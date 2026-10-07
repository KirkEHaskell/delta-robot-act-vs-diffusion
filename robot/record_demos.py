"""
record_demos.py -- record teleoperated demonstrations into a LeRobot (v3.0) dataset.

  python record_demos.py --root ../data/my_dataset --episodes 10

You move the leader arm by hand; the Teensy mirrors it onto the follower servos and streams both
(firmware/delta_teleop). The magnet button on the leader switches the electromagnet. Per episode:
press Enter to start, Ctrl+C to end it. Re-running the script appends to the same dataset.
This is the recorder that produced the study's 100-episode dataset (it was named
RecordTrialsWithElectromagnet3Cam.py then).

Robustness features:
  * Cameras captured via PyAV forcing MJPG (OpenCV's DSHOW can't force it and falls
    back to uncompressed YUY2, which starves the 3rd camera's USB bandwidth).
  * Cameras resolved by STABLE USB PORT ID via camera_resolver, not OpenCV index.
    Each role (side/top_right/top_left) always maps to the same physical camera.
  * Fails loud at startup if any camera is missing (never records a scrambled set).
  * If any camera DROPS mid-episode, that episode is aborted and discarded.

WINDOWS NOTE: all execution lives under `if __name__ == "__main__":`. LeRobot's
save_episode() spawns worker processes to encode video, and on Windows a spawned
process RE-IMPORTS this module -- without the guard it would re-open the serial
port / cameras in every worker (COM3 "Access is denied") and crash the encoder.

Teensy serial port: --port, or env DELTA_PORT (default COM3 on Windows, /dev/ttyACM0 on Linux).

Data schema:
  observation.state           : 3-D  [follower_j1, j2, j3]
  action                      : 4-D  [leader_j1, j2, j3, magnet]
  observation.images.side / top_right / top_left  (240x320x3)
"""

import argparse
import os
import sys
import av
import av.logging
# Silence per-frame ffmpeg/swscaler warnings ("deprecated pixel format used") that
# otherwise flood the console -- 3 cameras x 30 fps = thousands of lines. Real
# errors (ERROR and above) still print.
av.logging.set_level(av.logging.ERROR)
import serial
import time
import threading
import json
import shutil
import signal
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from camera_resolver import CAMERA_MAP, apply_controls, av_open_args, resolve_camera_paths

# =================== Config ===================
_ap = argparse.ArgumentParser(description="Record teleoperated demos into a LeRobot dataset.")
_ap.add_argument("--root", default=os.path.join("..", "data", "my_dataset"), help="dataset folder")
_ap.add_argument("--repo-id", default="local/delta_bolt_pick_place", help="LeRobot repo_id stored in the dataset")
_ap.add_argument("--episodes", type=int, default=10, help="episodes to record this session")
_ap.add_argument("--task", default="pick up the bolt and drop it in the bucket")
_ap.add_argument("--port", default=os.environ.get(
    "DELTA_PORT", "COM3" if sys.platform.startswith("win") else "/dev/ttyACM0"))
_args, _ = _ap.parse_known_args()      # parse_known_args: encoder worker processes re-import this module

SERIAL_PORT = _args.port
BAUD        = 115200
FPS         = 30
IMG_W, IMG_H = 320, 240
N_EPISODES   = _args.episodes
REPO_ID      = _args.repo_id
TASK_STR     = _args.task
DATASET_DIR  = _args.root
# Linux: put the cameras on auto for a fresh dataset. Set to 1 only to add episodes to the study
# dataset (reproduces its camera colour settings, see cameras_linux.py).
MATCH_TRAINING_CAMERAS = os.environ.get("DELTA_MATCH_TRAINING_CAMERAS", "0") == "1"

CAM_KEYS     = list(CAMERA_MAP.values())     # ["side", "top_right", "top_left"]
STALE_SEC    = 0.5                            # a camera silent this long = dropped

ROOT  = Path(DATASET_DIR)
_INFO = ROOT / "meta" / "info.json"

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (3,), "names": ["j1", "j2", "j3"]},
    "action":            {"dtype": "float32", "shape": (4,), "names": ["j1", "j2", "j3", "mag"]},
}
for _k in CAM_KEYS:
    FEATURES[f"observation.images.{_k}"] = {
        "dtype": "video", "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    }


# =================== Camera thread (PyAV, MJPG) ===================
class CamGrabber:
    """Background grabber over PyAV; exposes the latest RGB frame + its timestamp so
    the recorder can detect a camera that has stopped delivering frames."""
    def __init__(self, dev, name=""):
        self.name = name
        apply_controls(name, dev, MATCH_TRAINING_CAMERAS)
        f, kw = av_open_args(dev)
        self.container = av.open(f, **kw)
        self.frame = None
        self.last_ts = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        time.sleep(1.0)

    def _loop(self):
        try:
            for frame in self.container.decode(video=0):
                if not self.running:
                    break
                arr = frame.to_ndarray(format="rgb24")   # H x W x 3, RGB uint8
                with self.lock:
                    self.frame = arr
                    self.last_ts = time.perf_counter()
        except Exception:
            pass          # container closed / device dropped -> last_ts goes stale

    def read(self):
        with self.lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(), self.last_ts)

    def close(self):
        self.running = False
        try:
            self.container.close()
        except Exception:
            pass
        self.t.join(timeout=1.0)


def read_all_cams(cams):
    """({key: rgb_frame}, None) or (None, dropped_key) if any camera is stale."""
    now = time.perf_counter()
    frames = {}
    for key, g in cams.items():
        f, ts = g.read()
        if f is None or (now - ts) > STALE_SEC:
            return None, key
        frames[key] = f
    return frames, None


def parse_line(line):
    """Teensy CSV (8 fields) -> (leader[3], follower[3], mag). None on bad/-999."""
    parts = line.strip().split(",")
    if len(parts) != 8:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    leader   = np.array(vals[1:4], dtype=np.float32)
    follower = np.array(vals[4:7], dtype=np.float32)
    mag      = float(vals[7])
    if -999.0 in leader or -999.0 in follower:
        return None
    return leader, follower, mag


def dataset_has_data():
    if not _INFO.exists():
        return False
    try:
        n_ep = json.loads(_INFO.read_text()).get("total_episodes", 0)
    except Exception:
        n_ep = 0
    data_dir = ROOT / "data"
    return n_ep > 0 or (data_dir.exists() and any(data_dir.rglob("*.parquet")))


def record_episode(ds, ser, cams, ep_idx):
    print(f"\n=== Episode {ep_idx}/{N_EPISODES-1} ===")
    # Print the prompt to stdout, not via input() -- on Windows input() writes its
    # prompt to fd 2 (stderr), which we've redirected to null, so it would vanish.
    print("  Place bolt, get in start pose, press Enter to record (Ctrl+C to end episode)...",
          end="", flush=True)
    input()
    ser.reset_input_buffer()
    ser.write(b"s\n")
    dt = 1.0 / FPS
    t0 = time.perf_counter()
    n_written = 0
    last_print = 0
    aborted = False

    stop = False
    def on_sigint(sig, frame_):
        nonlocal stop
        stop = True
    old_handler = signal.signal(signal.SIGINT, on_sigint)

    try:
        while not stop:
            latest = None
            while ser.in_waiting:
                latest = ser.readline().decode(errors="ignore")
            if latest is None:
                latest = ser.readline().decode(errors="ignore")
            parsed = parse_line(latest) if latest else None
            if parsed is None:
                time.sleep(0.001)
                continue

            frames, dropped = read_all_cams(cams)
            if dropped is not None:
                print(f"\n  !! camera '{dropped}' dropped mid-episode -- aborting & discarding.")
                aborted = True
                break

            leader, follower, mag = parsed
            action = np.array([leader[0], leader[1], leader[2], mag], dtype=np.float32)
            frame_dict = {
                "observation.state": follower,
                "action": action,
                "task": TASK_STR,
            }
            for key in CAM_KEYS:
                frame_dict[f"observation.images.{key}"] = frames[key]
            ds.add_frame(frame_dict)
            n_written += 1

            if n_written - last_print >= FPS:
                print(f"  ...{n_written} frames (magnet={'ON' if mag > 0.5 else 'off'})", end="\r")
                last_print = n_written
            sleep_for = (t0 + n_written * dt) - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        ser.write(b"x\n")
        time.sleep(0.1)
        ser.reset_input_buffer()

    if aborted:
        ds.clear_episode_buffer()
        print("  episode discarded (camera drop).")
        return
    if n_written == 0:
        ds.clear_episode_buffer()
        print("  no frames captured — skipping save. Is the Teensy streaming 8-field CSV?")
        return
    ds.save_episode()
    print(f"  saved episode with {n_written} frames")


def main():
    # Redirect OS-level stderr (fd 2) to null so the video encoders' native chatter --
    # SVT-AV1 "Svt[info]" banners/progress, swscaler warnings, ffmpeg banners, emitted
    # by encoder worker processes and bypassing Python/av.logging -- can't flood the
    # console. Python's own stderr (tracebacks) stays on the real terminal.
    _real_err = os.dup(2)
    _dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_dn, 2); os.close(_dn)
    sys.stderr = os.fdopen(_real_err, "w", buffering=1)

    # Hide the per-episode "Map: 100%|..." progress bar (HuggingFace datasets tqdm).
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception:
        pass

    # ---- Resolve cameras (fail loud BEFORE touching hardware/dataset) ----
    resolved = resolve_camera_paths()        # {key: device_path}; raises if any missing
    print("Resolved cameras (by stable USB port):")
    for key in CAM_KEYS:
        print(f"  {key}: {resolved[key]}")

    # ---- Serial ----
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    time.sleep(2.0)
    ser.reset_input_buffer()

    # ---- Dataset (auto create / resume) ----
    if _INFO.exists() and not dataset_has_data():
        print(f"Clearing empty/partial dataset at {ROOT} (0 episodes).")
        shutil.rmtree(ROOT)
    if dataset_has_data():
        ds = LeRobotDataset.resume(REPO_ID, root=str(ROOT))
    else:
        ds = LeRobotDataset.create(REPO_ID, fps=FPS, features=FEATURES, root=str(ROOT))

    # ---- Cameras ----
    cams = {key: CamGrabber(resolved[key], key) for key in CAM_KEYS}
    _probe, _dropped = read_all_cams(cams)
    if _dropped is not None:
        for g in cams.values():
            g.close()
        ser.close()
        raise SystemExit(f"Camera '{_dropped}' opened but is not delivering frames. "
                         f"Check the cable/hub before recording.")

    # ---- Record loop ----
    try:
        for i in range(N_EPISODES):
            record_episode(ds, ser, cams, i)
        ds.finalize()
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C at prompt).")
    finally:
        for g in cams.values():
            g.close()
        ser.close()
        print("done.")


if __name__ == "__main__":
    main()
