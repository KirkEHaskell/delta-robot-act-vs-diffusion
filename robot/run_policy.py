"""
run_policy.py -- run a trained checkpoint on the delta robot, and run scored evaluation sessions.

Works on Windows (DirectShow cameras, COM port) and Linux (V4L2 cameras, /dev/tty*), CPU or GPU.
This is the script used for every robot trial in the study (the Windows copy ran ACT on a laptop
CPU, the Linux copy ran Diffusion on an RTX 3090; they were merged afterwards with the logic unchanged).

Quick start (from inside robot/):

  python run_policy.py dry --ckpt path/to/pretrained_model
      Live cameras + joint state through the policy; prints the actions. The robot is put in AUTO
      mode but NO commands are sent, so the servos just hold. Always do this first.

  python run_policy.py try --ckpt path/to/pretrained_model
      Run the policy on the robot as many times as you like (Enter = run, SPACE = stop the run).
      Type another checkpoint path at the prompt to switch models. Saves a trace per run.

Study workflow (checkpoints named like act_n50_s50k_005000 inside CKPT_DIR):

  python run_policy.py plan --name act --policies act --positions 10
      -> test_results/plan_act.csv : every checkpoint x every bolt position, seeded random order.
  python run_policy.py run --plan test_results/plan_act.csv [--blind]
      -> walks the plan, you score each trial. Resumable. Ctrl+C = safe stop.
  python run_policy.py summary
      -> success-rate table per cell + test_results/summary.png

Options (environment variables; the Diffusion ones are logged in the `inference` column):
  DELTA_PORT=COM3 | /dev/ttyACM0     Teensy serial port (default: COM3 on Windows, /dev/ttyACM0 on Linux)
  DELTA_CKPT_DIR=checkpoints         folder holding <run>_<step> checkpoint folders (study workflow)
  TC_DEVICE=cpu|cuda                 default: cuda if available
  TC_SCHED=DDPM TC_STEPS=0           Diffusion sampler. DDPM with all 100 steps is what the study used
                                     (~0.76 s per plan on an RTX 3090). On a CPU use TC_SCHED=DDIM
                                     TC_STEPS=10 (~2 s per plan; matched DDPM-100 within ~1 deg offline).
  TC_MATCH_TRAINING_CAMERAS=0        Linux only: leave the cameras on auto instead of reproducing the
                                     study's camera settings (do this for models trained on your own data).

Diffusion checkpoints trained with BLIND_STATE=1 (training/lerobot_train_fixed.py) get their joint-state
input zeroed here too. That happens automatically when the checkpoint holds a file named BLIND_STATE (the
published Diffusion checkpoint has one), when "nostate" appears in its path or in the training run's
output_dir (train_config.json), or when you set TC_BLIND_STATE=1. Diffusion runs print which applies.

Results: test_results/results.csv (one row per trial) and test_results/traces/*.npz (state, action,
timing per step).
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import random
import re
import sys
import threading
import time

import av
import av.logging
av.logging.set_level(av.logging.ERROR)
import numpy as np
import torch

# =================== Config ===================
SERIAL_PORT = os.environ.get("DELTA_PORT", "COM3" if sys.platform.startswith("win") else "/dev/ttyACM0")
BAUD        = 115200
FPS         = 30
IMG_W, IMG_H = 320, 240

CKPT_DIR    = os.environ.get("DELTA_CKPT_DIR", "checkpoints")
RESULTS_DIR = "test_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
DEVICE      = os.environ.get("TC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MATCH_TRAINING_CAMERAS = os.environ.get("TC_MATCH_TRAINING_CAMERAS", "1") == "1"

SERVO_MIN_DEG = 30.0        # match firmware DEG_MIN/DEG_MAX
SERVO_MAX_DEG = 110.0
MAGNET_THRESHOLD = 0.5

# Mean starting follower pose of the 100 dataset demos (std ~3.5 deg per joint).
# Every trial starts here so all checkpoints see the same initial state.
START_POSE  = (91.7, 93.2, 90.8)
SETTLE_SEC  = 1.5           # time to glide to START_POSE and settle before the policy starts
MAX_SEC     = 30.0          # trial timeout in seconds of robot motion (= 900 actions); demos took 8-17 s

# Diffusion inference setting. ("DDPM", None) = exactly as trained, 100 denoising steps: what the
# study used, on an RTX 3090 (~0.76 s per 24-action plan). ("DDIM", 10) = ~2 s per plan on a laptop
# CPU; offline it matched DDPM-100 within ~0.5-1 deg. Either way the robot pauses while a plan is
# computed (fine for this quasi-static task). Keep it the SAME for every checkpoint you compare.
DIFFUSION_SCHEDULER = os.environ.get("TC_SCHED", "DDPM")
DIFFUSION_STEPS     = int(os.environ.get("TC_STEPS", "0")) or None

# Opt-in kick-start for Diffusion (see _trial_loop). Demo openings: median -10 deg on all joints in
# the first 0.33 s of motion (95% of episodes move all three joints the same way).
KICK_DEG   = float(os.environ.get("TC_KICK_DEG", "0"))
KICK_SEC   = float(os.environ.get("TC_KICK_SEC", "0.33"))
KICK_STEPS = max(1, round(KICK_SEC * FPS))

# Opt-in Diffusion anti-jitter settings (logged in the `inference` column):
#  TC_ACTION_STEPS=16 : execute 16 actions per plan instead of the trained default 8 (fewer replans)
#  TC_FIXED_NOISE=1   : start every plan from the same initial noise -> consecutive plans agree
ACTION_STEPS = int(os.environ.get("TC_ACTION_STEPS", "0")) or None
FIXED_NOISE  = os.environ.get("TC_FIXED_NOISE", "0") == "1"

STALE_SEC = 0.5

RESULT_FIELDS = ["timestamp", "plan", "plan_order", "checkpoint", "policy", "n_episodes", "steps",
                 "position", "valid", "picked", "success", "failure_mode", "duration_s",
                 "end_reason", "magnet_presses", "n_plans", "mean_plan_s", "max_plan_s", "inference", "notes"]

FAILURE_MODES = {
    "1": "no_pick",          # never got the bolt on the magnet
    "2": "dropped",          # picked, lost it before the bucket
    "3": "missed_bucket",    # released, but not in the bucket
    "4": "stalled",          # froze / wandered / timed out without finishing
    "5": "unsafe",           # erratic motion, operator stopped it
    "6": "other",
}


def parse_ckpt_name(name):
    """act_n50_s50k_005000 -> ('act', 50, 5000). Variants keep their prefix as the policy name,
    e.g. diffusionnostateh48_n50_s5k_005000 -> ('diffusionnostateh48', 50, 5000), so results never
    mix. Any other name (e.g. a downloaded checkpoint path) -> (folder name, 0, 0)."""
    base = os.path.basename(os.path.normpath(name))
    m = re.match(r"((?:act|diffusion)[a-z0-9]*)_n(\d+)_s\d+k_(\d+)$", base)
    if not m:
        return base, 0, 0
    return m.group(1), int(m.group(2)), int(m.group(3))


def ckpt_path(name):
    """A checkpoint is either a folder path, or a folder name inside CKPT_DIR."""
    return name if os.path.isdir(name) else os.path.join(CKPT_DIR, name)


def needs_blind_state(path, cfg):
    """True for Diffusion checkpoints trained with BLIND_STATE=1. Detected from (in order): env
    TC_BLIND_STATE=1/0, a BLIND_STATE marker file, "nostate" anywhere in the checkpoint path, or
    "nostate" in the training run's output_dir recorded in train_config.json (a checkpoint's own
    folder is just .../checkpoints/050000/pretrained_model, so its name alone isn't enough)."""
    if cfg.type != "diffusion":
        return False
    if os.environ.get("TC_BLIND_STATE") in ("0", "1"):
        return os.environ["TC_BLIND_STATE"] == "1"
    if os.path.exists(os.path.join(path, "BLIND_STATE")):
        return True
    if "nostate" in os.path.abspath(path):
        return True
    try:
        with open(os.path.join(path, "train_config.json")) as f:
            return "nostate" in str(json.load(f).get("output_dir", ""))
    except (OSError, ValueError):
        return False


def all_checkpoints():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CKPT_DIR, "*_*_*_*"))
                  if os.path.isdir(p))


def inference_label(policy_type):
    if "diffusion" in policy_type:
        kick = f"+kick{KICK_DEG:g}deg/{KICK_SEC:g}s" if KICK_DEG > 0 else ""
        steps = f"+act{ACTION_STEPS}" if ACTION_STEPS else ""
        noise = "+fixednoise" if FIXED_NOISE else ""
        return f"{DIFFUSION_SCHEDULER}-{DIFFUSION_STEPS or 100}{steps}{noise}{kick}"
    return "chunk100"


# =================== Policy loading ===================
def load_policy(name):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    path = ckpt_path(name)
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = DEVICE                                   # saved as "cuda" on the training box
    if cfg.type == "diffusion":
        cfg.noise_scheduler_type = DIFFUSION_SCHEDULER
        cfg.num_inference_steps = DIFFUSION_STEPS
        if ACTION_STEPS:
            cfg.n_action_steps = ACTION_STEPS          # must be <= horizon - n_obs_steps + 1
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(DEVICE).eval()
    policy.reset()
    policy.delta_policy_type = cfg.type
    if cfg.type == "diffusion" and not needs_blind_state(path, cfg):
        print("  NOTE: joint-state input NOT blinded. If this checkpoint was trained with BLIND_STATE=1,"
              " set TC_BLIND_STATE=1 or it will stall.")
    if needs_blind_state(path, cfg):
        print("  joint-state input blinded (zeroed), as in training")
        # trained with observation.state zeroed (training/lerobot_train_fixed.py BLIND_STATE=1):
        # blind it identically at inference, AFTER the preprocessor has normalized it.
        from lerobot.utils.constants import OBS_STATE
        _sel = policy.select_action

        def _blind_select(batch, *a, **kw):
            batch = dict(batch)
            batch[OBS_STATE] = torch.zeros_like(batch[OBS_STATE])
            return _sel(batch, *a, **kw)
        policy.select_action = _blind_select
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": DEVICE}})
    return policy, pre, post


# =================== Camera thread (PyAV, MJPG) ===================
class CamGrabber:
    def __init__(self, dev, name=""):
        from camera_resolver import apply_controls, av_open_args
        self.name = name
        apply_controls(name, dev, MATCH_TRAINING_CAMERAS)   # Linux: study colour settings (cameras_linux.py)
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
                arr = frame.to_ndarray(format="rgb24")
                with self.lock:
                    self.frame = arr
                    self.last_ts = time.perf_counter()
        except Exception:
            pass

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
    now = time.perf_counter()
    frames = {}
    for key, g in cams.items():
        f, ts = g.read()
        if f is None or (now - ts) > STALE_SEC:
            return None, key
        frames[key] = f
    return frames, None


def parse_line(line):
    """Teensy CSV (8 fields) -> follower[3] (observation.state). None on bad/-999."""
    parts = line.strip().split(",")
    if len(parts) != 8:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    follower = np.array(vals[4:7], dtype=np.float32)
    if -999.0 in follower:
        return None
    global TEENSY_MAG
    TEENSY_MAG = int(vals[7])          # what the firmware says the magnet is doing
    return follower


TEENSY_MAG = None


def latest_state(ser):
    """Drain the serial buffer and return the newest valid follower state (or None)."""
    latest = None
    while ser.in_waiting:
        latest = ser.readline().decode(errors="ignore")
    if latest is None:
        latest = ser.readline().decode(errors="ignore")
    return parse_line(latest) if latest else None


def to_img_tensor(rgb):
    """H x W x 3 RGB uint8 -> (3,H,W) float32 in [0,1] (unbatched; preprocessor batches)."""
    return torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1)


class Rig:
    """Serial + three cameras, opened once for the whole session."""

    def __init__(self):
        import serial
        from camera_resolver import resolve_camera_paths, CAMERA_MAP
        self.cam_keys = list(CAMERA_MAP.values())          # ["side", "top_right", "top_left"]
        resolved = resolve_camera_paths()
        print("Resolved cameras (by stable USB port):")
        for key in self.cam_keys:
            print(f"  {key}: {resolved[key]}")
        self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.cams = {key: CamGrabber(resolved[key], key) for key in self.cam_keys}
        _, dropped = read_all_cams(self.cams)
        if dropped is not None:
            self.close()
            raise SystemExit(f"Camera '{dropped}' opened but is not delivering frames.")

    def send(self, a1, a2, a3, magnet_on):
        self.ser.write(f"c,{a1:.3f},{a2:.3f},{a3:.3f},{1 if magnet_on else 0}\n".encode())

    def wait_state(self, timeout=2.0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            s = latest_state(self.ser)
            if s is not None:
                return s
        return None

    def glide_to(self, target, seconds=SETTLE_SEC):
        """Interpolate from the current pose to target (magnet off), then hold."""
        cur = self.wait_state()
        if cur is None:
            raw = [self.ser.readline().decode(errors="ignore").strip() for _ in range(3)]
            raw = [r for r in raw if r]
            hint = ("Teensy is silent: firmware not running or hung (replug USB / re-upload)."
                    if not raw else
                    "a follower servo reads -999: it isn't answering (servo power / bus cable)."
                    if any("-999" in r for r in raw) else "lines don't parse as state.")
            raise RuntimeError(f"no valid state from Teensy. {hint}  Raw lines: {raw}")
        target = np.array(target, dtype=np.float32)
        n = max(1, int(seconds * FPS * 0.6))
        for k in range(1, n + 1):
            p = cur + (target - cur) * (k / n)
            self.send(*p, False)
            time.sleep(1.0 / FPS)
        time.sleep(seconds * 0.4)

    def obs(self):
        """(obs dict, follower) or (None, reason)."""
        follower = latest_state(self.ser)
        frames, dropped = read_all_cams(self.cams)
        if dropped is not None:
            return None, f"camera_drop:{dropped}"
        if follower is None:
            return None, "no_state"
        o = {"observation.state": torch.from_numpy(follower)}
        for key in self.cam_keys:
            o[f"observation.images.{key}"] = to_img_tensor(frames[key])
        return o, follower

    def park(self):
        """Magnet off where it is, then glide back to START_POSE (jump there if no state)."""
        try:
            cur = self.wait_state()
            if cur is not None:
                self.send(*cur, False)              # magnet off, stay put
            self.glide_to(START_POSE)
        except Exception:
            self.send(*START_POSE, False)

    def safe_idle(self):
        try:
            self.park()
            self.ser.write(b"x\n")
            time.sleep(0.1)
        except Exception:
            pass

    def close(self):
        self.safe_idle()
        for g in getattr(self, "cams", {}).values():
            g.close()
        self.ser.close()


# =================== Keyboard ===================
import contextlib

if sys.platform.startswith("win"):
    import msvcrt

    @contextlib.contextmanager
    def raw_keys():
        yield

    def key_pressed():
        return msvcrt.getwch() if msvcrt.kbhit() else None

    def flush_keys():
        while msvcrt.kbhit():
            msvcrt.getwch()
else:
    import select
    import termios
    import tty

    @contextlib.contextmanager
    def raw_keys():
        """Single-key reads (no Enter needed) for the duration of a run; restores the terminal after,
        so the normal input() prompts keep working."""
        if not sys.stdin.isatty():
            yield
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def key_pressed():
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return "\r" if ch == "\n" else ch
        return None

    def flush_keys():
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


# =================== One trial ===================
def run_trial(rig, policy, pre, post, send_commands=True, max_sec=MAX_SEC):
    """Policy loop at FPS. Slow plans (e.g. Diffusion on CPU) simply pause the robot: the clock is
    re-synced after an overrun instead of bursting the backlog of actions."""
    policy.reset()
    trace = {"t": [], "state": [], "action": [], "plan_s": []}
    plan_times = []
    t_start = time.perf_counter()
    flush_keys()
    print("  running... SPACE, Enter or Ctrl+C = stop this run")
    max_steps = int(max_sec * FPS)              # timeout counts executed actions, not wall time,
    try:                                        # so planning pauses don't eat into it
        with raw_keys():
            end_reason = _trial_loop(rig, policy, pre, post, send_commands, max_steps,
                                     trace, plan_times, t_start)
    except KeyboardInterrupt:                   # Ctrl+C ends the run, not the session
        end_reason = "operator"
    duration = time.perf_counter() - t_start
    return trace, plan_times, duration, end_reason


def _trial_loop(rig, policy, pre, post, send_commands, max_steps, trace, plan_times, t_start):
    dt_ = 1.0 / FPS
    next_t = t_start
    while True:
        if len(trace["t"]) >= max_steps:
            return "timeout"
        k = key_pressed()
        if k in (" ", "\r", "\n", "\x1b"):
            return "operator"

        o, follower = rig.obs()
        if o is None:
            if follower.startswith("camera_drop"):
                return follower
            time.sleep(0.001)
            continue

        t0 = time.perf_counter()
        with torch.inference_mode():
            kw = {}
            if FIXED_NOISE and getattr(policy.config, "type", "") == "diffusion":
                if "noise" not in trace:              # one noise tensor per trial, reused by every plan
                    g = torch.Generator(device="cpu").manual_seed(0)
                    trace["noise"] = torch.randn((1, policy.config.horizon, policy.config.action_feature.shape[0]),
                                                 generator=g).to(DEVICE)
                kw["noise"] = trace["noise"]
            a = post(policy.select_action(pre(o), **kw)).squeeze(0).cpu().numpy()   # .cpu() syncs CUDA
        call_s = time.perf_counter() - t0
        if call_s > 0.05:                       # this call computed a new plan / chunk
            plan_times.append(call_s)
            if hasattr(rig, "ser"):
                # Nobody read serial during the plan; the ~4 KB driver buffer filled with old
                # lines, so drop them and let the next state read be a fresh one.
                rig.ser.reset_input_buffer()

        # KICK-START (Diffusion only, opt-in): replay the demos' opening move -- all three joints
        # down KICK_DEG over KICK_SEC -- while still feeding observations to the policy, so its
        # 2-frame state history sees motion. Then drop its queued actions so it replans from there.
        n_done = len(trace["t"])
        if KICK_DEG > 0 and getattr(policy.config, "type", "") == "diffusion" and n_done < KICK_STEPS:
            if n_done == 0:
                trace["kick_from"] = follower.copy()
            frac = (n_done + 1) / KICK_STEPS
            a = np.concatenate([trace["kick_from"] - KICK_DEG * frac, [0.0]]).astype(np.float32)
            if n_done == KICK_STEPS - 1:
                policy._queues["action"].clear()

        a1, a2, a3 = (float(np.clip(v, SERVO_MIN_DEG, SERVO_MAX_DEG)) for v in a[:3])
        magnet_on = bool(a[3] > MAGNET_THRESHOLD)
        if send_commands:
            rig.send(a1, a2, a3, magnet_on)

        trace["t"].append(time.perf_counter() - t_start)
        trace["state"].append(follower)
        trace["action"].append(a)
        trace["plan_s"].append(call_s)
        n = len(trace["t"])
        if n % FPS == 0 or not send_commands:
            print(f"    t={trace['t'][-1]:5.1f}s  state={np.round(follower, 1)}  "
                  f"cmd=[{a1:.1f} {a2:.1f} {a3:.1f}]  magnet={'ON' if magnet_on else 'off'} "
                  f"(teensy says {'ON' if TEENSY_MAG else 'off'})"
                  f"  ({call_s*1000:.0f} ms)")

        next_t += dt_
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        elif sleep_for < -dt_:
            next_t = time.perf_counter()        # overran (planning) -> resync, don't burst


# =================== Commands ===================
def magnet_presses(trace):
    """How many times the policy switched the magnet on = pick attempts."""
    if not trace["action"]:
        return 0
    on = (np.array(trace["action"])[:, 3] > MAGNET_THRESHOLD).astype(int)
    return int(on[0] + np.sum(np.diff(on) == 1))


def cmd_plan(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"plan_{args.name}.csv")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite).")
    ckpts = [c for c in all_checkpoints() if parse_ckpt_name(c)[0] in args.policies]
    positions = [f"P{i:02d}" for i in range(1, args.positions + 1)]
    pairs = [(c, p) for c in ckpts for p in positions]
    random.Random(args.seed).shuffle(pairs)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan_order", "checkpoint", "position"])
        for i, (c, p) in enumerate(pairs, 1):
            w.writerow([i, c, p])
    print(f"wrote {out}: {len(ckpts)} checkpoints x {len(positions)} positions = {len(pairs)} trials "
          f"(seed {args.seed})")


def read_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_result(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def ask(prompt, options):
    while True:
        ans = input(prompt).strip().lower()
        if ans in options:
            return ans
        print(f"    please answer one of: {', '.join(options)}")


def cmd_run(args):
    plan_name = os.path.splitext(os.path.basename(args.plan))[0]
    with open(args.plan, newline="") as f:
        plan = list(csv.DictReader(f))
    done = {(r["plan"], r["plan_order"]) for r in read_results() if r["valid"] == "1"}
    pending = [r for r in plan if (plan_name, r["plan_order"]) not in done]
    print(f"{plan_name}: {len(plan) - len(pending)}/{len(plan)} trials already scored, "
          f"{len(pending)} to go.")
    if not pending:
        return
    torch.set_num_threads(os.cpu_count() or 1)
    os.makedirs(os.path.join(RESULTS_DIR, "traces"), exist_ok=True)

    rig = Rig()
    loaded = (None, None)
    try:
        rig.ser.reset_input_buffer()
        rig.ser.write(b"a\n")                    # AUTO: Teensy streams, host owns servos + magnet
        i = 0
        while i < len(pending):
            row = pending[i]
            ck, pos = row["checkpoint"], row["position"]
            ptype, n_ep, steps = parse_ckpt_name(ck)
            if loaded[0] != ck:
                print(f"\nloading {ck} ...")
                loaded = (ck, load_policy(ck))
            policy, pre, post = loaded[1]

            print(f"\n=== trial {row['plan_order']}/{len(plan)}  (remaining {len(pending) - i}) ===")
            print(f"    place the bolt at  >>> {pos} <<<   (checkpoint hidden: see log afterwards)"
                  if args.blind else
                  f"    checkpoint {ck}   place the bolt at  >>> {pos} <<<")
            cmd = ask("    Enter = go, s = skip for now, q = quit: ", ["", "s", "q"])
            if cmd == "q":
                break
            if cmd == "s":
                pending.append(pending.pop(i))
                continue

            rig.glide_to(START_POSE)
            trace, plan_times, duration, end_reason = run_trial(rig, policy, pre, post)
            rig.park()                            # magnet off, glide back to start pose for scoring
            print(f"  ended ({end_reason}) after {duration:.1f}s;  plans: {len(plan_times)}"
                  + (f", mean {np.mean(plan_times):.2f}s" if plan_times else ""))

            valid = ask("  valid trial? (y / n = setup problem, redo later): ", ["y", "n"])
            picked = success = ""
            mode = ""
            if valid == "y":
                picked = ask("  bolt picked up? (y/n): ", ["y", "n"])
                success = ask("  bolt ended in the bucket? (y/n): ", ["y", "n"]) if picked == "y" else "n"
                if success == "n":
                    print("  failure mode: " + "  ".join(f"{k}={v}" for k, v in FAILURE_MODES.items()))
                    mode = FAILURE_MODES[ask("  > ", list(FAILURE_MODES))]
            notes = input("  notes (optional): ").strip()

            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            np.savez_compressed(
                os.path.join(RESULTS_DIR, "traces", f"{stamp}_{plan_name}_{row['plan_order']}_{ck}_{pos}.npz"),
                t=np.array(trace["t"]), state=np.array(trace["state"]),
                action=np.array(trace["action"]), call_s=np.array(trace["plan_s"]))
            append_result({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "plan": plan_name,
                "plan_order": row["plan_order"], "checkpoint": ck, "policy": ptype,
                "n_episodes": n_ep, "steps": steps, "position": pos,
                "valid": 1 if valid == "y" else 0,
                "picked": {"y": 1, "n": 0}.get(picked, ""), "success": {"y": 1, "n": 0}.get(success, ""),
                "failure_mode": mode, "duration_s": round(duration, 2), "end_reason": end_reason,
                "magnet_presses": magnet_presses(trace),
                "n_plans": len(plan_times),
                "mean_plan_s": round(float(np.mean(plan_times)), 3) if plan_times else "",
                "max_plan_s": round(float(np.max(plan_times)), 3) if plan_times else "",
                "inference": inference_label(ptype), "notes": notes})
            if valid == "y":
                i += 1
            else:
                pending.append(pending.pop(i))      # redo at the end of the session
                print("  logged as invalid; it will come round again.")
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping (the interrupted trial was not scored).")
    finally:
        rig.close()
        print("rig closed (magnet off, idle).")


def cmd_dry(args):
    torch.set_num_threads(os.cpu_count() or 1)
    policy, pre, post = load_policy(args.ckpt)
    rig = Rig()
    try:
        rig.ser.reset_input_buffer()
        rig.ser.write(b"a\n")                    # AUTO with no 'c' commands: servos hold still
        print("\nDRY RUN: no commands are sent. Watch the printed actions (degrees, magnet 0..1).")
        trace, plan_times, duration, _ = run_trial(rig, policy, pre, post,
                                                   send_commands=False, max_sec=args.seconds)
        acts = np.array(trace["action"])
        if len(acts):
            np.set_printoptions(precision=2, suppress=True)
            print(f"\n{len(acts)} steps in {duration:.1f}s. action min {acts.min(0)}  max {acts.max(0)}")
            print(f"plan calls: {len(plan_times)}  times (s): {np.round(plan_times, 2)}")
    except KeyboardInterrupt:
        pass
    finally:
        rig.close()


def cmd_try(args):
    """Run checkpoints on the robot as many times as you like. Cameras and serial stay open the
    whole session; type another checkpoint name at the prompt to switch models."""
    torch.set_num_threads(os.cpu_count() or 1)
    known = all_checkpoints()
    ckpt = args.ckpt
    print(f"loading {ckpt} ...")
    policy, pre, post = load_policy(ckpt)
    print(f"inference: {inference_label(policy.delta_policy_type)}")
    rig = Rig()
    run_no = 0
    pos = "P?"
    try:
        rig.ser.reset_input_buffer()
        rig.ser.write(b"a\n")
        while True:
            cmd = input(f"\n[{ckpt} | bolt at {pos}]  Enter = run {run_no + 1},  p1..p9 = set position,  "
                        f"checkpoint name = switch model,  q = quit: ").strip()
            if cmd.lower() == "q":
                break
            if re.fullmatch(r"[pP]\d+", cmd):
                pos = cmd.upper()
                continue
            if cmd:
                if cmd not in known and not os.path.isdir(cmd):
                    print(f"  unknown checkpoint '{cmd}'. Give a folder path, or a name inside {CKPT_DIR}/.")
                    continue
                print(f"loading {cmd} ...")
                ckpt = cmd
                policy, pre, post = load_policy(ckpt)
                run_no = 0
                print(f"inference: {inference_label(policy.delta_policy_type)}")
                continue
            run_no += 1
            print(f"=== {ckpt}  run {run_no}  bolt at {pos} ===")
            try:
                rig.glide_to(START_POSE)
            except KeyboardInterrupt:
                rig.park()
                continue
            trace, plan_times, duration, end_reason = run_trial(rig, policy, pre, post, max_sec=args.seconds)
            rig.park()
            os.makedirs(os.path.join(RESULTS_DIR, "try_traces"), exist_ok=True)
            out = os.path.join(RESULTS_DIR, "try_traces",
                               f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(os.path.normpath(ckpt))}_{pos.replace('?', 'x')}_run{run_no}.npz")
            np.savez_compressed(out, t=np.array(trace["t"]), state=np.array(trace["state"]),
                                action=np.array(trace["action"]), call_s=np.array(trace["plan_s"]))
            print(f"  trace saved: {out}")
            print(f"  grab attempts (magnet switched on): {magnet_presses(trace)}")
            print(f"  ended ({end_reason}) after {duration:.1f}s;  plans: {len(plan_times)}"
                  + (f", mean {np.mean(plan_times):.2f}s" if plan_times else ""))
    except (KeyboardInterrupt, EOFError):        # Ctrl+C at the prompt = quit
        print("\nquitting.")
    finally:
        rig.close()
        print("rig closed (magnet off, idle).")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cmd_summary(args):
    rows = [r for r in read_results() if r["valid"] == "1"]
    if not rows:
        raise SystemExit("no valid results yet.")
    cells = {}
    for r in rows:
        key = (r["policy"], int(r["n_episodes"]), int(r["steps"]))
        c = cells.setdefault(key, {"n": 0, "succ": 0, "pick": 0, "inf": set()})
        c["n"] += 1
        c["succ"] += int(r["success"])
        c["pick"] += int(r["picked"])
        c["inf"].add(r["inference"])
    print(f"{'policy':10} {'eps':>4} {'steps':>6}  {'trials':>6}  {'success':>8}  {'95% CI':>13}  {'picked':>7}  inference")
    for key in sorted(cells):
        c = cells[key]
        lo, hi = wilson(c["succ"], c["n"])
        ci = f"[{lo:.0%}-{hi:.0%}]"
        print(f"{key[0]:10} {key[1]:>4} {key[2]:>6}  {c['n']:>6}  {c['succ']/c['n']:>8.0%}  {ci:>13}  "
              f"{c['pick']/c['n']:>7.0%}  {','.join(sorted(c['inf']))}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    policies = sorted({k[0] for k in cells})
    fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 4), squeeze=False)
    for ax, pol in zip(axes[0], policies):
        for steps in sorted({k[2] for k in cells if k[0] == pol}):
            ks = sorted(k for k in cells if k[0] == pol and k[2] == steps)
            xs = [k[1] for k in ks]
            ys = [cells[k]["succ"] / cells[k]["n"] for k in ks]
            ax.plot(xs, ys, marker="o", label=f"{steps // 1000}k steps")
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("demo episodes")
        ax.set_ylabel("success rate")
        ax.set_title(pol)
        ax.legend()
        ax.grid(alpha=0.3)
    out = os.path.join(RESULTS_DIR, "summary.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nplot: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="make a randomized trial order")
    p.add_argument("--name", required=True)
    p.add_argument("--policies", nargs="+", default=["act", "diffusion"],
                   help="checkpoint-name prefixes to include, e.g. act diffusionnostateh48")
    p.add_argument("--positions", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run a plan on the robot")
    p.add_argument("--plan", required=True)
    p.add_argument("--blind", action="store_true", help="hide which checkpoint is running while scoring")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("dry", help="policy on live inputs, no commands sent")
    p.add_argument("--ckpt", required=True, help="checkpoint folder path, or a name inside CKPT_DIR")
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(func=cmd_dry)

    p = sub.add_parser("try", help="run one checkpoint on the robot, no scoring")
    p.add_argument("--ckpt", required=True, help="checkpoint folder path, or a name inside CKPT_DIR")
    p.add_argument("--seconds", type=float, default=MAX_SEC * 4)
    p.set_defaults(func=cmd_try)

    p = sub.add_parser("summary", help="success table + plot")
    p.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
