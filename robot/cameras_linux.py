"""
Linux camera resolver for the delta-robot rig (replaces the Windows/DirectShow camera_resolver.py).

Cameras are identified by the physical USB port they sit in (/dev/v4l/by-path), which is stable
across replug and reboot. All three cameras are the same model with the same serial number
("SN0001"), so /dev/v4l/by-id cannot tell them apart. Only the port can.

THE "GREEN-YELLOW GLITCH" (why camera settings are forced here):
On the Windows laptop, the side and top_right cameras had been opened by an old OpenCV recorder
that set CAP_PROP_AUTO_WB=0 and CAP_PROP_EXPOSURE=-6. Windows remembered those settings per
device, so every 3-cam recording (= ALL training data) and all laptop inference ran those two
cameras with auto white balance OFF (stuck at the default 4600 K) and manual exposure 2^-6 s
(= exposure_time_absolute 156). top_left was never touched by that script and ran on full auto.
Verified 2026-10-04 by matching live Linux frames to the training videos (luma, Cb, and the
blue/green ratio across brightness all line up). So we reproduce the glitch at the camera,
not in software.

First-time setup: `ls -l /dev/v4l/by-path` with the cameras plugged in, then edit CAMERA_MAP below
so each USB port maps to its role. Needs `v4l2-ctl` (package v4l-utils).

To check mapping + settings + live colour vs the training data:
    python cameras_linux.py
"""

import os
import subprocess

BY_PATH = "/dev/v4l/by-path"

# USB port (as it appears in the by-path name) -> dataset camera key.
# Ports are on the study desktop's USB hub (3.1 / 3.2 / 3.4).   *** EDIT FOR YOUR CAMERAS ***
CAMERA_MAP = {
    "usb-0:3.1:1.0": "side",
    "usb-0:3.4:1.0": "top_right",
    "usb-0:3.2:1.0": "top_left",
}

# Per-camera V4L2 controls applied at every open (UVC controls reset when a camera is replugged).
GLITCH = {"auto_exposure": 1, "exposure_time_absolute": 156,             # manual, 15.6 ms
          "white_balance_automatic": 0, "white_balance_temperature": 4600}  # AWB off @ default
AUTO = {"auto_exposure": 3, "white_balance_automatic": 1}                  # camera defaults
CAMERA_CONTROLS = {"side": GLITCH, "top_right": GLITCH, "top_left": AUTO}

# Colour signature of each camera in the training videos (median luma Y and chroma Cb).
TRAINING_SIGNATURE = {"side": {"Y": 53, "Cb": 112}, "top_right": {"Y": 83, "Cb": 98},
                      "top_left": {"Y": 140, "Cb": 128}}


def _devices():
    """{port: /dev/videoN} for the capture node (video-index0) of every camera on a port."""
    out = {}
    if not os.path.isdir(BY_PATH):
        return out
    for name in sorted(os.listdir(BY_PATH)):
        if not name.endswith("video-index0") or "-usbv2-" in name:
            continue
        port = name.split("-usb-")[-1].replace("-video-index0", "")
        out["usb-" + port if not port.startswith("usb-") else port] = os.path.realpath(
            os.path.join(BY_PATH, name))
    return out


def resolve_camera_paths(camera_map=CAMERA_MAP):
    """{key: /dev/videoN}. Fails loud if a camera is missing or two keys hit the same device."""
    devs = _devices()
    resolved, missing = {}, []
    for port, key in camera_map.items():
        if port in devs:
            resolved[key] = devs[port]
        else:
            missing.append(f"    {key}: port {port} -- NOT FOUND")
    if missing:
        have = "\n".join(f"    {p}: {d}" for p, d in devs.items()) or "    (none)"
        raise RuntimeError("Camera resolve failed:\n" + "\n".join(missing)
                           + f"\n  Cameras present right now:\n{have}\n"
                           "  Fix: plug each camera into its hub port (or update CAMERA_MAP).")
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError(f"two cameras resolved to the same device: {resolved}")
    return resolved


def apply_controls(key, dev, match_training=True):
    """Set this camera's controls and read them back; raise if the camera didn't take them.
    match_training=True reproduces the study's camera settings (needed to run the published
    checkpoints); False puts every camera on auto (for recording a fresh dataset)."""
    ctrls = CAMERA_CONTROLS[key] if match_training else AUTO
    subprocess.run(["v4l2-ctl", "-d", dev] + [f"--set-ctrl={k}={v}" for k, v in ctrls.items()],
                   check=True, capture_output=True)
    got = subprocess.run(["v4l2-ctl", "-d", dev, "--get-ctrl=" + ",".join(ctrls)],
                         check=True, capture_output=True, text=True).stdout
    # menu controls read back as "1 (Manual Mode)", so compare the leading number only
    vals = {k: v.split()[0] for k, v in (line.split(": ") for line in got.strip().splitlines())}
    bad = {k: (v, vals.get(k)) for k, v in ctrls.items() if str(v) != vals.get(k)}
    if bad:
        raise RuntimeError(f"camera {key} ({dev}) did not accept controls (wanted, got): {bad}")


def av_open_args(dev):
    """(file, kwargs) for av.open(): MJPG at 320x240@30, same stream as the Windows dshow input."""
    return dev, {"format": "v4l2",
                 "options": {"input_format": "mjpeg", "video_size": "320x240", "framerate": "30"}}


if __name__ == "__main__":
    import av
    import numpy as np
    paths = resolve_camera_paths()
    print("Resolved cameras (by USB port):")
    for key, dev in paths.items():
        apply_controls(key, dev)
        f, kw = av_open_args(dev)
        c = av.open(f, **kw)
        for i, fr in enumerate(c.decode(video=0)):
            if i == 45:                       # let the auto camera settle
                y = fr.reformat(format="yuv420p").to_ndarray()
                Y, Cb = int(np.median(y[:240])), int(np.median(y[240:300]))
                break
        c.close()
        t = TRAINING_SIGNATURE[key]
        print(f"  {key:9s} {dev}  controls={'GLITCH' if CAMERA_CONTROLS[key] is GLITCH else 'AUTO'}  "
              f"live Y={Y:3d} Cb={Cb:3d}   training Y={t['Y']:3d} Cb={t['Cb']:3d}")
    print("Live numbers depend on the room lighting; they should be in the same ballpark as training.")
