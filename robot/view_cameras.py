"""
Live view of exactly what the policy sees: all three cameras at 320x240, with the per-camera
settings from camera_resolver.py applied (side + top_right = the Windows "glitch" settings,
top_left = auto). Same capture path as run_policy.py. Works on Windows and Linux.

    python view_cameras.py              # live window (2x zoom)
    python view_cameras.py --training   # bottom row: a training frame per camera, to line the scene up

Keys: s = save a snapshot PNG next to this script, q / Esc = quit.
"""
import argparse
import datetime as dt
import os
import sys
import tkinter as tk

import av
import av.logging
import numpy as np
from PIL import Image, ImageDraw, ImageTk

av.logging.set_level(av.logging.ERROR)
sys.path.insert(0, ".")
from run_policy import CamGrabber  # noqa: E402  (applies the camera controls on open)
from camera_resolver import CAMERA_MAP, resolve_camera_paths  # noqa: E402

KEYS = list(CAMERA_MAP.values())        # side, top_right, top_left (model input order)
TRAIN_VIDEOS = os.path.join(os.environ.get("DELTA_DATASET", "../data/delta_robot_bolt_pick_place"), "videos")


def training_frames(episode_frame=5):
    out = []
    for key in KEYS:
        c = av.open(f"{TRAIN_VIDEOS}/observation.images.{key}/chunk-000/file-000.mp4")
        for i, f in enumerate(c.decode(video=0)):
            if i == episode_frame:
                out.append(f.to_ndarray(format="rgb24"))
                break
        c.close()
    return np.concatenate(out, 1)


def label(img, texts):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(texts):
        d.text((i * 320 + 4, 4), t, fill=(255, 0, 0))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true", help="show a training frame row underneath")
    ap.add_argument("--zoom", type=int, default=2)
    a = ap.parse_args()

    paths = resolve_camera_paths()
    cams = {k: CamGrabber(paths[k], k) for k in KEYS}
    train_row = label(training_frames(), [f"TRAINING {k}" for k in KEYS]) if a.training else None

    root = tk.Tk()
    root.title("What the policy sees (q = quit, s = snapshot)")
    panel = tk.Label(root)
    panel.pack()
    state = {"img": None}

    def tick():
        frames = []
        for k in KEYS:
            f, _ = cams[k].read()
            frames.append(f if f is not None else np.zeros((240, 320, 3), np.uint8))
        img = label(np.concatenate(frames, 1), [f"LIVE {k}" for k in KEYS])
        if train_row is not None:
            img = np.concatenate([img, train_row], 0)
        state["img"] = img
        pil = Image.fromarray(img)
        pil = pil.resize((pil.width * a.zoom, pil.height * a.zoom), Image.NEAREST)
        tkimg = ImageTk.PhotoImage(pil)
        panel.configure(image=tkimg)
        panel.image = tkimg
        root.after(33, tick)

    def on_key(e):
        if e.keysym in ("q", "Escape"):
            root.destroy()
        elif e.keysym == "s" and state["img"] is not None:
            out = f"snapshot_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
            Image.fromarray(state["img"]).save(out)
            print("saved", out)

    root.bind("<Key>", on_key)
    tick()
    try:
        root.mainloop()
    finally:
        for g in cams.values():
            g.close()


if __name__ == "__main__":
    main()
