"""
render_episode.py

Render one episode of a LeRobot v3.0 dataset as a single MP4 with the three camera views side by
side (side | top_right | top_left), so you can look at a demonstration without loading LeRobot.

  python tools/render_episode.py --root path/to/dataset --episode 42 --out episode_042.mp4

Needs ffmpeg on PATH (with an AV1 decoder, e.g. any "full" build) plus pandas + pyarrow.
"""

import argparse
import glob
import os
import subprocess

import pandas as pd

CAMS = ["side", "top_right", "top_left"]


def episode_row(root, ep):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    row = df[df.episode_index == ep]
    if row.empty:
        raise SystemExit(f"episode {ep} not found ({len(df)} episodes in {root})")
    return row.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder (the one holding meta/, data/, videos/)")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()

    r = episode_row(a.root, a.episode)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for cam in CAMS:
        k = f"videos/observation.images.{cam}"
        path = os.path.join(a.root, "videos", f"observation.images.{cam}",
                            f"chunk-{int(r[k + '/chunk_index']):03d}", f"file-{int(r[k + '/file_index']):03d}.mp4")
        t0, t1 = float(r[k + "/from_timestamp"]), float(r[k + "/to_timestamp"])
        cmd += ["-ss", f"{t0:.4f}", "-t", f"{t1 - t0:.4f}", "-i", path]
    cmd += ["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an",
            "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out}  (episode {a.episode}, {int(r['length'])} frames)")


if __name__ == "__main__":
    main()
