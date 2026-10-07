"""
Offline "does it actually look at the cameras?" diagnostic for a checkpoint (no robot needed).

    cd robot && python ../analysis/diag_copycat.py <checkpoint_dir> [<another> ...]

Dataset root: env DELTA_DATASET (default ../data/delta_robot_bolt_pick_place).

For each checkpoint, on real training episodes (1, 42, 91 -- in every data subset):
  replay   : feed the recorded demo frames in order; mean |pred - demo| per joint (teacher-forced).
             Low error alone proves nothing: copying the joint state also scores well here.
  imgs_only: images play the demo, joint state frozen at frame 0 -> how much of the demo's motion
             range the policy reproduces from the cameras alone (ACT ~100%; copycat Diffusion ~35%).
  start    : the closed-loop start (robot still at the start pose, cameras on the first frame),
             3 s of calls, 2 seeds per episode -> max commanded move. Demos start moving at ~1.3 s,
             so a healthy policy should command a big move (>5 deg); a copycat stays < 1 deg.
"""
import sys

import numpy as np
import torch

sys.argv, NAMES = sys.argv[:1], sys.argv[1:]
import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import run_policy as T  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

K = ["side", "top_right", "top_left"]
EPS = [1, 42, 91]
N = 240                                    # 8 s of each episode


def obs(state, imgs_from):
    o = {"observation.state": state}
    for k in K:
        o[f"observation.images.{k}"] = imgs_from[f"observation.images.{k}"]
    return o


def main():
    ds = LeRobotDataset("local/delta_robot_bolt_pick_place",
                        root=os.environ.get("DELTA_DATASET", "../data/delta_robot_bolt_pick_place"),
                        episodes=EPS, video_backend="pyav")
    eps = []
    for e in EPS:
        a = ds._absolute_to_relative_idx[int(ds.meta.episodes["dataset_from_index"][e])]
        eps.append([ds[a + i] for i in range(N)])
    for name in NAMES:
        pol, pre, post = T.load_policy(name)
        rep_err, ratio, start = [], [], []
        with torch.inference_mode():
            for fr in eps:
                demo = np.stack([f["action"].numpy() for f in fr])[:, :3]
                rng = np.ptp(demo, 0)
                pol.reset()
                p = np.array([post(pol.select_action(pre(obs(f["observation.state"], f)))).squeeze(0).cpu().numpy()
                              for f in fr])
                rep_err.append(np.abs(p[:, :3] - demo).mean())
                pol.reset()
                p = np.array([post(pol.select_action(pre(obs(fr[0]["observation.state"], f)))).squeeze(0).cpu().numpy()
                              for f in fr])
                ratio.append((np.ptp(p[:, :3], 0) / rng).mean())
                for sd in range(2):
                    torch.manual_seed(sd)
                    pol.reset()
                    p = np.array([post(pol.select_action(pre(obs(fr[0]["observation.state"], fr[0])))).squeeze(0).cpu().numpy()
                                  for _ in range(90)])
                    start.append(float(np.abs(p[:, :3] - fr[0]["observation.state"].numpy()).max()))
        del pol
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{name:32s} replay err {np.mean(rep_err):5.2f} deg | imgs_only motion {np.mean(ratio)*100:5.0f}% of demo "
              f"| start moves (deg) {np.round(start, 1).tolist()}", flush=True)


if __name__ == "__main__":
    main()
