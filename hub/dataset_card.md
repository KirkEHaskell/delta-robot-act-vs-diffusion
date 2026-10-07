---
license: cc-by-4.0
task_categories:
- robotics
tags:
- LeRobot
- imitation-learning
- teleoperation
- delta-robot
- parallel-robot
- pick-and-place
- electromagnet
- real-world
- multi-view
pretty_name: Delta Robot Bolt Pick-and-Place (3 cameras, 100 demos)
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Delta Robot Bolt Pick-and-Place: 100 teleoperated demonstrations, 3 cameras

100 real-world demonstrations of a **3-DOF delta (parallel) robot** with an **electromagnet
gripper**. In each one, the robot picks up a standing steel bolt and drops it into a cup. The data
was recorded by leader–follower teleoperation and is stored in **LeRobot v3.0** format.

This is the dataset behind the study *ACT vs. Diffusion Policy on a Home-Built Delta Robot*
(240 scored real-robot trials, 24 trained models). Code, CAD, firmware and the full report are at
<https://github.com/KirkEHaskell/delta-robot-act-vs-diffusion>.

## At a glance

| | |
|---|---|
| Robot | 3-DOF delta robot (`robot_type: delta_3dof`), 3× Feetech STS3215 servos, Teensy 4.1 |
| Gripper | Electromagnet, binary on/off |
| Episodes / frames | 100 / 33,200 |
| Frequency | 30 Hz |
| Episode length | 8.1–17.3 s (median 10.7 s) |
| Cameras | `side`, `top_right`, `top_left`: 240×320 RGB, AV1 video |
| Task string | `"pick up the bolt and drop it in the bucket"` (single task) |
| Teleoperation | Leader–follower; the leader is a hand-moved arm with AS5600 magnetic encoders |
| License | CC BY 4.0. Commercial use and use in training data are fine, with attribution |

## Features

| Key | Shape | Meaning |
|---|---|---|
| `observation.state` | (3,) float32 | Follower (robot) joint angles j1, j2, j3 in **degrees** (servo frame, ~90° = arm horizontal; usable range 30–110°) |
| `action` | (4,) float32 | Leader joint angles j1, j2, j3 in **degrees** + `mag`, the magnet command (0 = off, 1 = on) |
| `observation.images.side` | (240, 320, 3) | Side view |
| `observation.images.top_right` | (240, 320, 3) | Overhead view, right |
| `observation.images.top_left` | (240, 320, 3) | Overhead view, left |

The actions are **absolute joint-space targets**, not Cartesian poses and not deltas. The robot's
firmware clamps them to 30–110° and drives the servos to them directly.

## Things you should know before using it

- **Action ≈ next state.** Because of leader–follower teleoperation, the action (leader angle) is
  always very close to the next observed state (follower angle). In this study, LeRobot's default
  Diffusion Policy exploited this and learned to copy the joint state instead of using the
  cameras. The robot never moved. Blinding the state input fixed it. If you train on this data,
  check that your policy is image-driven.
- **Colour cast.** The `side` and `top_right` cameras have a constant yellow-green cast in every
  episode (auto white balance off, fixed exposure). `top_left` is neutral. The cast is consistent
  across all 100 episodes.
- **Idle segments.** Each episode starts with about 1.2 s of stillness, and there is a short pause
  (median 0.33 s) after the magnet grabs.
- **One grasp per episode.** 99 of 100 episodes contain exactly one magnet activation. No failed
  grasps or retries are demonstrated.
- **Single scene.** One table, one cup position, bolt placed at various spots within a demonstrated
  region. The lighting is indoor and roughly constant.
- **One operator, one robot,** recorded on one day (2026-09-16, about 15:30–19:15) in 10 sessions of 10 episodes.

## Quick start

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("KirkHaskell/delta_robot_bolt_pick_place", video_backend="pyav")
frame = ds[0]
print(frame["observation.state"], frame["action"], frame["observation.images.side"].shape)
```

The study trained on nested subsets of 10/25/50/100 episodes. The exact episode lists are in
`training/run_grid.py` in the GitHub repo.

## Results obtained with this data (10 real-robot trials per model)

| Model | Trained on | Success |
|---|---|---|
| Diffusion Policy (pretrained ResNet-18, crop augmentation, state blinded, 1.6 s horizon), 50k steps | all 100 episodes | **10/10** |
| ACT (LeRobot defaults), 50k steps | all 100 episodes | 6/10 |
| Either policy, 10 episodes or 5k steps | — | ≤ 2/10 |

Full grid and analysis: see the GitHub repo.

## Citation

```bibtex
@misc{delta_act_vs_diffusion_2026,
  title  = {ACT vs. Diffusion Policy on a Home-Built Delta Robot: Data and Training-Length Scaling for Real-World Pick-and-Place},
  author = {Kirk Haskell},
  year   = {2026},
  url    = {https://github.com/KirkEHaskell/delta-robot-act-vs-diffusion}
}
```
