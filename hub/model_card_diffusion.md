---
license: mit
library_name: lerobot
pipeline_tag: robotics
tags:
- LeRobot
- diffusion-policy
- imitation-learning
- delta-robot
- pick-and-place
- real-world
datasets:
- KirkHaskell/delta_robot_bolt_pick_place
---

# Diffusion Policy: delta robot bolt pick-and-place (100 demos, 50k steps), 10/10 on the real robot

A [Diffusion Policy](https://arxiv.org/abs/2303.04137) trained with LeRobot 0.5.0 on all 100
demonstrations of [`KirkHaskell/delta_robot_bolt_pick_place`](https://huggingface.co/datasets/KirkHaskell/delta_robot_bolt_pick_place).
It drives a home-built 3-DOF delta robot with an electromagnet gripper: it picks up a standing bolt
and drops it into a cup.

**Real-robot result: 10/10 successes** (95% Wilson CI 72–100%). It succeeded at all 5 bolt
positions, including one partly hidden by the cup and one outside the demonstrated area, and every
trial needed a single grab attempt. It was the best of the 24 models in the study, and subjectively
the most impressive. Full study: <https://github.com/KirkEHaskell/delta-robot-act-vs-diffusion>

## ⚠️ This is a modified Diffusion Policy: the joint-state input must be zeroed

LeRobot's default Diffusion Policy failed on this robot. Trained on leader–follower teleoperation
data, it learned to copy its own joint state instead of using the cameras, and it never left the
start pose. This model differs from the defaults in four ways:

| Setting | LeRobot default | This model |
|---|---|---|
| Vision backbone | ResNet-18 from scratch, GroupNorm | ResNet-18 ImageNet-pretrained, BatchNorm |
| Augmentation | none | random crop 216×288 |
| Observation frames | 2 | 1 |
| Joint state | used | **zeroed after normalization, in training and at inference** |
| Horizon / executed actions | 16 / 8 | 48 / 24 (1.6 s / 0.8 s at 30 Hz) |

The checkpoint's config cannot express "state blinded." The model still has an `observation.state`
input, but it was trained with that input set to zeros. **At inference, zero
`batch["observation.state"]` after the preprocessor and before `select_action`.** The repo's
`robot/run_policy.py` does this automatically: this folder contains a `BLIND_STATE` marker file
that the script detects.

```python
batch = preprocessor(obs)
batch["observation.state"] = torch.zeros_like(batch["observation.state"])
action = postprocessor(policy.select_action(batch))
```

| | |
|---|---|
| Inputs | 3 cameras (`side`, `top_right`, `top_left`, 240×320); the joint state is present but zeroed |
| Output | 4-D action: 3 absolute joint targets (deg) + magnet (on if > 0.5) |
| Training | 50k steps, cosine LR, batch 32, seed 1000, fp32, one RTX 3090 (~3 h), with `BLIND_STATE=1` via `training/lerobot_train_fixed.py` |
| Inference in the study | DDPM, all 100 denoising steps, RTX 3090: ~0.76 s per 24-action plan. The robot pauses while it plans |
| CPU | DDPM-100 takes ~18 s per plan. DDIM with 10 steps takes ~2 s and matched DDPM-100 within ~1° offline, but was not used for the scored trials |

## Use

```bash
hf download KirkHaskell/diffusion_delta_robot_bolt_pick_place --local-dir models/diffusion
cd robot && python run_policy.py dry --ckpt ../models/diffusion
```

**Limitations:** this model is specific to one robot, camera placement, scene and camera colour
settings (two cameras have a fixed white balance). It is not real-time without a GPU. Undertrained
versions of this configuration tend to stall rather than act.
