---
license: mit
library_name: lerobot
pipeline_tag: robotics
tags:
- LeRobot
- act
- imitation-learning
- delta-robot
- pick-and-place
- real-world
datasets:
- KirkHaskell/delta_robot_bolt_pick_place
---

# ACT: delta robot bolt pick-and-place (100 demos, 50k steps)

An [ACT](https://arxiv.org/abs/2304.13705) policy trained with LeRobot 0.5.0 on all 100
demonstrations of [`KirkHaskell/delta_robot_bolt_pick_place`](https://huggingface.co/datasets/KirkHaskell/delta_robot_bolt_pick_place).
It drives a home-built 3-DOF delta robot with an electromagnet gripper: it picks up a standing bolt
and drops it into a cup.

**Real-robot result: 6/10 successes** (95% Wilson CI 31–83%). It was tied for the best ACT model in
a 12-model ACT grid, and it was never stopped as unsafe. Its failures were 4 stalls or timeouts. It
succeeded at both out-of-distribution trials (bolt outside the demonstrated area). Full study:
<https://github.com/KirkEHaskell/delta-robot-act-vs-diffusion>

| | |
|---|---|
| Inputs | `observation.state` (3 joint angles, deg) + 3 cameras (`side`, `top_right`, `top_left`, 240×320) |
| Output | 4-D action: 3 absolute joint targets (deg) + magnet (on if > 0.5) |
| Settings | LeRobot ACT defaults: ResNet-18 (ImageNet), chunk size 100, all 100 actions executed, VAE (KL weight 10), LR 1e-5 |
| Training | 50k steps, batch 32, seed 1000, fp32, one RTX 3090 (~3.1 h) |
| Inference speed | 0.1–0.37 s per 100-action chunk on a laptop CPU (Ryzen AI 9 HX 370), real time at 30 Hz |

## Use

```bash
hf download KirkHaskell/act_delta_robot_bolt_pick_place --local-dir models/act
cd robot && python run_policy.py dry --ckpt ../models/act
```

Normalization lives outside the policy in LeRobot 0.5.x. Load the bundled pre/post-processors with
`make_pre_post_processors(policy_cfg, pretrained_path=...)`, or the actions come out normalized.

**Limitations:** this model is specific to one robot, camera placement, scene and camera colour
settings (two cameras have a fixed white balance). Expect to fine-tune on your own demonstrations.
