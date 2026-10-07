"""Offline: does DDIM-10 plan like the trained DDPM-100? Held-out episode, several plan points.

    python analysis/ddim_vs_ddpm_check.py <diffusion_checkpoint_dir> <episode> [dataset_root]

Compares each 8-action plan to the recorded actions (and DDIM to DDPM)."""
import os, sys, time, numpy as np, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
torch.set_num_threads(os.cpu_count() or 1)
CK = sys.argv[1]; EP = int(sys.argv[2])
ROOT = sys.argv[3] if len(sys.argv) > 3 else "data/delta_robot_bolt_pick_place"
ds = LeRobotDataset("local/delta_robot_bolt_pick_place", root=ROOT, episodes=[EP], video_backend="pyav")
keys = ["observation.state"] + [k for k in ds.features if k.startswith("observation.images.")]
L = len(ds); points = list(range(1, L - 8, max(1, (L - 9) // 8)))[:8]

def load(sched, steps):
    cfg = PreTrainedConfig.from_pretrained(CK); cfg.device = "cpu"
    cfg.noise_scheduler_type = sched; cfg.num_inference_steps = steps
    p = get_policy_class(cfg.type).from_pretrained(CK, config=cfg).eval()
    pre, post = make_pre_post_processors(policy_cfg=p.config, pretrained_path=CK,
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    return p, pre, post

def plans(sched, steps):
    p, pre, post = load(sched, steps); out = []; t0 = time.perf_counter()
    for i in points:
        torch.manual_seed(0); p.reset()
        obs = pre({k: ds[i][k] for k in keys})   # after reset, obs queue is padded with copies of obs i
        with torch.inference_mode():
            acts = [p.select_action(obs) for _ in range(8)]   # first call plans, next 7 pop the queue
        out.append(np.stack([post(a).squeeze(0).numpy() for a in acts]))
    print(f"{sched}-{steps}: {(time.perf_counter()-t0)/len(points):.1f}s per plan", flush=True)
    return np.array(out)

gt = np.array([[ds[i + k]["action"].numpy() for k in range(8)] for i in points])
ddpm = plans("DDPM", 100); ddim = plans("DDIM", 10)
np.set_printoptions(precision=2, suppress=True)
print("plan points:", points, "of", L)
print("mean |err| vs recorded, joints (deg) [j1 j2 j3], magnet:")
for n, a in (("DDPM-100", ddpm), ("DDIM-10", ddim)):
    e = np.abs(a - gt).mean((0, 1)); print(f"  {n}: {e[:3]}  mag {e[3]:.2f}")
print("mean |DDIM-10 - DDPM-100| joints:", np.abs(ddim - ddpm).mean((0, 1))[:3])
print("magnet agreement DDIM vs DDPM:", ((ddim[..., 3] > .5) == (ddpm[..., 3] > .5)).mean())
