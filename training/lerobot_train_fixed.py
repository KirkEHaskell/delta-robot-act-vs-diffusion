#!/usr/bin/env python3
"""`lerobot-train` with a fix for a lerobot 0.5.0 bug: --dataset.episodes + EpisodeAwareSampler.

Policies with `drop_n_last_frames` (Diffusion) use EpisodeAwareSampler, which yields
*absolute* frame indices from dataset.meta. With an episode subset the dataset is indexed
by *position within the loaded subset*, so training crashes with
`IndexError: Invalid key: 17189 is out of bounds for size 3219` (or, worse, would read the
wrong frames). The dataset already holds the absolute->relative map
(`_absolute_to_relative_idx`); this wrapper remaps the sampler's indices through it.
No-op when all episodes are used (map is None) and for ACT (no EpisodeAwareSampler).
Same CLI as lerobot-train.
"""
import os
import sys

import lerobot.scripts.lerobot_train as lt
from lerobot.datasets.sampler import EpisodeAwareSampler

_datasets = []
_orig_make_dataset = lt.make_dataset


def _make_dataset(cfg):
    ds = _orig_make_dataset(cfg)
    _datasets.append(ds)
    return ds


class SubsetSafeEpisodeAwareSampler(EpisodeAwareSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        abs_to_rel = getattr(_datasets[-1], "_absolute_to_relative_idx", None) if _datasets else None
        if abs_to_rel is not None:
            self.indices = [abs_to_rel[i] for i in self.indices]
            print(f"[lerobot_train_fixed] remapped {len(self.indices)} sampler indices to subset positions "
                  f"(max {max(self.indices)} < {len(abs_to_rel)})", flush=True)


lt.make_dataset = _make_dataset
lt.EpisodeAwareSampler = SubsetSafeEpisodeAwareSampler

# Opt-in (env BLIND_STATE=1): train Diffusion WITHOUT joint-state information. lerobot's Diffusion
# requires observation.state to exist, so it's replaced by zeros (after normalization) in every
# training batch. Used to break the "copycat" shortcut (copying the follower's joint state instead
# of using the cameras). Inference must blind it the same way -- robot/run_policy.py
# does that for checkpoints named *nostate* or holding a BLIND_STATE file.
if os.environ.get("BLIND_STATE") == "1":
    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import OBS_STATE

    _orig_forward = DiffusionPolicy.forward

    def _blind_forward(self, batch, *a, **kw):
        batch = dict(batch)
        batch[OBS_STATE] = torch.zeros_like(batch[OBS_STATE])
        return _orig_forward(self, batch, *a, **kw)

    DiffusionPolicy.forward = _blind_forward
    print("[lerobot_train_fixed] BLIND_STATE=1: observation.state zeroed in every Diffusion batch", flush=True)

if __name__ == "__main__":
    sys.exit(lt.main())
