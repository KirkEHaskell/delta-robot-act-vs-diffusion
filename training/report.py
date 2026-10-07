#!/usr/bin/env python3
"""Parse, plot and summarize scaling-study training runs.

A run's logs are segments: <name>.log (starts at step 0) plus any
<name>.resume_from<NNNNNN>.log (starts at that checkpoint step). Later segments
override steps an earlier, crashed segment had already logged past its last
checkpoint, so curves are exact across resumes.

    python training/report.py [--v2]         # (re)write RESULTS.md + comparison plots from finished runs
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM = r"(nan|-?inf|[0-9.eE+-]+)"
LINE_RE = re.compile(r"INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ step:\S+ .*?loss:" + NUM + r" grdn:" + NUM)
LOGFREQ_RE = re.compile(r"(?:'log_freq':\s*|--log_freq=)(\d+)")
RESUME_RE = re.compile(r"\.resume_from(\d+)\.log$")


def segments(train_dir, name):
    segs = []
    base = Path(train_dir) / f"{name}.log"
    if base.exists():
        segs.append((0, os.path.getmtime(base), base))
    for p in Path(train_dir).glob(f"{name}.resume_from*.log"):
        segs.append((int(RESUME_RE.search(p.name).group(1)), os.path.getmtime(p), p))
    return sorted(segs)


def load_run(train_dir, name):
    by_step = {}
    wall_s = 0.0
    steps_timed = 0
    for start, _, path in segments(train_dir, name):
        text = path.read_text(errors="replace")
        m = LOGFREQ_RE.search(text)
        log_freq = int(m.group(1)) if m else 100
        pts = LINE_RE.findall(text)
        for i, (ts, loss, grdn) in enumerate(pts):
            by_step[start + (i + 1) * log_freq] = (float(loss), float(grdn))
        if len(pts) >= 2:
            t0 = datetime.strptime(pts[0][0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(pts[-1][0], "%Y-%m-%d %H:%M:%S")
            wall_s += (t1 - t0).total_seconds() * len(pts) / (len(pts) - 1)
            steps_timed += len(pts) * log_freq
    steps = sorted(by_step)
    return {
        "name": name,
        "step": steps,
        "loss": [by_step[s][0] for s in steps],
        "grdn": [by_step[s][1] for s in steps],
        "wall_s": wall_s,
        "steps_per_s": steps_timed / wall_s if wall_s else float("nan"),
    }


def loss_at(curve, step, window=10):
    """Mean logged loss over the last `window` log points at or before `step`."""
    vals = [l for s, l in zip(curve["step"], curve["loss"]) if s <= step][-window:]
    return sum(vals) / len(vals) if vals else float("nan")


def plot_runs(train_dir, names, out, labels=None, title=None, logy=True, max_step=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, name in enumerate(names):
        c = load_run(train_dir, name)
        st, lo, gr = c["step"], c["loss"], c["grdn"]
        if max_step:
            k = sum(1 for s in st if s <= max_step)
            st, lo, gr = st[:k], lo[:k], gr[:k]
        lab = labels[i] if labels else name
        ax1.plot(st, lo, lw=1.1, label=lab)
        ax2.plot(st, gr, lw=0.9, label=lab)
    ax1.set_ylabel("training loss")
    ax2.set_ylabel("grad norm")
    ax2.set_xlabel("step")
    if logy:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title(title or ", ".join(names))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cells(runs):
    """Grid cells -> (policy, n, steps, run_name, [checkpoint steps])."""
    out = []
    for name, policy, n, steps, save_freq in runs:
        if policy == "act":
            # one 50k run serves all three compute cells (constant LR)
            for cell_steps in (5_000, 20_000, 50_000):
                every = cell_steps // 10
                out.append((policy, n, cell_steps, name, list(range(every, cell_steps + 1, every))))
        else:
            out.append((policy, n, steps, name, list(range(save_freq, steps + 1, save_freq))))
    return out


def final_report(train_dir, runs):
    train_dir = Path(train_dir)
    done = {r[0] for r in runs if (train_dir / r[0] / "DONE.json").exists()}
    curves = {name: load_run(train_dir, name) for name in done}
    plots = []

    act = [r[0] for r in runs if r[1] == "act" and r[0] in done]
    act.sort(key=lambda s: int(re.search(r"_n(\d+)_", s).group(1)))
    if act:
        p = train_dir / "compare_act_by_data.png"
        plot_runs(train_dir, act, p, title="ACT: loss vs step by #episodes (batch 32)")
        plots.append(p)
    for steps in (5_000, 20_000, 50_000):
        ds = [r[0] for r in runs if r[1] == "diffusion" and r[3] == steps and r[0] in done]
        ds.sort(key=lambda s: int(re.search(r"_n(\d+)_", s).group(1)))
        if ds:
            p = train_dir / f"compare_diffusion_s{steps // 1000}k_by_data.png"
            plot_runs(train_dir, ds, p, title=f"Diffusion {steps // 1000}k steps: by #episodes (batch 32)")
            plots.append(p)
    for n in (10, 25, 50, 100):
        ds = [r[0] for r in runs if r[1] == "diffusion" and r[2] == n and r[0] in done]
        if len(ds) > 1:
            p = train_dir / f"compare_diffusion_n{n}_by_steps.png"
            plot_runs(train_dir, ds, p, title=f"Diffusion n={n}: by compute budget (cosine LR per run)")
            plots.append(p)

    lines = [
        "# Scaling-study results (training side)",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. All runs: batch 32, seed 1000, fp32, lerobot 0.5.0, "
        "torch 2.6.0, policy defaults, pyav decoding._",
        "",
        f"Paths are relative to `{train_dir.name}/`.",
        "",
        "**ACT cells** come from one 50k run per data size (ACT's LR is constant, so its 5k/20k "
        "checkpoints are exactly what a separate 5k/20k run would produce). **Diffusion cells** are "
        "separate runs (its cosine LR schedule depends on total steps).",
        "",
        "**Loss values are not comparable across policies** (ACT: L1 + KL; Diffusion: noise-prediction MSE), "
        "and training loss is not task success - judge the grid by robot success rate.",
        "",
        "| policy | episodes | steps | status | loss at end of cell | checkpoints (`<run>/checkpoints/<step>/pretrained_model`) |",
        "|---|---|---|---|---|---|",
    ]
    for policy, n, steps, run, ck in cells(runs):
        if run in done:
            ls = f"{loss_at(curves[run], steps):.4f}"
            status = "done"
        else:
            ls = "-"
            status = "FAILED" if (train_dir / run / "FAILED.json").exists() else "pending"
        cks = f"`{run}/checkpoints/` " + ", ".join(f"{s:06d}" for s in ck)
        lines.append(f"| {policy} | {n} | {steps // 1000}k | {status} | {ls} | {cks} |")
    lines += ["", "## Runs", "", "| run | wall time | step/s | loss start -> end |", "|---|---|---|---|"]
    for r in runs:
        if r[0] in done:
            info = json.loads((train_dir / r[0] / "DONE.json").read_text())
            lines.append(f"| {r[0]} | {info['wall_h']:.2f} h | {info['steps_per_s']:.2f} | "
                         f"{info['start_loss']:.3f} -> {info['final_loss']:.4f} |")
    lines += ["", "## Plots", "", *[f"- `{p.name}`" for p in plots],
              *[f"- `{r[0]}_plot.png`" for r in runs if r[0] in done]]
    (train_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from run_grid import ROOT, grid, grid_v2  # noqa: E402

    if "--v2" in sys.argv:
        final_report(ROOT / "outputs/train_v2", grid_v2())
    else:
        final_report(ROOT / "outputs/train", grid())
