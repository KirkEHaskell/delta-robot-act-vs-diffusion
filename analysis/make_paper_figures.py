"""
make_paper_figures.py -- publication-style figures (vector PDF) for paper/main.tex.

  python analysis/make_paper_figures.py

Reads results/trials_clean.csv (written by analyze_results.py), results/training/training_curves_all_runs.csv,
the photos/videos in media/ and, if present, the dataset (for the camera-view figure; env DELTA_DATASET,
default data/delta_robot_bolt_pick_place). Writes paper/figures/*.pdf (+ .jpg for photo panels).

Style: serif text, thin marks, light horizontal grid, colour-blind-validated categorical slots with a
distinct marker per series so every figure also reads in greyscale print.
"""

import os
import subprocess

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

COL_W, TXT_W = 3.4, 7.0                     # inches: one column / full text width
INK, INK2, GRID = "#111111", "#4a4a4a", "#e3e3e3"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"]   # categorical slots 1-5, fixed order
MARK = ["o", "s", "^", "D", "v"]
POL = {"ACT": (C[0], "o"), "Diffusion": (C[1], "s")}
POL_NAME = {"ACT": "ACT", "Diffusion": "Diffusion Policy (v2)"}
DEMOS, STEPS = [10, 25, 50, 100], [5000, 20000, 50000]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    if os.environ.get("PREVIEW_DIR"):                      # optional PNG previews
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")),
                    bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


df = pd.read_csv(os.path.join(ROOT, "results", "trials_clean.csv"))
cell = df.groupby(["policy", "demos", "steps"]).agg(k=("success", "sum"), n=("success", "size")).reset_index()
cell["rate"] = cell.k / cell.n
cell[["lo", "hi"]] = [wilson(k, n) for k, n in zip(cell.k, cell.n)]
pols = ["ACT", "Diffusion"]

# ---------------------------------------------------------------- 1. success grid (heatmap)
fig, axes = plt.subplots(1, 2, figsize=(TXT_W * 0.78, 2.35), constrained_layout=True)
for ax, pol in zip(axes, pols):
    M = np.zeros((len(DEMOS), len(STEPS)))
    for i, d in enumerate(DEMOS):
        for j, s in enumerate(STEPS):
            M[i, j] = cell[(cell.policy == pol) & (cell.demos == d) & (cell.steps == s)].k.iloc[0]
    im = ax.imshow(M / 10, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(DEMOS)):
        for j in range(len(STEPS)):
            v = int(M[i, j])
            ax.text(j, i, f"{v}/10", ha="center", va="center", fontsize=8.5,
                    color="white" if v >= 6 else INK, fontweight="bold" if v == M.max() else "normal")
    ax.set_xticks(range(3), ["5k", "20k", "50k"])
    ax.set_yticks(range(4), [str(d) for d in DEMOS])
    ax.set_xlabel("Training steps")
    ax.set_title(POL_NAME[pol], loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Demonstrations")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
cb.set_label("Success rate", fontsize=8)
cb.outline.set_visible(False)
save(fig, "success_grid.pdf")

# ---------------------------------------------------------------- 2. success vs demos (CI), per policy
fig, axes = plt.subplots(1, 2, figsize=(TXT_W, 2.3), sharey=True, constrained_layout=True)
for ax, pol in zip(axes, pols):
    for idx, s in enumerate(STEPS):
        r = cell[(cell.policy == pol) & (cell.steps == s)].sort_values("demos")
        off = (idx - 1) * 0.035                       # small horizontal dodge (log axis)
        x = r.demos * 10 ** off
        ax.errorbar(x, r.rate, yerr=[r.rate - r.lo, r.hi - r.rate], color=C[idx], marker=MARK[idx],
                    lw=1.3, ms=4.5, capsize=2, elinewidth=0.7, label=f"{s // 1000}k steps")
    ax.set_xscale("log")
    ax.set_xticks(DEMOS, [str(d) for d in DEMOS])
    ax.minorticks_off()
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Training demonstrations (log scale)")
    ax.set_title(POL_NAME[pol], loc="left")
    ygrid(ax)
axes[0].set_ylabel("Success rate")
axes[0].legend(loc="upper left", handlelength=1.6)
save(fig, "success_vs_demos.pdf")

# ---------------------------------------------------------------- 3. success vs steps, per policy
fig, axes = plt.subplots(1, 2, figsize=(TXT_W, 2.2), sharey=True, constrained_layout=True)
for ax, pol in zip(axes, pols):
    for idx, d in enumerate(DEMOS):
        r = cell[(cell.policy == pol) & (cell.demos == d)].sort_values("steps")
        ax.plot(r.steps, r.rate, color=C[idx], marker=MARK[idx], label=f"{d} demos")
    ax.set_xscale("log")
    ax.set_xticks(STEPS, ["5k", "20k", "50k"])
    ax.minorticks_off()
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Training steps (log scale)")
    ax.set_title(POL_NAME[pol], loc="left")
    ygrid(ax)
axes[0].set_ylabel("Success rate")
axes[0].legend(loc="upper left", handlelength=1.6)
save(fig, "success_vs_steps.pdf")

# ---------------------------------------------------------------- 4. outcomes per policy (stacked, counts)
modes = ["success", "KnockedBoltOver", "Stalled or timed out", "Unsafe", "Dropped in transit"]
mname = {"success": "Success", "KnockedBoltOver": "Knocked bolt over", "Stalled or timed out": "Stalled / timed out",
         "Unsafe": "Unsafe (operator stop)", "Dropped in transit": "Dropped in transit"}
hatch = {"success": "", "KnockedBoltOver": "////", "Stalled or timed out": "....", "Unsafe": "xxxx",
         "Dropped in transit": ""}
fig, ax = plt.subplots(figsize=(COL_W, 1.55), constrained_layout=True)
plt.rcParams["hatch.linewidth"] = 0.4
for i, pol in enumerate(pols):
    vc = df[df.policy == pol].failure_mode.value_counts()
    left = 0
    for j, m in enumerate(modes):
        w = int(vc.get(m, 0))
        if w:
            ax.barh(i, w, left=left, height=0.62, color=C[j], edgecolor="white", linewidth=0.8,
                    hatch=hatch[m], label=mname[m] if i == 0 or pol == "ACT" else None)
            if w >= 8:
                ax.text(left + w / 2, i, str(w), ha="center", va="center", fontsize=7.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.12", fc=C[j], ec="none"))
            left += w
ax.set_yticks([0, 1], [POL_NAME[p] for p in pols])
ax.invert_yaxis()
ax.set_xlim(0, 120)
ax.set_xlabel("Trials (of 120 per policy)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
h, l = ax.get_legend_handles_labels()
seen = dict(zip(l, h))
ax.legend(seen.values(), seen.keys(), ncol=3, loc="lower left", bbox_to_anchor=(-0.32, 1.02),
          fontsize=6.8, handlelength=1.2, columnspacing=0.8)
save(fig, "outcomes_by_policy.pdf")

# ---------------------------------------------------------------- 5. success by bolt position
P = ["P1", "P2", "P3", "P4", "P5"]
pdesc = ["P1", "P2", "P3", "P4\n(occluded)", "P5\n(out of dist.)"]
fig, ax = plt.subplots(figsize=(COL_W, 1.9), constrained_layout=True)
w = 0.38
for k, pol in enumerate(pols):
    sub = df[df.policy == pol]
    ks = [sub[sub.position == p].success.sum() for p in P]
    ns = [(sub.position == p).sum() for p in P]
    x = np.arange(5) + (k - 0.5) * w
    ax.bar(x, np.array(ks) / np.array(ns), width=w - 0.04, color=POL[pol][0],
           hatch="" if pol == "ACT" else "////", edgecolor="white", linewidth=0.6, label=POL_NAME[pol])
    for xi, kk, nn in zip(x, ks, ns):
        ax.text(xi, kk / nn + 0.015, f"{kk}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(range(5), pdesc)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, 0.62)
ax.set_ylabel("Success rate (24 trials each)")
ax.legend(loc="upper right", fontsize=7)
ygrid(ax)
save(fig, "success_by_position.pdf")

# ---------------------------------------------------------------- 6. copycat ablation (images-only motion)
# Values from analysis/diag_copycat.py runs reported in the training handoff (n50 / 5k pilots, ACT n50).
labels = ["ACT\n(default)", "Diffusion\nv1 default", "+ pretrained\nvision, crop", "+ state\nblinded", "+ 1.6 s\nhorizon (v2)"]
vals = [0.92, 0.37, 0.42, 0.96, 0.95]
cols = [C[0], C[1], C[1], C[1], C[1]]
fig, ax = plt.subplots(figsize=(COL_W, 1.95), constrained_layout=True)
bars = ax.bar(range(5), vals, width=0.62, color=cols, edgecolor="white", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(5), labels, fontsize=7)
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("Demo motion reproduced\nfrom images alone")
ax.axvline(0.5, color=GRID, lw=0.8)
ygrid(ax)
save(fig, "copycat_ablation.pdf")

# ---------------------------------------------------------------- 7. training loss curves (50k runs)
tc = pd.read_csv(os.path.join(ROOT, "results", "training", "training_curves_all_runs.csv"))
fig, axes = plt.subplots(1, 2, figsize=(TXT_W, 2.3), constrained_layout=True)
spec = [("ACT (L1 + KL)", tc[(tc.grid == "v1_lerobot_defaults") & (tc.policy == "act")]),
        ("Diffusion Policy v2 (noise MSE)", tc[(tc.grid == "v2_diffusion_fixed") & (tc.total_steps == 50000)])]
X0 = 500                                        # skip the first few hundred steps' initial plunge
for ax, (title, sub) in zip(axes, spec):
    ys = []
    for idx, d in enumerate(DEMOS):
        r = sub[sub.n_demos == d].sort_values("step")
        y = np.exp(np.log(r.train_loss).rolling(7, center=True, min_periods=1).mean())   # smooth in log space
        keep = r.step >= X0
        ax.plot(r.step[keep], y[keep], color=C[idx], lw=1.2, label=f"{d} demos")
        ax.plot(r.step.iloc[-1], y.iloc[-1], marker=MARK[idx], color=C[idx], ms=4)
        ys.append(y[keep])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X0, 50000 * 1.15)
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000, 50000], ["500", "1k", "2k", "5k", "10k", "20k", "50k"])
    ax.minorticks_off()
    yt = [0.03, 0.05, 0.1, 0.2, 0.5, 1, 2] if title.startswith("ACT") else [0.002, 0.005, 0.01, 0.02, 0.05]
    ax.set_yticks(yt, [f"{v:g}" for v in yt])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Training step (log scale)")
    ax.set_title(title, loc="left")
    ax.grid(which="major", axis="both", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Training loss (log scale)")
for ax in axes:
    ax.legend(loc="lower left", handlelength=1.6)
save(fig, "training_loss.pdf")

# ---------------------------------------------------------------- 8. photo panels (JPG for LaTeX)
PH = os.path.join(ROOT, "media", "photos")


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


ff("-i", os.path.join(PH, "leader_and_follower.jpg"), "-vf", "scale=1000:-2", "-q:v", "3",
   os.path.join(OUT, "robots.jpg"))
ff("-i", os.path.join(PH, "follower_robot_overview.jpg"), "-vf", "scale=700:-2", "-q:v", "3",
   os.path.join(OUT, "follower.jpg"))
ff("-i", os.path.join(PH, "leader_arm_top_closeup.jpg"), "-vf", "scale=700:-2", "-q:v", "3",
   os.path.join(OUT, "leader_closeup.jpg"))
# filmstrip of the best Diffusion model's run (6 frames)
vid = os.path.join(ROOT, "media", "diffusion_best_n100_s50k.mp4")
ff("-i", vid, "-vf", "fps=6/12.5,scale=-2:420,tile=6x1:padding=8:color=white", "-frames:v", "1", "-q:v", "3",
   os.path.join(OUT, "filmstrip_diffusion.jpg"))
print("wrote photo panels")

# camera views (one frame from each camera of one demo), if the dataset is available
demo = os.path.join(ROOT, "media", "demos", "episode_042.mp4")
if os.path.exists(demo):
    ff("-ss", "4", "-i", demo, "-frames:v", "1", "-q:v", "2", os.path.join(OUT, "camera_views.jpg"))
    print("wrote camera_views.jpg")
