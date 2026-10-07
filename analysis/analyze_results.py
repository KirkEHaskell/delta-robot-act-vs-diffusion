"""
analyze_results.py

Turns the robot trial log into clean tables, statistics and figures.

  python analysis/analyze_results.py [path/to/trials.csv or .xlsx]

Default input: results/robot_trials.csv (one row per trial, as scored by the operator). An .xlsx
with a "Trials" sheet in the same columns also works. Outputs go to results/ and figures/.

Scoring rules:
  * success = "In cup" == Y.
  * Rows marked "Unsafe" with Picked/In cup left blank = the operator stopped the robot. They are
    counted as FAILURES (picked = 0, success = 0), not dropped.
  * Grab attempts are as typed by the operator (count printed by the test script); 10 was often
    typed for stalled/timed-out runs, so treat attempts as approximate.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "robot_trials.csv")
OUT_DATA, OUT_FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(OUT_FIG, exist_ok=True)
LOG = open(os.path.join(OUT_DATA, "stats_output.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


# ---------------------------------------------------------------- load + clean
if SRC.lower().endswith(".xlsx"):
    df = pd.read_excel(SRC, sheet_name="Trials")
else:
    df = pd.read_csv(SRC)
df = df[df["Checkpoint"].notna()]
filled = df[["Picked up (Y/N)", "In cup (Y/N)", "Grab attempts", "Failure mode"]].notna().any(axis=1)
df = df[filled].copy()
df["unsafe_stop"] = df["Failure mode"].astype(str).str.lower().eq("unsafe") & df["In cup (Y/N)"].isna()
df["picked"] = (df["Picked up (Y/N)"] == "Y").astype(int)
df["success"] = (df["In cup (Y/N)"] == "Y").astype(int)
df["failure_mode"] = np.where(df["success"] == 1, "success", df["Failure mode"].fillna("unknown"))
df["Demos"] = df["Demos"].astype(int)
df["Train steps"] = df["Train steps"].astype(int)
df = df.rename(columns={"Trial #": "trial", "Checkpoint": "checkpoint", "Policy": "policy", "Demos": "demos",
                        "Train steps": "steps", "Position": "position", "Try": "try",
                        "Grab attempts": "grab_attempts", "Notes": "notes"})
keep = ["trial", "checkpoint", "policy", "demos", "steps", "position", "try", "picked", "success",
        "grab_attempts", "failure_mode", "unsafe_stop", "notes"]
df[keep].to_csv(os.path.join(OUT_DATA, "trials_clean.csv"), index=False)
say(f"source: {os.path.relpath(SRC, ROOT)}")
say(f"scored trials: {len(df)}  ({df['unsafe_stop'].sum()} operator-stopped 'Unsafe' rows counted as failures)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def trend_p(ks, ns, xs):
    """Cochran-Armitage trend test (two-sided)."""
    ks, ns, xs = map(np.asarray, (ks, ns, xs))
    N, p = ns.sum(), ks.sum() / ns.sum()
    if p in (0, 1):
        return np.nan
    T = np.sum(xs * (ks - ns * p))
    V = p * (1 - p) * (np.sum(ns * xs ** 2) - np.sum(ns * xs) ** 2 / N)
    return 2 * norm.sf(abs(T / np.sqrt(V)))


# ---------------------------------------------------------------- per model
g = df.groupby(["policy", "demos", "steps"])
by = g.agg(checkpoint=("checkpoint", "first"), n=("success", "size"), successes=("success", "sum"),
           picked=("picked", "sum"), unsafe_stops=("unsafe_stop", "sum"),
           mean_grab_attempts=("grab_attempts", lambda s: pd.to_numeric(s, errors="coerce").mean())).reset_index()
by["success_rate"] = by["successes"] / by["n"]
by["pick_rate"] = by["picked"] / by["n"]
ci = [wilson(k, n) for k, n in zip(by["successes"], by["n"])]
by["ci95_low"], by["ci95_high"] = [c[0] for c in ci], [c[1] for c in ci]
fm = pd.crosstab([df["policy"], df["demos"], df["steps"]], df["failure_mode"]).reset_index()
by = by.merge(fm, on=["policy", "demos", "steps"], how="left")
pos_ok = df[df.success == 1].groupby(["policy", "demos", "steps"])["position"].apply(lambda s: " ".join(sorted(set(s))))
by["positions_with_success"] = by.set_index(["policy", "demos", "steps"]).index.map(pos_ok).fillna("")
by.to_csv(os.path.join(OUT_DATA, "results_by_model.csv"), index=False)

STEPS, DEMOS = [5000, 20000, 50000], [10, 25, 50, 100]
say("\n=== Success (in cup) per cell: successes/trials ===")
for pol in sorted(df.policy.unique()):
    say(f"\n{pol}")
    say("demos \\ steps".ljust(15) + "".join(f"{s // 1000}k".rjust(10) for s in STEPS))
    for d in DEMOS:
        cells = []
        for s in STEPS:
            r = by[(by.policy == pol) & (by.demos == d) & (by.steps == s)]
            cells.append(f"{int(r.successes.iloc[0])}/{int(r.n.iloc[0])}" if len(r) else "not run")
        say(str(d).ljust(15) + "".join(c.rjust(10) for c in cells))

say("\n=== Trend tests (Cochran-Armitage, x = log2 of the varied quantity) ===")
for pol in sorted(df.policy.unique()):
    b = by[by.policy == pol]
    for s in STEPS:
        r = b[b.steps == s].sort_values("demos")
        if len(r) >= 3:
            say(f"{pol:9} demos trend @ {s // 1000:>2}k steps: {list(r.successes)} of {list(r.n)} "
                f"at demos {list(r.demos)}  p = {trend_p(r.successes, r.n, np.log2(r.demos)):.4f}")
    for d in DEMOS:
        r = b[b.demos == d].sort_values("steps")
        if len(r) >= 3:
            say(f"{pol:9} steps trend @ {d:>3} demos: {list(r.successes)} of {list(r.n)} "
                f"at steps {list(r.steps)}  p = {trend_p(r.successes, r.n, np.log2(r.steps)):.4f}")

say("\n=== Pooled by one factor (all cells of that level) ===")
for pol in sorted(df.policy.unique()):
    b = df[df.policy == pol]
    say(f"{pol}: by demos " + ", ".join(f"{d}: {b[b.demos == d].success.sum()}/{(b.demos == d).sum()}" for d in DEMOS)
        + " | by steps " + ", ".join(f"{s // 1000}k: {b[b.steps == s].success.sum()}/{(b.steps == s).sum()}" for s in STEPS))

say("\n=== ACT vs Diffusion, same cell (Fisher exact, two-sided) ===")
pols = sorted(df.policy.unique())
if len(pols) == 2:
    a, d_ = pols
    for dm in DEMOS:
        for s in STEPS:
            ra = by[(by.policy == a) & (by.demos == dm) & (by.steps == s)]
            rd = by[(by.policy == d_) & (by.demos == dm) & (by.steps == s)]
            if len(ra) and len(rd):
                ka, na, kd, nd = int(ra.successes.iloc[0]), int(ra.n.iloc[0]), int(rd.successes.iloc[0]), int(rd.n.iloc[0])
                p = fisher_exact([[ka, na - ka], [kd, nd - kd]])[1]
                say(f"  {dm:>3} demos {s // 1000:>2}k: {a} {ka}/{na} vs {d_} {kd}/{nd}  p = {p:.3f}")
    shared = by.groupby(["demos", "steps"]).policy.nunique()
    shared = shared[shared == 2].index
    tot = {p_: by[(by.policy == p_) & by.set_index(["demos", "steps"]).index.isin(shared)] for p_ in pols}
    ka, na = int(tot[a].successes.sum()), int(tot[a].n.sum())
    kd, nd = int(tot[d_].successes.sum()), int(tot[d_].n.sum())
    say(f"  pooled over the {len(shared)} cells both policies ran: {a} {ka}/{na} vs {d_} {kd}/{nd}  "
        f"p = {fisher_exact([[ka, na - ka], [kd, nd - kd]])[1]:.4f}")

say("\n=== Position effects (pooled success by bolt position) ===")
say(pd.crosstab(df.policy, df.position, values=df.success, aggfunc="mean").round(2).to_string())
pairs = df.pivot_table(index=["checkpoint", "position"], columns="try", values="success", aggfunc="first").dropna()
agree = (pairs.iloc[:, 0] == pairs.iloc[:, 1]).mean()
say(f"\nTry 1 and try 2 at the same position agree in {agree:.0%} of {len(pairs)} pairs "
    "(trials cluster by position, so each model's 10 trials act more like ~5 independent results).")

say("\n=== Failure modes (count of failed trials) ===")
say(pd.crosstab([df.policy], df.failure_mode).to_string())

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, INK2 = "#0b0b0b", "#52514e"
SERIES = {5000: "#2a78d6", 20000: "#eb6834", 50000: "#1baf7a"}    # validated categorical slots 1-3
POLCOL = {"ACT": "#2a78d6", "Diffusion": "#eb6834"}


def save(fig, name):
    fig.savefig(os.path.join(OUT_FIG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. success vs demos, one line per training length, one panel per policy
fig, axes = plt.subplots(1, len(pols), figsize=(5.2 * len(pols), 3.8), sharey=True)
axes = np.atleast_1d(axes)
for ax, pol in zip(axes, pols):
    for s in STEPS:
        r = by[(by.policy == pol) & (by.steps == s)].sort_values("demos")
        if r.empty:
            continue
        lo, hi = r.success_rate - r.ci95_low, r.ci95_high - r.success_rate
        ax.errorbar(r.demos, r.success_rate, yerr=[lo, hi], color=SERIES[s], lw=2, marker="o", ms=7,
                    capsize=3, elinewidth=1, alpha=0.95, label=f"{s // 1000}k steps")
    ax.set_xscale("log")
    ax.set_xticks(DEMOS, [str(d) for d in DEMOS])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Demonstrations used for training")
    ax.set_title(pol, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
axes[0].set_ylabel("Success rate (bolt in cup)")
axes[-1].legend(frameon=False, loc="upper left", labelcolor=INK2)
fig.suptitle("Success rate vs. amount of data (error bars: 95% Wilson interval, n = 10 per point)",
             color=INK2, fontsize=9, x=0.02, ha="left", y=0.995)
save(fig, "success_vs_demos.png")

# 2. grid heatmaps
fig, axes = plt.subplots(1, len(pols), figsize=(4.6 * len(pols), 3.6))
axes = np.atleast_1d(axes)
for ax, pol in zip(axes, pols):
    M = np.full((len(DEMOS), len(STEPS)), np.nan)
    for i, d in enumerate(DEMOS):
        for j, s in enumerate(STEPS):
            r = by[(by.policy == pol) & (by.demos == d) & (by.steps == s)]
            if len(r):
                M[i, j] = r.success_rate.iloc[0]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(DEMOS)):
        for j in range(len(STEPS)):
            v = M[i, j]
            txt = "not run" if np.isnan(v) else f"{v:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color=("white" if (not np.isnan(v) and v > 0.55) else INK))
    ax.set_xticks(range(len(STEPS)), [f"{s // 1000}k" for s in STEPS])
    ax.set_yticks(range(len(DEMOS)), [str(d) for d in DEMOS])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Demonstrations")
    ax.set_title(pol, color=INK, loc="left", fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
save(fig, "success_grid_heatmap.png")

# 3. success vs steps, one line per data size
fig, axes = plt.subplots(1, len(pols), figsize=(5.2 * len(pols), 3.8), sharey=True)
axes = np.atleast_1d(axes)
DCOL = {10: "#2a78d6", 25: "#eb6834", 50: "#1baf7a", 100: "#eda100"}
for ax, pol in zip(axes, pols):
    for d in DEMOS:
        r = by[(by.policy == pol) & (by.demos == d)].sort_values("steps")
        if len(r):
            ax.plot(r.steps, r.success_rate, color=DCOL[d], lw=2, marker="o", ms=7, label=f"{d} demos")
            ax.annotate(f"{d}", (r.steps.iloc[-1], r.success_rate.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(STEPS, ["5k", "20k", "50k"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Training steps")
    ax.set_title(pol, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
axes[0].set_ylabel("Success rate (bolt in cup)")
axes[-1].legend(frameon=False, loc="upper left", labelcolor=INK2)
save(fig, "success_vs_steps.png")

# 4. failure modes per policy (stacked shares)
modes = [m for m in ["success", "KnockedBoltOver", "Stalled or timed out", "Unsafe", "Dropped in transit"]
         if m in df.failure_mode.unique()]
mcol = {"success": "#1baf7a", "KnockedBoltOver": "#eb6834", "Stalled or timed out": "#2a78d6",
        "Unsafe": "#4a3aa7", "Dropped in transit": "#eda100"}
fig, ax = plt.subplots(figsize=(7, 1.0 + 0.6 * len(pols)))
for i, pol in enumerate(pols):
    sub = df[df.policy == pol].failure_mode.value_counts(normalize=True)
    left = 0
    for m in modes:
        w = sub.get(m, 0)
        if w:
            ax.barh(i, w, left=left, color=mcol[m], edgecolor="white", linewidth=2, height=0.6,
                    label=m if i == 0 else None)
            if w > 0.07:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", color="white", fontsize=9)
            left += w
ax.set_yticks(range(len(pols)), pols)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlim(0, 1)
ax.legend(frameon=False, ncol=len(modes), loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8, labelcolor=INK2)
save(fig, "outcomes_by_policy.png")

# 5. position effect
fig, ax = plt.subplots(figsize=(6, 3.2))
P = sorted(df.position.unique())
w = 0.38
for k, pol in enumerate(pols):
    rate = [df[(df.policy == pol) & (df.position == p)].success.mean() for p in P]
    ax.bar(np.arange(len(P)) + (k - 0.5) * w, rate, width=w - 0.02, color=POLCOL.get(pol, "#888"), label=pol)
ax.set_xticks(range(len(P)), P)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("Success rate, all models pooled")
ax.set_xlabel("Bolt position (P1-P4 inside the demonstrated area, P4 partly hidden by the cup;\n"
              "P5 outside the demonstrated area)")
ax.legend(frameon=False, labelcolor=INK2)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
save(fig, "success_by_position.png")

say(f"\nfigures written to {os.path.relpath(OUT_FIG, ROOT)}")
LOG.close()
