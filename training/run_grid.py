#!/usr/bin/env python3
"""Unattended driver for the DeltaRobot ACT-vs-Diffusion scaling grid.

Runs every grid run sequentially on the one GPU. Idempotent: re-launching it
skips finished runs (DONE marker), resumes interrupted ones from their last
checkpoint, and restarts runs that died before their first checkpoint.

    python training/run_grid.py --console          # ACT + default-Diffusion grid  -> outputs/train
    python training/run_grid.py --v2 --console     # the fixed Diffusion grid     -> outputs/train_v2
    (unattended on Linux: setsid nohup python training/run_grid.py > outputs/train/queue.out 2>&1 &)

Dataset: env DELTA_DATASET (default data/delta_robot_bolt_pick_place, i.e. the Hugging Face dataset
downloaded there). Run from anywhere; paths are relative to the repo root.

Graceful stop: `touch outputs/train/STOP` -> the queue exits after the current
run finishes (the current run is not interrupted).
Status:        outputs/train/STATUS.md (rewritten on every event).

Fixed-for-fairness settings for every run: batch_size=32, seed=1000,
pyav video backend, fp32 (lerobot default), policy defaults otherwise.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root
DATASET = Path(os.environ.get("DELTA_DATASET", ROOT / "data/delta_robot_bolt_pick_place")).resolve()
# lerobot-train + fix for the 0.5.0 episode-subset sampler bug (see training/lerobot_train_fixed.py)
TRAIN = [sys.executable, str(ROOT / "training/lerobot_train_fixed.py")]

BATCH_SIZE = 32
SEED = 1000
NUM_WORKERS = 8
LOG_FREQ = 100
MAX_ATTEMPTS = 3
MIN_FREE_GB = 30
CONSOLE = False      # --console: run in the foreground terminal, show live training output

# Nested, seeded subsets: random.Random(42).shuffle(range(100)), first N. n=100 -> all episodes.
EPISODES = {
    10: [1, 9, 15, 41, 42, 50, 65, 70, 78, 91],
    25: [1, 9, 10, 15, 21, 30, 32, 37, 41, 42, 45, 48, 50, 55, 56, 65, 70, 72, 73, 76, 78, 80, 91, 92, 96],
    50: [1, 7, 8, 9, 10, 12, 15, 18, 21, 23, 26, 30, 32, 33, 34, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48,
         49, 50, 52, 55, 56, 59, 60, 61, 63, 65, 66, 70, 72, 73, 74, 76, 78, 80, 83, 85, 87, 91, 92, 96, 98],
    100: None,
}


# ---- v2 Diffusion grid (2026-10-05): fixes for the "copycat" stall found in robot testing ----
# ImageNet-pretrained ResNet18 (+BatchNorm), random-crop augmentation, 1 obs frame, joint state
# BLINDED (zeroed, env BLIND_STATE=1 in lerobot_train_fixed.py), horizon 48 / execute 24
# (= the Diffusion Policy paper's 1.6 s / 0.8 s at 10 Hz, rescaled to this dataset's 30 Hz).
V2_FLAGS = [
    "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1", "--policy.use_group_norm=false",
    "--policy.crop_shape=[216, 288]", "--policy.crop_is_random=true",
    "--policy.n_obs_steps=1", "--policy.horizon=48", "--policy.n_action_steps=24",
    "--policy.drop_n_last_frames=24",
]
V2_ENV = {"BLIND_STATE": "1"}


def grid_v2():
    """Diffusion only, cosine LR -> one run per cell. Order = most useful first (the queue stops at
    the deadline): all 5k cells, then 20k (n50, n100, n25, n10), then 50k."""
    runs = []
    for n in (50, 10, 25, 100):
        runs.append((f"diffusionnostateh48_n{n}_s5k", "diffusion", n, 5_000, 500))
    for n in (50, 100, 25, 10):
        runs.append((f"diffusionnostateh48_n{n}_s20k", "diffusion", n, 20_000, 2_000))
    for n in (50, 100, 25, 10):
        runs.append((f"diffusionnostateh48_n{n}_s50k", "diffusion", n, 50_000, 5_000))
    return runs


def grid():
    """(name, policy, n_episodes, steps, save_freq) in execution order.

    ACT has a constant LR (no scheduler), so one 50k run per data size also yields the
    s5k / s20k cells: its checkpoints at 5000 / 20000 are what separate runs would produce.
    save_freq=500 for ACT so the checkpoint sets of the s5k (every 500), s20k (every 2000)
    and s50k (every 5000) cells are all contained in it.
    Diffusion uses a cosine LR schedule tied to --steps, so every cell is its own run.
    """
    runs = []
    for n in (50, 10, 25, 100):
        runs.append((f"act_n{n}_s50k", "act", n, 50_000, 500))
    for steps in (5_000, 20_000, 50_000):
        for n in (10, 25, 50, 100):
            runs.append((f"diffusion_n{n}_s{steps // 1000}k", "diffusion", n, steps, steps // 10))
    return runs


class Queue:
    def __init__(self, train_dir: Path, runs, extra_flags=(), extra_env=None, deadline=None, rate=None):
        self.train_dir = train_dir
        self.runs = runs
        self.extra_flags = list(extra_flags)
        self.extra_env = dict(extra_env or {})
        self.deadline = deadline          # datetime or None: don't START a run that can't finish by then
        self.rate = rate                  # steps/s used for that estimate
        self.events = []
        self.current = None
        self.started = datetime.now()

    # ---------- status ----------
    def event(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        if CONSOLE:                                   # stand out between lerobot's progress output
            print(f"\n{'=' * 100}\n>>> QUEUE: {line}\n{'=' * 100}", flush=True)
            with open(self.train_dir / "queue.out", "a") as q:
                q.write(line + "\n")
        else:
            print(line, flush=True)
        self.events.append(line)
        self.write_status()

    def state_of(self, name):
        d = self.train_dir / name
        if (d / "DONE.json").exists():
            return "done"
        if (d / "FAILED.json").exists():
            return "FAILED"
        if name == self.current:
            return "running"
        return "queued"

    def write_status(self):
        rows = []
        for name, policy, n, steps, _ in self.runs:
            st = self.state_of(name)
            extra = ""
            if st == "done":
                info = json.loads((self.train_dir / name / "DONE.json").read_text())
                extra = f"final loss {info['final_loss']:.4f}, {info['wall_h']:.2f} h, {info['steps_per_s']:.2f} step/s"
            elif st == "running":
                extra = self.progress(name)
            rows.append(f"| {name} | {st} | {extra} |")
        free = shutil.disk_usage(self.train_dir).free / 1e9
        txt = [
            "# Scaling-study training status",
            f"_Updated {datetime.now():%Y-%m-%d %H:%M:%S} · queue started {self.started:%Y-%m-%d %H:%M} · "
            f"disk free {free:.0f} GB · batch {BATCH_SIZE}, seed {SEED}_",
            "",
            "| run | state | info |",
            "|---|---|---|",
            *rows,
            "",
            "## Events",
            *[f"- {e}" for e in self.events[-40:]],
        ]
        (self.train_dir / "STATUS.md").write_text("\n".join(txt) + "\n")

    def progress(self, name):
        logs = sorted(self.train_dir.glob(f"{name}*.log"), key=os.path.getmtime)
        if not logs:
            return "starting"
        with open(logs[-1], "rb") as f:
            f.seek(max(0, os.path.getsize(logs[-1]) - 4000))
            tail = f.read().decode(errors="replace")
        m = re.findall(r"(\d+)/(\d+) \[([0-9:]+)<([0-9:]+)", tail)
        return f"segment progress {m[-1][0]}/{m[-1][1]}, ETA {m[-1][3]}" if m else "starting"

    # ---------- run handling ----------
    def last_ckpt_step(self, out):
        last = out / "checkpoints/last"
        if not last.exists():
            return None
        try:
            return int(json.loads((last / "training_state/training_step.json").read_text())["step"])
        except Exception:
            return None

    def train_cmd(self, name, policy, n, steps, save_freq, out):
        cmd = TRAIN + [
            "--dataset.repo_id=local/delta_robot_bolt_pick_place",
            f"--dataset.root={DATASET}",
            f"--policy.type={policy}", "--policy.push_to_hub=false",
            "--dataset.video_backend=pyav", "--policy.device=cuda",
            f"--batch_size={BATCH_SIZE}", f"--num_workers={NUM_WORKERS}", f"--seed={SEED}",
            f"--steps={steps}", f"--save_freq={save_freq}", f"--log_freq={LOG_FREQ}",
            f"--output_dir={out}",
        ] + self.extra_flags
        if EPISODES[n] is not None:
            cmd.append(f"--dataset.episodes={json.dumps(EPISODES[n])}")
        return cmd

    def run_one(self, name, policy, n, steps, save_freq):
        out = self.train_dir / name
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.wait_for_disk()
            resume_step = self.last_ckpt_step(out)
            if resume_step is not None and resume_step >= steps:
                break  # training finished; only post-processing was missing
            if resume_step is not None:
                cfg = out / "checkpoints/last/pretrained_model/train_config.json"
                cmd = TRAIN + ["--resume=true", f"--config_path={cfg}"]
                log = self.train_dir / f"{name}.resume_from{resume_step:06d}.log"
                self.event(f"{name}: attempt {attempt}, RESUMING from step {resume_step}")
            else:
                if out.exists():
                    shutil.rmtree(out)  # died before first checkpoint: nothing to resume
                cmd = self.train_cmd(name, policy, n, steps, save_freq, out)
                log = self.train_dir / f"{name}.log"
                self.event(f"{name}: attempt {attempt}, starting fresh")
            with open(log, "w") as f:
                f.write("CMD: " + " ".join(cmd) + "\n")
                f.flush()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if CONSOLE else f,
                                        stderr=subprocess.STDOUT, cwd=ROOT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1", **self.extra_env})
                if CONSOLE:                           # tee training output to the log AND the terminal
                    import threading

                    def _tee():
                        while True:
                            chunk = proc.stdout.read1(4096)
                            if not chunk:
                                break
                            txt = chunk.decode(errors="replace")
                            f.write(txt)
                            f.flush()
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                    tee = threading.Thread(target=_tee, daemon=True)
                    tee.start()
                last_prune = time.time()
                while proc.poll() is None:            # live-prune optimizer state while training
                    time.sleep(1)
                    if time.time() - last_prune > 60:
                        self.prune_live(out)
                        last_prune = time.time()
                if CONSOLE:
                    tee.join(timeout=5)
                rc = proc.returncode
            if rc == 0 and self.last_ckpt_step(out) == steps:
                break
            self.event(f"{name}: attempt {attempt} exited rc={rc}, last checkpoint "
                       f"{self.last_ckpt_step(out)} (log {log.name})")
            time.sleep(30)  # let GPU memory / dataloader workers clear
        else:
            return self.fail(name, f"no successful completion after {MAX_ATTEMPTS} attempts")
        return self.finalize(name, policy, n, steps, save_freq)

    def prune_live(self, out):
        """Drop training_state (optimizer, ~2 GB for Diffusion) from every checkpoint except the
        newest one that `last` points to -- resuming only ever needs `last`. Never touches a
        checkpoint newer than `last` (it may still be being written)."""
        try:
            last = int(os.path.basename(os.readlink(out / "checkpoints/last")))
        except OSError:
            return
        for d in (out / "checkpoints").glob("[0-9]*"):
            if int(d.name) < last and (d / "training_state").exists():
                shutil.rmtree(d / "training_state", ignore_errors=True)

    def finalize(self, name, policy, n, steps, save_freq):
        out = self.train_dir / name
        # 1) every expected checkpoint exists and is loadable-complete
        need = ["config.json", "model.safetensors", "train_config.json",
                "policy_preprocessor.json", "policy_postprocessor.json"]
        expected = list(range(save_freq, steps + 1, save_freq))
        missing = []
        for s in expected:
            pm = out / f"checkpoints/{s:06d}/pretrained_model"
            for fn in need:
                if not (pm / fn).exists():
                    missing.append(f"{s:06d}/{fn}")
            if not list(pm.glob("*normalizer_processor.safetensors")):
                missing.append(f"{s:06d}/normalizer")
        if missing:
            return self.fail(name, f"incomplete checkpoints: {missing[:5]}")
        # 2) loss curve sanity
        sys.path.insert(0, str(ROOT / "training"))
        import report  # noqa: E402
        curve = report.load_run(self.train_dir, name)
        losses = curve["loss"]
        if not losses:
            return self.fail(name, "no loss lines parsed from logs")
        bad = [l for l in losses if l != l or l in (float("inf"), float("-inf"))]
        if bad:
            return self.fail(name, f"{len(bad)} non-finite loss values")
        k = max(1, min(10, len(losses) // 10))
        head = sum(losses[:k]) / k
        tail = sum(losses[-k:]) / k
        if not tail < head:
            return self.fail(name, f"loss did not decrease (start {head:.4f}, end {tail:.4f})")
        # 3) prune optimizer state from all but the final checkpoint (only needed for resuming)
        freed = 0
        for s in expected[:-1]:
            ts = out / f"checkpoints/{s:06d}/training_state"
            if ts.exists():
                freed += sum(p.stat().st_size for p in ts.rglob("*") if p.is_file())
                shutil.rmtree(ts)
        # 4) plot + DONE marker
        try:
            report.plot_runs(self.train_dir, [name], self.train_dir / f"{name}_plot.png")
        except Exception as e:  # plotting must never block the queue
            self.event(f"{name}: plot failed ({e!r})")
        info = {
            "name": name, "policy": policy, "n_episodes": n, "steps": steps, "save_freq": save_freq,
            "batch_size": BATCH_SIZE, "seed": SEED,
            "final_loss": tail, "start_loss": head, "wall_h": curve["wall_s"] / 3600,
            "steps_per_s": curve["steps_per_s"], "checkpoints": [f"{s:06d}" for s in expected],
            "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "pruned_gb": freed / 1e9,
        }
        (out / "DONE.json").write_text(json.dumps(info, indent=2))
        self.event(f"{name}: DONE - loss {head:.3f} -> {tail:.4f}, {info['wall_h']:.2f} h, "
                   f"{len(expected)} checkpoints verified, pruned {freed / 1e9:.1f} GB optimizer state")
        return True

    def fail(self, name, why):
        (self.train_dir / name).mkdir(parents=True, exist_ok=True)
        (self.train_dir / name / "FAILED.json").write_text(json.dumps({"reason": why}, indent=2))
        self.event(f"{name}: FAILED - {why}")
        return False

    def wait_for_disk(self):
        while shutil.disk_usage(self.train_dir).free / 1e9 < MIN_FREE_GB:
            self.event(f"PAUSED: less than {MIN_FREE_GB} GB free disk; re-checking in 5 min")
            time.sleep(300)

    def main(self):
        self.event(f"queue started: {len(self.runs)} runs")
        for name, policy, n, steps, save_freq in self.runs:
            if (self.train_dir / "STOP").exists():
                self.event("STOP file found - exiting before next run")
                break
            st = self.state_of(name)
            if st in ("done", "FAILED"):
                continue
            if self.deadline and self.rate:
                resume = self.last_ckpt_step(self.train_dir / name) or 0
                need_s = (steps - resume) / self.rate + 240          # + startup/checkpoint overhead
                if resume < steps and datetime.now().timestamp() + need_s > self.deadline.timestamp():
                    self.event(f"DEADLINE: {name} needs ~{need_s / 60:.0f} min, would end after "
                               f"{self.deadline:%H:%M} -> stopping the queue here")
                    break
            self.current = name
            self.write_status()
            try:
                self.run_one(name, policy, n, steps, save_freq)
            except Exception as e:
                self.fail(name, f"driver exception {e!r}")
            self.current = None
            self.write_status()
        else:
            self.event("queue finished")
            try:
                sys.path.insert(0, str(ROOT / "training"))
                import report
                report.final_report(self.train_dir, self.runs)
                self.event("final report written")
            except Exception as e:
                self.event(f"final report failed ({e!r})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="tiny 2-run queue in outputs/queue_test")
    ap.add_argument("--v2", action="store_true", help="the fixed Diffusion grid in outputs/train_v2")
    ap.add_argument("--deadline", help="HH:MM -- don't start a run that can't finish by then")
    ap.add_argument("--rate", type=float, default=3.5, help="steps/s used for the deadline estimate")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--console", action="store_true",
                    help="foreground: stream live training output + queue messages to this terminal "
                         "(Ctrl+C stops; re-run to resume)")
    a = ap.parse_args()
    MIN_FREE_GB = a.min_free_gb
    CONSOLE = a.console
    deadline = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        deadline = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline < datetime.now():
            deadline = deadline.replace(day=deadline.day) + __import__("datetime").timedelta(days=1)
    flags, env = (), None
    if a.test:
        runs = [("act_n10_s300", "act", 10, 300, 100), ("diffusion_n10_s300", "diffusion", 10, 300, 100)]
        d = ROOT / "outputs/queue_test"
    elif a.v2:
        runs, d, flags, env = grid_v2(), ROOT / "outputs/train_v2", V2_FLAGS, V2_ENV
    else:
        runs, d = grid(), ROOT / "outputs/train"
    d.mkdir(parents=True, exist_ok=True)
    Queue(d, runs, flags, env, deadline, a.rate).main()
