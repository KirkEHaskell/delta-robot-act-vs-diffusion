# Scaling-study results (training side)
_Generated 2026-10-02 11:37. All runs: batch 32, seed 1000, fp32, lerobot 0.5.0, torch 2.6.0, policy defaults, pyav decoding._

Copy a checkpoint's `pretrained_model/` folder to the laptop for inference. Paths are relative to `outputs/train/`.

**ACT cells** come from one 50k run per data size (ACT's LR is constant, so its 5k/20k checkpoints are exactly what a separate 5k/20k run would produce). **Diffusion cells** are separate runs (its cosine LR schedule depends on total steps).

**Loss values are not comparable across policies** (ACT: L1 + KL; Diffusion: noise-prediction MSE), and training loss is not task success - judge the grid by robot success rate.

| policy | episodes | steps | status | loss at end of cell | checkpoints (`<run>/checkpoints/<step>/pretrained_model`) |
|---|---|---|---|---|---|
| act | 50 | 5k | done | 0.1914 | `act_n50_s50k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| act | 50 | 20k | done | 0.0662 | `act_n50_s50k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| act | 50 | 50k | done | 0.0405 | `act_n50_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| act | 10 | 5k | done | 0.1419 | `act_n10_s50k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| act | 10 | 20k | done | 0.0444 | `act_n10_s50k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| act | 10 | 50k | done | 0.0253 | `act_n10_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| act | 25 | 5k | done | 0.1678 | `act_n25_s50k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| act | 25 | 20k | done | 0.0561 | `act_n25_s50k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| act | 25 | 50k | done | 0.0329 | `act_n25_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| act | 100 | 5k | done | 0.2162 | `act_n100_s50k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| act | 100 | 20k | done | 0.0812 | `act_n100_s50k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| act | 100 | 50k | done | 0.0487 | `act_n100_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 10 | 5k | done | 0.0099 | `diffusion_n10_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 25 | 5k | done | 0.0127 | `diffusion_n25_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 50 | 5k | done | 0.0147 | `diffusion_n50_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 100 | 5k | done | 0.0156 | `diffusion_n100_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 10 | 20k | done | 0.0029 | `diffusion_n10_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 25 | 20k | done | 0.0048 | `diffusion_n25_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 50 | 20k | done | 0.0068 | `diffusion_n50_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 100 | 20k | done | 0.0087 | `diffusion_n100_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 10 | 50k | done | 0.0011 | `diffusion_n10_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 25 | 50k | done | 0.0022 | `diffusion_n25_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 50 | 50k | done | 0.0037 | `diffusion_n50_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 100 | 50k | done | 0.0058 | `diffusion_n100_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |

## Runs

| run | wall time | step/s | loss start -> end |
|---|---|---|---|
| act_n50_s50k | 3.17 h | 4.38 | 2.560 -> 0.0405 |
| act_n10_s50k | 3.42 h | 4.06 | 2.533 -> 0.0253 |
| act_n25_s50k | 3.23 h | 4.30 | 2.547 -> 0.0329 |
| act_n100_s50k | 3.13 h | 4.44 | 2.552 -> 0.0487 |
| diffusion_n10_s5k | 0.45 h | 3.09 | 0.244 -> 0.0090 |
| diffusion_n25_s5k | 0.41 h | 3.35 | 0.241 -> 0.0120 |
| diffusion_n50_s5k | 0.40 h | 3.44 | 0.241 -> 0.0140 |
| diffusion_n100_s5k | 0.40 h | 3.50 | 0.243 -> 0.0150 |
| diffusion_n10_s20k | 1.71 h | 3.24 | 0.148 -> 0.0029 |
| diffusion_n25_s20k | 1.59 h | 3.49 | 0.146 -> 0.0048 |
| diffusion_n50_s20k | 1.56 h | 3.56 | 0.146 -> 0.0068 |
| diffusion_n100_s20k | 1.53 h | 3.63 | 0.147 -> 0.0087 |
| diffusion_n10_s50k | 4.25 h | 3.27 | 0.148 -> 0.0011 |
| diffusion_n25_s50k | 4.00 h | 3.48 | 0.146 -> 0.0022 |
| diffusion_n50_s50k | 3.85 h | 3.61 | 0.146 -> 0.0037 |
| diffusion_n100_s50k | 3.81 h | 3.64 | 0.147 -> 0.0058 |

## Plots

- `compare_act_by_data.png`
- `compare_diffusion_s5k_by_data.png`
- `compare_diffusion_s20k_by_data.png`
- `compare_diffusion_s50k_by_data.png`
- `compare_diffusion_n10_by_steps.png`
- `compare_diffusion_n25_by_steps.png`
- `compare_diffusion_n50_by_steps.png`
- `compare_diffusion_n100_by_steps.png`
- `act_n50_s50k_plot.png`
- `act_n10_s50k_plot.png`
- `act_n25_s50k_plot.png`
- `act_n100_s50k_plot.png`
- `diffusion_n10_s5k_plot.png`
- `diffusion_n25_s5k_plot.png`
- `diffusion_n50_s5k_plot.png`
- `diffusion_n100_s5k_plot.png`
- `diffusion_n10_s20k_plot.png`
- `diffusion_n25_s20k_plot.png`
- `diffusion_n50_s20k_plot.png`
- `diffusion_n100_s20k_plot.png`
- `diffusion_n10_s50k_plot.png`
- `diffusion_n25_s50k_plot.png`
- `diffusion_n50_s50k_plot.png`
- `diffusion_n100_s50k_plot.png`
