# Scaling-study results (training side)
_Generated 2026-10-05 19:49. All runs: batch 32, seed 1000, fp32, lerobot 0.5.0, torch 2.6.0, policy defaults, pyav decoding._

Copy a checkpoint's `pretrained_model/` folder to the laptop for inference. Paths are relative to `outputs/train/`.

**ACT cells** come from one 50k run per data size (ACT's LR is constant, so its 5k/20k checkpoints are exactly what a separate 5k/20k run would produce). **Diffusion cells** are separate runs (its cosine LR schedule depends on total steps).

**Loss values are not comparable across policies** (ACT: L1 + KL; Diffusion: noise-prediction MSE), and training loss is not task success - judge the grid by robot success rate.

| policy | episodes | steps | status | loss at end of cell | checkpoints (`<run>/checkpoints/<step>/pretrained_model`) |
|---|---|---|---|---|---|
| diffusion | 50 | 5k | done | 0.0134 | `diffusionnostateh48_n50_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 10 | 5k | done | 0.0088 | `diffusionnostateh48_n10_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 25 | 5k | done | 0.0119 | `diffusionnostateh48_n25_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 100 | 5k | done | 0.0138 | `diffusionnostateh48_n100_s5k/checkpoints/` 000500, 001000, 001500, 002000, 002500, 003000, 003500, 004000, 004500, 005000 |
| diffusion | 50 | 20k | done | 0.0065 | `diffusionnostateh48_n50_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 100 | 20k | done | 0.0090 | `diffusionnostateh48_n100_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 25 | 20k | done | 0.0042 | `diffusionnostateh48_n25_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 10 | 20k | done | 0.0030 | `diffusionnostateh48_n10_s20k/checkpoints/` 002000, 004000, 006000, 008000, 010000, 012000, 014000, 016000, 018000, 020000 |
| diffusion | 50 | 50k | done | 0.0040 | `diffusionnostateh48_n50_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 100 | 50k | done | 0.0061 | `diffusionnostateh48_n100_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 25 | 50k | done | 0.0025 | `diffusionnostateh48_n25_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |
| diffusion | 10 | 50k | done | 0.0019 | `diffusionnostateh48_n10_s50k/checkpoints/` 005000, 010000, 015000, 020000, 025000, 030000, 035000, 040000, 045000, 050000 |

## Runs

| run | wall time | step/s | loss start -> end |
|---|---|---|---|
| diffusionnostateh48_n50_s5k | 0.36 h | 3.84 | 0.213 -> 0.0130 |
| diffusionnostateh48_n10_s5k | 0.36 h | 3.91 | 0.216 -> 0.0080 |
| diffusionnostateh48_n25_s5k | 0.34 h | 4.12 | 0.216 -> 0.0112 |
| diffusionnostateh48_n100_s5k | 0.33 h | 4.20 | 0.215 -> 0.0134 |
| diffusionnostateh48_n50_s20k | 1.26 h | 4.40 | 0.126 -> 0.0065 |
| diffusionnostateh48_n100_s20k | 1.26 h | 4.42 | 0.127 -> 0.0090 |
| diffusionnostateh48_n25_s20k | 1.29 h | 4.29 | 0.128 -> 0.0042 |
| diffusionnostateh48_n10_s20k | 1.36 h | 4.09 | 0.129 -> 0.0030 |
| diffusionnostateh48_n50_s50k | 3.10 h | 4.57 | 0.126 -> 0.0040 |
| diffusionnostateh48_n100_s50k | 2.97 h | 4.67 | 0.127 -> 0.0061 |
| diffusionnostateh48_n25_s50k | 3.05 h | 4.55 | 0.128 -> 0.0025 |
| diffusionnostateh48_n10_s50k | 3.23 h | 4.30 | 0.129 -> 0.0019 |

## Plots

- `compare_diffusion_s5k_by_data.png`
- `compare_diffusion_s20k_by_data.png`
- `compare_diffusion_s50k_by_data.png`
- `compare_diffusion_n10_by_steps.png`
- `compare_diffusion_n25_by_steps.png`
- `compare_diffusion_n50_by_steps.png`
- `compare_diffusion_n100_by_steps.png`
- `diffusionnostateh48_n50_s5k_plot.png`
- `diffusionnostateh48_n10_s5k_plot.png`
- `diffusionnostateh48_n25_s5k_plot.png`
- `diffusionnostateh48_n100_s5k_plot.png`
- `diffusionnostateh48_n50_s20k_plot.png`
- `diffusionnostateh48_n100_s20k_plot.png`
- `diffusionnostateh48_n25_s20k_plot.png`
- `diffusionnostateh48_n10_s20k_plot.png`
- `diffusionnostateh48_n50_s50k_plot.png`
- `diffusionnostateh48_n100_s50k_plot.png`
- `diffusionnostateh48_n25_s50k_plot.png`
- `diffusionnostateh48_n10_s50k_plot.png`
