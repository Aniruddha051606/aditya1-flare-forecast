# Table 7c. Separately trained soft-only vs soft + hard rise-phase models (existing ablation, 3 seeds)

| seed | peak log-MAE soft only (dex) | peak log-MAE soft + hard (dex) | reduction (dex) | relative | flares |
|---|---|---|---|---|---|
| 1337 | 0.2187 | 0.1965 | 0.0222 [0.0187, 0.0261] | 0.102 | 6120 |
| 42 | 0.2190 | 0.2027 | 0.0162 [0.0126, 0.0197] | 0.074 | 6120 |
| 7 | 0.2176 | 0.1987 | 0.0189 [0.0155, 0.0228] | 0.087 | 6120 |

From outputs/ablations/hel1os/hel1os_value.json: holds up: every seed's interval excludes zero. Peak flux predicted during the rise, 3 time-ordered folds; intervals: bootstrap over flares (1000 draws, seed 0).

Source files and script: scripts/paper/make_tables.py (2026-09-25 20:26:51 UTC).
