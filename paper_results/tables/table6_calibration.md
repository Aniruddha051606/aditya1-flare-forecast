# Table 6. Probability calibration (each experiment's own test windows)

| experiment | target (>= C1) | Brier raw | Brier calibrated | BSS raw | BSS calibrated | windows |
|---|---|---|---|---|---|---|
| E1 | flare in progress | 0.0595 | 0.0297 | 0.204 | 0.603 [0.577, 0.626] | 106570 |
| E1 | flare within 15 min | 0.1317 | 0.0788 | -0.098 | 0.343 [0.322, 0.360] | 106570 |
| E1 | flare within 30 min | 0.1686 | 0.1173 | -0.091 | 0.241 [0.220, 0.258] | 106570 |
| E1 | flare within 60 min | 0.2130 | 0.1709 | -0.059 | 0.150 [0.123, 0.170] | 106570 |
| E2 | flare in progress | 0.0809 | 0.0376 | -0.095 | 0.491 [0.467, 0.510] | 114785 |
| E2 | flare within 15 min | 0.1657 | 0.0852 | -0.401 | 0.279 [0.262, 0.294] | 114785 |
| E2 | flare within 30 min | 0.2037 | 0.1236 | -0.337 | 0.189 [0.171, 0.201] | 114785 |
| E2 | flare within 60 min | 0.2522 | 0.1788 | -0.270 | 0.099 [0.077, 0.113] | 114785 |
| E4 | flare in progress | 0.0597 | 0.0296 | 0.199 | 0.603 [0.574, 0.627] | 117185 |
| E4 | flare within 15 min | 0.1251 | 0.0769 | -0.048 | 0.355 [0.335, 0.372] | 117185 |
| E4 | flare within 30 min | 0.1608 | 0.1128 | -0.047 | 0.265 [0.246, 0.282] | 117185 |
| E4 | flare within 60 min | 0.2021 | 0.1617 | -0.013 | 0.190 [0.167, 0.209] | 117185 |

Isotonic calibration fitted on each experiment's validation windows only. BSS against the test-period base rate (solarflare.metrics.brier_skill_score).

Source files and script: scripts/paper/make_tables.py (2026-09-26 03:04:46 UTC).
