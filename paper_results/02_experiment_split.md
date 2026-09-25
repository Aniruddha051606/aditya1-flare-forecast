# Experiment split (frozen)

Rebuilt by `scripts/paper/data_and_split.py` with the training code and checked against the split recorded at training. one calendar cut: windows sorted by origin time, the first 60% train, next 20% validation, last 20% test, with every window within the embargo of a boundary dropped.

| Split | first origin (UTC) | last origin (UTC) | windows | both instruments at origin | in-flare rate | flare within 15 / 30 / 60 min | GOES flares (C/M/X) |
|---|---|---|---:|---:|---:|---|---|
| train | 2023-12-01 01:59:20 | 2025-08-07 20:49:40 | 307,030 | 218,313 | 0.158 | 0.269 / 0.369 / 0.535 | 5633 (4526/1047/60) |
| val | 2025-09-03 20:59:40 | 2026-03-03 03:51:40 | 120,062 | 100,614 | 0.104 | 0.177 / 0.241 / 0.348 | 1464 (1263/186/15) |
| test | 2026-03-30 03:53:40 | 2026-09-21 22:59:40 | 118,622 | 105,943 | 0.081 | 0.139 / 0.189 / 0.276 | 1090 (968/117/5) |

Training windows thinned from 414,171 to 307,030 (quiet training windows kept every 600 s instead of 120 s (training only)). Seed 1337.

## Checks (all must hold; the script stops otherwise)

- same_split_as_recorded_at_training: True
- splits_disjoint: True
- chronological: True
- gap_train_to_val_days: 27.01
- gap_val_to_test_days: 27.0
- embargo_days: 27.0
- gaps_at_least_embargo: True
- no_input_or_target_span_crosses_a_split: True
- input_window_s: 7200.0
- longest_horizon_s: 3600.0
