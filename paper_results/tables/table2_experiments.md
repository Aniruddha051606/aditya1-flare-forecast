# Table 2. Experiments

| experiment | inputs | model.inputs | seed | parameters | best epoch | modality dropout | training windows | validation windows | test windows (native) |
|---|---|---|---|---|---|---|---|---|---|
| E1 | SoLEXS only | soft | 1337 | 463615 | 26 | off | 220758 | 101541 | 107912 |
| E2 | HEL1OS only | hard | 1337 | 463615 | 18 | off | 303434 | 118296 | 116206 |
| E4 | SoLEXS + HEL1OS | both | 1337 | 463615 | 48 | 0.25 | 307030 | 120062 | 118622 |

Common test windows (both instruments observing): 105,512 on 162 days.
Same code, split, labels, seed and settings for all; only model.inputs differs (and modality dropout, which needs two instruments).

Source files and script: scripts/paper/make_tables.py (2026-09-26 03:04:45 UTC).
