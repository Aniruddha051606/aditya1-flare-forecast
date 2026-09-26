# Table 1. Data

| instrument | role | band | study data | observed days | native cadence | model grid | model features |
|---|---|---|---|---|---|---|---|
| SoLEXS (Aditya-L1) | model input | soft X-ray | 831 days | 764.9 | 1 s light curve, 340-channel spectra | 20 s | 24 |
| HEL1OS (Aditya-L1) | model input | hard X-ray | 2668 products | 968.4 | 1 s light curves (readout batches 2-8 s) | 20 s | 48 |
| GOES-18 XRS | truth only (labels, flux target, scoring) | 0.1-0.8 nm | 8511 labelled flares >= C1.0 |  | 1 min | 20 s (targets) | 0 |

Labelled flares by class: {'C': 7045, 'M': 1385, 'X': 81}.
Split: training 2023-12-01 to 2025-08-07, validation 2025-09-03 to 2026-03-03, test 2026-03-30 to 2026-09-21; 27-day embargoes.

Source files and script: scripts/paper/make_tables.py (2026-09-26 05:25:16 UTC).
