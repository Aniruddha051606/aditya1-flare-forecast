# Table 7. Robustness to a missing instrument (common test windows)

| model | AUC in progress | TSS in progress | AUC within 15 min | MAE flux +15 min (dex) |
|---|---|---|---|---|
| E4 | 0.951 [0.946, 0.956] | 0.780 [0.765, 0.792] | 0.831 [0.815, 0.847] | 0.075 [0.070, 0.079] |
| E4, HEL1OS withheld | 0.946 [0.940, 0.950] | 0.774 [0.760, 0.786] | 0.807 [0.792, 0.821] | 0.081 [0.076, 0.085] |
| E1 | 0.946 [0.941, 0.951] | 0.771 [0.759, 0.783] | 0.809 [0.796, 0.823] | 0.077 [0.073, 0.081] |
| E4, SoLEXS withheld | 0.925 [0.918, 0.931] | 0.725 [0.710, 0.741] | 0.783 [0.768, 0.798] | 0.330 [0.308, 0.354] |
| E2 | 0.920 [0.915, 0.926] | 0.727 [0.710, 0.743] | 0.763 [0.750, 0.775] | 0.336 [0.317, 0.358] |

'withheld': the deployed E4 with one instrument's mask set to zero at prediction time (thresholds and calibration unchanged). E1 and E2 are trained on one instrument. 95% intervals resampling whole test days.

Source files and script: scripts/paper/make_tables.py (2026-09-26 05:25:18 UTC).
