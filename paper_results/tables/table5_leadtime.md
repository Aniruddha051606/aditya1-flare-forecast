# Table 5. Alerts: detection, false alarms and lead time

| alert | method | operating point | validation false alarms/day | test false alarms/day | time alert on (test) | flares | detection rate | chance | event TSS | median lead (min) | IQR (min) | p10-p90 (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOES >= C1 flare within 15 min | network (E4) | best_TSS |  | 13.808 | 0.1409 | 968 | 0.998 | 0.487 | 0.511 | 20.0 | 8.0-34.0 |  |
| GOES >= C1 flare within 15 min | network (E4) | fa_1 | 0.86 | 0.493 | 0.0398 | 968 | 0.786 | 0.02 | 0.766 | 5.0 | 1.0-14.0 |  |
| GOES >= C1 flare within 15 min | network (E4) | fa_2 | 1.752 | 1.471 | 0.0527 | 968 | 0.905 | 0.061 | 0.844 | 6.0 | 1.0-16.0 | 0.0-31.0 |
| GOES >= C1 flare within 15 min | network (E4) | fa_5 | 4.615 | 3.842 | 0.0687 | 968 | 0.971 | 0.162 | 0.809 | 8.0 | 2.0-20.0 |  |
| GOES >= C1 flare within 15 min | SoLEXS trend (15-min extrapolation) | fa_2 | 1.982 | 1.391 | 0.0404 | 968 | 0.799 | 0.086 | 0.713 | 4.0 | 1.0-15.0 | 0.0-31.0 |
| GOES >= C1 flare within 15 min | SoLEXS flux now | fa_2 | 1.976 | 0.679 | 0.1127 | 968 | 0.63 | 0.179 | 0.451 | 10.0 | 2.0-34.0 | 0.0-39.1 |
| GOES >= C1 flare within 15 min | hot-onset trigger (SoLEXS) | fa_2 | 0.032 | 0.02 | 1.0 | 968 | 1.0 | 1.0 | 0.0 | 37.0 | 34.0-41.0 | 18.0-46.0 |
| GOES flux reaches M1 within 30 min | network (E4) | best_TSS |  | 2.25 | 0.069 | 114 | 0.991 | 0.494 | 0.497 | 28.0 | 6.0-40.0 |  |
| GOES flux reaches M1 within 30 min | network (E4) | fa_0.25 | 0.219 | 0.126 | 0.0074 | 114 | 0.877 | 0.04 | 0.837 | 5.0 | 2.0-13.0 |  |
| GOES flux reaches M1 within 30 min | network (E4) | fa_0.5 | 0.459 | 0.326 | 0.0099 | 114 | 0.921 | 0.08 | 0.841 | 6.0 | 2.0-15.0 | 0.0-36.6 |
| GOES flux reaches M1 within 30 min | network (E4) | fa_1 | 0.983 | 0.679 | 0.0157 | 114 | 0.956 | 0.144 | 0.812 | 8.0 | 2.0-21.0 |  |
| GOES flux reaches M1 within 30 min | network or SoLEXS flux now (E4) | best_TSS |  | 2.284 | 0.0764 | 114 | 1.0 | 0.534 | 0.466 | 28.0 | 5.0-40.0 |  |
| GOES flux reaches M1 within 30 min | network or SoLEXS flux now (E4) | fa_0.25 | 0.182 | 0.093 | 0.0086 | 114 | 0.93 | 0.029 | 0.901 | 4.0 | 1.0-12.0 |  |
| GOES flux reaches M1 within 30 min | network or SoLEXS flux now (E4) | fa_0.5 | 0.497 | 0.399 | 0.0125 | 114 | 0.982 | 0.103 | 0.879 | 6.0 | 1.0-16.0 | 0.0-38.9 |
| GOES flux reaches M1 within 30 min | network or SoLEXS flux now (E4) | fa_1 | 0.961 | 0.646 | 0.0183 | 114 | 0.991 | 0.149 | 0.842 | 8.0 | 2.0-22.0 |  |
| GOES flux reaches M1 within 30 min | SoLEXS trend (15-min extrapolation) | fa_0.5 | 0.47 | 0.546 | 0.0031 | 114 | 0.728 | 0.029 | 0.699 | 3.0 | 1.0-7.0 | 0.0-11.8 |
| GOES flux reaches M1 within 30 min | SoLEXS flux now | fa_0.5 | 0.497 | 0.266 | 0.0134 | 114 | 0.991 | 0.08 | 0.911 | 5.0 | 1.0-13.0 | 0.0-37.8 |
| GOES flux reaches M1 within 30 min | hot-onset trigger (SoLEXS) | fa_0.5 | 0.128 | 0.1 | 1.0 | 114 | 1.0 | 1.0 | 0.0 | 40.0 | 37.0-47.5 | 33.3-56.4 |

E4 alerts at every operating point, references at the primary one (2 and 0.5 false alarms per day on validation). Lead: GOES peak minus the first alert minute; chance: the same windows moved 2 h; event TSS = detection rate - chance. Thresholds fixed on validation. A false alarm is an ON episode with no flare, so a signal that is almost always ON has few false alarms but a high time on and chance rate (the hot-onset trigger row): read false alarms together with time on and chance.
Test period 2026-03-30 04:53 to 2026-09-21 00:00 UTC (150.2 days with data).

Source files and script: scripts/paper/make_tables.py (2026-09-25 20:26:51 UTC).
