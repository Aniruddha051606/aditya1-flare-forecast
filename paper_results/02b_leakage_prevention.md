# How leakage is prevented

Written before any paper experiment was run, from the code and the test suites.
Each protection names where it is implemented and what checks it. Measured values
(split dates, gaps, counts) are not restated here: they are in
`02_experiment_split.json`, produced by `scripts/paper/data_and_split.py`, which
stops if any split check fails.

## 1. Chronological split with a 27-day embargo

- **Implementation.** `solarflare/preprocess/dataset.py`, `_global_split`: every
  window is sorted by its prediction origin; the first 60% of windows train, the
  next 20% validate, the last 20% test (`train.split` in the run's
  `reports/config.json`). Windows within the embargo of a boundary are dropped.
  The embargo is `embargo_days = 27` in `config/project.toml` (one solar rotation),
  applied as `train.global_embargo_s`.
- **No random temporal splitting.** Splits are cut on the calendar; shuffling
  happens only inside the training split, batch by batch.
- **Checked on the real split** (`02_experiment_split.json`, `checks`): the
  rebuilt split equals the one recorded at training; splits are disjoint and in
  time order; both gaps are at least the embargo; and no input span (2 h before the
  origin) or target span (up to the longest horizon after it) reaches across a
  boundary.
- **Tests.** `test_correctness`: "27-day embargo between train and validation",
  "27-day embargo between validation and test", "every validation window is after
  all training + embargo", "every test window is after all validation + embargo",
  "splits are disjoint".

## 2. Causal inputs

- **Network.** Every temporal layer is causal: `solarflare/models/blocks.py`
  (`CausalConv1d`, the TCN stack), `solarflare/models/ssm.py`. Tests
  (`test_correctness`): "CausalConv1d does not see the future", "TCN encoder stack
  is causal", "SSM encoder is causal", "SqueezeExcite gate is causal", and each
  encoder in the zoo ("encoder '...' shape and causality").
- **Features.** Input features use trailing (past-only) backgrounds:
  `solarflare/preprocess/grid.py` / `features.py` (running percentile). Tests:
  "no SoLEXS feature depends on future data", "trailing background ignores the
  future", "changing data after t leaves the background at t alone", "chunked
  trailing percentile == single-chunk".
- **Labels may look ahead; inputs may not.** The centred background kept on each
  segment (`Segment.background`) is used for labels and plots only and is never an
  input (`solarflare/preprocess/dataset.py`).
- **Clock inputs are off** (`model.use_clock = false` in the final run's config):
  on a finite archive time-of-day identifies samples and was once used as a lookup
  table. Test (`test_robustness`): "...including the clock flag that caused the
  leakage bug". The modality ablation keeps a `clock_only` row as a guard rail.

## 3. GOES-18 is truth, never an input

- **Design.** With `labels = "goes"`, GOES-18 XRS enters only
  `build_segments_from_raw` to make targets (flare list, in-flare, phase, flux
  targets). Model inputs are SoLEXS and HEL1OS features and their masks.
- **New regression test** (`test_goes`, "test_goes_never_reaches_the_model_inputs"):
  the same Aditya-L1 data labelled with two different GOES records give identical
  input features, masks, clock, windows, split and normalised inputs, while the
  targets differ.
- **Two uses of GOES that are fitted parameters, not inputs, both on training data
  only:** the flux anchor (a linear SoLEXS-rate to GOES-flux map the flux heads
  start from; `solarflare/pipeline.py`, `fit_flux_anchor`, fitted on samples up to
  the training end) and the SoLEXS-to-GOES calibration used by the catalogue and
  the day-ahead study (`fit_calibration(..., train_end)`). `goes_long` on a segment
  is the SoLEXS band analogue of the GOES long channel, computed from SoLEXS.

## 4. Validation-only choices

- **Thresholds.** `solarflare/evaluate.py`: every yes/no threshold is
  `best_threshold` on the validation predictions, then held fixed on test. The
  frozen model's thresholds likewise (`solarflare/forward.py`, `freeze`). Alert
  rules: `solarflare/products/leadtime.py` chooses rules on validation (test:
  "alert rules carry the validation thresholds").
- **Calibration.** Isotonic calibration (`solarflare/probcal.py`) is fitted on
  validation predictions only (`forward.py`, `calibration_report.py`) and applied
  unchanged to test. Tests: "calibration lowers the Brier score out of sample",
  "calibration never reorders forecasts".
- **Model selection.** Early stopping and the best checkpoint use the validation
  score (`solarflare/train.py`, `quick_val_metrics`); the SHARP input decision uses
  validation scores (`outputs/ablations/sharp`).
- **Day-ahead study.** `solarflare/products/dayahead.py`: models fitted on training
  hours, family and threshold chosen on validation, scored once on test.

## 5. Duplicates

- SoLEXS days carrying a copy of the previous day are found
  (`solarflare/quality.py`) and masked (tests: "copied intervals are masked,
  half-open", "training mask blanks the copied samples only").
- Versions: the highest version of a SoLEXS day and of a HEL1OS observation wins
  (tests: "a tie in coverage goes to the higher version", "a re-processed day
  replaces the cached one, never doubles it"); duplicate HEL1OS minutes are counted
  once.
- The study cache is frozen (`cache/FROZEN`): data downloaded after training cannot
  enter it (test: `test_scale`, "a frozen cache does not take in a newly downloaded
  day").

## 6. Known exceptions and risks (to state in the paper)

- **Operational day models v2** (`outputs/dayahead/frozen_day_models_v2.pkl`) are
  recalibrated on the test period by design: they exist only to forecast days
  after the study. They must never be scored on the test period; no paper table
  uses them.
- **Development history.** The project was developed over several runs whose test
  scores were seen (for example, calibration was added after raw probabilities
  showed negative Brier skill; the SoLEXS energy scale was corrected). Choices made
  after seeing earlier test scores are a mild form of test-set reuse. The sealed
  forecasts (`outputs/tests/live_dayahead/`) and any later forward test are the
  only evaluations nothing could have been tuned on.
- **Base-rate shift.** Flare rates fall from training to validation to test
  (`02_experiment_split.json`); validation-chosen thresholds are therefore
  conservative on test for the longer horizons. This is a property of the solar
  cycle, not leakage, but it affects frequency bias.
