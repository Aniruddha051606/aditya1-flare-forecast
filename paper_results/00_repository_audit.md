# Repository audit (Solar Physics, Paper 1: SoLEXS + HEL1OS)

Audit of the existing code, tests and results, made before any paper experiment.
It describes what exists and where; measured values live in the result files it
points to and are not restated here.

**Scope of Paper 1.** Aditya-L1 SoLEXS (soft X-ray) and HEL1OS (hard X-ray) as
model inputs; GOES-18 XRS as reference and truth only. SUIT is excluded: the
`suit/` package stays in the repository for Paper 2, but nothing in
`paper_results/` runs, analyses or reports it. SHARP is not an input of any
Paper 1 experiment.

**Environment and repository state:** `00_environment.json` (git commit and
remotes, Python, OS, package versions, GPU), written by
`scripts/paper/environment.py`. Model and training configuration of every
experiment: its `reports/config.json` (E4: `outputs/model/reports/config.json`).

## 1. Architecture

`python -m solarflare pipeline` (`solarflare/runall.py`) runs the whole study as
resumable stages into `outputs/`, driven by `config/project.toml`
(`solarflare/settings.py`). Stage state: `outputs/pipeline/state.json`.

| Stage | What it produces | Code |
|---|---|---|
| cache, quality | per-product 20 s gridded rates; SoLEXS days that copy the previous day | `preprocess/cache.py`, `quality.py` |
| train | SoLEXHEL-Net on SoLEXS + HEL1OS (X-ray only) | `train.py`, `models/net.py` |
| train-sharp, choose-inputs | same network with SHARP inputs; validation decides in or out | `runall.py` |
| baselines | climatology, persistence, logistic regression, gradient-boosted trees on the same split and features | `baselines.py` |
| calibration | raw vs isotonic-calibrated probabilities | `products/calibration_report.py`, `probcal.py` |
| references | flux forecasts vs "no change" references an Aditya-only system has | `products/references.py` |
| freeze | frozen model: weights, normaliser, validation thresholds and calibration, SHA-256 | `forward.py` |
| catalog | master flare catalogue: SoLEXS and HEL1OS detections vs GOES | `catalog/` |
| alerts-predict(-nohard), alerts | minute-by-minute predictions over validation + test (and with HEL1OS hidden); alert rules chosen on validation; lead times per flare | `products/leadtime.py` |
| dayahead | 2-24 h forecasts from SoLEXS activity and SHARP | `products/dayahead.py` |
| hel1os-seed-*, hel1os-value | paired soft-only vs soft+hard ablation (rise-phase model, 3 seeds) | `forecast.py`, `products/hel1os_value.py` |
| temperature, hxr-spectra, hxr-timing, onset-study | flare physics | `products/` |
| report, summary | `outputs/model/reports/RESULTS.md`, `outputs/RESULTS.md` | `report.py`, `summary.py` |

Tests beyond the pipeline (`outputs/tests/`, `python -m solarflare model-tests`,
`blind-dayahead`, `scripts/run_tests.py`): baseline vs final on common windows,
peak flux 3 min after onset, frequency bias, blind day-ahead replay, sealed live
day-ahead forecasts.

## 2. Components

- **Readers** (`solarflare/io/`): SoLEXS L1 (`solexs.py`, read inside the zips),
  HEL1OS L1 light curves (`hel1os.py`) and photon event lists (`hel1os_events.py`,
  read from the zips), GOES-18 XRS L2 flsum/avg1m (`goes.py`), SHARP (`sharp.py`).
- **Preprocessing** (`solarflare/preprocess/`): common UTC grid (`grid.py`),
  multi-day timelines (`timeline.py`), features with trailing backgrounds
  (`features.py`), labels from the GOES flare list (`labels.py`), windows,
  chronological split and embargo, training-window thinning (`dataset.py`),
  per-product cache (`cache.py`; frozen study cache, `cache/FROZEN`).
- **Model** (`solarflare/models/`): SoLEXHEL-Net (`net.py`): a causal encoder per
  instrument, a mask-aware fusion gate, modality dropout during training, and
  heads for in-flare, flare phase, flare within 15/30/60 min, flux now, flux at
  +1/5/15/30/60 min as 10/50/90% quantiles, and time/size of the flare peak.
  Causal blocks (`blocks.py`), alternative encoders (`zoo.py`, `ssm.py`),
  multi-task losses (`losses.py`), rise-phase model (`riseflare.py`).
- **Evaluation** (`solarflare/evaluate.py`, `metrics.py`): TSS, HSS, POD, FAR,
  POFD, CSI, FB, F1, accuracy, AUC, Brier, BSS vs climatology, reliability; MAE,
  RMSE, R2, correlation, skill vs persistence, climatology and SoLEXS no-change;
  interval coverage raw and scaled on validation; 95% intervals resampling whole
  UTC days (`day_block_ci`); modality ablation (both, soft only, hard only, clock
  only) over all test windows and where HEL1OS observed.

## 3. Experiments that already exist (result files)

| Question | Result file |
|---|---|
| Network on the test split (all heads, CIs) | `outputs/model/reports/evaluation.json` |
| Same with one instrument blanked at inference | `evaluation.json` -> `modality_ablation*` |
| Classical baselines | `outputs/model/reports/baselines.json` |
| Raw vs calibrated probabilities | `outputs/model/reports/calibration.json` |
| Flux vs no-change references | `outputs/model/reports/fair_references.json` |
| Lead times, alert operating points | `outputs/alerts/lead_times.csv`, `leadtime_summary.json`, `alert_rules.json`, predictions `pred_final.npz`, `pred_final_nohard.npz` |
| HEL1OS value, separately trained, 3 seeds | `outputs/ablations/hel1os/` |
| SHARP in or out | `outputs/ablations/sharp/decision.json` |
| SoLEXS-only network vs final, same windows | `outputs/tests/baseline_vs_final/` |
| Peak flux 3 min after onset | `outputs/tests/peak_nowcast/` |
| Frequency bias | `outputs/tests/forecast_bias/` |
| Day-ahead | `outputs/dayahead/`, `outputs/tests/blind_dayahead/`, `outputs/tests/live_dayahead/` |
| Catalogue, physics | `outputs/catalog/`, `outputs/physics/` |

## 4. What the paper's E1/E2/E4 matrix needs and does not yet have

- **E4 (SoLEXS + HEL1OS)** is the final model (`outputs/model`, seed 1337).
- **E1 (SoLEXS only) and E2 (HEL1OS only) have not been trained on the frozen
  split.** What exists instead: (a) the final model with one instrument blanked at
  inference, which measures robustness to a missing instrument, not what a
  single-instrument model learns; (b) the separately trained soft-only vs
  soft+hard rise-phase ablation (peak size only, no HEL1OS-only arm); (c) a
  SoLEXS-only network trained before the HEL1OS ingest, on a different split.
- **Code gap.** `train` has no option to train on one instrument. Needed: a
  setting that blanks one instrument's inputs and masks *after* the windows and
  split are fixed (blanking before would change which windows exist and move the
  split), so E1, E2 and E4 share windows, split, labels, thresholds protocol and
  seed.
- **Cost.** The final model's `train` stage took 469.7 min
  (`outputs/pipeline/state.json`); E1 + E2 are about two such runs, run one at a
  time on the 6 GB GPU (two at once ran out of GPU memory on 2026-09-24).
- **Seeds.** One seed per experiment is affordable; training-seed variation for
  the main network is then unmeasured (it is measured for the rise-phase ablation,
  3 seeds).

## 5. Scientific risks found in the audit

1. **Test-set reuse during development.** Several design decisions followed
   earlier runs' test scores (see `02b_leakage_prevention.md`, section 6).
2. **Baselines are close on some tasks.** The in-flare comparison with gradient-
   boosted trees and the C vs >= M call 3 min after onset
   (`outputs/tests/peak_nowcast`) show small or non-significant margins; the paper
   must report baselines beside the network, not only persistence.
3. **Base-rate shift** across the chronological split affects frequency bias and
   thresholds (`outputs/tests/forecast_bias`).
4. **Single test block** of about six months near solar maximum; no cross-cycle
   generalisation can be claimed.
5. **Truth handling.** GOES flare starts define onsets; an Aditya-only system
   would detect onsets itself.
6. **Operational day models v2** are recalibrated on the test period; never to be
   scored on it (`02b_leakage_prevention.md`).
7. **The day-block bootstrap captures test-sample variation only**, not training
   variation.

## 6. Status of the Paper 1 experiments and exact commands

| Step | Command | Status |
|---|---|---|
| Whole study (E4 and all products) | `python -m solarflare pipeline` | done (25 stages, `outputs/pipeline/state.json`) |
| Environment | `python scripts/paper/environment.py` | done |
| Data inventory, frozen split, split checks | `python scripts/paper/data_and_split.py` | done |
| Test suites and lint | `python scripts/paper/test_status.py` | done (`03_test_status.md`) |
| E1 SoLEXS only | `python -m solarflare train --from-run outputs/model --out-dir outputs/paper/E1_solexs_only --set model.inputs=soft` | training (console job, resumable) |
| E2 HEL1OS only | `python -m solarflare train --from-run outputs/model --out-dir outputs/paper/E2_hel1os_only --set model.inputs=hard` | queued after E1 |
| E1/E2/E4 metrics, paired comparisons, robustness | `python scripts/paper/evaluate_experiments.py` | waits for E1 and E2 (refuses to run before) |
| Lead times and operating points (E4) | `python scripts/paper/leadtime_tables.py` | done, from `outputs/alerts/` |

E1 and E2 were launched from the working tree of commit 870eeb1 plus the
uncommitted `model.inputs` change (single-instrument training); the manifest
records the commit that contains it once it is committed.

## 7. Known issues found and fixed during this audit

- A `model-tests` run on 2026-09-25 cached a SoLEXS day downloaded after the
  study into the study cache (one entry). It was removed, nothing else had
  changed since training, and the study cache is now frozen (`cache/FROZEN`,
  regression test in `test_scale`).
- HEL1OS storm-day event lists (up to 9.3 GB) were read whole into memory by
  `hxr-spectra`; the reader now memory-maps them and converts only the rows
  needed (identical output, verified).
- GOES never reaching the model inputs was a design property without a test; a
  regression test now checks it (`test_goes`).
