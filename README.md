# Solar flare nowcasting and forecasting from Aditya-L1 (SoLEXS + HEL1OS)

A deep-learning system for ISRO Problem Statement 15. It reads Aditya-L1 soft X-ray (SoLEXS) and hard
X-ray (HEL1OS) Level-1 products and does four things:

- detects flares in each instrument and merges the two lists into one catalogue;
- nowcasts whether a flare is in progress and how bright it is;
- forecasts the next 1–60 minutes (flux, flare occurrence, peak) with a measured lead time;
- raises alerts at a chosen false-alarm rate, shown in a desktop console.

GOES-18 XRS is the reference truth for training and scoring only. It is never a model input.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU build (see requirements.txt)
# set data_root in config/project.toml (or SOLARFLARE_DATA_ROOT), then:
python scripts/download_goes.py          # GOES-18 truth from NOAA (public)
python scripts/download_sharp.py         # SHARP from JSOC (optional)
python -m solarflare pipeline            # the whole study into outputs/, resumable
python -m solarflare day-forecast --day 2026-09-21   # sealed C/M forecast for one day
```

Results land in **`outputs/RESULTS.md`**. The page is generated from the measured JSON reports, so it
cannot drift from what the code produced.

---

## What the problem statement asks, and where it is answered

| Asked for | Answered by | Output |
|---|---|---|
| Flare detection in SoLEXS and in HEL1OS | Two independent detectors: a SoLEXS rise rule and HEL1OS coincidence bursts. Each is scored against the GOES flare list, with a time-shifted chance control. | `outputs/catalog/` |
| A merged flare catalogue | Soft and hard detections are merged into one list, with GOES class estimated from SoLEXS alone. | `outputs/catalog/master_catalog.csv` |
| A forecasting model with quantified lead time | SoLEXHEL-Net (below) forecasts 1–60 min ahead. The alert study reports the lead time before each flare's peak, and before GOES first reaches M1. | `outputs/alerts/LEADTIME.md` |
| High TPR, low FAR | Every alert threshold is fixed on the validation period at a set false-alarm rate (C: 2/day, M: 0.5/day) and scored once on the test period. Hit rates are shown next to a chance baseline (the same windows moved 2 h). | `outputs/alerts/alert_rules.json` |
| An interface with light curves and visual alerts | The desktop console's *Flare Watch* tab replays the frozen model day by day: light curves, calibrated probabilities, alert shading. | `Training Console.exe` |
| Hours-ahead forecasting | Day-ahead models (2–24 h) are built from SoLEXS activity, from SDO/HMI SHARP magnetic parameters, and from both, all compared on the same hours. | `outputs/dayahead/DAYAHEAD.md` |

---

## Layout

```
config/project.toml        every path and model setting; nothing else in the code names a folder
solarflare/                the package (python -m solarflare <command>)
  io/                      readers: SoLEXS, HEL1OS (light curves + event lists), GOES-R XRS, SHARP
  preprocess/              cache, gridding, features, labels, windows, chronological splits
  models/                  SoLEXHEL-Net and the encoder zoo used in the ablations
  train.py evaluate.py     training (smoothed model selection, per-part gradient clipping), test metrics
  forward.py probcal.py    freezing a model with its normaliser, calibration and thresholds
  catalog/                 the two detectors, the merge, the GOES scoring, the overview figure
  products/                alerts (lead time), dayahead, references, calibration report, HEL1OS
                           value, hard X-ray spectra and timing, temperatures, onset study
  runall.py summary.py     the pipeline runner and the one-page results
  settings.py cli.py       project.toml -> settings; the command line
dashboard/                 the desktop console (Tkinter) and its detached job runner
suit/                      forecasting from SUIT full-disk UV images (own README; same truth, split, scores)
scripts/                   data chores: HEL1OS ingest, unzip and verify PRADAN archives, data status,
                           GOES and SHARP downloads
tests/                     13 suites (below)
outputs/                   everything the pipeline writes (reports, catalogues, figures and the frozen
                           model are kept in git; predictions, checkpoints and logs are rebuilt)
cache/                     preprocessed products, rebuilt only for new files (not in git)
```

`outputs/` after a full run:

```
outputs/
  quality/      copied SoLEXS days, masked in training
  model/        the network: reports, figures, checkpoint
    forward/final/   the frozen model: weights, normaliser, calibration, thresholds, SHA-256
  ablations/    sharp/ (with vs without SHARP inputs), hel1os/seed_*/ (paired soft vs soft+hard)
  catalog/      master catalogue and its scores
  alerts/       minute-by-minute predictions, lead times, alert rules, the Flare Watch replay
  dayahead/     2–24 h forecasts
  physics/      temperatures, hard X-ray spectra and timing, hot onsets, Neupert effect
  tests/        README.md over: suites/, baseline_vs_final/, peak_nowcast/, forecast_bias/,
                blind_dayahead/ (see "Tests")
  pipeline/     state.json and one log per stage
  RESULTS.md    one page over all of it
```

---

## Data

| Source | What | Where (project.toml) |
|---|---|---|
| PRADAN (ISSDC) | SoLEXS L1 day files (340-channel spectra, 1 s) and HEL1OS L1 products (light curves, event lists), as downloaded zips | `data_root` |
| NOAA NCEI | GOES-18 XRS L2 flare summary (`xrsf-l2-flsum`) and 1-minute flux (`xrsf-l2-avg1m`), used as truth: `python scripts/download_goes.py` | `goes_dir` |
| JSOC | SDO/HMI SHARP keywords (`hmi.sharp_cea_720s`), hourly, numbers only: `python scripts/download_sharp.py` | `sharp_dir` |

`goes_dir`, `sharp_dir` and `hel1os_extracted` default to folders under `data_root`
(`"{data_root}/goes"` and so on), so moving the data to another drive is one edit, or
`SOLARFLARE_DATA_ROOT=E:/AdityaData` for one session. If the data folder is empty, or has lost more
than 20% of the files the cache was built from, every command stops with a message instead of
quietly training on what is left (`SOLARFLARE_ALLOW_MISSING=1` continues on purpose).

```bash
python scripts/unzip_archive.py --data-root D:/Data --instrument solexs      # never deletes a zip
python scripts/unzip_archive.py --data-root D:/Data --instrument hel1os --members all
python scripts/verify_extract.py --data-root D:/Data
python scripts/data_status.py --data-root D:/Data                             # what is there, what is missing
```

Extraction stops before the disk falls below `min_free_gb`. A product split across two zips is
extracted once. A half-downloaded file is skipped, never read.

### Ingesting without filling the disk

Extracting the whole HEL1OS archive needs about 139 GB. `scripts/ingest_batch.py` does it one
product at a time instead: extract the light curves, cache them, read the entry back, then delete
the extracted files again and move on. The zips are never touched, so sub-second timing and the hard
X-ray spectra can still be redone from them.

```bash
python scripts/ingest_batch.py --dry-run     # what it would do
python scripts/ingest_batch.py               # about 0.5 s per product, ~0.4 MB of cache each
python scripts/ingest_batch.py --keep-extracted   # cache only, delete nothing
```

`[data] cache_is_source = true` then lets a day whose extracted files are gone still train, score and
appear in the catalogue: the cache holds everything the pipeline reads (about 0.3 GB for the whole
mission, against ~280 GB of zips and extracted products). Training runs once over all of it, not per
batch -- training month by month would make the model forget earlier ones and break the chronological
split. SoLEXS zips need no extraction at all; the reader reads inside the zip.

What the cache cannot answer comes from the zips directly: the photon event lists behind the hard
X-ray spectra (`hxr-spectra`), the sub-second timing study (`hxr-timing`) and the CdTe temperatures
(`temperature`). Those stages read `events/evt.fits` straight out of each product's zip for the
minutes around a flare (about a second per product), so nothing is extracted for them. Unpacked,
the event lists alone would need ~460 GB. Delete the zips and these three products are lost.

#### Protecting the cache

Once the extracted files are deleted the cache *is* the data of record, so the guard that used to
watch the raw archive watches the cache instead: if more than a fifth of the products the manifest
lists have lost their `.npz`, the run stops rather than quietly training on what is left.

```bash
python scripts/ingest_batch.py --check                  # every entry present and readable?
python scripts/ingest_batch.py --backup E:/cache-backup # copy it somewhere else afterwards
```

A hole is never fatal as long as the zips are there: re-running `ingest_batch.py` notices that an
entry's `.npz` has gone and rebuilds just that product. The zips are the real thing to protect; the
cache is a few hundred MB and cheap to keep a second copy of, ideally on another drive.

---

## The pipeline

`python -m solarflare pipeline` runs each stage as `python -m solarflare <command>` in its own
process. State is kept in `outputs/pipeline/state.json`, so an interrupted run resumes where it
stopped. `--list` shows the state of every stage. `--redo <stage>` reruns that stage and everything
built on it. A frozen model is never overwritten; the old one is moved aside.

Training also survives an interruption inside a stage. After every epoch it writes
`checkpoints/last.pt` (weights, optimiser, schedule, history and random state, written atomically),
and a training started again with the same settings and data carries on from the last finished
epoch with the result it would have reached uninterrupted. After a power cut, restart with plain
`python -m solarflare pipeline` (the console's **Run pipeline**), not `--redo`, which would retrain
stages that had already finished. `last.pt` is deleted when a training ends, and one left by other
settings or data is ignored.

| Stage | Command | Does |
|---|---|---|
| cache | `cache` | Grids every SoLEXS/HEL1OS product once (only new files are read). |
| quality | `quality` | Finds PRADAN SoLEXS day files that repeat the previous day. |
| train | `train` | Trains the network on X-ray inputs. |
| train-sharp | `train --sharp` | The same network with SHARP inputs (only when `use_sharp = "auto"`). |
| choose-inputs | — | Keeps SHARP only if it raises the validation score by `sharp_min_gain`. The test period plays no part. |
| baselines, calibration, references | `baselines`, `calibration-report`, `references` | Classical baselines, probability calibration, and "no change" flux references. |
| freeze | `forward-test freeze --name final` | Fixes the weights, normaliser, calibration and thresholds, all fitted on validation. |
| catalog, catalog-figure | `catalog`, `catalog-figure` | The master catalogue, scored against GOES. |
| alerts-predict(-nohard), alerts | `alerts --predict [--blank-hard]`, `alerts` | Frozen-model predictions at 1-min stride, lead time vs false alarms, the HEL1OS-hidden ablation, alert rules. |
| dayahead | `dayahead` | 2–24 h forecasts: SoLEXS activity, SHARP, and both; freezes the day models `day-forecast` uses. |
| hel1os-seed-N, hel1os-value | `fusion --seed N`, `hel1os-value` | Paired soft-only vs soft+hard ablation, repeated over seeds. |
| temperature, hxr-spectra, hxr-timing, onset-study | same names | Flare physics from both instruments. |
| report, summary | `report` | `outputs/model/reports/RESULTS.md`, then `outputs/RESULTS.md`. |

Before a long run, `config/rehearsal.toml` runs every stage on the real data with one training epoch
(about 2.5 h), so a wiring mistake shows up early:

```bash
SOLARFLARE_CONFIG=config/rehearsal.toml python -m solarflare pipeline --keep-going
```

Any command also runs on its own. `python -m solarflare <command> --help` lists its options. The
model commands also take `--set section.key=value`, which overrides any setting for one run (for
example `--set train.balance_head_gradients=false`). Commands that act on a trained run (evaluate,
freeze, predict, report, baselines) reload that run's own `reports/config.json`. A later edit to
`project.toml` therefore cannot change the inputs a saved model is fed.

### Forecasting new days

- `python -m solarflare day-forecast --day YYYY-MM-DD [--extra-cache DIR]` writes a sealed
  (SHA-256) probability of a >= C1 and a >= M1 flare for that UTC day, from the latest SoLEXS data
  and the frozen day models. No GOES is read, and a day is never re-issued. Keep new downloads out
  of the main cache (it would move the train/test split): cache them into their own folder with
  `python -m solarflare cache --data-root NEW --cache-dir NEW_CACHE` and pass that as `--extra-cache`.
- `python -m solarflare day-forecast --score --goes-dir NEWER_GOES --out FOLDER` checks every sealed
  forecast's SHA-256 and scores it against GOES once GOES covers its day (`SCORES.md`). Download
  newer GOES into a dated folder so the study's `goes_dir` stays as it was.
- The day models are version 2 (`outputs/dayahead/frozen_day_models_v2.pkl`, refrozen alone with
  `python -m solarflare dayahead --freeze-only`): fitted before the test period and recalibrated
  (isotonic) on it, probabilities kept within 1-99%. Version 1 issued raw probabilities, and its
  >= C1 models swung between extremes; its file is kept for the forecasts sealed with it.
- `scripts/forward_check/` replays the frozen network minute by minute on days uploaded after the
  run and draws model-vs-GOES PNGs (`predict_after_test.py`, `plot_days.py`); existing images are
  never overwritten.

### Training speed

Batches are gathered by tensor indexing from arrays kept on the GPU (`train.loader = "gather"`;
`"torch"` is the old per-window loader), and validation runs in batches of 1024. The windows are
identical to the old loader's (tested). On the 2026-09 archive the old loader took 5.4 h per training,
most of it slicing windows in Python.

---

## The network: SoLEXHEL-Net

```
SoLEXS 24 feat (+12 SHARP) + mask ─→ causal TCN encoder ─┐
                                                          ├─→ gated, mask-aware fusion ─→ TCN trunk ─→ causal attention pool
HEL1OS 48 feat + mask ─────────────→ causal TCN encoder ─┘                                                   │
          ┌──────────────┬───────────────┬─────────────────────┬─────────────────────────┬─────────────────────┘
      flare phase     P(flare now)    GOES flux now      flux q10/q50/q90 at         P(>= C1 flare within
      (4 classes)                     (anchored)         +1/5/15/30/60 min           15/30/60 min), peak size/time
```

The input is a 2-hour window on a 20 s grid (360 steps). The network is small on purpose (~0.46 M
parameters).

| Choice | Why |
|---|---|
| Every convolution is causal (left-padded only) | A symmetric kernel leaks a few future steps per layer; stacked over a dozen layers, the model would read the answer. |
| Modality dropout, mask as an input channel | The two instruments overlap on a minority of days. One network handles soft-only, hard-only and fused input, and the mask tells a dead link from a quiet Sun. |
| Gated fusion conditioned on availability | When HEL1OS is absent, the trunk is routed to the soft stream instead of averaging in zeros. |
| Anchored flux heads | Flux = calibrated SoLEXS flux now + a learned change. The anchor is refitted on the training period of every run. |
| Monotone quantile head | q10 ≤ q50 ≤ q90 by construction. |
| Learned per-task loss weights, focal loss | Five heads on different scales; long quiet stretches would otherwise swamp the loss. |
| `ChannelNorm1d`, not `GroupNorm` | GroupNorm pools statistics over time and breaks causality. |

**Features.**
- **SoLEXS (24):** band rates on the published energy scale (Sarwade et al. 2025); GOES 1–8 Å and
  0.5–4 Å analogues; excess and ratio over a *trailing* background; hardness ratios; the GOES
  temperature ratio; mean energy; causal Δlog-flux; coverage.
- **HEL1OS (48 = 12 × CZT1, CZT2, CdTe1, CdTe2):** band rates, hardness ratios, ratio to background,
  Δlog. Each feature comes from one detector only, because the detectors disagree by 0.9–1.5×.
- **SHARP (12, optional):** disk totals and maxima of unsigned flux, current helicity, free energy,
  R, shear and area. Only regions within 70° of the central meridian are used. Values are as an
  operator would have had them: the latest hour ≤ t − 2 h, with a freshness flag.

**Labels.** GOES-18 flare list ≥ C1.0 and 1-minute XRS-B flux. Targets: phase, in-flare, log flux
now, flux at each horizon, occurrence within 15/30/60 min, and peak size and time.

---

## Guarding against the usual ways this goes wrong

- **Leakage through the split.** Splits are chronological and global, with a 27-day embargo
  (`embargo_days`, one solar rotation) so the same active regions do not flare on both sides of a
  boundary. Every product reads the split dates from the trained run, never from code.
- **Leakage through features.** Backgrounds used as inputs are trailing, never centred. HEL1OS
  readout smoothing is a trailing mean. Tests perturb the future and assert that no earlier feature
  moves.
- **Thresholds tuned on the test period.** All thresholds, calibrations and the SHARP decision use
  validation only. The test period is scored once.
- **Probabilities that are not probabilities.** Occurrence heads are calibrated by isotonic
  regression on validation. The frozen model stores the knots, and its thresholds refer to
  calibrated values.
- **A lucky epoch.** Model selection and early stopping follow a trailing mean of the validation
  score over 3 epochs. One earlier run picked a noisy epoch-5 spike on the raw score and stopped.
- **One head drowning the others.** Gradients are clipped per network part. The peak head's gradient
  had been about 10× the rest.
- **HEL1OS readout batching.** Events arrive in batches of about 500–650, one every 2–8 s. Rates
  are averaged over 60 s (trailing) before use.
- **Copied data.** Four PRADAN SoLEXS day files (2024-07-26, 2024-09-01, 2024-10-02, 2025-02-20)
  repeat 6–12 h of the previous day under their own date. `quality` finds them, with GOES as
  referee, and training masks them.
- **Unfair baselines.** Every method is compared at the same false-alarm rate, including the hot
  onset precursor trigger (HOPE, Telikicherla et al. 2026) rebuilt on SoLEXS. Hit rates are shown
  with their chance level, and gains with day- or week-block bootstrap intervals.
- **Numbers without uncertainty.** Headline TSS, AUC, POD and flux errors carry 95% intervals that
  resample whole test days.
- **Flux scored where no Aditya system would forecast.** Flux errors are scored where SoLEXS observed
  the forecast origin, against two "no change" references on the same windows: GOES at the origin,
  and calibrated SoLEXS at the origin (what an Aditya-only system has).
- **Forecast ranges that are too narrow.** The 10–90% flux ranges are widened by a factor fitted on
  validation (split-conformal) and frozen with the model.
- **One seed.** The HEL1OS gain is quoted across seeds, with a verdict that says "mixed" when the
  seeds disagree.

---

## The console

`Training Console.exe` in the project folder (or `pythonw dashboard/mission_control.pyw`) opens **two
windows**, both refreshed every second, so they can sit on one screen each:

**Operations** — five screens: Pipeline, Training, Results, Forecasts and Data.
- *Pipeline*: every stage with its state (done, running, failed, needs redoing) and duration.
- *Training*: eight tiles (status, epoch, batch, elapsed, time left, best score, epochs since the
  best one, throughput), four charts — the loss and each head's share of it, validation skill,
  learning rate against epoch time, and the gradient reaching each part of the network — plus the
  network diagram and this epoch's live numbers.
- *Machine*: GPU use, GPU memory, GPU temperature, CPU, memory and training throughput, each traced
  over the last three minutes, with free space on both drives.
- Terminal (job, stage and training logs) and the watchdog: failed stages, a stalled or diverging
  run, an idle or hot GPU, low disk.
- Actions: run the full pipeline, redo a stage, check or extract data, update the cache, run the
  tests. Jobs run detached (closing a window does not stop them) and keep the laptop awake.
- The console follows `SOLARFLARE_CONFIG` like every command, so it can watch a rehearsal run.

- *Results*: the headline numbers read straight from the reports — both alerts (flares warned, chance
  level, false alarms per day, lead), flare-now and flare-soon skill, flux and peak errors — then the
  ≥ C1 alert against simple baselines, a reliability diagram, and the verdicts on SHARP, HEL1OS,
  calibration, the catalogue and the day-ahead forecasts. A result whose stage has not run says so.

- *Forecasts*: the sealed day-ahead forecasts (UTC day, P(>= C1), P(>= M1), lead, seal), a date box
  that issues the next one through `day-forecast`, and a viewer for the model-vs-actual pictures
  `scripts/forward_check` draws. A sealed forecast is never re-issued.
- *Data*: where the data lives (`data_root`), free space per drive, SoLEXS / HEL1OS / GOES / SHARP
  coverage month by month (the HEL1OS gap is visible at a glance), the cache size and how many of
  its sources are still on disk, and buttons to download GOES or SHARP. Scanning counts file names
  on a worker thread, so the console keeps its 1 s tick. The watchdog raises an alert when the data
  folder is empty or has lost files the cache was built from.

**Flare Watch** — the frozen model replayed one UTC day at a time, five panels on one time axis:
- SoLEXS flux in GOES units, with GOES for comparison and the GOES flare list;
- the master catalogue: a row each for what SoLEXS and HEL1OS detected on their own;
- the calibrated flare probability, shaded where the C alert is on, with the HOPE hot-onset
  trigger marked underneath at the same false-alarm rate;
- the M signal, shaded where the M alert is on;
- the HEL1OS light curves (CZT 20–40 keV, CdTe 5–20 keV).

Beside them: the alert state at the last minute of the replay, a list that switches between the
day's GOES flares (with the lead each alert gave) and the day's catalogue detections (which
instrument, SoLEXS class, highest HEL1OS band, GOES match), and the lead times over the test period.
Each value is drawn at the minute it became known, and days are marked as validation or test.

`--selftest out.json` builds both windows hidden, refreshes once and writes what they show, which is
how to check a packaged build without opening anything. Rebuild the exe after changing `dashboard/`:

```bash
python -m PyInstaller --onefile --windowed --name "Training Console" --icon dashboard/console.ico --paths . --paths dashboard --collect-submodules console --exclude-module torch --exclude-module scipy --exclude-module pandas --exclude-module sklearn --exclude-module astropy dashboard/mission_control.pyw
```

---

## Tests

```bash
python -m tests.test_correctness    # causality, leakage, masking, splits, metrics
python -m tests.test_robustness     # bad files, failure paths, the full CLI on synthetic data
python -m tests.test_scale          # cache, stitching, archive splits, zip reading
python -m tests.test_extract        # corrupt zips, path traversal, disk floor, HEL1OS layout
python -m tests.test_hel1os         # band alignment, zero counts, detector separation, versions
python -m tests.test_forward        # frozen normaliser, cutoff, append-only forecasts, GOES truth
python -m tests.test_goes           # flare summary reader, labels, targets
python -m tests.test_physics        # background, causal onset, Neupert lag
python -m tests.test_catalog        # soft rule, HEL1OS bursts, calibration, merge
python -m tests.test_products       # lead time, HXR spectra, timing, temperature, day-ahead
python -m tests.test_pipeline       # settings, calibration, SHARP, smoothing, clipping, runner, summary
python -m tests.test_suit           # SUIT reader, full-disk test, exposure independence, no look-ahead
python -m tests.test_suit_matrix    # E1-E6 pairing, no invented gains, X-ray blocks without look-ahead
```

CI (`.github/workflows/ci.yml`) runs lint and all thirteen suites on Python 3.12 and 3.13, with CPU
torch.

The finished study is tested further on the real archive, each result in `outputs/tests/` with
`outputs/tests/README.md` as the index:

```bash
python scripts/run_tests.py             # every suite + lint, logs and summary in outputs/tests/suites/
python -m solarflare model-tests        # baseline vs final, peak flux 3 min after onset, frequency bias
python -m solarflare blind-dayahead     # one day-ahead forecast per test day, models frozen before it
```

- **baseline_vs_final**: the SoLEXS-only baseline and the final model on the same windows (the
  period both held out), each fed as it was trained and using its own validation thresholds, with
  paired day-resampled intervals.
- **peak_nowcast**: each GOES flare's peak predicted 3 min after its start, the setting of arXiv
  2608.20062, against GOES/SoLEXS "no change" and "no change + typical rise".
- **forecast_bias**: frequency bias at the deployed, HSS-best and FB = 1 thresholds, raw and
  calibrated: whether over-forecasting is the model's or the threshold's.
- **blind_dayahead**: the day forecast replayed over the test period with models, calibration and
  thresholds fitted before it (the frozen day models were refitted on the test period, so replaying
  those would not be blind), scored against climatology and persistence.

---

## Limits worth knowing

- SoLEXS and HEL1OS overlap on a minority of days. Every hard X-ray result rests on that overlap,
  and the reports quote their sample sizes.
- HEL1OS readout batching means the L1 event lists resolve seconds, not milliseconds, below about
  2 000 events/s.
- Definitive SHARP data are published weeks after observation. The latest month may be missing, and
  the network then holds the last values with a "stale" flag.
- GOES is the truth, so a flare GOES misses counts as a false alarm. The catalogue checks HEL1OS-only
  events against SoLEXS 6–12 keV to show that some are real.
