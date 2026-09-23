# SUIT flare forecasting

Forecasting GOES flares from **Aditya-L1 SUIT** full-disk ultraviolet images. This is a separate
model from the X-ray one (`solarflare/`), but it shares its GOES truth, its train/test split and its
scoring, so the two are scored on the same days and can later be combined.

## Why SUIT

SoLEXS and HEL1OS see a flare in the corona once it has started. SUIT images the **chromosphere**,
where energy builds up before a flare. Its Mg II and Ca II filters show the brightenings and plage
growth of an active region hours ahead. On Aditya-L1 data, [arXiv:2607.26171](https://arxiv.org/abs/2607.26171)
found **102 chromospheric pre-flare transients** before 7 M/X flares, about 28 % of them with
10–30 keV counterparts. That makes SUIT the natural input for forecasts of hours to a day, the
horizon where X-ray-only forecasting is weakest.

Filters used by default (`suit/config.py`, from the instrument paper
[arXiv:2501.02274](https://arxiv.org/abs/2501.02274)):

| Filter | nm | Sees |
|---|---|---|
| NB03 | 279.6 | Mg II k, chromosphere |
| NB04 | 280.3 | Mg II h, chromosphere |
| NB08 | 396.85 | Ca II H, chromosphere |
| BB03 | 300–360 | photosphere (context) |

## Leakage rules, and how the code enforces them

SUIT is told about flares by SoLEXS and HEL1OS. On a trigger it points a small region of interest
(RoI) at the flare and sets its own exposure. Anything that reflects that behaviour gives the answer
away, so:

1. **Full-disk frames only.** Frames smaller than `min_full_px` (RoI) are never selected, and a
   frame is used only if the whole solar disk is in view (`features.fit_disk`).
2. **Contrast-only features.** Every feature is the image divided by the quiet disk at the same
   radius, so a changed exposure changes nothing. Tested: 3× exposure gives identical features.
3. **Exposure and observing mode are never features.** The mode is recorded for the inventory only.
4. **No look-ahead.** A forecast at hour `o` sees only frames taken at or before `o` (minus
   `latency_h`). Tested by altering every frame after a date: no earlier row changes.
5. **The observing schedule is not a feature either.** Image age and the number of filters present
   change when flare mode interrupts synoptic imaging, so they are left out
   (`schedule_features = false`).

## Data

Put SUIT Level-1 files under `<data_root>/suit` (`D:\Data\suit`) as FITS, `.fits.gz`, or zips of
either. Zips are read in place, like the SoLEXS and HEL1OS readers, so one copy of the archive is
enough.

Of PRADAN's ~800 images a day, the model reads one full-disk frame per filter per hour (~96 a day).
Each is reduced to 256 × 256 (11″ pixels) and about 13 numbers, so the feature tables stay in
megabytes. The raw archive is the storage question: at up to ~8 GB a day, a full month does **not**
fit on D: today (~135 GB free). Start with a **pilot of one to two weeks** of full-disk frames in the
four filters above. A flare-rich stretch such as May 2024 is ideal if PRADAN lets you choose filters.
Scale up when the larger drive arrives.

## Running it

Settings: `config/suit.toml` (TOML, like `project.toml`, but a separate file so SUIT work never
touches the X-ray pipeline's settings). A misspelt key is an error.

```bash
python -m suit inventory --show 3     # first: what is there, and does the reader understand it
python -m suit features               # per-frame features, incremental and cached
python -m suit matrix                 # E1-E6: does SUIT add to SoLEXS and HEL1OS?
python -m suit train --own-split      # SUIT alone on its own hours (a pilot smoke test)
```

`inventory` reads headers only. It reports files and frames, unreadable or corrupt files, frames with
no time or filter, **which header keyword each time and filter came from** (the check on this
reader's assumptions), filters, image sizes, observing modes, per-filter cadence and hourly coverage,
the time span and the longest gaps. It also writes all of it to `outputs/suit/inventory.json`.

## The E1-E6 matrix (`python -m suit matrix`)

| | Inputs |
|---|---|
| E1 | SoLEXS: the day-ahead system's own features, built by its own functions, unchanged |
| E2 | HEL1OS: new, built the same way (`suit/xray.py`): CdTe 5–20 and CZT 20–40 keV rates, hardness, catalogue bursts |
| E3 | SUIT |
| E4 | SoLEXS + HEL1OS |
| E5 | SoLEXS + SUIT |
| E6 | SoLEXS + HEL1OS + SUIT |

Everything but the inputs is the day-ahead system's, called unchanged
(`solarflare/products/dayahead.py`):
- **Forecasts:** every whole UTC hour.
- **Targets:** a GOES ≥C1 or ≥M1 flare peaking within 2, 6, 12 or 24 h, with GOES covering ≥ 80 %
  of the window.
- **Split:** the X-ray model's train end and test start. A target window never straddles a boundary.
- **Models:** logistic regression and gradient-boosted trees. The family and threshold are chosen on
  validation, and the test period is scored once.
- **Reference:** persistence.
- **Intervals:** 95 % intervals that resample whole weeks.

**Paired:** all six experiments train and score on the **same hours**, those where all three
instruments observed. A difference between two of them therefore comes from their inputs alone.
Hours without SUIT are left out of all six, and counted.

Reported per target: TSS [95 %], **FB**, AUC [95 %], BSS, POD and FAR for each experiment. For each
comparison: the baseline's AUC, the new AUC, the paired AUC gain [95 %], the gain as a share of the
baseline's skill above chance, the TSS and BSS changes, and the number of paired forecasts and
distinct flares. The comparisons:

- **E6 − E4:** SUIT added to SoLEXS + HEL1OS. This is **the decision gate**.
- **E5 − E1:** SUIT added to SoLEXS.
- **E4 − E1:** HEL1OS added to SoLEXS.
- **E3 − E1:** SUIT alone against SoLEXS alone.

**The decision gate is fixed in `config/suit.toml` before any SUIT data exists.** SUIT earns the
neural-network branch only if, for ≥M1 within 24 h, E6 beats E4 with a 95 % week-block interval on
the AUC gain above zero and no loss of Brier skill. The other seven targets are reported, not used
to decide, so the answer cannot come from picking the best-looking one. "The interval excludes zero"
is the criterion, as in the day-ahead system; no other significance test is implied.

Output: `outputs/suit/matrix/MATRIX.md` and `matrix_summary.json`, with the code version, the SUIT
feature fingerprint, the full configuration and the split, for reproducibility.

**Needs:** the finished X-ray run (its split and master catalogue), GOES, and SUIT data. It stops
and says which is missing.

## What is verified, and what is not yet

**Verified on synthetic data. This is pipeline validation, not a scientific result:**
- `tests/test_suit.py`, 24 checks:
  - the inventory: corrupt files, bad zip members and frames with no time reported, never dropped;
    which header keyword each time and filter came from; per-filter cadence;
  - the reader: plain, gzipped and zipped files, and a filter named only in the file name;
  - RoI and zoomed frames excluded;
  - exposure independence;
  - no look-ahead;
  - a cache that cannot go stale;
  - the full training path finding a planted pre-flare brightening.
- `tests/test_suit_matrix.py`: the matrix finds a planted SUIT build-up (E6 − E4 above zero, gate
  passes). With SUIT replaced by noise it **does not manufacture a gain** (gate does not pass). Also:
  - all six experiments on identical hours, each with exactly its own inputs;
  - no look-ahead in the SoLEXS and HEL1OS blocks;
  - duplicate and missing minutes, timestamp alignment;
  - imputation from training hours only;
  - the settings file.

**Assumed until real files arrive:**
- The header keyword names. `suit/io.py` tries several for time, filter and mode, then falls back to
  the file name. Run `inventory --show 3` on the first real files and adjust `TIME_KEYS`,
  `FILTER_KEYS` or `MODE_KEYS` if needed.
- That Level-1 full-disk frames are at least 1024 px (4096², or 2048² binned) and RoI frames
  smaller. The inventory prints the sizes it finds.
- That the disk-fit thresholds hold for real limb profiles. The quiet-disk normalisation uses ring
  medians, so it copes with limb darkening and limb brightening alike.

## Next steps

1. Download a pilot and run `inventory --show 3`. Confirm the keywords, sizes, filters and cadence.
2. Run `features`, then `train --own-split`, to check the pipeline end to end on real images.
3. With enough SUIT overlapping the X-ray model's test period, run `matrix`.
4. **Only if the gate passes:** a small CNN on disk-contrast maps and region crops, and a third
   branch in SoLEXHEL-Net (another `ModalityEncoder`, three-way gated fusion), as a new model version.
