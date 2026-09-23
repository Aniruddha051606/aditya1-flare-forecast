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

```bash
python -m suit inventory --show 3     # what is there; prints 3 real headers
python -m suit features               # per-frame features, incremental
python -m suit train                  # hourly table, models, scores
```

Results go to `outputs/suit/suit_summary.json`: for ≥C1 and ≥M1 within 6, 12 and 24 h, the TSS
with its 95 % interval, the **frequency bias**, AUC, Brier skill against climatology, a paired
comparison with persistence (a flare of that class in the last 24 h), and the most important
features. `--own-split` ignores the X-ray model's split, which a pilot shorter than that split needs
anyway. It then says so, because those numbers are not comparable with the X-ray results.

## What is verified, and what is not yet

**Verified** on a synthetic SUIT month (`tests/test_suit.py`, 18 checks): the reader (plain,
gzipped, zipped; a filter named only in the file name), RoI and zoomed frames excluded, exposure
independence, no look-ahead, and the full training path finding a planted pre-flare brightening
(AUC > 0.8).

**Assumed until real files arrive:**
- The header keyword names. `suit/io.py` tries several for time, filter and mode, then falls back to
  the file name. Run `inventory --show 3` on the first real files and adjust `TIME_KEYS`,
  `FILTER_KEYS` or `MODE_KEYS` if needed.
- That Level-1 full-disk frames are at least 1024 px (4096², or 2048² binned) and RoI frames
  smaller. The inventory prints the sizes it finds.
- That the disk-fit thresholds hold for real limb profiles. The quiet-disk normalisation uses ring
  medians, so it copes with limb darkening and limb brightening alike.

## Next steps

1. Download a pilot and run `inventory --show 3`. Confirm the keywords, sizes and filters.
2. Run `features`, then `train --own-split`, to check the pipeline end to end on real images.
3. With the full archive, run `train` on the X-ray model's split, so the numbers are comparable.
4. Join the SUIT and X-ray hourly features (both on whole UTC hours) in the day-ahead forecaster,
   and test whether SUIT adds skill over X-rays, as was done for SHARP.
