# Pre-registered physics analysis: when does HEL1OS help?

Written 2026-09-26, **before any per-flare E4-vs-E1 difference was computed or
looked at**. At the time of writing, the only E4-vs-E1 results that existed were
the window-level paired comparisons in `paper_results/paired_comparisons.csv`
(commit fdcd03e). The file is committed on its own, before the analysis script
runs, so git history dates it. `scripts/paper/physics_interpretation.py`
records the SHA-256 of this file in every output. If this file changes after that
commit, the change and the reason are listed under "Amendments" at the end, and
the original wording stays.

## Hypothesis

> Flares exhibiting earlier/stronger hard-X-ray signatures should show a larger
> incremental forecasting benefit from E4 (SoLEXS + HEL1OS) over E1 (SoLEXS only).

A null or opposite result is reported as found. The hypothesis is not revised
after the data are seen.

## Models, predictions, thresholds (all frozen)

- E1: `outputs/paper/E1_solexs_only`. E4: `outputs/model`.
  Predictions: `outputs/paper/predictions/{E1,E4}_{val,test}.npz`, one per
  2-min prediction origin (`y_t_unix` = origin time).
- Head: **flare within 60 min** (primary); flare within 15 and 30 min (secondary).
- Threshold: each model's own validation best-TSS threshold for that head
  (`paper_metrics.fit_on_validation`, the same rule as every paper table).
  Probabilities for the confidence outcome: each model's validation isotonic
  calibration.
- The test set is used only to compute the quantities below. No threshold,
  window or feature is tuned on it.

## Flare sample

- GOES-18 flares ≥ C1.0 (the label list, `labels = "goes"`,
  `goes_min_class = "C1.0"`) with peaks in the test period
  2026-03-30 → 2026-09-21.
- For flare *i*, with GOES start *s*, peak *p* and preceding-flare end *e*:
  *e* is the latest GOES end among ≥ C1.0 flares that started before *s*.
  The **opportunity window** is [*a*, *p*], with *a* = max(*s* − 60 min, *e*).
- Origins count only if they are in the **common** population: SoLEXS and
  HEL1OS both observed (`pop_soft & pop_hard`), the same population as the
  paired comparisons.
- Inclusion criteria:
  1. *a* ≤ *s* − 10 min, i.e. at least 10 min of flare-free lead-in. Excluded
     flares, which start in the decay of an earlier one, are counted.
  2. At least 80% of the expected 2-min origins in [*a*, *p*] are present in the
     common population.
  3. Neither model's alert is already on at the first origin of the window
     (left-censored). The number of such flares is reported, and a sensitivity
     analysis includes them with their censored values.

## Outcomes

- **Primary, warning time.** L_M,i = *p* − *t*\*, where *t*\* is the first origin
  in [*a*, *p*] at which model M's probability ≥ its threshold. It is 0 when there
  is no such origin (no warning before the peak). Resolution is 2 min.
  **ΔL_i = L_E4,i − L_E1,i** (minutes; positive means E4 warned earlier).
- **Secondary, pre-flare confidence.** ΔC_i = the mean over common origins in
  [max(*a*, *s* − 30 min), *s*) of (calibrated p_E4 − calibrated p_E1) for the
  60-min head. It needs ≥ 5 origins, otherwise it is missing.

## Hard-X-ray features (descriptors; not model inputs)

Matched to the GOES flare by GOES peak time from
`outputs/catalog/master_catalog.csv` (`goes_peak_utc`), and by GOES start from
`outputs/physics/flare_catalog_dec120s.csv`. HEL1OS spectra come from
`outputs/physics/hxr_spectra.csv` via the catalogue `id`.

**Primary features (the two tests of the hypothesis):**

- **P1, earlier:** HEL1OS alert time − SoLEXS alert time for the same event
  (`hard_alert_utc − soft_alert_utc`, in minutes). Negative means HXR first.
  **Prediction: Spearman ρ(ΔL, P1) < 0.**
- **P2, stronger:** log10 of the HEL1OS CZT 20–40 keV peak rate
  (`hard_peak_czt_20_40_cps`). **Prediction: ρ(ΔL, P2) > 0.**

**Secondary features (exploratory, reported without claims of significance):**

- spectral index γ at the HXR peak (`gamma`, lower = harder) and the maximum
  energy detected (`max_energy_keV`);
- Neupert behaviour: `neupert_r_best`, `neupert_lag_s`, and
  `hxr_peak_minus_impulsive_s` (HXR peak − steepest SXR rise);
- early HXR significance in the first 2 min after the SXR onset
  (`dec_hxr_sigma`);
- HXR rise time (`hard_peak_czt_20_40_utc − hard_alert_utc`) and
  SXR onset → impulsive phase (`onset_to_impulsive_min`).

## Statistics

- Unit: flare. Uncertainty: **day-block bootstrap** (whole UTC days resampled,
  2,000 draws, seed 0), because flares on one day share conditions.
- Primary tests: Spearman ρ of ΔL with P1 and with P2. Each has a day-block
  bootstrap 95% CI and a two-sided permutation p-value (10,000 shuffles of ΔL,
  seed 0), **Holm-corrected over the two**, α = 0.05. A primary test with fewer
  than 30 flares is reported as inconclusive, not as a null.
- Mandatory confounder check: the partial Spearman ρ controlling for log10 GOES
  peak flux, since bigger flares are brighter in HXR. If the raw association
  holds but the partial one vanishes, the result is described as flare size, not
  HXR behaviour.
- Descriptive group comparison: E4 earlier (ΔL ≥ 2 min) vs E4 not earlier
  (ΔL ≤ 0). For each feature: medians per group, the difference with its
  day-block CI, and group sizes.
- Detection contrast, secondary: mean ΔL for flares that HEL1OS detected (a
  HEL1OS alert in the catalogue) vs flares HEL1OS observed but did not detect.
  Prediction: larger for detected flares.
- **Placebo, secondary:** repeat the primary correlations with ΔL computed from
  E4 *with HEL1OS withheld* (`E4__HEL1OS_withheld_test.npz`, E4 thresholds)
  minus E1. The HXR input is removed there, so no association is expected. An
  association there would mean the features track something other than what
  HEL1OS adds.
- Sensitivity: 15- and 30-min heads; censored flares included; isolated flares
  only (no ≥ C1.0 flare ending within 60 min before the start).
- Everything by class (C vs ≥ M) is descriptive only.

## Outputs (`paper_results/physics/`)

`flare_level_comparison.csv`, `hxr_timing_features.csv`,
`hxr_spectral_features.csv`, `neupert_features.csv`,
`physics_gain_association.csv`, and a short `README.md` stating the result for
each prediction in one line. Every number is read from these files.

## Amendments

None.
