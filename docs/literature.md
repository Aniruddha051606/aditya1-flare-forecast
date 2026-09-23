# Solar flare prediction with machine learning: where this project sits

Reviewed 2026-09-23. Written to answer three questions: what scores are honestly
achievable, why published numbers disagree so wildly, and what this project can
claim that the literature does not already have.

---

## 1. The honest benchmark: what operational systems actually achieve

The only truly fair comparison in the field is the ISEE/Nagoya benchmark
(Leka et al. 2019, ApJS 243:36), which scored **19 operational forecasting
methods** on the *same* interval, with forecasts issued in advance rather than
fitted after the fact.

Its verdict is blunt: none of the methods score close to 1.0 on any meaningful
metric, though most stay above no-skill. Which method "wins" depends entirely on
the flare class and the metric chosen — there was no single winner.

The follow-up review (arXiv 2511.20465, 2025) puts human forecaster performance
for major events at **TSS 0.3–0.5**, and warns that offline TSS/HSS figures should
not be extrapolated to operational settings.

**Take this as the ceiling for day-ahead forecasting, not the published 0.8–0.9.**

### Their base rates vs ours — this matters enormously

Leka's test interval was 2016–2017, near **solar minimum**. Ours is 2024–2026,
near **solar maximum**. Measured from the GOES flare summary in `D:\Data\goes`:

| Class | Leka 2016–17 (731 d) | This project 2024-02→2026-09 (964 d) |
|---|---|---|
| C1.0+ | 188 event days — **25.7 %** | 930 event days — **96.5 %** |
| M1.0+ | 26 event days — **3.6 %** | 453 event days — **47.0 %** |
| X1.0+ | 3 event days — **0.4 %** | 63 event days — **6.5 %** |

Consequences, and they are not small:

* **C-class day-ahead is a dead question for us.** At a 96.5 % base rate the answer
  is "yes" almost every day. Any accuracy figure is meaningless; even TSS is
  strained. C-class belongs in *nowcasting*, where the high rate is an asset.
* **M-class is the only well-posed day-ahead target** — 47 % is close to balanced,
  so TSS, AUC and reliability all behave sensibly.
* **X-class is where "90 % accuracy" is a lie.** Predicting "no X flare" every
  single day scores **93.5 %**.
* Our numbers are **not comparable** to any paper whose test period sat at solar
  minimum. Say so explicitly when reporting.

---

## 2. Why published numbers are so much higher — five traps

Reported TSS for M-class 24 h forecasts spans roughly 0.5 → 0.95. The spread is
mostly methodology, not method quality.

**1. Threshold tuned on the test set (TSSmax).** Leka et al. deliberately refuse
to report maximum-TSS, because the optimal threshold must come from training
data, not from the interval being scored. A large fraction of ML papers report
exactly this. *We choose thresholds on validation only — keep it that way.*

**2. Random splits instead of chronological.** The same active region rotates
across the disk for days and can land in both train and test. The 2025 review
recommends **active-region-partitioned** splits. *Our 27-day embargo (one solar
rotation) is stricter than most published work, and will cost us score.*

**3. Class balancing that leaks into evaluation.** Bobra & Couvidat (2015) is the
most-cited SHARP+SVM result (TSS 0.76 at 24 h, 0.82 at 48 h), obtained with class
weighting on 303 flaring vs 5,000 non-flaring regions. It has since been
criticised for producing test-set-tailored results. Resampling the *training* set
is defensible; resampling or balancing the *test* set is not.

**4. TSS is gameable by overforecasting.** Leka et al. make this point directly:
at the low event rates typical of flares, a system that cries wolf can reach a
high TSS, while an underforecasting system cannot. **TSS must be reported
alongside frequency bias (FB).** We currently do not compute FB — see §5.

**5. Inconsistent preprocessing and self-built datasets.** The review's blunt
conclusion is that a strictly fair cross-paper comparison is essentially not
possible today.

---

## 3. The method landscape

Four generations, per the 2025 review:

| Generation | Typical approach | M-class 24 h TSS reported |
|---|---|---|
| Statistical | Sunspot/McIntosh class, Poisson rates | ~0.3–0.5 |
| Classical ML | SVM, Random Forest, LightGBM on SHARP features | 0.55–0.76 |
| Deep learning | CNN on magnetograms, LSTM on MVTS, Transformers | 0.70–0.88 |
| Multimodal LLM | Foundation models over heterogeneous inputs | 0.88–0.95 (claimed) |

Treat the bottom two rows with the §2 caveats firmly in mind. Selected results
often quoted: DeFN ≈ 0.80; ASAP_Deep ≈ 0.894 for ≥C; Liu et al. LSTM ≈ 0.607;
Wang et al. LSTM ≈ 0.553; LightGBM ≈ 0.69.

**Benchmark datasets.** SWAN-SF (4,098 multivariate time series, 51 parameters,
12-min cadence, Cycle 24) is the standard imbalanced-MVTS benchmark — partition 1
alone has a 1:364 X-to-quiet ratio. A large sub-literature exists purely on
handling that imbalance (SMOTE, ADASYN, TimeGAN, contrastive learning, MiniRocket).
The Boucheron set (950k HMI magnetograms, AR-partitioned) is the newer alternative.

**None of these use X-ray light curves as the primary input.** They are almost all
magnetogram/SHARP-driven. That is the gap this project sits in.

---

## 4. Where this project is genuinely novel — and where it is not

### Not novel
* Day-ahead forecasting from SHARP magnetic parameters. Heavily worked since 2015.
* C-class detection from soft X-rays. Trivially done by GOES thresholds.

### Genuinely novel
1. **Combined soft + hard X-ray input from a single platform.** The literature is
   dominated by magnetograms. HEL1OS gives 8–150 keV alongside SoLEXS 2–22 keV at
   L1, continuously. No published ML forecaster uses this pairing.
2. **Hard X-ray timing.** HEL1OS resolves quasi-periodic pulsations and impulsive
   structure that GOES simply cannot see. The Neupert relation (HXR leading SXR
   derivative) is a physically motivated precursor no magnetogram model can access.
3. **HOPE on a second instrument.** Hot onsets — the ~10 MK plasma appearing
   *before* the impulsive phase (Hudson et al.; A&A 2023) — were established on
   GOES. Telikicherla et al. (2026, Solar Phys.) built the first ML nowcaster on
   them, reporting all four >M5.0 test flares detected with **17.9 min mean lead
   over the NOAA R3 alert**. Reproducing HOPE independently on SoLEXS is a real
   contribution: instrument-independent confirmation plus a second vantage point.
4. **Operational realism.** Sealed day-ahead forecasts with SHA-256, refusal to
   re-issue, and chronological splits with a rotation-length embargo is stricter
   than most of the field.

### The direct competitor to beat
**arXiv 2608.20062 (Aug 2026)** — "Toward Operational Solar Flare Peak Flux
Nowcasting". Attention seq2seq LSTM (2×1024 encoder), input 60 min of GOES
0.1–0.8 nm at 1-min cadence, predicts the ongoing flare's peak flux from +3 min
after onset. 19,438 flares (1997–2024), 4-fold CV stratified by month.

| Target | RMSE (dex) | % error |
|---|---|---|
| ≥C | 0.26 | 3.11 % |
| ≥M | 0.45 | 5.59 % |
| X | 0.87 | 12.76 % |
| ≥M at +3 min | 0.67 | 9.95 % |

Classification C vs ≥M: **TSS 0.68, F1 0.77**. No baseline comparison given;
authors note that magnetograms or microwave data would likely help.

**This is our yardstick.** They use GOES soft X-rays alone. We have soft *and*
hard X-rays. If SoLEXS+HEL1OS cannot beat 0.26 dex at ≥C and TSS 0.68, the hard
X-ray channel is not earning its place — and that itself is a publishable
negative result.

Note their split is 4-fold CV stratified by month, i.e. **not chronological**.
Our comparison will be apples-to-oranges in our disfavour; state it.

---

## 5. What to change in this project as a result

1. **Add frequency bias (FB) next to every TSS.** Leka et al. show TSS alone is
   uninterpretable at low event rates. Currently missing from `metrics.py`.
2. **Score day-ahead at M+ , not C+.** `goes_min_class = "C1.0"` is right for
   training labels and nowcasting, but a C-class day-ahead forecast at a 96.5 %
   base rate says nothing. Report M1.0+ as the headline.
3. **Report the base rate beside every score**, always. `dayahead.py` already
   records `base_rate_train` / `base_rate_test` — surface them in RESULTS.md.
4. **Never report TSSmax.** Thresholds from validation only. Already the design;
   keep it explicit in the write-up so reviewers can see it.
5. **Add the peak-flux nowcast task** to match arXiv 2608.20062 directly —
   predict peak flux from +3 min after onset, report RMSE in dex. This is the
   single most comparable number we could produce.
6. **Frame X-class honestly.** 63 event days is too few for a defensible
   day-ahead X-class model. Report it as a limit, not a result.
7. **Cite the solar-cycle phase** in every comparison. Our maximum-phase base
   rates make our task different, not easier.

---

## 6. Reading list

**Evaluation and benchmarks (read first)**
- Leka et al. 2019, *A Comparison of Flare Forecasting Methods II*, ApJS 243:36 — [arXiv:1907.02905](https://arxiv.org/abs/1907.02905)
- Leka et al. 2019, *III. Systematic Behaviors* — [IOP](https://iopscience.iop.org/article/10.3847/1538-4357/ab2e11)
- Park et al. 2020, *IV. Consecutive-day Forecasting Patterns* — [IOP](https://iopscience.iop.org/article/10.3847/1538-4357/ab65f0)
- Barnes et al. 2016, *I. Results from the "All-Clear" Workshop* — [IOP](https://iopscience.iop.org/article/10.3847/0004-637X/829/2/89)
- *Advances and Challenges in Solar Flare Prediction: A Review*, 2025 — [arXiv:2511.20465](https://arxiv.org/html/2511.20465v2)

**Nowcasting (our task)**
- *Toward Operational Solar Flare Peak Flux Nowcasting*, 2026 — [arXiv:2608.20062](https://arxiv.org/abs/2608.20062)
- Telikicherla et al. 2026, *Advancing Solar Flare Nowcasting with ML Detection of HOPE*, Solar Phys. — [Springer](https://link.springer.com/article/10.1007/s11207-026-02705-2)
- *The existence of hot X-ray onsets in solar flares*, A&A 2023 — [arXiv:2310.04234](https://arxiv.org/pdf/2310.04234)
- *Observations of a Faint Nonthermal Onset before a GOES C-class Flare*, 2025 — [arXiv:2510.02052](https://arxiv.org/pdf/2510.02052)

**Forecasting methods**
- Bobra & Couvidat 2015, *SDO/HMI Vector Magnetic Field + ML* — [arXiv:1411.1405](https://arxiv.org/pdf/1411.1405)
- Florios et al. 2018, *Magnetogram-based Predictors and ML* — [arXiv:1801.05744](https://arxiv.org/pdf/1801.05744)
- *Predicting Solar Flares with ML: Solar Cycle Dependence* — [arXiv:1912.00502](https://arxiv.org/pdf/1912.00502)
- *A Framework for Designing and Evaluating Flare Forecasting Systems* — [arXiv:2005.02493](https://arxiv.org/pdf/2005.02493)
- *Comparative Analysis of ML Models: High-performing AR Flare Indicators* — [arXiv:2204.05910](https://arxiv.org/pdf/2204.05910)
- *Solar Flare Forecasting Using ML and SDO/HMI Data*, ApJS 2025 — [IOP](https://iopscience.iop.org/article/10.3847/1538-4365/adf8e0)

**Imbalance handling (SWAN-SF sub-literature)**
- *Class-Based Time Series Data Augmentation* — [arXiv:2405.20590](https://arxiv.org/html/2405.20590v1)
- *Contrastive Representation Learning (CONTREX)* — [arXiv:2410.00312](https://arxiv.org/pdf/2410.00312)
- *EXCON: Extreme Instance-based Contrastive Learning* — [arXiv:2411.11249](https://arxiv.org/pdf/2411.11249)

**Aditya-L1 instruments and early science**
- *HEL1OS — A Hard X-ray Spectrometer on Board Aditya-L1* — [arXiv:2512.12679](https://arxiv.org/pdf/2512.12679)
- *Iron Fluorescence in X-class Solar Flares: SoLEXS Observations* — [arXiv:2605.22573](https://arxiv.org/pdf/2605.22573)
- *Non-Stationary Quasi-Periodic Pulsations with HEL1OS and GOES* — [arXiv:2607.05526](https://arxiv.org/pdf/2607.05526)
- *Multi-Wavelength Diagnostics of Pre-Flare Evolution with Aditya-L1*, MNRAS — [arXiv:2607.26171](https://arxiv.org/abs/2607.26171)
  — 7 M/X flares, **102 chromospheric pre-flare transients**, ~28 % with 10–30 keV
  X-ray counterparts, 4 with confirmed hot onsets. Direct precedent for combining
  SUIT + HEL1OS + SoLEXS.
- *The Aditya-L1 mission of ISRO* — [arXiv:2212.13046](https://arxiv.org/pdf/2212.13046)
