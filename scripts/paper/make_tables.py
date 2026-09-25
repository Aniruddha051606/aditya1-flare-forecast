"""Paper step 19: the publication tables, as CSV (machine-readable) and Markdown.

    python scripts/paper/make_tables.py

Every cell is copied or formatted from a result file; nothing is typed. A table
whose source files do not exist yet is skipped, and paper_results/tables/README.md
says which and why.

  table1_data         01_data_inventory.json, 02_experiment_split.json
  table2_experiments  experiments.json, each run's reports/config.json and data_meta.json
  table3_classification        metrics_classification.csv (common windows; native in table3b)
  table4_flux         metrics_forecast.csv (common windows)
  table5_leadtime     leadtime_summary.csv
  table6_calibration_uncertainty  metrics_classification.csv, metrics_uncertainty.csv (native windows)
  table7_robustness_ablation      metrics_classification.csv, metrics_forecast.csv, paired_comparisons.csv,
                                  outputs/ablations/hel1os/hel1os_value.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import PAPER, ROOT, stamp  # noqa: E402  (sets sys.path)
from solarflare.settings import load_settings

S = load_settings()
TABLES = PAPER / "tables"
HEADS = ["in_flare", "flare_within_15min", "flare_within_30min", "flare_within_60min"]
HEAD_LABEL = {"in_flare": "flare in progress", "flare_within_15min": "flare within 15 min",
              "flare_within_30min": "flare within 30 min", "flare_within_60min": "flare within 60 min"}
EXPS = ["E1", "E2", "E4"]


class Missing(Exception):
    pass


def _csv(name: str) -> list[dict]:
    p = PAPER / name
    if not p.exists():
        raise Missing(f"{name} not found (scripts/paper/evaluate_experiments.py has not run)")
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _json(path) -> dict:
    if not path.exists():
        raise Missing(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} not found")
    return json.loads(path.read_text("utf-8"))


def _one(rows, **kw):
    got = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    return got[0] if got else None


def f(v, nd: int = 3) -> str:
    if v in (None, "", "None"):
        return "n/a"
    return f"{float(v):.{nd}f}"


def ci(r, nd: int = 3) -> str:
    if not r:
        return "n/a"
    s = f(r["value"], nd)
    return s + (f" [{f(r['ci95_low'], nd)}, {f(r['ci95_high'], nd)}]" if r.get("ci95_low") not in (None, "", "None")
                else "")


def write(name: str, title: str, rows: list[dict], notes: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    md += ["", *notes, "", f"Source files and script: scripts/paper/make_tables.py ({stamp()['generated_utc']} UTC)."]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  -> paper_results/tables/{name}.csv / .md")


# ---- the tables ------------------------------------------------------------------------

def table1():
    inv, sp = _json(PAPER / "01_data_inventory.json"), _json(PAPER / "02_experiment_split.json")
    s, h, g = inv["SoLEXS"], inv["HEL1OS"], inv["GOES"]
    rows = [
        {"instrument": "SoLEXS (Aditya-L1)", "role": "model input", "band": "soft X-ray",
         "study data": f"{s['days_in_frozen_study']} days", "observed days": s["observed_days_in_study"],
         "native cadence": "1 s light curve, 340-channel spectra", "model grid": f"{sp['window']['grid_s']:g} s",
         "model features": s["model_features"]},
        {"instrument": "HEL1OS (Aditya-L1)", "role": "model input", "band": "hard X-ray",
         "study data": f"{h['products_used_by_the_study']} products", "observed days": h["observed_days_in_study"],
         "native cadence": "1 s light curves (readout batches 2-8 s)", "model grid": f"{sp['window']['grid_s']:g} s",
         "model features": h["model_features"]},
        {"instrument": "GOES-18 XRS", "role": "truth only (labels, flux target, scoring)", "band": "0.1-0.8 nm",
         "study data": f"{sum(g['flares_labelled_in_observed_segments_by_class'].values())} labelled flares "
                       f">= {g['label_threshold']}", "observed days": "",
         "native cadence": "1 min", "model grid": f"{sp['window']['grid_s']:g} s (targets)", "model features": 0},
    ]
    notes = [f"Labelled flares by class: {g['flares_labelled_in_observed_segments_by_class']}.",
             f"Split: training {sp['splits']['train']['first_origin_utc'][:10]} to "
             f"{sp['splits']['train']['last_origin_utc'][:10]}, validation {sp['splits']['val']['first_origin_utc'][:10]}"
             f" to {sp['splits']['val']['last_origin_utc'][:10]}, test {sp['splits']['test']['first_origin_utc'][:10]} "
             f"to {sp['splits']['test']['last_origin_utc'][:10]}; {sp['checks']['embargo_days']:g}-day embargoes."]
    write("table1_data", "Table 1. Data", rows, notes)


def table2():
    ex = _json(PAPER / "experiments.json")
    rows = []
    for k in EXPS:
        e = ex["experiments"][k]
        run = Path(e["run"])
        meta = _json(run / "reports" / "data_meta.json")
        cfg = _json(run / "reports" / "config.json")
        rows.append({"experiment": k, "inputs": e["label"], "model.inputs": e["inputs"], "seed": e["seed"],
                     "parameters": e["parameters"], "best epoch": e["best_epoch"],
                     "modality dropout": cfg["model"]["modality_dropout"] if e["inputs"] == "both" else "off",
                     "training windows": meta["n_train"], "validation windows": meta["n_val"],
                     "test windows (native)": meta["n_test"]})
    pop = ex["populations_test"]["common"]
    write("table2_experiments", "Table 2. Experiments", rows,
          [f"Common test windows (both instruments observing): {pop['windows']:,} on {pop['days']} days.",
           "Same code, split, labels, seed and settings for all; only model.inputs differs (and modality dropout, "
           "which needs two instruments)."])


def _class_table(pop: str):
    cl = _csv("metrics_classification.csv")
    rows = []
    for k in EXPS:
        for h in HEADS:
            r = {m: _one(cl, experiment=k, population=pop, head=h, metric=m) for m in
                 ("ROC_AUC", "TSS", "POD_recall", "precision", "F1", "false_alarm_ratio", "false_positive_rate",
                  "frequency_bias", "BSS_calibrated")}
            if not r["TSS"]:
                continue
            rows.append({"experiment": k, "target (>= C1)": HEAD_LABEL[h], "AUC": ci(r["ROC_AUC"]), "TSS": ci(r["TSS"]),
                         "POD": f(r["POD_recall"]["value"]), "precision": f(r["precision"]["value"]),
                         "F1": f(r["F1"]["value"]), "FAR (ratio)": f(r["false_alarm_ratio"]["value"]),
                         "POFD": f(r["false_positive_rate"]["value"]), "FB": f(r["frequency_bias"]["value"], 2),
                         "BSS (calibrated)": ci(r["BSS_calibrated"]), "threshold": r["TSS"]["decision_threshold"],
                         "windows": r["TSS"]["n_samples"], "positive": r["TSS"]["n_positive"], "days": r["TSS"]["n_days"]})
    return rows


def table3():
    notes = ["Thresholds: best TSS on each experiment's own validation windows. FAR: false alarm ratio; POFD: false "
             "positive rate; FB: frequency bias. Brackets: 95% intervals resampling whole test days (n = 500 draws, "
             "200 for AUC, seed 0). Only the >= C1 threshold is modelled; M1 and X1 alerts are in Table 5."]
    write("table3_classification", "Table 3. Flare heads on the common test windows (both instruments observing)",
          _class_table("common"), notes)
    write("table3b_classification_native", "Table 3b. Flare heads on each experiment's own test windows",
          _class_table("native"), notes)


def table4():
    fx = _csv("metrics_forecast.csv")
    labels = {**{k: k for k in EXPS}, "reference: GOES flux now (persistence)": "GOES flux now (persistence)",
              "reference: SoLEXS flux now (calibrated on training)": "SoLEXS flux now",
              "reference: climatology (training mean)": "climatology"}
    hs = sorted({int(r["horizon_min"]) for r in fx if r["population"] == "common"})
    rows = []
    for h in hs:
        for e, label in labels.items():
            r = {m: _one(fx, experiment=e, population="common", horizon_min=h, metric=m)
                 for m in ("MAE", "RMSE", "bias", "correlation")}
            if r["MAE"]:
                rows.append({"horizon": "now" if h == 0 else f"+{h} min", "forecast": label, "MAE (dex)": ci(r["MAE"]),
                             "RMSE (dex)": f(r["RMSE"]["value"]), "bias (dex)": f(r["bias"]["value"]),
                             "correlation": f(r["correlation"]["value"]), "windows": r["MAE"]["n_samples"]})
    write("table4_flux", "Table 4. Flux forecasts on the common test windows", rows,
          ["log10 GOES-18 XRS-B flux; model = median forecast. Brackets: 95% intervals resampling whole test days."])


def table5():
    lt = _csv("leadtime_summary.csv")
    src = _json(PAPER / "leadtime_source.json")
    prim = src["primary_false_alarms_per_day"]
    rows = []
    for r in lt:
        c = "C" if r["alert"].startswith("GOES >= C1") else "M"
        if r["scope"] != "all flares" or r["operating_point"] != f"fa_{prim[c]:g}" and not r["method"].endswith("(E4)"):
            continue
        rows.append({"alert": r["alert"], "method": r["method"], "operating point": r["operating_point"],
                     "validation false alarms/day": r["validation_false_alarms_per_day"],
                     "test false alarms/day": r["false_alarms_per_day_test"],
                     "time alert on (test)": r.get("alert_on_fraction_test", ""), "flares": r["n_flares"],
                     "detection rate": r["detection_rate"], "chance": r["chance_rate"], "event TSS": r["event_TSS"],
                     "median lead (min)": r["median_lead_min"],
                     "IQR (min)": f"{r['q25_lead_min']}-{r['q75_lead_min']}",
                     "p10-p90 (min)": f"{r.get('p10_lead_min_all', '')}-{r.get('p90_lead_min_all', '')}"
                     if r.get("p10_lead_min_all") else ""})
    cls = [{"alert": r["alert"], "method": r["method"], "operating point": r["operating_point"], "class": r["scope"],
            "flares": r["n_flares"], "detection rate": r["detection_rate"], "median lead (min)": r["median_lead_min"],
            "note": r.get("note", "")}
           for r in lt if r["scope"].startswith("GOES class") and r["method"].endswith("(E4)")
           and r["operating_point"] == f"fa_{prim['C' if r['alert'].startswith('GOES >= C1') else 'M']:g}"]
    notes = [f"E4 alerts at every operating point, references at the primary one ({prim['C']:g} and {prim['M']:g} false "
             "alarms per day on validation). Lead: GOES peak minus the first alert minute; chance: the same windows "
             "moved 2 h; event TSS = detection rate - chance. Thresholds fixed on validation. A false alarm is an ON "
             "episode with no flare, so a signal that is almost always ON has few false alarms but a high time on "
             "and chance rate (the hot-onset trigger row): read false alarms together with time on and chance.",
             f"Test period {src['test_period'][0]} to {src['test_period'][1]} UTC ({src['test_days_with_data']} days "
             "with data)."]
    write("table5_leadtime", "Table 5. Alerts: detection, false alarms and lead time", rows, notes)
    write("table5b_leadtime_by_class", "Table 5b. E4 alerts by GOES class (primary operating point)", cls,
          [f"Classes with fewer than {src['min_events_for_class_statistics']} flares are flagged: indicative only."])


def table6():
    cl, un = _csv("metrics_classification.csv"), _csv("metrics_uncertainty.csv")
    cal = []
    for k in EXPS:
        for h in HEADS:
            r = {m: _one(cl, experiment=k, population="native", head=h, metric=m)
                 for m in ("Brier_raw", "Brier_calibrated", "BSS_raw", "BSS_calibrated")}
            if r["Brier_raw"]:
                cal.append({"experiment": k, "target (>= C1)": HEAD_LABEL[h], "Brier raw": f(r["Brier_raw"]["value"], 4),
                            "Brier calibrated": f(r["Brier_calibrated"]["value"], 4), "BSS raw": f(r["BSS_raw"]["value"]),
                            "BSS calibrated": ci(r["BSS_calibrated"]), "windows": r["Brier_raw"]["n_samples"]})
    write("table6_calibration", "Table 6. Probability calibration (each experiment's own test windows)", cal,
          ["Isotonic calibration fitted on each experiment's validation windows only. BSS against the test-period "
           "base rate (solarflare.metrics.brier_skill_score)."])
    unc = []
    for k in EXPS:
        hs = sorted({int(r["horizon_min"]) for r in un if r["experiment"] == k and r["population"] == "native"})
        for h in hs:
            r = {m: _one(un, experiment=k, population="native", horizon_min=h, metric=m)
                 for m in ("coverage_raw", "width_raw_dex", "scale_from_validation", "coverage_scaled", "width_scaled_dex",
                           "pinball_q50")}
            unc.append({"experiment": k, "horizon": f"+{h} min", "interval": r["coverage_raw"]["interval"],
                        "nominal": r["coverage_raw"]["nominal_coverage"], "coverage raw": f(r["coverage_raw"]["value"]),
                        "width raw (dex)": f(r["width_raw_dex"]["value"]),
                        "scale (validation)": f(r["scale_from_validation"]["value"]),
                        "coverage scaled": f(r["coverage_scaled"]["value"]),
                        "width scaled (dex)": f(r["width_scaled_dex"]["value"]),
                        "pinball q50 (dex)": f(r["pinball_q50"]["value"], 4), "windows": r["coverage_raw"]["n_samples"]})
    write("table6b_uncertainty", "Table 6b. Flux prediction intervals (each experiment's own test windows)", unc,
          ["Coverage: fraction of truths inside the q10-q90 interval; scaled: widened by the factor fitted on "
           "validation (split-conformal). Width: mean q90 - q10."])


def table7():
    cl, fx, pr = _csv("metrics_classification.csv"), _csv("metrics_forecast.csv"), _csv("paired_comparisons.csv")
    rows = []
    for e in ["E4", "E4, HEL1OS withheld", "E1", "E4, SoLEXS withheld", "E2"]:
        a = _one(cl, experiment=e, population="common", head="in_flare", metric="ROC_AUC")
        b = _one(cl, experiment=e, population="common", head="flare_within_15min", metric="ROC_AUC")
        t = _one(cl, experiment=e, population="common", head="in_flare", metric="TSS")
        m = _one(fx, experiment=e, population="common", horizon_min=15, metric="MAE")
        rows.append({"model": e, "AUC in progress": ci(a), "TSS in progress": ci(t), "AUC within 15 min": ci(b),
                     "MAE flux +15 min (dex)": ci(m)})
    write("table7_robustness", "Table 7. Robustness to a missing instrument (common test windows)", rows,
          ["'withheld': the deployed E4 with one instrument's mask set to zero at prediction time (thresholds and "
           "calibration unchanged). E1 and E2 are trained on one instrument. 95% intervals resampling whole test days."])
    diffs = [{"comparison": r["comparison"], "quantity": r["quantity"], "horizon (min)": r["horizon_min"],
              "metric": r["metric"], "difference": ci(r, 4), "interval excludes 0": r["ci_excludes_zero"],
              "windows": r["n_samples"], "days": r["n_days"]} for r in pr]
    write("table7b_paired_differences", "Table 7b. Paired differences on the common test windows", diffs,
          ["Positive AUC/TSS/BSS differences and negative MAE differences favour the first model. Intervals resample "
           "the same whole test days for both models (500 draws, 200 for AUC, seed 0)."])


def table7c():
    hv = _json(S.ablations / "hel1os" / "hel1os_value.json")
    abl = [{"seed": s["seed"].removeprefix("seed_"), "peak log-MAE soft only (dex)": f(s["mae_soft_only"], 4),
            "peak log-MAE soft + hard (dex)": f(s["mae_soft_hard"], 4),
            "reduction (dex)": f"{f(s['gain_dex'], 4)} [{f(s['ci'][0], 4)}, {f(s['ci'][1], 4)}]",
            "relative": f(s["relative"]), "flares": s["n_flares"]} for s in hv["seeds"]]
    write("table7c_hel1os_rise_ablation",
          "Table 7c. Separately trained soft-only vs soft + hard rise-phase models (existing ablation, 3 seeds)", abl,
          [f"From outputs/ablations/hel1os/hel1os_value.json: {hv['verdict']}. Peak flux predicted during the rise, "
           "3 time-ordered folds; intervals: bootstrap over flares (1000 draws, seed 0)."])


def main() -> int:
    status = []
    for fn in (table1, table2, table3, table4, table5, table6, table7, table7c):
        try:
            fn()
            status.append(f"- {fn.__name__}: written")
        except Missing as exc:
            status.append(f"- {fn.__name__}: SKIPPED -- {exc}")
            print(f"  {fn.__name__} skipped: {exc}")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "README.md").write_text("# Tables\n\nGenerated by scripts/paper/make_tables.py from result files.\n\n"
                                      + "\n".join(status) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
