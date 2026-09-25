"""Paper steps 5-11 and 14-16: E1 / E2 / E4 on the frozen test set.

    python scripts/paper/evaluate_experiments.py                 # predict (GPU, ~20 min) + metrics
    python scripts/paper/evaluate_experiments.py --metrics-only  # metrics from the saved predictions

Experiments -- same code, split, seed and settings; only ``model.inputs`` differs:
  E1  SoLEXS only       outputs/paper/E1_solexs_only  (train --from-run outputs/model --set model.inputs=soft)
  E2  HEL1OS only       outputs/paper/E2_hel1os_only  (train --from-run outputs/model --set model.inputs=hard)
  E4  SoLEXS + HEL1OS   outputs/model                 (the final model)
Robustness: E4 scored with SoLEXS, or HEL1OS, withheld at prediction time.

Each experiment predicts every validation and test window once; the predictions
are saved to outputs/paper/predictions/ and every metric is computed from them
(paper_metrics.py), so all experiments are scored by one code path.

Populations (test):
  common  windows where SoLEXS and HEL1OS both observed the origin and at least
          min_observed_fraction of the input: every experiment has its inputs,
          and the paired comparisons are made here
  native  each experiment's own windows (E1: SoLEXS observed, E2: HEL1OS, E4: either)
Thresholds, probability calibration and interval scales come from each
experiment's own validation windows (native), never from test.

Writes to paper_results/: metrics_classification.csv, metrics_forecast.csv,
metrics_uncertainty.csv, reliability.csv, paired_comparisons.csv, experiments.json.
Stops with a message naming what is missing if a training has not finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from common import PAPER, write_json  # noqa: E402  (sets sys.path)
from paper_metrics import (BOOT, classification, fit_on_validation, flux, flux_references, paired, reliability,
                           uncertainty)
from solarflare.settings import load_settings

S = load_settings()
PRED = S.outputs / "paper" / "predictions"
EXPERIMENTS = {
    "E1": {"label": "SoLEXS only", "run": S.outputs / "paper" / "E1_solexs_only", "inputs": "soft"},
    "E2": {"label": "HEL1OS only", "run": S.outputs / "paper" / "E2_hel1os_only", "inputs": "hard"},
    "E4": {"label": "SoLEXS + HEL1OS", "run": S.model_dir, "inputs": "both"},
}
#: robustness: the deployed E4 model with one instrument withheld at prediction time
ROBUSTNESS = {"E4, SoLEXS withheld": ("E4", "soft"), "E4, HEL1OS withheld": ("E4", "hard")}
KEEP = ("p_inflare", "p_occurrence", "nowcast", "forecast", "peak", "y_in_flare", "y_nowcast", "y_nowcast_mask",
        "y_forecast", "y_forecast_mask", "y_occurrence", "y_occurrence_mask", "y_peak", "y_peak_mask", "y_t_unix",
        "y_persistence", "y_soft_origin", "y_hard_origin")


def _file(name: str, split: str):
    return PRED / f"{name.replace(', ', '__').replace(' ', '_')}_{split}.npz"


def not_ready() -> list[str]:
    out = []
    for key, e in EXPERIMENTS.items():
        run = e["run"]
        if not (run / "checkpoints" / "best.pt").exists() or not (run / "reports" / "evaluation.json").exists():
            out.append(f"{key} ({e['label']}): training not finished in {run}")
            continue
        inputs = json.loads((run / "reports" / "config.json").read_text("utf-8"))["model"].get("inputs", "both")
        if inputs != e["inputs"]:
            out.append(f"{key}: {run} was trained with model.inputs={inputs!r}, expected {e['inputs']!r}")
    return out


def predict_all() -> dict:
    """Predict every experiment on the whole validation and test splits (GPU)."""
    from solarflare.config import Config
    from solarflare.evaluate import solexs_no_change
    from solarflare.models.net import count_parameters
    from solarflare.pipeline import prepare, resolve_device
    from solarflare.preprocess.dataset import build_targets, instrument_windows
    from solarflare.products.model_tests import load_run, predict

    t0 = time.time()
    device = resolve_device("auto")
    cfg = Config.from_json(S.model_dir / "reports" / "config.json")
    prep = prepare(cfg, verbose=False)
    targets = [build_targets(s, cfg) for s in prep.segments]
    sets = {"val": prep.splits["val"], "test": prep.splits["test"]}
    print(f"data ready: val {len(sets['val']):,}, test {len(sets['test']):,} windows ({time.time() - t0:.0f} s)",
          flush=True)
    extra = {}
    for split, idx in sets.items():
        soft = set(instrument_windows(prep.segments, prep.windows, idx, cfg, "soft"))
        hard = set(instrument_windows(prep.segments, prep.windows, idx, cfg, "hard"))
        t = np.array([prep.windows[i].t_unix for i in idx])
        extra[split] = {"window_index": np.asarray(idx), "pop_soft": np.array([i in soft for i in idx]),
                        "pop_hard": np.array([i in hard for i in idx]), "solexs_now": solexs_no_change(prep, t)[0]}
    # climatology reference: mean flux target over the training windows (as solarflare.evaluate)
    vals = [targets[prep.windows[i].seg]["nowcast"][prep.windows[i].end - 1] for i in prep.splits["train"]
            if targets[prep.windows[i].seg]["nowcast_mask"][prep.windows[i].end - 1] > 0]
    meta = {"climatology_log_flux": float(np.mean(vals)), "experiments": {}, "robustness": {}}
    PRED.mkdir(parents=True, exist_ok=True)

    def save(name, split, p):
        np.savez_compressed(_file(name, split), **{k: p[k] for k in KEEP}, **extra[split])

    for key, e in EXPERIMENTS.items():
        r = load_run(e["run"], device, hel1os=True)
        out = predict(r, prep, targets, prep.windows, sets, device)
        for split, p in out.items():
            save(key, split, p)
        meta["experiments"][key] = {"label": e["label"], "run": str(e["run"]), "inputs": e["inputs"],
                                    "best_epoch": r["epoch"], "parameters": count_parameters(r["model"]),
                                    "seed": r["cfg"].train.seed}
        print(f"{key} predicted ({time.time() - t0:.0f} s)", flush=True)
    for name, (base, blank) in ROBUSTNESS.items():
        r = load_run(EXPERIMENTS[base]["run"], device, hel1os=True)
        r["blank"] = blank
        save(name, "test", predict(r, prep, targets, prep.windows, {"test": sets["test"]}, device)["test"])
        meta["robustness"][name] = {"model": base, "withheld_at_prediction": blank}
        print(f"{name} predicted ({time.time() - t0:.0f} s)", flush=True)
    meta["window"] = {"occurrence_min": [int(h / 60) for h in cfg.win.occurrence_horizons_s],
                      "forecast_min": [int(h / 60) for h in cfg.win.forecast_horizons_s],
                      "quantiles": list(cfg.win.quantiles), "min_observed_fraction": cfg.win.min_observed_fraction}
    (PRED / "predictions_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    return meta


def _load(name: str, split: str) -> dict:
    with np.load(_file(name, split)) as z:
        return {k: z[k] for k in z.files}


def _write_csv(name: str, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(PAPER / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> paper_results/{name} ({len(rows)} rows)")


def metrics(meta: dict) -> None:
    w = meta["window"]
    occ, hor, qs = w["occurrence_min"], w["forecast_min"], w["quantiles"]
    val = {k: _load(k, "val") for k in EXPERIMENTS}
    test = {k: _load(k, "test") for k in EXPERIMENTS}
    rob = {k: _load(k, "test") for k in ROBUSTNESS}
    t4, v4 = test["E4"], val["E4"]
    common = t4["pop_soft"] & t4["pop_hard"]
    native_t = {"E1": t4["pop_soft"], "E2": t4["pop_hard"], "E4": np.ones(common.size, bool)}
    native_v = {"E1": v4["pop_soft"], "E2": v4["pop_hard"], "E4": np.ones(v4["pop_soft"].size, bool)}
    fitted = {k: fit_on_validation(val[k], native_v[k], occ) for k in EXPERIMENTS}
    cls, rel, fx, unc = [], [], [], []
    for k in EXPERIMENTS:
        for pop, sel in (("common", common), ("native", native_t[k])):
            cls += classification(k, pop, test[k], sel, fitted[k], occ)
            rel += reliability(k, pop, test[k], sel, fitted[k], occ)
            fx += flux(k, pop, test[k], sel, hor, qs)
            unc += uncertainty(k, pop, test[k], sel, val[k], native_v[k], hor, qs)
        print(f"{k} scored", flush=True)
    for pop, sel in (("common", common), ("native", native_t["E4"])):
        fx += flux_references(pop, t4, sel, hor, qs, t4["solexs_now"], meta["climatology_log_flux"])
    for name in ROBUSTNESS:
        cls += classification(name, "common", rob[name], common, fitted["E4"], occ)
        fx += flux(name, "common", rob[name], common, hor, qs)
    pairs = []
    for a, b in (("E4", "E1"), ("E4", "E2"), ("E1", "E2")):
        pairs += paired(a, b, "common", test[a], test[b], common, fitted[a], fitted[b], occ, hor, qs)
    for name in ROBUSTNESS:
        pairs += paired("E4", name, "common", t4, rob[name], common, fitted["E4"], fitted["E4"], occ, hor, qs)
    PAPER.mkdir(parents=True, exist_ok=True)
    _write_csv("metrics_classification.csv", cls)
    _write_csv("reliability.csv", rel)
    _write_csv("metrics_forecast.csv", fx)
    _write_csv("metrics_uncertainty.csv", unc)
    _write_csv("paired_comparisons.csv", pairs)
    days = lambda sel: int(np.unique(np.floor(t4["y_t_unix"][sel] / 86400)).size)  # noqa: E731
    write_json("experiments.json", {
        **meta, "bootstrap": BOOT,
        "populations_test": {"common": {"windows": int(common.sum()), "days": days(common)},
                             **{f"native_{k}": {"windows": int(v.sum()), "days": days(v)} for k, v in native_t.items()}},
        "validation_fits": {k: {h: {"threshold": f["threshold"]} for h, f in fitted[k].items()} for k in fitted},
        "calibration": "isotonic regression (solarflare.probcal) on each experiment's own validation windows",
        "thresholds": "best TSS on each experiment's own validation windows (solarflare.metrics.best_threshold)",
        "interval_scale": "split-conformal widening factor fitted on validation (solarflare.metrics.interval_scale)",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--metrics-only", action="store_true", help="recompute metrics from the saved predictions")
    args = ap.parse_args()
    missing = not_ready()
    if missing:
        print("Not run -- experiments missing:\n  " + "\n  ".join(missing))
        return 2
    if args.metrics_only:
        meta = json.loads((PRED / "predictions_meta.json").read_text("utf-8"))
    else:
        meta = predict_all()
    metrics(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
