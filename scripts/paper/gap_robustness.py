"""Paper step 14 (continued): performance against real observation gaps.

    python scripts/paper/gap_robustness.py

Every window needs its instrument(s) to have observed at least
min_observed_fraction of the 2 h input, so test windows range from half-covered
to complete. This groups each experiment's own test windows by how much of the
input its instrument actually observed -- real gaps in the data, no simulated
degradation -- and scores each group with the same metrics, thresholds and
calibration as evaluate_experiments.py (fitted on validation, from the saved
predictions in outputs/paper/predictions/).

Writes paper_results/metrics_gap_robustness.csv.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from common import PAPER  # noqa: E402  (sets sys.path)
from paper_metrics import classification, fit_on_validation, flux
from solarflare.settings import load_settings

S = load_settings()
PRED = S.outputs / "paper" / "predictions"
#: coverage groups (fraction of the input window the instrument observed). E1 and
#: E2 have no window under 50% (their window rule); E4 does wherever the other
#: instrument carried the window.
BINS = [(0.0, 0.5, "<50%"), (0.5, 0.9, "50-90%"), (0.9, 1.0, "90-<100%"), (1.0, 1.01, "100%")]
#: which instrument's coverage groups each experiment
GROUPING = {"E1": [("soft", "SoLEXS")], "E2": [("hard", "HEL1OS")], "E4": [("soft", "SoLEXS"), ("hard", "HEL1OS")]}


def _load(name: str, split: str) -> dict:
    with np.load(PRED / f"{name}_{split}.npz") as z:
        return {k: z[k] for k in z.files}


def coverage(window_index: np.ndarray) -> dict[str, np.ndarray]:
    """Fraction of each window's input that SoLEXS / HEL1OS observed."""
    from solarflare.config import Config
    from solarflare.pipeline import prepare

    cfg = Config.from_json(S.model_dir / "reports" / "config.json")
    prep = prepare(cfg, verbose=False)
    L = cfg.steps_per_window
    out = {}
    for key, attr in (("soft", "soft_mask"), ("hard", "hard_mask")):
        cs = [np.concatenate([[0.0], np.cumsum(getattr(s, attr))]) for s in prep.segments]
        out[key] = np.array([(cs[prep.windows[i].seg][prep.windows[i].end] - cs[prep.windows[i].seg][prep.windows[i].end - L]) / L
                             for i in window_index])
    return out


def main() -> int:
    meta = json.loads((PRED / "predictions_meta.json").read_text("utf-8"))
    occ, hor, qs = meta["window"]["occurrence_min"], meta["window"]["forecast_min"], meta["window"]["quantiles"]
    test = {k: _load(k, "test") for k in GROUPING}
    val = {k: _load(k, "val") for k in GROUPING}
    t4, v4 = test["E4"], val["E4"]
    native_t = {"E1": t4["pop_soft"], "E2": t4["pop_hard"], "E4": np.ones(t4["pop_soft"].size, bool)}
    native_v = {"E1": v4["pop_soft"], "E2": v4["pop_hard"], "E4": np.ones(v4["pop_soft"].size, bool)}
    print("measuring input coverage of every test window (rebuilds the data, ~5 min) ...", flush=True)
    cov = coverage(t4["window_index"])
    rows = []
    for k, groups in GROUPING.items():
        fitted = fit_on_validation(val[k], native_v[k], occ)
        for inst, name in groups:
            for lo, hi, label in BINS:
                sel = native_t[k] & (cov[inst] >= lo) & (cov[inst] < hi)
                pop = f"{name} input coverage {label}"
                if sel.sum() < 100:
                    print(f"  {k} {pop}: {int(sel.sum())} windows, skipped (fewer than 100)")
                    continue
                rows += [dict(r, instrument=name, coverage_group=label)
                         for r in classification(k, pop, test[k], sel, fitted, occ)
                         + flux(k, pop, test[k], sel, hor, qs)]
                print(f"  {k} {pop}: {int(sel.sum()):,} windows", flush=True)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(PAPER / "metrics_gap_robustness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> paper_results/metrics_gap_robustness.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
