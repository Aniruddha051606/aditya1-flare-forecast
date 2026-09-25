"""Metrics for the paper tables, computed from saved predictions.

Pure functions over the dictionaries ``solarflare.train.collect_predictions``
returns (one entry per window), using the repository's own definitions
(``solarflare.metrics``) and its resampling of whole UTC days
(``day_block_ci``): a flare fills many 20 s windows of one day, so windows are
never resampled one by one. Nothing here reads data or models; every threshold,
calibration map and interval scale is fitted on validation predictions passed
in by the caller and only applied to test predictions.
"""

from __future__ import annotations

import numpy as np

from solarflare import probcal
from solarflare.metrics import (best_threshold, brier_score, brier_skill_score, day_block_ci, forecast_score_mask,
                                interval_scale, roc_auc, skill_scores)

#: Resampling for every interval in the paper tables.
BOOT = {"method": "bootstrap over whole UTC days of the test period (all windows of a day drawn together); "
                  "paired differences resample the same days for both models",
        "n_boot": 500, "n_boot_auc": 200, "seed": 0, "interval": "2.5-97.5 percentile (95%)"}

EVENT_CLASS = ">=C1.0 (GOES-18 XRS flare list)"


# ---- the classification heads ----------------------------------------------------

def heads(p: dict, occurrence_min: list[int]) -> dict[str, tuple]:
    """name -> (target, horizon_min, labelled mask, truth, probability)."""
    out = {"in_flare": ("flare in progress", 0, p["y_nowcast_mask"] > 0, p["y_in_flare"], p["p_inflare"])}
    for k, h in enumerate(occurrence_min):
        out[f"flare_within_{h}min"] = ("flare starts or is in progress within the horizon", h,
                                       p["y_occurrence_mask"][:, k] > 0, p["y_occurrence"][:, k],
                                       p["p_occurrence"][:, k])
    return out


def fit_on_validation(pv: dict, sel: np.ndarray, occurrence_min: list[int]) -> dict[str, dict]:
    """Per head, from validation predictions only: the best-TSS threshold on the
    raw probability (as solarflare.evaluate chooses it) and an isotonic
    calibration map (solarflare.probcal)."""
    out = {}
    for name, (_, _, m, y, p) in heads(pv, occurrence_min).items():
        mm = m & sel
        thr = best_threshold(y[mm], p[mm], "TSS")[0] if mm.any() and len(np.unique(y[mm])) > 1 else 0.5
        out[name] = {"threshold": float(thr), "calibration": probcal.fit(p[mm], y[mm])}
    return out


def _row(**kw) -> dict:
    for k in ("value", "ci95_low", "ci95_high"):
        v = kw.get(k)
        kw[k] = None if v is None or not np.isfinite(v) else round(float(v), 6)
    return kw


def _head_rows(base: dict, y, p, t, thr: float, cal: dict, boot: dict) -> list[dict]:
    pc = probcal.apply(cal, p)
    s = skill_scores(y, p >= thr)

    def ci(stat, n=boot["n_boot"]):
        return day_block_ci(t, stat, n_boot=n, seed=boot["seed"]) or [None, None]

    vals = [
        ("ROC_AUC", roc_auc(y, p), ci(lambda i: roc_auc(y[i], p[i]), boot["n_boot_auc"])),
        ("TSS", s["TSS"], ci(lambda i: skill_scores(y[i], p[i] >= thr)["TSS"])),
        ("POD_recall", s["POD"], None), ("precision", s["precision"], None), ("F1", s["F1"], None),
        ("false_alarm_ratio", s["FAR"], None), ("false_positive_rate", s["POFD"], None),
        ("HSS", s["HSS"], None), ("frequency_bias", s["FB"], None), ("accuracy", s["accuracy"], None),
        ("TP", s["TP"], None), ("FP", s["FP"], None), ("FN", s["FN"], None), ("TN", s["TN"], None),
        ("Brier_raw", brier_score(y, p), None), ("BSS_raw", brier_skill_score(y, p), None),
        ("Brier_calibrated", brier_score(y, pc), None),
        ("BSS_calibrated", brier_skill_score(y, pc), ci(lambda i: brier_skill_score(y[i], pc[i]))),
    ]
    return [_row(**base, metric=m, value=v, ci95_low=(c or [None, None])[0], ci95_high=(c or [None, None])[1])
            for m, v, c in vals]


def classification(experiment: str, population: str, pt: dict, sel: np.ndarray, fitted: dict,
                   occurrence_min: list[int], boot: dict = BOOT) -> list[dict]:
    """Every supported classification metric of every head on the test windows ``sel``."""
    rows = []
    for name, (target, h, m, y, p) in heads(pt, occurrence_min).items():
        mm = m & sel
        if not mm.any() or len(np.unique(y[mm])) < 2:
            continue
        t = pt["y_t_unix"][mm]
        base = {"experiment": experiment, "population": population, "head": name, "target": target,
                "horizon_min": h, "event_class": EVENT_CLASS, "split": "test", "n_samples": int(mm.sum()),
                "n_positive": int(y[mm].sum()), "n_days": int(np.unique(np.floor(t / 86400)).size),
                "decision_threshold": round(fitted[name]["threshold"], 4)}
        rows += _head_rows(base, y[mm], p[mm], t, fitted[name]["threshold"], fitted[name]["calibration"], boot)
    return rows


def reliability(experiment: str, population: str, pt: dict, sel: np.ndarray, fitted: dict,
                occurrence_min: list[int], bins: int = 10) -> list[dict]:
    """Reliability-diagram data: equal-width probability bins, raw and calibrated."""
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for name, (_, h, m, y, p) in heads(pt, occurrence_min).items():
        mm = m & sel
        y, p = y[mm], p[mm]
        for kind, q in (("raw", p), ("calibrated", probcal.apply(fitted[name]["calibration"], p))):
            k = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
            for b in range(bins):
                inb = k == b
                if inb.any():
                    rows.append({"experiment": experiment, "population": population, "head": name,
                                 "horizon_min": h, "probability": kind, "bin_low": round(edges[b], 2),
                                 "bin_high": round(edges[b + 1], 2), "n_samples": int(inb.sum()),
                                 "mean_forecast": round(float(q[inb].mean()), 6),
                                 "observed_frequency": round(float(y[inb].mean()), 6)})
    return rows


# ---- flux ----------------------------------------------------------------------------

def flux_targets(p: dict, horizons_min: list[int], quantiles: list[float]) -> dict[int, tuple]:
    """horizon_min -> (scored mask, truth, median forecast); 0 = flux now (nowcast)."""
    i50 = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))
    out = {0: (p["y_nowcast_mask"] > 0, p["y_nowcast"], p["nowcast"])}
    for k, h in enumerate(horizons_min):
        out[h] = (forecast_score_mask(p["y_forecast_mask"][:, k], p["y_nowcast_mask"]),
                  p["y_forecast"][:, k], p["forecast"][:, k, i50])
    return out


def _flux_rows(label: str, population: str, h: int, y, f, t, boot, kind: str) -> list[dict]:
    e = f - y
    base = {"experiment": label, "population": population, "horizon_min": h, "split": "test", "kind": kind,
            "units": "dex (log10 W m^-2, GOES-18 XRS-B)", "n_samples": int(y.size),
            "n_days": int(np.unique(np.floor(t / 86400)).size)}
    c = day_block_ci(t, lambda i: float(np.mean(np.abs(e[i]))), n_boot=boot["n_boot"], seed=boot["seed"]) or [None, None]
    corr = float(np.corrcoef(f, y)[0, 1]) if y.size > 2 and np.std(f) > 0 and np.std(y) > 0 else float("nan")
    return [_row(**base, metric="MAE", value=np.mean(np.abs(e)), ci95_low=c[0], ci95_high=c[1]),
            _row(**base, metric="RMSE", value=np.sqrt(np.mean(e ** 2))),
            _row(**base, metric="bias", value=np.mean(e)),
            _row(**base, metric="correlation", value=corr)]


def flux(experiment: str, population: str, pt: dict, sel: np.ndarray, horizons_min: list[int],
         quantiles: list[float], boot: dict = BOOT) -> list[dict]:
    rows = []
    for h, (m, y, f) in flux_targets(pt, horizons_min, quantiles).items():
        mm = m & sel
        if mm.any():
            rows += _flux_rows(experiment, population, h, y[mm], f[mm], pt["y_t_unix"][mm], boot, "model (median)")
    return rows


def flux_references(population: str, pt: dict, sel: np.ndarray, horizons_min: list[int], quantiles: list[float],
                    solexs_now: np.ndarray, climatology: float, boot: dict = BOOT) -> list[dict]:
    """The "no change" and climatology references on the same windows as the models."""
    rows = []
    for h, (m, y, _) in flux_targets(pt, horizons_min, quantiles).items():
        mm = m & sel
        refs = {"reference: climatology (training mean)": np.full(y.shape, climatology),
                "reference: SoLEXS flux now (calibrated on training)": solexs_now}
        if h > 0:
            refs["reference: GOES flux now (persistence)"] = pt["y_persistence"]
        for label, r in refs.items():
            ok = mm & np.isfinite(r)
            if ok.any():
                rows += _flux_rows(label, population, h, y[ok], r[ok], pt["y_t_unix"][ok], boot, "reference")
    return rows


def pinball(y: np.ndarray, f: np.ndarray, q: float) -> float:
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def uncertainty(experiment: str, population: str, pt: dict, sel: np.ndarray, pv: dict, sel_v: np.ndarray,
                horizons_min: list[int], quantiles: list[float]) -> list[dict]:
    """Quantile forecasts: empirical coverage and width of the q_lo-q_hi interval,
    raw and widened by a factor fitted on validation (split-conformal, as
    solarflare.evaluate does), and the pinball loss of each quantile."""
    qs = list(quantiles)
    i50 = int(np.argmin(np.abs(np.array(qs) - 0.5)))
    nominal = round(qs[-1] - qs[0], 4)
    rows = []
    for k, h in enumerate(horizons_min):
        mv = forecast_score_mask(pv["y_forecast_mask"][:, k], pv["y_nowcast_mask"]) & sel_v
        scale = interval_scale(pv["y_forecast"][mv, k], pv["forecast"][mv, k, 0], pv["forecast"][mv, k, i50],
                               pv["forecast"][mv, k, -1], nominal)
        mm = forecast_score_mask(pt["y_forecast_mask"][:, k], pt["y_nowcast_mask"]) & sel
        y, lo, mid, hi = (pt["y_forecast"][mm, k], pt["forecast"][mm, k, 0], pt["forecast"][mm, k, i50],
                          pt["forecast"][mm, k, -1])
        lo_s, hi_s = mid - scale * (mid - lo), mid + scale * (hi - mid)
        base = {"experiment": experiment, "population": population, "horizon_min": h, "split": "test",
                "interval": f"q{int(qs[0] * 100)}-q{int(qs[-1] * 100)}", "nominal_coverage": nominal,
                "n_samples": int(y.size)}
        vals = [("coverage_raw", np.mean((y >= lo) & (y <= hi))), ("width_raw_dex", np.mean(hi - lo)),
                ("scale_from_validation", scale),
                ("coverage_scaled", np.mean((y >= lo_s) & (y <= hi_s))), ("width_scaled_dex", np.mean(hi_s - lo_s))]
        vals += [(f"pinball_q{int(q * 100)}", pinball(y, pt["forecast"][mm, k, j], q)) for j, q in enumerate(qs)]
        rows += [_row(**base, metric=n, value=v) for n, v in vals]
    return rows


# ---- paired differences ----------------------------------------------------------------

def _paired_head(label: str, population: str, name: str, h: int, y, p, q, t, fa: dict, fb: dict,
                 boot: dict) -> list[dict]:
    ta, tb = fa["threshold"], fb["threshold"]
    ca, cb = probcal.apply(fa["calibration"], p), probcal.apply(fb["calibration"], q)
    stats = {
        "ROC_AUC": (lambda i: roc_auc(y[i], p[i]) - roc_auc(y[i], q[i]), boot["n_boot_auc"]),
        "TSS": (lambda i: skill_scores(y[i], p[i] >= ta)["TSS"] - skill_scores(y[i], q[i] >= tb)["TSS"],
                boot["n_boot"]),
        "BSS_calibrated": (lambda i: brier_skill_score(y[i], ca[i]) - brier_skill_score(y[i], cb[i]), boot["n_boot"]),
    }
    rows = []
    for metric, (stat, n) in stats.items():
        c = day_block_ci(t, stat, n_boot=n, seed=boot["seed"]) or [None, None]
        rows.append(_row(comparison=label, population=population, quantity=name, horizon_min=h, metric=metric,
                         value=stat(np.arange(y.size)), ci95_low=c[0], ci95_high=c[1], n_samples=int(y.size),
                         n_days=int(np.unique(np.floor(t / 86400)).size),
                         ci_excludes_zero=bool(c[0] is not None and (c[0] > 0 or c[1] < 0))))
    return rows


def _paired_flux(label: str, population: str, h: int, ea, eb, t, boot: dict) -> dict:
    c = day_block_ci(t, lambda i: float(ea[i].mean() - eb[i].mean()), n_boot=boot["n_boot"],
                     seed=boot["seed"]) or [None, None]
    return _row(comparison=label, population=population, quantity="flux", horizon_min=h, metric="MAE_dex",
                value=ea.mean() - eb.mean(), ci95_low=c[0], ci95_high=c[1], n_samples=int(ea.size),
                n_days=int(np.unique(np.floor(t / 86400)).size),
                ci_excludes_zero=bool(c[0] is not None and (c[0] > 0 or c[1] < 0)))


def paired(a: str, b: str, population: str, pa: dict, pb: dict, sel: np.ndarray, fa: dict, fb: dict,
           occurrence_min: list[int], horizons_min: list[int], quantiles: list[float],
           boot: dict = BOOT) -> list[dict]:
    """a - b on the same test windows, each model with its own validation threshold
    and calibration; intervals resample the same whole days for both."""
    label, rows = f"{a} - {b}", []
    hb = heads(pb, occurrence_min)
    for name, (_, h, m, y, p) in heads(pa, occurrence_min).items():
        mm = m & sel
        if mm.any() and len(np.unique(y[mm])) > 1:
            rows += _paired_head(label, population, name, h, y[mm], p[mm], hb[name][4][mm], pa["y_t_unix"][mm],
                                 fa[name], fb[name], boot)
    fb_ = flux_targets(pb, horizons_min, quantiles)
    for h, (m, y, f) in flux_targets(pa, horizons_min, quantiles).items():
        mm = m & sel
        if mm.any():
            rows.append(_paired_flux(label, population, h, np.abs(f[mm] - y[mm]), np.abs(fb_[h][2][mm] - y[mm]),
                                     pa["y_t_unix"][mm], boot))
    return rows
