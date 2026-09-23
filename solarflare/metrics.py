"""Forecast verification metrics.

Accuracy is meaningless for flare forecasting: a model that always says "no
flare" scores 90%+ on a quiet week.  The operational standard is the True Skill
Statistic, which is insensitive to class balance, together with the Heidke
Skill Score; both are reported here alongside the usual POD/FAR/CSI and
probabilistic scores.
"""

from __future__ import annotations

import numpy as np


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    return {
        "TP": int(np.sum(y_true & y_pred)),
        "FP": int(np.sum(~y_true & y_pred)),
        "FN": int(np.sum(y_true & ~y_pred)),
        "TN": int(np.sum(~y_true & ~y_pred)),
    }


def skill_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = contingency(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn

    def d(a, b):
        return a / b if b else 0.0

    pod = d(tp, tp + fn)                 # recall / hit rate
    pofd = d(fp, fp + tn)                # false alarm rate
    far = d(fp, tp + fp)                 # false alarm ratio
    csi = d(tp, tp + fp + fn)            # critical success index
    tss = pod - pofd                     # true skill statistic
    # Frequency bias: how many events were forecast for every one that happened.
    # 1 is unbiased, >1 overforecasting, <1 underforecasting. Leka et al. (2019,
    # ApJS 243:36) require it beside TSS: at the low event rates typical of
    # flares an overforecasting system can reach a high TSS where a cautious one
    # cannot, so a TSS quoted on its own cannot be compared between methods.
    fb = d(tp + fp, tp + fn)

    exp_correct = d((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp), n) if n else 0.0
    hss = d((tp + tn) - exp_correct, n - exp_correct) if n else 0.0

    precision = d(tp, tp + fp)
    f1 = d(2 * precision * pod, precision + pod)

    return {
        **{k: float(v) for k, v in c.items()},
        "accuracy": float(d(tp + tn, n)),
        "POD": float(pod), "POFD": float(pofd), "FAR": float(far),
        "CSI": float(csi), "TSS": float(tss), "HSS": float(hss),
        "FB": float(fb),
        "precision": float(precision), "F1": float(f1),
        "base_rate": float(d(tp + fn, n)),
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray,
                   metric: str = "TSS") -> tuple[float, dict]:
    """Pick the operating point that maximises a skill score.

    0.5 is the right threshold only when the classes are balanced, which they
    never are here -- so the threshold is a fitted quantity, chosen on
    validation data and then held fixed for the test set.
    """
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, skill_scores(y_true, prob >= 0.5)
    cands = np.unique(np.round(np.clip(prob, 0, 1), 3))
    cands = np.unique(np.concatenate([cands, np.linspace(0.02, 0.98, 49)]))
    best, best_s, best_m = 0.5, -np.inf, {}
    for t in cands:
        s = skill_scores(y_true, prob >= t)
        if s[metric] > best_s:
            best, best_s, best_m = float(t), s[metric], s
    return best, best_m


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Brier score relative to always predicting the climatological rate."""
    if y_true.size == 0:
        return float("nan")
    base = float(np.mean(y_true))
    ref = float(np.mean((base - y_true) ** 2))
    if ref <= 0:
        return float("nan")
    return float(1.0 - brier_score(y_true, prob) / ref)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    # Average ranks over ties, otherwise AUC is biased for coarse scores.
    # Vectorised: a Python loop over ties made each call ~0.1 s on a test set,
    # too slow for the day-block bootstrap (hundreds of calls).
    order = np.argsort(score, kind="mergesort")
    _, inv, counts = np.unique(np.asarray(score)[order], return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    avg = (last - counts + 1 + last) / 2.0
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = avg[inv]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reliability(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict:
    """Reliability (calibration) curve.

    A forecast that says 30% should verify 30% of the time; without this the
    probabilities are just rankings.
    """
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    out = {"bin_centre": [], "predicted": [], "observed": [], "count": []}
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out["bin_centre"].append(float((edges[b] + edges[b + 1]) / 2))
        out["predicted"].append(float(prob[m].mean()))
        out["observed"].append(float(y_true[m].mean()))
        out["count"].append(int(m.sum()))
    return out


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        m = mask.astype(bool)
        y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def forecast_score_mask(future_valid: np.ndarray, origin_valid: np.ndarray) -> np.ndarray:
    """Windows on which a flux forecast is scored: the future target *and* the
    value at the forecast origin must both be real.

    Where the truth is missing the stored target is 0 (a GOES gap, a stretch
    with no soft X-ray truth). Scored there, persistence "forecasts" log flux 0
    -- six decades off in GOES W/m^2 -- and the model's skill over it is
    inflated. Model, persistence and climatology share this one sample.
    """
    return (np.asarray(future_valid) > 0) & (np.asarray(origin_valid) > 0)


def skill_vs_reference(y_true: np.ndarray, y_pred: np.ndarray,
                       y_ref: np.ndarray, mask: np.ndarray | None = None
                       ) -> float:
    """Fractional RMSE reduction against a reference forecast.

    For short horizons persistence is a genuinely strong baseline; a flare
    forecaster that cannot beat it has not earned its complexity, so this is
    reported next to every regression score.
    """
    if mask is not None:
        m = mask.astype(bool)
        y_true, y_pred, y_ref = y_true[m], y_pred[m], y_ref[m]
    if y_true.size == 0:
        return float("nan")
    rmse_m = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    rmse_r = float(np.sqrt(np.mean((y_ref - y_true) ** 2)))
    if rmse_r <= 0:
        return float("nan")
    return float(1.0 - rmse_m / rmse_r)


def multiclass_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "support": int(cm[c, :].sum()),
        }
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(n_classes)]))
    return {"confusion": cm.tolist(), "accuracy": acc,
            "macro_f1": macro_f1, "per_class": per_class}


# ---------------------------------------------------------------------------
# Uncertainty of the scores themselves, and of the forecast intervals
# ---------------------------------------------------------------------------

def day_block_ci(t_unix: np.ndarray, stat, n_boot: int = 500, seed: int = 0,
                 digits: int = 4) -> list[float] | None:
    """95% interval of ``stat(idx)`` resampling whole UTC days.

    Minutes of one day are not independent (one flare fills dozens of windows),
    so resampling windows would give intervals far too narrow. ``stat`` takes an
    index array into the arrays it closes over; draws where it fails (e.g. a
    resample with one class only) are skipped. None with fewer than 3 days."""
    days = np.floor(np.asarray(t_unix, dtype=np.float64) / 86400.0).astype(np.int64)
    order = np.argsort(days, kind="stable")
    uniq, starts = np.unique(days[order], return_index=True)
    if len(uniq) < 3:
        return None
    members = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        try:
            v = float(stat(idx))
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < n_boot // 2:
        return None
    return [round(float(np.percentile(draws, 2.5)), digits), round(float(np.percentile(draws, 97.5)), digits)]


def interval_scale(y: np.ndarray, lo: np.ndarray, mid: np.ndarray, hi: np.ndarray,
                   target: float) -> float:
    """Factor k that widens [lo, hi] about ``mid`` until a fraction ``target``
    of ``y`` falls inside: lo' = mid - k (mid - lo), hi' = mid + k (hi - mid).

    Fitted on validation (a split-conformal correction), so test coverage is an
    honest check. k > 1 widens intervals that were too narrow."""
    y, lo, mid, hi = (np.asarray(a, dtype=np.float64) for a in (y, lo, mid, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(mid) & np.isfinite(hi)
    if ok.sum() < 20:
        return 1.0
    y, lo, mid, hi = y[ok], lo[ok], mid[ok], hi[ok]
    eps = 1e-6
    need = np.where(y < mid, (mid - y) / np.maximum(mid - lo, eps), (y - mid) / np.maximum(hi - mid, eps))
    return float(max(np.quantile(need, target), 1e-3))


def apply_interval_scale(fc: np.ndarray, scales, mid_index: int) -> np.ndarray:
    """Quantile forecasts (N, horizons, quantiles) with each horizon's spread
    about its median multiplied by ``scales[h]`` (see interval_scale)."""
    out = np.array(fc, dtype=np.float64, copy=True)
    if not scales:
        return out
    k = np.asarray(scales, dtype=np.float64)[None, :, None]
    mid = out[:, :, mid_index:mid_index + 1]
    return mid + k * (out - mid)
