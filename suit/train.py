"""Train and score SUIT flare forecasts, exactly as the X-ray day-ahead model is scored.

For each flare class (>= C1, >= M1) and horizon (6, 12, 24 h): does a GOES flare
peak within ``H`` hours of the forecast origin? Two model families (logistic and
gradient-boosted trees, solarflare.products.dayahead) are fitted on training
hours; the family and its yes/no threshold are chosen on validation; the test
period is scored once, with week-block 95% intervals, the Brier skill against
training climatology, the frequency bias beside the TSS (Leka et al. 2019), and
a paired comparison with persistence ("a flare of that class in the last 24 h").

The split is the X-ray model's own (its train end and test start), so the two
are scored on the same days and can be compared or combined. When the SUIT data
is too short for that split -- a pilot month, say -- it falls back to its own
chronological split with an embargo and says so: those numbers are then not
comparable with the X-ray results.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from solarflare.products import dayahead as D

from .config import SuitConfig, SuitPaths
from .dataset import build_features, hourly_table, select_frames
from .io import index_frames

MIN_ROWS = 30


def xray_split(run_dir: Path) -> tuple[float, float] | None:
    meta = Path(run_dir) / "reports" / "data_meta.json"
    if not meta.exists():
        return None
    d = json.loads(meta.read_text("utf-8"))["split_dates"]
    return float(d["train_end"]), float(d["test_start"])


def masks(o, base, H, split, cfg):
    """(train, val, test, how). A target window never straddles a boundary."""
    end = o + 3600.0 * H
    if split is not None:
        t1, t2 = split
        tr, va, te = base & (end <= t1), base & (o > t1) & (end <= t2), base & (o >= t2)
        if min(tr.sum(), va.sum(), te.sum()) >= MIN_ROWS:
            return tr, va, te, "x-ray model split"
    ob = o[base]
    if ob.size < 3 * MIN_ROWS:
        return None
    f1, f2 = cfg.split_fractions
    t1, t2 = np.quantile(ob, f1), np.quantile(ob, f1 + f2)
    e = 86400.0 * cfg.embargo_days
    tr = base & (end <= t1)
    va = base & (o >= t1 + e) & (end <= t2)
    te = base & (o >= t2 + e)
    return tr, va, te, f"own chronological split, {cfg.embargo_days:g}-day embargo (not comparable with X-ray)"


def recent_flares(truth, o: np.ndarray, min_flux: float, hours: float = 24.0) -> np.ndarray:
    peaks = np.sort([f.peak_unix for f in truth.flares if f.peak_flux >= min_flux])
    return (np.searchsorted(peaks, o, side="right")
            - np.searchsorted(peaks, o - 3600.0 * hours, side="right")).astype(float)


def score(X, y, tr, va, te, weeks, truth, o, cls):
    """dayahead.fit_score on SUIT features, plus frequency bias and persistence."""
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # the logistic family needs finite inputs: training medians, never test ones
    med = np.nanmedian(np.where(tr[:, None], X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    clim = float(y[tr].mean())
    res, (model, pt) = D.fit_score(Xi, y, tr, va, te, weeks[te], clim)
    yt = y[te]
    res["FB"] = round(float((pt >= res["threshold"]).sum() / max(yt.sum(), 1)), 2)

    rp = recent_flares(truth, o, D.CLASSES[cls])[:, None]
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(rp[tr], y[tr])
    pp = pm.predict_proba(rp[te])[:, 1]
    thr = D.best_tss_threshold(y[va], pm.predict_proba(rp[va])[:, 1])
    res["persistence"] = {"AUC": round(float(roc_auc_score(yt, pp)), 3), "TSS": round(float(D.tss_at(yt, pp, thr)), 3)}
    res["gain_vs_persistence"] = D.paired_gain(yt, pt, pp, weeks[te])
    pi = permutation_importance(model, Xi[te], yt, scoring="roc_auc", n_repeats=5, random_state=0)
    res["_importance"] = pi.importances_mean
    return res


def run(paths: SuitPaths, cfg: SuitConfig, own_split: bool = False, verbose: bool = True, truth=None) -> dict:
    """``truth``: a GOES truth object (default: loaded from ``paths.goes``)."""
    from solarflare.io.goes import load_goes

    frames = index_frames(paths.data, paths.features / "frames_index.json", verbose)
    if not frames:
        raise SystemExit(f"no SUIT frames under {paths.data}")
    selected = select_frames(frames, cfg)
    feats = build_features(selected, cfg, paths.features, verbose)
    ts = np.concatenate([d["t"] for d in feats.values() if len(d["t"])] or [np.zeros(0)])
    if not ts.size:
        raise SystemExit("no full-disk frames in the chosen filters: see `python -m suit inventory`")
    o = np.arange(np.floor(ts.min() / 3600.0) * 3600.0 + 3600.0, ts.max() + 3600.0, 3600.0)
    X, names, have = hourly_table(feats, o, cfg)
    truth = truth if truth is not None else load_goes(paths.goes)
    split = None if own_split else xray_split(paths.split_run)
    weeks = np.floor(o / (7 * 86400.0))
    utc = lambda t: datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M")    # noqa: E731

    summary = {"frames": len(frames), "used_by_filter": {f: d["used"] for f, d in feats.items()},
               "rejected_by_filter": {f: d["rejected"] for f, d in feats.items()},
               "region_of_interest_frames_skipped": sum(min(f.nx, f.ny) < cfg.min_full_px for f in frames),
               "first": utc(ts.min()), "last": utc(ts.max()), "hours_with_suit": int(have.sum()),
               "features": len(names), "config": {k: v for k, v in vars(cfg).items() if k != "extra"},
               "results": {}}
    for cls in cfg.classes:
        for H in cfg.horizons_h:
            y, covered = D.window_targets(truth, o, 0.0, float(H))[cls]
            m = masks(o, have & covered, H, split, cfg)
            key = f">={cls}1 within {H} h"
            if m is None:
                summary["results"][key] = {"skipped": "too few hours with SUIT and GOES"}
                continue
            tr, va, te, how = m
            if min(tr.sum(), va.sum(), te.sum()) < MIN_ROWS or any(len(np.unique(y[s])) < 2 for s in (tr, va, te)):
                summary["results"][key] = {"skipped": "a split has too few hours or only one outcome", "split": how}
                continue
            res = score(X, y, tr, va, te, weeks, truth, o, cls)
            imp = res.pop("_importance")
            top = np.argsort(-imp)[:6]
            res.update(split=how, n={"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
                       base_rate_test=round(float(y[te].mean()), 3),
                       test_period=[utc(o[te].min()), utc(o[te].max())],
                       top_features=[[names[k], round(float(imp[k]), 4)] for k in top])
            summary["results"][key] = res
            if verbose:
                g = res["gain_vs_persistence"]
                print(f"  {key:18s} TSS {res['TSS']:+.3f} {res['TSS_ci']}  FB {res['FB']:.2f}  AUC {res['AUC']:.3f}  "
                      f"BSS {res['BSS_vs_training_climatology']:+.3f}  | persistence AUC {res['persistence']['AUC']:.3f}"
                      f"  gain {g['AUC_gain']:+.3f} {g['ci']}  [{res['selected']}; {how}]", flush=True)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    dest = paths.outputs / "suit_summary.json"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, dest)
    if verbose:
        print(f"wrote {dest}")
    return summary
