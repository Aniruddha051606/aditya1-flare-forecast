"""Sealed forecast for one UTC day: will the Sun produce a >= C1 / >= M1 flare?

    python -m solarflare day-forecast --day 2026-09-21
    python -m solarflare day-forecast --day 2026-09-21 --extra-cache outputs/_dev/forward_check/cache

Uses the day models frozen by ``python -m solarflare dayahead`` (SoLEXS activity
features, one model per class and lead) and the latest SoLEXS data in the
preprocessing cache plus any ``--extra-cache`` folders (e.g. days uploaded after
the last pipeline run). No GOES data are read: the SoLEXS -> GOES calibration
was frozen with the models.

The forecast is issued from the last full hour of SoLEXS data before the day
starts, with the model whose lead is nearest (day start - that hour). It is
written once to outputs/dayahead/forecasts/forecast_<day>.json with a SHA-256 of
its content; a day that already has a sealed forecast is never re-issued.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.products.dayahead import (FROZEN_DAY, activity_features, catalogue_flares,
                                          flux_series, solexs_minutes)
from solarflare.settings import load_settings

#: A forecast whose lead is further than this from every frozen lead is refused.
MAX_LEAD_MISMATCH_H = 3.0


def merge_minutes(parts: list[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    """Union of several SoLEXS minute series on one contiguous minute grid.

    Each part is (time, rate, counts, valid, high rate, high valid) as from
    ``solexs_minutes``. Where parts overlap, a later part's valid minutes win."""
    parts = [p for p in parts if len(p[0])]
    if not parts:
        raise ValueError("no SoLEXS minutes in any cache")
    t0 = min(float(p[0][0]) for p in parts)
    t1 = max(float(p[0][-1]) for p in parts)
    grid = t0 + 60.0 * np.arange(int(round((t1 - t0) / 60.0)) + 1)
    out = [grid, np.full(grid.size, np.nan), np.zeros(grid.size), np.zeros(grid.size, bool),
           np.full(grid.size, np.nan), np.zeros(grid.size, bool)]
    for tm, rm, cm, vm, hm, vh in parts:
        k = np.rint((np.asarray(tm) - t0) / 60.0).astype(np.int64)
        use = np.asarray(vm, bool)
        out[1][k[use]], out[2][k[use]], out[3][k[use]] = rm[use], cm[use], True
        use_h = use & np.asarray(vh, bool)
        out[4][k[use_h]], out[5][k[use_h]] = hm[use_h], True
    return tuple(out)


def pick_lead(keys: list[str], cls: str, lead_h: float) -> tuple[str, float]:
    """The frozen model for ``cls`` whose lead is nearest ``lead_h``."""
    leads = [float(k.split("@")[1]) for k in keys if k.startswith(f"{cls}@")]
    if not leads:
        raise KeyError(f"no frozen day model for class {cls}")
    best = min(leads, key=lambda x: abs(x - lead_h))
    return f"{cls}@{best:g}", abs(best - lead_h)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--day", required=True, help="UTC day to forecast, YYYY-MM-DD")
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--extra-cache", action="append", default=[],
                    help="another preprocessing cache with newer SoLEXS days (repeatable)")
    ap.add_argument("--catalog", default=str(S.catalog / "master_catalog.csv"))
    ap.add_argument("--models", default=str(S.dayahead / FROZEN_DAY))
    ap.add_argument("--out", default=str(S.dayahead / "forecasts"))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    day0 = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=UTC)
    dest = Path(args.out) / f"forecast_{args.day}.json"
    if dest.exists():
        print(f"{args.day} already has a sealed forecast ({dest}); it is never re-issued.")
        return 1
    with open(args.models, "rb") as fh:
        fz = pickle.load(fh)

    main_part = solexs_minutes(args.cache_dir)
    parts = [main_part] + [solexs_minutes(c) for c in args.extra_cache]
    tm, rm, cm, vm, hm, vh = merge_minutes(parts)
    lf, hard = flux_series(fz["calibration"], rm, vm, hm, vh)

    # Flares: the master catalogue for what the pipeline processed, then the same
    # SoLEXS rule on anything newer (the catalogue ends with the main cache).
    from solarflare.catalog.detect import noaa_events

    known, pf = catalogue_flares(args.catalog)
    cat_end = float(main_part[0][-1] + 60.0)
    w = tm >= cat_end - 86400.0              # a day of lead-in so a flare in progress is whole
    with np.errstate(divide="ignore", invalid="ignore"):
        new = [e for e in noaa_events(tm[w], fz["calibration"](np.where(vm[w], rm[w], np.nan)), vm[w],
                                      counts=cm[w])
               if e.peak_unix >= cat_end] if w.any() else []
    keep = known - 300.0 < cat_end
    known = np.concatenate([known[keep], [e.peak_unix + 300.0 for e in new]])
    pf = np.concatenate([pf[keep], [e.peak_flux for e in new]])
    order = np.argsort(known)
    known, pf = known[order], pf[order]

    feat, have, o = activity_features(tm, lf, hard, known, pf)
    feat = feat[fz["features"]]
    X = feat.to_numpy(dtype=np.float64)
    ok = np.flatnonzero(have & np.isfinite(X).all(1) & (o < day0.timestamp()))
    if not ok.size:
        print("no hour before the forecast day has enough SoLEXS data (>= 50% of the previous 6 h)")
        return 1
    i = int(ok[-1])
    lead = (day0.timestamp() - o[i]) / 3600.0

    result = {}
    for c in ("C", "M"):
        key, miss = pick_lead(list(fz["models"]), c, lead)
        if miss > MAX_LEAD_MISMATCH_H:
            print(f"the last usable data hour is {lead:.0f} h before {args.day}; the frozen leads "
                  f"({', '.join(k for k in fz['models'] if k.startswith(c))}) do not reach that far")
            return 1
        p = float(fz["models"][key].predict_proba(X[[i]])[0, 1])
        result[c] = {"probability": round(p, 3), "model": key, **fz["skill"][key]}

    sealed = {
        "forecast_for": f"{args.day} 00:00-24:00 UTC ({(day0 + timedelta(hours=5.5)):%d %b %H:%M} IST onwards)",
        "issued_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "origin_utc": datetime.fromtimestamp(o[i], UTC).strftime("%Y-%m-%d %H:%M"),
        "lead_hours": round(lead, 2),
        "data_used": f"SoLEXS minutes to {datetime.fromtimestamp(float(tm[-1]) + 60, UTC):%Y-%m-%d %H:%M} UTC; "
                     f"day models frozen {fz['created_utc']} UTC; no GOES",
        "new_solexs_flares_after_catalogue": [
            [datetime.fromtimestamp(e.peak_unix, UTC).strftime("%Y-%m-%d %H:%M"), f"{e.peak_flux:.2e}"]
            for e in new],
        "features_at_origin": {k: (None if not np.isfinite(v) else round(float(v), 4))
                               for k, v in pd.Series(X[i], index=fz["features"]).items()},
        "C": result["C"], "M": result["M"],
    }
    body = json.dumps(sealed, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"sha256": digest, **sealed}, indent=2), encoding="utf-8")
    for c in ("C", "M"):
        r = result[c]
        print(f">= {c}1 on {args.day}: {100 * r['probability']:.0f}%  (model {r['model']}, "
              f"test AUC {r.get('test_AUC')}, last-30-day base rate {r.get('base_rate_last_30d')})")
    print(f"issued from {sealed['origin_utc']} UTC, {lead:.0f} h before the day; sealed "
          f"sha256 {digest[:16]} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
