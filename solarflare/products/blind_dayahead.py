"""One day-ahead forecast per test day, from models frozen before the test began.

    python -m solarflare blind-dayahead

Replays the operational day forecast (``python -m solarflare day-forecast``) over
every UTC day of the test period, with day models that never saw it. The frozen
models day-forecast uses were refitted on every labelled hour, test period
included, so replaying them over past days would not be blind. Here:

- the SoLEXS -> GOES calibration and each class/lead model are fitted on the
  training period only; the model family (logistic or trees) and the yes/no
  threshold are chosen on validation; a second version is recalibrated
  (isotonic) on validation, since the flare rate fell between training and test;
- each day's forecast comes from the last full hour of SoLEXS data before the
  day starts, with the model whose lead is nearest, exactly as day-forecast does;
- outcomes (a GOES flare >= C1 / >= M1 peaking that UTC day, where GOES
  observed >= 80% of the day) are attached only after every forecast is made.

References: the training-period rate of flare days (climatology) and
persistence (the training-period rate of flare days after a flare day and
after a quiet day, applied to yesterday's GOES outcome). Scores per class:
Brier skill against both, AUC, TSS at the validation threshold, accuracy
beside the accuracy of always answering the more common outcome, and a
reliability table; 95% intervals resample whole weeks.

Writes outputs/tests/blind_dayahead/{REPORT.md, blind_dayahead.json, forecasts.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from solarflare import probcal
from solarflare.products import dayahead as da
from solarflare.settings import load_settings
from solarflare.util import utc

DAY = 86400.0
#: a forecast is issued only if a frozen lead lies within this of the actual lead
MAX_LEAD_MISMATCH_H = 3.0


def fit_day_models(X, have, o, truth, train_end: float, test_start: float) -> dict:
    """Per class and lead: the family chosen on validation, fitted on training
    hours only, its validation TSS threshold, and an isotonic recalibration
    fitted on its validation predictions."""
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(X).all(1)
    out = {}
    for L0 in da.DAY_LEADS_H:
        L1 = L0 + 24.0
        for c, (y, gok) in da.window_targets(truth, o, L0, L1).items():
            base = have & gok & finite
            tr = base & (o + 3600.0 * L1 <= train_end)
            va = base & (o > train_end) & (o + 3600.0 * L1 <= test_start)
            if min(tr.sum(), va.sum()) < 50 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            best = None
            for name, m in da._models().items():
                m.fit(X[tr], y[tr])
                pv = m.predict_proba(X[va])[:, 1]
                a = float(roc_auc_score(y[va], pv))
                if best is None or a > best["val_AUC"]:
                    best = {"family": name, "model": m, "val_AUC": a, "pv": pv}
            out[(c, L0)] = {"family": best["family"], "model": best["model"],
                            "val_AUC": round(best["val_AUC"], 3),
                            "threshold": da.best_tss_threshold(y[va], best["pv"]),
                            "recal": probcal.fit(best["pv"], y[va]),
                            "base_rate_train": round(float(y[tr].mean()), 3), "n_train": int(tr.sum())}
    return out


def day_outcomes(truth, days: np.ndarray) -> dict:
    """Per class: (flare that UTC day, GOES observed >= 80% of it)."""
    return da.window_targets(truth, days, 0.0, 24.0)


def persistence_rates(y: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
    """P(flare day | yesterday flare day), P(flare day | yesterday quiet) over
    consecutive observed days."""
    prev, cur = y[:-1], y[1:]
    both = ok[:-1] & ok[1:]
    after_yes = both & (prev == 1)
    after_no = both & (prev == 0)
    p1 = float(cur[after_yes].mean()) if after_yes.any() else float(cur[both].mean())
    p0 = float(cur[after_no].mean()) if after_no.any() else float(cur[both].mean())
    return p1, p0


def bss(y, p, ref) -> float:
    bs, br = np.mean((p - y) ** 2), np.mean((ref - y) ** 2)
    return float(1.0 - bs / br) if br > 0 else float("nan")


def reliability_table(y, p, edges=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "days": int(m.sum()),
                         "mean_forecast": round(float(p[m].mean()), 3),
                         "observed_rate": round(float(y[m].mean()), 3)})
    return rows


def score_class(y, p_raw, p_rec, yes, clim, pers, weeks) -> dict:
    from sklearn.metrics import roc_auc_score

    def auc(a, b):
        return float(roc_auc_score(a, b)) if len(np.unique(a)) == 2 else float("nan")

    def ci(stat, *cols):
        return da.week_ci(y, np.stack(cols, 1), stat, weeks)

    majority = max(y.mean(), 1 - y.mean())
    out = {"days": int(y.size), "flare_days": int(y.sum()), "base_rate_test": round(float(y.mean()), 3),
           "training_climatology": round(clim, 3),
           "AUC": round(auc(y, p_raw), 3),
           "AUC_ci": ci(lambda a, b: roc_auc_score(a, b[:, 0]), p_raw),
           "AUC_persistence": round(auc(y, pers), 3),
           "TSS": round(float(da.tss_at(y, yes.astype(float), 0.5)), 3),
           "accuracy": round(float(np.mean(yes == (y == 1))), 3),
           "accuracy_always_majority": round(float(majority), 3),
           "hits": int(np.sum(yes & (y == 1))), "misses": int(np.sum(~yes & (y == 1))),
           "false_alarms": int(np.sum(yes & (y == 0))), "correct_quiet": int(np.sum(~yes & (y == 0)))}
    for name, p in (("raw", p_raw), ("recalibrated", p_rec)):
        out[f"BSS_vs_climatology_{name}"] = round(bss(y, p, np.full_like(p, clim)), 3)
        out[f"BSS_vs_climatology_{name}_ci"] = ci(
            lambda a, b, _c=clim: bss(a, b[:, 0], np.full(len(a), _c)), p)
        out[f"BSS_vs_persistence_{name}"] = round(bss(y, p, pers), 3)
        out[f"BSS_vs_persistence_{name}_ci"] = ci(lambda a, b: bss(a, b[:, 0], b[:, 1]), p, pers)
        out[f"reliability_{name}"] = reliability_table(y, p)
    out["BSS_persistence_vs_climatology"] = round(bss(y, pers, np.full_like(pers, clim)), 3)
    # hindsight reference: the test period's own flare-day rate, unknown when forecasting
    rate = float(y.mean())
    out["BSS_vs_test_rate_recalibrated"] = round(bss(y, p_rec, np.full_like(p_rec, rate)), 3)
    out["BSS_vs_test_rate_recalibrated_ci"] = ci(
        lambda a, b, _r=rate: bss(a, b[:, 0], np.full(len(a), _r)), p_rec)
    return out


def render(r: dict) -> str:
    def c(x):
        return f"[{x[0]:+.2f}, {x[1]:+.2f}]" if x else "[n/a]"

    L = ["# Blind day-ahead replay", "",
         f"One forecast per UTC day, {r['period'][0]} -> {r['period'][1]}, from the last full hour of SoLEXS "
         f"data before the day (median lead {r['median_lead_h']:.1f} h). Models, calibration, family and "
         f"threshold fixed from data before {r['train_end'][:10]} (training) and the validation period to "
         f"{r['test_start'][:10]}; nothing from the test period. {r['days_forecast']} days forecast, "
         f"{r['days_skipped']} skipped (no SoLEXS data close enough before the day).", ""]
    for cls, s in r["classes"].items():
        L += [f"## >= {cls}1 flare in the day", "",
              f"{s['flare_days']} of {s['days']} days had one (test rate {s['base_rate_test']:.2f}; "
              f"training rate {s['training_climatology']:.2f}).", "",
              "| Score | raw | recalibrated on validation |", "|---|---|---|",
              f"| Brier skill vs climatology | {s['BSS_vs_climatology_raw']:+.3f} {c(s['BSS_vs_climatology_raw_ci'])} "
              f"| {s['BSS_vs_climatology_recalibrated']:+.3f} {c(s['BSS_vs_climatology_recalibrated_ci'])} |",
              f"| Brier skill vs persistence | {s['BSS_vs_persistence_raw']:+.3f} {c(s['BSS_vs_persistence_raw_ci'])} "
              f"| {s['BSS_vs_persistence_recalibrated']:+.3f} "
              f"{c(s['BSS_vs_persistence_recalibrated_ci'])} |",
              f"| Brier skill vs the test period's own rate (hindsight) | | "
              f"{s['BSS_vs_test_rate_recalibrated']:+.3f} {c(s['BSS_vs_test_rate_recalibrated_ci'])} |", "",
              f"AUC {s['AUC']:.3f} {c(s['AUC_ci'])} (persistence {s['AUC_persistence']:.3f}); persistence's own "
              f"Brier skill vs climatology {s['BSS_persistence_vs_climatology']:+.3f}.", "",
              f"Yes/no at each lead model's validation threshold: TSS {s['TSS']:.3f}; "
              f"{s['hits']} hits, {s['misses']} misses, {s['false_alarms']} false alarms, "
              f"{s['correct_quiet']} correct quiet days; accuracy {100 * s['accuracy']:.0f}% against "
              f"{100 * s['accuracy_always_majority']:.0f}% for always answering the more common outcome.", "",
              "Reliability (recalibrated): " + "; ".join(
                  f"{b['bin']}: said {b['mean_forecast']:.2f}, happened {b['observed_rate']:.2f} ({b['days']} d)"
                  for b in s["reliability_recalibrated"]), ""]
    L += ["## Reading it", "", *[f"- {x}" for x in r["notes"]], ""]
    return "\n".join(L)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--catalog", default=str(S.catalog / "master_catalog.csv"))
    ap.add_argument("--run-dir", default=None, help="trained run whose split to use (default: the final model)")
    ap.add_argument("--out", default=str(S.tests / "blind_dayahead"))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    split = S.split_dates(Path(args.run_dir) if args.run_dir else None)
    train_end, test_start = split["train_end"], split["test_start"]

    from solarflare.io.goes import load_goes

    truth = load_goes(Path(args.goes_dir))
    tm, rm, _, vm, hm, vh = da.solexs_minutes(args.cache_dir)
    cal = da.fit_calibration(tm, rm, vm, truth, train_end)
    lf, hard = da.flux_series(cal, rm, vm, hm, vh)
    known, pf = da.catalogue_flares(args.catalog)
    feat, have, o = da.activity_features(tm, lf, hard, known, pf)
    X = feat.to_numpy(dtype=np.float64)
    usable = have & np.isfinite(X).all(1)
    models = fit_day_models(X, have, o, truth, train_end, test_start)
    print(f"{len(models)} class/lead models fitted on training hours ({time.time() - t0:.0f} s)", flush=True)

    # 1. forecasts, before any outcome is looked at
    first = np.ceil(test_start / DAY) * DAY
    last = np.floor(float(tm[-1]) / DAY) * DAY + DAY          # the day after the last SoLEXS minute
    days = np.arange(first, last + 1, DAY)
    rows, skipped = [], 0
    for d in days:
        ok = np.flatnonzero(usable & (o < d))
        if not ok.size:
            skipped += 1
            continue
        i = int(ok[-1])
        lead = (d - o[i]) / 3600.0
        rec = {"day": utc(d, "%Y-%m-%d"), "origin_utc": utc(o[i], "%Y-%m-%d %H:%M"), "lead_h": round(lead, 1)}
        for c in da.CLASSES:
            leads = [L0 for (cc, L0) in models if cc == c]
            L0 = min(leads, key=lambda x: abs(x - lead))
            if abs(L0 - lead) > MAX_LEAD_MISMATCH_H:
                rec = None
                break
            m = models[(c, L0)]
            p = float(m["model"].predict_proba(X[[i]])[0, 1])
            rec.update({f"model_{c}": f"{c}@{L0:g}", f"p_{c}": round(p, 4),
                        f"p_{c}_recal": round(float(probcal.apply(m["recal"], np.array([p]))[0]), 4),
                        f"thr_{c}": m["threshold"]})
        if rec is None:
            skipped += 1
            continue
        rec["_t"] = d
        rows.append(rec)
    print(f"{len(rows)} daily forecasts issued, {skipped} days skipped", flush=True)

    # 2. outcomes, then scores
    td = np.array([r_["_t"] for r_ in rows])
    outc = day_outcomes(truth, td)
    # climatology and persistence from training-period days with SoLEXS data
    all_days = np.arange(np.floor(float(tm[0]) / DAY) * DAY, td[-1] + DAY, DAY)
    hist = day_outcomes(truth, all_days)
    gp = np.array([f.peak_unix for f in truth.flares])
    gcls = np.array([f.goes_class for f in truth.flares])
    for r_ in rows:
        on = (gp >= r_["_t"]) & (gp < r_["_t"] + DAY)
        r_["largest_goes_flare"] = (max(gcls[on], key=lambda s_: ("ABCMX".index(s_[0]), float(s_[1:])))
                                    if on.any() else "")
    weeks = np.floor(td / (7 * DAY)).astype(int)
    summary = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
               "train_end": utc(train_end), "test_start": utc(test_start),
               "period": [rows[0]["day"], rows[-1]["day"]], "days_forecast": len(rows), "days_skipped": skipped,
               "median_lead_h": float(np.median([r_["lead_h"] for r_ in rows])),
               "models": {f"{c}@{L0:g}": {k: v for k, v in m.items() if k not in ("model", "recal")}
                          for (c, L0), m in models.items()}, "classes": {}}
    for c in da.CLASSES:
        y, gok = outc[c]
        yh, okh = hist[c]
        train_days = okh & (all_days + DAY <= train_end)
        clim = float(yh[train_days].mean())
        p1, p0 = persistence_rates(yh, okh & (all_days + DAY <= train_end))
        # yesterday's outcome for each forecast day (GOES, known when the day ends)
        k = np.searchsorted(all_days, td - DAY)
        yday = np.where(okh[k], yh[k], np.nan)
        pers = np.where(yday == 1, p1, np.where(yday == 0, p0, clim))
        for r_, v, g in zip(rows, y, gok):
            r_[f"flare_{c}"] = int(v) if g else ""
        sel = gok.astype(bool)
        p_raw = np.array([r_[f"p_{c}"] for r_ in rows])
        p_rec = np.array([r_[f"p_{c}_recal"] for r_ in rows])
        yes = p_raw >= np.array([r_[f"thr_{c}"] for r_ in rows])
        s = score_class(y[sel], p_raw[sel], p_rec[sel], yes[sel], clim, pers[sel], weeks[sel])
        s["persistence_rates"] = {"after_flare_day": round(p1, 3), "after_quiet_day": round(p0, 3)}
        summary["classes"][c] = s
    summary["notes"] = [
        "A replay, not a live forecast: every input and model setting predates the test period, but the "
        "forecasts were computed after it. The sealed forecasts from `python -m solarflare day-forecast` "
        "are the live test.",
        "Test period near the Cycle 25 maximum, about six months; the >= C1 rate fell from ~0.99 of days "
        "in training to the test rate shown, which is why the raw probabilities over-forecast and the "
        "validation recalibration matters.",
        "Accuracy is shown only beside the always-the-common-answer accuracy: with most days flaring "
        "(>= C1) or quiet (>= M1), a high accuracy alone says nothing.",
    ]
    summary["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind_dayahead.json").write_text(json.dumps(summary, indent=2), "utf-8")
    cols = ["day", "origin_utc", "lead_h", "model_C", "p_C", "p_C_recal", "flare_C", "model_M", "p_M",
            "p_M_recal", "flare_M", "largest_goes_flare"]
    with open(out / "forecasts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "REPORT.md").write_text(render(summary), "utf-8")
    for c, s in summary["classes"].items():
        print(f">= {c}1: {s['days']} days, rate {s['base_rate_test']:.2f}; AUC {s['AUC']:.3f} (persistence "
              f"{s['AUC_persistence']:.3f}); BSS vs climatology raw {s['BSS_vs_climatology_raw']:+.3f}, "
              f"recalibrated {s['BSS_vs_climatology_recalibrated']:+.3f}; vs persistence "
              f"{s['BSS_vs_persistence_recalibrated']:+.3f}; TSS {s['TSS']:.3f}", flush=True)
    from solarflare.products.model_tests import write_index

    write_index(out.parent)
    print(f"-> {out / 'REPORT.md'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
