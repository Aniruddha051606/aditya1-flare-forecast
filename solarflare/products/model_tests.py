"""Tests of the trained network beyond its own evaluation, written to outputs/tests/.

    python -m solarflare model-tests                  # all three
    python -m solarflare model-tests --only peak,bias

baseline_vs_final/
    The SoLEXS-only baseline (outputs/ablations/baseline_solexs_only) and the
    final SoLEXS + HEL1OS model on the same windows. Their own test reports
    cannot be compared: the split moved when HEL1OS was ingested. Here both are
    scored on the period both held out (the baseline's test start to the end of
    the data), on origins SoLEXS observed with enough of the input window, each
    as it was trained: the baseline with its own normaliser and HEL1OS blanked
    (it trained without it), each with the thresholds it chose on its own
    validation split and probabilities calibrated on that split. Differences
    final - baseline carry 95% intervals that resample whole days; they are
    given for all these windows and for those HEL1OS also observed.
peak_nowcast/
    The ongoing flare's peak flux predicted 3 min after its GOES start, the
    setting of arXiv 2608.20062 (GOES-only seq2seq LSTM, 1997-2024). One
    prediction per GOES flare >= C1 in the test period, RMSE in dex by class and
    C vs >= M classification, against simple references: GOES flux at that
    minute, the same plus the typical training-period rise, and calibrated
    SoLEXS flux (all an Aditya-only system has).
forecast_bias/
    Frequency bias (FB = forecast / observed events) of the in-flare and
    occurrence heads at three operating points chosen on validation (best TSS,
    best HSS, FB = 1), raw and calibrated, beside the base rate of each split:
    is over-forecasting a property of the model or of the TSS-optimal threshold?
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from solarflare import probcal
from solarflare.config import Config
from solarflare.metrics import best_threshold, brier_skill_score, day_block_ci, roc_auc, skill_scores
from solarflare.preprocess.dataset import (PEAK_TIME_SCALE_S, Normalizer, WindowIndex, build_targets)
from solarflare.settings import load_settings
from solarflare.util import utc

TESTS = ("compare", "peak", "bias")
#: log10 W/m^2 at the bottom of each GOES class
CLASS_FLOOR = {"C": -6.0, "M": -5.0, "X": -4.0}
#: arXiv 2608.20062 as reported (RMSE in dex). Its one figure for predictions made
#: exactly 3 min after onset is >= M; the class rows and C vs >= M scores appear to
#: pool prediction times from +3 min on, when the flare is better developed.
PAPER = {"reference": "arXiv 2608.20062 (2026), attention seq2seq LSTM on GOES 0.1-0.8 nm, 1997-2024, "
                      "19,438 flares, 4-fold CV by month",
         "RMSE_dex_at_3min": {">=M": 0.67},
         "RMSE_dex_as_reported": {">=C": 0.26, ">=M": 0.45, "X": 0.87},
         "C_vs_M": {"TSS": 0.68, "F1": 0.77}}
ONSET_DELAY_S = 180.0


# ---- models and predictions ----------------------------------------------------

def load_run(run: Path, device: torch.device, hel1os: bool) -> dict:
    from solarflare.train import load_model

    ck = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_json(run / "reports" / "config.json")
    ev = json.loads((run / "reports" / "evaluation.json").read_text("utf-8"))
    split = json.loads((run / "reports" / "data_meta.json").read_text("utf-8"))["split_dates"]
    return {"run": run, "cfg": cfg, "model": load_model(run / "checkpoints" / "best.pt", cfg, device),
            "norm": Normalizer(**{k: np.asarray(v) for k, v in ck["norm"].items()}),
            "epoch": ck.get("epoch"), "hel1os": hel1os,
            "thr": {"in_flare": float(ev["nowcast_in_flare"]["threshold"]),
                    **{f"occ_{k}": float(v["threshold"]) for k, v in ev["forecast_occurrence"].items()}},
            "train_end": float(split["train_end"]), "test_start": float(split["test_start"])}


def _withheld(seg, which: str):
    """A segment with one instrument withheld ("soft" or "hard"): the zeros and
    zero mask the dataset writes where that instrument did not observe. With
    "hard" this is how the SoLEXS-only baseline saw every day it trained on."""
    s = copy.copy(seg)
    if which == "hard":
        s.hard, s.hard_mask = np.zeros_like(seg.hard), np.zeros_like(seg.hard_mask)
    else:
        s.soft, s.soft_mask = np.zeros_like(seg.soft), np.zeros_like(seg.soft_mask)
    return s


def predict(r: dict, prep, targets: list, windows: list, sets: dict[str, list[int]],
            device: torch.device) -> dict[str, dict | None]:
    """Predictions of one run on several window sets, inputs normalised with the
    run's own normaliser; device memory is released afterwards. ``r["blank"]``
    ("soft"/"hard") withholds an instrument; a run loaded with hel1os=False
    withholds HEL1OS."""
    from solarflare.torch_data import GatherBatches, normalized_inputs
    from solarflare.train import collect_predictions

    blank = r.get("blank") or (None if r["hel1os"] else "hard")
    segs = prep.segments if blank is None else [_withheld(s, blank) for s in prep.segments]
    shared = {"targets": targets, **normalized_inputs(segs, r["norm"])}
    cfg = r["cfg"]
    out = {}
    for name, idx in sets.items():
        if not idx:
            out[name] = None
            continue
        gb = GatherBatches(segs, windows, idx, cfg, shared, device, int(cfg.train.eval_batch_size))
        out[name] = collect_predictions(r["model"], gb, device)
    shared.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ---- window sets ----------------------------------------------------------------

def soft_windows(prep, cfg: Config, idx, t0: float, t1: float = np.inf) -> list[int]:
    """Windows in [t0, t1] whose origin SoLEXS observed, with SoLEXS over
    ``min_observed_fraction`` of the input: the windows a SoLEXS-only model had."""
    L = cfg.steps_per_window
    cs = [np.concatenate([[0.0], np.cumsum(s.soft_mask)]) for s in prep.segments]
    keep = []
    for i in idx:
        w = prep.windows[i]
        if not (t0 <= w.t_unix <= t1):
            continue
        c, m = cs[w.seg], prep.segments[w.seg].soft_mask
        if m[w.end - 1] > 0 and (c[w.end] - c[w.end - L]) / L >= cfg.win.min_observed_fraction:
            keep.append(i)
    return keep


def onset_windows(prep, cfg: Config, t0: float, t1: float = np.inf) -> tuple[list[WindowIndex], list[dict]]:
    """One window per GOES flare whose origin, 3 min after its GOES start, lies in
    [t0, t1], where SoLEXS observed the origin and enough of the input."""
    L = cfg.steps_per_window
    k = int(round(ONSET_DELAY_S / cfg.pre.dt_seconds))
    wins, info = [], []
    for si, s in enumerate(prep.segments):
        cs = np.concatenate([[0.0], np.cumsum(s.soft_mask)])
        for e in s.events:
            o = e.start_idx + k
            if o >= e.end_idx or o + 1 > len(s) or o + 1 < L or not (t0 <= s.time_unix[o] <= t1):
                continue
            if s.soft_mask[o] <= 0 or (cs[o + 1] - cs[o + 1 - L]) / L < cfg.win.min_observed_fraction:
                continue
            wins.append(WindowIndex(seg=si, end=o + 1, t_unix=float(s.time_unix[o])))
            info.append({"start_utc": utc(e.start_unix), "peak_utc": utc(e.peak_unix),
                         "peak_flux": float(e.peak_rate), "peak_after_origin": bool(e.peak_idx > o)})
    return wins, info


# ---- scores ----------------------------------------------------------------------

def heads(p: dict, cfg: Config) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per classification head: (mask, truth, probability) over a prediction set."""
    out = {"in_flare": (p["y_nowcast_mask"] > 0, p["y_in_flare"], p["p_inflare"])}
    for h, hs in enumerate(cfg.win.occurrence_horizons_s):
        out[f"occ_{int(hs / 60)}min"] = (p["y_occurrence_mask"][:, h] > 0, p["y_occurrence"][:, h],
                                         p["p_occurrence"][:, h])
    return out


def regressions(p: dict, cfg: Config) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per flux target: (mask, absolute error), on origins SoLEXS observed."""
    from solarflare.metrics import forecast_score_mask

    soft = p["y_soft_origin"] > 0
    i50 = int(np.argmin(np.abs(np.array(cfg.win.quantiles) - 0.5)))
    out = {"nowcast_flux": ((p["y_nowcast_mask"] > 0) & soft, np.abs(p["nowcast"] - p["y_nowcast"]))}
    for h, hs in enumerate(cfg.win.forecast_horizons_s):
        m = forecast_score_mask(p["y_forecast_mask"][:, h], p["y_nowcast_mask"]) & soft
        out[f"flux_{int(hs / 60)}min"] = (m, np.abs(p["forecast"][:, h, i50] - p["y_forecast"][:, h]))
    out["peak_flux"] = (p["y_peak_mask"][:, 1] > 0, np.abs(p["peak"][:, 1] - p["y_peak"][:, 1]))
    out["peak_time_min"] = (p["y_peak_mask"][:, 0] > 0,
                            np.abs(p["peak"][:, 0] - p["y_peak"][:, 0]) * PEAK_TIME_SCALE_S / 60.0)
    return out


def r4(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def paired_ci(t: np.ndarray, stat, n_boot: int = 300) -> list[float] | None:
    return day_block_ci(t, stat, n_boot=n_boot)


# ---- 1. baseline vs final ----------------------------------------------------------

def paired_head(y, pf, pb, cf, cb, thf: float, thb: float, t) -> dict:
    """One classification head, both models on the same windows: raw scores at
    each model's own threshold, calibrated Brier skill, paired differences."""
    row = {}
    for name, pp, cc, th in (("final", pf, cf, thf), ("baseline", pb, cb, thb)):
        s = skill_scores(y, pp >= th)
        row[name] = {"AUC": r4(roc_auc(y, pp)), "TSS": r4(s["TSS"]), "HSS": r4(s["HSS"]),
                     "POD": r4(s["POD"]), "FAR": r4(s["FAR"]), "FB": r4(s["FB"]), "threshold": th,
                     "BSS_raw": r4(brier_skill_score(y, pp)), "BSS_calibrated": r4(brier_skill_score(y, cc))}
    row["n"], row["events"] = int(y.size), int(y.sum())
    row["diff"] = {
        "AUC": r4(roc_auc(y, pf) - roc_auc(y, pb)),
        "AUC_ci95": paired_ci(t, lambda i: roc_auc(y[i], pf[i]) - roc_auc(y[i], pb[i]), 200),
        "TSS": r4(row["final"]["TSS"] - row["baseline"]["TSS"]),
        "TSS_ci95": paired_ci(t, lambda i: skill_scores(y[i], pf[i] >= thf)["TSS"]
                              - skill_scores(y[i], pb[i] >= thb)["TSS"]),
        "BSS_calibrated": r4(row["final"]["BSS_calibrated"] - row["baseline"]["BSS_calibrated"]),
        "BSS_calibrated_ci95": paired_ci(t, lambda i: brier_skill_score(y[i], cf[i])
                                         - brier_skill_score(y[i], cb[i])),
    }
    return row


def paired_error(ef: np.ndarray, eb: np.ndarray, t: np.ndarray) -> dict:
    """Absolute errors of both models on the same windows."""
    if not ef.size:
        return {"n": 0}
    return {"n": int(ef.size), "final_MAE": r4(ef.mean()), "baseline_MAE": r4(eb.mean()),
            "diff": r4(ef.mean() - eb.mean()),
            "diff_ci95": paired_ci(t, lambda i: float(ef[i].mean() - eb[i].mean())),
            "final_better_share": r4(float(np.mean(ef < eb)))}


def compare(runs: dict, preds: dict, cfg: Config) -> dict:
    """Paired scores of the two runs on the common windows (``preds[name]['common']``)."""
    fin, base = preds["final"]["common"], preds["baseline"]["common"]
    t = fin["y_t_unix"]
    cal = {}
    for name in ("final", "baseline"):
        v = preds[name]["val"]
        cal[name] = {k: probcal.fit(pp[m], y[m]) for k, (m, y, pp) in heads(v, cfg).items()}
    subsets = {"all": np.ones(t.size, bool), "hel1os_observed": fin["y_hard_origin"] > 0}
    hf, hb = heads(fin, cfg), heads(base, cfg)
    rf, rb = regressions(fin, cfg), regressions(base, cfg)
    out = {}
    for sub, keep in subsets.items():
        res = {"windows": int(keep.sum()), "days": int(np.unique(np.floor(t[keep] / 86400)).size),
               "classification": {}, "regression": {}}
        for k, (m, y, pf) in hf.items():
            m = m & keep
            pb = hb[k][2][m]
            pf, y = pf[m], y[m]
            res["classification"][k] = paired_head(
                y, pf, pb, probcal.apply(cal["final"][k], pf), probcal.apply(cal["baseline"][k], pb),
                runs["final"]["thr"][k], runs["baseline"]["thr"][k], t[m])
        for k, (m, ef) in rf.items():
            m = m & keep
            res["regression"][k] = paired_error(ef[m], rb[k][1][m], t[m])
        out[sub] = res
    return out


def render_compare(r: dict) -> str:
    L = ["# Baseline (SoLEXS only) vs final (SoLEXS + HEL1OS), same windows", "",
         f"Period {r['period'][0]} -> {r['period'][1]} UTC (both models held it out); origins SoLEXS observed. "
         f"Baseline: epoch {r['baseline']['epoch']}, own normaliser, HEL1OS blanked as in its training. "
         f"Final: epoch {r['final']['epoch']}. Each uses the thresholds it chose on its own validation split; "
         "calibrated = isotonic fit on that split. Intervals: 95%, resampling whole days. "
         "A difference is called real only when its interval excludes zero.", ""]
    for sub, title in (("all", "All common windows"), ("hel1os_observed", "Windows HEL1OS also observed")):
        s = r["results"][sub]
        L += [f"## {title} ({s['windows']:,} windows, {s['days']} days)", "",
              "| Head | AUC base -> final | TSS base -> final (FB) | calibrated BSS base -> final | "
              "AUC diff [95%] | TSS diff [95%] | BSS diff [95%] |", "|---|---|---|---|---|---|---|"]
        for k, v in s["classification"].items():
            b, f, d = v["baseline"], v["final"], v["diff"]
            L.append(f"| {k} | {_n(b['AUC'], '.3f')} -> {_n(f['AUC'], '.3f')} | {_n(b['TSS'], '.3f')} ({_n(b['FB'], '.2f')}) -> "
                     f"{_n(f['TSS'], '.3f')} ({_n(f['FB'], '.2f')}) | {_n(b['BSS_calibrated'], '.3f')} -> {_n(f['BSS_calibrated'], '.3f')} | "
                     f"{_n(d['AUC'], '+.3f')} {_ci(d['AUC_ci95'])} | {_n(d['TSS'], '+.3f')} {_ci(d['TSS_ci95'])} | "
                     f"{_n(d['BSS_calibrated'], '+.3f')} {_ci(d['BSS_calibrated_ci95'])} |")
        L += ["", "| Flux target | n | baseline MAE | final MAE | diff [95%] | final better on |",
              "|---|---:|---:|---:|---|---:|"]
        for k, v in s["regression"].items():
            if v["n"]:
                unit = " min" if k.endswith("_min") else " dex"
                L.append(f"| {k} | {v['n']:,} | {_n(v['baseline_MAE'], '.3f')}{unit} | {_n(v['final_MAE'], '.3f')}{unit} | "
                         f"{_n(v['diff'], '+.4f')} {_ci(v['diff_ci95'])} | {_n(100 * v['final_better_share'], '.0f')}% |")
        L.append("")
    L += ["## Verdict", "", *[f"- {x}" for x in r["verdict"]], "",
          "## Reading it", "",
          "- Two separate trainings: besides HEL1OS they differ in training span (the final model's ended "
          f"{r['final']['train_end'][:10]}, the baseline's {r['baseline']['train_end'][:10]}), initialisation "
          "and stopping epoch. This compares the deployed model with the one it replaces; the single-variable "
          "test of HEL1OS is the paired fusion ablation over three seeds (outputs/ablations/hel1os).",
          "- HEL1OS observed almost every common window, so the two sections are nearly the same sample.", ""]
    return "\n".join(L)


def compare_verdict(res: dict) -> list[str]:
    out = []
    for sub, label in (("all", "all common windows"), ("hel1os_observed", "where HEL1OS observed")):
        s = res[sub]
        better, worse = [], []
        for k, v in s["classification"].items():
            for m in ("AUC", "TSS", "BSS_calibrated"):
                ci = v["diff"].get(f"{m}_ci95")
                if ci and ci[0] > 0:
                    better.append(f"{k} {m}")
                elif ci and ci[1] < 0:
                    worse.append(f"{k} {m}")
        for k, v in s["regression"].items():
            ci = v.get("diff_ci95")
            if ci and ci[1] < 0:
                better.append(f"{k} MAE")
            elif ci and ci[0] > 0:
                worse.append(f"{k} MAE")
        out.append(f"{label}: final better on {', '.join(better) or 'nothing'}; "
                   f"worse on {', '.join(worse) or 'nothing'} (every other difference's interval includes 0)")
    return out


# ---- 2. peak flux 3 min after onset ------------------------------------------------

def peak_scores(y: np.ndarray, p: np.ndarray, t: np.ndarray) -> dict:
    e = p - y
    return {"n": int(y.size), "RMSE": r4(np.sqrt(np.mean(e ** 2))) if y.size else None,
            "RMSE_ci95": day_block_ci(t, lambda i: float(np.sqrt(np.mean(e[i] ** 2)))) if y.size > 20 else None,
            "MAE": r4(np.mean(np.abs(e))) if y.size else None, "bias": r4(np.mean(e)) if y.size else None}


def goes_class(log_flux: float | None) -> str:
    """log10 W/m^2 -> GOES class, e.g. -5.3 -> 'C5.0'."""
    if log_flux is None or not np.isfinite(log_flux):
        return "n/a"
    f = 10.0 ** log_flux
    for letter, floor in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if f >= floor:
            return f"{letter}{f / floor:.1f}"
    return f"A{f / 1e-8:.1f}"


def m_cut(y: np.ndarray, p: np.ndarray) -> float:
    """The predicted log peak above which a flare is called >= M: the value that
    maximises TSS on validation flares, as every threshold here is chosen (the
    M1 floor itself when validation has only one class)."""
    ok = np.isfinite(p) & np.isfinite(y)
    t, q = y[ok] >= CLASS_FLOOR["M"], p[ok]
    if t.all() or not t.any():
        return CLASS_FLOOR["M"]
    cands = np.unique(np.concatenate([np.quantile(q, np.linspace(0.01, 0.99, 197)), [CLASS_FLOOR["M"]]]))
    return float(cands[int(np.argmax([skill_scores(t, q >= c)["TSS"] for c in cands]))])


def m_class_scores(y: np.ndarray, p: np.ndarray, cut: float = CLASS_FLOOR["M"]) -> dict:
    s = skill_scores(y >= CLASS_FLOOR["M"], p >= cut)
    return {k: r4(s[k]) for k in ("TSS", "F1", "HSS", "POD", "FAR", "FB")} | {
        "M_flares": int(np.sum(y >= CLASS_FLOOR["M"])), "C_flares": int(np.sum(y < CLASS_FLOOR["M"]))}


def rmse_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> dict:
    """RMSE(a) - RMSE(b) on the same flares, 95% interval resampling days."""
    ea, eb = (a - y) ** 2, (b - y) ** 2
    if y.size < 3:
        return {"n": int(y.size), "diff": None, "ci95": None}
    return {"n": int(y.size), "diff": r4(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
            "ci95": day_block_ci(t, lambda i: float(np.sqrt(ea[i].mean()) - np.sqrt(eb[i].mean())))}


def peak_nowcast(pred: dict, info: list[dict], refs: dict[str, np.ndarray], against: str,
                 cuts: dict[str, float] | None = None) -> dict:
    """Peak-flux scores per true class for the network and each reference, the
    network's paired RMSE difference against reference ``against``, and the
    C vs >= M call at the M1 floor and at each forecast's ``cuts`` (validation)."""
    y = pred["y_peak"][:, 1]
    ok = pred["y_peak_mask"][:, 1] > 0
    t = pred["y_t_unix"]
    ahead = np.array([d["peak_after_origin"] for d in info])
    forecasts = {"model": pred["peak"][:, 1], **refs}
    groups = {">=C": y >= CLASS_FLOOR["C"], ">=M": y >= CLASS_FLOOR["M"], "X": y >= CLASS_FLOOR["X"],
              "C": (y >= CLASS_FLOOR["C"]) & (y < CLASS_FLOOR["M"]),
              "M": (y >= CLASS_FLOOR["M"]) & (y < CLASS_FLOOR["X"])}
    out = {"n_flares": int(ok.sum()), "peak_still_ahead": int((ok & ahead).sum()), "by_class": {},
           "by_class_peak_ahead": {}, "C_vs_M": {}}
    for sel_name, sel in (("by_class", ok), ("by_class_peak_ahead", ok & ahead)):
        for g, gm in groups.items():
            m = sel & gm & np.all([np.isfinite(f) for f in forecasts.values()], axis=0)
            out[sel_name][g] = {k: peak_scores(y[m], f[m], t[m]) for k, f in forecasts.items()}
    m = ok & np.all([np.isfinite(f) for f in forecasts.values()], axis=0)
    out["C_vs_M"] = {k: m_class_scores(y[m], f[m]) for k, f in forecasts.items()}
    if cuts:
        out["C_vs_M_validation_cut"] = {k: {**m_class_scores(y[m], f[m], cuts[k]), "cut": r4(cuts[k])}
                                        for k, f in forecasts.items() if k in cuts}
        if "model" in cuts and against in cuts:
            truth = y[m] >= CLASS_FLOOR["M"]
            a, b, tm_ = forecasts["model"][m] >= cuts["model"], forecasts[against][m] >= cuts[against], t[m]
            out["C_vs_M_validation_cut_paired"] = {
                "reference": against,
                "TSS_diff": r4(skill_scores(truth, a)["TSS"] - skill_scores(truth, b)["TSS"]),
                "ci95": day_block_ci(tm_, lambda i: skill_scores(truth[i], a[i])["TSS"]
                                     - skill_scores(truth[i], b[i])["TSS"])}
    out["paired_vs"] = {"reference": against, **{
        g: rmse_diff(y[m & gm], forecasts["model"][m & gm], forecasts[against][m & gm], t[m & gm])
        for g, gm in groups.items()}}
    tt = np.abs(pred["peak"][:, 0] - pred["y_peak"][:, 0]) * PEAK_TIME_SCALE_S / 60.0
    out["time_to_peak_MAE_min"] = r4(tt[ok & ahead].mean()) if (ok & ahead).any() else None
    return out


def render_peak(r: dict) -> str:
    names = {"model": "Network (final)", "goes_now": "GOES flux at +3 min (no change)",
             "goes_now_plus_rise": "GOES at +3 min + typical rise", "solexs_now": "SoLEXS at +3 min (calibrated)",
             "solexs_now_plus_rise": "SoLEXS at +3 min + typical rise", "baseline": "Network (SoLEXS-only baseline)"}
    s = r["final_test_period"]
    L = ["# Peak flux of the ongoing flare, predicted 3 min after GOES start", "",
         f"One prediction per GOES flare >= C1 starting in {r['period'][0]} -> {r['period'][1]} UTC, from the "
         "window ending 3 min after the flare's GOES start, where SoLEXS observed it. Truth: GOES XRS-B peak, "
         "log10 W/m^2. 'Typical rise' = median log(peak / flux at +3 min) over training-period flares "
         f"({_n(r['typical_rise_dex'], '+.3f')} dex). RMSE in dex, 95% intervals resampling days.", "",
         f"**{s['n_flares']:,} flares** ({s['peak_still_ahead']:,} still rising at +3 min). "
         f"Time-to-peak MAE at +3 min: {_n(s['time_to_peak_MAE_min'], '.1f')} min.", "",
         "| Class (truth) | n | " + " | ".join(names[k] for k in s["by_class"][">=C"])
         + " | paper at +3 min | paper, as reported |",
         "|---|---:|" + "---|" * (len(s["by_class"][">=C"]) + 2)]
    for g, row in s["by_class"].items():
        n = next(iter(row.values()))["n"]
        at3, rep = PAPER["RMSE_dex_at_3min"].get(g, "-"), PAPER["RMSE_dex_as_reported"].get(g, "-")
        L.append(f"| {g} | {n:,} | " + " | ".join(
            f"{_n(v['RMSE'], '.3f')} {_ci(v['RMSE_ci95'])}" if v["RMSE"] is not None else "-" for v in row.values())
                 + f" | {at3} | {rep} |")
    pv = s["paired_vs"]
    L += ["", f"Network minus {names[pv['reference']]}, same flares (negative = network better):", "",
          "| Class | n | RMSE difference [95%] |", "|---|---:|---|"]
    L += [f"| {g} | {v['n']:,} | {_n(v['diff'], '+.3f')} {_ci(v['ci95'])} |"
          for g, v in pv.items() if g != "reference" and v["n"]]
    L += ["", "C vs >= M from the predicted peak (>= 1e-5 W/m^2 called M):", "",
          "| Forecast | TSS | F1 | HSS | POD | FAR | FB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in s["C_vs_M"].items():
        L.append(f"| {names[k]} | {_n(v['TSS'], '.3f')} | {_n(v['F1'], '.3f')} | {_n(v['HSS'], '.3f')} | {_n(v['POD'], '.3f')} | "
                 f"{_n(v['FAR'], '.3f')} | {_n(v['FB'], '.2f')} |")
    L.append(f"| paper (arXiv 2608.20062, as reported) | {PAPER['C_vs_M']['TSS']} | {PAPER['C_vs_M']['F1']} "
             "| | | | |")
    if s.get("C_vs_M_validation_cut"):
        L += ["", f"Same call with each forecast's cut-off chosen on the {r['validation_flares']:,} validation-period "
              f"flares (best TSS; {r['validation_period'][0]} -> {r['validation_period'][1]}), then held fixed:", "",
              "| Forecast | cut-off (GOES class) | TSS | F1 | HSS | POD | FAR | FB |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in s["C_vs_M_validation_cut"].items():
            L.append(f"| {names[k]} | {goes_class(v['cut'])} | {_n(v['TSS'], '.3f')} | {_n(v['F1'], '.3f')} | "
                     f"{_n(v['HSS'], '.3f')} | {_n(v['POD'], '.3f')} | {_n(v['FAR'], '.3f')} | {_n(v['FB'], '.2f')} |")
        pd_ = s.get("C_vs_M_validation_cut_paired")
        if pd_:
            L += ["", f"Network minus {names[pd_['reference']]}, same flares: TSS {_n(pd_['TSS_diff'], '+.3f')} "
                  f"{_ci(pd_['ci95'])} (95%, resampling days)."]
    if r.get("common_period"):
        c = r["common_period"]
        L += ["", f"## Baseline vs final, same flares ({c['period'][0]} -> {c['period'][1]})", "",
              "| Class | n | baseline RMSE | final RMSE | final minus baseline [95%] |",
              "|---|---:|---:|---:|---|"]
        for g, row in c["by_class"].items():
            if row["model"]["n"]:
                d = c["paired_vs"][g]
                L.append(f"| {g} | {row['model']['n']:,} | {_n(row['baseline']['RMSE'], '.3f')} | "
                         f"{_n(row['model']['RMSE'], '.3f')} | {_n(d['diff'], '+.3f')} {_ci(d['ci95'])} |")
    L += ["", "## Reading this against the paper", "", *[f"- {x}" for x in r["caveats"]], ""]
    return "\n".join(L)


# ---- 3. frequency bias -------------------------------------------------------------

def fb_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The threshold at which as many windows are forecast positive as are positive."""
    rate = float(np.mean(y))
    return float(np.quantile(p, 1.0 - rate)) if 0 < rate < 1 else 0.5


def forecast_bias(val: dict, test: dict, cfg: Config, base_rates: dict) -> dict:
    out = {}
    hv, ht = heads(val, cfg), heads(test, cfg)
    for k in hv:
        mv, yv, pv = hv[k]
        mt, yt, pt = ht[k]
        yv, pv, yt, pt = yv[mv], pv[mv], yt[mt], pt[mt]
        cal = probcal.fit(pv, yv)
        ct = probcal.apply(cal, pt)
        points = {"best TSS (as deployed)": best_threshold(yv, pv, "TSS")[0],
                  "best HSS": best_threshold(yv, pv, "HSS")[0],
                  "FB = 1 on validation": fb_threshold(yv, pv)}
        row = {"base_rate": {**base_rates[k], "val": r4(yv.mean()), "test": r4(yt.mean())},
               "mean_probability_test": {"raw": r4(pt.mean()), "calibrated": r4(ct.mean())},
               "BSS_test": {"raw": r4(brier_skill_score(yt, pt)), "calibrated": r4(brier_skill_score(yt, ct))},
               "operating_points": {}}
        for name, th in points.items():
            s = skill_scores(yt, pt >= th)
            row["operating_points"][name] = {"threshold": r4(th), **{m: r4(s[m]) for m in
                                                                    ("TSS", "HSS", "POD", "FAR", "FB")}}
        s = skill_scores(yt, ct >= 0.5)
        row["operating_points"]["calibrated p >= 0.5"] = {"threshold": 0.5, **{m: r4(s[m]) for m in
                                                                             ("TSS", "HSS", "POD", "FAR", "FB")}}
        out[k] = row
    return out


def render_bias(r: dict) -> str:
    L = ["# Frequency bias: over-forecasting or the threshold?", "",
         "FB = windows forecast positive / windows positive (1 unbiased, > 1 over-forecasting; Leka et al. 2019). "
         "Operating points chosen on validation, scored on the final model's test split "
         f"({r['period'][0]} -> {r['period'][1]}).", ""]
    for k, v in r["heads"].items():
        br = v["base_rate"]
        L += [f"## {k}", "", f"Base rate: train {_n(br['train'], '.3f')}, validation {_n(br['val'], '.3f')}, test {_n(br['test'], '.3f')}. "
              f"Mean forecast probability on test: raw {_n(v['mean_probability_test']['raw'], '.3f')}, calibrated "
              f"{_n(v['mean_probability_test']['calibrated'], '.3f')}. BSS raw {_n(v['BSS_test']['raw'], '+.3f')}, "
              f"calibrated {_n(v['BSS_test']['calibrated'], '+.3f')}.", "",
              "| Operating point | threshold | TSS | HSS | POD | FAR | FB |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, s in v["operating_points"].items():
            L.append(f"| {name} | {_n(s['threshold'], '.3f')} | {_n(s['TSS'], '.3f')} | {_n(s['HSS'], '.3f')} | {_n(s['POD'], '.3f')} | "
                     f"{_n(s['FAR'], '.3f')} | {_n(s['FB'], '.2f')} |")
        L.append("")
    L += ["## Verdict", "", *[f"- {x}" for x in r["verdict"]], ""]
    return "\n".join(L)


def bias_verdict(heads_: dict) -> list[str]:
    out = []
    for k, v in heads_.items():
        op = v["operating_points"]
        a, b = op["best TSS (as deployed)"], op["FB = 1 on validation"]
        mp, br = v["mean_probability_test"]["calibrated"], v["base_rate"]["test"]
        if a["FB"] > 1.2:
            why = ("over-forecasting, set by the TSS-optimal threshold (TSS rewards it at low base rates), "
                   "not by the model: the FB = 1 threshold removes it at a cost in TSS and a gain in HSS.")
        elif a["FB"] < 0.8:
            why = ("under-forecasting: the flare rate kept falling (validation > test), so the threshold "
                   "chosen on validation is conservative on test.")
        else:
            why = "close to unbiased."
        out.append(f"{k}: as deployed FB {_n(a['FB'], '.2f')} at TSS {_n(a['TSS'], '.3f')} / HSS "
                   f"{_n(a['HSS'], '.3f')}; at the FB = 1 threshold FB {_n(b['FB'], '.2f')}, TSS "
                   f"{_n(b['TSS'], '.3f')}, HSS {_n(b['HSS'], '.3f')}: {why} Calibrated mean probability "
                   f"{_n(mp)} against a test rate of {_n(br)}.")
    return out


# ---- helpers -----------------------------------------------------------------------

def _n(x, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "[n/a]"


def _write(folder: Path, name: str, data: dict, md: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), "utf-8")
    (folder / "REPORT.md").write_text(md, "utf-8")
    print(f"  -> {folder / 'REPORT.md'}", flush=True)


def split_base_rates(prep, targets: list, cfg: Config) -> dict[str, dict]:
    """Training-split base rate of every head, from the targets alone."""
    idx = prep.splits["train"]
    seg = np.array([prep.windows[i].seg for i in idx])
    j = np.array([prep.windows[i].end - 1 for i in idx])
    out = {}
    for name, key, h in [("in_flare", "in_flare", None)] + [
            (f"occ_{int(hs / 60)}min", "occurrence", n) for n, hs in enumerate(cfg.win.occurrence_horizons_s)]:
        vals = []
        for s in np.unique(seg):
            tg = targets[s]
            jj = j[seg == s]
            mk = "nowcast_mask" if h is None else "occurrence_mask"
            m = (tg[mk][jj] if h is None else tg[mk][jj, h]) > 0
            y = tg[key][jj] if h is None else tg[key][jj, h]
            vals.append(y[m])
        out[name] = {"train": r4(np.concatenate(vals).mean())}
    return out


def typical_rise(prep, cfg: Config, t_end: float) -> float:
    """Median log10(peak / GOES flux 3 min after start) over flares before ``t_end``."""
    k = int(round(ONSET_DELAY_S / cfg.pre.dt_seconds))
    d = []
    for s in prep.segments:
        for e in s.events:
            o = e.start_idx + k
            if o < e.end_idx and o < len(s) and s.time_unix[o] <= t_end and s.target_valid[o] > 0:
                d.append(np.log10(max(e.peak_rate, 1e-9)) - float(s.log_flux[o]))
    return float(np.median(d)) if d else 0.0


# ---- index of everything in outputs/tests --------------------------------------------

def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def write_index(folder: Path) -> Path:
    """outputs/tests/README.md: one paragraph per test that has results, each
    linking to its full report. Called by every test command."""
    L = ["# Tests", "", "Every result here is on data the models never trained on, scored with the "
         "project's own rules (thresholds from validation, intervals resampling whole days or weeks). "
         "Each folder holds the full report (REPORT.md or summary.md) and its JSON.", ""]
    s = _read(folder / "suites" / "summary.json")
    if s:
        lint = ", ".join(f"{r['name']} {'clean' if r['ok'] else 'PROBLEMS'}" for r in s["lint"])
        L += ["## Code test suites -- [suites/summary.md](suites/summary.md)", "",
              f"{'All passed' if s['all_passed'] else 'FAILURES'}: {s['checks_passed']} checks passed, "
              f"{s['checks_failed']} failed across {len(s['suites'])} suites ({s['started_utc']} UTC); "
              f"lint: {lint or 'not run'}.", ""]
    c = _read(folder / "baseline_vs_final" / "baseline_vs_final.json")
    if c:
        a = c["results"]["all"]
        h = c["results"]["hel1os_observed"]
        ci, pk = a["classification"]["in_flare"], a["regression"]["peak_flux"]
        hi = h["classification"]["in_flare"]
        L += ["## Baseline (SoLEXS only) vs final (SoLEXS + HEL1OS) -- "
              "[baseline_vs_final/REPORT.md](baseline_vs_final/REPORT.md)", "",
              f"Same {a['windows']:,} windows ({c['period'][0]} -> {c['period'][1]}). In-flare TSS "
              f"{_n(ci['baseline']['TSS'])} -> {_n(ci['final']['TSS'])} (diff {_n(ci['diff']['TSS'], '+.3f')} "
              f"{_ci(ci['diff']['TSS_ci95'])}); where HEL1OS observed {_n(hi['baseline']['TSS'])} -> "
              f"{_n(hi['final']['TSS'])} ({_n(hi['diff']['TSS'], '+.3f')} {_ci(hi['diff']['TSS_ci95'])}). "
              f"Peak-flux MAE {_n(pk.get('baseline_MAE'))} -> {_n(pk.get('final_MAE'))} dex "
              f"({_n(pk.get('diff'), '+.4f')} {_ci(pk.get('diff_ci95'))}).", "",
              *[f"- {v}" for v in c["verdict"]], ""]
    p = _read(folder / "peak_nowcast" / "peak_nowcast.json")
    if p:
        f = p["final_test_period"]
        row = f["by_class"]
        L += ["## Peak flux 3 min after onset vs arXiv 2608.20062 -- [peak_nowcast/REPORT.md](peak_nowcast/REPORT.md)",
              "", f"{f['n_flares']:,} GOES flares. RMSE (dex) network / GOES-now-plus-typical-rise / paper: "
              + "; ".join(f"{g} {_n(row[g]['model']['RMSE'])} / {_n(row[g]['goes_now_plus_rise']['RMSE'])} / "
                          f"{PAPER['RMSE_dex_at_3min'].get(g, PAPER['RMSE_dex_as_reported'].get(g, '-'))}"
                          for g in (">=C", ">=M", "X") if row[g]["model"]["n"])
              + " (paper: >= M at +3 min; the others as reported, pooled over prediction times). C vs >= M: "
                f"TSS {_n(f['C_vs_M']['model']['TSS'])}, F1 {_n(f['C_vs_M']['model']['F1'])} (paper as reported "
                f"{PAPER['C_vs_M']['TSS']}, {PAPER['C_vs_M']['F1']}). Different data and period: see the "
                "report's caveats."]
        vc = f.get("C_vs_M_validation_cut")
        if vc:
            m_, g_ = vc["model"], vc["goes_now_plus_rise"]
            pd_ = f.get("C_vs_M_validation_cut_paired") or {}
            L.append(f"- C vs >= M with the cut-off chosen on {p['validation_flares']:,} validation flares: network "
                     f"TSS {_n(m_['TSS'])}, F1 {_n(m_['F1'])}, FB {_n(m_['FB'], '.2f')} (calls M above "
                     f"{goes_class(m_['cut'])}); 'GOES now + typical rise' TSS {_n(g_['TSS'])}, F1 {_n(g_['F1'])}; "
                     f"difference {_n(pd_.get('TSS_diff'), '+.3f')} {_ci(pd_.get('ci95'))}.")
        pv = f["paired_vs"]
        L.append("- Network minus 'GOES at +3 min + typical rise', same flares: "
                 + "; ".join(f"{g} {_n(pv[g]['diff'], '+.3f')} {_ci(pv[g]['ci95'])}"
                             for g in (">=C", ">=M", "C", "M") if pv[g]["n"]) + ".")
        cp = p.get("common_period")
        if cp:
            d = cp["paired_vs"]
            L.append("- Final minus SoLEXS-only baseline at +3 min, same flares: "
                     + "; ".join(f"{g} {_n(d[g]['diff'], '+.3f')} {_ci(d[g]['ci95'])}"
                                 for g in (">=C", ">=M", "C", "M") if d[g]["n"]) + ".")
        L.append("")
    b = _read(folder / "forecast_bias" / "forecast_bias.json")
    if b:
        L += ["## Frequency bias -- [forecast_bias/REPORT.md](forecast_bias/REPORT.md)", "",
              *[f"- {v}" for v in b["verdict"]], ""]
    d = _read(folder / "blind_dayahead" / "blind_dayahead.json")
    if d:
        L += ["## Blind day-ahead replay -- [blind_dayahead/REPORT.md](blind_dayahead/REPORT.md)", "",
              f"{d['days_forecast']} days ({d['period'][0]} -> {d['period'][1]}), models frozen before the test."]
        for k, v in d["classes"].items():
            L.append(f"- >= {k}1 day ({v['flare_days']}/{v['days']} days): AUC {v['AUC']:.3f} (persistence "
                     f"{v['AUC_persistence']:.3f}); Brier skill (recalibrated) vs persistence "
                     f"{v['BSS_vs_persistence_recalibrated']:+.3f} {_ci(v['BSS_vs_persistence_recalibrated_ci'])}, "
                     f"vs the test period's own rate {v['BSS_vs_test_rate_recalibrated']:+.3f} "
                     f"{_ci(v['BSS_vs_test_rate_recalibrated_ci'])}; accuracy {100 * v['accuracy']:.0f}% "
                     f"(always-the-common-answer {100 * v['accuracy_always_majority']:.0f}%).")
        L.append("")
    sealed = sorted((folder / "live_dayahead" / "sealed").glob("forecast_*.json"))
    if sealed:
        L += ["## Sealed day-ahead forecasts -- live_dayahead/sealed/", "",
              "Issued by `python -m solarflare day-forecast` from SoLEXS days downloaded after the models were "
              "frozen, each with a SHA-256 of its content and never re-issued; scored once GOES covers the day.", "",
              "| Day | >= C1 | >= M1 | issued (UTC) | lead | sha256 | outcome (GOES) |",
              "|---|---:|---:|---|---:|---|---|"]
        scores = _read(folder / "live_dayahead" / "sealed" / "scores.json") or {}
        outcome = {r["day"]: r for r in scores.get("forecasts", [])}
        for f in sealed:
            s = _read(f) or {}
            day = f.stem.removeprefix("forecast_")
            o = outcome.get(day, {})
            res = (f"C1 {'yes' if o['C']['happened'] else 'no'}, M1 {'yes' if o['M']['happened'] else 'no'} "
                   f"(largest {o.get('largest_goes_flare') or 'none'})" if o.get("status") == "scored"
                   else o.get("status", "pending"))
            L.append(f"| {day} | {100 * s['C']['probability']:.0f}% | {100 * s['M']['probability']:.0f}% | "
                     f"{s['issued_utc']} | {s['lead_hours']:.0f} h | {s['sha256'][:16]} | {res} |")
        if scores.get("scored"):
            L += ["", f"Mean Brier score over {scores['scored']} scored days, forecast vs the last-30-day rate: "
                  f">= C1 {scores['mean_brier_C']:.3f} vs {scores['mean_reference_brier_C']:.3f}; "
                  f">= M1 {scores['mean_brier_M']:.3f} vs {scores['mean_reference_brier_M']:.3f} "
                  "(a record, not yet a measurement: see live_dayahead/sealed/SCORES.md)."]
        L.append("")
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "README.md"
    out.write_text("\n".join(L), "utf-8")
    return out


# ---- main --------------------------------------------------------------------------

def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--final", default=str(S.model_dir), help="the final run folder")
    ap.add_argument("--baseline", default=str(S.ablations / "baseline_solexs_only"),
                    help="the SoLEXS-only run folder")
    ap.add_argument("--only", default=",".join(TESTS), help=f"comma-separated subset of {', '.join(TESTS)}")
    ap.add_argument("--out", default=str(S.tests))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    todo = [x.strip() for x in args.only.split(",") if x.strip()]
    bad = set(todo) - set(TESTS)
    if bad:
        ap.error(f"unknown test(s) {sorted(bad)}; choose from {TESTS}")
    out = Path(args.out)
    t0 = time.time()

    from solarflare.evaluate import solexs_no_change
    from solarflare.pipeline import prepare, resolve_device

    device = resolve_device("auto")
    runs = {"final": load_run(Path(args.final), device, hel1os=True)}
    if "compare" in todo or "peak" in todo:
        runs["baseline"] = load_run(Path(args.baseline), device, hel1os=False)
    cfg = runs["final"]["cfg"]
    prep = prepare(cfg, verbose=False)
    data_end = float(max(s.time_unix[-1] for s in prep.segments))
    print(f"data ready: {len(prep.windows):,} windows, test {len(prep.splits['test']):,} "
          f"({time.time() - t0:.0f} s)", flush=True)
    targets = [build_targets(s, cfg) for s in prep.segments]

    fin_start = runs["final"]["test_start"]
    sets = {"final": {}, "baseline": {}}
    if "compare" in todo:
        b = runs["baseline"]
        emb = max(cfg.train.embargo_s, float(getattr(cfg.train, "global_embargo_s", 0.0)))
        common = soft_windows(prep, cfg, prep.splits["test"], b["test_start"])
        sets["final"].update(common=common, val=prep.splits["val"])
        sets["baseline"].update(common=common, val=soft_windows(prep, cfg, range(len(prep.windows)),
                                                                b["train_end"] + emb, b["test_start"] - emb))
    if "bias" in todo:
        sets["final"].update(val=prep.splits["val"], test=prep.splits["test"])
    windows = prep.windows
    onset_info = []
    val_period = None
    if "peak" in todo:
        extra, onset_info = onset_windows(prep, cfg, fin_start)
        tv = [prep.windows[i].t_unix for i in prep.splits["val"]]
        val_period = (min(tv), max(tv))
        extra_val, _ = onset_windows(prep, cfg, *val_period)
        windows = list(prep.windows) + extra + extra_val
        onset = list(range(len(prep.windows), len(prep.windows) + len(extra)))
        sets["final"]["onset"] = onset
        sets["final"]["onset_val"] = list(range(len(prep.windows) + len(extra), len(windows)))
        sets["baseline"]["onset"] = onset
    preds = {}
    for name in ("final", "baseline"):
        if name in runs and sets[name]:
            print(f"predicting with the {name} model on {', '.join(f'{k} {len(v):,}' for k, v in sets[name].items())}",
                  flush=True)
            preds[name] = predict(runs[name], prep, targets, windows, sets[name], device)
    period = [utc(fin_start, "%Y-%m-%d"), utc(data_end, "%Y-%m-%d")]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    if "compare" in todo:
        b = runs["baseline"]
        res = compare(runs, preds, cfg)
        r = {"generated_utc": stamp, "period": [utc(b["test_start"], "%Y-%m-%d"), period[1]],
             "final": {"run": str(runs["final"]["run"]), "epoch": runs["final"]["epoch"],
                       "train_end": utc(runs["final"]["train_end"])},
             "baseline": {"run": str(b["run"]), "epoch": b["epoch"], "train_end": utc(b["train_end"]),
                          "validation_windows": len(sets["baseline"]["val"])},
             "results": res, "verdict": compare_verdict(res)}
        _write(out / "baseline_vs_final", "baseline_vs_final", r, render_compare(r))

    if "peak" in todo:
        p = preds["final"]["onset"]
        rise = typical_rise(prep, cfg, runs["final"]["train_end"])

        def references(q: dict) -> dict[str, np.ndarray]:
            now = q["y_persistence"].astype(np.float64)
            now[q["y_nowcast_mask"] <= 0] = np.nan
            slx, _ = solexs_no_change(prep, q["y_t_unix"])
            return {"goes_now": now, "goes_now_plus_rise": now + rise, "solexs_now": slx,
                    "solexs_now_plus_rise": slx + rise}

        refs = references(p)
        goes_now = refs["goes_now"]
        pv = preds["final"]["onset_val"]
        ok_v = pv["y_peak_mask"][:, 1] > 0
        yv = pv["y_peak"][ok_v, 1]
        cuts = {k: m_cut(yv, f[ok_v]) for k, f in {"model": pv["peak"][:, 1], **references(pv)}.items()}
        r = {"generated_utc": stamp, "period": period, "typical_rise_dex": round(rise, 4), "paper": PAPER,
             "validation_period": [utc(val_period[0], "%Y-%m-%d"), utc(val_period[1], "%Y-%m-%d")],
             "validation_flares": int(ok_v.sum()), "validation_M_flares": int(np.sum(yv >= CLASS_FLOOR["M"])),
             "M_cut_offs": {k: r4(v) for k, v in cuts.items()},
             "final_test_period": peak_nowcast(p, onset_info, refs, "goes_now_plus_rise", cuts)}
        pb = preds["baseline"]["onset"]
        in_common = p["y_t_unix"] >= runs["baseline"]["test_start"]
        if in_common.any():
            pc = {k: v[in_common] for k, v in p.items()}
            info_c = [d for d, keep in zip(onset_info, in_common) if keep]
            refs_c = {"baseline": pb["peak"][in_common, 1]}
            r["common_period"] = {"period": [utc(runs["baseline"]["test_start"], "%Y-%m-%d"), period[1]],
                                  **peak_nowcast(pc, info_c, refs_c, "baseline")}
        r["caveats"] = [
            "Like for like: the paper's figure for >= M flares predicted exactly 3 min after onset is "
            f"{PAPER['RMSE_dex_at_3min']['>=M']} dex. Its 0.26 / 0.45 / 0.87 rows and its C vs >= M TSS/F1 "
            "appear to pool prediction times from +3 min on (easier, the flare is further along); check the "
            "paper before quoting those head to head with numbers made at +3 min only.",
            "Different data: the paper trains and tests on GOES itself over 1997-2024 (all cycle phases); this "
            "network reads SoLEXS + HEL1OS and is scored against GOES near the Cycle 25 maximum, a single "
            "chronological test block of about six months.",
            "Different truth handling: here the flare start is GOES's own start time; an Aditya-only system "
            "would know the start from its own detector, a minute or two later or earlier.",
            "The paper gives no simple reference; 'GOES at +3 min + typical rise' is the fair floor any "
            "peak nowcaster has to beat.",
            "Classes with few flares (X) have wide intervals; read them as indicative.",
        ]
        rows = [{**d, "true_log_peak": r4(y), "predicted_log_peak": r4(m), "goes_at_origin": r4(g)}
                for d, y, m, g in zip(onset_info, p["y_peak"][:, 1], p["peak"][:, 1], goes_now)]
        r["flares"] = rows
        _write(out / "peak_nowcast", "peak_nowcast", r, render_peak(r))

    if "bias" in todo:
        f = preds["final"]
        hb = forecast_bias(f["val"], f["test"], cfg, split_base_rates(prep, targets, cfg))
        r = {"generated_utc": stamp, "period": period, "heads": hb, "verdict": bias_verdict(hb)}
        _write(out / "forecast_bias", "forecast_bias", r, render_bias(r))

    print(f"index -> {write_index(out)}; done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
