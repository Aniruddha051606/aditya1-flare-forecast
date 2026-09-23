"""The E1-E6 day-ahead matrix: what does SUIT add to SoLEXS and HEL1OS?

    python -m suit matrix

    E1 SoLEXS   E2 HEL1OS   E3 SUIT   E4 SoLEXS+HEL1OS   E5 SoLEXS+SUIT   E6 all three

Everything but the feature blocks is the day-ahead system's own
(solarflare.products.dayahead), called unchanged: hourly origins, targets (a GOES
>= C1 / >= M1 flare peaking within 2, 6, 12, 24 h, with GOES covering >= 80% of
the window), the split (the X-ray model's train end and test start, a target
window never straddling a boundary), the two model families with family and
threshold chosen on validation, the persistence reference, and the 95% intervals
that resample whole weeks.

Paired: all six experiments are trained and scored on the same hours -- those
where all three instruments were observing -- so a difference between two of
them comes from their inputs alone. Hours without SUIT are left out of all six,
and counted. The comparison that decides whether SUIT earns a neural-network
branch, E6 - E4, and its target are fixed in config/suit.toml before any SUIT
data is seen; the rest are reported, not used to decide.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import SuitConfig, SuitPaths

EXPERIMENTS = {
    "E1": ("solexs",), "E2": ("hel1os",), "E3": ("suit",),
    "E4": ("solexs", "hel1os"), "E5": ("solexs", "suit"), "E6": ("solexs", "hel1os", "suit"),
}
NAMES = {"solexs": "SoLEXS", "hel1os": "HEL1OS", "suit": "SUIT"}
COMPARISONS = (
    ("E6", "E4", "SUIT added to SoLEXS + HEL1OS"),       # the decision gate
    ("E5", "E1", "SUIT added to SoLEXS"),
    ("E4", "E1", "HEL1OS added to SoLEXS"),
    ("E3", "E1", "SUIT alone vs SoLEXS alone"),
)
MIN_ROWS = 50            # the day-ahead system's own rule


@dataclass
class Block:
    """One instrument's hourly features at the matrix origins."""
    X: np.ndarray                # (n_origins, n_features)
    names: list[str]
    observed: np.ndarray         # bool per origin
    impute: bool = False         # NaNs allowed (training medians); else the row must be complete


def inputs(E: str) -> str:
    return " + ".join(NAMES[b] for b in EXPERIMENTS[E])


def impute(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """NaNs -> the median over ``rows`` (the training hours) only, so nothing about
    the validation or test period leaks into the filled values."""
    med = np.nanmedian(np.where(rows[:, None], X, np.nan), axis=0)
    return np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0.0))


def _distinct_events(peaks: np.ndarray, o_te: np.ndarray, H: float) -> int:
    """Flares whose peak falls inside some scored target window (o, o + H]."""
    if not peaks.size or not o_te.size:
        return 0
    j = np.searchsorted(o_te, peaks, side="left") - 1
    ok = j >= 0
    return int(((peaks[ok] - o_te[j[ok]]) <= 3600.0 * H).sum())


def evaluate(o: np.ndarray, blocks: dict[str, Block], targets: dict, split: tuple[float, float],
             event_peaks: dict[str, np.ndarray], cfg: SuitConfig, provenance: str) -> dict:
    """Score E1-E6 and their paired differences for every target ``(class, H) ->
    (y, GOES covered)``. ``provenance`` is written into every output ("synthetic"
    marks a pipeline check, never a scientific result)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from solarflare.metrics import skill_scores
    from solarflare.products import dayahead as D

    t1, t2 = split
    order = ("solexs", "hel1os", "suit")
    complete = {b: np.isfinite(blocks[b].X).all(1) if not blocks[b].impute else np.ones(o.size, bool) for b in order}
    common = np.ones(o.size, bool)
    for b in order:
        common &= blocks[b].observed & complete[b]
    weeks = np.floor(o / (7 * 86400.0)).astype(int)
    utc = lambda t: datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M")    # noqa: E731
    summary = {
        "provenance": provenance,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "split": {"train_end": utc(t1), "test_start": utc(t2)},
        "hours": {"origins": int(o.size), **{f"{b}_observed": int(blocks[b].observed.sum()) for b in order},
                  "all_three": int(common.sum())},
        "features": {b: len(blocks[b].names) for b in order},
        "experiments": {E: inputs(E) for E in EXPERIMENTS},
        "gate": {"target": f">={cfg.gate_class}1 within {cfg.gate_horizon_h} h", "comparison": "E6 - E4",
                 "rule": "AUC gain > 0 with its 95% week-block interval above zero, and Brier skill not lower"},
        "results": {},
    }
    for (cls, H), (y, covered) in targets.items():
        key = f">={cls}1 within {H} h"
        base = common & covered
        tr = base & (o + 3600.0 * H <= t1)
        va = base & (o > t1) & (o + 3600.0 * H <= t2)
        te = base & (o >= t2)
        if min(tr.sum(), va.sum(), te.sum()) < MIN_ROWS or len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
            summary["results"][key] = {"skipped": "fewer than 50 hours in a split, or one outcome only",
                                       "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}
            continue
        clim = float(y[tr].mean())
        yt, wk = y[te], weeks[te]
        mats = {}
        for b in order:
            X = blocks[b].X.astype(np.float64)
            mats[b] = impute(X, tr) if blocks[b].impute else X      # this target's training hours only
        exp, preds = {}, {}
        for E, bl in EXPERIMENTS.items():
            Xe = np.hstack([mats[b] for b in bl])
            res, (_, pt) = D.fit_score(Xe, y, tr, va, te, wk, clim)
            s = skill_scores(yt, pt >= res["threshold"])
            exp[E] = {"inputs": inputs(E), "family": res["selected"], "TSS": res["TSS"], "TSS_ci": res["TSS_ci"],
                      "FB": round(s["FB"], 2), "AUC": res["AUC"], "AUC_ci": res["AUC_ci"],
                      "BSS": res["BSS_vs_training_climatology"], "POD": round(s["POD"], 3),
                      "FAR": round(s["FAR"], 3), "val_AUC": res["val_AUC"], "threshold": res["threshold"],
                      "n_features": int(Xe.shape[1])}
            preds[E] = pt
        pers = None
        if f"n_{cls}_24h" in blocks["solexs"].names:          # the day-ahead reference, as there
            pj = blocks["solexs"].names.index(f"n_{cls}_24h")
            pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(mats["solexs"][tr][:, [pj]], y[tr])
            pp = pm.predict_proba(mats["solexs"][te][:, [pj]])[:, 1]
            thr = D.best_tss_threshold(y[va], pm.predict_proba(mats["solexs"][va][:, [pj]])[:, 1])
            pers = {"AUC": round(float(roc_auc_score(yt, pp)), 3), "TSS": round(float(D.tss_at(yt, pp, thr)), 3)}
        n = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
             "test_positive": int(yt.sum()),
             "test_events": _distinct_events(np.sort(event_peaks.get(cls, np.zeros(0))), o[te], H)}
        comps = []
        for a, b, what in COMPARISONS:
            g = D.paired_gain(yt, preds[a], preds[b], wk)
            ea, eb = exp[a], exp[b]
            comps.append({"comparison": f"{a} - {b}", "what": what, "baseline_AUC": eb["AUC"], "AUC": ea["AUC"],
                          "AUC_gain": g["AUC_gain"], "AUC_gain_ci": g["ci"],
                          # the gain as a share of the baseline's skill above chance
                          "relative_to_baseline_skill": (round(g["AUC_gain"] / (eb["AUC"] - 0.5), 3)
                                                         if eb["AUC"] > 0.55 else None),
                          "TSS_change": round(ea["TSS"] - eb["TSS"], 3),
                          "BSS_change": round(ea["BSS"] - eb["BSS"], 3),
                          "paired_forecasts": n["test"], "events": n["test_events"]})
        summary["results"][key] = {"n": n, "base_rate_test": round(float(yt.mean()), 3), "persistence": pers,
                                   "experiments": exp, "comparisons": comps}
    summary["gate"]["verdict"] = _verdict(summary, cfg)
    return summary


def _verdict(summary: dict, cfg: SuitConfig) -> str:
    r = summary["results"].get(f">={cfg.gate_class}1 within {cfg.gate_horizon_h} h", {})
    if "comparisons" not in r:
        return "not evaluable: " + r.get("skipped", "target not scored")
    c = next(x for x in r["comparisons"] if x["comparison"] == "E6 - E4")
    lo, hi = c["AUC_gain_ci"] or (None, None)
    if lo is not None and lo > 0 and c["BSS_change"] >= 0:
        return "PASSED: SUIT adds skill over SoLEXS + HEL1OS"
    if hi is not None and hi < 0:
        return "not passed: SUIT lowers skill"
    return "not passed: no gain shown (the interval includes zero, or Brier skill fell)"


def git_version(root: Path) -> str:
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_real(paths: SuitPaths, cfg: SuitConfig, verbose: bool = True):
    """The three blocks and the targets from the archive. Stops with what is
    missing: the X-ray run (its split and catalogue), GOES, or SUIT data."""
    from solarflare.catalog.build import hel1os_minutes
    from solarflare.io.goes import load_goes
    from solarflare.products import dayahead as D
    from solarflare.settings import load_settings

    from .dataset import build_features, hourly_table, select_frames
    from .io import index_frames
    from .xray import catalogue_bursts, hel1os_block, solexs_block

    S = load_settings()
    meta = paths.split_run / "reports" / "data_meta.json"
    catalog = S.catalog / "master_catalog.csv"
    for need, why in ((meta, "the X-ray model's split: run the X-ray pipeline first"),
                      (catalog, "the master catalogue: run the X-ray pipeline's catalog stage first")):
        if not need.exists():
            raise SystemExit(f"missing {need} ({why})")
    d = json.loads(meta.read_text("utf-8"))["split_dates"]
    split = (float(d["train_end"]), float(d["test_start"]))
    frames = index_frames(paths.data, paths.features / "frames_index.json", verbose)
    if not frames:
        raise SystemExit(f"no SUIT frames under {paths.data}: download them, then `python -m suit inventory`")

    truth = load_goes(paths.goes)
    xf, have_s, o = solexs_block(S.cache, catalog, truth, split[0])
    hf, have_h = hel1os_block(hel1os_minutes(S.cache), catalogue_bursts(catalog), o, cfg.hel1os_min_coverage)
    feats = build_features(select_frames(frames, cfg), cfg, paths.features, verbose)
    Xs, names_s, have_u = hourly_table(feats, o, cfg)
    blocks = {"solexs": Block(xf.to_numpy(np.float64), list(xf.columns), have_s),
              "hel1os": Block(hf.to_numpy(np.float64), list(hf.columns), have_h),
              "suit": Block(Xs, names_s, have_u, impute=True)}
    by_h = {H: D.window_targets(truth, o, 0.0, H) for H in D.HORIZONS_H}
    targets = {(c, H): by_h[H][c] for c in D.CLASSES for H in D.HORIZONS_H}
    peaks = {c: np.sort([f.peak_unix for f in truth.flares if f.peak_flux >= lo]) for c, lo in D.CLASSES.items()}
    return o, blocks, targets, split, peaks


def render(s: dict) -> str:
    """MATRIX.md: one table per target, then the paired differences."""
    synth = s["provenance"] == "synthetic"
    out = ["# SUIT day-ahead matrix (E1-E6)", ""]
    if synth:
        out += ["> **SYNTHETIC DATA: pipeline validation only. These numbers say nothing about the Sun.**", ""]
    h = s["hours"]
    out += [f"Split: training to {s['split']['train_end']} UTC, test from {s['split']['test_start']} UTC. "
            f"Hours with all three instruments: {h['all_three']} of {h['origins']} "
            f"(SoLEXS {h['solexs_observed']}, HEL1OS {h['hel1os_observed']}, SUIT {h['suit_observed']}). "
            f"Every experiment is trained and scored on those same hours.", "",
            f"**Decision gate** ({s['gate']['target']}, {s['gate']['comparison']}): **{s['gate']['verdict']}**", ""]
    ci = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]" if c else ""                  # noqa: E731
    for key, r in s["results"].items():
        out += [f"## {key}", ""]
        if "skipped" in r:
            out += [f"Not scored: {r['skipped']} (n = {r['n']}).", ""]
            continue
        n = r["n"]
        out += [f"Test: {n['test']} hourly forecasts, {n['test_positive']} with a flare in the window, "
                f"{n['test_events']} distinct flares; base rate {r['base_rate_test']}."
                + (f" Persistence: AUC {r['persistence']['AUC']}, TSS {r['persistence']['TSS']}." if r["persistence"] else ""),
                "", "| | Inputs | TSS [95%] | FB | AUC [95%] | BSS | POD | FAR |", "|---|---|---|---|---|---|---|---|"]
        for E, e in r["experiments"].items():
            out.append(f"| {E} | {e['inputs']} | {e['TSS']:.3f}{ci(e['TSS_ci'])} | {e['FB']:.2f} | "
                       f"{e['AUC']:.3f}{ci(e['AUC_ci'])} | {e['BSS']:+.3f} | {e['POD']:.2f} | {e['FAR']:.2f} |")
        out += ["", "| Comparison | What | Baseline AUC | AUC | Gain [95%] | Share of baseline skill | "
                    "TSS change | BSS change | Paired forecasts | Flares |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in r["comparisons"]:
            rel = f"{c['relative_to_baseline_skill']:+.1%}" if c["relative_to_baseline_skill"] is not None else "-"
            out.append(f"| {c['comparison']} | {c['what']} | {c['baseline_AUC']:.3f} | {c['AUC']:.3f} | "
                       f"{c['AUC_gain']:+.3f}{ci(c['AUC_gain_ci'])} | {rel} | {c['TSS_change']:+.3f} | "
                       f"{c['BSS_change']:+.3f} | {c['paired_forecasts']} | {c['events']} |")
        out.append("")
    out += ["Intervals resample whole weeks (hourly forecasts with day-long windows overlap), as in the "
            "day-ahead system. An interval that excludes zero is the criterion used here; no other "
            "significance test is implied.", ""]
    return "\n".join(out)


def write(summary: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in (("matrix_summary.json", json.dumps(summary, indent=1, default=float)),
                       ("MATRIX.md", render(summary))):
        tmp = dest / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest / name)


def run(paths: SuitPaths, cfg: SuitConfig, verbose: bool = True) -> dict:
    from .features import feature_fingerprint

    o, blocks, targets, split, peaks = build_real(paths, cfg, verbose)
    s = evaluate(o, blocks, targets, split, peaks, cfg, provenance="real")
    root = Path(__file__).resolve().parents[1]
    s["reproducibility"] = {"code": git_version(root), "suit_features": feature_fingerprint(cfg),
                            "config": {k: v for k, v in vars(cfg).items() if k != "extra"},
                            "split_run": str(paths.split_run)}
    write(s, paths.outputs / "matrix")
    if verbose:
        print(render(s))
        print(f"wrote {paths.outputs / 'matrix'}")
    return s
