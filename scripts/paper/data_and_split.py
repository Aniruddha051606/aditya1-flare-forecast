"""Paper steps 2-3: the data on disk, and the frozen chronological split.

    python scripts/paper/data_and_split.py

Writes paper_results/01_data_inventory.{json,md} and 02_experiment_split.{json,md}.

Read only. The PRADAN zips are listed (and their member lists read, to tell
products with light curves from event-list-only versions); nothing under
data_root is written. The split is rebuilt with the code the final model was
trained with (solarflare.pipeline.prepare on its reports/config.json and the
frozen study cache) and checked against the dates recorded at training, and the
leakage guards are verified on it: disjoint splits, the embargo, and no input or
target span crossing from one split into the next.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np

from common import ROOT, goes_class_letter, utc, write_json, write_text  # noqa: E402  (sets sys.path)
from solarflare.settings import load_settings

S = load_settings()
PRADAN = S.data_root / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"
SLX = re.compile(r"AL1_SLX_L1_(\d{8})_v([\d.]+)\.zip$")
HLS = re.compile(r"HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V(\d+)\.zip$")
DAY = 86400.0


def _day(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d").replace(tzinfo=UTC)


def missing_runs(present: set[str], first: str, last: str) -> list[dict]:
    """Consecutive runs of UTC days in [first, last] with no file."""
    out, d, end = [], _day(first), _day(last)
    run = None
    while d <= end:
        k = d.strftime("%Y%m%d")
        if k not in present:
            run = run or {"from": d.strftime("%Y-%m-%d"), "days": 0}
            run["days"] += 1
            run["to"] = d.strftime("%Y-%m-%d")
        elif run:
            out.append(run)
            run = None
        d += timedelta(days=1)
    if run:
        out.append(run)
    return out


def study_entries() -> list[dict]:
    return [e for e in json.loads((S.cache / "manifest.json").read_text("utf-8")) if e.get("status") == "ok"]


def solexs_inventory(meta: dict, used: list[dict]) -> dict:
    zips = sorted(PRADAN.glob("solexs/level1/**/AL1_SLX_L1_*.zip"))
    versions: dict[str, list[str]] = {}
    for z in zips:
        m = SLX.search(z.name)
        if m:
            versions.setdefault(m.group(1), []).append(m.group(2))
    dates = sorted(versions)
    in_study = {e["source"]["date"] for e in used if e["source"]["kind"] == "solexs"}
    dup = json.loads((S.outputs / "quality" / "solexs_duplicates.json").read_text("utf-8"))
    obs = sum(s["soft_observed"] * s["steps"] for s in meta["segments"]) * meta["dt"] / DAY
    return {
        "product": "SoLEXS Level-1 day files (PRADAN zips, read in place)",
        "zip_files": len(zips), "days_with_files": len(dates),
        "first_day": dates[0], "last_day": dates[-1],
        "days_in_two_or_more_versions": {d: v for d, v in versions.items() if len(v) > 1},
        "version_rule": "the highest version of a day is used",
        "missing_days": missing_runs(set(dates), dates[0], dates[-1]),
        "days_in_frozen_study": len(in_study),
        "days_on_disk_after_the_study": sorted(set(dates) - in_study),
        "copied_days": {"days_checked": dup["days_checked"], "files_repeating_the_previous_day": [
            {"file": d["file"], "copies": d["real_date"], "masked_utc": d["intervals_utc"]} for d in dup["duplicates"]],
            "samples_masked": meta["excluded_solexs_samples"]},
        "cadence": "L1 1 s light curve and 340-channel spectra; gridded to "
                   f"{meta['dt']:.0f} s for the model",
        "model_features": meta["soft_features"],
        "observed_days_in_study": round(obs, 1),
    }


def hel1os_inventory(meta: dict, used: list[dict]) -> dict:
    spec = importlib.util.spec_from_file_location("ingest_batch", ROOT / "scripts" / "ingest_batch.py")
    ib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ib)
    zips = sorted(PRADAN.glob("hel1os/level1/**/HLS_*.zip"))
    obs: dict[str, list[int]] = {}
    days: set[str] = set()
    no_lc = []
    for z in zips:
        m = HLS.search(z.name)
        if not m:
            continue
        obs.setdefault(m.group(1) + m.group(2), []).append(int(m.group(4)))
        t0 = _day(m.group(1)) + timedelta(hours=int(m.group(2)[:2]), minutes=int(m.group(2)[2:4]),
                                          seconds=int(m.group(2)[4:]))
        d = t0.replace(hour=0, minute=0, second=0)
        while d < t0 + timedelta(seconds=int(m.group(3))):
            days.add(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        if not ib.product_in_zip(z)[1]:
            no_lc.append(z.name)
    first, last = min(days), max(days)
    in_study = [e for e in used if e["source"]["kind"] == "hel1os"]
    last_study_day = utc(max(e["t_stop"] for e in in_study) - 1.0, "%Y%m%d")
    obs_days = sum(s["hard_observed"] * s["steps"] for s in meta["segments"]) * meta["dt"] / DAY
    return {
        "product": "HEL1OS Level-1 products (PRADAN zips): light curves, GTIs, photon event lists",
        "zip_files": len(zips), "observations": len(obs),
        "observations_in_two_or_more_versions": sum(len(v) > 1 for v in obs.values()),
        "version_rule": "the higher version of an observation is used (overlapping versions carry the same telemetry)",
        "zips_without_light_curves": {"count": len(no_lc), "note": "older versions shipping only housekeeping, "
                                      "GTIs and event lists; each has a newer version with light curves", "files": no_lc},
        "calibration_zips": sorted(p.name for p in PRADAN.glob("hel1os/cal/**/*.zip")),
        "first_day": first, "last_day": last, "days_with_data": len(days),
        "missing_days": missing_runs(days, first, last),
        "products_used_by_the_study": len(in_study),
        "days_on_disk_after_the_study": sorted(d for d in days if d > last_study_day),
        "study_first_utc": utc(min(e["t_start"] for e in in_study)),
        "study_last_utc": utc(max(e["t_stop"] for e in in_study)),
        "extraction": "light curves extracted one product at a time, cached (20 s grid), extracted files deleted, "
                      "zips kept (scripts/ingest_batch.py); photon event lists read straight from the zips",
        "cadence": f"L1 light curves 1 s (readout batches every 2-8 s); smoothed over 60 s and gridded to "
                   f"{meta['dt']:.0f} s for the model",
        "model_features": meta["hard_features"],
        "observed_days_in_study": round(obs_days, 1),
    }


def goes_inventory(meta: dict, study: list[dict], prep) -> dict:
    from solarflare.io.goes import load_goes

    g = load_goes(S.goes_dir)
    ok = np.isfinite(g.xrsb)
    t0, t1 = min(e["t_start"] for e in study), max(e["t_stop"] for e in study)
    in_span = (g.time_unix >= t0) & (g.time_unix < t1)
    fl = [f for f in g.flares if t0 <= f.peak_unix < t1]
    return {
        "product": "GOES-18 XRS L2 science: flare summary (flsum) and 1-min averages (avg1m)",
        "role": "truth only: flare labels, reference flux, scoring. Never a model input (see 02b_leakage_prevention.md)",
        "files": sorted(g.source_files),
        "flare_list_first_peak": utc(g.flares[0].peak_unix), "flare_list_last_peak": utc(g.flares[-1].peak_unix),
        "flux_first_valid_minute": utc(g.time_unix[ok][0]), "flux_last_valid_minute": utc(g.time_unix[ok][-1]),
        "flux_valid_fraction_over_study_span": round(float(ok[in_span].mean()), 4),
        "flares_in_study_time_span_by_class": dict(sorted(Counter(goes_class_letter(f.peak_flux) for f in fl).items())),
        "flares_in_study_time_span_note": "every GOES-listed flare between the first and last Aditya-L1 product, "
                                          "whether or not Aditya-L1 was observing",
        "flares_labelled_in_observed_segments_by_class": dict(sorted(Counter(
            goes_class_letter(e.peak_rate) for sg in prep.segments for e in sg.events).items())),
        "label_threshold": meta["goes_min_class"],
    }


def other_inventory() -> dict:
    """Data sources outside Paper 1's inputs. (SUIT is excluded from Paper 1 altogether.)"""
    sharp = sorted(S.sharp_dir.glob("*.csv")) if S.sharp_dir.exists() else []
    return {
        "SHARP": {"files": len(sharp), "first": sharp[0].name if sharp else None,
                  "last": sharp[-1].name if sharp else None,
                  "status": "downloaded; left out of the final network by the pipeline's own ablation "
                            "(outputs/ablations/sharp); not an input of any Paper 1 experiment"},
    }


# ---- the split ---------------------------------------------------------------------

def split_definition(meta: dict):
    from solarflare.config import Config
    from solarflare.pipeline import prepare
    from solarflare.preprocess.dataset import build_targets

    cfg = Config.from_json(S.model_dir / "reports" / "config.json")
    prep = prepare(cfg, verbose=False)
    W = prep.windows
    L = cfg.steps_per_window * cfg.pre.dt_seconds
    horizon = max(cfg.win.forecast_horizons_s + cfg.win.occurrence_horizons_s)
    emb = max(cfg.train.embargo_s, cfg.train.global_embargo_s)
    targets = [build_targets(s, cfg) for s in prep.segments]
    occ_names = [f"{int(h / 60)}min" for h in cfg.win.occurrence_horizons_s]
    splits, spans = {}, {}
    for name in ("train", "val", "test"):
        idx = np.asarray(prep.splits[name])
        t = np.array([W[i].t_unix for i in idx])
        seg = np.array([W[i].seg for i in idx])
        j = np.array([W[i].end - 1 for i in idx])
        soft = np.array([prep.segments[s].soft_mask[k] > 0 for s, k in zip(seg, j)])
        hard = np.array([prep.segments[s].hard_mask[k] > 0 for s, k in zip(seg, j)])
        inf_y, inf_m = [], []
        occ_y, occ_m = [], []
        for s in np.unique(seg):
            tg, jj = targets[s], j[seg == s]
            inf_y.append(tg["in_flare"][jj]), inf_m.append(tg["nowcast_mask"][jj] > 0)
            occ_y.append(tg["occurrence"][jj]), occ_m.append(tg["occurrence_mask"][jj] > 0)
        iy, im = np.concatenate(inf_y), np.concatenate(inf_m)
        oy, om = np.concatenate(occ_y), np.concatenate(occ_m)
        t0, t1 = float(t.min()), float(t.max())
        ev = [e for s in prep.segments for e in s.events if t0 <= e.peak_unix <= t1]
        splits[name] = {
            "first_origin_utc": utc(t0), "last_origin_utc": utc(t1), "windows": int(idx.size),
            "windows_soft_at_origin": int(soft.sum()), "windows_hard_at_origin": int(hard.sum()),
            "windows_both_at_origin": int((soft & hard).sum()),
            "in_flare": {"labelled": int(im.sum()), "positive": int(iy[im].sum()),
                         "rate": round(float(iy[im].mean()), 4)},
            "flare_within": {h: {"labelled": int(om[:, k].sum()), "positive": int(oy[om[:, k], k].sum()),
                                 "rate": round(float(oy[om[:, k], k].mean()), 4)} for k, h in enumerate(occ_names)},
            "goes_flares_peaking_in_span": len(ev),
            "goes_flares_by_class": dict(sorted(Counter(goes_class_letter(e.peak_rate) for e in ev).items())),
        }
        spans[name] = (t0, t1, set(idx.tolist()))
    tr, va, te = spans["train"], spans["val"], spans["test"]
    # the split must be the one the model was trained on
    rec = meta["split_dates"]
    same = abs(tr[1] - rec["train_end"]) < 1 and abs(te[0] - rec["test_start"]) < 1
    checks = {
        "same_split_as_recorded_at_training": same,
        "splits_disjoint": not (tr[2] & va[2] or va[2] & te[2] or tr[2] & te[2]),
        "chronological": tr[1] < va[0] and va[1] < te[0],
        "gap_train_to_val_days": round((va[0] - tr[1]) / DAY, 2),
        "gap_val_to_test_days": round((te[0] - va[1]) / DAY, 2),
        "embargo_days": emb / DAY,
        "gaps_at_least_embargo": (va[0] - tr[1]) >= emb and (te[0] - va[1]) >= emb,
        # a window reads [origin - input, origin] and is labelled up to origin + horizon
        "no_input_or_target_span_crosses_a_split": (tr[1] + horizon < va[0] - L) and (va[1] + horizon < te[0] - L),
        "input_window_s": L, "longest_horizon_s": horizon,
    }
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise SystemExit(f"split check failed: {checks}")
    return prep, {
        "mode": meta["split_mode"], "fractions_train_val_test": list(cfg.train.split),
        "rule": "one calendar cut: windows sorted by origin time, the first 60% train, next 20% validation, "
                "last 20% test, with every window within the embargo of a boundary dropped",
        "training_thinning": {"train_windows_before": meta["n_train_before_thinning"],
                              "train_windows_after": meta["n_train"],
                              "rule": "quiet training windows kept every 600 s instead of 120 s (training only)"},
        "window": {"input_s": L, "stride_s": cfg.win.stride_seconds, "grid_s": cfg.pre.dt_seconds,
                   "forecast_horizons_s": list(cfg.win.forecast_horizons_s),
                   "occurrence_horizons_s": list(cfg.win.occurrence_horizons_s)},
        "labels": {"source": cfg.pre.label_source, "min_class": cfg.pre.goes_min_class},
        "seed": cfg.train.seed,
        "splits": splits, "checks": checks,
        "total_windows": len(W), "total_flares_in_study": sum(len(s.events) for s in prep.segments),
    }


def render_inventory(d: dict) -> str:
    s, h, g, o = d["SoLEXS"], d["HEL1OS"], d["GOES"], d["other"]
    def miss(r: list[dict]) -> str:
        if not r:
            return "none"
        top = sorted(r, key=lambda x: -x["days"])[:3]
        return (f"{sum(x['days'] for x in r)} days in {len(r)} gaps; longest "
                + ", ".join(f"{x['from']}..{x['to']} ({x['days']} d)" for x in top) + " (full list in the JSON)")
    return "\n".join([
        "# Data inventory", "", f"Generated by `scripts/paper/data_and_split.py` ({d['provenance']['generated_utc']} UTC, "
        f"commit {d['provenance']['git_commit']}). Frozen study cache: `{d['study_cache']}` (marker present: "
        f"{d['study_cache_frozen']}).", "",
        f"Cache build for the study: {d['cache_build']['ok']} products read, {d['cache_build']['empty']} empty, "
        f"{d['cache_build']['failed']} unreadable/corrupt.", "",
        "## SoLEXS", "",
        f"- {s['zip_files']} zips, {s['days_with_files']} days, {s['first_day']} to {s['last_day']}; "
        f"{len(s['days_in_two_or_more_versions'])} days in two versions (highest used).",
        f"- Missing days: {miss(s['missing_days'])}.",
        f"- Files repeating the previous day: {len(s['copied_days']['files_repeating_the_previous_day'])} "
        f"({s['copied_days']['samples_masked']} samples masked).",
        f"- In the frozen study: {s['days_in_frozen_study']} days, {s['observed_days_in_study']} observed days. "
        f"On disk but after the study: {', '.join(s['days_on_disk_after_the_study']) or 'none'}.",
        f"- {s['cadence']}; {s['model_features']} model features.", "",
        "## HEL1OS", "",
        f"- {h['zip_files']} zips, {h['observations']} observations "
        f"({h['observations_in_two_or_more_versions']} in two or more versions), {h['days_with_data']} days, "
        f"{h['first_day']} to {h['last_day']}; {h['zips_without_light_curves']['count']} zips without light curves.",
        f"- Missing days: {miss(h['missing_days'])}.",
        f"- Used by the study: {h['products_used_by_the_study']} products, {h['study_first_utc']} to "
        f"{h['study_last_utc']}, {h['observed_days_in_study']} observed days. On disk but after the study: "
        f"{', '.join(h['days_on_disk_after_the_study']) or 'none'}.",
        f"- {h['cadence']}; {h['model_features']} model features.", f"- {h['extraction']}.", "",
        "## GOES-18 (truth only)", "",
        f"- Flare list {g['flare_list_first_peak']} to {g['flare_list_last_peak']}; 1-min flux "
        f"{g['flux_first_valid_minute']} to {g['flux_last_valid_minute']} "
        f"({100 * g['flux_valid_fraction_over_study_span']:.1f}% valid over the study span).",
        f"- Flares in the study time span by class: {g['flares_in_study_time_span_by_class']} "
        f"({g['flares_in_study_time_span_note']}); labelled flares inside observed segments: "
        f"{g['flares_labelled_in_observed_segments_by_class']}; labels >= {g['label_threshold']}.",
        "", "## Other", "",
        f"- SHARP: {o['SHARP']['files']} files; {o['SHARP']['status']}.", ""])


def render_split(d: dict) -> str:
    L = ["# Experiment split (frozen)", "", f"Rebuilt by `scripts/paper/data_and_split.py` with the training code "
         f"and checked against the split recorded at training. {d['rule']}.", "",
         "| Split | first origin (UTC) | last origin (UTC) | windows | both instruments at origin | in-flare rate | "
         "flare within 15 / 30 / 60 min | GOES flares (C/M/X) |", "|---|---|---|---:|---:|---:|---|---|"]
    for k, v in d["splits"].items():
        fw = " / ".join(f"{x['rate']:.3f}" for x in v["flare_within"].values())
        L.append(f"| {k} | {v['first_origin_utc']} | {v['last_origin_utc']} | {v['windows']:,} | "
                 f"{v['windows_both_at_origin']:,} | {v['in_flare']['rate']:.3f} | {fw} | "
                 f"{v['goes_flares_peaking_in_span']} ({'/'.join(str(v['goes_flares_by_class'].get(c, 0)) for c in 'CMX')}) |")
    c = d["checks"]
    L += ["", f"Training windows thinned from {d['training_thinning']['train_windows_before']:,} to "
          f"{d['training_thinning']['train_windows_after']:,} ({d['training_thinning']['rule']}). Seed {d['seed']}.", "",
          "## Checks (all must hold; the script stops otherwise)", "",
          *[f"- {k}: {v}" for k, v in c.items()], ""]
    return "\n".join(L)


def main() -> int:
    meta = json.loads((S.model_dir / "reports" / "data_meta.json").read_text("utf-8"))
    study = study_entries()
    from solarflare.preprocess.cache import cache_frozen

    print("rebuilding the training data (~5 min) ...", flush=True)
    prep, sp = split_definition(meta)
    used = [e for e in prep.cache_entries if e.get("status") == "ok"]
    build = Counter(e.get("status") for e in prep.cache_entries)
    inv = {"study_cache": str(S.cache), "study_cache_frozen": cache_frozen(S.cache),
           "cache_build": {k: int(build.get(k, 0)) for k in ("ok", "empty", "failed")},
           "SoLEXS": solexs_inventory(meta, used), "HEL1OS": hel1os_inventory(meta, used),
           "GOES": goes_inventory(meta, study, prep), "other": other_inventory()}
    p = write_json("01_data_inventory.json", inv)
    write_text("01_data_inventory.md", render_inventory(json.loads(p.read_text("utf-8"))))
    p = write_json("02_experiment_split.json", sp)
    write_text("02_experiment_split.md", render_split(json.loads(p.read_text("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
