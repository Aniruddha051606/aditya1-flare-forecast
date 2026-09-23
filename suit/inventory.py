"""What SUIT data is there, and does the reader understand it?

    python -m suit inventory --show 3

Run this on the first real download before anything else. It reads headers only
and reports: files and frames; unreadable or corrupt files; frames with no time
or no filter; *which header keyword* each time and filter came from (or the
file name) -- the check on suit/io.py's assumptions; filters, image sizes and
observing modes; per-filter cadence and hourly coverage of the full-disk frames;
the time span, days observed and the longest gaps. It also writes all of it to
outputs/suit/inventory.json, and with --show N prints N real headers.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import FILTERS, SuitConfig, SuitPaths
from .io import SuitFrame, _open_fits, image_hdu, index_frames


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M")


def summarise(frames: list[SuitFrame], problems: list, cfg: SuitConfig, n_files: int) -> dict:
    full = [f for f in frames if min(f.nx, f.ny) >= cfg.min_full_px]
    out: dict = {"files": n_files, "frames": len(frames), "problems": len(problems),
                 "problem_examples": [list(p) for p in problems[:20]],
                 "full_disk_frames": len(full), "region_of_interest_frames": len(frames) - len(full),
                 "time_from": dict(Counter(f.time_from or "?" for f in frames)),
                 "filter_from": dict(Counter(f.filt_from or "none" for f in frames)),
                 "no_filter": sum(not f.filt for f in frames),
                 "image_sizes": dict(Counter(f"{f.nx}x{f.ny}" for f in frames).most_common(10)),
                 "modes": dict(Counter(f.mode or "(no keyword)" for f in frames).most_common(10)),
                 "filters": {}}
    if not frames:
        return out
    t = np.array([f.t_unix for f in frames])
    days = sorted({_utc(x)[:10] for x in t})
    ts = np.sort(t)
    gaps = np.diff(ts)
    big = np.argsort(gaps)[::-1][:3] if gaps.size else []
    out.update(first=_utc(ts[0]), last=_utc(ts[-1]), days_observed=len(days),
               longest_gaps=[{"from": _utc(ts[i]), "to": _utc(ts[i + 1]), "hours": round(gaps[i] / 3600.0, 1)}
                             for i in big if gaps[i] > 3600.0])
    span_h = max((ts[-1] - ts[0]) / 3600.0, 1.0)
    for filt in sorted({f.filt or "?" for f in full}):
        tf = np.sort([f.t_unix for f in full if (f.filt or "?") == filt])
        d = np.diff(tf)
        out["filters"][filt] = {
            "what": FILTERS.get(filt, ("", "not a SUIT filter name: check FILTER_KEYS in suit/io.py"))[1],
            "full_disk_frames": int(tf.size),
            "median_cadence_min": round(float(np.median(d)) / 60.0, 1) if d.size else None,
            "hours_covered": int(np.unique(np.floor(tf / 3600.0)).size),
            "hours_covered_fraction": round(np.unique(np.floor(tf / 3600.0)).size / span_h, 3),
            "configured": filt in cfg.filters}
    return out


def print_report(r: dict, root: Path) -> None:
    print(f"SUIT under {root}: {r['files']} files, {r['frames']} frames, {r['problems']} problem(s)")
    for where, why in r["problem_examples"][:5]:
        print(f"  ! {Path(where.split('|')[0]).name}{' | ' + where.split('|', 1)[1] if '|' in where else ''}: {why}")
    if not r["frames"]:
        return
    print(f"  {r['first']} -> {r['last']} UTC, {r['days_observed']} days with data")
    for g in r.get("longest_gaps", []):
        print(f"  gap: {g['from']} -> {g['to']} ({g['hours']} h)")
    print(f"  time from: {r['time_from']}   filter from: {r['filter_from']}   no filter: {r['no_filter']}")
    print(f"  full-disk: {r['full_disk_frames']}; smaller (regions of interest, never used): "
          f"{r['region_of_interest_frames']}")
    print(f"  image sizes: {r['image_sizes']}")
    print(f"  observing modes: {r['modes']}")
    print("  full-disk frames by filter:")
    for filt, d in r["filters"].items():
        print(f"    {filt:5s} {d['full_disk_frames']:7d}  cadence {d['median_cadence_min']} min, "
              f"{d['hours_covered_fraction']:.0%} of hours  {'[used]' if d['configured'] else ''}  {d['what']}")
    missing = [f for f in r.get("configured_filters", []) if f not in r["filters"]]
    if missing:
        print(f"  configured but not found: {missing} -- check the names in config/suit.toml")


def print_header(f: SuitFrame) -> None:
    import zipfile

    print(f"\n--- header of {Path(f.member or f.path).name}  (time from {f.time_from or '?'}, "
          f"filter from {f.filt_from or 'none'})")
    if f.member:
        with zipfile.ZipFile(f.path) as z, z.open(f.member) as fh, _open_fits(fh, f.member) as h:
            print(repr(image_hdu(h)[1]))
    else:
        with open(f.path, "rb") as fh, _open_fits(fh, f.path) as h:
            print(repr(image_hdu(h)[1]))


def run(paths: SuitPaths, cfg: SuitConfig, show: int = 0) -> dict:
    from .io import FITS_NAME

    problems: list = []
    frames = index_frames(paths.data, paths.features / "frames_index.json", verbose=True, problems=problems)
    n_files = sum(1 for p in Path(paths.data).rglob("*") if p.is_file()
                  and (p.suffix.lower() == ".zip" or FITS_NAME.search(p.name))) if Path(paths.data).exists() else 0
    r = summarise(frames, problems, cfg, n_files)
    r["configured_filters"] = list(cfg.filters)
    print_report(r, paths.data)
    seen: set[str] = set()
    for f in frames:
        if len(seen) >= show:
            break
        if f.path not in seen:
            seen.add(f.path)
            print_header(f)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    dest = paths.outputs / "inventory.json"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(r, indent=1), encoding="utf-8")
    os.replace(tmp, dest)
    return r
