"""python -m suit inventory | features | train  (see suit/README.md)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .config import FILTERS, SuitConfig, SuitPaths, default_paths


def _paths(args) -> SuitPaths:
    p = default_paths()
    return replace(p, data=Path(args.data)) if args.data else p


def _cfg(args) -> SuitConfig:
    cfg = SuitConfig()
    if args.filters:
        cfg.filters = tuple(f.strip().upper() for f in args.filters.split(","))
    if getattr(args, "embargo_days", None) is not None:
        cfg.embargo_days = args.embargo_days
    return cfg


def _print_header(f) -> None:
    """The merged header of one frame, so the keyword names can be checked."""
    import zipfile

    from .io import _image_hdu, _open_fits

    print(f"\n--- header of {Path(f.member or f.path).name}")
    if f.member:
        with zipfile.ZipFile(f.path) as z, z.open(f.member) as fh, _open_fits(fh, f.member) as h:
            print(repr(_image_hdu(h)[1]))
    else:
        with open(f.path, "rb") as fh, _open_fits(fh, f.path) as h:
            print(repr(_image_hdu(h)[1]))


def cmd_inventory(args) -> int:
    from .io import index_frames

    p, cfg = _paths(args), _cfg(args)
    frames = index_frames(p.data, p.features / "frames_index.json", verbose=True)
    print(f"{len(frames)} frames under {p.data}")
    if not frames:
        return 1
    day = lambda t: datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d")   # noqa: E731
    full = [f for f in frames if min(f.nx, f.ny) >= cfg.min_full_px]
    print(f"  {day(frames[0].t_unix)} -> {day(frames[-1].t_unix)}, {len({day(f.t_unix) for f in frames})} days")
    print(f"  full-disk size (>= {cfg.min_full_px} px): {len(full)}; smaller (regions of interest, never used): "
          f"{len(frames) - len(full)}")
    print("  by filter (full-disk frames):")
    for filt, n in sorted(Counter(f.filt or "?" for f in full).items()):
        what = FILTERS.get(filt, ("", "filter not recognised: check FILTER_KEYS in suit/io.py"))[1]
        print(f"    {filt:5s} {n:7d}  {what}")
    print(f"  image sizes: {dict(Counter(f'{f.nx}x{f.ny}' for f in frames).most_common(6))}")
    print(f"  observing modes: {dict(Counter(f.mode or '(no keyword)' for f in frames).most_common(6))}")
    seen: set[str] = set()
    for f in frames:
        if len(seen) >= args.show:
            break
        if f.path not in seen:
            seen.add(f.path)
            _print_header(f)
    return 0


def cmd_features(args) -> int:
    from .dataset import build_features, select_frames
    from .io import index_frames

    p, cfg = _paths(args), _cfg(args)
    frames = index_frames(p.data, p.features / "frames_index.json", verbose=True)
    build_features(select_frames(frames, cfg), cfg, p.features, verbose=True)
    return 0


def cmd_train(args) -> int:
    from .train import run

    run(_paths(args), _cfg(args), own_split=args.own_split, verbose=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m suit", description="SUIT full-disk flare forecasting")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("inventory", cmd_inventory, "what SUIT data is there; --show N prints headers"),
                            ("features", cmd_features, "compute per-frame features (incremental)"),
                            ("train", cmd_train, "hourly table, models, scores -> outputs/suit/suit_summary.json")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", help="SUIT folder (default: <data_root>/suit)")
        p.add_argument("--filters", help="comma-separated, e.g. NB03,NB04,NB08,BB03")
        p.set_defaults(func=fn)
        if name == "inventory":
            p.add_argument("--show", type=int, default=0, help="print the headers of this many files")
        if name == "train":
            p.add_argument("--own-split", action="store_true", help="ignore the X-ray model's split")
            p.add_argument("--embargo-days", type=float, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
