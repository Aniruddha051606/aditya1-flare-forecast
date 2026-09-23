"""python -m suit inventory | features | train | matrix  (see suit/README.md).

Settings: config/suit.toml; the options here override it for one run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import SuitConfig, SuitPaths, default_paths, load_config


def _paths(args) -> SuitPaths:
    p = default_paths()
    return replace(p, data=Path(args.data)) if args.data else p


def _cfg(args) -> SuitConfig:
    cfg = load_config(Path(args.config) if args.config else None)
    if args.filters:
        cfg.filters = tuple(f.strip().upper() for f in args.filters.split(","))
    if getattr(args, "embargo_days", None) is not None:
        cfg.embargo_days = args.embargo_days
    return cfg


def cmd_inventory(args) -> int:
    from .inventory import run

    r = run(_paths(args), _cfg(args), show=args.show)
    return 0 if r["frames"] else 1


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


def cmd_matrix(args) -> int:
    from .matrix import run

    run(_paths(args), _cfg(args), verbose=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m suit", description="SUIT full-disk flare forecasting")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (
            ("inventory", cmd_inventory, "what SUIT data is there, and does the reader understand it"),
            ("features", cmd_features, "compute per-frame features (incremental)"),
            ("train", cmd_train, "SUIT alone -> outputs/suit/suit_summary.json"),
            ("matrix", cmd_matrix, "E1-E6: SoLEXS, HEL1OS, SUIT and combinations -> outputs/suit/matrix/")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", help="SUIT folder (default: <data_root>/suit)")
        p.add_argument("--config", help="settings file (default: config/suit.toml)")
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
