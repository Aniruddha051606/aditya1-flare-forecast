"""Reading the metric tables written by scripts/paper/evaluate_experiments.py."""

from __future__ import annotations

import csv

from style import PAPER

HEAD_ORDER = ["in_flare", "flare_within_15min", "flare_within_30min", "flare_within_60min"]
HEAD_LABEL = {"in_flare": "flare in progress", "flare_within_15min": "flare within 15 min",
              "flare_within_30min": "flare within 30 min", "flare_within_60min": "flare within 60 min"}


def rows(name: str) -> list[dict]:
    path = PAPER / name
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/paper/evaluate_experiments.py first")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v):
    return None if v in ("", None, "None") else float(v)


def pick(table: list[dict], **kw) -> list[dict]:
    return [r for r in table if all(str(r.get(k)) == str(v) for k, v in kw.items())]


def one(table: list[dict], **kw) -> dict | None:
    got = pick(table, **kw)
    return got[0] if got else None
