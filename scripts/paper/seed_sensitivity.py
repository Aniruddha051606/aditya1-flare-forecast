"""Paper step 17: training-seed sensitivity of E4 - E1 (a robustness study only).

    python scripts/paper/seed_sensitivity.py --train --refresh  # train what is missing (resumes), then everything below
    python scripts/paper/seed_sensitivity.py                 # predict finished seed runs (GPU, ~5 min each) + metrics
    python scripts/paper/seed_sensitivity.py --wait          # wait for the seed training job to end, then run
    python scripts/paper/seed_sensitivity.py --metrics-only  # metrics from the saved predictions
    add --refresh to rebuild the tables, summary and manifest afterwards

The reported E1 and E4 results are the primary runs (train.seed 1337) and are not
changed by this script. E1 and E4 were retrained with the other two seeds
declared in config/project.toml (7, 42) -- same code, data, split and settings,
only ``train.seed`` differs:

    python -m solarflare train --from-run outputs/model --out-dir outputs/paper/seeds/E1_seed7 --set model.inputs=soft --seed 7
    python -m solarflare train --from-run outputs/model --out-dir outputs/paper/seeds/E4_seed7 --seed 7
    (and the same with 42)

Every seed pair is scored exactly as paired_comparisons.csv scores the primary
pair (paper_metrics.paired: common test windows, each model's own validation
threshold and calibration, day-block intervals). All seeds are reported; none
is selected. The seed-1337 rows are recomputed from the saved primary
predictions and checked against paired_comparisons.csv.

Writes paper_results/seed_sensitivity.csv (every paired metric per seed),
seed_sensitivity_models.csv (each model's own scores per seed) and
seed_sensitivity_summary.csv (per metric: every seed's value, mean, range, and
how many seeds' intervals exclude zero in each direction).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
import time

import numpy as np

import evaluate_experiments as ev
from common import PAPER, ROOT  # noqa: E402  (sets sys.path)
from paper_metrics import classification, fit_on_validation, paired
from solarflare.settings import load_settings

S = load_settings()
PRED = S.outputs / "paper" / "predictions"
SEED_PRED = PRED / "seeds"
PRIMARY_SEED = 1337
SEEDS = (PRIMARY_SEED, 7, 42)
INPUTS = {"E1": "soft", "E4": "both"}
#: metrics where a lower value is better (so a negative E4 - E1 favours E4)
LOWER_IS_BETTER = {"MAE_dex"}


def run_dir(key: str, seed: int):
    if seed == PRIMARY_SEED:
        return ev.EXPERIMENTS[key]["run"]
    return S.outputs / "paper" / "seeds" / f"{key}_seed{seed}"


def _file(key: str, seed: int, split: str):
    return SEED_PRED / f"{key}_seed{seed}_{split}.npz"


def trained(key: str, seed: int) -> str | None:
    """None when the run is finished and is what it should be, else the reason."""
    run = run_dir(key, seed)
    if not (run / "checkpoints" / "best.pt").exists() or not (run / "reports" / "evaluation.json").exists():
        return f"{key} seed {seed}: training not finished in {run.relative_to(ROOT)}"
    cfg = json.loads((run / "reports" / "config.json").read_text("utf-8"))
    if cfg["model"].get("inputs", "both") != INPUTS[key] or cfg["train"]["seed"] != seed:
        return f"{key} seed {seed}: {run} has inputs={cfg['model'].get('inputs')}, seed={cfg['train']['seed']}"
    return None


def launch_training() -> str:
    """Start one detached console job for every seed run not yet finished, pair by pair
    (E1 then E4 for each seed), so the first complete pair is ready soonest."""
    from dashboard.console import jobs

    steps = []
    for seed in SEEDS:
        for key in INPUTS:
            if seed == PRIMARY_SEED or trained(key, seed) is None:
                continue
            extra = ["--set", f"model.inputs={INPUTS[key]}"] if INPUTS[key] != "both" else []
            steps.append({"label": f"{key} seed {seed}",
                          "cmd": ["PY", "-u", "-m", "solarflare", "train", "--from-run", "outputs/model",
                                  "--out-dir", run_dir(key, seed).relative_to(ROOT).as_posix(), *extra,
                                  "--seed", str(seed)]})
    if not steps:
        return "every seed run is already trained"
    err = jobs.launch("Paper seed sensitivity: E1 and E4, seeds 7 and 42", steps)
    return err or "training started: " + ", ".join(s["label"] for s in steps)


def load(key: str, seed: int, split: str) -> dict:
    if seed == PRIMARY_SEED:
        return ev._load(key, split)
    with np.load(_file(key, seed, split)) as z:
        return {k: z[k] for k in z.files}


def predict_seeds(todo: list[tuple[str, int]]) -> None:
    """Predict each finished seed run on every validation and test window (GPU)."""
    from solarflare.config import Config
    from solarflare.pipeline import prepare, resolve_device
    from solarflare.preprocess.dataset import build_targets
    from solarflare.products.model_tests import load_run, predict

    t0 = time.time()
    device = resolve_device("auto")
    cfg = Config.from_json(S.model_dir / "reports" / "config.json")
    prep = prepare(cfg, verbose=False)
    targets = [build_targets(s, cfg) for s in prep.segments]
    sets = {"val": prep.splits["val"], "test": prep.splits["test"]}
    SEED_PRED.mkdir(parents=True, exist_ok=True)
    for key, seed in todo:
        r = load_run(run_dir(key, seed), device, hel1os=True)
        assert r["cfg"].train.seed == seed, (key, seed, r["cfg"].train.seed)
        for split, p in predict(r, prep, targets, prep.windows, sets, device).items():
            ref = ev._load("E4", split)
            assert np.array_equal(p["y_t_unix"], ref["y_t_unix"]), f"{key} seed {seed} {split}: windows differ"
            np.savez_compressed(_file(key, seed, split), **{k: p[k] for k in ev.KEEP}, best_epoch=r["epoch"])
        print(f"{key} seed {seed} predicted, best epoch {r['epoch']} ({time.time() - t0:.0f} s)", flush=True)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _write(name: str, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(PAPER / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> paper_results/{name} ({len(rows)} rows)")


def metrics(seeds: list[int]) -> int:
    meta = json.loads((PRED / "predictions_meta.json").read_text("utf-8"))
    occ, hor, qs = (meta["window"][k] for k in ("occurrence_min", "forecast_min", "quantiles"))
    t4, v4 = ev._load("E4", "test"), ev._load("E4", "val")
    common = t4["pop_soft"] & t4["pop_hard"]
    native_v = {"E1": v4["pop_soft"], "E4": np.ones(v4["pop_soft"].size, bool)}
    pairs, models = [], []
    for seed in seeds:
        test = {k: load(k, seed, "test") for k in INPUTS}
        val = {k: load(k, seed, "val") for k in INPUTS}
        fitted = {k: fit_on_validation(val[k], native_v[k], occ) for k in INPUTS}
        tag = {"seed": seed, "primary": seed == PRIMARY_SEED}
        pairs += [{**tag, **r} for r in paired("E4", "E1", "common", test["E4"], test["E1"], common,
                                               fitted["E4"], fitted["E1"], occ, hor, qs)]
        for k in INPUTS:
            models += [{**tag, **r} for r in classification(k, "common", test[k], common, fitted[k], occ)
                       if r["metric"] in ("ROC_AUC", "TSS", "BSS_calibrated")]
        print(f"seed {seed} scored", flush=True)

    # the primary seed must reproduce the published paired comparison exactly
    with open(PAPER / "paired_comparisons.csv", encoding="utf-8") as fh:
        pub = {(r["quantity"], r["horizon_min"], r["metric"]): r for r in csv.DictReader(fh)
               if r["comparison"] == "E4 - E1" and r["population"] == "common"}
    mism = [r for r in pairs if r["seed"] == PRIMARY_SEED
            and not np.isclose(_num(pub[(r["quantity"], str(r["horizon_min"]), r["metric"])]["value"]),
                               _num(r["value"]), rtol=0, atol=1e-9, equal_nan=True)]
    if mism:
        print(f"ERROR: seed {PRIMARY_SEED} does not reproduce paired_comparisons.csv ({len(mism)} rows); not written")
        return 1

    summary = []
    for key in dict.fromkeys((r["quantity"], r["horizon_min"], r["metric"]) for r in pairs):
        rs = {r["seed"]: r for r in pairs if (r["quantity"], r["horizon_min"], r["metric"]) == key}
        vals = np.array([_num(rs[s]["value"]) for s in seeds])
        lo, hi = ({s: _num(rs[s][c]) for s in seeds} for c in ("ci95_low", "ci95_high"))
        e4_better = [s for s in seeds if (hi[s] < 0 if key[2] in LOWER_IS_BETTER else lo[s] > 0)]
        e1_better = [s for s in seeds if (lo[s] > 0 if key[2] in LOWER_IS_BETTER else hi[s] < 0)]
        summary.append({"quantity": key[0], "horizon_min": key[1], "metric": key[2],
                        **{f"seed_{s}": rs[s]["value"] for s in seeds},
                        **{f"seed_{s}_ci95": f"[{rs[s]['ci95_low']}, {rs[s]['ci95_high']}]" for s in seeds},
                        "mean_over_seeds": round(float(np.nanmean(vals)), 6),
                        "min_over_seeds": round(float(np.nanmin(vals)), 6),
                        "max_over_seeds": round(float(np.nanmax(vals)), 6),
                        "n_seeds": len(seeds), "n_seeds_interval_favours_E4": len(e4_better),
                        "n_seeds_interval_favours_E1": len(e1_better)})
    _write("seed_sensitivity.csv", pairs)
    _write("seed_sensitivity_models.csv", models)
    _write("seed_sensitivity_summary.csv", summary)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--wait", action="store_true", help="wait for the seed training job to finish first")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="then rerun make_tables, summary and manifest")
    ap.add_argument("--train", action="store_true",
                    help="first start a detached console job training the seed runs not yet finished "
                         "(an interrupted run resumes from its last epoch); implies --wait")
    args = ap.parse_args()
    if sys.platform == "win32":                    # keep the machine awake while waiting and predicting
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if args.train:
        print(launch_training(), flush=True)
        args.wait = True
    if args.wait:
        from dashboard.console import jobs

        print("waiting for the training job to finish (closing this window does not stop it; "
              "rerun with --refresh afterwards) ...", flush=True)
        while jobs.job_running()[2]:
            time.sleep(300)
    missing = [m for s in SEEDS for k in INPUTS if (m := trained(k, s))]
    for m in missing:
        print("missing:", m)
    seeds = [s for s in SEEDS if all(trained(k, s) is None for k in INPUTS)]
    if len(seeds) < 2:
        print("fewer than two complete seed pairs; nothing to compare")
        return 2
    if not args.metrics_only:
        todo = [(k, s) for s in seeds if s != PRIMARY_SEED for k in INPUTS if not _file(k, s, "test").exists()]
        if todo:
            predict_seeds(todo)
    rc = metrics(seeds)
    if rc == 0 and args.refresh:
        import subprocess
        from pathlib import Path

        for step in ("make_tables.py", "summary.py", "manifest.py"):
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name(step))], cwd=ROOT, check=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
