"""Reproduce every Paper 1 result, in order, into paper_results/.

    python scripts/paper/run_all_paper_experiments.py                     # all steps; stops if E1/E2 are missing
    python scripts/paper/run_all_paper_experiments.py --train             # also trains E1 and E2 if missing (~6 h each, GPU)
    python scripts/paper/run_all_paper_experiments.py --wait-for-training # wait for a running E1/E2 job, then run
    python scripts/paper/run_all_paper_experiments.py --skip-tests        # skip the test suites (5-7 min)

Steps (each a script of its own, rerunnable alone):
   1 environment.py            git commit, Python, OS, packages, GPU
   2 data_and_split.py         data inventory, frozen split and its leakage checks (rebuilds the data, ~5 min)
   3 test_status.py            every test suite and lint
   4 E1 / E2 training          only with --train: python -m solarflare train --from-run outputs/model ...
   5 evaluate_experiments.py   E1/E2/E4 predictions (GPU, ~20 min) and every metric table
   6 leadtime_tables.py        lead times and operating points (from the alerts stage)
   7 make_tables.py            publication tables
   8 paper_results/plotting/fig*.py   every figure
   9 manifest.py               RESULTS_MANIFEST.json
  10 summary.py                PAPER_RESULTS_SUMMARY.md

E4 is the final model of ``python -m solarflare pipeline`` (outputs/model); this
script never retrains it. Nothing under data_root is written: the study cache is
frozen and the zips are only read. Log: outputs/paper/run_all.log.
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import PAPER, ROOT  # noqa: E402  (sets sys.path)
from solarflare.settings import load_settings

S = load_settings()
HERE = Path(__file__).resolve().parent
LOG = S.outputs / "paper" / "run_all.log"
TRAIN = {"E1": ["--out-dir", "outputs/paper/E1_solexs_only", "--set", "model.inputs=soft"],
         "E2": ["--out-dir", "outputs/paper/E2_hel1os_only", "--set", "model.inputs=hard"]}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(label: str, args: list[str], cwd: Path = ROOT) -> None:
    log(f"{label}: {' '.join(args)}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-u", *args], cwd=cwd).returncode
    if rc != 0:
        log(f"{label} FAILED (exit {rc}); stopping. Fix the cause and rerun; finished steps are rerun cheaply.")
        raise SystemExit(rc)
    log(f"{label} done ({time.time() - t0:.0f} s)")


def not_ready() -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_experiments", HERE / "evaluate_experiments.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.not_ready()


def wait_for_training() -> None:
    from dashboard.console import jobs

    log("waiting for E1 and E2 to finish training")
    while not_ready():
        _, job, running = jobs.job_running()
        if not running:
            missing = not_ready()
            if missing:
                log("no training job is running and experiments are missing:\n  " + "\n  ".join(missing))
                raise SystemExit(2)
            break
        time.sleep(300)
    log("E1 and E2 trained")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--train", action="store_true", help="train E1 and E2 when missing")
    ap.add_argument("--wait-for-training", action="store_true", help="wait for a running E1/E2 training job")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    if sys.platform == "win32":                    # keep the machine awake for a long unattended run
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        if args.wait_for_training:
            wait_for_training()
        run("1 environment", [str(HERE / "environment.py")])
        run("2 data and split", [str(HERE / "data_and_split.py")])
        if not args.skip_tests:
            run("3 tests", [str(HERE / "test_status.py")])
        missing = not_ready()
        if missing and args.train:
            for key, extra in TRAIN.items():
                if any(m.startswith(key) for m in missing):
                    run(f"4 train {key}", ["-m", "solarflare", "train", "--from-run", "outputs/model", *extra])
            missing = not_ready()
        if missing:
            log("stopping before evaluation -- experiments missing:\n  " + "\n  ".join(missing)
                + "\n  (rerun with --train, or train them with the commands in the module docstring)")
            return 2
        run("5 evaluate E1/E2/E4", [str(HERE / "evaluate_experiments.py")])
        run("6 lead-time tables", [str(HERE / "leadtime_tables.py")])
        run("7 tables", [str(HERE / "make_tables.py")])
        for fig in sorted((PAPER / "plotting").glob("fig*.py")):
            run(f"8 {fig.stem}", [str(fig)], cwd=fig.parent)
        run("9 manifest", [str(HERE / "manifest.py")])
        run("10 summary", [str(HERE / "summary.py")])
        log("all Paper 1 results produced: paper_results/")
        return 0
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    raise SystemExit(main())
