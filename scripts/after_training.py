"""When the running pipeline's `train` stage finishes: keep that model, then rerun on the full data.

    python scripts/after_training.py              # waits for `train`, then acts
    python scripts/after_training.py --dry-run    # reports what it would do, now

The pipeline was started before the HEL1OS zips were ingested, so its `train`
stage is the SoLEXS-only model. That model is a baseline worth keeping. The stages
after it would run for hours on two HEL1OS products: three HEL1OS ablation
trainings, hard X-ray spectra and timing. So once `train` is done, this script:

1. copies outputs/ablations/sharp/xray_only to outputs/ablations/baseline_solexs_only.
   The rerun trains into the same folder, so without the copy the baseline would
   be overwritten. An existing copy is never replaced; a new name is used instead.
2. stops the pipeline job, the way the console's Stop does (the whole process tree).
   The pipeline records its state per stage, so nothing finished is lost.
3. starts one console job, visible in the console, which runs:
   ingest HEL1OS -> check the cache -> ``pipeline --redo cache`` (every stage again).

If `train` fails or the pipeline is stopped by hand, it touches nothing and exits.
It logs to outputs/jobs/after_training.log. Launch it detached (see the memory note
on long runs) so it outlives the chat session.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.console import jobs  # noqa: E402
from dashboard.console.common import pid_alive  # noqa: E402
from solarflare import runall  # noqa: E402
from solarflare.settings import load_settings  # noqa: E402

S = load_settings()
LOG = S.outputs / "jobs" / "after_training.log"
POLL_S = 30
BACKUP = "C:/Users/pawar/Backups/solarflare-cache"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def train_status() -> str:
    try:
        st = runall.State(S.outputs / "pipeline" / "state.json")
    except ValueError:                     # caught mid-write
        return "running"
    return (st.get("train") or {}).get("status", "-")


def pipeline_job():
    """The console job running the pipeline, if it is the one running now."""
    path, job, running = jobs.job_running()
    if running and any("pipeline" in s.get("cmd", []) for s in job.get("steps", [])):
        return path, job
    return None, None


def save_baseline(dry: bool) -> Path | None:
    src = S.ablations / "sharp" / "xray_only"
    if not (src / "reports" / "evaluation.json").exists():
        log(f"no finished run in {src}; nothing to keep")
        return None
    dest = S.ablations / "baseline_solexs_only"
    if dest.exists():
        dest = dest.with_name(f"{dest.name}_{datetime.now():%Y%m%d_%H%M%S}")
    if dry:
        log(f"would copy {src} -> {dest}")
        return dest
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("live.json"))
    log(f"kept the SoLEXS-only model: {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log(f"watching `train` (status now: {train_status()})")
    misses = 0
    while not args.dry_run:
        s = train_status()
        if s == "done":
            break
        if s in ("failed", "interrupted"):
            log(f"`train` is {s!r}: nothing done")
            return 1
        # two misses in a row, so a job file caught mid-write is not mistaken for a stop
        misses = misses + 1 if pipeline_job()[1] is None else 0
        if misses >= 2:
            log(f"no pipeline job running and `train` is {s!r} (stopped by hand?): nothing done")
            return 1
        time.sleep(POLL_S)

    if save_baseline(args.dry_run) is None and not args.dry_run:
        return 1

    path, job = pipeline_job()
    if job is None:
        log("the pipeline job is no longer running; nothing to stop")
    elif args.dry_run:
        log(f"would stop job {job['id']} (runner pid {job.get('runner_pid')})")
    else:
        jobs.stop(path, job)
        for _ in range(60):
            if not pid_alive(job.get("runner_pid")):
                break
            time.sleep(1)
        log(f"stopped job {job['id']} after `train`; the rest reruns on the full data")

    steps = [
        {"label": "ingest HEL1OS", "cmd": ["PY", "-u", "scripts/ingest_batch.py", "--backup", BACKUP,
                                           "--max-failed-pct", "2"]},
        {"label": "check cache", "cmd": ["PY", "-u", "scripts/ingest_batch.py", "--check"]},
        {"label": "pipeline", "cmd": ["PY", "-u", "-m", "solarflare", "pipeline", "--redo", "cache",
                                      "--keep-going"]},
    ]
    if args.dry_run:
        for st in steps:
            log(f"would run: {st['label']}: {' '.join(st['cmd'][1:])}")
        return 0
    err = jobs.launch("Ingest HEL1OS + full pipeline", steps)
    if err:
        log(f"could not start the job: {err}")
        return 1
    log("started the job: ingest HEL1OS -> check cache -> pipeline --redo cache (see the console)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
