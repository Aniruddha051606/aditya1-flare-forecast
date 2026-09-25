"""Run every test suite and the linters, and keep the results.

    python scripts/run_tests.py                 # all suites -> outputs/tests/suites/
    python scripts/run_tests.py test_scale      # just these

Each suite runs in its own process (as in CI), its full output goes to
``<outputs>/tests/suites/<suite>.log``, and ``summary.json`` / ``summary.md``
list PASS/FAIL counts and run time per suite. Exit status 1 if anything failed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solarflare.settings import load_settings  # noqa: E402

LINT = {
    "pyflakes": ["-m", "pyflakes", "solarflare", "suit", "tests", "scripts", "dashboard/console",
                 "dashboard/job_runner.py", "dashboard/mission_control.pyw"],
    "ruff": ["-m", "ruff", "check", "solarflare", "suit", "tests", "scripts", "dashboard"],
}


def run(name: str, args: list[str], log: Path) -> dict:
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    text = log.read_text("utf-8", errors="replace")
    fails = re.findall(r"^\s*FAIL\s+(.+?)\s*$", text, re.M)
    return {"name": name, "exit": rc, "ok": rc == 0, "seconds": round(time.time() - t0, 1),
            "passed": len(re.findall(r"^\s*PASS\s", text, re.M)), "failed": len(fails),
            "failures": fails[:20], "errors": re.findall(r"^\s*ERROR .+$", text, re.M)[:10],
            "log": log.name}


def main(argv: list[str]) -> int:
    out = load_settings().tests / "suites"
    out.mkdir(parents=True, exist_ok=True)
    suites = argv or sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    started = datetime.now(UTC)
    rows = []
    for s in suites:
        print(f"{s} ...", end=" ", flush=True)
        r = run(s, ["-m", f"tests.{s}"], out / f"{s}.log")
        rows.append(r)
        print(f"{'ok' if r['ok'] else 'FAILED'}  {r['passed']} passed, {r['failed']} failed, {r['seconds']:.0f} s")
    lint = []
    if not argv:
        for name, args in LINT.items():
            r = run(name, args, out / f"lint_{name}.log")
            lint.append(r)
            print(f"{name}: {'clean' if r['ok'] else 'problems, see ' + r['log']}")
    ok = all(r["ok"] for r in rows + lint)
    summary = {"started_utc": started.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
               "all_passed": ok, "suites": rows, "lint": lint,
               "checks_passed": sum(r["passed"] for r in rows), "checks_failed": sum(r["failed"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    md = [f"# Test suites ({summary['started_utc']} UTC, Python {summary['python']})", "",
          f"**{'All passed' if ok else 'FAILURES'}**: {summary['checks_passed']} checks passed, "
          f"{summary['checks_failed']} failed across {len(rows)} suites.", "",
          "| Suite | Result | Passed | Failed | Time |", "|---|---|---:|---:|---:|"]
    md += [f"| {r['name']} | {'ok' if r['ok'] else '**FAILED**'} | {r['passed']} | {r['failed']} | "
           f"{r['seconds']:.0f} s |" for r in rows]
    md += [f"| lint: {r['name']} | {'clean' if r['ok'] else '**problems**'} | | | {r['seconds']:.0f} s |" for r in lint]
    for r in rows:
        if r["failures"] or r["errors"]:
            md += ["", f"## {r['name']}", *[f"- FAIL {f}" for f in r["failures"]], *[f"- {e.strip()}" for e in r["errors"]]]
    (out / "summary.md").write_text("\n".join(md) + "\n", "utf-8")
    from solarflare.products.model_tests import write_index

    write_index(out.parent)
    print(f"\n{'All passed' if ok else 'FAILURES'}; results in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
