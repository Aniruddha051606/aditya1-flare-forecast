"""Paper step 4: run every test suite and the linters, and record the outcome.

    python scripts/paper/test_status.py            # runs scripts/run_tests.py first
    python scripts/paper/test_status.py --no-run   # only re-render the last run

Writes paper_results/03_test_status.md from outputs/tests/suites/summary.json
and the suite logs (warnings are counted from the logs).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from common import ROOT, stamp, write_text  # noqa: E402  (sets sys.path)
from solarflare.settings import load_settings

CMD = [sys.executable, "scripts/run_tests.py"]


def main(argv: list[str]) -> int:
    if "--no-run" not in argv:
        subprocess.run(CMD, cwd=ROOT, check=False)
    folder = load_settings().tests / "suites"
    s = json.loads((folder / "summary.json").read_text("utf-8"))
    warn, where = {}, []
    for r in s["suites"]:
        group = ""
        warn[r["name"]] = 0
        for line in (folder / r["log"]).read_text("utf-8", errors="replace").splitlines():
            if re.match(r"^test_\w+:$", line):
                group = line[:-1]
            if re.search(r"Warning\b", line):
                warn[r["name"]] += 1
                where.append(f"- {r['name']} / {group}: `{line.strip()[:150]}`")
    total_s = sum(r["seconds"] for r in s["suites"] + s["lint"])
    p = stamp()
    L = ["# Test status", "",
         f"Command: `python scripts/run_tests.py` (each suite in its own process, as in CI), run "
         f"{s['started_utc']} UTC on Python {s['python']}; rendered {p['generated_utc']} UTC, commit {p['git_commit']}.",
         "", f"**{'All passed' if s['all_passed'] else 'FAILURES'}**: {s['checks_passed']} checks passed, "
         f"{s['checks_failed']} failed, across {len(s['suites'])} suites; total run time {total_s:.0f} s.", "",
         "| Suite | Result | Passed | Failed | Warnings in log | Time |", "|---|---|---:|---:|---:|---:|"]
    L += [f"| {r['name']}{' (SUIT code, Paper 2: not used by Paper 1)' if r['name'].startswith('test_suit') else ''} | "
          f"{'ok' if r['ok'] else '**FAILED**'} | {r['passed']} | {r['failed']} | {warn[r['name']]} | "
          f"{r['seconds']:.0f} s |" for r in s["suites"]]
    L += [f"| lint: {r['name']} | {'clean' if r['ok'] else '**problems**'} | | | | {r['seconds']:.0f} s |" for r in s["lint"]]
    bad = [r for r in s["suites"] if r["failures"] or r["errors"]]
    L += ["", "## Failures", ""] + ([f"- {r['name']}: {', '.join(r['failures'] + r['errors'])}" for r in bad]
                                   or ["None."])
    L += ["", "## Warnings", "", *(where or ["None."])]
    L += ["", "No suite skips tests; each group runs every check it defines.",
          "", f"Logs: `{folder.relative_to(ROOT).as_posix()}/`."]
    write_text("03_test_status.md", "\n".join(L))
    return 0 if s["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
