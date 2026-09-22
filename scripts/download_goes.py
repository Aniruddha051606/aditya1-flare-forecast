"""Download the GOES-18 X-ray files the pipeline scores against (flare truth).

    python scripts/download_goes.py                  # into goes_dir from config/project.toml
    python scripts/download_goes.py --years 2024 2025 2026 --out E:/AdityaData/goes

Source: NOAA NCEI, public, no account. Two science products of GOES-18 XRS:

* ``sci_xrsf-l2-flsum_g18_s..._e<date>_v*.nc`` -- the flare summary (start, peak,
  end and class of every flare), one file for the whole mission, re-issued as it
  grows. Only the newest issue is fetched; older issues in the folder are left.
* ``sci_xrsf-l2-avg1m_g18_y<year>_v*.nc`` -- 1-minute XRS-A/B flux, one file per
  year (about 20-35 MB). The current year's file is re-issued as it grows, so it
  is fetched again whenever NOAA's copy is newer (by size) than the local one.

Files are written as ``*.part`` and renamed when complete, so an interrupted
download never leaves a half file that the pipeline would read.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data"
FLSUM = f"{BASE}/xrsf-l2-flsum_science/"
AVG1M = f"{BASE}/xrsf-l2-avg1m_science/"
TIMEOUT_S = 120


def listing(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/]+\.nc)"', html)))


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except OSError:
        return None


def fetch(url: str, dest: Path) -> int:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(part, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    part.replace(dest)
    return dest.stat().st_size


def main() -> int:
    from solarflare.settings import load_settings

    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", default=str(S.goes_dir), help="default: goes_dir in config/project.toml")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2024, date.today().year + 1)),
                    help="years of 1-minute flux (default: 2024 to this year)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, Path]] = []
    flsum = [f for f in listing(FLSUM) if "_g18_" in f]
    if flsum:
        newest = max(flsum, key=lambda f: re.search(r"_e(\d{8})_", f).group(1) if re.search(r"_e(\d{8})_", f) else "")
        if not (out / newest).exists():
            todo.append((FLSUM + newest, out / newest))
        else:
            print(f"{newest}: already here")
    avg = [f for f in listing(AVG1M) if "_g18_y" in f]
    for y in args.years:
        names = [f for f in avg if f"_y{y}_" in f]
        if not names:
            print(f"no 1-minute file for {y} on NOAA's server")
            continue
        name = max(names)                              # highest version
        dest = out / name
        if dest.exists():
            size = remote_size(AVG1M + name)
            if size is None or size == dest.stat().st_size:
                print(f"{name}: already here")
                continue
            print(f"{name}: NOAA has a newer issue ({size:,} vs {dest.stat().st_size:,} bytes)")
        todo.append((AVG1M + name, dest))

    if not todo:
        print("GOES files are up to date.")
        return 0
    for url, dest in todo:
        if args.dry_run:
            print(f"would download {url}")
            continue
        n = fetch(url, dest)
        print(f"downloaded {dest.name} ({n / 1e6:.1f} MB)")
    print(f"GOES folder: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
