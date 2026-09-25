"""Turn downloaded zips into cache entries one at a time, freeing the space as it goes.

    python scripts/ingest_batch.py --dry-run          # what it would do
    python scripts/ingest_batch.py                    # extract -> cache -> verify -> delete extracted
    python scripts/ingest_batch.py --limit 20         # a few products, then stop
    python scripts/ingest_batch.py --keep-extracted   # cache only, delete nothing
    python scripts/ingest_batch.py --check            # is every cached product still readable?
    python scripts/ingest_batch.py --backup E:/cache-backup     # ingest, then copy the cache

For each HEL1OS product: extract the light curves (and their GTI and housekeeping),
build its cache entry, check the entry can be read back, then delete the extracted
files again. The zip is never touched, so anything the 20 s cache cannot answer
(sub-second timing, the hard X-ray spectra) can still be redone from it.

SoLEXS zips are cached directly -- the reader reads inside the zip -- so nothing
is extracted or deleted for them.

Why one at a time: extracting the whole HEL1OS archive needs about 139 GB;
extracting one product needs about 400 MB, and after caching it drops to a few
hundred kB. The work is resumable: a product already cached is skipped, so the
script can be stopped and re-run.

With ``[data] cache_is_source = true`` in config/project.toml (the default here),
a day whose extracted files are gone still trains, scores and appears in the
catalogue, because the cache holds everything the pipeline reads. That makes the
cache the data of record, so it is worth protecting: ``--check`` reports any hole
in it, ``--backup`` copies it elsewhere, and re-running this script rebuilds
whatever is missing straight from the zips.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.preprocess.cache import (FROZEN_MARKER, build_cache, cache_frozen, index_sources,  # noqa: E402
                                         load_cached)
from solarflare.settings import load_settings, model_config  # noqa: E402

#: What the pipeline reads from a HEL1OS product (scripts/unzip_archive.py's
#: "lightcurves" member set).
WANTED = re.compile(r"/(czt|cdte)/lightcurve_[a-z0-9]+\.fits$|/aux/gti[a-z0-9]+\.fits$|/aux/hk\.fits$",
                    re.I)
HLS_ZIP = re.compile(r"HLS_\d{8}_\d{6}_\d+sec_lev\d+_V\d+\.zip$", re.I)


def gb_free(path: Path) -> float:
    return shutil.disk_usage(path if path.exists() else path.anchor).free / 1e9


def check_cache(cache: Path) -> int:
    """Is every product the manifest claims actually there and readable?

    With ``cache_is_source`` the .npz files are the data of record, so a silent
    hole in them is the failure that matters. Exits non-zero if there is one."""
    import numpy as np

    man = cache / "manifest.json"
    if not man.exists():
        print(f"no manifest in {cache}: nothing cached yet")
        return 1
    try:
        entries = json.loads(man.read_text("utf-8"))
    except (ValueError, OSError) as exc:
        print(f"manifest unreadable: {exc}")
        return 1
    ok = [e for e in entries if e.get("status") == "ok"]
    missing, bad, bins = [], [], 0
    for e in ok:
        f = cache / f"{e.get('key')}.npz"
        if not f.exists():
            missing.append(e)
            continue
        try:
            with np.load(f) as z:
                bins += int(z["time_unix"].size)
        except (OSError, ValueError, KeyError):
            bad.append(e)
    kinds: dict[str, int] = {}
    deleted = 0
    for e in ok:
        src = e.get("source") or {}
        kinds[src.get("kind", "?")] = kinds.get(src.get("kind", "?"), 0) + 1
        if src.get("path") and not Path(src["path"]).exists():
            deleted += 1
    size = sum(f.stat().st_size for f in cache.glob("*.npz")) / 1e9
    print(f"{len(entries)} manifest entries, {len(ok)} usable "
          f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")
    print(f"{bins} time bins, {size:.2f} GB in {cache}")
    print(f"{deleted} of {len(ok)} have no raw file left (cache_is_source carries these)")
    if missing or bad:
        for e in (missing + bad)[:5]:
            print(f"  ! {(e.get('source') or {}).get('path') or e.get('key')}")
        print(f"{len(missing)} missing, {len(bad)} unreadable. Rebuild just those from the zips:"
              "\n    python scripts/ingest_batch.py")
        return 1
    print("every entry is present and readable")
    return 0


def backup_cache(cache: Path, dest: Path) -> int:
    """Copy the cache somewhere else and verify the copy.

    The cache is small (a few hundred MB for the whole archive) and, once the
    extracted files are deleted, it is the only copy of what the model reads.
    Keep it on a different drive than the zips."""
    if not (cache / "manifest.json").exists():
        print(f"nothing to back up: no manifest in {cache}")
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    marker = [cache / FROZEN_MARKER] if cache_frozen(cache) else []
    for f in [cache / "manifest.json", *marker, *sorted(cache.glob("*.npz"))]:
        out = dest / f.name
        if out.exists() and out.stat().st_size == f.stat().st_size and out.stat().st_mtime_ns >= f.stat().st_mtime_ns:
            skipped += 1
            continue
        shutil.copy2(f, out)
        copied += 1
    here = {f.name for f in cache.glob("*.npz")}
    there = {f.name for f in dest.glob("*.npz")}
    size = sum((dest / n).stat().st_size for n in there) / 1e9
    print(f"{copied} file(s) copied, {skipped} already current; "
          f"{len(there)} cache files ({size:.2f} GB) in {dest}")
    if here - there:
        print(f"backup incomplete: {len(here - there)} file(s) did not copy")
        return 1
    if dest.resolve().drive.lower() == cache.resolve().drive.lower():
        print(f"note: the backup is on the same drive ({dest.drive}) as the cache; "
              "a drive failure would take both")
    return 0


def this_product(entries: list[dict]) -> list[dict]:
    """The usable entries this call built. build_cache also returns every earlier
    product it carries from the cache (``source_deleted``); reading all of those
    back after each product would make the ingest quadratic -- about 3.7 million
    cache reads over the whole HEL1OS archive instead of 2,700."""
    return [e for e in entries if e.get("status") == "ok" and not e.get("source_deleted")]


def product_in_zip(zip_path: Path) -> tuple[str | None, bool]:
    """(the product's folder inside its zip, whether it has light curves).

    Some older product versions ship without light curves -- only housekeeping,
    GTIs and the photon event list; 34 of the 2,719 zips, each with a newer
    version of the same observation that has them. Nothing in those is for the
    cache, and they are not failures."""
    name = zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        own = [m for m in z.namelist() if f"/{name}/" in "/" + m]
    if not own:
        return None, False
    s = "/" + own[0]
    return s[1:s.index(f"/{name}/") + len(name) + 1], any("/lightcurve_" in m for m in own)


def extract_lightcurves(zip_path: Path, dest: Path) -> Path | None:
    """Extract just the light curves of one HEL1OS zip; returns the product folder.

    The folder comes from the zip, not from the product name: PRADAN files a
    product that starts just before midnight (e.g. 23:59:50) under the next
    day's date folder, and deriving the path from the name skipped those."""
    name = zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if WANTED.search(m)]
        own = [m for m in members if f"/{name}/" in "/" + m]
        if not any("/lightcurve_" in m for m in own):
            return None
        for m in members:
            z.extract(m, dest)
    rel = ("/" + own[0])[1:("/" + own[0]).index(f"/{name}/") + len(name) + 1]
    out = dest / rel
    return out if out.is_dir() else None


def main() -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data-root", default=str(S.data_root))
    ap.add_argument("--dest", default=str(S.hel1os_extracted), help="where products are extracted")
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many products (0 = all)")
    ap.add_argument("--keep-extracted", action="store_true", help="cache only; delete nothing")
    ap.add_argument("--min-free-gb", type=float, default=S.min_free_gb,
                    help="stop before the disk falls below this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-failed-pct", type=float, default=0.0,
                    help="still exit 0 if at most this %% of products could not be cached (default 0: any failure "
                         "is an error). For unattended chains, where one bad zip should not stop the rest.")
    ap.add_argument("--check", action="store_true",
                    help="report whether every cached product is present and readable, then stop")
    ap.add_argument("--backup", metavar="DIR",
                    help="copy the cache there afterwards (or on its own, with --check)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    root, dest, cache = Path(args.data_root), Path(args.dest), Path(args.cache_dir)
    if args.check:
        rc = check_cache(cache)
        return backup_cache(cache, Path(args.backup)) or rc if args.backup else rc
    if cache_frozen(cache):
        print(f"{cache} is frozen ({cache / FROZEN_MARKER}): it holds the data the final model was trained "
              "and tested on, so nothing is ingested into it. Ingest new days into their own cache with "
              "--cache-dir; restore a damaged frozen cache from its backup (--check lists holes).")
        return 2
    cfg = model_config(S).pre
    zips = sorted(p for p in root.rglob("*.zip") if HLS_ZIP.search(p.name))
    if not zips:
        print(f"no HEL1OS zips under {root}")
    # Which products the cache already holds, by product folder name, so one that
    # was cached and then deleted is not ingested twice. An entry whose .npz has
    # since gone counts as missing, so re-running this script repairs the hole
    # instead of skipping past it.
    cached_names: set[str] = set()
    holes = 0
    man = cache / "manifest.json"
    if man.exists():
        try:
            for e in json.loads(man.read_text("utf-8")):
                p = (e.get("source") or {}).get("path", "")
                if not p:
                    continue
                if e.get("status") == "empty":
                    cached_names.add(Path(p).name)   # nothing usable inside; no .npz to lose
                elif e.get("status") == "ok":
                    if (cache / f"{e.get('key')}.npz").exists():
                        cached_names.add(Path(p).name)
                    else:
                        holes += 1
        except (ValueError, OSError):
            pass

    todo = [z for z in zips if z.stem not in cached_names]
    print(f"{len(zips)} HEL1OS zips, {len(zips) - len(todo)} already cached, {len(todo)} to do")
    if holes:
        print(f"  ({holes} cache file(s) listed in the manifest are missing and will be rebuilt)")
    if args.limit:
        todo = todo[:args.limit]
    done = failed = no_lc = 0
    freed = 0.0
    t0 = time.time()
    for i, z in enumerate(todo, 1):
        if gb_free(cache) < args.min_free_gb:
            print(f"stopping: only {gb_free(cache):.0f} GB free, floor is {args.min_free_gb:g} GB")
            break
        if args.dry_run:
            print(f"[{i}/{len(todo)}] would ingest {z.name}")
            continue
        try:
            folder, has_lc = product_in_zip(z)
            if folder and not has_lc:
                # an event-list-only version: nothing to cache. Remove what an
                # earlier run of this script extracted for it, and only that.
                left = dest / folder
                if left.is_dir() and all(WANTED.search("/" + f.relative_to(dest).as_posix())
                                         for f in left.rglob("*") if f.is_file()):
                    shutil.rmtree(left, ignore_errors=True)
                print(f"[{i}/{len(todo)}] {z.name}: no light curves in this version (event list only)")
                no_lc += 1
                continue
            prod = extract_lightcurves(z, dest)
            if prod is None:
                print(f"[{i}/{len(todo)}] {z.name}: product not found inside the zip, skipped")
                failed += 1
                continue
            size = sum(f.stat().st_size for f in prod.rglob("*") if f.is_file())
            sources = [s for s in index_sources([prod]) if s.kind == "hel1os"]
            entries = build_cache(sources, cfg, cache, workers=1, verbose=False)
            ok = this_product(entries)
            # read the entry back before anything is deleted
            good = bool(ok) and all((cache / f"{e['key']}.npz").exists() for e in ok)
            if good:
                load_cached(ok, cache, "hel1os")
            if not good:
                print(f"[{i}/{len(todo)}] {z.name}: cache entry missing or unreadable, extracted files kept")
                failed += 1
                continue
            if not args.keep_extracted:
                shutil.rmtree(prod, ignore_errors=True)
                freed += size / 1e9
            done += 1
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / max(i, 1)
                print(f"[{i}/{len(todo)}] {done} cached, {failed} skipped, {freed:.1f} GB freed, "
                      f"{rate:.1f} s/product, {gb_free(cache):.0f} GB free", flush=True)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            print(f"[{i}/{len(todo)}] {z.name}: {type(exc).__name__}: {exc}")
            failed += 1
    size = sum(f.stat().st_size for f in cache.glob("*.npz")) / 1e9 if cache.is_dir() else 0.0
    print(f"\n{done} product(s) cached, {failed} failed, {no_lc} without light curves (event lists only); "
          f"{freed:.1f} GB of extracted files removed; cache now {size:.2f} GB in {cache}")
    print("SoLEXS zips need no extraction: python -m solarflare cache reads them in place.")
    if args.backup and not args.dry_run:
        print()
        backup_cache(cache, Path(args.backup))
    attempted = done + failed
    if failed and 100.0 * failed / max(attempted, 1) <= args.max_failed_pct:
        print(f"{failed} of {attempted} product(s) not cached: within --max-failed-pct {args.max_failed_pct:g}")
        return 0
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
