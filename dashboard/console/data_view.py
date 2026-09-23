"""Data screen: where the mission data lives, what is there, and what is missing.

Every command reads ``data_root`` from config/project.toml (or SOLARFLARE_DATA_ROOT),
so this screen shows that folder, the drives holding it, and a month-by-month
coverage strip for each instrument -- the quickest way to see a gap such as
HEL1OS 2024-07 to 2026-05, or to notice that a drive is no longer there.

Scanning counts files only (never opens them) and runs on a worker thread every
SCAN_EVERY_S seconds, so the console keeps its 1 s tick.
"""

from __future__ import annotations

import ctypes
import re
import threading
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, PANEL, RED, S, SMALL, STEEL, TEAL, TEXT,
                     UI_B, Tile, blend, flat_button, read_json)

SCAN_EVERY_S = 60.0
#: The mission starts here; the strip runs from this month to the present.
FIRST_MONTH = (2024, 2)
SLX = re.compile(r"AL1_SLX_L1_(\d{8})_v", re.I)
HLS = re.compile(r"HLS_(\d{8})_\d{6}_\d+sec", re.I)


def free_gb(path: Path) -> float | None:
    """Free space on the drive holding ``path`` (GB), or None if it is gone."""
    try:
        p = path if path.exists() else path.anchor or path
        free = ctypes.c_ulonglong(0)
        if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(str(p), None, None, ctypes.byref(free)):
            return None
        return free.value / 1e9
    except Exception:                                    # noqa: BLE001 - a missing drive is normal here
        return None


def months(first=FIRST_MONTH) -> list[tuple[int, int]]:
    now = datetime.now(UTC)
    y, m = first
    out = []
    while (y, m) <= (now.year, now.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def scan() -> dict:
    """File counts per instrument, per month. Names only: nothing is opened."""
    out: dict = {"scanned_unix": time.time(), "data_root": S.data_root, "exists": S.data_root.is_dir()}
    slx: dict[str, set] = {}
    hls: dict[str, set] = {}
    if out["exists"]:
        for f in S.data_root.rglob("AL1_SLX_L1_*.zip"):
            m = SLX.search(f.name)
            if m:
                slx.setdefault(m.group(1)[:6], set()).add(m.group(1))
        for f in S.data_root.rglob("HLS_*.zip"):
            m = HLS.search(f.name)
            if m:
                hls.setdefault(m.group(1)[:6], set()).add(m.group(1))
    ext = S.hel1os_extracted
    hls_ext: dict[str, set] = {}
    if ext.is_dir():
        for d in ext.glob("*/*/*/HLS_*"):
            m = HLS.search(d.name)
            if m:
                hls_ext.setdefault(m.group(1)[:6], set()).add(m.group(1))
    out["solexs_days"] = {k: len(v) for k, v in slx.items()}
    out["hel1os_days"] = {k: len(v) for k, v in hls.items()}
    out["hel1os_extracted_days"] = {k: len(v) for k, v in hls_ext.items()}

    goes = sorted(S.goes_dir.glob("*xrsf-l2-flsum*.nc")) if S.goes_dir.is_dir() else []
    avg = sorted(S.goes_dir.glob("*xrsf-l2-avg1m*.nc")) if S.goes_dir.is_dir() else []
    end = ""
    for f in goes:
        m = re.search(r"_e(\d{8})_", f.name)
        end = max(end, m.group(1) if m else "")
    out["goes"] = {"flare_list_to": end, "flux_years": len(avg), "n": len(goes) + len(avg)}
    sharp = sorted(S.sharp_dir.glob("sharp_*.csv")) if S.sharp_dir.is_dir() else []
    out["sharp"] = {"months": len(sharp), "last": sharp[-1].stem.replace("sharp_", "") if sharp else ""}

    man = read_json(S.cache / "manifest.json") or []
    ok = [e for e in man if isinstance(e, dict) and e.get("status") == "ok"]
    size = sum(f.stat().st_size for f in S.cache.glob("*.npz")) if S.cache.is_dir() else 0
    # With [data] cache_is_source the extracted files are deleted on purpose once
    # they are cached (scripts/ingest_batch.py), so a missing raw file is normal
    # and the .npz is what must be watched instead.
    kept = bool(S.data.get("cache_is_source", False))
    if kept:
        gone = sum(1 for e in ok if not (S.cache / f"{e.get('key')}.npz").exists())
        carried = sum(1 for e in ok if not Path(e.get("source", {}).get("path", "")).exists())
    else:
        gone = sum(1 for e in ok if not Path(e.get("source", {}).get("path", "")).exists())
        carried = 0
    out["cache"] = {"entries": len(ok), "sources_missing": gone, "gb": size / 1e9,
                    "from_cache_alone": carried, "cache_is_source": kept}
    out["free"] = {"data": free_gb(S.data_root), "cache": free_gb(S.cache), "outputs": free_gb(S.outputs)}
    return out


class DataTab(tk.Frame):
    """Where the data is, how much of it, and what is missing."""

    def __init__(self, parent, run_job=None):
        super().__init__(parent, bg=GROUND)
        self.run_job = run_job
        self.info: dict = {}
        self._scanning = False
        self._last = 0.0

        head = tk.Frame(self, bg=GROUND)
        head.pack(fill="x", pady=(0, 6))
        self.where = tk.Label(head, text="", bg=GROUND, fg=TEXT, font=UI_B, anchor="w", justify="left")
        self.where.pack(side="left")
        flat_button(head, "Rescan", self.rescan, MUTED).pack(side="right")
        flat_button(head, "Open data folder", self._open, MUTED).pack(side="right", padx=(0, 6))
        flat_button(head, "Download SHARP", lambda: self._job("Download SHARP", "scripts/download_sharp.py"),
                    STEEL).pack(side="right", padx=(0, 6))
        flat_button(head, "Download GOES", lambda: self._job("Download GOES", "scripts/download_goes.py"),
                    STEEL).pack(side="right", padx=(0, 6))
        # Turns the zips into cache entries one at a time, deleting each extracted
        # product once it is cached. Resumable: pressing it again fills what is left.
        flat_button(head, "Ingest HEL1OS", lambda: self._job("Ingest HEL1OS", "scripts/ingest_batch.py"),
                    GREEN).pack(side="right", padx=(0, 6))

        self.banner = tk.Label(self, text="", bg=PANEL, fg=RED, font=UI_B, anchor="w", justify="left",
                               wraplength=900)

        tiles = tk.Frame(self, bg=GROUND)
        tiles.pack(fill="x")
        self.tiles = {}
        for i, (key, label) in enumerate((("solexs", "SoLEXS days"), ("hel1os", "HEL1OS days"),
                                          ("both", "Days with both"), ("cache", "Preprocessing cache"),
                                          ("truth", "GOES truth"), ("disk", "Free space"))):
            t = Tile(tiles, label)
            t.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 8, 0), pady=(0, 8))
            tiles.grid_columnconfigure(i % 3, weight=1, uniform="d")
            self.tiles[key] = t

        box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        box.pack(fill="both", expand=True)
        tk.Label(box, text="COVERAGE BY MONTH  ·  one cell per month, brighter = more days", bg=PANEL, fg=MUTED,
                 font=SMALL).pack(anchor="w", padx=12, pady=(9, 2))
        self.canvas = tk.Canvas(box, bg=PANEL, highlightthickness=0, height=150)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda e: self.hint.config(text=""))
        self.hint = tk.Label(box, text="", bg=PANEL, fg=MUTED, font=SMALL, anchor="w")
        self.hint.pack(fill="x", padx=12, pady=(0, 8))
        self.foot = tk.Label(self, text="", bg=GROUND, fg=FAINT, font=SMALL, anchor="w", justify="left",
                             wraplength=900)
        self.foot.pack(fill="x", pady=(6, 0))
        self._cells: list[tuple[int, int, int, int, str]] = []

    # ---- scanning ------------------------------------------------------------------
    def rescan(self) -> None:
        if self._scanning:
            return
        self._scanning = True

        def work():
            try:
                info = scan()
            except Exception as exc:                     # noqa: BLE001 - never kill the console
                info = {"error": f"{type(exc).__name__}: {exc}", "scanned_unix": time.time()}
            self.info = info
            self._scanning = False

        threading.Thread(target=work, daemon=True).start()

    def _open(self) -> None:
        import os
        p = S.data_root if S.data_root.is_dir() else S.root
        os.startfile(p)

    def _job(self, name: str, script: str) -> None:
        if self.run_job:
            self.run_job(name, [{"label": name.lower(), "cmd": ["PY", "-u", script]}])

    # ---- refresh -------------------------------------------------------------------
    def refresh(self) -> None:
        if time.time() - self._last > SCAN_EVERY_S:
            self._last = time.time()
            self.rescan()
        i = self.info
        if not i:
            self.where.config(text=f"Scanning {S.data_root} …")
            return
        self.where.config(text=f"data_root: {i.get('data_root', S.data_root)}"
                               + ("" if i.get("exists") else "   (FOLDER NOT FOUND)"))
        self._banner(i)
        self._tiles(i)
        self._strip(i)
        age = time.time() - i.get("scanned_unix", 0)
        self.foot.config(text=f"Scanned {age:.0f} s ago · counts come from file names only. "
                              f"goes_dir {S.goes_dir} · sharp_dir {S.sharp_dir} · cache {S.cache}")

    def _banner(self, i: dict) -> None:
        msg, col = "", RED
        if i.get("error"):
            msg = f"Scan failed: {i['error']}"
        elif not i.get("exists"):
            msg = (f"The data folder {i.get('data_root')} does not exist. Set data_root in config/project.toml "
                   "or SOLARFLARE_DATA_ROOT; every command refuses to run until then.")
        elif not (i.get("solexs_days") or i.get("hel1os_days")):
            msg = ("No SoLEXS or HEL1OS files under the data folder. Downloads go there; the pipeline refuses "
                   "to train on an empty archive.")
        elif i.get("cache", {}).get("sources_missing"):
            c = i["cache"]
            g, n = c["sources_missing"], c["entries"]
            msg, col = ((f"{g} of {n} cache files are missing. The cache is the data of record here "
                         "(cache_is_source), so the pipeline stops above 20%: restore the backup, or rebuild "
                         "those days from the zips with scripts/ingest_batch.py.")
                        if c.get("cache_is_source") else
                        (f"{g} of {n} files this cache was built from are missing. The pipeline stops above "
                         "20%: restore them, or point data_root at their new home."), AMBER)
        if msg:
            self.banner.config(text=msg, fg=col)
            self.banner.pack(fill="x", pady=(0, 8), ipady=6, before=self.tiles["solexs"].master)
        else:
            self.banner.pack_forget()

    def _tiles(self, i: dict) -> None:
        T = self.tiles
        slx, hls = i.get("solexs_days", {}), i.get("hel1os_days", {})
        ext = i.get("hel1os_extracted_days", {})
        n_slx, n_hls, n_ext = sum(slx.values()), sum(hls.values()), sum(ext.values())
        both = sum(min(slx.get(k, 0), max(hls.get(k, 0), ext.get(k, 0))) for k in set(slx) | set(hls) | set(ext))
        T["solexs"].set(f"{n_slx}", f"{len(slx)} months · zips under data_root", TEAL if n_slx else MUTED)
        T["hel1os"].set(f"{n_hls}", f"{n_ext} days extracted · the pipeline reads those",
                        TEAL if n_ext else (AMBER if n_hls else MUTED))
        T["both"].set(f"~{both}", "days where both instruments have files (per month)",
                      TEAL if both else MUTED)
        c = i.get("cache", {})
        T["cache"].set(f"{c.get('gb', 0):.2f} GB", f"{c.get('entries', 0)} products cached"
                       + (f" · {c['from_cache_alone']} read from the cache alone"
                          if c.get("from_cache_alone") else "")
                       + (f" · {c['sources_missing']} missing" if c.get("sources_missing") else ""),
                       AMBER if c.get("sources_missing") else (TEAL if c.get("entries") else MUTED))
        g = i.get("goes", {})
        sh = i.get("sharp", {})
        T["truth"].set(g.get("flare_list_to") or "--",
                       f"flare list to this date · {g.get('flux_years', 0)} year files · "
                       f"SHARP {sh.get('months', 0)} months", TEAL if g.get("n") else MUTED)
        fr = i.get("free", {})
        vals = [v for v in fr.values() if v is not None]
        low = min(vals) if vals else None
        T["disk"].set("--" if low is None else f"{low:.0f} GB",
                      " · ".join(f"{k} {v:.0f} GB" for k, v in fr.items() if v is not None),
                      RED if (low is not None and low < S.min_free_gb) else TEXT)

    def _strip(self, i: dict) -> None:
        c = self.canvas
        c.delete("all")
        self._cells = []
        ms = months()
        w = max(c.winfo_width(), 200)
        left = 58
        cw = max((w - left - 8) / max(len(ms), 1), 3.0)
        rows = (("SoLEXS", i.get("solexs_days", {}), TEAL),
                ("HEL1OS", i.get("hel1os_days", {}), STEEL),
                ("extracted", i.get("hel1os_extracted_days", {}), GREEN))
        for r, (name, data, col) in enumerate(rows):
            y = 16 + r * 34
            c.create_text(4, y + 9, text=name, anchor="w", fill=MUTED, font=SMALL)
            for k, (yy, mm) in enumerate(ms):
                key = f"{yy:04d}{mm:02d}"
                days = data.get(key, 0)
                days_in = 31 if mm in (1, 3, 5, 7, 8, 10, 12) else (28 if mm == 2 else 30)
                frac = min(days / days_in, 1.0)
                x0 = left + k * cw
                fill = blend(PANEL, col, 0.12 + 0.88 * frac) if days else PANEL
                c.create_rectangle(x0, y, x0 + max(cw - 1.5, 1), y + 18, fill=fill,
                                   outline=LINE if not days else "")
                self._cells.append((int(x0), int(y), int(x0 + cw), int(y + 18),
                                    f"{yy}-{mm:02d} · {name}: {days} day(s)"))
        y = 16 + len(rows) * 34
        for k, (yy, mm) in enumerate(ms):
            if mm in (1, 7):
                c.create_text(left + k * cw, y, text=f"{yy if mm == 1 else ''}{'' if mm == 1 else 'Jul'}",
                              anchor="nw", fill=FAINT, font=SMALL)
        gap = [f"{yy}-{mm:02d}" for yy, mm in ms
               if not i.get("hel1os_days", {}).get(f"{yy:04d}{mm:02d}") and i.get("solexs_days", {}).get(f"{yy:04d}{mm:02d}")]
        if gap:
            c.create_text(left, y + 16, text=f"months with SoLEXS but no HEL1OS: {len(gap)}"
                                             + (f"  ({gap[0]} … {gap[-1]})" if len(gap) > 1 else f"  ({gap[0]})"),
                          anchor="nw", fill=AMBER, font=SMALL)

    def _hover(self, ev) -> None:
        for x0, y0, x1, y1, text in self._cells:
            if x0 <= ev.x <= x1 and y0 <= ev.y <= y1:
                self.hint.config(text=text)
                return
        self.hint.config(text="")
