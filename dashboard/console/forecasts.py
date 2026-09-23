"""Forecasts screen: sealed day-ahead forecasts, and the blind test against GOES.

A day forecast is issued once, from the last hour of SoLEXS data before that UTC
day, by ``python -m solarflare day-forecast --day YYYY-MM-DD``. It is written
with a SHA-256 of its content and never re-issued, so what was claimed before the
day cannot drift afterwards. This screen lists them, and shows the model-vs-actual
pictures the forward check draws once the day's data and GOES arrive.
"""

from __future__ import annotations

import os
import tkinter as tk
from datetime import UTC, datetime, timedelta
from tkinter import ttk

from .common import (AMBER, FAINT, GROUND, LINE, MUTED, OUTPUTS, PANEL, S, SMALL, TEAL, TEXT, UI, UI_B,
                     flat_button, read_json)

FORECASTS = S.dayahead / "forecasts"
#: Where scripts/forward_check writes its model-vs-actual PNGs.
PICTURES = OUTPUTS / "_dev" / "forward_check" / "png"


class ForecastsTab(tk.Frame):
    def __init__(self, parent, run_job=None):
        super().__init__(parent, bg=GROUND)
        self.run_job = run_job
        self._stamp = "unset"
        self._img = None

        head = tk.Frame(self, bg=GROUND)
        head.pack(fill="x", pady=(0, 6))
        tk.Label(head, text="Sealed day-ahead forecasts", bg=GROUND, fg=TEXT, font=UI_B).pack(side="left")
        flat_button(head, "Open folder", self._open_folder, MUTED).pack(side="right")
        self.day = tk.StringVar(value=(datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d"))
        flat_button(head, "Issue forecast", self._issue, TEAL).pack(side="right", padx=(0, 6))
        e = tk.Entry(head, textvariable=self.day, width=11, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=UI, justify="center")
        e.pack(side="right", padx=(0, 6))
        tk.Label(head, text="for UTC day", bg=GROUND, fg=MUTED, font=UI).pack(side="right", padx=(0, 6))

        box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        box.pack(fill="x")
        self.table = tk.Frame(box, bg=PANEL)
        self.table.pack(fill="x", padx=12, pady=10)
        for c, w in enumerate((0, 0, 0, 0, 1)):
            self.table.grid_columnconfigure(c, weight=w)
        self.rows: list[list[tk.Label]] = []
        self.empty = tk.Label(box, text="", bg=PANEL, fg=FAINT, font=UI, anchor="w", justify="left",
                              wraplength=880)

        pics = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        pics.pack(fill="both", expand=True, pady=(10, 0))
        bar = tk.Frame(pics, bg=PANEL)
        bar.pack(fill="x", padx=12, pady=(9, 0))
        tk.Label(bar, text="MODEL VS ACTUAL  ·  scripts/forward_check", bg=PANEL, fg=MUTED,
                 font=SMALL).pack(side="left")
        self.pick = ttk.Combobox(bar, width=30, state="readonly", style="Run.TCombobox", font=UI)
        self.pick.pack(side="right")
        self.pick.bind("<<ComboboxSelected>>", lambda e: self._show_picture())
        flat_button(bar, "Open picture", self._open_picture, MUTED).pack(side="right", padx=(0, 6))
        self.pic = tk.Label(pics, bg=PANEL, fg=FAINT, font=UI, text="", anchor="center")
        self.pic.pack(fill="both", expand=True, padx=12, pady=(6, 10))

    # ---- actions -------------------------------------------------------------------
    def _issue(self) -> None:
        day = self.day.get().strip()
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            self.empty.config(text=f"'{day}' is not a date (use YYYY-MM-DD).")
            return
        if self.run_job:
            cmd = ["PY", "-u", "-m", "solarflare", "day-forecast", "--day", day]
            extra = OUTPUTS / "_dev" / "forward_check" / "cache"
            if extra.is_dir():                     # days uploaded after the last pipeline run
                cmd += ["--extra-cache", str(extra)]
            self.run_job(f"Day forecast {day}", [{"label": f"day-forecast {day}", "cmd": cmd}])

    def _open_folder(self) -> None:
        FORECASTS.mkdir(parents=True, exist_ok=True)
        os.startfile(FORECASTS)

    def _open_picture(self) -> None:
        p = PICTURES / self.pick.get() if self.pick.get() else None
        if p and p.exists():
            os.startfile(p)

    # ---- refresh -------------------------------------------------------------------
    def refresh(self) -> None:
        files = sorted(FORECASTS.glob("forecast_*.json")) if FORECASTS.is_dir() else []
        pics = sorted(PICTURES.glob("*.png")) if PICTURES.is_dir() else []
        stamp = (tuple((f.name, f.stat().st_mtime) for f in files), tuple(p.name for p in pics))
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._table([read_json(f) for f in files])
        names = [p.name for p in pics]
        if list(self.pick["values"]) != names:
            self.pick["values"] = names
            if names and self.pick.get() not in names:
                self.pick.set(names[-1])
                self._show_picture()
        if not names:
            self.pic.config(text="No pictures yet. After a day's SoLEXS and HEL1OS are in, run\n"
                                 "  python scripts/forward_check/predict_after_test.py\n"
                                 "  python scripts/forward_check/plot_days.py", image="")

    def _table(self, rows: list[dict]) -> None:
        for lab in [w for r in self.rows for w in r]:
            lab.destroy()
        self.rows = []
        if not rows:
            self.empty.config(text="No sealed forecasts yet. Pick a UTC day above and press Issue forecast; it "
                                   "needs the day models from the dayahead stage and SoLEXS up to the day before.")
            self.empty.pack(fill="x", padx=12, pady=(0, 10))
            return
        self.empty.pack_forget()
        heads = ("UTC day", "≥ C1", "≥ M1", "lead", "issued · model · seal")
        for c, h in enumerate(heads):
            tk.Label(self.table, text=h.upper(), bg=PANEL, fg=MUTED, font=SMALL,
                     anchor="w").grid(row=0, column=c, sticky="w", padx=(0, 14), pady=(0, 4))
        for r, d in enumerate(sorted([x for x in rows if x], key=lambda x: x.get("forecast_for", "")), start=1):
            day = (d.get("forecast_for") or "")[:10]
            c_, m_ = d.get("C", {}), d.get("M", {})
            cells = [
                (day, TEXT, UI_B),
                (f"{100 * c_.get('probability', 0):.0f}%", AMBER if c_.get("probability", 0) >= 0.5 else TEXT, UI_B),
                (f"{100 * m_.get('probability', 0):.0f}%", AMBER if m_.get("probability", 0) >= 0.3 else TEXT, UI_B),
                (f"{d.get('lead_hours', 0):.0f} h", MUTED, UI),
                (f"{(d.get('issued_utc') or '')[:16]} UTC · {c_.get('model', '')}/{m_.get('model', '')} · "
                 f"{(d.get('sha256') or '')[:8]}", FAINT, SMALL),
            ]
            row = []
            for c, (text, fg, font) in enumerate(cells):
                lab = tk.Label(self.table, text=text, bg=PANEL, fg=fg, font=font, anchor="w")
                lab.grid(row=r, column=c, sticky="w", padx=(0, 14))
                row.append(lab)
            self.rows.append(row)
        skill = [x.get("C", {}).get("test_AUC") for x in rows if x and x.get("C", {}).get("test_AUC")]
        if skill:
            lab = tk.Label(self.table, text=f"Day models: C test AUC {skill[-1]}, from the dayahead stage. "
                                            "A sealed forecast is never re-issued.",
                           bg=PANEL, fg=FAINT, font=SMALL, anchor="w")
            lab.grid(row=len(self.rows) + 1, column=0, columnspan=5, sticky="w", pady=(6, 0))
            self.rows.append([lab])

    def _show_picture(self) -> None:
        name = self.pick.get()
        p = PICTURES / name if name else None
        if not p or not p.exists():
            return
        try:
            img = tk.PhotoImage(file=str(p))
            w = max(self.pic.winfo_width(), 400)
            factor = max(1, int(img.width() / max(w - 20, 200)) + 1)
            self._img = img.subsample(factor, factor)
            self.pic.config(image=self._img, text="")
        except Exception as exc:                          # noqa: BLE001 - a bad PNG must not kill the console
            self.pic.config(image="", text=f"cannot show {name}: {exc}")
