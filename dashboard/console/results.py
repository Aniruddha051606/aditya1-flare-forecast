"""Results screen: the headline numbers, read straight from the reports.

Every figure here comes from a JSON file a pipeline stage wrote -- nothing is
typed in -- and a result whose stage has not run yet says which stage it is
waiting for. The screen redraws only when one of those files changes.

  tiles     the two alerts, flare-now, flare-soon, flux now, peak of a rising flare
  chart     the >= C1 alert against simple baselines at the same false-alarm rate
  chart     reliability: do the calibrated probabilities mean what they say
  verdicts  SHARP, HEL1OS (alerts and peak size), calibration, catalogue, day-ahead
"""

from __future__ import annotations

import os
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, OUTPUTS, PANEL, S, SMALL, STEEL, TEAL, TEXT,
                     UI, UI_B, Tile, flat_button, muted_legend, read_json, style_axes)

FILES = {
    "rules": S.alerts / "alert_rules.json",
    "alerts": S.alerts / "leadtime_summary.json",
    "eval": S.model_dir / "reports" / "evaluation.json",
    "cal": S.model_dir / "reports" / "calibration.json",
    "sharp": S.ablations / "sharp" / "decision.json",
    "hel1os": S.ablations / "hel1os" / "hel1os_value.json",
    "catalog": S.catalog / "catalog_summary.json",
    "dayahead": S.dayahead / "dayahead_summary.json",
}
C_METHODS = (("model", "network"), ("hope", "HOPE hot onset"), ("trend", "trend"), ("current", "flux now"),
             ("rule", "rise rule"), ("rule_C_level", "rule ≥ C1"))


def pct(v) -> str:
    return "--" if v is None else f"{100 * float(v):.0f}%"


def ci(d: dict, key: str, nd: int = 2) -> str:
    """" [lo, hi]" for a metric that carries a 95% interval, else nothing."""
    c = (d or {}).get(f"{key}_ci95")
    return f" [{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if c else ""


def fb(d: dict) -> str:
    """Frequency bias, flagged when the score is flattered by overforecasting.

    A TSS means little without it (Leka et al. 2019): a system that raises far
    more alarms than there are flares can score well while being unusable."""
    v = (d or {}).get("FB")
    if v is None:
        return ""
    mark = "" if 0.7 <= v <= 1.5 else ("  over-forecasting" if v > 1.5 else "  under-forecasting")
    return f" · FB {v:.2f}{mark}"


class ResultsTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=GROUND)
        self._stamp = "unset"
        head = tk.Frame(self, bg=GROUND)
        head.pack(fill="x", pady=(0, 8))
        self.caption = tk.Label(head, text="", bg=GROUND, fg=MUTED, font=UI, anchor="w", justify="left")
        self.caption.pack(side="left")
        flat_button(head, "Open RESULTS.md", self._open, MUTED).pack(side="right")

        tiles = tk.Frame(self, bg=GROUND)
        tiles.pack(fill="x")
        self.tiles = {}
        for i, (key, label) in enumerate((("c", "≥ C1 alert · flares warned"), ("m", "M1 alert · flares warned"),
                                          ("now", "Flare in progress"), ("soon", "Flare within 15 min"),
                                          ("flux", "GOES flux from Aditya data"), ("peak", "Peak of a rising flare"))):
            t = Tile(tiles, label)
            t.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 8, 0), pady=(0, 8))
            tiles.grid_columnconfigure(i % 3, weight=1, uniform="r")
            self.tiles[key] = t

        charts = tk.Frame(self, bg=GROUND)
        charts.pack(fill="x")
        charts.grid_columnconfigure(0, weight=3, uniform="k")
        charts.grid_columnconfigure(1, weight=2, uniform="k")
        self.fig_b, self.ax_b, self.cv_b = self._chart(charts, 0, (5.2, 2.3))
        self.fig_r, self.ax_r, self.cv_r = self._chart(charts, 1, (3.4, 2.3))

        box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        box.pack(fill="both", expand=True, pady=(10, 0))
        tk.Label(box, text="DECISIONS AND VERDICTS  ·  each fixed before the test period was scored", bg=PANEL,
                 fg=MUTED, font=SMALL).pack(anchor="w", padx=12, pady=(9, 2))
        self.text = tk.Text(box, bg=PANEL, fg=TEXT, font=UI, relief="flat", highlightthickness=0, wrap="word",
                            height=9)
        self.text.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        for tag, col, font in (("k", MUTED, UI_B), ("good", GREEN, UI_B), ("no", AMBER, UI_B), ("wait", FAINT, UI)):
            self.text.tag_configure(tag, foreground=col, font=font)

    def _chart(self, parent, col, size):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
        fig = Figure(figsize=size, dpi=100, facecolor=PANEL, layout="constrained")
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        return fig, ax, canvas

    def _open(self):
        p = OUTPUTS / "RESULTS.md"
        if p.exists():
            os.startfile(p)

    # ---- refresh -------------------------------------------------------------------
    def refresh(self) -> None:
        stamp = tuple(f.stat().st_mtime if f.exists() else 0.0 for f in FILES.values())
        if stamp == self._stamp:
            return
        self._stamp = stamp
        R = {k: read_json(f) for k, f in FILES.items()}
        self._tiles(R)
        self._baselines(R.get("alerts"))
        self._reliability(R.get("cal"))
        self._verdicts(R)
        a = R.get("alerts")
        self.caption.config(text=(f"Test period {a['test_period'][0][:10]} to {a['test_period'][1][:10]}, "
                                  f"{a['test_days_with_data']} days with data. Thresholds fixed on validation."
                                  if a else "Results appear here as the pipeline's stages finish."))

    def _tiles(self, R) -> None:
        T = self.tiles
        rules = R.get("rules")
        for key, c in (("c", "C"), ("m", "M")):
            t = (rules or {}).get(c, {}).get("test")
            if t:
                T[key].set(pct(t["TPR"]), f"chance {pct(t['chance'])} · {t['false_per_day']:.1f}/day · "
                                          f"{t['median_lead_min']:.0f} min lead", TEXT)
            else:
                T[key].set("--", "after the alerts stage", MUTED)
        ev = R.get("eval")
        if ev:
            n = ev["nowcast_in_flare"]
            T["now"].set(f"TSS {n['TSS']:.3f}", f"95% {ci(n, 'TSS').strip()} · AUC {n['AUC']:.2f} · "
                                                f"POD {n['POD']:.2f} · FAR {n['FAR']:.2f}"
                                                + fb(n))
            occ = ev.get("forecast_occurrence", {})
            if "15min" in occ:
                T["soon"].set(f"TSS {occ['15min']['TSS']:.3f}",
                              f"95% {ci(occ['15min'], 'TSS').strip()} · "
                              + " · ".join(f"{h[:-3]} min {occ[h]['TSS']:.2f}" for h in ("30min", "60min") if h in occ)
                              + fb(occ["15min"]))
            fr = ev.get("forecast_regression", {})
            nr = ev["nowcast_regression"]
            slx = nr.get("solexs_calibration_MAE")
            T["flux"].set(f"{nr['MAE']:.3f} dex",
                          (f"95% {ci(nr, 'MAE', 3).strip()} · " if nr.get("MAE_ci95") else "")
                          + (f"a plain SoLEXS calibration: {slx:.3f} · " if slx else "")
                          + "+" + " · +".join(f"{h[:-3]} min {fr[h]['MAE']:.3f}" for h in ("15min", "60min") if h in fr))
            pk = ev.get("peak", {})
            if pk:
                T["peak"].set(f"{pk['log_peak_flux_MAE']:.3f} dex", f"timing {pk['time_to_peak_MAE_min']:.1f} min")
        else:
            for k in ("now", "soon", "flux", "peak"):
                T[k].set("--", "after the train stage", MUTED)

    def _baselines(self, a) -> None:
        ax = self.ax_b
        ax.clear()
        style_axes(ax, "≥ C1 alert vs simple baselines")
        if not a:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "after the alerts stage", transform=ax.transAxes, ha="center", color=FAINT)
            self.cv_b.draw_idle()
            return
        R = a["results"]["C"]
        op = f"fa_{a['primary_false_alarms_per_day']['C']:g}"
        rows = []
        for key, label in C_METHODS:
            d = (R.get(key) or {}).get(op) or (R.get(key) or {}).get("fixed")
            if d and d.get("event_TSS") is not None:
                rows.append((label, d))
        rows.reverse()
        y = range(len(rows))
        cols = [TEAL if lab == "network" else STEEL for lab, _ in rows]
        ax.barh(list(y), [d["event_TSS"] for _, d in rows], color=cols, height=0.62)
        for i, (lab, d) in zip(y, rows):
            ax.text(d["event_TSS"] + 0.01, i, f"{d['event_TSS']:.2f} · {pct(d['TPR'])} · {d['false_per_day']:.1f}/day",
                    va="center", fontsize=7,
                    color=TEXT if lab == "network" else MUTED)
        ax.set_yticks(list(y))
        ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("event TSS · flares warned · false alarms/day", color=MUTED, fontsize=8)
        self.cv_b.draw_idle()

    def _reliability(self, cal) -> None:
        ax = self.ax_r
        ax.clear()
        style_axes(ax, "Do the probabilities mean it?")
        h = (cal or {}).get("heads", {}).get("within_15min")
        if not h:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "after the calibration stage", transform=ax.transAxes, ha="center", color=FAINT)
        else:
            ax.plot([0, 1], [0, 1], color=LINE, lw=1)
            for key, col, lab in (("reliability_raw", FAINT, "raw network"),
                                  ("reliability_calibrated", TEAL, "calibrated")):
                r = h[key]
                ax.plot(r["predicted"], r["observed"], "o-", color=col, ms=3, lw=1.3, label=lab)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("forecast P(flare within 15 min)", color=MUTED, fontsize=8)
            ax.set_ylabel("observed frequency", color=MUTED, fontsize=8)
            muted_legend(ax, loc="upper left")
        self.cv_r.draw_idle()

    def _verdicts(self, R) -> None:
        t = self.text
        t.config(state="normal")
        t.delete("1.0", "end")

        def line(head: str, body: str, tag: str = "") -> None:
            t.insert("end", f"{head}  ", "k")
            t.insert("end", body + "\n", tag)

        s = R.get("sharp")
        if s:
            line("SHARP magnetic data as a network input:",
                 f"{'kept' if s['use_sharp'] else 'left out'} — validation {s['xray_only']['val_score']} without, "
                 f"{s['with_sharp']['val_score']} with (needed +{s['criterion'].split('>=')[-1].strip()}).",
                 "good" if s["use_sharp"] else "no")
        else:
            line("SHARP:", "after the choose-inputs stage", "wait")

        a = R.get("alerts")
        if a:
            for c, name in (("C", "≥ C1"), ("M", "M1")):
                h = a["results"][c].get("hel1os_ablation", {})
                mf = h.get("matched_false_alarms", {})
                ci = mf.get("warned_gain_ci")
                if h and ci:
                    sig = ci[0] > 0
                    line(f"HEL1OS in the {name} alert:",
                         f"{100 * h['warned_with']:.1f}% of {h['flares']} flares warned with it vs "
                         f"{100 * mf['warned_without']:.1f}% "
                         f"without, at equal false alarms (interval {ci[0]:+.3f} to {ci[1]:+.3f}) — "
                         + ("a real gain." if sig else "no significant gain."), "good" if sig else "no")
        v = R.get("hel1os")
        if v:
            sd = f" ± {100 * v['relative_sd']:.1f}%" if v.get("relative_sd") is not None else ""
            line("HEL1OS for the peak size of a rising flare:",
                 f"{100 * v['mean_relative']:.1f}%{sd} lower error over {len(v['seeds'])} seeds; "
                 f"{v['intervals_excluding_zero']} of {len(v['seeds'])} intervals exclude zero — {v['verdict']}.",
                 "good" if v["verdict"].startswith("holds") else "no")
        else:
            line("HEL1OS for peak size:", "after the hel1os-value stage (3 seeds)", "wait")

        cal = (R.get("cal") or {}).get("heads", {}).get("within_15min")
        if cal:
            line("Calibration:", f"Brier skill at 15 min {cal['raw']['BSS_vs_climatology']:+.3f} raw → "
                                 f"{cal['calibrated']['BSS_vs_climatology']:+.3f} calibrated; ranking (AUC) unchanged.",
                 "good" if cal["calibrated"]["BSS_vs_climatology"] > 0 else "no")
        c = R.get("catalog")
        if c:
            r = c["test"]["recall"]
            bits = [f"{k} {pct(r[k]['soft_recall'])}" for k in ("C", "M", "X") if k in r]
            both = r.get("C", {})
            line("Catalogue, test period:", "SoLEXS finds " + ", ".join(bits) + " of GOES flares; where both "
                 f"instruments observed, C flares {pct(both.get('both_combined_recall'))} vs "
                 f"{pct(both.get('both_combined_recall_chance'))} by chance.")
        ev = R.get("eval")
        fr = (ev or {}).get("forecast_regression", {})
        h60 = fr.get("60min")
        if h60 and h60.get("skill_vs_solexs_no_change") is not None:
            better = h60["skill_vs_solexs_no_change"] > 0
            line("Flux forecast an hour ahead:",
                 f"{100 * h60['skill_vs_solexs_no_change']:+.0f}% error against 'no change' from Aditya's own flux "
                 f"({100 * h60['skill_vs_persistence']:+.0f}% against GOES no change, which an Aditya-only system "
                 f"never sees); 80% range covers {pct(h60.get('interval_coverage'))} → "
                 f"{pct(h60.get('interval_coverage_calibrated'))} after the validation-fitted widening.",
                 "good" if better else "no")
        if a:
            hp = (a["results"]["C"].get("hope") or {}).get(f"fa_{a['primary_false_alarms_per_day']['C']:g}")
            md = (a["results"]["C"].get("model") or {}).get(f"fa_{a['primary_false_alarms_per_day']['C']:g}")
            if hp and md:
                win = md["event_TSS"] >= hp["event_TSS"]
                cut = (a.get("hope_trigger") or {}).get("cut")
                line("HOPE hot onset trigger (the published method, on SoLEXS):",
                     f"event TSS {hp['event_TSS']:.2f} vs the network's {md['event_TSS']:.2f} at the same false "
                     f"alarms; warns {pct(hp['TPR'])} of flares, median lead {hp['median_lead_min']} min"
                     + (f" (hot cut {cut} chosen on validation)." if cut else "."),
                     "good" if win else "no")
        d = R.get("dayahead")
        if d:
            r = d["results"].get(">=M1 within 24 h")
            if r:
                line("Day-ahead:", f"≥ M1 within 24 h, AUC {r['sets']['xray']['AUC']} from SoLEXS activity vs "
                                   f"{r['persistence']['AUC']} for persistence.")
        t.config(state="disabled")
