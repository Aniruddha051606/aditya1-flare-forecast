"""Figure 3: SoLEXHEL-Net, the network of every experiment.

    python paper_results/plotting/fig03_architecture.py

Every size in the figure is read from the final model's configuration
(outputs/model/reports/config.json, data_meta.json) and from the network built
with it (solarflare.models.net.SolexHelNet): parameter counts per block are
counted, not typed.
"""

from __future__ import annotations

import json

from diagram import arrow, blank, box, left, right
from style import DOUBLE, FILL, save, setup
import matplotlib.pyplot as plt

from solarflare.config import Config
from solarflare.models.net import SolexHelNet, count_parameters
from solarflare.settings import load_settings


def main() -> int:
    setup()
    S = load_settings()
    cfg = Config.from_json(S.model_dir / "reports" / "config.json")
    meta = json.loads((S.model_dir / "reports" / "data_meta.json").read_text("utf-8"))
    ns, nh = int(meta["soft_features"]), int(meta["hard_features"])
    net = SolexHelNet(ns, nh, 2, cfg.model, cfg.win)
    n = {name: count_parameters(mod) for name, mod in net.named_children()}
    heads = sum(v for k, v in n.items() if k.startswith("head_"))
    m, w = cfg.model, cfg.win
    L = int(round(w.input_seconds / cfg.pre.dt_seconds))
    dil = list(m.dilations)
    trunk = dil[: max(len(dil) // 2, 1)]
    occ = "/".join(str(int(h / 60)) for h in w.occurrence_horizons_s)
    fh = "/".join(str(int(h / 60)) for h in w.forecast_horizons_s)
    qs = "/".join(f"q{int(q * 100)}" for q in w.quantiles)

    fig = plt.figure(figsize=(DOUBLE, 2.7))
    ax = blank(fig)
    inp = f"{L} x {cfg.pre.dt_seconds:g} s ({w.input_seconds / 3600:g} h)"
    si = box(ax, 0.005, 0.62, 0.14, 0.24, f"SoLEXS (soft X-ray)\n{ns} features + mask\n{inp}", face=FILL["SoLEXS"], size=6)
    hi = box(ax, 0.005, 0.14, 0.14, 0.24, f"HEL1OS (hard X-ray)\n{nh} features + mask\n{inp}", face=FILL["HEL1OS"], size=6)
    enc_txt = (f"causal TCN encoder\nhidden {m.hidden}, kernel {m.kernel_size}\ndilations {dil[0]}-{dil[-1]}\n"
               "{p:,} parameters")
    se = box(ax, 0.175, 0.62, 0.16, 0.24, enc_txt.format(p=n["soft_enc"]), face=FILL["SoLEXS"], size=6)
    he = box(ax, 0.175, 0.14, 0.16, 0.24, enc_txt.format(p=n["hard_enc"]), face=FILL["HEL1OS"], size=6)
    fu = box(ax, 0.37, 0.36, 0.145, 0.28, f"gated fusion\n(gate sees both\nobservation masks)\n{n['fusion']:,} parameters",
             face=FILL["both"], size=6)
    tr = box(ax, 0.545, 0.36, 0.125, 0.28, f"causal TCN trunk\ndilations {trunk[0]}-{trunk[-1]}\n{n['trunk']:,}\nparameters",
             size=6)
    po = box(ax, 0.7, 0.36, 0.125, 0.28, f"causal attention\npooling + last\nstep, {m.attn_heads} heads\n"
             f"{n['pool']:,} parameters", size=6)
    hd = box(ax, 0.855, 0.06, 0.14, 0.86,
             f"heads\n({heads:,} parameters)\n\nflare in progress\n(probability)\n\nflare phase\n({m.n_phase_classes} classes)"
             f"\n\nflare within\n{occ} min\n(probabilities)\n\nflux now\n\nflux at\n+{fh} min\n({qs})\n\npeak time\nand size",
             size=6)
    for a, b in ((right(si), left(se)), (right(hi), left(he)), (right(fu), left(tr)), (right(tr), left(po)),
                 (right(po), left(hd))):
        arrow(ax, a, b)
    arrow(ax, right(se), (fu[0], fu[1] + 0.2))
    arrow(ax, right(he), (fu[0], fu[1] + 0.08))
    ax.text(0.175, 0.935, f"training: whole-instrument dropout p = {m.modality_dropout:g} (E4 only)", fontsize=6,
            transform=ax.transAxes)
    ax.text(0.175, 0.025, "E1 / E2: the same network with the HEL1OS / SoLEXS mask held at zero (model.inputs)",
            fontsize=6, transform=ax.transAxes)
    ax.text(0.37, 0.72, "flux heads add the calibrated\nSoLEXS flux now (training fit)" if m.anchor_flux else "",
            fontsize=5.5, transform=ax.transAxes)
    total = count_parameters(net)
    cap = (f"SoLEXHEL-Net ({total:,} trainable parameters), used by all experiments. Each instrument's {L}-step "
           f"({w.input_seconds / 3600:g} h at {cfg.pre.dt_seconds:g} s) input and its observation mask enter a causal "
           f"temporal convolutional encoder (hidden size {m.hidden}, kernel {m.kernel_size}, dilations "
           f"{', '.join(map(str, dil))}); a gate conditioned on both masks fuses them; a causal trunk (dilations "
           f"{', '.join(map(str, trunk))}) and causal attention pooling feed the heads. Every layer sees only the past. "
           f"During E4 training a whole instrument is dropped with probability {m.modality_dropout:g}. Clock inputs are "
           f"{'on' if m.use_clock else 'off'}. The single-instrument experiments E1 and E2 use the same network with "
           "the other instrument's mask held at zero in training and prediction.")
    save(fig, "fig03_architecture", cap, ["outputs/model/reports/config.json", "outputs/model/reports/data_meta.json",
                                           "solarflare/models/net.py"], "fig03_architecture.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
