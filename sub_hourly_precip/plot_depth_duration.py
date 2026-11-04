"""Depth-duration curves by mesh, and where the meshes converge.

Per event (plots/EVENT/depth_duration.png):
  a) Regional max depth vs duration (wettest land cell).
  b) Area-weighted quantile (plots.depth_duration.quantile) of per-cell max
     depth over land cells.
  c) (b) relative to ref_resolution, with a +-band shading.
  Solid = native mesh, dashed = finer mesh aggregated onto ref cells
  (build.coarsen_to). If dashed stays near solid, the difference is
  physical at common scale; if dashed collapses toward 1, it is grid scale.

Multi-event (plots/multi_event/depth_duration_ratio.png): panel (c) for every
event, one panel per finer mesh.

Run:  python plot_depth_duration.py   (after build_blocks_nc.py)
"""
import os

import numpy as np

import sh_common as sh

opts = sh.load_options()
B, DURS = sh.check_durations(opts)
P = opts["plots"]["depth_duration"]
Q = float(P["quantile"])
REF = P["ref_resolution"]
BAND = float(P.get("band", 0.1))
plt = sh.style()
from matplotlib.lines import Line2D  # noqa: E402

D = np.array(DURS)


def land(d, prefix=""):
    lf = d.get(f"{prefix}landfrac")
    n = d[f"{prefix}lat"].size
    return np.ones(n, bool) if lf is None else np.isfinite(lf) & (lf >= opts.get("min_landfrac", 0.5))


def stats(d, prefix=""):
    """(max over land, area-weighted quantile over land), each (ndur,)."""
    key = f"{prefix}maxdepth"
    if key not in d:
        return None
    m = land(d, prefix)
    md, w = d[key][:, m], d[f"{prefix}area"][m]
    mx = np.nanmax(md, axis=1)
    qq = np.array([sh.weighted_quantile(md[i], w, Q) for i in range(len(DURS))])
    return mx, qq


def fmt_row(name, vals, f="{:7.1f}"):
    return f"  {name:18s}" + "".join(f.format(v) for v in vals)


summary = {}   # event -> res -> {"native": ratio, "coarse": ratio}
for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    label = cfg["event_label"]
    print(f"\n=== {label}")
    built = sh.built_cases(opts, cfg)
    if not built:
        continue
    S = {}
    for case, d in built:
        S[case["resolution"]] = dict(case=case, native=stats(d), coarse=stats(d, "coarse_"))
    head = "  " + " " * 18 + "".join(f"{sh.duration_label(x):>7s}" for x in DURS)
    print(f"  max over land (mm) / q{Q:g} (mm):")
    print(head)
    for res, s in S.items():
        print(fmt_row(f"{res} max", s["native"][0]))
        print(fmt_row(f"{res} q{Q:g}", s["native"][1]))
        if s["coarse"] is not None:
            print(fmt_row(f"{res}->{REF} q{Q:g}", s["coarse"][1]))

    fig, axes = plt.subplots(1, 3, figsize=sh.fig_size(10.5, 3.4), constrained_layout=True)
    ratios = {}
    for res, s in S.items():
        col = sh.case_color(s["case"])
        lab = sh.case_label(cfg, s["case"])
        axes[0].plot(D, s["native"][0], color=col, marker="o", ms=3, label=lab)
        axes[1].plot(D, s["native"][1], color=col, marker="o", ms=3, label=lab)
        if s["coarse"] is not None:
            axes[0].plot(D, s["coarse"][0], color=col, ls="--", lw=1.2)
            axes[1].plot(D, s["coarse"][1], color=col, ls="--", lw=1.2)
    if REF in S:
        ref_q = S[REF]["native"][1]
        axes[2].axhspan(1 - BAND, 1 + BAND, color=sh.GRID, alpha=0.6, lw=0)
        axes[2].axhline(1, color=sh.MUTED, lw=0.8)
        print(f"  q{Q:g} ratio to {REF}:")
        print(head)
        for res, s in S.items():
            if res == REF:
                continue
            col = sh.case_color(s["case"])
            rn = s["native"][1] / ref_q
            axes[2].plot(D, rn, color=col, marker="o", ms=3)
            ratios[res] = {"native": rn}
            print(fmt_row(f"{res}", rn, "{:7.2f}"))
            if s["coarse"] is not None:
                rc = s["coarse"][1] / ref_q
                axes[2].plot(D, rc, color=col, ls="--", lw=1.2)
                ratios[res]["coarse"] = rc
                print(fmt_row(f"{res}->{REF}", rc, "{:7.2f}"))
    summary[label] = ratios

    for i, ax in enumerate(axes):
        sh.duration_axis(ax, DURS)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_ylabel("Max depth (mm)")
    axes[2].set_ylabel(f"Ratio to {sh.case_label(cfg, {'resolution': REF})}")
    sh.panel_title(axes[0], 0, "Wettest land cell")
    sh.panel_title(axes[1], 1, f"q{Q:g} of cell maxima (area-weighted)")
    sh.panel_title(axes[2], 2, f"q{Q:g} ratio")
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=sh.INK, lw=1.6), Line2D([], [], color=sh.INK, lw=1.2, ls="--")]
    l += ["native mesh", f"on {REF} cells"]
    axes[0].legend(h, l, loc="upper left")
    sh.save_png(fig, sh.plot_path(opts, label, "depth_duration.png"))
    plt.close(fig)

# ── multi-event ──────────────────────────────────────────────────────────
fine = sorted({r for v in summary.values() for r in v}, key=lambda r: int(r.split("x")[1]))
if fine:
    fig, axes = plt.subplots(1, len(fine), figsize=sh.fig_size(4.2 * len(fine), 3.4),
                             constrained_layout=True, squeeze=False, sharey=True)
    for j, res in enumerate(fine):
        ax = axes[0, j]
        ax.axhspan(1 - BAND, 1 + BAND, color=sh.GRID, alpha=0.6, lw=0)
        ax.axhline(1, color=sh.MUTED, lw=0.8)
        for i, (label, ratios) in enumerate(summary.items()):
            if res not in ratios:
                continue
            col = sh.EVENT_COLORS[i % len(sh.EVENT_COLORS)]
            ax.plot(D, ratios[res]["native"], color=col, marker="o", ms=3, label=sh.event_short(label))
            if "coarse" in ratios[res]:
                ax.plot(D, ratios[res]["coarse"], color=col, ls="--", lw=1.2)
        sh.duration_axis(ax, DURS)
        sh.panel_title(ax, j, f"{res} / {REF}, q{Q:g} of cell maxima")
        if j == 0:
            ax.set_ylabel("Ratio")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], color=sh.INK, lw=1.6), Line2D([], [], color=sh.INK, lw=1.2, ls="--")]
    l += ["native mesh", f"on {REF} cells"]
    axes[0, -1].legend(h, l, loc="upper right")
    d = os.path.join(sh.rel(opts["plot_dir"]), "multi_event")
    os.makedirs(d, exist_ok=True)
    sh.save_png(fig, os.path.join(d, "depth_duration_ratio.png"))
    plt.close(fig)
