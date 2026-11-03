"""Convective (ZM: PRECC, PRECSC) vs large-scale (P3: PRECL = PRECT - PRECC,
PRECSL) phase partition, all events pooled.

a) convective share of precip vs wet-bulb temperature
b) large-scale snow fraction vs wet bulb
c) convective snow fraction vs wet bulb
Tests whether sub-freezing rain (cold-end snow fraction < 90%) and the 14 km
light-rate behavior come from the convective scheme or from P3.

Needs prect_conv / snow_conv in the phase files (build_phase_nc.py appends
them to existing files). Output: plots/multi_event/snowfrac_conv_ls.png
"""
import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
cold_c = opts["plots"]["threshold_summary"].get("cold_end_c", -3.0)
lo, hi, dw = so["temp_bins_c"]
edges = np.arange(lo, hi + 1e-9, dw)
centers = 0.5 * (edges[:-1] + edges[1:])
ms = so["min_samples"]

acc, labels, colors = {}, {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    print(f"=== {cfg['event_label']}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        if "prect_conv" not in d or not np.isfinite(d["prect_conv"]).any():
            print(f"  {res}: no prect_conv in phase file; rerun build_phase_nc.py")
            continue
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        conv = np.nan_to_num(d["prect_conv"])
        sconv = np.nan_to_num(d["snow_conv"])
        ls = np.clip(p - conv, 0, None)
        sls = np.clip(s - sconv, 0, None)
        tw = d["tw"]
        parts = {"share": rc.binned(tw, p, conv, w, ok, day, edges).sum(0),       # S/P = conv share
                 "ls": rc.binned(tw, ls, sls, w, ok, day, edges).sum(0),
                 "conv": rc.binned(tw, conv, sconv, w, ok, day, edges).sum(0)}
        a = acc.setdefault(res, {k: np.zeros_like(v) for k, v in parts.items()})
        for k, v in parts.items():
            a[k] += v
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not acc:
    raise SystemExit("no phase files with prect_conv")

res_order = [r for r in rc.RES_COLORS if r in acc]
cold = centers < cold_c
print(f"\n{'res':<7s} {'conv % of precip':>17s} {'conv % near 0C':>15s} "
      f"{'LS snow%<' + str(cold_c):>13s} {'conv snow%<' + str(cold_c):>15s} {'LS T50':>7s} {'conv T50':>9s}")
near = (centers >= -1) & (centers < 2)
for r in res_order:
    a = acc[r]
    P, C = a["share"][0], a["share"][1]
    cold_ls = a["ls"][1][cold].sum() / max(a["ls"][0][cold].sum(), 1e-12)
    cold_cv = a["conv"][1][cold].sum() / max(a["conv"][0][cold].sum(), 1e-12)
    print(f"{r:<7s} {100 * C.sum() / P.sum():17.1f} {100 * C[near].sum() / P[near].sum():15.1f} "
          f"{100 * cold_ls:13.1f} {100 * cold_cv:15.1f} "
          f"{rc.threshold(a['ls'], centers, 0.5, ms):7.2f} {rc.threshold(a['conv'], centers, 0.5, ms):9.2f}")

fig, axes = plt.subplots(1, 3, figsize=rc.fig_size(11, 3.8), sharex=True)
titles = ["convective share of precip", "large-scale (P3) snow fraction", "convective (ZM) snow fraction"]
for j, key in enumerate(["share", "ls", "conv"]):
    ax = axes[j]
    for r in res_order:
        f = rc.snow_fraction(acc[r][key], ms)
        ax.plot(centers, 100 * f, color=colors[r], marker="o", ms=2.5, label=labels[r])
    ax.axvline(0, color=rc.MUTED, lw=0.6)
    if j:
        ax.axhline(50, color=rc.MUTED, lw=0.6)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("wet-bulb temperature (C)")
    ax.set_ylabel("% of precip" if j == 0 else "snow fraction (%)")
    rc.panel_title(ax, j, titles[j])
axes[0].legend(loc="upper right")
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", "snowfrac_conv_ls.png"))
