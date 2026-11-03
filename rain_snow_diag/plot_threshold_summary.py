"""Figure 1: rain-snow transition threshold and width, all events.

Thresholds come from a weighted monotone (non-increasing) fit of snow
fraction vs temperature, so sparse noisy cold bins can't create spurious
crossings. Per event and resolution: T50 (dot), bootstrap CI over event days
(thick bar), and the transition range between width_levels snow (thin bar).
Bottom row pools all events (precip-weighted, so wet events dominate).

Printed table adds the cold-end snow fraction (samples colder than
cold_end_c), which caps how high the curve gets.

Output: plots/multi_event/threshold_summary.png
"""
import numpy as np
from matplotlib.lines import Line2D

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["threshold_summary"]
lo, hi, dw = so["temp_bins_c"]
edges = np.arange(lo, hi + 1e-9, dw)
centers = 0.5 * (edges[:-1] + edges[1:])
TEMPS = [("tw", "wet-bulb temperature (C)"), ("trefht", "2-m air temperature (C)")]
q_cold, q_warm = po["width_levels"]
cold_c = po.get("cold_end_c", -3.0)
ms = so["min_samples"]
rng = np.random.default_rng(po["seed"])


def stats(daily):
    """daily (ndays, 3, nb) -> T50, T(q_cold), T(q_warm), CI of T50, cold-end fraction."""
    tot = daily.sum(axis=0)
    t50 = rc.threshold(tot, centers, 0.5, ms)
    tc = rc.threshold(tot, centers, q_cold, ms)
    tw = rc.threshold(tot, centers, q_warm, ms)
    cold = centers < cold_c
    P, S = tot[0][cold].sum(), tot[1][cold].sum()
    plateau = S / P if P > 0 else np.nan
    nd = daily.shape[0]
    boot = np.array([rc.threshold(daily[rng.integers(0, nd, nd)].sum(axis=0), centers, 0.5, ms)
                     for _ in range(po["n_boot"])])
    ok = np.isfinite(boot)
    ci = np.percentile(boot[ok], po["ci"]) if ok.sum() > 0.5 * len(boot) else [np.nan, np.nan]
    return t50, tc, tw, ci, plateau


rows, labels, colors = {}, {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    ev = cfg["event_label"]
    print(f"=== {ev}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        rows.setdefault(ev, {})[res] = {k: rc.binned(d[k], p, s, w, ok, day, edges) for k, _ in TEMPS}
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not rows:
    raise SystemExit("no built events")

res_order = [r for r in rc.RES_COLORS if r in labels] + [r for r in labels if r not in rc.RES_COLORS]
pooled = {r: {k: np.concatenate([rows[e][r][k] for e in rows if r in rows[e]]) for k, _ in TEMPS}
          for r in res_order}
ylabels = [rc.event_short(e) for e in rows] + ["All events"]
blocks = list(rows.values()) + [pooled]

fig, axes = plt.subplots(1, 2, figsize=rc.fig_size(10, 1.4 + 0.6 * len(blocks)), sharey=True)
off = np.linspace(-0.22, 0.22, len(res_order)) if len(res_order) > 1 else [0.0]
qc, qw = int(q_cold * 100), int(q_warm * 100)
print(f"\n{'event':<10s} {'res':<7s} {'var':<7s} {'T50':>6s} {'CI lo':>6s} {'CI hi':>6s} "
      f"{'T' + str(qc):>6s} {'T' + str(qw):>6s} {'width':>6s} {'snow%<' + str(cold_c):>10s}")
for j, (key, xlabel) in enumerate(TEMPS):
    ax = axes[j]
    for i, (name, block) in enumerate(zip(ylabels, blocks)):
        y = len(blocks) - 1 - i
        for r, dy in zip(res_order, off):
            if r not in block:
                continue
            t50, tc, tw, ci, plateau = stats(block[r][key])
            print(f"{name:<10s} {r:<7s} {key:<7s} {t50:6.2f} {ci[0]:6.2f} {ci[1]:6.2f} "
                  f"{tc:6.2f} {tw:6.2f} {tw - tc:6.2f} {100 * plateau:10.1f}")
            ax.plot([tc, tw], [y + dy] * 2, color=colors[r], lw=1.2, alpha=0.45,
                    solid_capstyle="round")
            ax.plot(ci, [y + dy] * 2, color=colors[r], lw=3.2, solid_capstyle="round")
            ax.plot(t50, y + dy, "o", color=colors[r], ms=6, mec="white", mew=1.0, zorder=4)
    ax.axvline(0, color=rc.MUTED, lw=0.6)
    ax.axhline(0.5, color=rc.GRID, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    rc.panel_title(ax, j, "wet bulb" if key == "tw" else "2-m air temperature")
axes[0].set_yticks(range(len(blocks)))
axes[0].set_yticklabels(ylabels[::-1])
handles = [Line2D([], [], color=colors[r], marker="o", lw=3.2, ms=6, mec="white") for r in res_order]
handles += [Line2D([], [], color=rc.MUTED, lw=1.2, alpha=0.6)]
names = [f"{labels[r]}: T50 and {po['ci'][0]}-{po['ci'][1]}% CI" for r in res_order] + [
    f"{qc}% to {qw}% snow"]
fig.legend(handles, names, loc="lower left", ncol=len(names), fontsize=7,
           bbox_to_anchor=(0.01, 1.0))
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", "threshold_summary.png"))
