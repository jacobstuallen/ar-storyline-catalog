"""Figure 1: rain-snow transition threshold and width, all events.

Per event and resolution: T50 (temperature where snow is 50% of precip) with
a bootstrap CI from resampling event days, and the transition width from
T(90% snow) to T(10% snow). Left panel wet-bulb, right panel 2-m T. Bottom
row pools all events.

Output: plots/multi_event/threshold_summary.png, and a printed table.
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
rng = np.random.default_rng(po["seed"])


def stats(daily):
    """daily (ndays, 3, nb) -> T50, T(q_cold), T(q_warm), CI of T50."""
    tot = daily.sum(axis=0)
    f = rc.snow_fraction(tot, so["min_samples"])
    t50 = rc.crossing(f, centers, 0.5)
    tc, tw = rc.crossing(f, centers, q_cold), rc.crossing(f, centers, q_warm)
    nd = daily.shape[0]
    boot = []
    for _ in range(po["n_boot"]):
        pick = rng.integers(0, nd, nd)
        boot.append(rc.crossing(rc.snow_fraction(daily[pick].sum(axis=0), so["min_samples"]),
                                centers, 0.5))
    boot = np.array(boot)
    ok = np.isfinite(boot)
    ci = np.percentile(boot[ok], po["ci"]) if ok.sum() > 0.5 * len(boot) else [np.nan, np.nan]
    return t50, tc, tw, ci


# rows[event_label][res] = {temp_key: daily sums}; pooled stacks days across events.
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

fig, axes = plt.subplots(1, 2, figsize=(11, 1.0 + 0.75 * len(blocks)), sharey=True)
off = np.linspace(-0.22, 0.22, len(res_order)) if len(res_order) > 1 else [0.0]
print(f"\n{'event':<10s} {'res':<7s} {'var':<7s} {'T50':>6s} {'CI lo':>6s} {'CI hi':>6s} "
      f"{'T' + str(int(q_cold * 100)):>6s} {'T' + str(int(q_warm * 100)):>6s} {'width':>6s}")
for j, (key, xlabel) in enumerate(TEMPS):
    ax = axes[j]
    for i, (name, block) in enumerate(zip(ylabels, blocks)):
        y = len(blocks) - 1 - i
        for r, dy in zip(res_order, off):
            if r not in block:
                continue
            t50, tc, tw, ci = stats(block[r][key])
            print(f"{name:<10s} {r:<7s} {key:<7s} {t50:6.2f} {ci[0]:6.2f} {ci[1]:6.2f} "
                  f"{tc:6.2f} {tw:6.2f} {tw - tc:6.2f}")
            ax.plot([tc, tw], [y + dy] * 2, color=colors[r], lw=1.2, alpha=0.45,
                    solid_capstyle="round")
            ax.plot(ci, [y + dy] * 2, color=colors[r], lw=3.2, solid_capstyle="round")
            ax.plot(t50, y + dy, "o", color=colors[r], ms=6, mec="white", mew=1.0, zorder=4)
    ax.axvline(0, color=rc.MUTED, lw=0.6)
    ax.axhline(0.5, color=rc.GRID, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(blocks)))
axes[0].set_yticklabels(ylabels[::-1])
handles = [Line2D([], [], color=colors[r], marker="o", lw=3.2, ms=6, mec="white") for r in res_order]
handles += [Line2D([], [], color=rc.MUTED, lw=1.2, alpha=0.6)]
names = [labels[r] for r in res_order] + [
    f"{int(q_cold * 100)}% to {int(q_warm * 100)}% snow (transition width)"]
fig.legend(handles, names, loc="upper left", ncol=len(names), bbox_to_anchor=(0.01, 1.02))
fig.suptitle(f"Temperature of 50% snow (dot) with {po['ci'][0]}-{po['ci'][1]}% bootstrap CI over days "
             f"(thick bar); land, PRECT >= {so['min_rate_mmhr']} mm/hr", x=0.01, ha="left", y=1.08)
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", "threshold_summary.png"))
