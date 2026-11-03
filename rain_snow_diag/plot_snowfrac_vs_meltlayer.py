"""Figure 3: snow fraction vs freezing-level height above ground.

Depth = freezing level (6-hr, nearest each 3-hr precip interval) minus
surface elevation, a proxy for melting-layer depth. Columns below 0 C all
the way to the ground are shown as their own point at the left ("frozen").
If the 14 km excess snow near 0 C disappears here, it comes from how surface
temperature relates to the melting layer; if it stays, from melting itself.

Columns: one per event plus all events pooled. Top row: share of precip per
depth bin; bottom row: snow fraction. Prints D50 (depth of 50% snow).
Output: plots/multi_event/snowfrac_vs_meltlayer.png
"""
import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["meltlayer"]
d0, d1, dd = po["depth_bins_m"]
edges = np.arange(d0, d1 + 1e-9, dd)
if po.get("frozen_to_ground", True):
    edges = np.concatenate([[d0 - dd], edges])     # first bin holds "frozen" columns
centers = 0.5 * (edges[:-1] + edges[1:])

sums, labels, colors = {}, {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    ev = cfg["event_label"]
    print(f"=== {ev}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        fz = rc.fzl_on_precip_times(d)
        if not np.isfinite(fz).any():
            print(f"  {res}: no freezing level in phase file, skipped")
            continue
        depth = fz - d["zsfc"][None, :]
        frozen = np.abs(depth) < 1.0                  # build sets fzl = zsfc for these
        depth = np.where(frozen, d0 - 0.5 * dd, np.maximum(depth, d0))
        if not po.get("frozen_to_ground", True):
            depth = np.where(frozen, np.nan, depth)
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        sums.setdefault(ev, {})[res] = rc.binned(depth, p, s, w, ok, day, edges).sum(axis=0)
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not sums:
    raise SystemExit("no built events with freezing level")

res_order = [r for r in rc.RES_COLORS if r in labels]
pooled = {r: sum(sums[e][r] for e in sums if r in sums[e]) for r in res_order}
cols = [(rc.event_short(e), sums[e]) for e in sums] + [("All events", pooled)]

fig, axes = plt.subplots(2, len(cols), figsize=(2.9 * len(cols) + 0.6, 5.6), sharex=True,
                         sharey="row", squeeze=False, gridspec_kw={"height_ratios": [1, 2]})
for j, (name, block) in enumerate(cols):
    for r in res_order:
        if r not in block:
            continue
        P = block[r][0]
        f = rc.snow_fraction(block[r], so["min_samples"])
        x = centers / 1000
        axes[0, j].plot(x, 100 * P / P.sum(), color=colors[r], label=labels[r])
        axes[1, j].plot(x, 100 * f, color=colors[r], marker="o", ms=2.5, label=labels[r])
        d50 = rc.crossing(f[1:], centers[1:], 0.5) if po.get("frozen_to_ground", True) else \
            rc.crossing(f, centers, 0.5)
        print(f"  {name:<10s} {r:<6s} D50 = {d50:6.0f} m   frozen-to-ground share "
              f"{100 * P[0] / P.sum() if po.get('frozen_to_ground', True) else np.nan:5.1f}%")
    axes[0, j].set_title(name, loc="left")
    axes[1, j].axhline(50, color=rc.MUTED, lw=0.6)
    axes[1, j].set_xlabel("freezing level above ground (km)")
    if po.get("frozen_to_ground", True):
        axes[1, j].axvline(edges[1] / 1000, color=rc.GRID, lw=1.2)
        axes[1, j].text(centers[0] / 1000, -9, "frozen", ha="center", fontsize=7, color=rc.MUTED)
axes[0, 0].set_ylabel("% of precip")
axes[1, 0].set_ylabel("snow fraction (%)")
axes[1, 0].set_ylim(-12, 103)
axes[1, 0].legend(loc="upper right", fontsize=8)
fig.suptitle(f"Snow fraction vs freezing-level height above ground; land, PRECT >= "
             f"{so['min_rate_mmhr']} mm/hr", x=0.01, ha="left")
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", "snowfrac_vs_meltlayer.png"))
