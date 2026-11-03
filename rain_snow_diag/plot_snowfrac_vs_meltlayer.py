"""Figure 3: snow fraction vs freezing-level height above ground.

Depth = freezing level (6-hr, nearest each 3-hr precip interval) minus
surface elevation, a proxy for melting-layer depth. Columns below 0 C all
the way to the ground are shown as their own point at the left ("frozen").
If the 14 km excess snow near 0 C disappears here, it comes from how surface
temperature relates to the melting layer; if it stays, from melting itself.

One panel per event plus all events pooled, 3 per row. Each panel: share of
precip per depth bin (top) and snow fraction (bottom). D50 (depth of 50%
snow, from a monotone fit) is printed and marked.
Output: plots/multi_event/snowfrac_vs_meltlayer.png
"""
import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["meltlayer"]
d0, d1, dd = po["depth_bins_m"]
frozen_bin = po.get("frozen_to_ground", True)
edges = np.arange(d0, d1 + 1e-9, dd)
if frozen_bin:
    edges = np.concatenate([[d0 - dd], edges])     # first bin holds "frozen" columns
centers = 0.5 * (edges[:-1] + edges[1:])
NCOL = 3

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
        if not frozen_bin:
            depth = np.where(frozen, np.nan, depth)
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        sums.setdefault(ev, {})[res] = rc.binned(depth, p, s, w, ok, day, edges).sum(axis=0)
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not sums:
    raise SystemExit("no built events with freezing level")

res_order = [r for r in rc.RES_COLORS if r in labels]
pooled = {r: sum(sums[e][r] for e in sums if r in sums[e]) for r in res_order}
blocks = [(rc.event_short(e), sums[e]) for e in sums] + [("All events", pooled)]
nrow = int(np.ceil(len(blocks) / NCOL))

fig = plt.figure(figsize=rc.fig_size(11, 3.6 * nrow + 0.4))
gs = fig.add_gridspec(2 * nrow, NCOL, height_ratios=[1, 2.2] * nrow, hspace=0.12)
x = centers / 1000
k0 = 1 if frozen_bin else 0
for b, (name, block) in enumerate(blocks):
    row, col = divmod(b, NCOL)
    ax_s = fig.add_subplot(gs[2 * row, col])
    ax_f = fig.add_subplot(gs[2 * row + 1, col], sharex=ax_s)
    for r in res_order:
        if r not in block:
            continue
        P = block[r][0]
        f = rc.snow_fraction(block[r], so["min_samples"])
        ax_s.plot(x, 100 * P / P.sum(), color=colors[r], label=labels[r])
        ax_f.plot(x, 100 * f, color=colors[r], marker="o", ms=2.5, label=labels[r])
        d50 = rc.threshold(block[r][:, k0:], centers[k0:], 0.5, so["min_samples"])
        if np.isfinite(d50):
            ax_f.axvline(d50 / 1000, color=colors[r], lw=0.8, ls=":")
        print(f"  {name:<10s} {r:<6s} D50 = {d50:6.0f} m   frozen-to-ground share "
              f"{100 * P[0] / P.sum() if frozen_bin else np.nan:5.1f}%")
    rc.panel_title(ax_s, b, name)
    plt.setp(ax_s.get_xticklabels(), visible=False)
    ax_f.axhline(50, color=rc.MUTED, lw=0.6)
    ax_f.set_ylim(-12, 103)
    if frozen_bin:
        ax_f.axvline(edges[1] / 1000, color=rc.GRID, lw=1.2)
        ax_f.text(x[0], -9, "frozen", ha="center", fontsize=7, color=rc.MUTED)
    if row == nrow - 1 or b + NCOL >= len(blocks):
        ax_f.set_xlabel("freezing level above ground (km)")
    if col == 0:
        ax_s.set_ylabel("% of precip")
        ax_f.set_ylabel("snow fraction (%)")
    if b == 0:
        ax_f.legend(loc="upper right", fontsize=7)
rc.save_png(fig, rc.plot_path(opts, "multi_event", "snowfrac_vs_meltlayer.png"))
