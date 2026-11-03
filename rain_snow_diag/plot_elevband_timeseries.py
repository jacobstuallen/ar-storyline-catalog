"""Rain and snow by elevation band, one row per resolution.

Left: area-mean rain (solid) and snow (dashed) over land cells in each band.
Right: median freezing level across land cells in the box (shading = 25th
to 75th percentile across those cells, i.e. spatial spread) and the snow line.
Output: plots/EVENT/elevband_timeseries.png. Prints area per band per case,
since hypsometry differs by mesh.
"""
import numpy as np
from matplotlib.lines import Line2D

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
po = opts["plots"]["elevband_timeseries"]
sl = opts["snowline"]
edges = po["band_edges_m"]
nb = len(edges) - 1
# Spectral, low = red, high = blue, skipping the pale yellow centre.
x = np.linspace(0, 1, nb)
colors = [plt.get_cmap("Spectral")(v) for v in np.where(x < 0.5, 0.76 * x, 0.24 + 0.76 * x)]

for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    label = cfg["event_label"]
    print(f"=== {label}")
    cases = rc.built_cases(opts, cfg)
    if not cases:
        continue

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.0 * n + 0.6), sharex=True,
                             sharey="col", squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (case, d) in enumerate(cases):
        land = rc.land_mask(d, opts)
        z, w = d["zsfc"], d["area"]
        ax = axes[i, 0]
        print(f"  {case['resolution']}: band, cells, area km2")
        for b in range(nb):
            m = land & (z >= edges[b]) & (z < edges[b + 1])
            print(f"    {edges[b]:>5.0f}-{edges[b + 1]:<5.0f} m  {m.sum():6d}  {rc.km2(w[m].sum()):9.0f}")
            if m.sum() < po["min_cells"]:
                continue
            ww = w[m] / w[m].sum()
            ax.plot(d["time"], np.nansum(d["rain"][:, m] * ww, axis=1), color=colors[b], lw=1.4)
            ax.plot(d["time"], np.nansum(d["snow"][:, m] * ww, axis=1), color=colors[b],
                    lw=1.4, ls=(0, (3, 1.5)))
        ax.set_ylabel("mm/hr")
        ax.set_title(f"{d['label']}: rain and snow by elevation band", loc="left")

        ax = axes[i, 1]
        if d["fzl"].shape[0]:
            f = np.where(land, d["fzl"], np.nan)
            q25, q50, q75 = np.nanpercentile(f, [25, 50, 75], axis=1)
            ax.fill_between(d["time6"], q25, q75, color=rc.case_color(case), alpha=0.18, lw=0,
                            label="freezing level, 25-75th pct of cells")
            ax.plot(d["time6"], q50, color=rc.case_color(case), lw=1.6, label="freezing level, median of land cells")
        snl = rc.snow_line(z[land], d["rain"][:, land], d["snow"][:, land], w[land],
                           sl["bin_m"], sl["min_rate_mmhr"], sl["min_cells"], sl["frac"])
        ax.plot(d["time"], snl, color=rc.INK, lw=0, marker="o", ms=3, label="snow line (elevation of 50% snow)")
        for e in edges[1:-1]:
            ax.axhline(e, color=rc.MUTED, lw=0.5, ls=":")
        ax.set_ylabel("m MSL")
        ax.set_ylim(0, None)
        ax.set_title("freezing level and snow line", loc="left")
        if np.isfinite(snl).any():
            print(f"  {case['resolution']}: snow line median {np.nanmedian(snl):.0f} m")

    handles = [Line2D([], [], color=colors[b], lw=1.6) for b in range(nb)]
    labels = [f"{edges[b]:.0f}-{edges[b + 1]:.0f} m" for b in range(nb)]
    handles += [Line2D([], [], color=rc.MUTED, lw=1.4), Line2D([], [], color=rc.MUTED, lw=1.4, ls=(0, (3, 1.5)))]
    labels += ["rain", "snow"]
    fig.legend(handles, labels, loc="upper left", ncol=nb + 2, bbox_to_anchor=(0.01, 1.0))
    axes[0, 1].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        rc.date_axis(ax)
    fig.suptitle(f"{label}: rain-snow partition, box {opts['region_bbox']}", x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    rc.save_png(fig, rc.plot_path(opts, label, "elevband_timeseries.png"))
    plt.close(fig)
