"""Figure 4: snow fraction vs wet-bulb temperature, split by precip rate.

Heavier precipitation cools the melting layer (latent cooling), so snow can
survive to warmer surface temperatures. If the finer meshes' extra snow at
0-2 C comes from heavier rates, the resolutions should agree within each
rate class. Columns = rate classes, line per resolution, all events pooled.
Prints each resolution's precip share by rate class for Tw in near_zero_c.

Output: plots/multi_event/snowfrac_vs_rate.png
"""
import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["rate_classes"]
lo, hi, dw = so["temp_bins_c"]
edges = np.arange(lo, hi + 1e-9, dw)
centers = 0.5 * (edges[:-1] + edges[1:])
classes = [(a, np.inf if b is None else b) for a, b in po["classes_mmhr"]]
z0, z1 = po["near_zero_c"]

sums, near, labels, colors = {}, {}, {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    print(f"=== {cfg['event_label']}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        tw = d["tw"]
        for c, (a, b) in enumerate(classes):
            okc = ok & (p >= a) & (p < b)
            x = rc.binned(tw, p, s, w, okc, day, edges).sum(axis=0)
            sums.setdefault(res, {}).setdefault(c, np.zeros_like(x))
            sums[res][c] += x
            m = okc & (tw >= z0) & (tw < z1)
            near.setdefault(res, np.zeros(len(classes)))[c] += np.sum((p * w)[m])
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not sums:
    raise SystemExit("no built events")

res_order = [r for r in rc.RES_COLORS if r in sums]
cname = [f"{a:g}-{b:g} mm/hr" if np.isfinite(b) else f">= {a:g} mm/hr" for a, b in classes]
print(f"\nShare of precip with Tw in [{z0}, {z1}) C by rate class (%):")
print(f"{'res':<7s}" + "".join(f"{n:>16s}" for n in cname))
for r in res_order:
    print(f"{r:<7s}" + "".join(f"{100 * v / near[r].sum():16.1f}" for v in near[r]))

fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes) + 0.6, 4.2),
                         sharey=True, squeeze=False)
for c, ax in enumerate(axes[0]):
    for r in res_order:
        f = rc.snow_fraction(sums[r][c], so["min_samples"])
        t50 = rc.crossing(f, centers, 0.5)
        ax.plot(centers, 100 * f, color=colors[r], marker="o", ms=2.5,
                label=f"{labels[r]} (T50 {t50:.1f} C)")
        print(f"  {cname[c]:<14s} {r:<6s} T50 = {t50:5.2f} C  samples {int(sums[r][c][2].sum())}")
    ax.axhline(50, color=rc.MUTED, lw=0.6)
    ax.axvline(0, color=rc.MUTED, lw=0.6)
    ax.set_title(cname[c], loc="left")
    ax.set_xlabel("wet-bulb temperature (C)")
    ax.legend(loc="upper right", fontsize=7)
axes[0, 0].set_ylabel("snow fraction (%)")
axes[0, 0].set_ylim(-3, 103)
fig.suptitle("Snow fraction vs wet-bulb temperature by precip rate, all events pooled, land",
             x=0.01, ha="left")
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", "snowfrac_vs_rate.png"))
