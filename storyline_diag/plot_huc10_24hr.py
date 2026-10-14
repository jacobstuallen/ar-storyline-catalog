"""
plot_huc10_24hr.py

HUC-10 watershed map of precipitation (inches) over one 24-hr window
(observed_comparison.day_window), for AORC and every E3SM case that covers
it. Watersheds at or above huc10_24hr.threshold_in switch from the blue
scale to the red scale and get a black outline.

Station reports ("N stations over 10 inches") are point values, so the
per-watershed statistic defaults to `max` (wettest grid cell in each
watershed).

Run:  python plot_huc10_24hr.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs

from obs_compare import (MM_PER_IN, load_obs_cfg, parse_window, e3sm_window_accum,
                         aorc_window_accum)
from storyline_io import available_cases

cfg, oc = load_obs_cfg()
opt = oc.get('huc10_24hr', {})
t0, t1 = parse_window(oc['day_window'])
thresh = float(opt.get('threshold_in', 10))
stat = opt.get('stat', 'max')
bins = np.array(opt.get('bins_in', [0, 1, 2, 3, 4, 6, 8, 10, 12, 15]), dtype=float)
extent = list(opt.get('extent', [-124.6, -119.5, 36.5, 42.1]))
bbox = sp.extent_to_bbox(extent)
stride = int(opt.get('aorc_stride', 1))
print(f"24-hr window: {t0} to {t1} UTC, per-watershed stat: {stat}")
if thresh not in bins:
    raise ValueError(f"threshold_in ({thresh}) must be one of bins_in {bins.tolist()}.")


# ── data ─────────────────────────────────────────────────────────────────
print("Reading HUC-10 watersheds ...")
hucs = sp.wbd_in_extent('WBDHU10', opt.get('hu2_regions', ['18']), extent)

panels = []
print("Reading AORC ...")
a = aorc_window_accum(cfg, t0, t1, bbox)
if a is None:
    panels.append(("AORC", None))
else:
    a_lat, a_lon, a_acc = a
    a_lon2d, a_lat2d = np.meshgrid(a_lon, a_lat)
    panels.append(("AORC", sp.zonal_stat(a_lat2d[::stride, ::stride], a_lon2d[::stride, ::stride],
                                         a_acc[::stride, ::stride] / MM_PER_IN, hucs, how=stat)))

for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    lat, lon, acc = e3sm_window_accum(case, t0, t1, bbox)
    panels.append((case['label'], sp.zonal_stat(lat, lon, acc / MM_PER_IN, hucs, how=stat)))


# ── colors: blue below threshold, red at/above ───────────────────────────
n_below = int(np.sum(bins[1:] <= thresh))
n_above = len(bins) - 1 - n_below
blues = plt.cm.Blues(np.linspace(0.12, 0.80, n_below))
reds = plt.cm.Reds(np.linspace(0.55, 0.85, n_above + 1))
cmap = mcolors.ListedColormap(np.vstack([blues, reds[:-1]]))
cmap.set_over(reds[-1])
norm = mcolors.BoundaryNorm(bins, cmap.N)


# ── figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 5.6), squeeze=False,
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes[0]
print(f"\nWatersheds with 24-hr {stat} >= {thresh:g} in:")
for ax, (label, hw) in zip(axes, panels):
    sp.plot_background(ax, extent)
    if hw is None:
        sp.na_panel(ax, "AORC n/a (pre-1979)")
        ax.set_title(label, loc='left')
        continue
    hw.plot(ax=ax, column='value', cmap=cmap, norm=norm, edgecolor='white', linewidth=0.05,
            missing_kwds={'color': 'lightgrey'}, transform=ccrs.PlateCarree())
    hit = hw[hw['value'] >= thresh]
    if len(hit):
        hit.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6,
                 transform=ccrs.PlateCarree(), zorder=5)
    ax.set_title(f"{label}\n{len(hit)} watersheds ≥ {thresh:g} in", loc='left')
    print(f"  {label}: {len(hit)}")
    for _, r in hit.sort_values('value', ascending=False).head(10).iterrows():
        print(f"      {r['huc10']}  {r['name']:<35} {r['value']:.1f} in")

cax = fig.add_axes([0.25, 0.10, 0.5, 0.025])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                  extend='max', label=f"24-hr precipitation, watershed {stat} (in)")
cb.set_ticks(bins)
cb.set_ticklabels([f"{b:g}" for b in bins])
cb.ax.get_xticklabels()[int(np.flatnonzero(bins == thresh)[0])].set_fontweight('bold')
cb.ax.axvline(thresh, color='black', lw=1.5)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.86, wspace=0.05)

sp.save_png(fig, sp.out_path(cfg, f"huc10_24hr_{t0:%Y%m%d}"))
