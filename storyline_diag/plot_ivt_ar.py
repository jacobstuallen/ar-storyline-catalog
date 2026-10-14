"""
plot_ivt_ar.py

Integrated vapor transport (IVT) from TUQ/TVQ (instantaneous, h0 stream):
  1. IVT time series at each named location (nearest cell), with the
     Ralph et al. (2019) AR category in the legend.
  2. Peak IVT during the event at every cell, one map per case.
Options: plots.ivt_ar in defaults.yaml (region, extent, locations).

VERIFY: AR_CATEGORY_TABLE is from memory of Ralph et al. (2019, BAMS).
Check it against the CW3E AR Scale (https://cw3e.ucsd.edu/arscale/) before
quoting a category. Duration is (number of 3-hourly records with IVT > 250)
x 3 hr, so a single dry record splits one AR into two shorter ones.

Run:  python plot_ivt_ar.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import numpy as np
from scipy.spatial import cKDTree

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from matplotlib.tri import Triangulation

from storyline_config import load_event, event_window, plot_opts, resolve_locations
from storyline_io import read_e3sm, available_cases

IVT_AR_THRESHOLD = 250.0       # kg m-1 s-1
AR_CATEGORY_TABLE = [          # ivt_lo, ivt_hi, <24h, 24-48h, >48h
    (250, 500, 1, 2, 3),
    (500, 750, 2, 3, 4),
    (750, 1000, 3, 4, 5),
    (1000, 1250, 4, 5, 5),
    (1250, np.inf, 5, 5, 5),
]
CAT_BOUNDS = [0, 250, 500, 750, 1000, 1250, 2000]


def ar_category(max_ivt, duration_hr):
    if max_ivt < IVT_AR_THRESHOLD or duration_hr <= 0:
        return 0
    col = 0 if duration_hr < 24 else (1 if duration_hr <= 48 else 2)
    for lo, hi, *cats in AR_CATEGORY_TABLE:
        if lo <= max_ivt < hi:
            return cats[col]
    return 5


cfg = load_event()
opt = plot_opts(cfg, 'ivt_ar')
t0, t1 = event_window(cfg)
hours_step = cfg['time'].get('hours_step', 3)
locs = resolve_locations(cfg, opt.get('locations'))
bbox = tuple(cfg['regions'][opt.get('region', 'WUS')])
extent = list(opt.get('extent') or sp.bbox_to_extent(bbox))

# ── read each case once over the map region ──────────────────────────────
results = []      # (case, times, lat, lon, ivt)
for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM TUQ/TVQ: {case['name']} ...")
    times, lat, lon, v = read_e3sm(case, t0, t1, bbox, ('TUQ', 'TVQ'))
    results.append((case, times, lat, lon, np.hypot(v['TUQ'], v['TVQ'])))

# ── 1. time series at locations ──────────────────────────────────────────
ncols = min(3, len(locs))
nrows = int(np.ceil(len(locs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
axes = axes.ravel()
for ax, (name, (la, lo)) in zip(axes, locs.items()):
    for case, times, lat, lon, ivt in results:
        _, j = cKDTree(np.column_stack([lat, lon])).query([la, lo])
        series = ivt[:, j]
        mx = float(np.nanmax(series))
        dur = float(np.sum(series > IVT_AR_THRESHOLD) * hours_step)
        cat = ar_category(mx, dur)
        print(f"  {name} / {case['label']}: max {mx:.0f}, {dur:.0f} hr > 250, AR{cat}")
        ax.plot(times, series, color=sp.case_color(case), lw=1.5,
                label=f"{case['label']} (AR{cat}, max {mx:.0f})")
    ax.axhline(IVT_AR_THRESHOLD, color='gray', ls=':', lw=1, label="AR threshold (250)")
    ax.set_xlim(t0, t1)
    ax.set_ylabel("IVT (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title(name.replace('_', ' '), loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8)
for ax in axes[len(locs):]:
    ax.axis('off')
fig.tight_layout()
sp.save_png(fig, sp.out_path(cfg, "ivt_ar_timeseries"))

# ── 2. peak IVT map ──────────────────────────────────────────────────────
cmap = plt.get_cmap('viridis', len(CAT_BOUNDS) - 1)
norm = mcolors.BoundaryNorm(CAT_BOUNDS, cmap.N)
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5), squeeze=False,
                         subplot_kw={'projection': ccrs.PlateCarree()})
for ax, (case, times, lat, lon, ivt) in zip(axes[0], results):
    sp.plot_background(ax, extent)
    cf = ax.tricontourf(Triangulation(lon, lat), np.nanmax(ivt, axis=0), levels=CAT_BOUNDS,
                        cmap=cmap, norm=norm, extend='max', transform=ccrs.PlateCarree())
    ax.set_title(f"{case['label']}\nPeak IVT during event", loc='left', fontsize=10)
fig.subplots_adjust(bottom=0.15)
cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
fig.colorbar(cf, cax=cax, orientation='horizontal',
             label="Peak IVT (kg m$^{-1}$ s$^{-1}$), AR scale bin edges")
sp.save_png(fig, sp.out_path(cfg, "ivt_ar_peakmap"))
