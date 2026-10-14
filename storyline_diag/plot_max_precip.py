"""
plot_max_precip.py

2-row map of peak intensity: max hourly (row 1) and max 24-hr (row 2)
precipitation, for AORC plus each E3SM case.

E3SM output is 3-hourly instantaneous PRECT, so "max hourly" is the peak
rate (mm/hr) times 1 hr, and max 24-hr is the largest trapezoidal depth
over any 8 consecutive intervals. AORC uses true hourly values and a
rolling 24-hr sum.
Options: plots.max_precip in defaults.yaml.

Run:  python plot_max_precip.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from storyline_config import load_event, event_window, plot_opts, data_dir
from storyline_io import (read_e3sm_precip, cumulative_depth, read_aorc_rolling_max,
                          aorc_available, available_cases)

WINDOW_HOURS = 24

cfg = load_event()
opt = plot_opts(cfg, 'max_precip')
t0, t1 = event_window(cfg)
hours_step = cfg['time'].get('hours_step', 3)
region = opt.get('region', 'WUS')
if opt.get('extent'):
    extent = list(opt['extent'])
    bbox = sp.pad_bbox(sp.extent_to_bbox(extent), 1.0)
    tag = "zoom"
else:
    bbox = tuple(cfg['regions'][region])
    extent = sp.bbox_to_extent(bbox)
    tag = region

boundaries = sp.load_boundaries(opt, extent)


def e3sm_peaks(times, rate):
    cum = cumulative_depth(times, rate)
    steps = int(round(WINDOW_HOURS / hours_step))
    if steps >= len(cum):
        raise ValueError(f"Only {len(cum)} records; need more than {steps} for a {WINDOW_HOURS}-hr window.")
    return np.nanmax(rate, axis=0), np.nanmax(cum[steps:] - cum[:-steps], axis=0)


# ── data ─────────────────────────────────────────────────────────────────
panels = []
if aorc_available(t0):
    try:
        print("Reading AORC ...")
        lat, lon, m1, m24 = read_aorc_rolling_max(data_dir(cfg, 'aorc_dir'), t0, t1, bbox, WINDOW_HOURS)
        panels.append({'label': 'AORC', 'kind': 'grid', 'lat': lat, 'lon': lon, 'h': m1, 'd': m24})
    except Exception as e:
        print(f"  AORC SKIPPED ({type(e).__name__}: {e})")
        panels.append({'label': 'AORC', 'na': 'AORC n/a'})
else:
    panels.append({'label': 'AORC', 'na': 'AORC n/a (pre-1979)'})

for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    h, d = e3sm_peaks(times, prect)
    panels.append({'label': case['label'], 'kind': 'mesh', 'lat': lat, 'lon': lon, 'h': h, 'd': d})

# ── figure ───────────────────────────────────────────────────────────────
rows = [('h', opt.get('vmax_hourly'), "Max hourly precipitation (mm)"),
        ('d', opt.get('vmax_daily'), f"Max {WINDOW_HOURS}-hr precipitation (mm)")]
fig, axes = plt.subplots(2, len(panels), figsize=(2.3 * len(panels) + 1.0, 6.0), squeeze=False,
                         subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(right=0.92, wspace=0.05, hspace=0.15)
for r, (key, vmax, label) in enumerate(rows):
    vmax = vmax or sp.nice_max([p[key] for p in panels if key in p])
    levels = np.linspace(0, vmax, 30)
    cf = None
    for c, p in enumerate(panels):
        ax = axes[r, c]
        sp.plot_background(ax, extent)
        if 'na' in p:
            sp.na_panel(ax, p['na'])
        else:
            cf = sp.draw_field(ax, {**p, 'val': p[key]}, levels, sp.PRECIP_CMAP)
            sp.add_boundaries(ax, boundaries)
        if r == 0:
            ax.set_title(p['label'], loc='left')
    if cf is not None:
        sp.cbar_right(fig, axes[r, -1], cf, label)
sp.save_png(fig, sp.out_path(cfg, f"max_precip_{tag}"))

# ── county summary ───────────────────────────────────────────────────────
if 'county' in boundaries:
    print(f"\nMax {WINDOW_HOURS}-hr precip, {opt.get('county', 'Sonoma')} County "
          f"(mean over points inside / wettest point):")
    for p in panels:
        if 'd' not in p:
            continue
        lat, lon = (np.meshgrid(p['lat'], p['lon'], indexing='ij') if p['kind'] == 'grid'
                    else (p['lat'], p['lon']))
        m = sp.points_in(boundaries['county'], lat, lon) & np.isfinite(p['d'])
        if m.any():
            print(f"  {p['label']}: {np.mean(p['d'][m]):.1f} / {np.max(p['d'][m]):.1f} mm  (n={int(m.sum())})")
        else:
            print(f"  {p['label']}: no points inside")
