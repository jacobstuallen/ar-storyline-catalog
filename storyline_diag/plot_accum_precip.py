"""
plot_accum_precip.py

Map of total event precipitation: each obs product (obs: in defaults.yaml)
plus each E3SM case, one row.
E3SM: trapezoidal integral of instantaneous PRECT. Obs: sum of hourly (AORC,
Stage IV) or daily (NClimGrid) values.
Options: plots.accum_precip in defaults.yaml (region, extent, boundary, vmax).

Run:  python plot_accum_precip.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from storyline_config import load_event, event_window, plot_opts
from storyline_io import (read_e3sm_precip, total_depth, read_obs_sum, obs_products, obs_label,
                          available_cases)

cfg = load_event()
opt = plot_opts(cfg, 'accum_precip')
t0, t1 = event_window(cfg)
region = opt.get('region', 'WUS')
if opt.get('extent'):
    extent = list(opt['extent'])
    bbox = sp.pad_bbox(sp.extent_to_bbox(extent), 1.0)
    tag = "zoom"
else:
    bbox = tuple(cfg['regions'][region])
    extent = sp.bbox_to_extent(bbox)
    tag = region
print(f"Window {t0} to {t1}, data bbox {bbox}")

boundaries = sp.load_boundaries(opt, extent)

# ── data ─────────────────────────────────────────────────────────────────
panels = []
for name in obs_products(cfg, t0, t1):
    try:
        print(f"Reading {name} ...")
        panels.append(read_obs_sum(cfg, name, t0, t1, bbox))
    except Exception as e:
        print(f"  {name} SKIPPED ({type(e).__name__}: {e})")
        panels.append({'label': obs_label(name), 'na': f"{obs_label(name)} n/a"})

for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    panels.append({'label': case['label'], 'kind': 'mesh', 'lat': lat, 'lon': lon,
                   'val': total_depth(times, prect)})

# ── figure ───────────────────────────────────────────────────────────────
vmax = opt.get('vmax') or sp.nice_max([p['val'] for p in panels if 'val' in p])
levels = np.linspace(0, vmax, 30)
fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 1.0, 3.2), squeeze=False,
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes[0]
cf = None
for ax, p in zip(axes, panels):
    sp.plot_background(ax, extent)
    if 'na' in p:
        sp.na_panel(ax, p['na'])
    else:
        cf = sp.draw_field(ax, p, levels, sp.PRECIP_CMAP)
        sp.add_boundaries(ax, boundaries)
    ax.set_title(p['label'], loc='left')
fig.subplots_adjust(right=0.92, wspace=0.05)
if cf is not None:
    sp.cbar_right(fig, axes[-1], cf, "Accumulated precipitation (mm)")
sp.save_png(fig, sp.out_path(cfg, f"accum_precip_{tag}"))

# ── county summary ───────────────────────────────────────────────────────
if 'county' in boundaries:
    print(f"\nEvent total, {opt.get('county', 'Sonoma')} County (simple mean of points inside):")
    for p in panels:
        if 'val' not in p:
            continue
        m = sp.points_in(boundaries['county'], p['lat'], p['lon']) & np.isfinite(p['val'])
        if m.any():
            print(f"  {p['label']}: {np.mean(p['val'][m]):.1f} mm  (n={int(m.sum())})")
        else:
            print(f"  {p['label']}: no points inside")
