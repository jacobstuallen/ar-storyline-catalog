"""
plot_accum_precip.py

Map of total event precipitation: AORC plus each E3SM case, one row.
E3SM: trapezoidal integral of instantaneous PRECT. AORC: sum of hourly values.
Options: plots.accum_precip in defaults.yaml (region, extent, boundary, vmax).

Run:  python plot_accum_precip.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from storyline_config import load_event, event_window, plot_opts, data_dir
from storyline_io import (read_e3sm_precip, total_depth, read_aorc_sum, aorc_available,
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
if aorc_available(t0):
    try:
        print("Reading AORC ...")
        lat, lon, tot = read_aorc_sum(data_dir(cfg, 'aorc_dir'), t0, t1, bbox)
        panels.append({'label': 'AORC', 'kind': 'grid', 'lat': lat, 'lon': lon, 'val': tot})
    except Exception as e:
        print(f"  AORC SKIPPED ({type(e).__name__}: {e})")
        panels.append({'label': 'AORC', 'na': 'AORC n/a'})
else:
    panels.append({'label': 'AORC', 'na': 'AORC n/a (pre-1979)'})

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
        lat, lon = (np.meshgrid(p['lat'], p['lon'], indexing='ij') if p['kind'] == 'grid'
                    else (p['lat'], p['lon']))
        m = sp.points_in(boundaries['county'], lat, lon) & np.isfinite(p['val'])
        if m.any():
            print(f"  {p['label']}: {np.mean(p['val'][m]):.1f} mm  (n={int(m.sum())})")
        else:
            print(f"  {p['label']}: no points inside")
