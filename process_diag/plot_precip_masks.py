"""
plot_precip_masks.py  (block 4, check before computing)

Maps of the reference-mesh surface fields used for the precipitation masks:
elevation, terrain slope, and each mask group as classes. Also prints the
land slope percentiles and the column count per mask, to set the slope
classes in process.yaml before running compute_precip_structure_functions.py.
The surface is the same for every event; the event only says which run's
files to read.

Run:  python plot_precip_masks.py
"""

from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import process_io as pio
import storyline_plot as sp

EVENT_YEAR = pio.event_year(2019)
MARKER_SIZE = 2

pcfg = pio.load_process_cfg()
opt = pcfg['precip_structure_function']
cfg = pio.find_event(pcfg, EVENT_YEAR)
t0, t1 = pio.analysis_window(cfg, pcfg['window'])
bbox = opt['region']
surf = pio.surface_fields(cfg, opt, t0, t1, bbox, pad=0.0)
masks = pio.surface_masks(surf, opt['masks'])

sl = surf['slope'][surf['land'] & np.isfinite(surf['slope'])]
print(f"{opt['surface']['resolution']} columns in {bbox}: {len(surf['lat'])}, land {surf['land'].sum()}")
print("land slope percentiles (m/km): " + ", ".join(
    f"p{q} {np.percentile(sl, q):.0f}" for q in (10, 25, 33, 50, 67, 75, 90)))
for name, m in masks.items():
    print(f"  {name:16} {m.sum():6d} columns  ({pio.mask_title(name, opt)})")

groups = pio.mask_groups(opt, list(masks))
fig, axes = pio.panel_grid(2 + len(groups), 2, w=5.0, h=5.0)
lon, lat, land = surf['lon'], surf['lat'], surf['land']


def base(ax, n, title):
    ax.scatter(lon[~land], lat[~land], s=MARKER_SIZE, color='0.85', lw=0)
    ax.set_title(f"{ascii_lowercase[n]}) {title}", loc='left', fontsize=10)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


for n, (key, title, cmap, vmax) in enumerate((("elev", "Elevation (m)", "terrain", None),
                                              ("slope", "Terrain slope (m/km)", "magma_r",
                                               np.nanpercentile(sl, 98)))):
    ax = axes[n]
    base(ax, n, f"{title}, {opt['surface']['resolution']}")
    sc = ax.scatter(lon[land], lat[land], c=surf[key][land], s=MARKER_SIZE, cmap=cmap,
                    vmin=0, vmax=vmax, lw=0)
    fig.colorbar(sc, ax=ax, shrink=0.8)

for g, (group, gmasks) in enumerate(groups.items()):
    ax = axes[2 + g]
    base(ax, 2 + g, f"{group} classes")
    classes = [m for m in gmasks if m != "land"]
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.9, max(len(classes), 1)))
    for m, c in zip(classes, colors):
        sel = masks[m]
        ax.scatter(lon[sel], lat[sel], s=MARKER_SIZE, color=c, lw=0,
                   label=f"{pio.mask_title(m, opt)} ({sel.sum()})")
    unassigned = land & ~np.logical_or.reduce([masks[m] for m in classes]) if classes else land
    if unassigned.any():
        ax.scatter(lon[unassigned], lat[unassigned], s=MARKER_SIZE, color='red', lw=0,
                   label=f"land in no class ({unassigned.sum()})")
    ax.legend(frameon=False, fontsize=7, markerscale=4, loc='lower left')

fig.tight_layout()
sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', 'summary')}/precip_masks.png")
