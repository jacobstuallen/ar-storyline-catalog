"""
plot_precip_masks.py  (block 4, check before computing)

For each active region in process.yaml:
  1. precip_masks.png: elevation, terrain slope, and each mask group as
     classes, from the reference mesh. Prints the land slope percentiles
     and the column count per mask, for setting the slope classes.
  2. precip_mesh_spacing.png: local column spacing of every mesh (distance
     to the nearest column), with its median. The r/dx analysis assumes one
     spacing per mesh in the box; colours away from the median show where a
     region reaches a mesh's transition zone.
The surface is the same for every event; the event only says which run's
files to read.

Run:  python plot_precip_masks.py
"""

from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree

import process_io as pio
import storyline_io as sio
import storyline_plot as sp

EVENT_YEAR = pio.event_year(2019)
MARKER_SIZE = 2
SPACING_TOLERANCE = 1.25   # columns with spacing above this x the median are counted as "coarse"

pcfg = pio.load_process_cfg()
cfg = pio.find_event(pcfg, EVENT_YEAR)


def base(ax, n, title, lon, lat, land=None):
    if land is not None:
        ax.scatter(lon[~land], lat[~land], s=MARKER_SIZE, color='0.85', lw=0)
    ax.set_title(f"{ascii_lowercase[n]}) {title}", loc='left', fontsize=10)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


for REGION, opt in pio.precip_regions(pcfg):
    t0, t1 = pio.analysis_window(cfg, pcfg['window'])
    bbox = opt['region']
    print(f"\n##### region {REGION}: {bbox}")
    plot_dir = pio.out_dir(pcfg, 'plot_dir', 'summary', REGION)

    # ── 1. surface fields and mask classes ───────────────────────────────
    surf = pio.surface_fields(cfg, opt, t0, t1, bbox, pad=0.0)
    masks = pio.surface_masks(surf, opt['masks'])
    sl = surf['slope'][surf['land'] & np.isfinite(surf['slope'])]
    print(f"{opt['surface']['resolution']} columns: {len(surf['lat'])}, land {surf['land'].sum()}")
    print("land slope percentiles (m/km): " + ", ".join(
        f"p{q} {np.percentile(sl, q):.0f}" for q in (10, 25, 33, 50, 67, 75, 90)))
    for name, m in masks.items():
        print(f"  {name:16} {m.sum():6d} columns  ({pio.mask_title(name, opt)})")

    groups = pio.mask_groups(opt, list(masks))
    fig, axes = pio.panel_grid(2 + len(groups), 2, w=5.0, h=5.0)
    lon, lat, land = surf['lon'], surf['lat'], surf['land']
    for n, (key, title, cmap, vmax) in enumerate((("elev", "Elevation (m)", "terrain", None),
                                                  ("slope", "Terrain slope (m/km)", "magma_r",
                                                   np.nanpercentile(sl, 98)))):
        ax = axes[n]
        base(ax, n, f"{title}, {opt['surface']['resolution']}", lon, lat, land)
        sc = ax.scatter(lon[land], lat[land], c=surf[key][land], s=MARKER_SIZE, cmap=cmap,
                        vmin=0, vmax=vmax, lw=0)
        fig.colorbar(sc, ax=ax, shrink=0.8)
    for g, (group, gmasks) in enumerate(groups.items()):
        ax = axes[2 + g]
        base(ax, 2 + g, f"{group} classes", lon, lat, land)
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
    fig.suptitle(f"{REGION} {bbox}", fontsize=10)
    fig.tight_layout()
    sp.save_png(fig, f"{plot_dir}/precip_masks.png")

    # ── 2. local column spacing of every mesh ────────────────────────────
    meshes = []
    for case in cfg['cases']:
        try:
            index = sio._e3sm_file_index(dict(case, e3sm_stream=opt['surface']['stream']), t0, t1)
        except FileNotFoundError as e:
            print(f"  SKIP {case['resolution']}: {e}")
            continue
        if not index:
            continue
        cols = pio.Columns(index[0][0], bbox)
        d, _ = cKDTree(pio.sphere_xyz(cols.lat, cols.lon)).query(pio.sphere_xyz(cols.lat, cols.lon), k=2)
        meshes.append((case, cols.lat, cols.lon, d[:, 1]))
    if not meshes:
        continue
    fig, axes = pio.panel_grid(len(meshes), 3, w=4.6, h=5.0)
    finest = min(float(np.median(m[3])) for m in meshes)
    for n, ((case, la, lo, sp_km), ax) in enumerate(zip(meshes, axes)):
        med = float(np.median(sp_km))
        coarse = (sp_km > SPACING_TOLERANCE * med).mean()
        print(f"  {case['resolution']}: {len(la)} columns, spacing median {med:.1f} km, "
              f"p5-p95 {np.percentile(sp_km, 5):.1f}-{np.percentile(sp_km, 95):.1f} km, "
              f"{100 * coarse:.1f}% above {SPACING_TOLERANCE} x median")
        base(ax, n, f"{case['label']}: median {med:.1f} km", lo, la)
        size = MARKER_SIZE * min((med / finest) ** 2, 30)     # dots grow with spacing
        sc = ax.scatter(lo, la, c=sp_km / med, s=size, cmap='coolwarm', vmin=0.5, vmax=1.5, lw=0)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="spacing / median")
    fig.suptitle(f"{REGION}: local column spacing (nearest-neighbour distance)", fontsize=10)
    fig.tight_layout()
    sp.save_png(fig, f"{plot_dir}/precip_mesh_spacing.png")
