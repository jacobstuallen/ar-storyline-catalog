"""
print_basin_totals.py

Storm-total precipitation (inches) in each basin under
observed_comparison.basins, over observed_comparison.storm_window, for each
obs product and every E3SM case that covers the window. NClimGrid totals
cover whole days, so their period can be offset from the window (printed).

  mean   simple (not area-weighted) mean of grid cells inside the basin
  max    wettest grid cell inside the basin, and where it is

Station observations are point maxima, so compare them to `max`. Grid cells
average over their area, so expect models and gridded obs to fall short.

Run:  python print_basin_totals.py      (event: STORYLINE_CONFIG, see storyline_config.py)
Writes plots/<event_label>/basin_storm_totals.csv
"""

import os

import numpy as np

import storyline_plot as sp
from obs_compare import (MM_PER_IN, load_obs_cfg, parse_window, e3sm_window_accum,
                         obs_window_accum)
from storyline_io import available_cases

cfg, oc = load_obs_cfg()
t0, t1 = parse_window(oc['storm_window'])
obs_total_in = oc.get('observed_storm_total_in')
print(f"Storm window: {t0} to {t1} UTC")


# ── basin polygons ───────────────────────────────────────────────────────
def load_basin(name, spec):
    """Dissolve the WBD polygons matching spec ({hu2, layer, name_contains}
    or {hu2, layer, codes: [...]}) into one geometry."""
    gdf = sp.wbd_layer(spec.get('hu2', '18'), spec['layer'])
    code_col = 'huc' + spec['layer'][5:]
    if 'codes' in spec:
        sel = gdf[gdf[code_col].astype(str).isin([str(c) for c in spec['codes']])]
    else:
        sel = gdf[gdf['name'].str.contains(spec['name_contains'], case=False, na=False)]
    if sel.empty:
        raise ValueError(f"No {spec['layer']} polygons matched basin '{name}' ({spec}).")
    print(f"\n{name}: {len(sel)} {spec['layer']} polygon(s) matched:")
    for _, r in sel.iterrows():
        print(f"    {r[code_col]}  {r['name']}")
    return sel.geometry.union_all() if hasattr(sel.geometry, 'union_all') else sel.geometry.unary_union


basins = {name: load_basin(name, spec) for name, spec in oc['basins'].items()}
bounds = np.array([g.bounds for g in basins.values()])      # minx, miny, maxx, maxy
pad = 0.5
region_bbox = (bounds[:, 1].min() - pad, bounds[:, 0].min() - pad,
               bounds[:, 3].max() + pad, bounds[:, 2].max() + pad)


# ── data ─────────────────────────────────────────────────────────────────
sources = []   # (label, lat_flat, lon_flat, depth_in_flat)
print()
for o in obs_window_accum(cfg, t0, t1, region_bbox):
    sources.append((o['label'], o['lat'].ravel(), o['lon'].ravel(), o['val'].ravel() / MM_PER_IN))

for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    lat, lon, acc = e3sm_window_accum(case, t0, t1, region_bbox)
    sources.append((case['label'], lat, lon, acc / MM_PER_IN))


# ── table ────────────────────────────────────────────────────────────────
rows = []
for bname, geom in basins.items():
    for label, lat, lon, acc_in in sources:
        inside = sp.points_in(geom, lat, lon) & np.isfinite(acc_in)
        n = int(inside.sum())
        if n == 0:      # grid coarser than the basin: nearest cell to its center
            rp = geom.representative_point()
            j = int(np.nanargmin(np.where(np.isfinite(acc_in),
                                          (lat - rp.y) ** 2 + (lon - rp.x) ** 2, np.nan)))
            rows.append((bname, label, acc_in[j], acc_in[j], lat[j], lon[j], 0))
            continue
        vals = acc_in[inside]
        j = np.flatnonzero(inside)[np.argmax(vals)]
        rows.append((bname, label, float(vals.mean()), float(acc_in[j]),
                     float(lat[j]), float(lon[j]), n))

hdr = (f"{'Basin':<10}{'Source':<22}{'Mean (in)':>10}{'Max (in)':>10}"
       f"{'Max lat':>9}{'Max lon':>10}{'Cells':>7}")
print(f"\nStorm totals, {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M} UTC")
if obs_total_in:
    print(f"Observed (station max): ~{obs_total_in} in")
print(hdr)
print("-" * len(hdr))
prev = None
for b, s, mean, mx, la, lo, n in rows:
    if prev and b != prev:
        print()
    flag = "  (nearest cell, none inside)" if n == 0 else ""
    print(f"{b:<10}{s:<22}{mean:>10.1f}{mx:>10.1f}{la:>9.3f}{lo:>10.3f}{n:>7}{flag}")
    prev = b

out_csv = os.path.join("plots", cfg['event_label'], "basin_storm_totals.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, 'w') as f:
    f.write("basin,source,mean_in,max_in,max_lat,max_lon,n_cells\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print(f"\nWritten: {out_csv}")
