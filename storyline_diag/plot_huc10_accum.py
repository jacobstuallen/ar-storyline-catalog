"""
plot_huc10_accum.py

HUC-10 watershed choropleth of accumulated precipitation over the event (or
a sub-window): per-watershed mean of each obs product and each E3SM case, plus an
E3SM-minus-obs difference figure for each obs product. Watersheds smaller than an E3SM cell
spacing take the nearest cell's value.
Options: plots.huc10_accum in defaults.yaml. The hu2_regions list there is
a best guess at which WBD regions cover the extent; read the printed
warnings for any region that fails to load.

Run:  python plot_huc10_accum.py      (event: STORYLINE_CONFIG, see storyline_config.py)
Writes accum_by_huc10_<window>_raw.png and _diff_vs_<obs>.png
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from storyline_config import load_event, event_window, plot_opts
from storyline_io import (read_e3sm_precip, total_depth, read_obs_sum, obs_products,
                          available_cases)

cfg = load_event()
opt = plot_opts(cfg, 'huc10_accum')
start_day = int(opt.get('start_day') or 0)
t0, t1 = event_window(cfg, start_day, opt.get('window_days'))
extent = list(opt.get('extent', [-125.0, -102.0, 32.0, 51.0]))
display_extent = list(opt.get('display_extent') or extent)
bbox = sp.extent_to_bbox(extent)
stride = int(opt.get('aorc_stride', 6))
window_tag = f"day{start_day}-{start_day + (t1 - t0).days}"
print(f"Accumulation window: {t0} to {t1}")

print("Reading HUC-10 watersheds ...")
polys = sp.wbd_in_extent('WBDHU10', opt.get('hu2_regions', ['18']), extent)
print(f"  {len(polys)} watersheds")

obs = []                    # (name, label, GeoDataFrame)
for name in obs_products(cfg, t0, t1):
    try:
        print(f"Reading {name} ...")
        o = read_obs_sum(cfg, name, t0, t1, bbox)
        k = stride if name == 'AORC' else 1
        obs.append((name, o['label'], sp.zonal_stat(o['lat'][::k, ::k], o['lon'][::k, ::k],
                                                    o['val'][::k, ::k], polys, how='mean')))
    except Exception as e:
        print(f"  {name} SKIPPED ({type(e).__name__}: {e})")
if not obs:
    print("  No obs read; no difference figures.")

raw = [(label, g) for _, label, g in obs]
diffs = {name: [] for name, _, _ in obs}     # one difference figure per obs product
for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    g = sp.zonal_stat(lat, lon, total_depth(times, prect), polys, how='mean')
    raw.append((case['label'], g))
    for name, label, og in obs:
        d = g.copy()
        d['value'] = g['value'] - og['value']
        diffs[name].append((f"{case['label']} minus {label}", d))

vmax = opt.get('vmax') or sp.nice_max([g['value'] for _, g in raw if g is not None], pct=99)
fig, _ = sp.choropleth_row(raw, display_extent, sp.PRECIP_CMAP, mcolors.Normalize(0, vmax),
                           "Accumulated precipitation (mm)", (2.9 * len(raw) + 1.0, 5.5), linewidth=0.01)
sp.save_png(fig, sp.out_path(cfg, f"accum_by_huc10_{window_tag}_raw"))

dmax = opt.get('diffmax') or sp.nice_max(
    [g['value'] for dl in diffs.values() for _, g in dl] or [[1.0]], pct=99)
for name, diff in diffs.items():
    if not diff:
        continue
    fig, _ = sp.choropleth_row(diff, display_extent, plt.get_cmap('BrBG'),
                               mcolors.Normalize(-dmax, dmax), "Accumulation difference (mm)",
                               (2.9 * len(diff) + 1.0, 5.5), linewidth=0.01)
    sp.save_png(fig, sp.out_path(cfg, f"accum_by_huc10_{window_tag}_diff_vs_{name}"))
