"""
plot_huc10_accum.py

HUC-10 watershed choropleth of accumulated precipitation over the event (or
a sub-window): per-watershed mean of AORC and each E3SM case, plus an
E3SM-minus-AORC difference figure. Watersheds smaller than an E3SM cell
spacing take the nearest cell's value.
Options: plots.huc10_accum in defaults.yaml. The hu2_regions list there is
a best guess at which WBD regions cover the extent; read the printed
warnings for any region that fails to load.

Run:  python plot_huc10_accum.py      (event: STORYLINE_CONFIG, see storyline_config.py)
Writes accum_by_huc10_<window>_raw.png and _diff.png
"""

import numpy as np

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from storyline_config import load_event, event_window, plot_opts, data_dir
from storyline_io import (read_e3sm_precip, total_depth, read_aorc_sum, aorc_available,
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

aorc = None
if aorc_available(t0):
    try:
        print("Reading AORC ...")
        lat, lon, tot = read_aorc_sum(data_dir(cfg, 'aorc_dir'), t0, t1, bbox)
        lon2d, lat2d = np.meshgrid(lon, lat)
        aorc = sp.zonal_stat(lat2d[::stride, ::stride], lon2d[::stride, ::stride],
                             tot[::stride, ::stride], polys, how='mean')
    except Exception as e:
        print(f"  AORC SKIPPED ({type(e).__name__}: {e})")
else:
    print(f"  AORC n/a for {t0.year} (record starts 1979); no difference figure.")

raw, diff = [("AORC", aorc)], []
for case in available_cases(cfg, t0, t1):
    print(f"Reading E3SM: {case['name']} ...")
    times, lat, lon, prect = read_e3sm_precip(case, t0, t1, bbox)
    g = sp.zonal_stat(lat, lon, total_depth(times, prect), polys, how='mean')
    raw.append((case['label'], g))
    if aorc is not None:
        d = g.copy()
        d['value'] = g['value'] - aorc['value']
        diff.append((f"{case['label']} minus AORC", d))

vmax = opt.get('vmax') or sp.nice_max([g['value'] for _, g in raw if g is not None], pct=99)
fig, _ = sp.choropleth_row(raw, display_extent, sp.PRECIP_CMAP, mcolors.Normalize(0, vmax),
                           "Accumulated precipitation (mm)", (2.9 * len(raw) + 1.0, 5.5), linewidth=0.01)
sp.save_png(fig, sp.out_path(cfg, f"accum_by_huc10_{window_tag}_raw"))

if diff:
    dmax = opt.get('diffmax') or sp.nice_max([g['value'] for _, g in diff], pct=99)
    fig, _ = sp.choropleth_row(diff, display_extent, plt.get_cmap('BrBG'),
                               mcolors.Normalize(-dmax, dmax), "Accumulation difference (mm)",
                               (2.9 * len(diff) + 1.0, 5.5), linewidth=0.01)
    sp.save_png(fig, sp.out_path(cfg, f"accum_by_huc10_{window_tag}_diff"))
