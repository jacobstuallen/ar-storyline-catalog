"""
plot_multicase_overlay.py

Area-weighted mean cumulative precipitation for each E3SM resolution against
AORC (and Stage IV from 2002 on), one figure per box in
plots.overlay.mean_regions.

Reads the netCDFs from build_comparison_nc.py (one per case and file region
in plots.overlay.regions), so run that first (the runner does). Each
mean_region is a name in `regions:` and must lie inside a file region; its
cells are cut out of the file region's netCDF, so no rebuild is needed to
add or change a mean_region.

Weights are E3SM cell areas, read from the case's first eam.h0 file
('area'), else from the SCRIP file ('grid_area'). Obs were sampled onto the
E3SM cells, so they use the same weights. When AORC is present, every series
is restricted to cells where AORC has data, so all lines describe the same
land area.

show_spread: true adds the (unweighted) 2.5-97.5 percentile spread across
cells as shading. It is a spatial spread, not a confidence interval.

Run:  python plot_multicase_overlay.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import glob
import os
from datetime import datetime, timedelta

import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree

import storyline_plot as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from storyline_config import load_event, event_window, plot_opts
from storyline_io import cumulative_depth

DEFAULT_STYLE = {'color': [31, 119, 180], 'lw': 1.5, 'zorder': 3, 'alpha_ci': 0.15}
AREA_MATCH_TOL_DEG = 1e-3      # max lat/lon mismatch when matching cells to area sources

cfg = load_event()
t0, t1 = event_window(cfg)
ov = cfg['overlay']
popt = plot_opts(cfg, 'overlay')
file_regions = popt.get('regions', ['WUS'])
mean_regions = popt.get('mean_regions') or file_regions
show_spread = bool(popt.get('show_spread', False))
ref_res = ov.get('ref_resolution')


def nc_path(case, region):
    return ov['comparison_nc_tmpl'].format(
        event_label=cfg['event_label'],
        resolution_tag=ov.get('resolution_tag_map', {}).get(case['resolution'], case['resolution']),
        region=region, year=cfg['time']['start_year'])


def cell_areas(case, lat, lon):
    """Area of each E3SM cell at (lat, lon), from the h0 'area' variable, else
    the SCRIP file's grid_area. Units don't matter (weights are normalized)."""
    stream = case.get('e3sm_stream', 'eam.h0')
    files = sorted(glob.glob(os.path.join(case['e3sm_dir'], f"*.{stream}.*.nc")))
    src = None
    if files:
        with nc.Dataset(files[0]) as ds:
            if 'area' in ds.variables:
                src = (ds['lat'][:], ds['lon'][:], ds['area'][:], f"'area' in {files[0]}")
    if src is None:
        path = case.get('scrip_file') or cfg.get('resolution_scrip_files', {}).get(case['resolution'])
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"No 'area' in the h0 files and no SCRIP file for '{case['name']}'.")
        with nc.Dataset(path) as ds:
            la, lo = ds['grid_center_lat'][:], ds['grid_center_lon'][:]
            if 'rad' in getattr(ds['grid_center_lat'], 'units', '').lower():
                la, lo = np.degrees(la), np.degrees(lo)
            src = (la, lo, ds['grid_area'][:], f"grid_area in {path}")
    la, lo, area = (np.asarray(x, dtype=float) for x in src[:3])
    lo = np.where(lo > 180, lo - 360, lo)
    dist, idx = cKDTree(np.column_stack([la, lo])).query(np.column_stack([lat, lon]))
    if dist.max() > AREA_MATCH_TOL_DEG:
        raise ValueError(f"Cell match for areas off by up to {dist.max():.4f} deg ({src[3]}).")
    print(f"  {case['name']}: area weights from {src[3]}")
    return area[idx]


def area_mean(cum, w):
    """Weighted mean across cells (axis 1), ignoring NaN cells per time."""
    ok = np.isfinite(cum)
    return np.nansum(cum * w, axis=1) / np.sum(ok * w, axis=1)


def spread(cum):
    return np.nanpercentile(cum, 2.5, axis=1), np.nanpercentile(cum, 97.5, axis=1)


def obs_cum(vals, name):
    n_bad = int(np.sum(~np.isfinite(vals[1:]).all(axis=1)))
    if n_bad:
        print(f"  WARNING: {name} has {n_bad} intervals with no data; they count as 0.")
    return np.nancumsum(vals, axis=0)


# ── load each case's comparison file(s) ──────────────────────────────────
loaded = {}      # file_region -> [dict per case]
for region in file_regions:
    print(f"\n== file region {region} ==")
    loaded[region] = []
    for case in cfg['cases']:
        path = nc_path(case, region)
        if not os.path.isfile(path):
            print(f"  SKIP {case['name']}: no comparison file {path}")
            continue
        with nc.Dataset(path) as ds:
            times = np.array([datetime(1970, 1, 1) + timedelta(seconds=float(s))
                              for s in ds.variables['time'][:]])
            d = {'case': case, 'times': times,
                 'lat': np.asarray(ds.variables['lat'][:], dtype=float),
                 'lon': np.asarray(ds.variables['lon'][:], dtype=float),
                 'e3sm': np.asarray(ds.variables['e3sm_prect_mmhr'][:], dtype=float)}
            for k, v in (('aorc', 'aorc_precip_mm'), ('st4', 'st4_precip_mm'), ('valid', 'aorc_valid')):
                d[k] = np.asarray(ds.variables[v][:], dtype=float) if v in ds.variables else None
        keep = d['valid'] > 0 if d['valid'] is not None else np.ones(len(d['lat']), dtype=bool)
        for k in ('lat', 'lon'):
            d[k] = d[k][keep]
        for k in ('e3sm', 'aorc', 'st4'):
            if d[k] is not None:
                d[k] = d[k][:, keep]
        d['area'] = cell_areas(case, d['lat'], d['lon'])
        d['e3sm_cum'] = cumulative_depth(d['times'], d['e3sm'])
        for k, label in (('aorc', 'AORC'), ('st4', 'Stage IV')):
            d[k + '_cum'] = obs_cum(d[k], f"{case['name']} {label}") if d[k] is not None else None
        loaded[region].append(d)


# ── one figure per mean region ───────────────────────────────────────────
n_figs = 0
for mreg in mean_regions:
    min_lat, min_lon, max_lat, max_lon = cfg['regions'][mreg]
    src_region = next((r for r in file_regions if loaded[r] and all(
        np.any((d['lat'] >= min_lat) & (d['lat'] <= max_lat) &
               (d['lon'] >= min_lon) & (d['lon'] <= max_lon)) for d in loaded[r])), None)
    if src_region is None:
        print(f"\n{mreg}: no comparison file region contains it; skipped.")
        continue
    print(f"\n== {mreg} (cells from {src_region} files) ==")

    series = []      # (d, mask)
    for d in loaded[src_region]:
        m = (d['lat'] >= min_lat) & (d['lat'] <= max_lat) & (d['lon'] >= min_lon) & (d['lon'] <= max_lon)
        print(f"  {d['case']['name']}: {int(m.sum())} cells")
        series.append((d, m))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    totals = []
    ref, ref_m = next(((d, m) for d, m in series if d['case']['resolution'] == ref_res), series[0])
    for key, label, color, ls in (('aorc', 'AORC', 'black', '-'), ('st4', 'Stage IV', 'dimgray', '--')):
        cum = ref[key + '_cum']
        if cum is None:
            continue
        cum, w = cum[:, ref_m], ref['area'][ref_m]
        mean = area_mean(cum, w)
        ax.plot(ref['times'], mean, color=color, lw=1.8, ls=ls, zorder=10, label=label)
        if show_spread:
            ax.fill_between(ref['times'], *spread(cum), color=color, alpha=0.1, zorder=1)
        totals.append((label, mean[-1]))

    styles = ov.get('styles', {})
    for d, m in series:
        st = {**DEFAULT_STYLE, **styles.get(d['case']['resolution'], {})}
        color = np.array(st['color']) / 255.0
        cum, w = d['e3sm_cum'][:, m], d['area'][m]
        mean = area_mean(cum, w)
        ax.plot(d['times'], mean, color=color, lw=st['lw'], zorder=st['zorder'], label=d['case']['label'])
        if show_spread:
            ax.fill_between(d['times'], *spread(cum), color=color, alpha=st['alpha_ci'], zorder=1)
        totals.append((d['case']['label'], mean[-1]))

    print(f"  Event totals, area-weighted mean over {mreg}:")
    for label, tot in totals:
        print(f"    {label:<16} {tot:7.1f} mm")

    ax.set_xlim(t0, t1)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{cfg['event_label']}: {mreg}, area-weighted mean", loc='left')
    ax.set_ylabel("Cumulative precipitation (mm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    sp.save_png(fig, sp.out_path(cfg, f"multicase_overlay_{mreg}"))
    n_figs += 1

if n_figs == 0:
    raise SystemExit("No overlay figures made (run build_comparison_nc.py first).")
