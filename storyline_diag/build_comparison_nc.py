"""
build_comparison_nc.py

Writes the comparison netCDFs that plot_multicase_overlay.py reads: one per
case and region, at overlay.comparison_nc_tmpl. Always overwrites, so a
stale file from an older schema can't be picked up by mistake.

For each case and region (plots.overlay.regions):
  - E3SM PRECT (mm/hr, instantaneous) on the native mesh, cropped to the region.
  - AORC sampled at the AORC cell nearest each E3SM cell, with the hourly
    values SUMMED into each E3SM output interval (t[i-1], t[i]]. Row 0 is 0.
    (The previous version kept only the single hour nearest each 3-hourly
    E3SM time, which undercounted AORC by about 3x.)
  - Stage IV the same way, for events from 2002 on.
  - aorc_valid (ncol): 1 where AORC has data for every interval. The overlay
    uses it so E3SM and AORC are compared over the same cells (AORC is NaN
    over the ocean and outside CONUS).

Run:  python build_comparison_nc.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree

from storyline_config import load_event, event_window, plot_opts, data_dir
from storyline_io import (read_e3sm_precip, read_aorc_at_points, read_st4, sum_into_intervals,
                          available_cases, aorc_available, ST4_EPOCH)

cfg = load_event()
t0, t1 = event_window(cfg)
regions = plot_opts(cfg, 'overlay').get('regions', ['WUS'])
ov = cfg['overlay']
tag_map = ov.get('resolution_tag_map', {})
aorc_dir, st4_dir = data_dir(cfg, 'aorc_dir'), data_dir(cfg, 'st4_dir')
use_aorc = aorc_available(t0) and aorc_dir
use_st4 = t0.date() >= ST4_EPOCH and st4_dir
if not use_aorc:
    print(f"AORC n/a for {t0.year} (record starts 1979); writing E3SM only.")


def nc_path(case, region):
    return ov['comparison_nc_tmpl'].format(
        event_label=cfg['event_label'], resolution_tag=tag_map.get(case['resolution'], case['resolution']),
        region=region, year=cfg['time']['start_year'])


n_written = 0
for case in available_cases(cfg, t0, t1):
    for region in regions:
        print(f"\n== {case['name']} / {region} ==")
        bbox = cfg['regions'][region]
        e_times, e_lat, e_lon, e_prect = read_e3sm_precip(case, t0, t1, bbox)
        print(f"  E3SM: {len(e_times)} records, {len(e_lat)} cells")

        aorc = st4 = None
        if use_aorc:
            try:
                a_times, a_vals = read_aorc_at_points(aorc_dir, t0, t1, e_lat, e_lon)
                aorc = sum_into_intervals(a_times, a_vals, e_times)
            except Exception as e:
                print(f"  AORC SKIPPED ({type(e).__name__}: {e})")
        if use_st4:
            try:
                s_times, s_lat, s_lon, s_vals = read_st4(st4_dir, t0, t1, bbox)
                _, idx = cKDTree(np.column_stack([s_lat.ravel(), s_lon.ravel()])).query(
                    np.column_stack([e_lat, e_lon]))
                st4 = sum_into_intervals(s_times, s_vals.reshape(len(s_times), -1)[:, idx], e_times)
            except Exception as e:
                print(f"  Stage IV SKIPPED ({type(e).__name__}: {e})")

        out = nc_path(case, region)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with nc.Dataset(out, 'w') as ds:
            ds.createDimension('time', len(e_times))
            ds.createDimension('ncol', len(e_lat))
            tv = ds.createVariable('time', 'f8', ('time',))
            tv.units = "seconds since 1970-01-01 00:00:00"
            tv.calendar = "standard"
            tv[:] = [(t - datetime(1970, 1, 1)).total_seconds() for t in e_times]
            ds.createVariable('lat', 'f8', ('ncol',))[:] = e_lat
            ds.createVariable('lon', 'f8', ('ncol',))[:] = e_lon
            v = ds.createVariable('e3sm_prect_mmhr', 'f4', ('time', 'ncol'), fill_value=np.nan)
            v[:] = e_prect
            v.units, v.long_name = "mm/hr", "E3SM PRECT, instantaneous rate"
            for name, arr, src in (('aorc_precip_mm', aorc, 'AORC'), ('st4_precip_mm', st4, 'Stage IV')):
                if arr is None:
                    continue
                v = ds.createVariable(name, 'f4', ('time', 'ncol'), fill_value=np.nan)
                v[:] = arr
                v.units = "mm"
                v.long_name = (f"{src} summed over each E3SM interval (t[i-1], t[i]], nearest "
                               f"obs cell to each E3SM cell; row 0 = 0; NaN if any hour missing")
            if aorc is not None:
                valid = np.all(np.isfinite(aorc[1:]), axis=0) if len(e_times) > 1 \
                    else np.isfinite(aorc[0])
                ds.createVariable('aorc_valid', 'i1', ('ncol',))[:] = valid.astype(np.int8)
                print(f"  AORC valid at {int(valid.sum())}/{len(valid)} cells")
            ds.case_name = case['name']
            ds.resolution = case['resolution']
            ds.region = region
            ds.event_label = cfg['event_label']
        print(f"  Written: {out}")
        n_written += 1

if n_written == 0:
    raise SystemExit("No comparison files written (no case covers the event window).")
