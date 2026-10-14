"""
build_comparison_nc.py

Writes the comparison netCDFs that plot_multicase_overlay.py reads: one per
case and region, at overlay.comparison_nc_tmpl. Always overwrites, so a
stale file from an older schema can't be picked up by mistake.

For each case and region (plots.overlay.regions):
  - E3SM PRECT (mm/hr, instantaneous) on the native mesh, cropped to the region.
  - Each obs product the event uses (obs: in defaults.yaml) sampled at the
    obs cell nearest each E3SM cell, as <name>_precip_mm with <name> the
    lowercase product name (aorc, stageiv, nclimgrid).
      Hourly (AORC, Stage IV): values SUMMED into each E3SM output interval
      (t[i-1], t[i]] on the E3SM time axis. Row 0 is 0.
      Daily (NClimGrid): one row per day on its own axis <name>_time (period
      end), since days can't be split into 3-hr intervals.
  - obs_valid (ncol): 1 where every obs product written has data for every
    record (records missing at every cell, i.e. a missing hour or file, are
    ignored here). The overlay uses it so E3SM and obs are compared over the same
    cells (obs are NaN over the ocean and outside CONUS).
  - Global attribute obs_products: comma-separated product names written.

Run:  python build_comparison_nc.py      (event: STORYLINE_CONFIG, see storyline_config.py)
"""

import os
from datetime import datetime

import numpy as np
import netCDF4 as nc

from storyline_config import load_event, event_window, plot_opts
from storyline_io import (read_e3sm_precip, read_obs_at_points, sum_into_intervals,
                          available_cases, obs_products, obs_label)

cfg = load_event()
t0, t1 = event_window(cfg)
regions = plot_opts(cfg, 'overlay').get('regions', ['WUS'])
ov = cfg['overlay']
tag_map = ov.get('resolution_tag_map', {})
products = obs_products(cfg, t0, t1)


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

        obs = []      # (name, times or None for the E3SM axis, values)
        for name in products:
            try:
                o_times, o_vals, step = read_obs_at_points(cfg, name, t0, t1, e_lat, e_lon)
                if step == 1:
                    obs.append((name, None, sum_into_intervals(o_times, o_vals, e_times)))
                else:
                    obs.append((name, o_times, o_vals))
            except Exception as e:
                print(f"  {name} SKIPPED ({type(e).__name__}: {e})")

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
            valid = np.ones(len(e_lat), dtype=bool)
            for name, o_times, arr in obs:
                key = name.lower()
                if o_times is None:
                    v = ds.createVariable(f"{key}_precip_mm", 'f4', ('time', 'ncol'), fill_value=np.nan)
                    v.long_name = (f"{obs_label(name)} summed over each E3SM interval (t[i-1], t[i]], "
                                   f"nearest obs cell to each E3SM cell; row 0 = 0; NaN if any hour missing")
                    full = arr[1:] if len(e_times) > 1 else arr
                else:
                    ds.createDimension(f"{key}_time", len(o_times))
                    ot = ds.createVariable(f"{key}_time", 'f8', (f"{key}_time",))
                    ot.units = "seconds since 1970-01-01 00:00:00"
                    ot.calendar = "standard"
                    ot.long_name = "end of each 24-hr obs period"
                    ot[:] = [(t - datetime(1970, 1, 1)).total_seconds() for t in o_times]
                    v = ds.createVariable(f"{key}_precip_mm", 'f4', (f"{key}_time", 'ncol'),
                                          fill_value=np.nan)
                    v.long_name = (f"{obs_label(name)} daily total ending at {key}_time, nearest obs "
                                   f"cell to each E3SM cell")
                    full = arr
                v[:] = arr
                v.units = "mm"
                # A record missing everywhere (a whole missing hour/file) is skipped here
                # so one gap doesn't invalidate every cell; the overlay warns about it.
                rows = np.isfinite(full).any(axis=1)
                ok = (np.all(np.isfinite(full[rows]), axis=0) if rows.any()
                      else np.zeros(len(e_lat), dtype=bool))
                print(f"  {obs_label(name)} valid at {int(ok.sum())}/{len(ok)} cells")
                valid &= ok
            if obs:
                ds.createVariable('obs_valid', 'i1', ('ncol',))[:] = valid.astype(np.int8)
                print(f"  All obs valid at {int(valid.sum())}/{len(valid)} cells")
            ds.obs_products = ",".join(n for n, _, _ in obs)
            ds.case_name = case['name']
            ds.resolution = case['resolution']
            ds.region = region
            ds.event_label = cfg['event_label']
        print(f"  Written: {out}")
        n_written += 1

if n_written == 0:
    raise SystemExit("No comparison files written (no case covers the event window).")
