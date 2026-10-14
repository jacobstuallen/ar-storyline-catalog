"""
storyline_io.py

Data readers and accumulation helpers shared by every plot script.

Conventions
  - region bbox order is (min_lat, min_lon, max_lat, max_lon), same as the
    `regions:` block in defaults.yaml. Map extents are
    [min_lon, max_lon, min_lat, max_lat] (cartopy order).
  - Windows are (t0, t1) datetimes in UTC.
  - E3SM PRECT is an instantaneous rate. Records with t0 <= t <= t1 are read,
    and depths come from a trapezoidal time integral.
  - AORC and Stage IV values are hourly accumulations stamped at the END of
    the hour, so a window (t0, t1] sums hours with t0 < t <= t1.
    VERIFY: if AORC stamps the start of the hour, flip OBS_STAMP_AT_END.
  - Times are decoded with cftime using each file's own calendar (E3SM is
    noleap), then converted to plain datetimes. No event here crosses a
    Feb 29 in a leap year, where noleap and real-calendar spacing differ.

Schemas assumed (from earlier ncdump -h output in this project):
  AORC:     <aorc_dir>/<year>/APCP_surface_<year>_<MM>.nc, hourly, 1-D
            latitude/longitude, APCP_surface(time, latitude, longitude) in mm.
  Stage IV: <st4_dir>/PrecipitationST4_<YYYYMMDD>.nc, x/y on a polar
            stereographic grid described by the Polar_Stereographic variable.
  E3SM:     <e3sm_dir>/*.<stream>.*.nc, unstructured (time, ncol), 1-D lat/lon.
"""

import glob
import os
import re
from datetime import date, datetime, timedelta

import numpy as np
import netCDF4 as nc
import cftime

AORC_EPOCH_YEAR = 1979                 # AORC period of record starts Feb 1979
ST4_EPOCH = date(2002, 1, 1)           # Stage IV period of record
OBS_STAMP_AT_END = True                # see module docstring


# ── time ─────────────────────────────────────────────────────────────────
def nc_times(time_var):
    """Decode a netCDF time variable to an array of python datetimes."""
    cal = getattr(time_var, 'calendar', 'standard')
    vals = cftime.num2date(np.asarray(time_var[:]), time_var.units, calendar=cal,
                           only_use_cftime_datetimes=True)
    return np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
                     for d in np.atleast_1d(vals)])


def hours_since(times, t0):
    return np.array([(t - t0).total_seconds() / 3600.0 for t in times])


def obs_mask(times, t0, t1):
    """Boolean mask for hourly obs inside the window, per OBS_STAMP_AT_END."""
    if OBS_STAMP_AT_END:
        return np.array([(t > t0) and (t <= t1) for t in times], dtype=bool)
    return np.array([(t >= t0) and (t < t1) for t in times], dtype=bool)


def aorc_available(t0):
    return t0.year >= AORC_EPOCH_YEAR


# ── E3SM ─────────────────────────────────────────────────────────────────
_FNAME_DATE = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})(?:-(\d{5}))?\.nc$")
_index_cache = {}


def _e3sm_file_index(case, t0, t1):
    """[(path, times, keep_idx)] for files with records in [t0, t1]. Filenames
    with a date stamp are pre-filtered so months of spin-up output are not
    opened; files without one are always opened."""
    key = (case['e3sm_dir'], case.get('e3sm_stream', 'eam.h0'), t0, t1)
    if key in _index_cache:
        return _index_cache[key]
    stream = case.get('e3sm_stream', 'eam.h0')
    pattern = os.path.join(case['e3sm_dir'], f"*.{stream}.*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    lo, hi = t0.date() - timedelta(days=35), t1.date()
    out = []
    for f in files:
        m = _FNAME_DATE.search(f)
        if m:
            fd = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if not (lo <= fd <= hi):
                continue
        with nc.Dataset(f) as ds:
            if 'time' not in ds.variables:
                continue
            times = nc_times(ds.variables['time'])
        keep = np.where((times >= t0) & (times <= t1))[0]
        if len(keep):
            out.append((f, times, keep))
    _index_cache[key] = out
    return out


def case_coverage(case, t0, t1):
    """(first, last, n_records) of E3SM records inside [t0, t1], or None."""
    try:
        idx = _e3sm_file_index(case, t0, t1)
    except FileNotFoundError:
        return None
    if not idx:
        return None
    times = np.unique(np.concatenate([t[k] for _, t, k in idx]))
    return times[0], times[-1], len(times)


def available_cases(cfg, t0, t1, slack_hours=None, required=True):
    """Cases whose output spans [t0, t1] within `slack_hours` (default: one
    output interval) at each end. Prints why any case is dropped, so a
    still-running 30x32 or a wrong `time:` block shows up in the log. With
    required=True (default) and no case left, the script exits with an error."""
    slack = timedelta(hours=slack_hours if slack_hours is not None
                      else cfg['time'].get('hours_step', 3))
    ok = []
    for case in cfg['cases']:
        cov = case_coverage(case, t0, t1)
        if cov is None:
            print(f"  SKIP case {case['name']}: no output in {t0} to {t1}")
            continue
        first, last, n = cov
        if first > t0 + slack or last < t1 - slack:
            print(f"  SKIP case {case['name']}: output covers {first} to {last}, "
                  f"window is {t0} to {t1}")
            continue
        ok.append(case)
    if required and not ok:
        raise SystemExit(f"No case has output covering {t0} to {t1}; check `time:` and e3sm_dir.")
    return ok


def read_e3sm(case, t0, t1, region_bbox, varnames=('PRECT',)):
    """(times, lat, lon, {var: (ntime, ncell)}) for records in [t0, t1],
    cells inside region_bbox. Duplicate timestamps (restart overlap) dropped."""
    min_lat, min_lon, max_lat, max_lon = region_bbox
    idx = _e3sm_file_index(case, t0, t1)
    if not idx:
        raise ValueError(f"No E3SM records in {t0} to {t1} for case '{case['name']}'.")

    cell_idx = lat = lon = None
    all_t, all_v = [], {v: [] for v in varnames}
    for f, times, keep in idx:
        with nc.Dataset(f) as ds:
            if cell_idx is None:
                lat_full = np.asarray(ds.variables['lat'][:])
                lon_full = np.asarray(ds.variables['lon'][:])
                lon_full = np.where(lon_full > 180, lon_full - 360, lon_full)
                cell_idx = np.where((lat_full >= min_lat) & (lat_full <= max_lat) &
                                    (lon_full >= min_lon) & (lon_full <= max_lon))[0]
                if len(cell_idx) == 0:
                    raise ValueError(f"No E3SM cells inside {region_bbox} for '{case['name']}'.")
                lat, lon = lat_full[cell_idx], lon_full[cell_idx]
            sl = slice(keep[0], keep[-1] + 1)
            for v in varnames:
                block = np.asarray(ds.variables[v][sl])[keep - keep[0]]
                all_v[v].append(block[:, cell_idx])
        all_t.append(times[keep])

    t_cat = np.concatenate(all_t)
    t_u, first = np.unique(t_cat, return_index=True)
    vals = {v: np.concatenate(all_v[v], axis=0)[first] for v in varnames}
    return t_u, lat, lon, vals


def read_e3sm_precip(case, t0, t1, region_bbox):
    """(times, lat, lon, PRECT in mm/hr)."""
    times, lat, lon, v = read_e3sm(case, t0, t1, region_bbox, ('PRECT',))
    return times, lat, lon, v['PRECT'] * 3.6e6


def cumulative_depth(times, rate_mmhr):
    """Trapezoidal running depth (mm), same shape as rate, 0 at times[0]."""
    dt = np.diff(hours_since(times, times[0]))
    dt = dt.reshape((-1,) + (1,) * (rate_mmhr.ndim - 1))
    incr = 0.5 * (rate_mmhr[:-1] + rate_mmhr[1:]) * dt
    return np.concatenate([np.zeros((1,) + rate_mmhr.shape[1:]), np.cumsum(incr, axis=0)])


def total_depth(times, rate_mmhr):
    """Trapezoidal total depth (mm) per cell."""
    return cumulative_depth(times, rate_mmhr)[-1]


def window_depth(times, rate_mmhr, t0, t1):
    """Depth (mm) per cell over [t0, t1], with the cumulative curve linearly
    interpolated to window edges that fall between records."""
    th = hours_since(times, t0)
    L = (t1 - t0).total_seconds() / 3600.0
    if th[0] > 0 or th[-1] < L:
        raise ValueError(f"E3SM records span {times[0]} to {times[-1]}, which does not "
                         f"cover {t0} to {t1}.")
    cum = cumulative_depth(times, rate_mmhr)

    def at(x):
        i = int(np.clip(np.searchsorted(th, x, side='right') - 1, 0, len(th) - 2))
        w = (x - th[i]) / (th[i + 1] - th[i])
        return cum[i] + w * (cum[i + 1] - cum[i])
    return at(L) - at(0.0)


# ── AORC ─────────────────────────────────────────────────────────────────
def _aorc_months(t0, t1):
    out, d = [], (t0 + timedelta(hours=1)).replace(day=1, hour=0)
    while d <= t1:
        out.append((d.year, d.month))
        d = (d + timedelta(days=32)).replace(day=1)
    return out


def iter_aorc(aorc_dir, t0, t1, region_bbox, buffer=0.5, chunk_hours=24):
    """Yield (lat, lon, times, block) with block (nt, nlat, nlon) float32 mm,
    NaN for missing, cropped to region_bbox + buffer degrees, `chunk_hours`
    at a time so large regions never sit in memory as a full stack."""
    min_lat, min_lon, max_lat, max_lon = region_bbox
    lat_sl = lon_sl = lat = lon = None
    n_yield = 0
    for y, m in _aorc_months(t0, t1):
        fpath = os.path.join(aorc_dir, str(y), f"APCP_surface_{y}_{m:02d}.nc")
        if not os.path.isfile(fpath):
            print(f"  SKIP (no AORC file): {fpath}")
            continue
        with nc.Dataset(fpath) as ds:
            if lat_sl is None:
                la = np.asarray(ds.variables['latitude'][:])
                lo = np.asarray(ds.variables['longitude'][:])
                iy = np.where((la >= min_lat - buffer) & (la <= max_lat + buffer))[0]
                ix = np.where((lo >= min_lon - buffer) & (lo <= max_lon + buffer))[0]
                if len(iy) == 0 or len(ix) == 0:
                    raise ValueError(f"AORC grid does not overlap {region_bbox}.")
                lat_sl, lon_sl = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
                lat, lon = la[lat_sl], lo[lon_sl]
            times = nc_times(ds.variables['time'])
            keep = np.where(obs_mask(times, t0, t1))[0]
            var = ds.variables['APCP_surface']
            for s in range(0, len(keep), chunk_hours):
                k = keep[s:s + chunk_hours]
                block = var[k[0]:k[-1] + 1, lat_sl, lon_sl]
                block = np.ma.filled(block.astype(np.float32), np.nan)[k - k[0]]
                n_yield += 1
                yield lat, lon, times[k], block
    if n_yield == 0:
        raise FileNotFoundError(f"No AORC data in {aorc_dir} for {t0} to {t1}.")


def _check_hours(n_got, t0, t1, what="AORC"):
    n_exp = int(round((t1 - t0).total_seconds() / 3600.0))
    if n_got != n_exp:
        print(f"  WARNING: {what} has {n_got} hourly records in {t0} to {t1}, expected {n_exp}.")


def read_aorc_sum(aorc_dir, t0, t1, region_bbox, buffer=0.5):
    """(lat, lon, total_mm_2d). Cells NaN for every hour stay NaN."""
    total = count = lat = lon = None
    n = 0
    for lat, lon, times, block in iter_aorc(aorc_dir, t0, t1, region_bbox, buffer):
        if total is None:
            total = np.zeros(block.shape[1:], dtype=np.float64)
            count = np.zeros(block.shape[1:], dtype=np.int32)
        total += np.nansum(block, axis=0)
        count += np.isfinite(block).sum(axis=0)
        n += len(times)
    _check_hours(n, t0, t1)
    return lat, lon, np.where(count > 0, total, np.nan)


def read_aorc_stack(aorc_dir, t0, t1, region_bbox, buffer=0.5):
    """(times, lat, lon, stack (nt, nlat, nlon)). Small regions only."""
    parts, tparts, lat, lon = [], [], None, None
    for lat, lon, times, block in iter_aorc(aorc_dir, t0, t1, region_bbox, buffer):
        parts.append(block)
        tparts.append(times)
    times = np.concatenate(tparts)
    _check_hours(len(times), t0, t1)
    return times, lat, lon, np.concatenate(parts, axis=0)


def read_aorc_rolling_max(aorc_dir, t0, t1, region_bbox, window_hours=24, buffer=0.5):
    """(lat, lon, max_1hr_mm, max_window_mm), streamed day by day."""
    w = int(window_hours)
    max1 = maxw = count = tail = lat = lon = None
    n = 0
    for lat, lon, times, block in iter_aorc(aorc_dir, t0, t1, region_bbox, buffer):
        if max1 is None:
            max1 = np.full(block.shape[1:], -np.inf, dtype=np.float32)
            maxw = np.full(block.shape[1:], -np.inf, dtype=np.float32)
            count = np.zeros(block.shape[1:], dtype=np.int32)
            tail = np.zeros((0,) + block.shape[1:], dtype=np.float32)
        n += len(times)
        count += np.isfinite(block).sum(axis=0)
        filled = np.nan_to_num(block, nan=0.0)
        max1 = np.maximum(max1, filled.max(axis=0))
        comb = np.concatenate([tail, filled], axis=0)
        if comb.shape[0] >= w:
            cs = np.concatenate([np.zeros((1,) + comb.shape[1:], np.float32),
                                 np.cumsum(comb, axis=0, dtype=np.float32)])
            maxw = np.maximum(maxw, (cs[w:] - cs[:-w]).max(axis=0))
        tail = comb[-(w - 1):] if w > 1 else comb[:0]
    _check_hours(n, t0, t1)
    bad = count == 0
    max1 = np.where(bad, np.nan, max1)
    maxw = np.where(bad | ~np.isfinite(maxw), np.nan, maxw)
    return lat, lon, max1, maxw


def read_aorc_at_points(aorc_dir, t0, t1, plat, plon):
    """(times, vals (nt, npts)): AORC at the grid cell nearest each point.
    Points off the AORC grid get NaN. Streams day by day."""
    pad = 0.1
    bbox = (np.min(plat) - pad, np.min(plon) - pad, np.max(plat) + pad, np.max(plon) + pad)
    iy = ix = valid = None
    tparts, vparts = [], []
    for lat, lon, times, block in iter_aorc(aorc_dir, t0, t1, bbox, buffer=0.05):
        if iy is None:
            dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
            fy = (plat - lat[0]) / dlat
            fx = (plon - lon[0]) / dlon
            valid = (fy >= -0.5) & (fy <= len(lat) - 0.5) & (fx >= -0.5) & (fx <= len(lon) - 0.5)
            iy = np.clip(np.rint(fy).astype(int), 0, len(lat) - 1)
            ix = np.clip(np.rint(fx).astype(int), 0, len(lon) - 1)
        v = block[:, iy, ix]
        v[:, ~valid] = np.nan
        tparts.append(times)
        vparts.append(v)
    times = np.concatenate(tparts)
    _check_hours(len(times), t0, t1)
    return times, np.concatenate(vparts, axis=0)


# ── Stage IV ─────────────────────────────────────────────────────────────
def _st4_transformer(ds):
    import pyproj
    gm = ds.variables['Polar_Stereographic']
    lat_0 = getattr(gm, 'latitude_of_projection_origin', 90.0)
    lon_0 = getattr(gm, 'straight_vertical_longitude_from_pole', -105.0)
    lat_ts = getattr(gm, 'standard_parallel', 60.0)
    a = getattr(gm, 'semi_major_axis', 6371200.0)
    crs = pyproj.CRS.from_proj4(f"+proj=stere +lat_0={lat_0} +lat_ts={lat_ts} +lon_0={lon_0} "
                                f"+k=1 +x_0=0 +y_0=0 +a={a} +b={a} +units=m +no_defs")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return fwd, inv


def read_st4(st4_dir, t0, t1, region_bbox, buffer_m=50000.0):
    """(times, lat2d, lon2d, stack mm). Daily files of hourly records."""
    min_lat, min_lon, max_lat, max_lon = region_bbox
    lat2d = lon2d = xs = ys = None
    tparts, vparts = [], []
    d = t0.date()
    while d <= t1.date():
        fpath = os.path.join(st4_dir, f"PrecipitationST4_{d:%Y%m%d}.nc")
        d += timedelta(days=1)
        if not os.path.isfile(fpath):
            print(f"  SKIP (no Stage IV file): {fpath}")
            continue
        with nc.Dataset(fpath) as ds:
            if xs is None:
                x, y = np.asarray(ds.variables['x'][:]), np.asarray(ds.variables['y'][:])
                fwd, inv = _st4_transformer(ds)
                bx, by = fwd.transform([min_lon, max_lon, min_lon, max_lon],
                                       [min_lat, min_lat, max_lat, max_lat])
                ix = np.where((x >= min(bx) - buffer_m) & (x <= max(bx) + buffer_m))[0]
                iy = np.where((y >= min(by) - buffer_m) & (y <= max(by) + buffer_m))[0]
                xs, ys = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)
                xx, yy = np.meshgrid(x[xs], y[ys])
                lon2d, lat2d = inv.transform(xx, yy)
            vname = ('Total_precipitation_surface_1_Hour_Accumulation'
                     if 'Total_precipitation_surface_1_Hour_Accumulation' in ds.variables
                     else 'precip')
            times = nc_times(ds.variables['time'])
            keep = np.where(obs_mask(times, t0, t1))[0]
            if len(keep):
                block = np.ma.filled(ds.variables[vname][:, ys, xs].astype(np.float32), np.nan)
                tparts.append(times[keep])
                vparts.append(block[keep])
    if not tparts:
        raise FileNotFoundError(f"No Stage IV data in {st4_dir} for {t0} to {t1}.")
    times = np.concatenate(tparts)
    order = np.argsort(times)
    return times[order], lat2d, lon2d, np.concatenate(vparts, axis=0)[order]


# ── obs onto the E3SM time axis ───────────────────────────────────────────
def sum_into_intervals(obs_times, obs_vals, target_times):
    """Sum hourly obs into the intervals (target[i-1], target[i]] of the E3SM
    time axis. Row 0 is 0 (nothing has accumulated at the first record). An
    interval missing any hour is NaN. obs_vals is (nt_obs, npts)."""
    ref = target_times[0]
    oh = hours_since(obs_times, ref)
    th = hours_since(target_times, ref)
    out = np.full((len(target_times), obs_vals.shape[1]), np.nan, dtype=np.float32)
    out[0] = 0.0
    step = np.median(np.diff(oh)) if len(oh) > 1 else 1.0
    for i in range(1, len(th)):
        a = np.searchsorted(oh, th[i - 1] + 1e-6, side='left')
        b = np.searchsorted(oh, th[i] + 1e-6, side='left')
        if (b - a) == int(round((th[i] - th[i - 1]) / step)):
            out[i] = obs_vals[a:b].sum(axis=0)
    return out
