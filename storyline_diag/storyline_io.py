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
  - NClimGrid is daily. Each value covers the 24 hours ending in the early
    morning of its date (NCEI metadata). Here that is taken as ending at
    obs.nclimgrid_day_end_utc (default 12 UTC = 04 PST). A window uses every
    day whose period MIDPOINT falls in [t0, t1), so NClimGrid totals can be
    offset from the window by up to 12 hr at each end. The period actually
    used is printed and returned.
  - Times are decoded with cftime using each file's own calendar (E3SM is
    noleap), then converted to plain datetimes. No event here crosses a
    Feb 29 in a leap year, where noleap and real-calendar spacing differ.

Which obs an event uses: obs_products(cfg, t0, t1), from the `obs:` block
in defaults.yaml. Scripts call the read_obs_* functions with a product name
("AORC", "StageIV", "NClimGrid"); every one returns 2-D lat/lon arrays.

Schemas assumed (from ncdump -h output in this project):
  AORC:      <aorc_dir>/<year>/APCP_surface_<year>_<MM>.nc, hourly, 1-D
             latitude/longitude, APCP_surface(time, latitude, longitude) in mm.
  Stage IV:  <st4_dir>/PrecipitationST4_<YYYYMMDD>.nc, pcp(hour, g5_y_1,
             g5_x_0) in mm/hour, no coordinate or time variables. The HRAP grid
             is rebuilt in st4_grid(); hour k ends at obs.stageiv.first_hour_end_utc + k.
  NClimGrid: <nclimgrid_dir>/<year>/ncdd-prcp-<YYYYMM>-grd-scaled.nc
             (nClimGrid-Daily v1.0.0, prcp only), 1-D lat/lon,
             prcp(time, lat, lon) in mm, time in days since 1800-01-01.
  E3SM:      <e3sm_dir>/*.<stream>.*.nc, unstructured (time, ncol), 1-D lat/lon.
"""

import glob
import os
import re
from datetime import date, datetime, timedelta

import numpy as np
import netCDF4 as nc
import cftime

OBS_START = {                          # first date each product covers
    'AORC': datetime(1979, 2, 1),
    'StageIV': datetime(2002, 1, 1),
    'NClimGrid': datetime(1951, 1, 1),
}
OBS_LABELS = {'AORC': 'AORC', 'StageIV': 'Stage IV', 'NClimGrid': 'NClimGrid'}
OBS_DIR_KEYS = {'AORC': 'aorc_dir', 'StageIV': 'st4_dir', 'NClimGrid': 'nclimgrid_dir'}
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




# ── generic reductions over streamed obs blocks ──────────────────────────
# Every iter_* below yields (lat, lon, times, block): lat/lon 1-D (regular
# grid) or 2-D (Stage IV), times = period END stamps, block (nt, ny, nx)
# float32 mm with NaN for missing.
def _as_2d(lat, lon):
    if np.ndim(lat) == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d
    return lat, lon


def _check_count(n_got, n_exp, t0, t1, what):
    if n_got != n_exp:
        print(f"  WARNING: {what} has {n_got} records in {t0} to {t1}, expected {n_exp}.")


def _sum_blocks(it):
    """(lat2d, lon2d, total, n_records). Cells NaN for every record stay NaN."""
    total = count = lat = lon = None
    n = 0
    for lat, lon, times, block in it:
        if total is None:
            total = np.zeros(block.shape[1:], dtype=np.float64)
            count = np.zeros(block.shape[1:], dtype=np.int32)
        total += np.nansum(block, axis=0)
        count += np.isfinite(block).sum(axis=0)
        n += len(times)
    lat2d, lon2d = _as_2d(lat, lon)
    return lat2d, lon2d, np.where(count > 0, total, np.nan), n


def _stack_blocks(it):
    """(times, lat2d, lon2d, stack). Small regions only."""
    parts, tparts, lat, lon = [], [], None, None
    for lat, lon, times, block in it:
        parts.append(block)
        tparts.append(times)
    lat2d, lon2d = _as_2d(lat, lon)
    return np.concatenate(tparts), lat2d, lon2d, np.concatenate(parts, axis=0)


def _rolling_max_blocks(it, w):
    """(lat2d, lon2d, max_1_record, max_w_records, n_records), streamed."""
    max1 = maxw = count = tail = lat = lon = None
    n = 0
    for lat, lon, times, block in it:
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
    bad = count == 0
    max1 = np.where(bad, np.nan, max1)
    maxw = np.where(bad | ~np.isfinite(maxw), np.nan, maxw)
    lat2d, lon2d = _as_2d(lat, lon)
    return lat2d, lon2d, max1, maxw, n


def _nearest_index(lat, lon, plat, plon):
    """(flat index into the grid, valid) for the cell nearest each point.
    Regular 1-D grids use index arithmetic; 2-D grids use a KD-tree."""
    plat, plon = np.asarray(plat, float), np.asarray(plon, float)
    if np.ndim(lat) == 1:
        dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
        fy, fx = (plat - lat[0]) / dlat, (plon - lon[0]) / dlon
        valid = (fy >= -0.5) & (fy <= len(lat) - 0.5) & (fx >= -0.5) & (fx <= len(lon) - 0.5)
        iy = np.clip(np.rint(fy).astype(int), 0, len(lat) - 1)
        ix = np.clip(np.rint(fx).astype(int), 0, len(lon) - 1)
        return iy * len(lon) + ix, valid
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(np.column_stack([lat.ravel(), lon.ravel()])).query(
        np.column_stack([plat, plon]))
    spacing = np.nanmedian(np.abs(np.diff(lat, axis=0)))
    return idx, dist <= 2.0 * spacing


def _points_blocks(it, plat, plon):
    """(times, vals (nt, npts)) at the cell nearest each point; off-grid NaN."""
    idx = valid = None
    tparts, vparts = [], []
    for lat, lon, times, block in it:
        if idx is None:
            idx, valid = _nearest_index(lat, lon, plat, plon)
        v = block.reshape(len(times), -1)[:, idx]
        v[:, ~valid] = np.nan
        tparts.append(times)
        vparts.append(v)
    return np.concatenate(tparts), np.concatenate(vparts, axis=0)


def _points_bbox(plat, plon, pad=0.1):
    return (np.min(plat) - pad, np.min(plon) - pad, np.max(plat) + pad, np.max(plon) + pad)


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




# ── Stage IV ─────────────────────────────────────────────────────────────
# The PrecipitationST4_YYYYMMDD.nc files carry no coordinates or time axis:
# pcp(hour, g5_y_1, g5_x_0) in mm per hour on the 1121 x 881 HRAP grid.
# The grid is rebuilt from the standard Stage IV definition (polar
# stereographic, true at 60N, LoV 105W, sphere R = 6371.2 km, 4762.5 m
# spacing, first point 23.117N 119.023W at the SW corner). Its corners
# reproduce the published ones (53.509N 134.04W, 45.619N 59.959W).
ST4_NX, ST4_NY = 1121, 881
ST4_DX = 4762.5
ST4_LAT1, ST4_LON1 = 23.117, -119.023
_st4_grid_cache = {}


def st4_grid(flip_rows=False):
    """(lat2d, lon2d), shape (881, 1121). Row 0 is the southern edge unless
    flip_rows (GRIB scan mode 64, kept by the GRIB-to-netCDF conversion)."""
    if flip_rows in _st4_grid_cache:
        return _st4_grid_cache[flip_rows]
    import pyproj
    crs = pyproj.CRS.from_proj4("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=1 +x_0=0 +y_0=0 "
                                "+a=6371200 +b=6371200 +units=m +no_defs")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = fwd.transform(ST4_LON1, ST4_LAT1)
    xx, yy = np.meshgrid(x0 + ST4_DX * np.arange(ST4_NX), y0 + ST4_DX * np.arange(ST4_NY))
    lon2d, lat2d = inv.transform(xx, yy)
    if flip_rows:
        lat2d, lon2d = lat2d[::-1], lon2d[::-1]
    _st4_grid_cache[flip_rows] = (lat2d, lon2d)
    return lat2d, lon2d


def iter_st4(st4_dir, t0, t1, region_bbox, first_hour_end=1, flip_rows=False, buffer=0.3):
    """Yield (lat2d, lon2d, times, block) one daily file at a time, cropped to
    region_bbox + buffer degrees. Record k of the file for date D ends at
    D 00Z + (first_hour_end + k) hours."""
    min_lat, min_lon, max_lat, max_lon = region_bbox
    glat, glon = st4_grid(flip_rows)
    inside = ((glat >= min_lat - buffer) & (glat <= max_lat + buffer) &
              (glon >= min_lon - buffer) & (glon <= max_lon + buffer))
    rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError(f"Stage IV grid does not overlap {region_bbox}.")
    ys, xs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    lat2d, lon2d = glat[ys, xs], glon[ys, xs]
    n_yield = 0
    d = (t0 - timedelta(hours=first_hour_end + 23)).date()   # first file that can hold an hour > t0
    while datetime.combine(d, datetime.min.time()) + timedelta(hours=first_hour_end) <= t1:
        fpath = os.path.join(st4_dir, f"PrecipitationST4_{d:%Y%m%d}.nc")
        day0 = datetime.combine(d, datetime.min.time())
        d += timedelta(days=1)
        if not os.path.isfile(fpath):
            print(f"  SKIP (no Stage IV file): {fpath}")
            continue
        with nc.Dataset(fpath) as ds:
            v = ds.variables['pcp']
            if v.shape[1:] != (ST4_NY, ST4_NX):
                raise ValueError(f"{fpath}: pcp shape {v.shape}, expected (hour, {ST4_NY}, {ST4_NX}).")
            times = np.array([day0 + timedelta(hours=first_hour_end + k) for k in range(v.shape[0])])
            keep = np.where(obs_mask(times, t0, t1))[0]
            if len(keep) == 0:
                continue
            block = v[keep[0]:keep[-1] + 1]
            block = np.ma.filled(block.astype(np.float32), np.nan)
            if flip_rows:
                block = block[:, ::-1]
            block = block[keep - keep[0]][:, ys, xs]
            n_yield += 1
            yield lat2d, lon2d, times[keep], block
    if n_yield == 0:
        raise FileNotFoundError(f"No Stage IV data in {st4_dir} for {t0} to {t1}.")


# ── NClimGrid (daily) ────────────────────────────────────────────────────
def nclimgrid_days(t0, t1, day_end_utc=12):
    """(dates, period_start, period_end): NClimGrid dates whose 24-hr period
    (ending day_end_utc on that date) has its midpoint in [t0, t1)."""
    days = []
    d = (t0 - timedelta(days=1)).date()
    while datetime.combine(d, datetime.min.time()) <= t1 + timedelta(days=1):
        mid = datetime.combine(d, datetime.min.time()) + timedelta(hours=day_end_utc - 12)
        if t0 <= mid < t1:
            days.append(d)
        d += timedelta(days=1)
    if not days:
        raise ValueError(f"No NClimGrid day has its midpoint in {t0} to {t1}.")
    end = datetime.combine(days[-1], datetime.min.time()) + timedelta(hours=day_end_utc)
    start = datetime.combine(days[0], datetime.min.time()) + timedelta(hours=day_end_utc - 24)
    return days, start, end


def iter_nclimgrid(ncg_dir, days, region_bbox, day_end_utc=12, buffer=0.1):
    """Yield (lat, lon, period_end_times, block) one monthly file at a time."""
    min_lat, min_lon, max_lat, max_lon = region_bbox
    lat_sl = lon_sl = lat = lon = None
    months = sorted({(d.year, d.month) for d in days})
    want = set(days)
    n_yield = 0
    for y, m in months:
        fpath = os.path.join(ncg_dir, str(y), f"ncdd-prcp-{y}{m:02d}-grd-scaled.nc")
        if not os.path.isfile(fpath):
            print(f"  SKIP (no NClimGrid file): {fpath}")
            continue
        with nc.Dataset(fpath) as ds:
            if lat_sl is None:
                la = np.asarray(ds.variables['lat'][:])
                lo = np.asarray(ds.variables['lon'][:])
                iy = np.where((la >= min_lat - buffer) & (la <= max_lat + buffer))[0]
                ix = np.where((lo >= min_lon - buffer) & (lo <= max_lon + buffer))[0]
                if len(iy) == 0 or len(ix) == 0:
                    raise ValueError(f"NClimGrid grid does not overlap {region_bbox}.")
                lat_sl, lon_sl = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
                lat, lon = la[lat_sl], lo[lon_sl]
            dates = [t.date() for t in nc_times(ds.variables['time'])]
            keep = np.array([i for i, dd in enumerate(dates) if dd in want])
            if len(keep) == 0:
                continue
            block = ds.variables['prcp'][keep[0]:keep[-1] + 1, lat_sl, lon_sl]
            block = np.ma.filled(block.astype(np.float32), np.nan)[keep - keep[0]]
            ends = np.array([datetime.combine(dates[i], datetime.min.time()) +
                             timedelta(hours=day_end_utc) for i in keep])
            n_yield += 1
            yield lat, lon, ends, block
    if n_yield == 0:
        raise FileNotFoundError(f"No NClimGrid data in {ncg_dir} for {days[0]} to {days[-1]}.")


# ── which obs, and dataset-agnostic readers ───────────────────────────────
def obs_products(cfg, t0, t1):
    """Obs product names to use for a window, from the `obs:` block. Primaries
    are used when their record covers t0 and their data dir is set; backups
    only when no primary qualifies, unless obs.always_show_backup."""
    o = cfg.get('obs', {}) or {}
    data = cfg.get('data', {}) or {}

    def ok(name):
        if name not in OBS_START:
            raise KeyError(f"Unknown obs product '{name}'. Known: {sorted(OBS_START)}")
        if t0 < OBS_START[name]:
            print(f"  {OBS_LABELS[name]} n/a: record starts {OBS_START[name]:%Y-%m-%d}")
            return False
        if not data.get(OBS_DIR_KEYS[name]):
            print(f"  {OBS_LABELS[name]} n/a: data.{OBS_DIR_KEYS[name]} not set")
            return False
        return True

    prim = [n for n in o.get('primary', ['AORC', 'StageIV']) if ok(n)]
    back = [n for n in o.get('backup', ['NClimGrid']) if ok(n)]
    use = prim + back if (not prim or o.get('always_show_backup')) else prim
    print(f"  Obs used: {', '.join(OBS_LABELS[n] for n in use) or 'none'}")
    return use


def obs_label(name):
    return OBS_LABELS[name]


def obs_is_daily(name):
    return name == 'NClimGrid'


def _day_end(cfg):
    return int((cfg.get('obs', {}) or {}).get('nclimgrid_day_end_utc', 12))


def _obs_iter(cfg, name, t0, t1, bbox):
    """(iterator, expected record count, period_start, period_end)."""
    d = (cfg.get('data', {}) or {}).get(OBS_DIR_KEYS[name])
    n_hours = int(round((t1 - t0).total_seconds() / 3600.0))
    if name == 'AORC':
        return iter_aorc(d, t0, t1, bbox), n_hours, t0, t1
    if name == 'StageIV':
        s4 = (cfg.get('obs', {}) or {}).get('stageiv', {}) or {}
        return (iter_st4(d, t0, t1, bbox, int(s4.get('first_hour_end_utc', 1)),
                         bool(s4.get('flip_rows', False))), n_hours, t0, t1)
    if name == 'NClimGrid':
        days, p0, p1 = nclimgrid_days(t0, t1, _day_end(cfg))
        if (p0, p1) != (t0, t1):
            print(f"  NClimGrid: {len(days)} days, period {p0:%Y-%m-%d %H}Z to {p1:%Y-%m-%d %H}Z "
                  f"(window {t0:%Y-%m-%d %H}Z to {t1:%Y-%m-%d %H}Z)")
        return iter_nclimgrid(d, days, bbox, _day_end(cfg)), len(days), p0, p1
    raise KeyError(f"Unknown obs product '{name}'.")


def read_obs_sum(cfg, name, t0, t1, bbox):
    """{'name', 'label', 'kind': 'grid', 'lat', 'lon' (2-D), 'val' (mm),
    'period': (start, end)} for the window total."""
    it, n_exp, p0, p1 = _obs_iter(cfg, name, t0, t1, bbox)
    lat, lon, tot, n = _sum_blocks(it)
    _check_count(n, n_exp, p0, p1, obs_label(name))
    return {'name': name, 'label': obs_label(name), 'kind': 'grid',
            'lat': lat, 'lon': lon, 'val': tot, 'period': (p0, p1)}


def read_obs_peaks(cfg, name, t0, t1, bbox, window_hours=24):
    """Like read_obs_sum, with 'h' = max single hour and 'd' = max running
    window_hours total. Daily products: 'h' is None and 'd' is the wettest
    single day (fixed days, so it can miss a peak that straddles two)."""
    it, n_exp, p0, p1 = _obs_iter(cfg, name, t0, t1, bbox)
    w = 1 if obs_is_daily(name) else int(window_hours)
    lat, lon, m1, mw, n = _rolling_max_blocks(it, w)
    _check_count(n, n_exp, p0, p1, obs_label(name))
    return {'name': name, 'label': obs_label(name), 'kind': 'grid', 'lat': lat, 'lon': lon,
            'h': None if obs_is_daily(name) else m1, 'd': mw, 'period': (p0, p1)}


def read_obs_stack(cfg, name, t0, t1, bbox):
    """(end_times, lat2d, lon2d, stack mm, step_hours). Small regions only."""
    it, n_exp, p0, p1 = _obs_iter(cfg, name, t0, t1, bbox)
    times, lat, lon, stack = _stack_blocks(it)
    _check_count(len(times), n_exp, p0, p1, obs_label(name))
    return times, lat, lon, stack, (24 if obs_is_daily(name) else 1)


def read_obs_at_points(cfg, name, t0, t1, plat, plon):
    """(end_times, vals (nt, npts), step_hours) at the obs cell nearest each
    point. Points off the grid (or farther than 2 cells away) get NaN."""
    bbox = _points_bbox(plat, plon)
    it, n_exp, p0, p1 = _obs_iter(cfg, name, t0, t1, bbox)
    times, vals = _points_blocks(it, plat, plon)
    _check_count(len(times), n_exp, p0, p1, obs_label(name))
    return times, vals, (24 if obs_is_daily(name) else 1)


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
