"""
process_io.py

Shared helpers for process_diag. Reuses the storyline_diag config and file
index (case paths, dates, storm windows) and adds:
  - readers for 3D fields on one or more model levels, NORCAL columns only
  - second-order structure functions on the native unstructured columns

Structure function: S2(r) = mean over column pairs at separation r of
(f_i - f_j)^2. No regridding. A spectrum E(k) ~ k^-n corresponds to
S2 ~ r^(n-1) for 1 < n < 3, so k^-5/3 -> r^(2/3) and k^-3 -> r^2.

Pairs depend only on the mesh, so they are built once per case:
  - every pair closer than r_exact_km (KD-tree), plus
  - uniformly sampled random pairs at larger separations.
r_exact_km is snapped to a bin edge, so no bin mixes the two samples.
"""

import os
import sys
import warnings
from datetime import timedelta

import numpy as np
import netCDF4 as nc
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.normpath(os.path.join(HERE, "..", "storyline_diag"))
sys.path.insert(0, DIAG_DIR)

import storyline_io as sio                      # noqa: E402
from storyline_config import load_event         # noqa: E402
from obs_compare import parse_window            # noqa: E402

R_EARTH_KM = 6371.0


# ── config ───────────────────────────────────────────────────────────────
def load_process_cfg(path=os.path.join(HERE, "process.yaml")):
    with open(path) as f:
        return yaml.safe_load(f)


def load_events(pcfg):
    """Resolved storyline_diag configs for every event in process.yaml."""
    return [load_event(os.path.join(DIAG_DIR, p)) for p in pcfg['events']]


def event_year(default):
    """PROCESS_EVENT (set by run_process.py) if present, else `default`."""
    return int(os.environ.get("PROCESS_EVENT", default))


def find_event(pcfg, year):
    """The resolved config whose event_label starts with `year`."""
    for p in pcfg['events']:
        if p.split('.')[1] == str(year):
            return load_event(os.path.join(DIAG_DIR, p))
    raise KeyError(f"No event for {year} in process.yaml events.")


def analysis_window(cfg, which):
    """(t0, t1): the event's storm_window, or the full time: block."""
    if which == "storm":
        win = (cfg.get('observed_comparison') or {}).get('storm_window')
        if win is None:
            raise KeyError(f"{cfg['event_label']}: no observed_comparison.storm_window.")
        return parse_window(win)
    from storyline_config import event_window
    return event_window(cfg)


def out_dir(pcfg, key, *parts):
    path = os.path.join(HERE, pcfg[key], *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ── E3SM records ─────────────────────────────────────────────────────────
def records(case, stream, t0, t1, every_hours=1):
    """[(path, record_index, time)] in [t0, t1], one per timestamp (restart
    duplicates dropped), thinned to every Nth hour from t0."""
    c = dict(case, e3sm_stream=stream)
    out, seen = [], set()
    try:
        index = sio._e3sm_file_index(c, t0, t1)
    except FileNotFoundError as e:
        print(f"  {case['name']}: {e}")
        return out
    for path, times, keep in index:
        for k in keep:
            t = times[k]
            if t in seen:
                continue
            hours = (t - t0).total_seconds() / 3600.0
            if every_hours > 1 and (abs(hours / every_hours - round(hours / every_hours)) > 1e-6):
                continue
            seen.add(t)
            out.append((path, int(k), t))
    out.sort(key=lambda x: x[2])
    return out


def check_coverage(recs, t0, t1, name, slack=timedelta(hours=1)):
    """True if records span [t0, t1] within `slack`; prints why not."""
    if not recs:
        print(f"  SKIP {name}: no records in {t0} to {t1}")
        return False
    first, last = recs[0][2], recs[-1][2]
    if first > t0 + slack or last < t1 - slack:
        print(f"  SKIP {name}: records cover {first} to {last}, window is {t0} to {t1}")
        return False
    return True


class Columns:
    """Columns of one mesh inside a bbox. Reads a contiguous ncol span with
    netCDF4 (fast), then picks the columns in numpy."""

    def __init__(self, path, bbox):
        min_lat, min_lon, max_lat, max_lon = bbox
        with nc.Dataset(path) as ds:
            lat = np.asarray(ds.variables['lat'][:])
            lon = np.asarray(ds.variables['lon'][:])
            lon = np.where(lon > 180, lon - 360, lon)
            idx = np.where((lat >= min_lat) & (lat <= max_lat) &
                           (lon >= min_lon) & (lon <= max_lon))[0]
            if len(idx) == 0:
                raise ValueError(f"No columns inside {bbox} in {path}")
            self.lat, self.lon = lat[idx], lon[idx]
            self.area = np.asarray(ds.variables['area'][:])[idx]
            self.lev = np.asarray(ds.variables['lev'][:])
            self.hyam = np.asarray(ds.variables['hyam'][:])
            self.hybm = np.asarray(ds.variables['hybm'][:])
            self.p0 = float(ds.variables['P0'][:])
        self.span = slice(int(idx[0]), int(idx[-1]) + 1)
        self.local = idx - idx[0]
        self.n = len(idx)

    def level_index(self, levels_hpa):
        """Nearest model level for each requested reference pressure."""
        return [int(np.argmin(np.abs(self.lev - p))) for p in levels_hpa]

    def read(self, path, rec, var, lev_idx=None):
        """var at one record: (ncol,) for 2D, (nlev, ncol) for 3D (all levels,
        or the list lev_idx)."""
        with nc.Dataset(path) as ds:
            v = ds.variables[var]
            if v.ndim == 2:
                return np.asarray(v[rec, self.span])[self.local]
            levs = slice(None) if lev_idx is None else list(lev_idx)
            return np.asarray(v[rec, levs, self.span])[:, self.local]


# ── geometry ─────────────────────────────────────────────────────────────
def sphere_xyz(lat, lon):
    """Cartesian points (km) on a sphere. Chord ~ great-circle distance to
    better than 0.01% below 500 km."""
    la, lo = np.radians(lat), np.radians(lon)
    return R_EARTH_KM * np.column_stack([np.cos(la) * np.cos(lo),
                                         np.cos(la) * np.sin(lo),
                                         np.sin(la)])


def grid_spacing_km(lat, lon):
    """Median nearest-neighbour distance between columns (km)."""
    d, _ = cKDTree(sphere_xyz(lat, lon)).query(sphere_xyz(lat, lon), k=2)
    return float(np.median(d[:, 1]))


def r_edges(sf_cfg):
    return np.geomspace(sf_cfg['r_min_km'], sf_cfg['r_max_km'], sf_cfg['n_bins'] + 1)


def build_pairs(lat, lon, edges, r_exact_km, n_random, seed=0):
    """(i, j, bin) for column pairs with edges[0] <= d < edges[-1]."""
    xyz = sphere_xyz(lat, lon)
    r_exact = edges[np.argmin(np.abs(edges - r_exact_km))]

    ij = cKDTree(xyz).query_pairs(r_exact, output_type='ndarray')
    i_ex, j_ex = ij[:, 0], ij[:, 1]
    d_ex = np.linalg.norm(xyz[i_ex] - xyz[j_ex], axis=1)
    keep = d_ex < r_exact                       # query_pairs is <=; keep the edge for the random set
    i_ex, j_ex, d_ex = i_ex[keep], j_ex[keep], d_ex[keep]

    rng = np.random.default_rng(seed)
    n = len(lat)
    i_r = rng.integers(0, n, int(n_random))
    j_r = rng.integers(0, n, int(n_random))
    d_r = np.linalg.norm(xyz[i_r] - xyz[j_r], axis=1)
    keep = (i_r != j_r) & (d_r >= r_exact)
    i_r, j_r, d_r = i_r[keep], j_r[keep], d_r[keep]

    i = np.concatenate([i_ex, i_r])
    j = np.concatenate([j_ex, j_r])
    d = np.concatenate([d_ex, d_r])
    b = np.digitize(d, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    print(f"    pairs: {ok[:len(i_ex)].sum():,} exact (< {r_exact:.0f} km), "
          f"{ok[len(i_ex):].sum():,} random")
    return i[ok].astype(np.int32), j[ok].astype(np.int32), b[ok].astype(np.int16)


def structure_function(f, i, j, b, n_bins):
    """S2 per bin for one field f(ncol). Returns (S2, pair counts)."""
    dsq = (f[i] - f[j]) ** 2
    good = np.isfinite(dsq)
    counts = np.bincount(b[good], minlength=n_bins).astype(float)
    sums = np.bincount(b[good], weights=dsq[good], minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums / counts, counts


# ── structure-function files ─────────────────────────────────────────────
def sf_path(pcfg, event_label, resolution):
    return os.path.join(out_dir(pcfg, 'data_dir', 'structure_functions'),
                        f"sf_{event_label}_{resolution}.nc")


def write_sf(path, s2, counts, edges, times, fields, levels_hpa, lev_actual,
             attrs):
    """s2: (field, level, time, bin)."""
    with nc.Dataset(path, 'w') as ds:
        ds.createDimension('field', len(fields))
        ds.createDimension('level', len(levels_hpa))
        ds.createDimension('time', len(times))
        ds.createDimension('bin', len(edges) - 1)
        ds.createDimension('edge', len(edges))
        v = ds.createVariable('field_name', str, ('field',))
        for k, name in enumerate(fields):
            v[k] = name
        ds.createVariable('level_hpa', 'f8', ('level',))[:] = levels_hpa
        ds['level_hpa'].long_name = "requested reference pressure"
        ds.createVariable('lev', 'f8', ('level',))[:] = lev_actual
        ds['lev'].long_name = "model lev used (1000*(A+B), hPa)"
        ds.createVariable('r_edges', 'f8', ('edge',))[:] = edges
        ds['r_edges'].units = "km"
        ds.createVariable('r', 'f8', ('bin',))[:] = np.sqrt(edges[:-1] * edges[1:])
        ds['r'].units = "km"
        t = ds.createVariable('time', 'f8', ('time',))
        t.units = f"hours since {times[0]:%Y-%m-%d %H:%M:%S}"
        t[:] = [(x - times[0]).total_seconds() / 3600.0 for x in times]
        ds.createVariable('S2', 'f4', ('field', 'level', 'time', 'bin'))[:] = s2
        ds['S2'].units = "m2 s-2"
        ds['S2'].long_name = "second-order structure function"
        ds.createVariable('npairs', 'f8', ('bin',))[:] = counts
        for k, val in attrs.items():
            ds.setncattr(k, val)


def read_sf(path):
    """Dict with S2 (field, level, time, bin), r, fields, levels, attrs."""
    with nc.Dataset(path) as ds:
        out = {
            'S2': np.ma.filled(ds['S2'][:].astype(float), np.nan),
            'r': ds['r'][:], 'r_edges': ds['r_edges'][:],
            'fields': [str(x) for x in ds['field_name'][:]],
            'level_hpa': ds['level_hpa'][:], 'lev': ds['lev'][:],
            'npairs': ds['npairs'][:],
            'attrs': {k: ds.getncattr(k) for k in ds.ncattrs()},
        }
    return out


def finite_bins(r, s2):
    """Only the bins with a value. Near the grid scale, column separations
    come in discrete values (dx, sqrt(2) dx, 2 dx, ...) and the bins between
    them are empty, so a plain line would leave isolated bins undrawn."""
    ok = np.isfinite(s2)
    return r[ok], s2[ok]


def anchor(r, s2, r_km):
    """(r, S2) at the filled bin nearest r_km, for placing reference slopes."""
    rf, sf_ = finite_bins(r, s2)
    k = int(np.argmin(np.abs(rf - r_km)))
    return rf[k], sf_[k]


def sf_mean(sf, field, level_hpa):
    """Time-mean S2(r) for one field and requested level."""
    k = sf['fields'].index(field)
    m = int(np.argmin(np.abs(np.asarray(sf['level_hpa']) - level_hpa)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # all-NaN bins (too few pairs)
        return np.nanmean(sf['S2'][k, m], axis=0), float(sf['lev'][m])


# ── precipitation (block 4) ──────────────────────────────────────────────
# Periods: k = 1..n covers (t0 + (k-1)A, t0 + kA] with A = accum_hours.
# Values are depths in mm per period. A period missing any record is NaN.
def n_periods(t0, t1, accum_hours):
    return int(round((t1 - t0).total_seconds() / 3600.0 / accum_hours))


def period_ends(t0, t1, accum_hours):
    return [t0 + timedelta(hours=accum_hours * (k + 1))
            for k in range(n_periods(t0, t1, accum_hours))]


def _period_index(times, t0, accum_hours):
    """0-based period for each stamp, from its (seconds since t0)."""
    s = np.array([(t - t0).total_seconds() for t in times])
    return np.ceil(s / (3600.0 * accum_hours) - 1e-9).astype(int) - 1


def read_e3sm_precip_periods(case, stream, var, t0, t1, bbox, accum_hours, chunk=64):
    """(Columns, vals (nper, ncol) mm per period) from an averaged precip
    tape (e.g. h5 PRECT, m/s). Each record is the mean over the interval
    ending at its stamp, so it is assigned by its interval midpoint. None if
    the case has no records."""
    c = dict(case, e3sm_stream=stream)
    try:
        index = sio._e3sm_file_index(c, t0, t1)
    except FileNotFoundError as e:
        print(f"  {case['name']}: {e}")
        return None
    if not index:
        return None
    stamps = sorted({t for _, times, keep in index for t in times[keep]})
    if len(stamps) < 2:
        return None
    step = float(np.median(np.diff([(t - stamps[0]).total_seconds() for t in stamps])))
    expected = accum_hours * 3600.0 / step
    if abs(expected - round(expected)) > 1e-6:
        print(f"  WARNING {case['name']}: {step:.0f} s records don't divide {accum_hours} h")
    expected = int(round(expected))

    cols = Columns(index[0][0], bbox)
    nper = n_periods(t0, t1, accum_hours)
    sums = np.zeros((nper, cols.n))
    counts = np.zeros(nper, dtype=int)
    seen = set()
    for path, times, keep in index:
        ks = np.array([k for k in keep if times[k] not in seen], dtype=int)
        seen.update(times[ks])
        if len(ks) == 0:
            continue
        mids = [times[k] - timedelta(seconds=step / 2) for k in ks]
        p = _period_index(mids, t0, accum_hours)
        with nc.Dataset(path) as ds:
            v = ds.variables[var]
            for s in range(0, len(ks), chunk):
                k, pp = ks[s:s + chunk], p[s:s + chunk]
                block = np.asarray(v[k[0]:k[-1] + 1, cols.span], dtype=np.float64)
                block = block[k - k[0]][:, cols.local]
                ok = (pp >= 0) & (pp < nper)
                np.add.at(sums, pp[ok], block[ok])
                np.add.at(counts, pp[ok], 1)
    bad = counts != expected
    if bad.any():
        print(f"  {case['resolution']}: {bad.sum()} of {nper} periods incomplete "
              f"(expected {expected} records each); set to NaN")
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = sums / counts[:, None] * 3.6e6 * accum_hours     # m/s -> mm per period
    vals[bad] = np.nan
    return cols, vals.astype(np.float32)


def block_mean(a, f):
    """Mean over f x f blocks of the last two axes (edges trimmed). NaN in a
    block gives NaN."""
    if f == 1:
        return a
    ny, nx = a.shape[-2] // f * f, a.shape[-1] // f * f
    a = a[..., :ny, :nx]
    return a.reshape(a.shape[:-2] + (ny // f, f, nx // f, f)).mean(axis=(-3, -1))


def read_obs_precip_periods(cfg, name, t0, t1, bbox, accum_hours, coarsen=1):
    """(lat, lon, vals (nper, npts)) for an hourly obs product: hours summed
    into periods, optionally block-averaged by `coarsen`, points inside bbox.
    A period missing any hour is NaN."""
    it, _, _, _ = sio._obs_iter(cfg, name, t0, t1, bbox)
    nper = n_periods(t0, t1, accum_hours)
    sums = counts = lat = lon = None
    for la, lo, times, block in it:
        if sums is None:
            lat, lon = la, lo
            sums = np.zeros((nper,) + block.shape[1:])
            counts = np.zeros(nper, dtype=int)
        p = _period_index(times, t0, accum_hours)
        ok = (p >= 0) & (p < nper)
        np.add.at(sums, p[ok], block[ok].astype(np.float64))
        np.add.at(counts, p[ok], 1)
    bad = counts != accum_hours
    if bad.any():
        print(f"  {name}: {bad.sum()} of {nper} periods incomplete; set to NaN")
    sums[bad] = np.nan
    lat2d, lon2d = sio._as_2d(lat, lon)
    sums = block_mean(sums, coarsen)
    lat2d, lon2d = block_mean(lat2d, coarsen), block_mean(lon2d, coarsen)
    min_lat, min_lon, max_lat, max_lon = bbox
    inside = ((lat2d >= min_lat) & (lat2d <= max_lat) &
              (lon2d >= min_lon) & (lon2d <= max_lon))
    return lat2d[inside], lon2d[inside], sums[:, inside].astype(np.float32)


def surface_fields(cfg, opt, t0, t1, bbox, pad=0.3):
    """(lat, lon, elevation m, land bool) on the columns of the reference
    mesh (opt['surface']), bbox + pad degrees."""
    s = opt['surface']
    case = next((c for c in cfg['cases'] if c['resolution'] == s['resolution']), None)
    if case is None:
        raise KeyError(f"{cfg['event_label']}: no {s['resolution']} case for the surface fields.")
    index = sio._e3sm_file_index(dict(case, e3sm_stream=s['stream']), t0, t1)
    if not index:
        raise FileNotFoundError(f"No {s['stream']} records for {case['name']} in {t0} to {t1}.")
    path, _, keep = index[0]
    min_lat, min_lon, max_lat, max_lon = bbox
    cols = Columns(path, (min_lat - pad, min_lon - pad, max_lat + pad, max_lon + pad))
    elev = cols.read(path, int(keep[0]), 'PHIS') / 9.80616
    land = cols.read(path, int(keep[0]), 'OCNFRAC') < s['ocnfrac_max']
    return cols.lat, cols.lon, elev, land


def mask_points(lat, lon, surf, masks):
    """{mask name: bool (npts)} from the nearest surface column. Points
    farther than 2 surface spacings from any column are outside every mask."""
    slat, slon, elev, land = surf
    tree = cKDTree(sphere_xyz(slat, slon))
    d, k = tree.query(sphere_xyz(lat, lon))
    near = d < 2 * grid_spacing_km(slat, slon)
    out = {}
    for name, zmax in masks.items():
        m = near & land[k]
        if zmax is not None:
            m &= elev[k] < zmax
        out[name] = m
    return out


def psf_path(pcfg, event_label, product):
    return os.path.join(out_dir(pcfg, 'data_dir', 'precip_structure_functions'),
                        f"psf_{event_label}_{product}.nc")


def write_psf(path, s2, npairs, edges, times, mask_names, kind_names, attrs):
    """s2: (mask, kind, time, bin); npairs: (mask, bin)."""
    with nc.Dataset(path, 'w') as ds:
        ds.createDimension('mask', len(mask_names))
        ds.createDimension('kind', len(kind_names))
        ds.createDimension('time', len(times))
        ds.createDimension('bin', len(edges) - 1)
        ds.createDimension('edge', len(edges))
        for dim, names in (('mask', mask_names), ('kind', kind_names)):
            v = ds.createVariable(f'{dim}_name', str, (dim,))
            for k, n in enumerate(names):
                v[k] = n
        ds.createVariable('r_edges', 'f8', ('edge',))[:] = edges
        ds['r_edges'].units = "km"
        ds.createVariable('r', 'f8', ('bin',))[:] = np.sqrt(edges[:-1] * edges[1:])
        ds['r'].units = "km"
        t = ds.createVariable('time', 'f8', ('time',))
        t.units = f"hours since {times[0]:%Y-%m-%d %H:%M:%S}"
        t.long_name = "period end"
        t[:] = [(x - times[0]).total_seconds() / 3600.0 for x in times]
        ds.createVariable('S2', 'f4', ('mask', 'kind', 'time', 'bin'))[:] = s2
        ds['S2'].units = "mm2 (per accumulation period squared)"
        ds['S2'].long_name = "second-order structure function of precipitation depth"
        ds.createVariable('npairs', 'f8', ('mask', 'bin'))[:] = npairs
        for k, val in attrs.items():
            ds.setncattr(k, val)


def read_psf(path):
    with nc.Dataset(path) as ds:
        return {
            'S2': np.ma.filled(ds['S2'][:].astype(float), np.nan),
            'r': ds['r'][:], 'npairs': ds['npairs'][:],
            'masks': [str(x) for x in ds['mask_name'][:]],
            'kinds': [str(x) for x in ds['kind_name'][:]],
            'attrs': {k: ds.getncattr(k) for k in ds.ncattrs()},
        }


def psf_mean(psf, mask, kind):
    """Time-mean S2(r) for one mask and kind."""
    m, k = psf['masks'].index(mask), psf['kinds'].index(kind)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(psf['S2'][m, k], axis=0)
