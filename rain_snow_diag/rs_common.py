"""Shared helpers for rain_snow_diag: config, EAM readers, phase physics, plotting.

Self-contained on purpose: event configs come from storyline_diag (YAML only),
but no storyline_diag Python modules are imported.
"""
import glob
import os
import re
from datetime import datetime, timedelta

import cftime
import netCDF4 as nc
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.80616
T0 = 273.15
R_EARTH_KM = 6371.22
EPOCH = datetime(1970, 1, 1)

# Categorical slots 1-3 (validated all-pairs) keyed by resolution, so a
# missing case never shifts the others' colors.
RES_COLORS = {"30x8": "#2a78d6", "30x16": "#eb6834", "30x32": "#1baf7a"}
# Ordinal blue ramp for elevation bands, low -> high.
BAND_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b",
               "#0a2a52"]
INK = "#222222"
MUTED = "#6b6b66"
GRID = "#e1e0d9"


# ── config ──────────────────────────────────────────────────────────────
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def rel(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def load_options():
    return load_yaml(os.path.join(HERE, "rain_snow.yaml"))


def load_event(opts, config_name):
    sdir = rel(opts["storyline_dir"])
    dpath = os.path.join(sdir, "defaults.yaml")
    base = load_yaml(dpath) if os.path.isfile(dpath) else {}
    cfg = deep_merge(base, load_yaml(os.path.join(sdir, config_name)))
    if "cases" not in cfg and "case" in cfg:
        cfg["cases"] = [cfg["case"]]
    return cfg


def event_window(cfg):
    t = cfg["time"]
    t0 = datetime(t["start_year"], t["start_month"], t["start_day"])
    return t0, t0 + timedelta(days=t["num_days"])


def case_label(cfg, case):
    res = case.get("resolution", "")
    return case.get("label") or cfg.get("resolution_labels", {}).get(res, res)


def case_color(case):
    return RES_COLORS.get(case.get("resolution"), INK)


def phase_path(opts, event_label, res):
    return os.path.join(rel(opts["data_dir"]), event_label, f"phase_{res}.nc")


def plot_path(opts, event_label, name):
    d = os.path.join(rel(opts["plot_dir"]), event_label)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def built_cases(opts, cfg):
    """[(case, phase dict)] for cases whose phase file exists."""
    out = []
    for case in cfg["cases"]:
        p = phase_path(opts, cfg["event_label"], case["resolution"])
        if os.path.isfile(p):
            out.append((case, read_phase(p)))
        else:
            print(f"  skip {case['resolution']}: no {p} (run build_phase_nc.py)")
    return out


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


# ── EAM files and times ─────────────────────────────────────────────────
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")


def stream_files(case, stream, t0, t1, margin_days=35):
    pat = os.path.join(case["e3sm_dir"], f"*.eam.{stream}.*.nc")
    out = []
    for f in sorted(glob.glob(pat)):
        m = DATE_RE.search(f)
        if m:
            fd = datetime(int(m[1]), int(m[2]), int(m[3])) + timedelta(seconds=int(m[4]))
            if fd > t1 or fd < t0 - timedelta(days=margin_days):
                continue
        out.append(f)
    return out


def file_times(ds):
    tv = ds.variables["time"]
    d = cftime.num2date(np.atleast_1d(tv[:]), tv.units,
                        calendar=getattr(tv, "calendar", "standard"))
    out = []
    for x in d:
        t = datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
        if getattr(x, "microsecond", 0) >= 500000:
            t += timedelta(seconds=1)
        out.append(t.replace(second=0) + timedelta(minutes=round(t.second / 60)))
    return np.array(out)


def grid_cols(case, box, t0, t1, stream="h0"):
    """Columns inside box [min_lat, min_lon, max_lat, max_lon]."""
    files = stream_files(case, stream, t0, t1) or sorted(
        glob.glob(os.path.join(case["e3sm_dir"], "*.eam.h*.nc")))
    if not files:
        raise FileNotFoundError(f"no EAM files in {case['e3sm_dir']}")
    with nc.Dataset(files[0]) as ds:
        lat = np.asarray(ds.variables["lat"][:], float)
        lon = np.asarray(ds.variables["lon"][:], float)
        lon = np.where(lon > 180, lon - 360, lon)
        area = np.asarray(ds.variables["area"][:], float)
    a, b, c, d = box
    cols = np.where((lat >= a) & (lat <= c) & (lon >= b) & (lon <= d))[0]
    if cols.size == 0:
        raise ValueError(f"no columns of {case['name']} inside {box}")
    return cols, lat[cols], lon[cols], area[cols]


def _filled(a):
    return np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)


def read_2d(case, stream, names, t0, t1, cols, optional=()):
    """Times in [t0, t1] and {name: (time, ncell)} on cols. Duplicate times
    (restarts) keep the first occurrence. Optional names missing from the
    files come back as None."""
    times, vals = [], {n: [] for n in names}
    missing = set()
    for f in stream_files(case, stream, t0, t1):
        with nc.Dataset(f) as ds:
            tt = file_times(ds)
            k = np.where((tt >= t0) & (tt <= t1))[0]
            if k.size == 0:
                continue
            sl = slice(k.min(), k.max() + 1)
            for n in names:
                if n not in ds.variables:
                    if n in optional:
                        missing.add(n)
                        continue
                    raise KeyError(f"{n} not in {f}")
                vals[n].append(_filled(ds.variables[n][sl, :])[k - k.min()][:, cols])
            times.append(tt[k])
    if not times:
        raise FileNotFoundError(f"no eam.{stream} records in {t0}..{t1} for {case['name']}")
    t = np.concatenate(times)
    _, first = np.unique(t, return_index=True)
    out = {}
    for n in names:
        if n in missing or not vals[n]:
            out[n] = None
        else:
            out[n] = np.concatenate(vals[n])[first]
    return t[first], out


def read_static(case, stream, names, t0, t1, cols):
    files = stream_files(case, stream, t0, t1) or sorted(
        glob.glob(os.path.join(case["e3sm_dir"], f"*.eam.{stream}.*.nc")))
    if not files:
        raise FileNotFoundError(f"no eam.{stream} files for {case['name']}")
    with nc.Dataset(files[0]) as ds:
        return {n: (_filled(ds.variables[n][0, :])[cols] if n in ds.variables else None)
                for n in names}


def iter_3d(case, stream, names, t0, t1, cols, want=None):
    """Yield (time, {name: array}) one record at a time. 3D fields come back
    as (lev, ncell), 2D as (ncell,). `want` = set of times to keep (or None)."""
    seen = set()
    for f in stream_files(case, stream, t0, t1):
        with nc.Dataset(f) as ds:
            tt = file_times(ds)
            hy = {k: np.asarray(ds.variables[k][:], float) for k in ("hyam", "hybm")
                  if k in ds.variables}
            p0 = float(ds.variables["P0"][:]) if "P0" in ds.variables else 1.0e5
            for i, t in enumerate(tt):
                if t < t0 or t > t1 or t in seen or (want is not None and t not in want):
                    continue
                seen.add(t)
                rec = {}
                for n in names:
                    v = ds.variables[n]
                    a = _filled(v[i])
                    rec[n] = a[..., cols]
                rec.update(hy)
                rec["P0"] = p0
                yield t, rec


# ── physics ─────────────────────────────────────────────────────────────
def std_atm_ps(z):
    return 101325.0 * (1.0 - 2.25577e-5 * np.asarray(z)) ** 5.25588


def rh_from_q(q, p, t):
    """RH (%) over water from specific humidity (kg/kg), p (Pa), T (K)."""
    e = q * p / (0.622 + 0.378 * q)
    tc = t - T0
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def wetbulb_stull(t, rh):
    """Wet-bulb temperature (C), Stull (2011). Valid ~5-99% RH, -20 to 50 C."""
    tc = t - T0
    rh = np.clip(rh, 1.0, 100.0)
    return (tc * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(tc + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def freezing_level(T, Z, zsfc, method="highest", zmax=10000.0):
    """Height (m MSL) of the 0 C crossing. T, Z: (lev, ncol), index 0 = model
    top. A crossing is cold above a warm level below, searched only below
    zmax so the warm stratopause (model top ~60 km) is never picked. Columns
    with no crossing and a sub-freezing bottom level get the surface elevation."""
    tc = T - T0
    cross = (tc[:-1] <= 0) & (tc[1:] > 0) & (Z[:-1] <= zmax)
    has = cross.any(axis=0)
    if method == "highest":
        k = np.argmax(cross, axis=0)
    else:
        k = cross.shape[0] - 1 - np.argmax(cross[::-1], axis=0)
    j = np.arange(T.shape[1])
    ta, tb = tc[k, j], tc[k + 1, j]
    za, zb = Z[k, j], Z[k + 1, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        fz = zb + tb / (tb - ta) * (za - zb)
    return np.where(has, fz, np.where(tc[-1] <= 0, zsfc, np.nan))


def snow_line(z, rain, snow, w, bin_m=100, min_rate=0.1, min_cells=3, frac=0.5):
    """Elevation (m) where the binned, area-weighted snow fraction first
    reaches `frac` going uphill. rain, snow: (ncell,) or (time, ncell).
    Returns NaN if no bin reaches frac; the lowest valid bin centre if it is
    already snow there."""
    rain = np.atleast_2d(rain)
    snow = np.atleast_2d(snow)
    edges = np.arange(np.floor(np.nanmin(z) / bin_m) * bin_m, np.nanmax(z) + bin_m, bin_m)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(z, edges) - 1, 0, len(centers) - 1)
    out = np.full(rain.shape[0], np.nan)
    for i in range(rain.shape[0]):
        p = rain[i] + snow[i]
        ok = np.isfinite(p) & (p >= min_rate)
        if ok.sum() < min_cells:
            continue
        P = np.bincount(ib[ok], weights=(p * w)[ok], minlength=len(centers))
        S = np.bincount(ib[ok], weights=(snow[i] * w)[ok], minlength=len(centers))
        N = np.bincount(ib[ok], minlength=len(centers))
        good = N >= min_cells
        if not good.any():
            continue
        f = np.where(good, S / np.where(P > 0, P, 1), np.nan)
        zc, fc = centers[good], f[good]
        hit = np.where(fc >= frac)[0]
        if hit.size == 0:
            continue
        h = hit[0]
        if h == 0:
            out[i] = zc[0]
        else:
            f0, f1 = fc[h - 1], fc[h]
            out[i] = zc[h - 1] + (frac - f0) / (f1 - f0) * (zc[h] - zc[h - 1])
    return out


# ── phase file ──────────────────────────────────────────────────────────
def to_seconds(times):
    return np.array([(t - EPOCH).total_seconds() for t in times])


def from_seconds(s):
    return np.array([EPOCH + timedelta(seconds=float(x)) for x in s])


def read_phase(path):
    d = {}
    with nc.Dataset(path) as ds:
        for n, v in ds.variables.items():
            a = _filled(v[:])
            d[n] = from_seconds(a) if n in ("time", "time6") else a
        for k in ds.ncattrs():
            d[k] = ds.getncattr(k)
    return d


def land_mask(d, opts):
    return np.isfinite(d["zsfc"]) & (d["landfrac"] >= opts.get("min_landfrac", 0.5))


def km2(area_sr):
    return area_sr * R_EARTH_KM ** 2


# ── plotting ────────────────────────────────────────────────────────────
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })
    return plt


def save_png(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  wrote {path}")


def date_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def triangulation(lon, lat, edge_factor=3.0):
    """Delaunay triangulation of the columns with long (hull) triangles masked."""
    from matplotlib.tri import Triangulation
    tri = Triangulation(lon, lat)
    x, y = lon[tri.triangles], lat[tri.triangles]
    edge = np.max(np.hypot(x - np.roll(x, 1, axis=1), y - np.roll(y, 1, axis=1)), axis=1)
    tri.set_mask(edge > edge_factor * np.median(edge))
    return tri


def map_axes(fig, nrows, ncols, extent):
    """Grid of map axes; cartopy if available, plain lon/lat axes otherwise."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        # Fails here (not at savefig) if Natural Earth is neither cached nor downloadable.
        shpreader.natural_earth("50m", "physical", "coastline")
        shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes")
        axes = fig.subplots(nrows, ncols, squeeze=False,
                            subplot_kw={"projection": ccrs.PlateCarree()})
        for ax in axes.flat:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.4, edgecolor=MUTED)
            ax.coastlines("50m", lw=0.5, color=MUTED)
        return axes
    except Exception as e:  # cartopy missing or no Natural Earth cache
        print(f"  cartopy unavailable ({e}); plain lon/lat axes")
        fig.clf()
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax in axes.flat:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(extent[2:]))))
        return axes
