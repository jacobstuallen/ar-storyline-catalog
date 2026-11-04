"""Shared helpers for sub_hourly_precip: config, h5 block reader, depth-duration
maths, coarsening, ASOS 1-minute reader, plotting.

Self-contained on purpose: event configs come from storyline_diag (YAML only),
but no storyline_diag Python modules are imported.

Time convention: every series is cut into blocks of `block_minutes` starting at
the analysis-window start t0. Block k covers (t0 + k*B, t0 + (k+1)*B] and is
stamped at its end. E3SM h5 records go to the block holding their averaging
interval's midpoint (from time_bnds). ASOS 1-minute values are taken as the
accumulation over the minute ending at `valid`.
"""
import csv
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
R_EARTH_KM = 6371.22
EPOCH = datetime(1970, 1, 1)
MM_PER_IN = 25.4

# Categorical slots 1-3 keyed by resolution, so a missing case never shifts
# the others' colors. Events use slots 1-5 in event order (validated, light).
RES_COLORS = {"30x8": "#2a78d6", "30x16": "#eb6834", "30x32": "#1baf7a"}
EVENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
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
    return load_yaml(os.path.join(HERE, "sub_hourly.yaml"))


def load_event(opts, config_name):
    sdir = rel(opts["storyline_dir"])
    dpath = os.path.join(sdir, "defaults.yaml")
    base = load_yaml(dpath) if os.path.isfile(dpath) else {}
    cfg = deep_merge(base, load_yaml(os.path.join(sdir, config_name)))
    if "cases" not in cfg and "case" in cfg:
        cfg["cases"] = [cfg["case"]]
    return cfg


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def analysis_window(opts, cfg):
    """(t0, t1) UTC. `window: storm` uses observed_comparison.storm_window,
    falling back to the event's time: block if it is missing."""
    sw = (cfg.get("observed_comparison") or {}).get("storm_window")
    if opts.get("window", "storm") == "storm" and sw:
        return parse_utc(sw["start"]), parse_utc(sw["end"])
    t = cfg["time"]
    t0 = datetime(t["start_year"], t["start_month"], t["start_day"])
    return t0, t0 + timedelta(days=t["num_days"])


def case_label(cfg, case):
    res = case.get("resolution", "")
    return case.get("label") or cfg.get("resolution_labels", {}).get(res, res)


def case_color(case):
    return RES_COLORS.get(case.get("resolution"), INK)


def event_short(label):
    return label.split("_")[0]


def blocks_path(opts, event_label, res):
    return os.path.join(rel(opts["data_dir"]), event_label, f"blocks_{res}.nc")


def plot_path(opts, event_label, name):
    d = os.path.join(rel(opts["plot_dir"]), event_label)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def check_durations(opts):
    B = int(opts["block_minutes"])
    durs = [int(d) for d in opts["durations_min"]]
    bad = [d for d in durs if d % B]
    if bad:
        raise ValueError(f"durations {bad} are not multiples of block_minutes={B}")
    return B, durs


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


def _to_dt(x):
    t = datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    if getattr(x, "microsecond", 0) >= 500000:
        t += timedelta(seconds=1)
    return t


def decode(values, tvar):
    d = cftime.num2date(np.atleast_1d(values), tvar.units,
                        calendar=getattr(tvar, "calendar", "standard"))
    return np.array([_to_dt(x) for x in np.atleast_1d(d)])


def _filled(a):
    return np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)


def grid_cols(case, box, files):
    """Columns inside box [min_lat, min_lon, max_lat, max_lon] of the first file."""
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


def read_blocks(case, stream, t0, t1, box, block_min, var="PRECT"):
    """Block depths (mm) of a time-mean rate variable (m/s).

    Returns dict: block_end (nb,), depth (nb, ncell) float32 (NaN where a
    block is not fully covered), cols, lat, lon, area, record_s (h5 interval).
    Restart duplicates are dropped (first occurrence kept)."""
    files = stream_files(case, stream, t0, t1)
    if not files:
        raise FileNotFoundError(f"no eam.{stream} files for {case['name']}")
    cols, lat, lon, area = grid_cols(case, box, files)
    c0, c1 = cols.min(), cols.max() + 1
    B = block_min * 60.0
    nb = int(round((t1 - t0).total_seconds() / B))
    depth = np.zeros((nb, cols.size))
    cover = np.zeros(nb)
    seen = set()
    rec_s = []
    for f in files:
        with nc.Dataset(f) as ds:
            tv = ds.variables["time"]
            if "time_bnds" not in ds.variables:
                raise KeyError(f"{f} has no time_bnds; cannot place time-mean records")
            tb = np.asarray(ds.variables["time_bnds"][:])
            b0, b1 = decode(tb[:, 0], tv), decode(tb[:, 1], tv)
            mid = np.array([a + (b - a) / 2 for a, b in zip(b0, b1)])
            dt = np.array([(b - a).total_seconds() for a, b in zip(b0, b1)])
            k = np.where((mid > t0) & (mid < t1) & (dt > 0))[0]
            k = np.array([i for i in k if mid[i] not in seen], dtype=int)
            if k.size == 0:
                continue
            sl = slice(k.min(), k.max() + 1)
            blk = _filled(ds.variables[var][sl, c0:c1])[k - k.min()][:, cols - c0]
            for j, i in enumerate(k):
                seen.add(mid[i])
                ib = int((mid[i] - t0).total_seconds() // B)
                depth[ib] += blk[j] * dt[i] * 1000.0
                cover[ib] += dt[i]
                rec_s.append(dt[i])
    full = np.abs(cover - B) < 1.0
    depth[~full] = np.nan
    if not full.any():
        raise ValueError(f"no complete {block_min}-min blocks for {case['name']} in {t0}..{t1}")
    rs = float(np.median(rec_s))
    if abs(B / rs - round(B / rs)) > 1e-6:
        raise ValueError(f"{case['name']}: h5 interval {rs:.0f} s does not divide {B:.0f} s")
    n_bad = int((~full).sum())
    if n_bad:
        print(f"    {n_bad} of {nb} blocks incomplete (NaN)")
    ends = np.array([t0 + timedelta(seconds=B * (i + 1)) for i in range(nb)])
    return dict(block_end=ends, depth=depth.astype(np.float32), cols=cols, lat=lat,
                lon=lon, area=area, record_s=rs)


def read_static(case, stream, cols, names, t0, t1):
    files = stream_files(case, stream, t0, t1) or sorted(
        glob.glob(os.path.join(case["e3sm_dir"], f"*.eam.{stream}.*.nc")))
    if not files:
        print(f"    no eam.{stream} files for {case['name']}; static fields missing")
        return {n: None for n in names}
    with nc.Dataset(files[0]) as ds:
        return {n: (_filled(ds.variables[n][0, :])[cols] if n in ds.variables else None)
                for n in names}


# ── depth-duration maths ────────────────────────────────────────────────
def rolling_max(depth, steps, max_missing_frac=0.0):
    """Max running sum over `steps` consecutive blocks, per column.
    depth (nb, n). A window may contain up to floor(max_missing_frac * steps)
    NaN blocks, counted as zero (so the result is a lower bound); windows with
    more are skipped. A column with no valid window is NaN. Returns (n,)."""
    depth = np.atleast_2d(np.asarray(depth, float))
    if depth.shape[0] < steps:
        return np.full(depth.shape[1], np.nan)
    bad = ~np.isfinite(depth)
    cs = np.vstack([np.zeros(depth.shape[1]), np.cumsum(np.where(bad, 0.0, depth), axis=0)])
    cb = np.vstack([np.zeros(depth.shape[1]), np.cumsum(bad, axis=0)])
    s = cs[steps:] - cs[:-steps]
    nbad = cb[steps:] - cb[:-steps]
    s[nbad > np.floor(max_missing_frac * steps)] = -np.inf
    m = s.max(axis=0)
    return np.where(np.isfinite(m), m, np.nan)


def max_depths(depth, durations_min, block_min, max_missing_frac=0.0):
    """(ndur, n) max depth for each duration."""
    return np.vstack([rolling_max(depth, d // block_min, max_missing_frac) for d in durations_min])


def weighted_quantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    o = np.argsort(x[ok])
    xs, ws = x[ok][o], w[ok][o]
    cw = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return float(np.interp(q, cw, xs))


def wet_fraction_top(depth, hours, block_min):
    """Share of the window total falling in the wettest `hours` (running
    window), per column."""
    steps = int(round(hours * 60 / block_min))
    tot = np.nansum(depth, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, rolling_max(depth, steps) / tot, np.nan)


# ── high-frequency characteristics (location- and timing-free) ──────────
WET_MM = 0.254   # one 0.01-in tip; a block at or above this counts as wet


def _col(depth):
    """(nb,) -> (nb, 1); (nb, n) unchanged. A 1-D series is one site."""
    d = np.asarray(depth, float)
    return d[:, None] if d.ndim == 1 else d


def aggregate(depth, k):
    """Sum k consecutive blocks (from the first block; a partial tail is
    dropped). NaN if any block in the group is NaN. depth (nb, n)."""
    d = _col(depth)
    if k == 1:
        return d
    m = (d.shape[0] // k) * k
    return d[:m].reshape(m // k, k, d.shape[1]).sum(axis=1)


def contribution_curve(depth, block_min, rates_mmhr):
    """Share of the total falling in blocks with rate >= r, for each r.
    depth (nb, n) -> (nr, n). NaN blocks are left out of both sums."""
    d = _col(depth)
    rate = d * 60.0 / block_min
    tot = np.nansum(d, axis=0)
    out = np.vstack([np.nansum(np.where(rate >= r, d, 0.0), axis=0) for r in rates_mmhr])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, out / tot, np.nan)


def hf_metrics(depth, block_min, rates_mmhr, top_hours, max_missing_frac=0.05):
    """Per column: total, missing fraction, share of total at rate >= each
    threshold, share in the wettest `top_hours` running windows, wet-block
    fraction, and peak block / mean wet block. depth (nb, n); returns
    {name: (n,)}. Every metric except total is independent of the storm total."""
    d = _col(depth)
    ok = np.isfinite(d)
    tot = np.nansum(d, axis=0)
    out = {"total": tot, "missing_frac": 1.0 - ok.mean(axis=0)}
    cc = contribution_curve(d, block_min, rates_mmhr)
    for r, row in zip(rates_mmhr, cc):
        out[f"share_ge_{r:g}"] = row
    with np.errstate(invalid="ignore", divide="ignore"):
        for h in top_hours:
            steps = int(round(h * 60 / block_min))
            out[f"top_{h:g}h"] = np.where(tot > 0, rolling_max(d, steps, max_missing_frac) / tot, np.nan)
        wet = ok & (np.where(ok, d, 0.0) >= WET_MM)
        out["wet_frac"] = wet.sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
        mean_wet = np.where(wet.sum(axis=0) > 0, np.where(wet, d, 0.0).sum(axis=0) / np.maximum(wet.sum(axis=0), 1), np.nan)
        peak = np.where(ok.any(axis=0), np.max(np.where(ok, d, -np.inf), axis=0), np.nan)
        out["peak_over_wetmean"] = peak / mean_wet
    return out


# ── CW3E landfalling AR catalog ─────────────────────────────────────────
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y%m%d%H", "%Y%m%d%H%M", "%Y-%m-%d_%H", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M")


def parse_time(s):
    s = s.strip().strip('"')
    for f in TIME_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    raise ValueError(f"unrecognised time {s!r}; add its format to TIME_FORMATS")


def read_ar_catalog(path, columns, header_rows=1, verbose=True):
    """CW3E per-point event CSV -> list of dict(start, end, scale). `columns`
    gives 1-based column numbers {start, end, scale} (CW3E documents column 6
    as the AR scale). Prints the header and first rows so the mapping can be
    checked."""
    rows = []
    with open(path) as f:
        rd = list(csv.reader(f))
    if verbose:
        print(f"  {os.path.basename(path)}: {len(rd)} lines; first lines:")
        for r in rd[:header_rows + 2]:
            print(f"    | {r}")
    for r in rd[header_rows:]:
        if not r or not "".join(r).strip():
            continue
        try:
            rows.append(dict(start=parse_time(r[columns["start"] - 1]),
                             end=parse_time(r[columns["end"] - 1]),
                             scale=int(float(r[columns["scale"] - 1]))))
        except (ValueError, IndexError) as e:
            raise ValueError(f"{path}: cannot read row {r}: {e}") from None
    return rows


# ── geometry ────────────────────────────────────────────────────────────
def xyz(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)]


def dist_km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat)
    dp, dl = p - p0, np.radians(lon - lon0)
    a = np.sin(dp / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def median_spacing_km(lat, lon):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz(lat, lon)).query(xyz(lat, lon), k=2)
    return float(np.median(d[:, 1]) * R_EARTH_KM)


def coarsen_map(lat_f, lon_f, lat_c, lon_c):
    """Index of the nearest coarse centre for each fine cell (Voronoi
    assignment, an approximation to conservative remapping)."""
    from scipy.spatial import cKDTree
    _, j = cKDTree(xyz(lat_c, lon_c)).query(xyz(lat_f, lon_f))
    return j


def coarsen(depth, area_f, j, ncoarse):
    """Area-weighted mean of fine columns per coarse cell. depth (nb, nf).
    Returns (nb, ncoarse) and the assigned fine area per coarse cell."""
    W = np.bincount(j, weights=area_f, minlength=ncoarse)
    out = np.full((depth.shape[0], ncoarse), np.nan)
    for t in range(depth.shape[0]):
        if np.isfinite(depth[t]).all():
            out[t] = np.bincount(j, weights=depth[t] * area_f, minlength=ncoarse)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / np.where(W > 0, W, np.nan)
    return out, W


# ── blocks file ─────────────────────────────────────────────────────────
def to_seconds(times):
    return np.array([(t - EPOCH).total_seconds() for t in times])


def from_seconds(s):
    return np.array([EPOCH + timedelta(seconds=float(x)) for x in s])


def read_blocks_nc(path):
    d = {}
    with nc.Dataset(path) as ds:
        for n, v in ds.variables.items():
            a = _filled(v[:])
            d[n] = from_seconds(a) if n == "block_end" else a
        for k in ds.ncattrs():
            d[k] = ds.getncattr(k)
    return d


def built_cases(opts, cfg):
    """[(case, blocks dict)] for cases whose blocks file exists."""
    out = []
    for case in cfg["cases"]:
        p = blocks_path(opts, cfg["event_label"], case["resolution"])
        if os.path.isfile(p):
            out.append((case, read_blocks_nc(p)))
        else:
            print(f"  skip {case['resolution']}: no {p} (run build_blocks_nc.py)")
    return out


def land(d, opts):
    m = np.ones(d["lat"].shape, bool)
    if "landfrac" in d:
        m &= np.isfinite(d["landfrac"]) & (d["landfrac"] >= opts.get("min_landfrac", 0.5))
    return m


# ── ASOS ────────────────────────────────────────────────────────────────
def asos_dir(opts):
    return rel(opts["asos"]["dir"])


def asos_stations(opts):
    """{id: dict(name, lat, lon, elev_m)} from data/asos/stations.csv."""
    p = os.path.join(asos_dir(opts), "stations.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} missing; run fetch_asos.py on a login node")
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[r["id"]] = dict(name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]),
                                elev_m=float(r["elev_m"]) if r["elev_m"] else np.nan)
    return out


def asos_csv_path(opts, event_label, sid):
    return os.path.join(asos_dir(opts), event_label, f"{sid}.csv")


def read_asos_csv(path):
    """IEM asos1min.py download (comma, gis=1). Returns times (minute stamps,
    UTC) and {var: float array} with NaN for missing; ptype kept as strings."""
    times, cols = [], {}
    with open(path) as f:
        rd = csv.reader(f)
        head = next(rd)
        vix = [i for i, h in enumerate(head) if h.startswith("valid")][0]
        names = head[vix + 1:]
        for n in names:
            cols[n] = []
        for row in rd:
            if len(row) < vix + 1 + len(names):
                continue
            times.append(datetime.strptime(row[vix], "%Y-%m-%d %H:%M"))
            for n, v in zip(names, row[vix + 1:]):
                cols[n].append(v)
    out = {}
    for n, v in cols.items():
        if n == "ptype":
            out[n] = np.array(v, dtype=object)
        else:
            out[n] = np.array([np.nan if x in ("M", "", "None") else float(x) for x in v])
    return np.array(times), out


NCEI_PG2_COLS = ["precip", "ptype", "pres1", "tmpf"]


def _real(s, lo, hi):
    try:
        v = float(s)
    except ValueError:
        return None
    return v if lo <= v <= hi else None


def parse_ncei_pg2(lines, t0=None, t1=None):
    """NCEI DSI-6406 (ASOS 1-minute page 2) fixed-width lines -> list of
    (valid_utc, {precip, ptype, pres1, tmpf}); None for missing. Column
    positions and the local/UTC date rule follow IEM's
    parse_ncei_asos1minute.py (brackets blanked before slicing). US stations
    only: UTC is ahead of local time, so a UTC hour below the local hour means
    the next UTC day. Keeps t0 < valid <= t1 when given."""
    out = []
    for ln in lines:
        ln = ln.rstrip("\n")
        if len(ln) < 30 or ln[13] not in "12":
            continue
        s = ln[13:29]
        try:
            day = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
            lhr, uhr, umin = int(s[8:10]), int(s[12:14]), int(s[14:16])
        except ValueError:
            continue
        if uhr < lhr:
            day += timedelta(days=1)
        valid = day + timedelta(hours=uhr, minutes=umin)
        if (t0 is not None and valid <= t0) or (t1 is not None and valid > t1):
            continue
        b = ln.replace("[", " ").replace("]", " ").replace("\\", " ")
        pt = b[31:34].strip()
        tf = b[94:97].strip()
        out.append((valid, {
            "precip": _real(b[43:48].strip(), 0.0, 0.5),
            "ptype": pt[:2] if pt else None,
            "pres1": _real(b[69:77].strip(), 10.0, 40.0),
            "tmpf": int(tf) if tf.lstrip("-").isdigit() and -90 <= int(tf) <= 150 else None,
        }))
    return out


def write_asos_csv(path, sid, name, lat, lon, records, cols=NCEI_PG2_COLS):
    """Write records [(valid, {col: value})] in the IEM asos1min.py CSV layout
    (gis=1), M for missing, so read_asos_csv reads both sources alike."""
    seen = set()
    with open(path, "w") as f:
        f.write(",".join(["station", "station_name", "lat", "lon", "valid(UTC)"] + cols) + "\n")
        for valid, r in sorted(records, key=lambda x: x[0]):
            if valid in seen:
                continue
            seen.add(valid)
            vals = ["M" if r.get(c) is None else str(r[c]) for c in cols]
            f.write(f"{sid},{name.replace(',', '_')},{lat},{lon},{valid:%Y-%m-%d %H:%M}," + ",".join(vals) + "\n")


UA = {"User-Agent": "ar-storyline-catalog sub_hourly_precip (python urllib)"}


def http_get(url, timeout=300):
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _ncei_window(A, st, s0, s1, path, raw_dir):
    """NCEI monthly page-2 files for the months spanning (s0, s1]. Returns
    True if any minute in the window was written to path."""
    import urllib.error
    recs = []
    m = s0.replace(day=1, hour=0, minute=0)
    sid4 = st["id"] if len(st["id"]) == 4 else f"K{st['id']}"
    while m <= s1:
        fn = f"asos-1min-pg2-{sid4}-{m:%Y%m}.dat"
        cached = os.path.join(raw_dir, fn)
        try:
            if os.path.isfile(cached):
                txt = open(cached).read()
            else:
                txt = http_get(f"{A['ncei_url']}/{m:%Y/%m}/{fn}")
                os.makedirs(raw_dir, exist_ok=True)
                with open(cached, "w") as fh:
                    fh.write(txt)
            lines = txt.splitlines()
            got = parse_ncei_pg2(lines, s0, s1)
            print(f"    {fn}: {len(lines)} lines, {len(got)} minutes in window")
            if not got and lines:
                print(f"      | {lines[0][:110]!r}")
            recs += got
        except urllib.error.HTTPError as e:
            print(f"    {fn}: HTTP {e.code}")
        except Exception as e:
            print(f"    {fn}: FAILED {type(e).__name__}: {e}")
        m = (m + timedelta(days=32)).replace(day=1)
    if recs:
        write_asos_csv(path, st["id"], st["name"], st["lat"], st["lon"], recs)
    return bool(recs)


def fetch_station_window(opts, st, s0, s1, path):
    """1-minute ASOS for one station over (s0, s1] into path (IEM CSV layout):
    IEM first, then NCEI monthly files if IEM has nothing and
    asos.ncei_fallback is true. Prints a one-line summary. Returns True if a
    file was written."""
    import time
    import urllib.parse
    A = opts["asos"]
    q = urllib.parse.urlencode({
        "station": st["id"], "vars": ",".join(A["vars"]),
        "sts": s0.strftime("%Y-%m-%dT%H:%MZ"), "ets": s1.strftime("%Y-%m-%dT%H:%MZ"),
        "sample": "1min", "what": "download", "tz": "UTC", "gis": "1", "delim": "comma"})
    try:
        txt = http_get(f"{A['request_url']}?{q}")
    except Exception as e:
        print(f"  {st['id']}: IEM request FAILED {type(e).__name__}: {e}")
        txt = ""
    time.sleep(float(A.get("sleep_s", 2)))
    src = "IEM"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(txt.strip().splitlines()) > 1:
        with open(path, "w") as fh:
            fh.write(txt)
    elif A.get("ncei_fallback", True):
        print(f"  {st['id']}: no records at IEM; trying NCEI monthly files")
        src = "NCEI"
        if not _ncei_window(A, st, s0, s1, path, os.path.join(asos_dir(opts), "ncei_raw")):
            return False
    else:
        print(f"  {st['id']}: no records")
        return False
    times, v = read_asos_csv(path)
    p = v.get("precip", np.array([]))
    nmin = int((s1 - s0).total_seconds() // 60)
    print(f"  {st['id']} ({src}): {len(times)} of {nmin} minutes, precip valid "
          f"{np.isfinite(p).sum()} min, total {np.nansum(p) * MM_PER_IN:.1f} mm")
    return True


def asos_blocks(times, precip_in, t0, nb, block_min, min_valid_frac):
    """Block depths (mm) from 1-minute precip (inches). A block is NaN unless
    at least min_valid_frac of its minutes have a value. Returns depth (nb,),
    valid minutes per block (nb,)."""
    depth = np.zeros(nb)
    nval = np.zeros(nb, int)
    seen = set()
    for t, p in zip(times, precip_in):
        if t in seen:
            continue
        seen.add(t)
        dm = int(round((t - t0).total_seconds() / 60.0))
        k = (dm - 1) // block_min
        if 0 <= k < nb and np.isfinite(p):
            depth[k] += p * MM_PER_IN
            nval[k] += 1
    depth[nval < min_valid_frac * block_min] = np.nan
    return depth, nval


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
        "axes.spines.right": False, "font.size": 8, "axes.titlesize": 10,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "legend.title_fontsize": 7,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })
    return plt


MAX_IN = 11.9  # figures stay under 12 x 12 inches


def fig_size(w, h):
    k = min(1.0, MAX_IN / w, MAX_IN / h)
    return (w * k, h * k)


def letter(i):
    return f"{chr(ord('a') + i)})"


def panel_title(ax, i, text=""):
    ax.set_title(f"{letter(i)} {text}".rstrip(), loc="left")


def save_png(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  wrote {path}")


LABELLED_MIN = (20, 60, 180, 360, 720, 1440, 4320)


def duration_label(d):
    return f"{d}m" if d < 60 else f"{d // 60}h"


def duration_axis(ax, durations_min):
    """Log x axis in minutes; ticks at every duration, labels on a subset
    (LABELLED_MIN) so neighbours on the log axis don't collide."""
    ax.set_xscale("log")
    ax.set_xticks(durations_min)
    ax.set_xticklabels([duration_label(d) if d in LABELLED_MIN else "" for d in durations_min])
    ax.minorticks_off()
    ax.set_xlabel("Duration")
