"""
budget_io.py

Helpers for the regional water budget: settings and event configs, E3SM file
index, box geometry, boundary interpolation weights, window integrals, and
the per-case netCDF that compute_budget.py writes.

Budget (area means over the box, kg m-2 = mm):
    P = F_in - F_out + E - dW + R
    W = TMQ + TMCLDLIQ + TMCLDICE + TMRAINQM
    F_in, F_out: line integral of max(-Q.n, 0), max(Q.n, 0), Q = (TUQ, TVQ),
                 n = outward normal, divided by box area
"""

import glob
import os
import re
from datetime import datetime, timedelta

import cftime
import netCDF4
import numpy as np
import yaml
from matplotlib.path import Path
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
R_EARTH = 6.371e6  # m, EAM value

H0_REQUIRED = ["TUQ", "TVQ", "QFLX", "TMQ"]
CONDENSATE = ["TMCLDLIQ", "TMCLDICE", "TMRAINQM"]


# ---------------------------------------------------------------------------
# Settings and event configs
# ---------------------------------------------------------------------------
def load_settings():
    with open(os.path.join(HERE, "budget.yaml")) as f:
        s = yaml.safe_load(f)
    s["event_config_dir"] = os.path.normpath(os.path.join(HERE, s["event_config_dir"]))
    s["output_dir"] = os.path.join(HERE, s["output_dir"])
    s["plot_dir"] = os.path.join(HERE, s["plot_dir"])
    s["boxes"] = [{**s.get("box_options", {}), **b} for b in s["boxes"]]
    names = [b["name"] for b in s["boxes"]]
    if len(set(names)) != len(names):
        raise ValueError(f"box names must be unique: {names}")
    for b in s["boxes"]:
        if b.get("inside") and b["inside"] not in names:
            print(f"WARNING: box {b['name']} is marked inside '{b['inside']}', which is not "
                  "an active box; its share panel is skipped")
    return s


def _parse_utc(text):
    return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M")


def load_event(settings, config_name):
    """Event label, cases and windows (event, storm, day) from an event config."""
    path = os.path.join(settings["event_config_dir"], config_name)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cases = cfg.get("cases") or [cfg["case"]]
    t = cfg["time"]
    t0 = datetime(t["start_year"], t["start_month"], t["start_day"])
    windows = {"event": (t0, t0 + timedelta(days=t["num_days"]))}
    obs = cfg.get("observed_comparison") or {}
    for key, name in [("storm_window", "storm"), ("day_window", "day")]:
        w = obs.get(key)
        if w and w.get("start") and w.get("end"):
            a, b = _parse_utc(w["start"]), _parse_utc(w["end"])
            if b > a:
                windows[name] = (a, b)
    return {"label": cfg["event_label"], "config": config_name,
            "cases": cases, "windows": windows}


def case_nc_path(settings, box_name, event_label, resolution):
    return os.path.join(settings["output_dir"], box_name,
                        f"budget_{event_label}_{resolution}.nc")


def plot_dir(settings, event_label, box_name):
    """plots/<event_label>/<box>/, created if needed."""
    d = os.path.join(settings["plot_dir"], event_label, box_name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# E3SM files
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})-\d{5}\.nc$")


def stream_files(e3sm_dir, stream, t0, t1, pad_days=2):
    """Files of one stream whose date stamp lies within [t0 - pad, t1 + pad]."""
    files = sorted(glob.glob(os.path.join(e3sm_dir, f"*.{stream}.*.nc")))
    keep = []
    for f in files:
        m = _DATE_RE.search(f)
        if m is None:
            keep.append(f)
            continue
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if t0 - timedelta(days=pad_days) <= d <= t1 + timedelta(days=pad_days):
            keep.append(f)
    return keep


def _to_datetime(vals, units, calendar):
    out = []
    for c in np.atleast_1d(cftime.num2date(vals, units, calendar=calendar)):
        out.append(datetime(c.year, c.month, c.day, c.hour, c.minute, c.second))
    return out


def file_times(nc):
    """Record times and, if present, bounds (list of (lo, hi)) as datetimes."""
    tv = nc.variables["time"]
    cal = getattr(tv, "calendar", "noleap")
    times = _to_datetime(tv[:], tv.units, cal)
    bounds = None
    bname = getattr(tv, "bounds", "time_bnds")
    if bname in nc.variables:
        b = nc.variables[bname][:]
        lo = _to_datetime(b[:, 0], tv.units, cal)
        hi = _to_datetime(b[:, 1], tv.units, cal)
        bounds = list(zip(lo, hi))
    return times, bounds


def grid_from_file(path):
    with netCDF4.Dataset(path) as nc:
        lat = np.asarray(nc.variables["lat"][:], dtype=float)
        lon = np.asarray(nc.variables["lon"][:], dtype=float)
        area = np.asarray(nc.variables["area"][:], dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lat, lon, area


# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------
def box_geometry(box):
    """Boundary samples (midpoint rule), outward normals, and box area.

    Returns dict: lon, lat, dl (m), nx, ny (east/north outward unit normal),
    face (name per sample), dist_km (along boundary to sample), face_edges
    (list of (name, start_km, end_km) per edge), area_m2, vertices.
    """
    v = np.asarray(box["vertices"], dtype=float)
    faces = box["faces"]
    nv = len(v)
    if len(faces) != nv:
        raise ValueError(f"box.faces has {len(faces)} names for {nv} edges")
    ds = box["sample_km"] * 1e3

    # Orientation from the signed equal-area polygon area (x = R lon, y = R sin lat).
    x = R_EARTH * np.radians(v[:, 0])
    y = R_EARTH * np.sin(np.radians(v[:, 1]))
    signed = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    sense = 1.0 if signed > 0 else -1.0   # +1 counterclockwise

    out = {k: [] for k in ["lon", "lat", "dl", "nx", "ny", "face", "dist_km"]}
    edges, dense = [], []
    s0 = 0.0
    for i in range(nv):
        a, b = v[i], v[(i + 1) % nv]
        latm = 0.5 * (a[1] + b[1])
        dx = R_EARTH * np.cos(np.radians(latm)) * np.radians(b[0] - a[0])
        dy = R_EARTH * np.radians(b[1] - a[1])
        n = max(1, int(np.ceil(np.hypot(dx, dy) / ds)))
        f = (np.arange(n) + 0.5) / n
        lons = a[0] + f * (b[0] - a[0])
        lats = a[1] + f * (b[1] - a[1])
        for k in range(n + 1):
            dense.append(a + (k / n) * (b - a))
        # local edge vector at each sample latitude, then outward = sense * right normal
        ex = R_EARTH * np.cos(np.radians(lats)) * np.radians(b[0] - a[0]) / n
        ey = np.full(n, dy / n)
        seg = np.hypot(ex, ey)
        out["lon"].append(lons)
        out["lat"].append(lats)
        out["dl"].append(seg)
        out["nx"].append(sense * ey / seg)
        out["ny"].append(-sense * ex / seg)
        out["face"] += [faces[i]] * n
        cum = s0 + np.cumsum(seg) - 0.5 * seg
        out["dist_km"].append(cum / 1e3)
        edges.append((faces[i], s0 / 1e3, (s0 + seg.sum()) / 1e3))
        s0 += seg.sum()
    geo = {k: (np.concatenate(val) if k != "face" else np.array(val))
           for k, val in out.items()}

    d = np.asarray(dense)
    xd = R_EARTH * np.radians(d[:, 0])
    yd = R_EARTH * np.sin(np.radians(d[:, 1]))
    geo["area_m2"] = abs(0.5 * np.sum(xd * np.roll(yd, -1) - np.roll(xd, -1) * yd))
    geo["face_edges"] = edges
    geo["vertices"] = v
    geo["face_names"] = list(dict.fromkeys(faces))
    return geo


def inside_box(lon, lat, vertices):
    return Path(np.asarray(vertices)).contains_points(np.column_stack([lon, lat]))


def in_extent(lon, lat, vertices, margin_deg):
    v = np.asarray(vertices)
    return ((lon >= v[:, 0].min() - margin_deg) & (lon <= v[:, 0].max() + margin_deg) &
            (lat >= v[:, 1].min() - margin_deg) & (lat <= v[:, 1].max() + margin_deg))


def interp_weights(col_lon, col_lat, pt_lon, pt_lat):
    """Linear (Delaunay barycentric) weights from columns to points.

    Coordinates are a local tangent plane around the points' centre, in km.
    Returns (idx, w), both (npt, 3): value at point = sum(w * field[idx]).
    """
    lon0, lat0 = np.mean(pt_lon), np.mean(pt_lat)
    k = np.pi / 180 * R_EARTH / 1e3

    def xy(lo, la):
        return np.column_stack([(lo - lon0) * k * np.cos(np.radians(lat0)), (la - lat0) * k])

    tri = Delaunay(xy(col_lon, col_lat))
    p = xy(pt_lon, pt_lat)
    simp = tri.find_simplex(p)
    if np.any(simp < 0):
        raise ValueError(f"{np.sum(simp < 0)} boundary samples fall outside the column hull; "
                         "increase box.stencil_margin_deg")
    T = tri.transform[simp]
    b = np.einsum("nij,nj->ni", T[:, :2, :], p - T[:, 2, :])
    w = np.column_stack([b, 1.0 - b.sum(axis=1)])
    return tri.simplices[simp], w


def median_spacing_km(lon, lat):
    """Median nearest-neighbour distance between columns (km)."""
    from scipy.spatial import cKDTree
    lat0 = np.mean(lat)
    k = np.pi / 180 * R_EARTH / 1e3
    xy = np.column_stack([lon * k * np.cos(np.radians(lat0)), lat * k])
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


# ---------------------------------------------------------------------------
# Window integrals. Times are datetimes; integrals use seconds.
# ---------------------------------------------------------------------------
def seconds(times, ref):
    return np.array([(t - ref).total_seconds() for t in times])


def _covers(t, t0, t1):
    return t0 >= t[0] - 1e-6 and t1 <= t[-1] + 1e-6


def integrate_inst(t, v, t0, t1):
    """Trapezoid integral of instantaneous series v(t) over [t0, t1] (seconds).

    Window edges between records use linear interpolation of the cumulative
    integral. NaN if the window is not covered.
    """
    if not _covers(t, t0, t1):
        return np.nan
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return np.interp(t1, t, cum) - np.interp(t0, t, cum)


def integrate_mean(lo, hi, rate, t0, t1):
    """Integral over [t0, t1] of interval-mean rates on contiguous [lo, hi]."""
    edges = np.concatenate([[lo[0]], hi])
    if not _covers(edges, t0, t1):
        return np.nan
    cum = np.concatenate([[0.0], np.cumsum(rate * (hi - lo))])
    return np.interp(t1, edges, cum) - np.interp(t0, edges, cum)


def change_inst(t, v, t0, t1):
    if not _covers(t, t0, t1):
        return np.nan
    return np.interp(t1, t, v) - np.interp(t0, t, v)


# ---------------------------------------------------------------------------
# Per-case netCDF
# ---------------------------------------------------------------------------
TIME_UNITS = "hours since 1900-01-01 00:00:00"
TIME_CAL = "noleap"


def _num(times):
    return cftime.date2num([cftime.datetime(t.year, t.month, t.day, t.hour, t.minute,
                                            calendar=TIME_CAL) for t in times],
                           TIME_UNITS, calendar=TIME_CAL)


def write_case_nc(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("t0", len(d["t0"]))
        nc.createDimension("t1", len(d["p_lo"]))
        nc.createDimension("face", len(d["face_names"]))
        nc.createDimension("sample", len(d["s_lon"]))
        nc.createDimension("vertex", len(d["vertices"]))
        nc.createDimension("col", len(d["map_lon"]))

        def var(name, dims, data, units="", long_name=""):
            x = nc.createVariable(name, "f8", dims)
            x[:] = data
            x.units = units
            x.long_name = long_name

        var("t0", ("t0",), _num(d["t0"]), TIME_UNITS, "h0 instant")
        nc["t0"].calendar = TIME_CAL
        var("p_lo", ("t1",), _num(d["p_lo"]), TIME_UNITS, "precip interval start")
        var("p_hi", ("t1",), _num(d["p_hi"]), TIME_UNITS, "precip interval end")
        nc["p_lo"].calendar = TIME_CAL
        nc["p_hi"].calendar = TIME_CAL
        var("P", ("t1",), d["P"], "kg m-2 s-1", "box-mean precipitation, interval mean")
        var("E", ("t0",), d["E"], "kg m-2 s-1", "box-mean QFLX")
        var("W_vap", ("t0",), d["W_vap"], "kg m-2", "box-mean TMQ")
        var("W_cond", ("t0",), d["W_cond"], "kg m-2", "box-mean TMCLDLIQ+TMCLDICE+TMRAINQM")
        var("F_in", ("t0", "face"), d["F_in"], "kg s-1", "vapour inflow through face")
        var("F_out", ("t0", "face"), d["F_out"], "kg s-1", "vapour outflow through face")
        var("qn", ("t0", "sample"), d["qn"], "kg m-1 s-1", "outward normal vapour flux")
        var("s_lon", ("sample",), d["s_lon"], "degrees_east")
        var("s_lat", ("sample",), d["s_lat"], "degrees_north")
        var("s_dl", ("sample",), d["s_dl"], "m", "boundary length per sample")
        var("s_dist", ("sample",), d["s_dist"], "km", "distance along boundary")
        s_face = nc.createVariable("s_face", "i4", ("sample",))
        s_face[:] = [d["face_names"].index(f) for f in d["s_face"]]
        var("v_lon", ("vertex",), d["vertices"][:, 0], "degrees_east")
        var("v_lat", ("vertex",), d["vertices"][:, 1], "degrees_north")
        var("map_lon", ("col",), d["map_lon"], "degrees_east")
        var("map_lat", ("col",), d["map_lat"], "degrees_north")
        var("map_tuq", ("col",), d["map_tuq"], "kg m-1 s-1", "window-mean TUQ")
        var("map_tvq", ("col",), d["map_tvq"], "kg m-1 s-1", "window-mean TVQ")
        nc.face_names = ",".join(d["face_names"])
        for k in ["box", "box_area_m2", "n_cols_inside", "spacing_km", "p_source",
                  "cond_vars", "case", "resolution", "event_label", "map_window"]:
            setattr(nc, k, d[k])


def read_case_nc(path):
    d = {}
    with netCDF4.Dataset(path) as nc:
        for k, x in nc.variables.items():
            d[k] = np.asarray(x[:])
        for k in ["t0", "p_lo", "p_hi"]:
            d[k] = _to_datetime(d[k], TIME_UNITS, TIME_CAL)
        for k in nc.ncattrs():
            d[k] = nc.getncattr(k)
    d["face_names"] = d["face_names"].split(",")
    return d


def window_budget(d, t0, t1):
    """Budget terms over [t0, t1], area means in mm (kg m-2). NaN if not covered."""
    ref = d["t0"][0]
    ts = seconds(d["t0"], ref)
    lo, hi = seconds(d["p_lo"], ref), seconds(d["p_hi"], ref)
    a, b = (t0 - ref).total_seconds(), (t1 - ref).total_seconds()
    A = d["box_area_m2"]
    out = {"P": integrate_mean(lo, hi, d["P"], a, b),
           "E": integrate_inst(ts, d["E"], a, b),
           "dW_vap": change_inst(ts, d["W_vap"], a, b),
           "dW_cond": change_inst(ts, d["W_cond"], a, b)}
    for j, f in enumerate(d["face_names"]):
        out[f"F_in_{f}"] = integrate_inst(ts, d["F_in"][:, j], a, b) / A
        out[f"F_out_{f}"] = integrate_inst(ts, d["F_out"][:, j], a, b) / A
    out["F_in"] = sum(out[f"F_in_{f}"] for f in d["face_names"])
    out["F_out"] = sum(out[f"F_out_{f}"] for f in d["face_names"])
    out["dW"] = out["dW_vap"] + out["dW_cond"]
    out["R"] = out["P"] - (out["F_in"] - out["F_out"] + out["E"] - out["dW"])
    out["DR"] = out["P"] / out["F_in"] if out["F_in"] > 0 else np.nan
    out["R_over_Fin"] = out["R"] / out["F_in"] if out["F_in"] > 0 else np.nan
    return out


def interval_rates(d):
    """All terms as interval means (mm/h) on the precip intervals."""
    ref = d["t0"][0]
    ts = seconds(d["t0"], ref)
    lo, hi = seconds(d["p_lo"], ref), seconds(d["p_hi"], ref)
    A = d["box_area_m2"]
    fin = d["F_in"].sum(axis=1) / A
    fout = d["F_out"].sum(axis=1) / A
    W = d["W_vap"] + d["W_cond"]
    r = {k: np.full(len(lo), np.nan) for k in ["F_in", "F_out", "E", "dW", "P", "R"]}
    for k in range(len(lo)):
        dt = hi[k] - lo[k]
        r["F_in"][k] = integrate_inst(ts, fin, lo[k], hi[k]) / dt
        r["F_out"][k] = integrate_inst(ts, fout, lo[k], hi[k]) / dt
        r["E"][k] = integrate_inst(ts, d["E"], lo[k], hi[k]) / dt
        r["dW"][k] = change_inst(ts, W, lo[k], hi[k]) / dt
    r["P"] = d["P"].astype(float).copy()
    r["R"] = r["P"] - (r["F_in"] - r["F_out"] + r["E"] - r["dW"])
    return {k: v * 3600.0 for k, v in r.items()}
