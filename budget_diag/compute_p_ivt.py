"""
compute_p_ivt.py

Native-grid window totals for spatial precipitation efficiency, one file per
event and mesh: output/spatial/p_ivt_<event>_<res>.nc

Per column, over the window (spatial.window in budget.yaml):
    P_acc     precipitation (mm), from the precip stream (h1 PRECT interval means)
    IVT_int   integral of |IVT| dt (kg m-1), |IVT| = |(TUQ, TVQ)| from h0
    TUQ_int, TVQ_int   integrals of the components (for flow vectors)
    P_cond, IVT_cond, hours_cond   same, only over 3-h intervals with
              interval-mean |IVT| >= spatial.ivt_threshold (AR-conditioned)
    zsfc      model terrain height PHIS / g (m), landfrac

Everything is accumulated on the precip intervals, so P and IVT cover exactly
the same time. h0 |IVT| is interpolated linearly in time to each interval's
start and end and integrated with the trapezoid rule; intervals cut by the
window edge count by their overlap.

Efficiency (computed in the plot scripts) = P_acc / IVT_int, in 1/m;
x 1e7 gives % per 100 km.

Run: python compute_p_ivt.py
"""

import os

import netCDF4
import numpy as np

import budget_io as bio

G = 9.80616   # m s-2, EAM value
RHO_W = 1000.0

S = bio.load_settings()
SP = S["spatial"]
OUT_DIR = os.path.join(bio.HERE, SP["output_dir"])
os.makedirs(OUT_DIR, exist_ok=True)
LON0, LON1, LAT0, LAT1 = SP["region"]
THRESH = float(SP["ivt_threshold"])


def time_interp(t, v, x):
    """Linear interpolation of v (nt, ncol) at times t to times x (nx,)."""
    i = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
    f = ((x - t[i]) / (t[i + 1] - t[i]))[:, None]
    return v[i] * (1.0 - f) + v[i + 1] * f


def read_stream(files, names, idx):
    """Records from several files, subset to columns idx, sorted, duplicates dropped."""
    times, bounds, data = [], [], {n: [] for n in names}
    for f in files:
        with netCDF4.Dataset(f) as nc:
            if not all(n in nc.variables for n in names):
                continue
            t, b = bio.file_times(nc)
            times += t
            bounds += b if b is not None else [None] * len(t)
            for n in names:
                data[n].append(np.asarray(nc.variables[n][:], dtype=float)[:, idx])
    if not times:
        return None
    order = np.argsort(np.array(times))
    times = [times[k] for k in order]
    bounds = [bounds[k] for k in order]
    data = {n: np.concatenate(v, axis=0)[order] for n, v in data.items()}
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = [times[k] != times[k - 1] for k in range(1, len(times))]
    return {"times": [t for t, k in zip(times, keep) if k],
            "bounds": [b for b, k in zip(bounds, keep) if k],
            **{n: v[keep] for n, v in data.items()}}


def read_topo(path, lat_h0, lon_h0):
    with netCDF4.Dataset(path) as nc:
        phis = np.asarray(nc.variables["PHIS"][:], dtype=float)
        lat = np.asarray(nc.variables["lat"][:], dtype=float)
        lon = np.asarray(nc.variables["lon"][:], dtype=float)
        lf = (np.asarray(nc.variables["LANDFRAC"][:], dtype=float)
              if "LANDFRAC" in nc.variables else np.full(phis.shape, np.nan))
    if phis.shape != lat_h0.shape:
        raise ValueError(f"topo ncol {phis.size} != h0 ncol {lat_h0.size}: {path}")
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    dlon = np.abs((lon - lon_h0 + 180.0) % 360.0 - 180.0)
    err = max(np.max(np.abs(lat - lat_h0)), np.max(dlon))
    if err > 1e-4:
        raise ValueError(f"topo lat/lon do not match h0 column order (max diff {err:.3g} deg): {path}")
    return phis / G, lf


for cfg_name in S["events"]:
    ev = bio.load_event(S, cfg_name)
    if SP["window"] not in ev["windows"]:
        print(f"{ev['label']}: no '{SP['window']}' window, skipped")
        continue
    w0, w1 = ev["windows"][SP["window"]]

    for case in ev["cases"]:
        res = case["resolution"]
        tag = f"{ev['label']} {res}"
        h0_files = bio.stream_files(case["e3sm_dir"], S["streams"]["h0"], w0, w1)
        if not h0_files:
            print(f"{tag}: no h0 files, skipped")
            continue

        lat, lon, area = bio.grid_from_file(h0_files[0])
        area_m2 = area * bio.R_EARTH ** 2 if area.sum() < 13.0 else area   # steradians or m2
        idx = np.where((lon >= LON0) & (lon <= LON1) & (lat >= LAT0) & (lat <= LAT1))[0]

        topo = SP.get("topo_files", {}).get(res)
        if not topo:
            with netCDF4.Dataset(h0_files[0]) as nc:
                topo = getattr(nc, "topography_file", None)
        if not topo or not os.path.exists(topo):
            print(f"{tag}: topography file not found ({topo}), skipped")
            continue
        zsfc, landfrac = read_topo(topo, lat, lon)

        # h0 instantaneous vapour flux
        h0 = read_stream(h0_files, ["TUQ", "TVQ"], idx)
        ref = h0["times"][0]
        th = bio.seconds(h0["times"], ref)
        ivt_mag = np.hypot(h0["TUQ"], h0["TVQ"])

        # precip intervals: h1 interval means, else h0 instantaneous PRECT
        p_files = bio.stream_files(case["e3sm_dir"], S["streams"]["precip"], w0, w1)
        pr = read_stream(p_files, ["PRECT"], idx) if p_files else None
        if pr is not None and all(b is not None for b in pr["bounds"]):
            lo = bio.seconds([b[0] for b in pr["bounds"]], ref)
            hi = bio.seconds([b[1] for b in pr["bounds"]], ref)
            rate = pr["PRECT"] * RHO_W
            p_source = f"{S['streams']['precip']} PRECT interval mean"
        else:
            hp = read_stream(h0_files, ["PRECT"], idx)
            tp = bio.seconds(hp["times"], ref)
            lo, hi = tp[:-1], tp[1:]
            rate = 0.5 * (hp["PRECT"][:-1] + hp["PRECT"][1:]) * RHO_W
            p_source = "h0 PRECT instantaneous, endpoint mean (fallback)"
            print(f"{tag}: WARNING precip stream missing or not a time mean; using h0 PRECT")

        a, b = (w0 - ref).total_seconds(), (w1 - ref).total_seconds()
        overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
        k = overlap > 0
        if not k.any() or lo[k].min() > a + 1 or hi[k].max() < b - 1 or \
                lo[k].min() < th[0] - 1 or hi[k].max() > th[-1] + 1:
            print(f"{tag}: data do not cover the {SP['window']} window, skipped")
            continue
        lo, hi, rate, overlap = lo[k], hi[k], rate[k], overlap[k]
        if np.sum(overlap) < (b - a) - 1:
            print(f"{tag}: WARNING gaps in precip intervals ({np.sum(overlap)/3600:.1f} of "
                  f"{(b - a)/3600:.1f} h covered)")

        # interval-mean |IVT| and components, trapezoid between interval ends
        def interval_mean(v):
            return 0.5 * (time_interp(th, v, lo) + time_interp(th, v, hi))

        ivt_k = interval_mean(ivt_mag)
        tuq_k = interval_mean(h0["TUQ"])
        tvq_k = interval_mean(h0["TVQ"])
        o = overlap[:, None]
        cond = ivt_k >= THRESH

        out = {
            "P_acc": np.sum(rate * o, axis=0),
            "IVT_int": np.sum(ivt_k * o, axis=0),
            "TUQ_int": np.sum(tuq_k * o, axis=0),
            "TVQ_int": np.sum(tvq_k * o, axis=0),
            "P_cond": np.sum(rate * o * cond, axis=0),
            "IVT_cond": np.sum(ivt_k * o * cond, axis=0),
            "hours_cond": np.sum(o * cond, axis=0) / 3600.0,
        }

        path = os.path.join(OUT_DIR, f"p_ivt_{ev['label']}_{res}.nc")
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("col", idx.size)
            meta = {
                "lon": (lon[idx], "degrees_east", ""),
                "lat": (lat[idx], "degrees_north", ""),
                "area": (area_m2[idx], "m2", "column area"),
                "col_km": (np.sqrt(area_m2[idx]) / 1e3, "km", "sqrt(column area)"),
                "zsfc": (zsfc[idx], "m", "model terrain height, PHIS/g"),
                "landfrac": (landfrac[idx], "1", "land fraction"),
                "P_acc": (out["P_acc"], "kg m-2", "window precipitation"),
                "IVT_int": (out["IVT_int"], "kg m-1", "window integral of |IVT|"),
                "TUQ_int": (out["TUQ_int"], "kg m-1", "window integral of TUQ"),
                "TVQ_int": (out["TVQ_int"], "kg m-1", "window integral of TVQ"),
                "P_cond": (out["P_cond"], "kg m-2", "precipitation, intervals with IVT >= threshold"),
                "IVT_cond": (out["IVT_cond"], "kg m-1", "|IVT| integral, intervals with IVT >= threshold"),
                "hours_cond": (out["hours_cond"], "h", "hours with IVT >= threshold"),
            }
            for name, (data, units, long_name) in meta.items():
                x = nc.createVariable(name, "f8", ("col",))
                x[:] = data
                x.units = units
                x.long_name = long_name
            nc.event_label = ev["label"]
            nc.case = case["name"]
            nc.resolution = res
            nc.window = SP["window"]
            nc.window_start = w0.strftime("%Y-%m-%d %H:%M")
            nc.window_end = w1.strftime("%Y-%m-%d %H:%M")
            nc.window_seconds = b - a
            nc.ivt_threshold = THRESH
            nc.p_source = p_source
            nc.topography_file = topo

        land = landfrac[idx] > 0.5
        print(f"{tag}: {idx.size} cols, {np.sum(overlap)/3600:.0f} h, "
              f"max P {out['P_acc'].max():.0f} mm, land max zsfc {zsfc[idx][land].max():.0f} m -> {path}")
