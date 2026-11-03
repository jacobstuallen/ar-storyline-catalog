"""Step 1. Per event and case, write data/EVENT/phase_RES.nc on the native
columns inside region_bbox:

  time  (3-hr, h1 stamps): prect, snow, rain (mm/hr), trefht, tw (C), snowh (mm SWE)
  time6 (6-hr, h2 stamps): fzl (m MSL, 0 C level)
  ncol: lat, lon, area (sr), zsfc (m), landfrac

Runs every event in rain_snow.yaml; existing files are skipped unless
build.overwrite is true.
"""
import os

import netCDF4 as nc
import numpy as np

import rs_common as rc

opts = rc.load_options()
box = opts["region_bbox"]
st = opts["streams"]
fz_method = opts["build"].get("freezing_level", "highest")
fz_zmax = opts["build"].get("freezing_level_max_m", 10000.0)
overwrite = opts["build"].get("overwrite", False)

for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    label = cfg["event_label"]
    t0, t1 = rc.event_window(cfg)
    print(f"=== {label}: {t0} to {t1}")

    for case in cfg["cases"]:
        res = case["resolution"]
        out = rc.phase_path(opts, label, res)
        if os.path.isfile(out) and not overwrite:
            print(f"  {res}: exists, skip ({out})")
            continue
        try:
            cols, lat, lon, area = rc.grid_cols(case, box, t0, t1)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {res}: SKIP, {e}")
            continue
        dx = np.median(np.sqrt(area)) * rc.R_EARTH_KM
        print(f"  {res}: {cols.size} columns, median spacing {dx:.1f} km")

        try:
            stat = rc.read_static(case, st["static"], ["PHIS", "OCNFRAC"], t0, t1, cols)
            tp, pr = rc.read_2d(case, st["precip"], ["PRECT", "PRECSL", "PRECSC"], t0, t1, cols)
            ts, sf = rc.read_2d(case, st["sfc"], ["TREFHT", "QREFHT", "PS"], t0, t1, cols,
                                optional=("PS",))
        except (FileNotFoundError, KeyError) as e:
            print(f"  {res}: SKIP, {e}")
            continue
        zsfc = stat["PHIS"] / rc.G
        landfrac = 1.0 - stat["OCNFRAC"] if stat["OCNFRAC"] is not None else np.ones_like(zsfc)

        # Precip in mm/hr; rain is the remainder (clipped at 0 for round-off).
        prect = pr["PRECT"] * 3.6e6
        snow = (pr["PRECSL"] + pr["PRECSC"]) * 3.6e6
        rain = np.clip(prect - snow, 0.0, None)
        snow = np.minimum(snow, prect)

        # Align surface fields to precip stamps (both 3-hr means).
        idx = {t: i for i, t in enumerate(ts)}
        take = np.array([idx.get(t, -1) for t in tp])
        print(f"  {res}: {len(tp)} precip records, {np.sum(take < 0)} without surface match")

        def align(a):
            o = np.full((len(tp), cols.size), np.nan)
            o[take >= 0] = a[take[take >= 0]]
            return o

        trefht = align(sf["TREFHT"])
        qref = align(sf["QREFHT"])
        if sf["PS"] is not None:
            ps = align(sf["PS"])
        else:
            print(f"  {res}: no PS on {st['sfc']}, standard-atmosphere PS from zsfc")
            ps = np.broadcast_to(rc.std_atm_ps(zsfc), trefht.shape)
        tw = rc.wetbulb_stull(trefht, rc.rh_from_q(qref, ps, trefht))

        # Snowpack (optional; for rain-on-snow).
        snowh = np.full_like(prect, np.nan)
        try:
            th, sh = rc.read_2d(case, st["snowpack"], ["SNOWHLND"], t0, t1, cols)
            ih = {t: i for i, t in enumerate(th)}
            for i, t in enumerate(tp):
                if t in ih:
                    snowh[i] = sh["SNOWHLND"][ih[t]] * 1000.0
        except (FileNotFoundError, KeyError) as e:
            print(f"  {res}: no snowpack ({e})")

        # Freezing level from 6-hr T and Z3.
        t6, fzl = [], []
        try:
            for t, rec in rc.iter_3d(case, st["profile"], ["T", "Z3"], t0, t1, cols):
                t6.append(t)
                fzl.append(rc.freezing_level(rec["T"], rec["Z3"], zsfc, fz_method, fz_zmax))
        except KeyError as e:
            print(f"  {res}: no freezing level ({e})")
        order = np.argsort(t6) if t6 else []
        t6 = np.array(t6)[order] if t6 else np.array([])
        fzl = np.array(fzl)[order] if t6.size else np.zeros((0, cols.size))
        print(f"  {res}: {t6.size} freezing-level records")

        os.makedirs(os.path.dirname(out), exist_ok=True)
        with nc.Dataset(out, "w") as ds:
            ds.createDimension("time", len(tp))
            ds.createDimension("time6", t6.size)
            ds.createDimension("ncol", cols.size)
            for name, dim, val in [("time", "time", rc.to_seconds(tp)),
                                   ("time6", "time6", rc.to_seconds(t6))]:
                v = ds.createVariable(name, "f8", (dim,))
                v.units = "seconds since 1970-01-01 00:00:00"
                v[:] = val
            fields = [
                ("ncol_index", ("ncol",), cols, "1", "column index in the EAM file", "i4"),
                ("lat", ("ncol",), lat, "degrees_north", "", "f8"),
                ("lon", ("ncol",), lon, "degrees_east", "", "f8"),
                ("area", ("ncol",), area, "sr", "physics grid area", "f8"),
                ("zsfc", ("ncol",), zsfc, "m", "PHIS / g", "f4"),
                ("landfrac", ("ncol",), landfrac, "1", "1 - OCNFRAC", "f4"),
                ("prect", ("time", "ncol"), prect, "mm/hr", "PRECT, 3-hr mean", "f4"),
                ("snow", ("time", "ncol"), snow, "mm/hr", "PRECSL + PRECSC", "f4"),
                ("rain", ("time", "ncol"), rain, "mm/hr", "PRECT - snow", "f4"),
                ("trefht", ("time", "ncol"), trefht - rc.T0, "degC", "TREFHT", "f4"),
                ("tw", ("time", "ncol"), tw, "degC", "wet bulb, Stull 2011", "f4"),
                ("snowh", ("time", "ncol"), snowh, "mm", "SNOWHLND (SWE)", "f4"),
                ("fzl", ("time6", "ncol"), fzl, "m", f"0 C level ({fz_method} crossing), MSL", "f4"),
            ]
            for name, dims, val, units, ln, dt in fields:
                kw = {} if dt == "i4" else {"fill_value": np.nan}
                v = ds.createVariable(name, dt, dims, zlib=True, **kw)
                v.units = units
                v.long_name = ln
                v[:] = val
            ds.event_label = label
            ds.resolution = res
            ds.case_name = case["name"]
            ds.label = rc.case_label(cfg, case)
            ds.region_bbox = str(box)
        print(f"  {res}: wrote {out}")
