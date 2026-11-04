"""Step 1: E3SM h5 precip aggregated to common blocks, per event and mesh.

For each event and case, reads h5 PRECT (time-mean) for cells in region_bbox
over the analysis window, sums it into block_minutes blocks, and writes
data/EVENT/blocks_RES.nc with:
    depth(block, cell)          mm per block
    maxdepth(dur, cell)         max running depth for each duration
    lat, lon, area, zsfc, landfrac (cell)
and, for meshes finer than build.coarsen_to, the same fields aggregated onto
that mesh's cells (prefix coarse_). Also writes data/EVENT/blocks_run_RES.nc
over the event's whole time: block (no coarsening) for
plot_hf_characteristics.py, unless build.full_run_blocks is false.
Options: sub_hourly.yaml.

Run:  python build_blocks_nc.py
"""
import glob
import os

import netCDF4 as nc
import numpy as np

import sh_common as sh

opts = sh.load_options()
B, DURS = sh.check_durations(opts)
BOX = opts["region_bbox"]
ST = opts["streams"]
bopt = opts.get("build", {})
COARSE_RES = bopt.get("coarsen_to")
MIN_COVER = float(bopt.get("coarsen_min_cover", 0.8))


def coarse_grid(case, t0, t1):
    """lat, lon, area of the target mesh inside the box padded by 0.5 deg."""
    files = sh.stream_files(case, ST["precip"], t0, t1) or sorted(
        glob.glob(os.path.join(case["e3sm_dir"], "*.eam.h*.nc")))
    if not files:
        return None
    a, b, c, d = BOX
    _, lat, lon, area = sh.grid_cols(case, [a - 0.5, b - 0.5, c + 0.5, d + 0.5], files)
    return lat, lon, area


def write(path, r, static, md, coarse, attrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("block", r["depth"].shape[0])
        ds.createDimension("cell", r["depth"].shape[1])
        ds.createDimension("dur", len(DURS))
        v = ds.createVariable("block_end", "f8", ("block",))
        v.units = "seconds since 1970-01-01 00:00:00 UTC"
        v[:] = sh.to_seconds(r["block_end"])
        ds.createVariable("duration_min", "i4", ("dur",))[:] = DURS
        for n in ("lat", "lon", "area"):
            ds.createVariable(n, "f8", ("cell",))[:] = r[n]
        ds.createVariable("col", "i4", ("cell",))[:] = r["cols"]
        for n, a in static.items():
            if a is not None:
                ds.createVariable(n, "f4", ("cell",))[:] = a
        v = ds.createVariable("depth", "f4", ("block", "cell"), zlib=True, complevel=4)
        v.units = "mm"
        v[:] = r["depth"]
        v = ds.createVariable("maxdepth", "f4", ("dur", "cell"))
        v.units = "mm"
        v[:] = md
        if coarse is not None:
            ds.createDimension("ccell", coarse["depth"].shape[1])
            for n in ("lat", "lon", "area", "zsfc", "landfrac"):
                if coarse.get(n) is not None:
                    ds.createVariable(f"coarse_{n}", "f8", ("ccell",))[:] = coarse[n]
            v = ds.createVariable("coarse_depth", "f4", ("block", "ccell"), zlib=True, complevel=4)
            v.units = "mm"
            v[:] = coarse["depth"]
            ds.createVariable("coarse_maxdepth", "f4", ("dur", "ccell"))[:] = coarse["maxdepth"]
        for k, val in attrs.items():
            ds.setncattr(k, val)
    print(f"  wrote {path}")


for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    label = cfg["event_label"]
    t0, t1 = sh.analysis_window(opts, cfg)
    print(f"\n=== {label}: {t0} to {t1} UTC, {B}-min blocks")
    by_res = {c["resolution"]: c for c in cfg["cases"]}
    for case in cfg["cases"]:
        res = case["resolution"]
        path = sh.blocks_path(opts, label, res)
        if os.path.isfile(path) and not bopt.get("overwrite", False):
            print(f"  {res}: exists, skip ({path})")
            continue
        print(f"  {res}: {case['name']}")
        try:
            r = sh.read_blocks(case, ST["precip"], t0, t1, BOX, B)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"    SKIP: {e}")
            continue
        print(f"    h5 interval {r['record_s']:.0f} s ({B * 60 / r['record_s']:.0f} records/block), "
              f"{r['depth'].shape[1]} cells, median spacing {sh.median_spacing_km(r['lat'], r['lon']):.1f} km")
        st = sh.read_static(case, ST["static"], r["cols"], ["PHIS", "OCNFRAC"], t0, t1)
        static = {"zsfc": None if st["PHIS"] is None else st["PHIS"] / sh.G,
                  "landfrac": None if st["OCNFRAC"] is None else 1.0 - st["OCNFRAC"]}
        md = sh.max_depths(r["depth"], DURS, B)

        coarse = None
        if COARSE_RES and res != COARSE_RES and COARSE_RES in by_res:
            g = coarse_grid(by_res[COARSE_RES], t0, t1)
            if g is None:
                print(f"    no {COARSE_RES} files; not coarsened")
            else:
                clat, clon, carea = g
                j = sh.coarsen_map(r["lat"], r["lon"], clat, clon)
                cd, W = sh.coarsen(r["depth"].astype(float), r["area"], j, clat.size)
                keep = W >= MIN_COVER * carea
                jk = j[keep[j]]
                def amean(x):
                    if x is None:
                        return None
                    s = np.bincount(j, weights=x * r["area"], minlength=clat.size)
                    return (s / np.where(W > 0, W, np.nan))[keep]
                coarse = dict(lat=clat[keep], lon=clon[keep], area=carea[keep],
                              zsfc=amean(static["zsfc"]), landfrac=amean(static["landfrac"]),
                              depth=cd[:, keep].astype(np.float32))
                coarse["maxdepth"] = sh.max_depths(coarse["depth"], DURS, B)
                print(f"    coarsened onto {keep.sum()} {COARSE_RES} cells "
                      f"(cover >= {MIN_COVER:.0%}; {jk.size} of {j.size} fine cells used)")

        land = np.ones(r["lat"].size, bool) if static["landfrac"] is None else static["landfrac"] >= opts.get("min_landfrac", 0.5)
        tot = np.nansum(r["depth"], axis=0)
        print(f"    land cells {land.sum()}: window total max {np.nanmax(tot[land]):.0f} mm, "
              f"max 20-min {np.nanmax(md[0][land]):.1f} mm, max 1-h "
              f"{np.nanmax(md[DURS.index(60)][land]) if 60 in DURS else np.nan:.1f} mm")
        attrs = dict(event_label=label, resolution=res, case=case["name"],
                     block_minutes=B, record_seconds=r["record_s"],
                     window_start=str(t0), window_end=str(t1),
                     region_bbox=str(BOX), coarsened_to=(COARSE_RES if coarse is not None else ""))
        write(path, r, static, md, coarse, attrs)

# ── full-run blocks for plot_hf_characteristics.py (CW3E-matched windows) ──
if bopt.get("full_run_blocks", True):
    for cfg_name in opts["events"]:
        cfg = sh.load_event(opts, cfg_name)
        label = cfg["event_label"]
        t0, t1 = sh.event_window(cfg)
        print(f"\n=== {label} full run: {t0} to {t1} UTC")
        for case in cfg["cases"]:
            res = case["resolution"]
            path = sh.blocks_path(opts, label, res, "run")
            if os.path.isfile(path) and not bopt.get("overwrite", False):
                print(f"  {res}: exists, skip ({path})")
                continue
            try:
                r = sh.read_blocks(case, ST["precip"], t0, t1, BOX, B)
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"  {res}: SKIP: {e}")
                continue
            st = sh.read_static(case, ST["static"], r["cols"], ["PHIS", "OCNFRAC"], t0, t1)
            static = {"zsfc": None if st["PHIS"] is None else st["PHIS"] / sh.G,
                      "landfrac": None if st["OCNFRAC"] is None else 1.0 - st["OCNFRAC"]}
            md = sh.max_depths(r["depth"], DURS, B)
            write(path, r, static, md, None,
                  dict(event_label=label, resolution=res, case=case["name"], block_minutes=B,
                       record_seconds=r["record_s"], window_start=str(t0), window_end=str(t1),
                       region_bbox=str(BOX), coarsened_to=""))
