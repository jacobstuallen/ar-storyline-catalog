"""
compute_budget.py

Regional water budget for every event, resolution and box in budget.yaml.
Each case's files are read once and all boxes are computed in the same pass.
For each case, reads h0 (TUQ, TVQ, QFLX, TMQ, TM* condensate burdens) and the
precip stream (3-hourly mean PRECT), and writes time series to
output/budget/<box>/budget_<event_label>_<resolution>.nc:

  F_in, F_out (t0, face)  vapour flux through each face, kg/s
  qn (t0, sample)         outward normal flux at each boundary sample
  E, W_vap, W_cond (t0)   box area means of QFLX, TMQ, condensate burden
  P (t1)                  box area mean PRECT, interval means
  map_tuq, map_tvq        storm-window mean IVT components around the box

Window totals and the drying ratio are computed later from these files
(budget_io.window_budget), so windows can change without rerunning this.

Boundary flux: TUQ/TVQ interpolated linearly (Delaunay) from native columns
to samples every sample_km along the box edges. Area means: native columns
with centres inside the box, weighted by `area`.

Run:  python compute_budget.py
"""

from datetime import timedelta

import netCDF4
import numpy as np

import budget_io as bio

settings = bio.load_settings()
PAD = timedelta(hours=3)

geos = {}
for box in settings["boxes"]:
    g = bio.box_geometry(box)
    geos[box["name"]] = g
    print(f"Box '{box['name']}': area {g['area_m2'] / 1e6:,.0f} km2, "
          f"perimeter {g['dl'].sum() / 1e3:,.0f} km, {len(g['dl'])} boundary samples")
    for name, a, b in g["face_edges"]:
        print(f"  edge {name:6s} {a:7.1f} - {b:7.1f} km")


def read_cols(nc, var, sel):
    x = nc.variables[var][sel, :]
    return np.asarray(np.ma.filled(x.astype(float), np.nan))


for cfg_name in settings["events"]:
    ev = bio.load_event(settings, cfg_name)
    t0, t1 = ev["windows"]["event"]
    map_name = "storm" if "storm" in ev["windows"] else "event"
    m0, m1 = ev["windows"][map_name]
    print(f"\n=== {ev['label']}  event window {t0} to {t1}")

    for case in ev["cases"]:
        res, cname, edir = case["resolution"], case["name"], case["e3sm_dir"]
        h0s = case.get("e3sm_stream", settings["streams"]["h0"])
        files = bio.stream_files(edir, h0s, t0, t1)
        if not files:
            print(f"  {res}: SKIP, no {h0s} files in {edir}")
            continue

        with netCDF4.Dataset(files[0]) as nc:
            have = set(nc.variables)
        missing = [v for v in bio.H0_REQUIRED if v not in have]
        if missing:
            print(f"  {res}: SKIP, {h0s} lacks {missing}")
            continue
        cond_vars = [v for v in bio.CONDENSATE if v in have]
        if len(cond_vars) < len(bio.CONDENSATE):
            print(f"  {res}: WARNING, condensate burdens missing "
                  f"{sorted(set(bio.CONDENSATE) - set(cond_vars))}; storage uses {cond_vars}")
        have_prect = "PRECT" in have

        # Grid and, per box: columns inside, interpolation stencil, map columns
        lat, lon, area = bio.grid_from_file(files[0])
        bx = {}
        for box in settings["boxes"]:
            name, g = box["name"], geos[box["name"]]
            in_idx = np.where(bio.inside_box(lon, lat, g["vertices"]))[0]
            if len(in_idx) < 3:
                print(f"  {res}: box {name} SKIP, only {len(in_idx)} columns inside")
                continue
            st_idx = np.where(bio.in_extent(lon, lat, g["vertices"], box["stencil_margin_deg"]))[0]
            tri_idx, tri_w = bio.interp_weights(lon[st_idx], lat[st_idx], g["lon"], g["lat"])
            map_idx = np.where(bio.in_extent(lon, lat, g["vertices"], settings["map_margin_deg"]))[0]
            bx[name] = dict(g=g, in_idx=in_idx, wts=area[in_idx] / area[in_idx].sum(),
                            tri_idx=st_idx[tri_idx], tri_w=tri_w, map_idx=map_idx,
                            rec={}, prec={}, map_sum=np.zeros((2, len(map_idx))), map_n=0)
        if not bx:
            continue

        # ---- h0 records (each file read once for all boxes)
        seen = set()
        for f in files:
            with netCDF4.Dataset(f) as nc:
                times, _ = bio.file_times(nc)
                sel = [i for i, t in enumerate(times) if t0 - PAD <= t <= t1 + PAD and t not in seen]
                if not sel:
                    continue
                tuq = read_cols(nc, "TUQ", sel)
                tvq = read_cols(nc, "TVQ", sel)
                qflx = read_cols(nc, "QFLX", sel)
                tmq = read_cols(nc, "TMQ", sel)
                cond = sum(read_cols(nc, v, sel) for v in cond_vars) if cond_vars \
                    else np.zeros_like(tmq)
                prect = read_cols(nc, "PRECT", sel) if have_prect else None
            for k, i in enumerate(sel):
                t = times[i]
                seen.add(t)
                for b in bx.values():
                    g, ii, w = b["g"], b["in_idx"], b["wts"]
                    qu = (tuq[k, b["tri_idx"]] * b["tri_w"]).sum(axis=1)
                    qv = (tvq[k, b["tri_idx"]] * b["tri_w"]).sum(axis=1)
                    b["rec"][t] = dict(
                        qn=qu * g["nx"] + qv * g["ny"],
                        E=np.sum(qflx[k, ii] * w), W_vap=np.sum(tmq[k, ii] * w),
                        W_cond=np.sum(cond[k, ii] * w),
                        P_inst=np.sum(prect[k, ii] * w) * 1000.0 if have_prect else np.nan)
                    if m0 <= t <= m1:
                        b["map_sum"] += np.vstack([tuq[k, b["map_idx"]], tvq[k, b["map_idx"]]])
                        b["map_n"] += 1

        tt = sorted(seen)
        if len(tt) < 2:
            print(f"  {res}: SKIP, fewer than 2 h0 records in the event window")
            continue

        # ---- precipitation: interval means from the precip stream
        ps = settings["streams"]["precip"]
        pfiles = bio.stream_files(edir, ps, t0, t1)
        pseen = set()
        for f in pfiles:
            with netCDF4.Dataset(f) as nc:
                if "PRECT" not in nc.variables or \
                        "mean" not in getattr(nc.variables["PRECT"], "cell_methods", ""):
                    continue
                times, bnds = bio.file_times(nc)
                if bnds is None:
                    continue
                sel = [i for i, (lo, hi) in enumerate(bnds)
                       if hi > lo and hi > t0 - PAD and lo < t1 + PAD and hi not in pseen]
                if not sel:
                    continue
                p = read_cols(nc, "PRECT", sel)
            for k, i in enumerate(sel):
                pseen.add(bnds[i][1])
                for b in bx.values():
                    b["prec"][bnds[i][1]] = (bnds[i][0],
                                             np.sum(p[k, b["in_idx"]] * b["wts"]) * 1000.0)

        for name, b in bx.items():
            g, rec = b["g"], b["rec"]
            qn = np.array([rec[t]["qn"] for t in tt])
            F_in = np.column_stack([(np.maximum(-qn[:, g["face"] == f], 0) *
                                     g["dl"][g["face"] == f]).sum(axis=1) for f in g["face_names"]])
            F_out = np.column_stack([(np.maximum(qn[:, g["face"] == f], 0) *
                                      g["dl"][g["face"] == f]).sum(axis=1) for f in g["face_names"]])

            if b["prec"]:
                p_source = f"{ps} PRECT (time mean)"
                p_hi = sorted(b["prec"])
                p_lo = [b["prec"][h][0] for h in p_hi]
                P = np.array([b["prec"][h][1] for h in p_hi])
                gaps = sum(1 for x, y in zip(p_hi[:-1], p_lo[1:]) if x != y)
                if gaps:
                    print(f"  {res}: WARNING, {gaps} gaps between precip intervals")
            elif have_prect:
                p_source = f"{h0s} PRECT (instantaneous, trapezoid)"
                print(f"  {res}: WARNING, no time-mean PRECT in {ps}; using {h0s} PRECT")
                pi = np.array([rec[t]["P_inst"] for t in tt])
                p_lo, p_hi = tt[:-1], tt[1:]
                P = 0.5 * (pi[1:] + pi[:-1])
            else:
                print(f"  {res}: SKIP, no PRECT in {ps} or {h0s}")
                break

            n = max(b["map_n"], 1)
            out = dict(
                t0=tt, p_lo=p_lo, p_hi=p_hi, P=P,
                E=np.array([rec[t]["E"] for t in tt]),
                W_vap=np.array([rec[t]["W_vap"] for t in tt]),
                W_cond=np.array([rec[t]["W_cond"] for t in tt]),
                F_in=F_in, F_out=F_out, qn=qn, face_names=g["face_names"],
                s_lon=g["lon"], s_lat=g["lat"], s_dl=g["dl"], s_dist=g["dist_km"],
                s_face=list(g["face"]), vertices=g["vertices"],
                map_lon=lon[b["map_idx"]], map_lat=lat[b["map_idx"]],
                map_tuq=b["map_sum"][0] / n, map_tvq=b["map_sum"][1] / n,
                box=name, box_area_m2=g["area_m2"], n_cols_inside=len(b["in_idx"]),
                spacing_km=bio.median_spacing_km(lon[b["in_idx"]], lat[b["in_idx"]]),
                p_source=p_source, cond_vars=",".join(cond_vars), case=cname,
                resolution=res, event_label=ev["label"], map_window=f"{map_name}: {m0} to {m1}")
            path = bio.case_nc_path(settings, name, ev["label"], res)
            bio.write_case_nc(path, out)

            print(f"  {res} [{name}]: {len(tt)} h0 records {tt[0]} to {tt[-1]}, {len(P)} precip "
                  f"intervals ({p_source}); {len(b['in_idx'])} columns inside, "
                  f"spacing {out['spacing_km']:.1f} km")
            d = bio.read_case_nc(path)
            for wname, (wa, wb_) in ev["windows"].items():
                wb = bio.window_budget(d, wa, wb_)
                if np.isnan(wb["P"]) or np.isnan(wb["F_in"]):
                    print(f"      {wname:5s}: window not covered")
                    continue
                print(f"      {wname:5s}: P {wb['P']:6.1f}  F_in {wb['F_in']:7.1f}  "
                      f"F_out {wb['F_out']:7.1f}  E {wb['E']:5.1f}  dW {wb['dW']:6.1f}  "
                      f"R {wb['R']:6.1f} mm   DR {wb['DR']:.3f}  "
                      f"R/F_in {100 * wb['R_over_Fin']:5.1f}%")
            print(f"      -> {path}")
