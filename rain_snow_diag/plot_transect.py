"""Along-transect height cross-sections from the 6-hr profile tape (h2).

Per transect and time, rows = resolutions, columns = air temperature (0 C
contour = freezing level), wind along the transect (positive = toward the
end point, i.e. upslope for a coast-to-crest line) and RH. Columns within
half_width_km of the line are interpolated to fixed heights and averaged in
along-track bins of max(bin_km, mesh spacing).

3D cloud liquid / rain / ice are not on any tape; add CLDLIQ, RAINQM, CLDICE
(and OMEGA) to fincl to extend this.

Output: plots/EVENT/transect_NAME_YYYYMMDDHH.png and transect_locator.png.
"""
import warnings

import numpy as np
from matplotlib.colors import TwoSlopeNorm

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
po = opts["plots"]["transect"]
st = opts["streams"]
heights = np.arange(0, po["z_top_m"] + po["dz_m"], po["dz_m"])
z_edges = np.concatenate([[heights[0] - po["dz_m"] / 2], heights + po["dz_m"] / 2])


def geometry(lat, lon, tr):
    (la0, lo0), (la1, lo1) = tr["start"], tr["end"]
    kx = 111.32 * np.cos(np.deg2rad(0.5 * (la0 + la1)))
    ky = 110.57
    ex, ey = (lo1 - lo0) * kx, (la1 - la0) * ky
    L = np.hypot(ex, ey)
    ex, ey = ex / L, ey / L
    x, y = (lon - lo0) * kx, (lat - la0) * ky
    s, nrm = x * ex + y * ey, -x * ey + y * ex
    sel = (np.abs(nrm) <= tr["half_width_km"]) & (s >= 0) & (s <= L)
    return sel, s, ex, ey, L


def tr_bbox(tr, pad=0.5):
    la = [tr["start"][0], tr["end"][0]]
    lo = [tr["start"][1], tr["end"][1]]
    return [min(la) - pad, min(lo) - pad, max(la) + pad, max(lo) + pad]


def auto_times(d, tr):
    """n wettest h2 times along the corridor (reference case), min_sep apart."""
    sel, *_ = geometry(d["lat"], d["lon"], tr)
    sel &= rc.land_mask(d, opts)
    t6 = set(d["time6"])
    score = np.nanmean(d["prect"][:, sel], axis=1)
    cand = [(score[i], t) for i, t in enumerate(d["time"]) if t in t6 and np.isfinite(score[i])]
    chosen = []
    for _, t in sorted(cand, key=lambda x: x[0], reverse=True):
        if all(abs((t - c).total_seconds()) >= po["min_sep_hours"] * 3600 for c in chosen):
            chosen.append(t)
        if len(chosen) == po["n_auto"]:
            break
    return sorted(chosen)


def to_heights(F, Z):
    """(lev, n) on model levels -> (nz, n) on fixed heights; NaN below the lowest level."""
    out = np.full((heights.size, F.shape[1]), np.nan)
    for j in range(F.shape[1]):
        zz, ff = Z[::-1, j], F[::-1, j]
        out[:, j] = np.interp(heights, zz, ff, left=np.nan, right=np.nan)
    return out


def bin_mean(F, ib, nbin):
    """Average columns (last axis) into along-track bins."""
    out = np.full(F.shape[:-1] + (nbin,), np.nan)
    for b in range(nbin):
        m = ib == b
        if m.any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[..., b] = np.nanmean(F[..., m], axis=-1)
    return out


def fill_gaps(F, sc):
    """Linearly fill empty interior along-track bins (coarse mesh, narrow corridor)."""
    G = F.copy()
    flat = G.reshape(-1, G.shape[-1])
    for row in flat:
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            inside = (sc >= sc[ok][0]) & (sc <= sc[ok][-1]) & ~ok
            row[inside] = np.interp(sc[inside], sc[ok], row[ok])
    return G


PANELS = [
    ("T", "air temperature (C)", "RdBu_r", np.arange(-30, 22, 2), 0.0),
    ("W", "wind along transect (m/s)", "PuOr_r", np.arange(-10, 42, 2), 0.0),
    ("RH", "relative humidity (%)", "Blues", np.arange(0, 105, 5), None),
]

for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    label = cfg["event_label"]
    t0, t1 = rc.event_window(cfg)
    print(f"=== {label}")
    built = rc.built_cases(opts, cfg)
    if not built:
        continue
    ref = built[0][1]

    # Locator map on the finest built mesh.
    fine = built[-1][1]
    box = opts["region_bbox"]
    extent = [box[1] - 1.0, box[3] + 0.5, box[0] - 0.5, box[2] + 0.5]
    fig = plt.figure(figsize=(5.5, 5))
    ax = rc.map_axes(fig, 1, 1, extent)[0, 0]
    tri = rc.triangulation(fine["lon"], fine["lat"])
    pc = ax.tripcolor(tri, fine["zsfc"], cmap="Greys", vmin=0, vmax=3500, shading="gouraud")
    for tr in po["transects"]:
        (la0, lo0), (la1, lo1) = tr["start"], tr["end"]
        ax.plot([lo0, lo1], [la0, la1], color=rc.RES_COLORS["30x16"], lw=2)
        ax.text(lo0, la0, f" {tr['name']} start", fontsize=7, va="top")
    fig.colorbar(pc, ax=ax, shrink=0.7, label="surface elevation (m)")
    ax.set_title(f"{fine['label']} terrain and transects", loc="left")
    rc.save_png(fig, rc.plot_path(opts, label, "transect_locator.png"))
    plt.close(fig)

    for tr in po["transects"]:
        times = [rc.parse_utc(s) for s in po.get("times", {}).get(label, [])] or auto_times(ref, tr)
        if not times:
            print(f"  {tr['name']}: no times")
            continue
        print(f"  {tr['name']}: times {[t.strftime('%Y-%m-%d %H') for t in times]}")

        # Read every case once for all times.
        sections = {}  # res -> {time: {field: (nz, nbin)}}, plus geometry
        for case, _ in built:
            res = case["resolution"]
            try:
                cols, lat, lon, area = rc.grid_cols(case, tr_bbox(tr), t0, t1)
            except (FileNotFoundError, ValueError) as e:
                print(f"  {res}: {e}")
                continue
            sel, s, ex, ey, L = geometry(lat, lon, tr)
            cols, s, area = cols[sel], s[sel], area[sel]
            if cols.size == 0:
                print(f"  {res}: no columns in corridor")
                continue
            dx = np.median(np.sqrt(area)) * rc.R_EARTH_KM
            bw = max(tr["bin_km"], dx)
            nbin = int(np.ceil(L / bw))
            ib = np.minimum((s / bw).astype(int), nbin - 1)
            zs = rc.read_static(case, st["static"], ["PHIS"], t0, t1, cols)["PHIS"] / rc.G
            print(f"  {res}: {cols.size} columns, bin {bw:.1f} km")
            want = set(times)
            got = {}
            for t, rec in rc.iter_3d(case, st["profile"], ["T", "U", "V", "Q", "Z3", "PS"],
                                     t0, t1, cols, want=want):
                p = rec["hyam"][:, None] * rec["P0"] + rec["hybm"][:, None] * rec["PS"][None, :]
                f = {"T": rec["T"] - rc.T0,
                     "W": rec["U"] * ex + rec["V"] * ey,
                     "RH": rc.rh_from_q(rec["Q"], p, rec["T"])}
                got[t] = {k: bin_mean(to_heights(v, rec["Z3"]), ib, nbin) for k, v in f.items()}
            missing = want - set(got)
            if missing:
                print(f"  {res}: no h2 record at {sorted(missing)}")
            s_edges = np.arange(nbin + 1) * bw
            s_edges[-1] = L
            sc = 0.5 * (s_edges[:-1] + s_edges[1:])
            zb = fill_gaps(bin_mean(zs[None, :], ib, nbin), sc)[0]
            empty = np.isnan(bin_mean(zs[None, :], ib, nbin)[0]).sum()
            if empty:
                print(f"  {res}: {empty} of {nbin} bins empty, filled along track")
            got = {t: {k: fill_gaps(v, sc) for k, v in f.items()} for t, f in got.items()}
            # Re-mask below the (filled) terrain so fills don't bleed under ground.
            for f in got.values():
                for v in f.values():
                    v[heights[:, None] < zb[None, :]] = np.nan
            sections[res] = (case, got, s_edges, zb)

        for t in times:
            rows = [(r, v) for r, v in sections.items() if t in v[1]]
            if not rows:
                continue
            fig, axes = plt.subplots(len(rows), 3, figsize=(15, 2.8 * len(rows) + 0.8),
                                     sharex=True, sharey=True, squeeze=False)
            for i, (res, (case, got, s_edges, zb)) in enumerate(rows):
                sc = 0.5 * (s_edges[:-1] + s_edges[1:])
                for j, (key, title, cmap, lev, center) in enumerate(PANELS):
                    ax = axes[i, j]
                    F = got[t][key]
                    norm = TwoSlopeNorm(center, lev[0], lev[-1]) if center is not None else None
                    pm = ax.pcolormesh(s_edges, z_edges / 1000, F, cmap=cmap, norm=norm,
                                       vmin=None if norm else lev[0], vmax=None if norm else lev[-1])
                    if key == "T":
                        ax.contour(sc, heights / 1000, F, levels=[0], colors=rc.INK, linewidths=1.6)
                    if key == "RH":
                        ax.contour(sc, heights / 1000, F, levels=[90], colors=rc.INK, linewidths=0.8)
                    ok = np.isfinite(zb)
                    ax.fill_between(sc[ok], 0, zb[ok] / 1000, color="#4a4a46", lw=0, zorder=3)
                    ax.set_title(f"{rc.case_label(cfg, case)}: {title}", loc="left")
                    if i == len(rows) - 1:
                        fig.colorbar(pm, ax=axes[:, j], orientation="horizontal",
                                     fraction=0.05 / len(rows) ** 0.5, pad=0.18 / len(rows) ** 0.5)
                        ax.set_xlabel(f"distance from {tr['start']} (km)")
                axes[i, 0].set_ylabel("height (km MSL)")
            fig.suptitle(f"{label} {tr['name']} transect, {t:%Y-%m-%d %H} UTC "
                         "(black: 0 C in T, 90% in RH)", x=0.01, ha="left")
            rc.save_png(fig, rc.plot_path(opts, label, f"transect_{tr['name']}_{t:%Y%m%d%H}.png"))
            plt.close(fig)
