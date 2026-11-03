"""
plot_p_ivt_elevation.py

Efficiency P / IVT binned by each mesh's own model terrain height, within each
region listed in spatial.bin_boxes: a budget box name, or a lon/lat rectangle
from spatial.bin_regions (e.g. the whole West). Land columns in the refined region
only. Each bin is a ratio of totals: sum(P A) / sum(IVT_int A).

One figure per box and mode (all intervals, AR-conditioned):
    rows: efficiency (% per 100 km), precipitation (mm), mean |IVT|
    columns: events, then an all-event composite (ratio of totals summed over
    events; longer windows weigh more)
    lines: meshes; bins with fewer than min_cols_per_bin columns are dropped

Also prints, per box, event and mesh: box-mean efficiency over all refined
columns (land and ocean), the box drying ratio from compute_budget.py output
if present, and the 95th percentile / max model terrain height in the box.

Reads output/spatial/p_ivt_<event>_<res>.nc from compute_p_ivt.py.
Writes plots/multi_event/<box>/p_ivt_elevation_<mode>_<window>.png

Run: python plot_p_ivt_elevation.py
"""

import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

import budget_io as bio

S = bio.load_settings()
SP = S["spatial"]
IN_DIR = os.path.join(bio.HERE, SP["output_dir"])
EDGES = np.arange(0.0, SP["bin_max_m"] + SP["bin_width_m"], SP["bin_width_m"])
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
NB = MID.size
BOXES = {b["name"]: b for b in S["boxes"]}
for name, (x0, x1, y0, y1) in (SP.get("bin_regions") or {}).items():   # lon/lat rectangles
    BOXES[name] = {"name": name, "vertices": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
RESES = S["resolution_order"]
MODES = {"all": ("P_acc", "IVT_int", None),
         "cond": ("P_cond", "IVT_cond", "hours_cond")}


def load(event_label, res):
    path = os.path.join(IN_DIR, f"p_ivt_{event_label}_{res}.nc")
    if not os.path.exists(path):
        return None
    with netCDF4.Dataset(path) as nc:
        d = {k: np.asarray(v[:], dtype=float) for k, v in nc.variables.items()}
        d.update({k: nc.getncattr(k) for k in nc.ncattrs()})
    d["refined"] = d["col_km"] <= SP["coarse_factor"] * SP["nominal_km"][res]
    return d


def bin_sums(d, sel, pk, ik, hk):
    """Per-bin sums of P*A, IVT*A, A, seconds*A and column counts."""
    z = np.clip(d["zsfc"][sel], 0.0, None)
    b = np.minimum((z // SP["bin_width_m"]).astype(int), NB - 1)
    A = d["area"][sel]
    secs = d[hk][sel] * 3600.0 if hk else np.full(A.size, d["window_seconds"])
    out = {}
    for name, w in [("PA", d[pk][sel] * A), ("IA", d[ik][sel] * A), ("A", A),
                    ("TA", secs * A), ("n", np.ones(A.size))]:
        out[name] = np.bincount(b, weights=w, minlength=NB)
    return out


events = [bio.load_event(S, c) for c in S["events"]]
data = {(ev["label"], r): load(ev["label"], r) for ev in events for r in RESES}
labels = [ev["label"] for ev in events if any(data[(ev["label"], r)] is not None for r in RESES)]
if not labels:
    raise SystemExit("no spatial files found; run compute_p_ivt.py first")

for box_name in SP["bin_boxes"]:
    if box_name not in BOXES:
        print(f"{box_name} is not a budget box or a spatial.bin_regions entry, skipped")
        continue
    box = BOXES[box_name]

    # ---- printed summary ----
    print(f"\n{box_name}  ({SP['window']} window)")
    print(f"{'event':>14} {'mesh':>6} {'eff all':>8} {'eff cond':>9} {'DR':>6} "
          f"{'z95 m':>6} {'zmax m':>7} {'ncol':>6}")
    for lab in labels:
        for r in RESES:
            d = data[(lab, r)]
            if d is None:
                continue
            inb = bio.inside_box(d["lon"], d["lat"], box["vertices"]) & d["refined"]
            land = inb & (d["landfrac"] >= SP["land_frac_min"])
            A = d["area"][inb]
            e_all = np.sum(d["P_acc"][inb] * A) / np.sum(d["IVT_int"][inb] * A) * 1e7
            ic = np.sum(d["IVT_cond"][inb] * A)
            e_c = np.sum(d["P_cond"][inb] * A) / ic * 1e7 if ic > 0 else np.nan
            dr = np.nan
            bpath = bio.case_nc_path(S, box_name, lab, r)
            ev = next(e for e in events if e["label"] == lab)
            if os.path.exists(bpath) and SP["window"] in ev["windows"]:
                dr = bio.window_budget(bio.read_case_nc(bpath), *ev["windows"][SP["window"]])["DR"]
            z = d["zsfc"][land]
            z95, zmax = (np.percentile(z, 95), z.max()) if z.size else (np.nan, np.nan)
            print(f"{lab:>14} {r:>6} {e_all:8.2f} {e_c:9.2f} {dr:6.3f} {z95:6.0f} {zmax:7.0f} "
                  f"{inb.sum():6d}")

    # ---- figures ----
    for mode, (pk, ik, hk) in MODES.items():
        cols = labels + ["all events"]
        fig, axes = plt.subplots(3, len(cols), figsize=(3.2 * len(cols), 8.4),
                                 sharex=True, sharey="row", squeeze=False)
        comp = {r: None for r in RESES}
        for j, lab in enumerate(cols):
            for r in RESES:
                if lab == "all events":
                    s = comp[r]
                    if s is None:
                        continue
                    nev = s.pop("nev")
                else:
                    d = data[(lab, r)]
                    if d is None:
                        continue
                    sel = (bio.inside_box(d["lon"], d["lat"], box["vertices"]) & d["refined"] &
                           (d["landfrac"] >= SP["land_frac_min"]))
                    s = bin_sums(d, sel, pk, ik, hk)
                    if comp[r] is None:
                        comp[r] = {k: v.copy() for k, v in s.items()} | {"nev": 1}
                    else:
                        for k in s:
                            comp[r][k] += s[k]
                        comp[r]["nev"] += 1
                    nev = 1
                ok = s["n"] >= SP["min_cols_per_bin"] * nev
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff = np.where(ok & (s["IA"] > 0), s["PA"] / s["IA"] * 1e7, np.nan)
                    pmm = np.where(ok, s["PA"] / s["A"], np.nan)
                    ivt = np.where(ok & (s["TA"] > 0), s["IA"] / s["TA"], np.nan)
                kw = dict(color=S["resolution_colors"][r], marker="o", markersize=2.5,
                          linewidth=1.3, label=S["resolution_labels"].get(r, r))
                axes[0, j].plot(MID, eff, **kw)
                axes[1, j].plot(MID, pmm, **kw)
                axes[2, j].plot(MID, ivt, **kw)
            axes[0, j].set_title(lab, fontsize=10)
            for i in range(3):
                axes[i, j].grid(alpha=0.3, linewidth=0.5)
                axes[i, j].ticklabel_format(axis="y", useOffset=False)
            axes[2, j].set_xlabel("Model terrain height (m)")
        axes[0, 0].set_ylabel("P / IVT (% per 100 km)")
        axes[1, 0].set_ylabel("Precipitation (mm" + (", per event)" if len(labels) > 1 else ")"))
        axes[2, 0].set_ylabel("Mean |IVT| (kg m$^{-1}$ s$^{-1}$)")
        axes[0, 0].legend(fontsize=8, frameon=False)
        what = ("all intervals" if mode == "all"
                else f"intervals with IVT $\\geq$ {SP['ivt_threshold']:.0f} kg m$^{{-1}}$ s$^{{-1}}$")
        fig.suptitle(f"{box_name}: efficiency by model terrain height, land columns, "
                     f"{SP['window']} window, {what}", fontsize=11)
        fig.tight_layout()
        out = os.path.join(S["plot_dir"], "multi_event", box_name)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"p_ivt_elevation_{mode}_{SP['window']}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(path)
