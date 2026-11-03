"""
plot_p_ivt_maps.py

Native-grid maps per event: rows are window precipitation, window-mean |IVT|,
efficiency P / IVT (unconditioned, masked where window-mean IVT < threshold)
and AR-conditioned efficiency (masked where fewer than min_cond_hours had
IVT >= threshold); columns are meshes. Model terrain contours and the budget
boxes are overlaid. Columns outside the refined region are left blank.

Reads output/spatial/p_ivt_<event>_<res>.nc from compute_p_ivt.py.
Writes plots/<event>/spatial/p_ivt_maps_<window>.png

Run: python plot_p_ivt_maps.py
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import netCDF4
import numpy as np

import budget_io as bio

S = bio.load_settings()
SP = S["spatial"]
IN_DIR = os.path.join(bio.HERE, SP["output_dir"])
THRESH = float(SP["ivt_threshold"])
EXTENT = SP["map_extent"]
PC = ccrs.PlateCarree()


def load(event_label, res):
    path = os.path.join(IN_DIR, f"p_ivt_{event_label}_{res}.nc")
    if not os.path.exists(path):
        return None
    with netCDF4.Dataset(path) as nc:
        d = {k: np.asarray(v[:], dtype=float) for k, v in nc.variables.items()}
        d.update({k: nc.getncattr(k) for k in nc.ncattrs()})
    refined = d["col_km"] <= SP["coarse_factor"] * SP["nominal_km"][res]
    ivt_mean = d["IVT_int"] / d["window_seconds"]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = d["P_acc"] / d["IVT_int"] * 1e7
        eff_c = d["P_cond"] / d["IVT_cond"] * 1e7
    eff[ivt_mean < THRESH] = np.nan
    eff_c[d["hours_cond"] < SP["min_cond_hours"]] = np.nan
    d["fields"] = {"P": np.where(refined, d["P_acc"], np.nan),
                   "IVT": np.where(refined, ivt_mean, np.nan),
                   "eff": np.where(refined, eff, np.nan),
                   "eff_c": np.where(refined, eff_c, np.nan)}
    # triangulation; drop long triangles so masked gaps are not bridged
    tri = mtri.Triangulation(d["lon"], d["lat"])
    t = tri.triangles
    k = np.pi / 180 * bio.R_EARTH / 1e3
    x = d["lon"] * k * np.cos(np.radians(np.mean(d["lat"])))
    y = d["lat"] * k
    edge = np.max([np.hypot(x[t[:, i]] - x[t[:, j]], y[t[:, i]] - y[t[:, j]])
                   for i, j in [(0, 1), (1, 2), (2, 0)]], axis=0)
    d["tri"] = tri
    d["long_tri"] = edge > 3.0 * np.max(d["col_km"][t], axis=1)
    return d


def tri_plot(ax, d, values, **kw):
    mask = d["long_tri"] | np.isnan(values)[d["tri"].triangles].any(axis=1)
    tri = mtri.Triangulation(d["lon"], d["lat"], d["tri"].triangles, mask=mask)
    return ax.tripcolor(tri, np.nan_to_num(values), shading="flat", transform=PC, **kw)


ROWS = [
    ("P", "Window precipitation (mm)", dict(cmap="Blues", vmin=0, vmax=SP["p_max_mm"])),
    ("IVT", "Window-mean |IVT|\n(kg m$^{-1}$ s$^{-1}$)", dict(cmap="viridis", vmin=0, vmax=SP["ivt_max"])),
    ("eff", f"P / IVT, IVT$\\geq${THRESH:.0f} mean\n(% per 100 km)",
     dict(cmap="magma_r", vmin=0, vmax=SP["eff_max"])),
    ("eff_c", f"P / IVT\n(% per 100 km)",
     dict(cmap="magma_r", vmin=0, vmax=SP["eff_max"])),
]

for cfg_name in S["events"]:
    ev = bio.load_event(S, cfg_name)
    data = {r: load(ev["label"], r) for r in S["resolution_order"]}
    data = {r: d for r, d in data.items() if d is not None}
    if not data:
        print(f"{ev['label']}: no spatial files, skipped")
        continue
    ncol = len(data)
    aspect = (EXTENT[1] - EXTENT[0]) / (EXTENT[3] - EXTENT[2])   # width / height, PlateCarree
    pw = 4.5                                                      # panel width (in)
    fig, axes = plt.subplots(len(ROWS), ncol, figsize=(9,8),
                             subplot_kw={"projection": PC})#, squeeze=False)
    fig.subplots_adjust(left=0.05, right=1.00, top=0.90, hspace=0.10, wspace=0.05, bottom=0.04)
    for j, (res, d) in enumerate(data.items()):
        for i, (key, label, kw) in enumerate(ROWS):
            ax = axes[i, j]
            ax.set_extent(EXTENT, crs=PC)
            m = tri_plot(ax, d, d["fields"][key], **kw)
            ax.tricontour(d["lon"], d["lat"], d["zsfc"], levels=SP["terrain_contours_m"],
                          colors="0.25", linewidths=0.4, transform=PC)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.4")
            for box in S["boxes"]:
                v = np.asarray(box["vertices"])
                ax.plot(np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]],
                        color="tab:red", linewidth=0.8, transform=PC)
            if i == 0:
                ax.set_title(S["resolution_labels"].get(res, res), fontsize=11)
            if j == ncol - 1:
                cb = fig.colorbar(m, ax=axes[i, :], shrink=0.85, pad=0.02, extend="max")
                cb.set_label(label, fontsize=9)
    d0 = next(iter(data.values()))
    fig.suptitle(f"{ev['label']}  {d0['window']} window  {d0['window_start']} to "
                 f"{d0['window_end']} UTC   (contours: model terrain "
                 f"{', '.join(str(z) for z in SP['terrain_contours_m'])} m)", fontsize=11)
    out = os.path.join(S["plot_dir"], ev["label"], "spatial")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"p_ivt_maps_{SP['window']}.png")
    fig.savefig(path, dpi=300)#, bbox_inches="tight")
    plt.close(fig)
    print(path)
