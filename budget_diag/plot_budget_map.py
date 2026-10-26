"""
plot_budget_map.py

Check figure, one per event and box:
  top row     storm-window mean IVT (|TUQ, TVQ|) and vectors around the box,
              one map per resolution, box outline with face names
  bottom      storm-window inward vapour flux along the box boundary, as each
              km of boundary's contribution to the box-mean budget (mm per km),
              all resolutions on one axis. Positive = into the box. The area
              under the curve is F_in - F_out for the window.

Needs compute_budget.py first.

Output: plots/<event_label>/<box>/budget_map.png

Run:  python plot_budget_map.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import budget_io as bio

QUIVER_DEG = 0.5        # one vector per this many degrees
WINDOW = "storm"        # storm | day | event; falls back to event if missing
USE_CARTOPY = True      # coastlines/states from the Natural Earth cache (prefetch.py)

PROJ = None
if USE_CARTOPY:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        PROJ = ccrs.PlateCarree()
    except ImportError:
        print("cartopy not available, maps drawn without coastlines")

settings = bio.load_settings()
order = settings["resolution_order"]
labels = settings["resolution_labels"]
colors = settings["resolution_colors"]

plt.rcParams.update({"font.size": 9})

for cfg_name in settings["events"]:
    ev = bio.load_event(settings, cfg_name)
    for box in settings["boxes"]:
        cases = [(res, bio.read_case_nc(bio.case_nc_path(settings, box["name"], ev["label"], res)))
                 for res in order
                 if os.path.exists(bio.case_nc_path(settings, box["name"], ev["label"], res))]
        if not cases:
            print(f"{ev['label']} [{box['name']}]: no budget files, skipping")
            continue
        wname = WINDOW if WINDOW in ev["windows"] else "event"
        a, b = ev["windows"][wname]

        n = len(cases)
        fig = plt.figure(figsize=(4.0 * n, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(2, n, height_ratios=[1.5, 1])
        vmax = np.nanpercentile(np.concatenate(
            [np.hypot(d["map_tuq"], d["map_tvq"]) for _, d in cases]), 99)

        for j, (res, d) in enumerate(cases):
            ax = fig.add_subplot(gs[0, j], projection=PROJ) if PROJ else fig.add_subplot(gs[0, j])
            kw = {"transform": PROJ} if PROJ else {}
            ivt = np.hypot(d["map_tuq"], d["map_tvq"])
            tri = mtri.Triangulation(d["map_lon"], d["map_lat"])
            pc = ax.tripcolor(tri, ivt, cmap="Blues", vmin=0, vmax=vmax, shading="gouraud", **kw)
            key = np.round(d["map_lon"] / QUIVER_DEG) * 1000 + np.round(d["map_lat"] / QUIVER_DEG)
            _, q = np.unique(key, return_index=True)
            ax.quiver(d["map_lon"][q], d["map_lat"][q], d["map_tuq"][q], d["map_tvq"][q],
                      color="0.25", scale=vmax * 25, width=0.003, **kw)
            vx = np.append(d["v_lon"], d["v_lon"][0])
            vy = np.append(d["v_lat"], d["v_lat"][0])
            ax.plot(vx, vy, color="black", lw=1.4, **kw)
            for f_i, f in enumerate(d["face_names"]):
                m = d["s_face"] == f_i
                ax.text(d["s_lon"][m].mean(), d["s_lat"][m].mean(), f, fontsize=7, ha="center",
                        va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1), **kw)
            if PROJ:
                ax.coastlines("50m", lw=0.6, color="0.3")
                ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.4, edgecolor="0.5")
                ax.set_extent([d["map_lon"].min(), d["map_lon"].max(),
                               d["map_lat"].min(), d["map_lat"].max()], crs=PROJ)
            ax.set_title(f"{labels.get(res, res)}  ({d['n_cols_inside']} columns in box, "
                         f"{d['spacing_km']:.1f} km)", fontsize=8)
        fig.colorbar(pc, ax=fig.axes[:n], orientation="horizontal", shrink=0.5, pad=0.02,
                     label=f"{wname}-window mean IVT (kg m$^{{-1}}$ s$^{{-1}}$)")

        ax = fig.add_subplot(gs[1, :])
        for res, d in cases:
            ref = d["t0"][0]
            ts = bio.seconds(d["t0"], ref)
            ta, tb = (a - ref).total_seconds(), (b - ref).total_seconds()
            inward = np.array([bio.integrate_inst(ts, -d["qn"][:, k], ta, tb)
                               for k in range(d["qn"].shape[1])])          # kg m-1
            contrib = inward * 1e3 / d["box_area_m2"]                       # mm per km of boundary
            net = np.sum(inward * d["s_dl"]) / d["box_area_m2"]
            ax.plot(d["s_dist"], contrib, color=colors[res], lw=1.5,
                    label=f"{labels.get(res, res)}  (net {net:.0f} mm)")
        edges = bio.box_geometry(box)["face_edges"]
        for name, s0, s1 in edges:
            ax.axvline(s0, color="0.7", lw=0.6)
            ax.text(0.5 * (s0 + s1), 1.0, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlim(0, edges[-1][2])
        ax.set_xlabel("distance along box boundary from first vertex (km)")
        ax.set_ylabel("inward flux, mm per km of boundary")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{ev['label']}, box '{box['name']}': {wname} window {a:%Y-%m-%d %H} to {b:%Y-%m-%d %H} UTC",
                     fontsize=10)

        out_dir = bio.plot_dir(settings, ev["label"], box["name"])
        png = os.path.join(out_dir, "budget_map.png")
        fig.savefig(png, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"{ev['label']} [{box['name']}]: wrote {png}")
