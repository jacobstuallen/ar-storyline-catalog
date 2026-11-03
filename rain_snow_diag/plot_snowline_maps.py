"""Rain-snow transition maps per AR pulse.

Pulses come from plots.snowline_maps.pulses, or are found automatically from
the land-mean precip rate of the first built case (wet spells above
auto_rate_mmhr, merged across short gaps), so every resolution uses the same
pulses.

Figure 1 (snowline_maps.png): rows = resolutions, columns = pulses. Rain
fraction of pulse precip (blue = snow, red = rain), 50% contour in black,
terrain contours in grey. Title gives the pulse snow line.
Figure 2 (rain_on_snow_maps.png): rain falling where snowpack SWE exceeds
rain_on_snow_swe_mm.
"""
from datetime import timedelta

import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
po = opts["plots"]["snowline_maps"]
sl = opts["snowline"]
box = opts["region_bbox"]
extent = [box[1], box[3], box[0], box[2]]


def auto_pulses(d):
    land = rc.land_mask(d, opts)
    w = d["area"][land] / d["area"][land].sum()
    r = np.nansum(d["prect"][:, land] * w, axis=1)
    wet = r >= po["auto_rate_mmhr"]
    t = d["time"]
    spells, start = [], None
    for i, flag in enumerate(wet):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(wet) - 1):
            end = i if flag else i - 1
            spells.append([max(t[0], t[start] - timedelta(hours=3)), t[end]])
            start = None
    merged = []
    for s in spells:
        if merged and (s[0] - merged[-1][1]).total_seconds() < po["auto_gap_hours"] * 3600:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    keep = [s for s in merged if (s[1] - s[0]).total_seconds() >= po["auto_min_hours"] * 3600]
    # Largest pulses by land-mean depth, kept in time order.
    depth = [r[(t > a) & (t <= b)].sum() * 3 for a, b in keep]
    top = sorted(np.argsort(depth)[::-1][: po["max_pulses"]])
    return [tuple(keep[i]) for i in top]


for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    label = cfg["event_label"]
    print(f"=== {label}")
    cases = rc.built_cases(opts, cfg)
    if not cases:
        continue
    manual = po.get("pulses", {}).get(label)
    pulses = [(rc.parse_utc(a), rc.parse_utc(b)) for a, b in manual] if manual else auto_pulses(cases[0][1])
    if not pulses:
        print("  no pulses found; lower auto_rate_mmhr or set pulses")
        continue
    for a, b in pulses:
        print(f"  pulse {a:%m-%d %H} to {b:%m-%d %H} UTC")

    n, m = len(cases), len(pulses)
    pw = min(3.0, 10.4 / m, 9.0 / n)           # panel width (in), keeps figure < 12 x 12
    size = rc.fig_size(pw * m + 1.2, (pw * 1.1 + 0.75) * n + 0.3)
    fig1 = plt.figure(figsize=size)
    ax1 = rc.map_axes(fig1, n, m, extent)
    fig2 = plt.figure(figsize=size)
    ax2 = rc.map_axes(fig2, n, m, extent)
    ros_all, store = [], []
    for i, (case, d) in enumerate(cases):
        land = rc.land_mask(d, opts)
        for j, (a, b) in enumerate(pulses):
            k = (d["time"] > a) & (d["time"] <= b)        # 3-hr means ending in (a, b]
            P = np.nansum(d["prect"][k], axis=0) * 3.0
            R = np.nansum(d["rain"][k], axis=0) * 3.0
            S = np.nansum(d["snow"][k], axis=0) * 3.0
            frac = np.where(P >= po["min_total_mm"], R / np.where(P > 0, P, 1), np.nan)
            ros = np.nansum(np.where(d["snowh"][k] >= po["rain_on_snow_swe_mm"], d["rain"][k], 0.0),
                            axis=0) * 3.0
            ros = np.where(np.isfinite(d["snowh"][k]).any(axis=0), ros, np.nan)
            ros_all.append(ros[land])
            snl = rc.snow_line(d["zsfc"][land], R[land], S[land], d["area"][land],
                               sl["bin_m"], po["min_total_mm"], sl["min_cells"], sl["frac"])[0]
            store.append((i, j, frac, ros, snl, d, case))
            print(f"  {case['resolution']} pulse {j + 1}: land-mean {np.nansum(P[land] * d['area'][land]) / d['area'][land].sum():.0f} mm, "
                  f"snow line {snl:.0f} m, max rain-on-snow {np.nanmax(ros[land]) if np.isfinite(ros[land]).any() else np.nan:.0f} mm")

    finite = np.concatenate(ros_all)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    ros_max = float(np.ceil(np.percentile(finite, 99.5) / 10) * 10) if finite.size else 10.0

    pcs = {}
    for i, j, frac, ros, snl, d, case in store:
        a, b = pulses[j]
        title = f"{rc.case_label(cfg, case)}\n{a:%m-%d %H}Z to {b:%m-%d %H}Z"
        letter = rc.letter(i * m + j)
        terrain = rc.triangulation(d["lon"], d["lat"])
        for key, ax, field, kw in [
            ("frac", ax1[i, j], frac, dict(cmap="RdBu_r", vmin=0, vmax=1)),
            ("ros", ax2[i, j], ros, dict(cmap="Blues", vmin=0, vmax=ros_max)),
        ]:
            tri = rc.triangulation(d["lon"], d["lat"])
            tri.set_mask(tri.mask | ~np.isfinite(field[tri.triangles]).all(axis=1))
            pcs[key] = ax.tripcolor(tri, np.nan_to_num(field), shading="flat", **kw)
            if key == "frac":
                tri_frac = tri
            ax.tricontour(terrain, d["zsfc"], levels=po["terrain_contours_m"], colors="#8a8a85",
                          linewidths=0.5)
            ax.set_title(f"{letter} {title}", loc="left")
        ok = np.isfinite(frac)
        if ok.sum() > 10 and np.nanmin(frac) < 0.5 < np.nanmax(frac):
            ax1[i, j].tricontour(tri_frac, np.nan_to_num(frac), levels=[0.5], colors=rc.INK,
                                 linewidths=1.0)
        if np.isfinite(snl):
            ax1[i, j].set_title(f"{letter} {title}\nsnow line {snl:.0f} m", loc="left")
    fig1.colorbar(pcs["frac"], ax=ax1, shrink=0.6,
                  label=f"rain fraction of pulse precip (black 50%; blank < {po['min_total_mm']} mm)")
    fig2.colorbar(pcs["ros"], ax=ax2, shrink=0.6,
                  label=f"rain on snowpack, SWE >= {po['rain_on_snow_swe_mm']} mm (mm)")
    rc.save_png(fig1, rc.plot_path(opts, label, "snowline_maps.png"))
    rc.save_png(fig2, rc.plot_path(opts, label, "rain_on_snow_maps.png"))
    plt.close(fig1)
    plt.close(fig2)
