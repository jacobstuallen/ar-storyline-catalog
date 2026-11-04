"""E3SM vs ASOS 1-minute gauges, on common blocks.

Per event with ASOS data (plots/EVENT/asos_compare.png), one row per station
(wettest first, up to compare.max_stations):
  a) Cumulative depth: station and the nearest land cell of each mesh.
  b) Depth-duration: station max depth vs duration; nearest land cell (line)
     and the range over land cells within compare.radius_km and
     compare.elev_tol_m of the station elevation (shaded).
  c) The N wettest blocks, ranked: station vs nearest land cell.

A gauge is a point; a cell is a 5-20 km mean. Expect the station to sit at or
above the cell range at short durations even for a perfect model. The shaded
range shows how much of that is placement within the neighbourhood.

Prints a table per event and a pooled table of model / station max-depth
ratios (median over stations and events) at the end.

Run:  python plot_asos_compare.py   (after build_blocks_nc.py and fetch_asos.py)
"""
import os

import numpy as np

import sh_common as sh

opts = sh.load_options()
B, DURS = sh.check_durations(opts)
A = opts["asos"]
C = opts["compare"]
plt = sh.style()
import matplotlib.dates as mdates  # noqa: E402

D = np.array(DURS)
N_RANK = 30
stations = sh.asos_stations(opts)
pooled = {}   # res -> list of (ndur,) ratios nearest / station


def land_mask(d):
    lf = d.get("landfrac")
    return np.ones(d["lat"].size, bool) if lf is None else np.isfinite(lf) & (lf >= opts.get("min_landfrac", 0.5))


def neighbours(d, st):
    """(nearest land cell index, distance km, envelope indices)."""
    lm = land_mask(d)
    dist = sh.dist_km(st["lat"], st["lon"], d["lat"], d["lon"])
    cand = np.where(lm)[0] if lm.any() else np.arange(dist.size)
    i = cand[np.argmin(dist[cand])]
    env = lm & (dist <= C["radius_km"])
    if C.get("elev_tol_m") is not None and "zsfc" in d and np.isfinite(st["elev_m"]):
        env &= np.abs(d["zsfc"] - st["elev_m"]) <= C["elev_tol_m"]
    return i, float(dist[i]), np.where(env)[0]


dlab = sh.duration_label


for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    label = cfg["event_label"]
    t0, t1 = sh.analysis_window(opts, cfg)
    edir = os.path.join(sh.asos_dir(opts), label)
    if not os.path.isdir(edir):
        print(f"\n=== {label}: no ASOS data ({edir}); skip")
        continue
    print(f"\n=== {label}: {t0} to {t1} UTC")
    built = sh.built_cases(opts, cfg)
    nb = int(round((t1 - t0).total_seconds() / (B * 60)))
    ends = np.array([t0 + (i + 1) * (t1 - t0) / nb for i in range(nb)])

    obs = []
    for sid, st in stations.items():
        p = sh.asos_csv_path(opts, label, sid)
        if not os.path.isfile(p):
            continue
        times, v = sh.read_asos_csv(p)
        if "precip" not in v:
            print(f"  {sid}: no precip column")
            continue
        dep, nval = sh.asos_blocks(times, v["precip"], t0, nb, B, A["min_valid_frac"])
        tot = np.nansum(dep)
        miss = np.mean(~np.isfinite(dep))
        print(f"  {sid}: total {tot:.1f} mm, {miss:.0%} of blocks missing/incomplete")
        if tot < A["min_storm_total_mm"]:
            continue
        obs.append(dict(sid=sid, st=st, depth=dep, tot=tot, miss=miss,
                        md=sh.max_depths(dep[:, None], DURS, B, A.get("window_max_missing_frac", 0.0))[:, 0]))
    obs.sort(key=lambda o: -o["tot"])
    obs = obs[: int(C["max_stations"])]
    if not obs:
        print("  no station passes min_storm_total_mm; skip")
        continue

    n = len(obs)
    fig, axes = plt.subplots(n, 3, figsize=sh.fig_size(10.5, 2.3 * n + 0.4),
                             constrained_layout=True, squeeze=False)
    print(f"\n  {'stn':5s} {'elev':>5s} {'mesh':6s} {'dist':>5s} {'zcell':>6s} {'total':>6s}"
          + "".join(f"{dlab(x):>7s}" for x in DURS))
    for r, o in enumerate(obs):
        st = o["st"]
        ax0, ax1, ax2 = axes[r]
        ax0.plot(ends, np.nancumsum(o["depth"]), color=sh.INK, lw=1.8, label="ASOS")
        ax1.plot(D, o["md"], color=sh.INK, marker="s", ms=3.5, lw=1.8, label="ASOS")
        wet = np.sort(o["depth"][np.isfinite(o["depth"])])[::-1][:N_RANK]
        ax2.plot(np.arange(1, wet.size + 1), wet, color=sh.INK, marker="s", ms=2.5, lw=1.6, label="ASOS")
        print(f"  {o['sid']:5s} {st['elev_m']:5.0f} {'obs':6s} {'':>5s} {'':>6s} {o['tot']:6.1f}"
              + "".join(f"{x:7.1f}" for x in o["md"]))
        for case, d in built:
            res = case["resolution"]
            col = sh.case_color(case)
            lab = sh.case_label(cfg, case)
            i, dist, env = neighbours(d, st)
            dep = d["depth"][:, i]
            ax0.plot(d["block_end"], np.nancumsum(dep), color=col, lw=1.3, label=lab)
            ax1.plot(D, d["maxdepth"][:, i], color=col, marker="o", ms=3, lw=1.3, label=lab)
            if env.size > 1:
                lo = np.nanmin(d["maxdepth"][:, env], axis=1)
                hi = np.nanmax(d["maxdepth"][:, env], axis=1)
                ax1.fill_between(D, lo, hi, color=col, alpha=0.15, lw=0)
            mw = np.sort(dep[np.isfinite(dep)])[::-1][:N_RANK]
            ax2.plot(np.arange(1, mw.size + 1), mw, color=col, marker="o", ms=2, lw=1.2, label=lab)
            z = d["zsfc"][i] if "zsfc" in d else np.nan
            print(f"  {o['sid']:5s} {'':>5s} {res:6s} {dist:5.1f} {z:6.0f} {np.nansum(dep):6.1f}"
                  + "".join(f"{x:7.1f}" for x in d["maxdepth"][:, i])
                  + f"   env n={env.size}, max " + " ".join(
                      f"{x:.1f}" for x in (np.nanmax(d["maxdepth"][:, env], axis=1) if env.size else [])))
            with np.errstate(invalid="ignore", divide="ignore"):
                pooled.setdefault(res, []).append(d["maxdepth"][:, i] / o["md"])

        name = st["name"].title()[:24]
        sh.panel_title(ax0, 3 * r, f"{o['sid']} {name}, {st['elev_m']:.0f} m")
        sh.panel_title(ax1, 3 * r + 1, f"{o['sid']} depth-duration")
        sh.panel_title(ax2, 3 * r + 2, f"{o['sid']} wettest {B}-min blocks")
        ax0.set_ylabel("Cumulative (mm)")
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax0.xaxis.set_major_locator(loc)
        ax0.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        sh.duration_axis(ax1, DURS)
        ax1.set_yscale("log")
        ax1.set_ylabel("Max depth (mm)")
        ax2.set_xscale("log")
        ax2.set_xlabel("Rank")
        ax2.set_ylabel(f"{B}-min depth (mm)")
        if r < n - 1:
            ax1.set_xlabel("")
            ax2.set_xlabel("")
    h, l = axes[0, 1].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color=sh.MUTED, alpha=0.25, lw=0))
    l.append(f"cells within {C['radius_km']} km" + (f", +-{C['elev_tol_m']} m" if C.get("elev_tol_m") is not None else ""))
    axes[0, 0].legend(h, l, loc="lower right")
    sh.save_png(fig, sh.plot_path(opts, label, "asos_compare.png"))
    plt.close(fig)

if pooled:
    print("\n=== pooled: median (nearest land cell / station) max depth, all stations and events")
    print("  " + " " * 8 + "".join(f"{dlab(x):>7s}" for x in DURS) + "   n")
    for res in sorted(pooled, key=lambda r: int(r.split("x")[1])):
        a = np.array(pooled[res])
        print(f"  {res:8s}" + "".join(f"{x:7.2f}" for x in np.nanmedian(a, axis=0)) + f"   {a.shape[0]}")
