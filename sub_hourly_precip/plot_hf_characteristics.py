"""High-frequency precipitation characteristics: E3SM meshes vs an ASOS
AR3+ climatology. Every metric compares a site with itself (share of its own
storm total, its own wet blocks), so storm placement and timing don't matter.

Observations: every station-storm from fetch_ar_storms.py with at least
climatology.min_storm_total_mm and at most climatology.max_missing_frac
missing 20-min blocks.
Model: the same CW3E storm selection and padding as the gauges, applied to
every AR storm lying fully inside each run (after hf.spinup_hours), read from
blocks_run_RES.nc. An event with no such storm falls back to its strongest
catalog event overlapping its storm_window. Land cells in hf.model_bbox at or
below hf.elev_max_m with at least the station storm-total threshold, pooled
over storms. Model blocks are rounded to whole gauge tips (hf.tip_mm), with
the remainder carried forward as in a tipping bucket.

Figures (plots/multi_event/):
  hf_contribution.png    share of the storm total falling at rate >= r, vs r,
                         one panel per hf.block_minutes, evaluated on the
                         gauge tip ladder (one 0.01-in tip per block = the
                         step width). Line = median, band = interquartile
                         range; top axis = depth per block.
  hf_metrics_NNmin.png   boxplots (5-95% whiskers) per metric and group.

A gauge is a point and a cell is an area mean, so even a perfect model sits
below the gauges on the high-rate metrics; the question is which mesh moves
toward them.

Run:  python plot_hf_characteristics.py
      (after build_blocks_nc.py, fetch_asos.py, fetch_ar_storms.py)
"""
import csv
import os
from datetime import timedelta

import numpy as np

import sh_common as sh

opts = sh.load_options()
B, _ = sh.check_durations(opts)
CL, HF, A = opts["climatology"], opts["hf"], opts["asos"]
BLOCKS = [int(x) for x in HF["block_minutes"]]
if any(b % B for b in BLOCKS):
    raise ValueError(f"hf.block_minutes {BLOCKS} must be multiples of block_minutes={B}")
RATES = [float(r) for r in HF["rates_mmhr"]]
TOPH = [float(h) for h in HF["top_hours"]]
TIP = HF.get("tip_mm") or sh.WET_MM
CURVE_MAX_MMHR = 40.0
# thresholds on the gauge ladder: k tips per block, as a rate
CURVE = {b: TIP * 60.0 / b * np.arange(1, int(CURVE_MAX_MMHR / (TIP * 60.0 / b)) + 1) for b in BLOCKS}
MINTOT = float(CL["min_storm_total_mm"])
plt = sh.style()

# ── observations ─────────────────────────────────────────────────────────
odir = sh.rel(CL["dir"])
spath = os.path.join(odir, "storms.csv")
if not os.path.isfile(spath):
    raise FileNotFoundError(f"{spath} missing; run fetch_ar_storms.py")
with open(spath) as f:
    storms = list(csv.DictReader(f))
obs = {b: [] for b in BLOCKS}       # list of (metrics dict, curve, storm row, sid)
n_seen = n_kept = 0
for s in storms:
    w0, w1 = sh.parse_utc(s["window_start"]), sh.parse_utc(s["window_end"])
    nb = int((w1 - w0).total_seconds() // (B * 60))
    for sid in CL["stations"]:
        p = os.path.join(odir, s["id"], f"{sid}.csv")
        if not os.path.isfile(p):
            continue
        n_seen += 1
        times, v = sh.read_asos_csv(p)
        if "precip" not in v:
            continue
        dep, _ = sh.asos_blocks(times, v["precip"], w0, nb, B, A["min_valid_frac"])
        miss = float(np.mean(~np.isfinite(dep)))
        if np.nansum(dep) < MINTOT or miss > CL["max_missing_frac"]:
            continue
        n_kept += 1
        for b in BLOCKS:
            d = sh.aggregate(dep, b // B)
            m = sh.hf_metrics(d, b, RATES, TOPH, HF["max_missing_frac"])
            obs[b].append(({k: float(x[0]) for k, x in m.items()},
                           sh.contribution_curve(d, b, CURVE[b])[:, 0], s, sid))
n_storms = len({o[2]["id"] for o in obs[BLOCKS[0]]})
print(f"ASOS: {n_kept} of {n_seen} station-storms kept (>= {MINTOT:g} mm, <= "
      f"{CL['max_missing_frac']:.0%} missing), from {n_storms} of {len(storms)} storms")
if not n_kept:
    raise SystemExit("no station-storm passes; nothing to compare")

# ── model ────────────────────────────────────────────────────────────────
storms_all = sh.catalog_storms(opts, years=None, verbose=False)
events_all = sh.catalog_events_all(opts)
AC = opts["ar_catalog"]
pre, post = timedelta(hours=AC["hours_before"]), timedelta(hours=AC["hours_after"])
spin = timedelta(hours=float(HF.get("spinup_hours", 0)))
mod = {}       # res -> {b: {"m": [metric dicts per storm], "c": [curves]}}
labels = {}
mb = HF["model_bbox"]
print("\nModel storms (CW3E windows inside each run):")
for cfg_name in opts["events"]:
    cfg = sh.load_event(opts, cfg_name)
    ev = sh.event_short(cfg["event_label"])
    built = sh.built_cases(opts, cfg, "run")
    if not built:
        print(f"  {ev}: no blocks_run files (rerun build_blocks_nc.py); skip")
        continue
    r0, r1 = sh.event_window(cfg)
    sel = [dict(st, fallback=False) for st in storms_all if st["w0"] >= r0 + spin and st["w1"] <= r1]
    if not sel:
        s0, s1 = sh.analysis_window(opts, cfg)
        over = [e for e in events_all if e["start"] < s1 and e["end"] > s0]
        if over:
            e = max(over, key=lambda e: (e["scale"], (e["end"] - e["start"])))
            w0, w1 = max(e["start"] - pre, r0 + spin), min(e["end"] + post, r1)
            sel = [dict(id=e["start"].strftime("%Y%m%d%H"), start=e["start"], end=e["end"],
                        scale=e["scale"], points={e["point"]}, w0=w0, w1=w1, fallback=True)]
    for st in sel:
        print(f"  {ev}: {st['id']} AR{st['scale']} {st['w0']:%Y-%m-%d %H} to {st['w1']:%Y-%m-%d %H} UTC"
              + ("  (fallback: no AR" + str(AC['min_scale']) + "+ storm in run)" if st["fallback"] else ""))
    for case, d in built:
        res = case["resolution"]
        labels[res] = sh.case_label(cfg, case)
        base = sh.land(d, opts) & (d["lat"] >= mb[0]) & (d["lat"] <= mb[2]) & \
            (d["lon"] >= mb[1]) & (d["lon"] <= mb[3])
        if "zsfc" in d:
            base &= d["zsfc"] <= HF["elev_max_m"]
        ends = d["block_end"]
        for st in sel:
            k = np.where((ends > st["w0"]) & (ends <= st["w1"]))[0]
            if k.size == 0:
                continue
            dep = d["depth"][k]
            m = base & (np.nansum(dep, axis=0) >= MINTOT)
            if not m.any():
                continue
            dep = dep[:, m]
            if HF.get("tip_mm"):
                dep = sh.tipify(dep, float(HF["tip_mm"]))
            for b in BLOCKS:
                dd = sh.aggregate(dep, b // B)
                mt = sh.hf_metrics(dd, b, RATES, TOPH, HF["max_missing_frac"])
                mt["event"] = np.full(m.sum(), f"{ev}/{st['id']}", dtype=object)
                slot = mod.setdefault(res, {}).setdefault(b, {"m": [], "c": []})
                slot["m"].append(mt)
                slot["c"].append(sh.contribution_curve(dd, b, CURVE[b]))
RES = sorted(mod, key=lambda r: int(r.split("x")[1]))
print(f"model cell-storms per mesh ({BLOCKS[0]}-min): "
      + ", ".join(f"{r}: {sum(x['total'].size for x in mod[r][BLOCKS[0]]['m'])}" for r in RES))


def pooled(res, b, key):
    return np.concatenate([x[key] for x in mod[res][b]["m"]])


def obs_vals(b, key):
    return np.array([o[0][key] for o in obs[b]])


METRICS = [(f"share_ge_{RATES[2]:g}", f"Share of total at >= {RATES[2]:g} mm/h"),
           (f"share_ge_{RATES[3]:g}", f"Share of total at >= {RATES[3]:g} mm/h"),
           (f"top_{TOPH[0]:g}h", f"Share of total in wettest {TOPH[0]:g} h"),
           (f"top_{TOPH[1]:g}h", f"Share of total in wettest {TOPH[1]:g} h"),
           ("wet_frac", "Fraction of blocks wet"),
           ("peak_over_wetmean", "Peak block / mean wet block")]

# ── printed tables ───────────────────────────────────────────────────────
def q3(x):
    x = x[np.isfinite(x)]
    return (np.nan,) * 3 if x.size == 0 else tuple(np.percentile(x, [25, 50, 75]))


for b in BLOCKS:
    print(f"\n=== {b}-min blocks: median [IQR]")
    keys = [k for k, _ in METRICS] + [f"share_ge_{r:g}" for r in RATES if f"share_ge_{r:g}" not in dict(METRICS)]
    print(f"  {'metric':22s} {'ASOS':>20s}" + "".join(f"{r:>20s}" for r in RES))
    for k in keys:
        row = [q3(obs_vals(b, k))] + [q3(pooled(r, b, k)) for r in RES]
        print(f"  {k:22s}" + "".join(f"  {m:6.2f} [{lo:4.2f},{hi:4.2f}]" for lo, m, hi in row))
    print("  per-storm model medians, " + METRICS[0][0] + " / " + METRICS[2][0] + ":")
    for r in RES:
        parts = []
        for x in mod[r][b]["m"]:
            parts.append(f"{x['event'][0]} {np.nanmedian(x[METRICS[0][0]]):.2f}/{np.nanmedian(x[METRICS[2][0]]):.2f} (n={x['total'].size})")
        print(f"    {r:6s} " + "\n           ".join(parts))
    sim = [o for o in obs[b] if o[2]["simulated_event"]]
    if sim:
        print("  ASOS station-storms overlapping simulated events, " + METRICS[0][0] + " / " + METRICS[2][0] + ":")
        for o in sim:
            print(f"    {o[2]['simulated_event']:12s} {o[3]:4s} AR{o[2]['max_scale']} "
                  f"total {o[0]['total']:6.1f} mm  {o[0][METRICS[0][0]]:.2f} / {o[0][METRICS[2][0]]:.2f}")

# ── figures ──────────────────────────────────────────────────────────────
outdir = os.path.join(sh.rel(opts["plot_dir"]), "multi_event")
os.makedirs(outdir, exist_ok=True)
obs_label = f"ASOS AR{opts['ar_catalog']['min_scale']}+ ({n_kept} station-storms)"

fig, axes = plt.subplots(1, len(BLOCKS), figsize=sh.fig_size(4.3 * len(BLOCKS), 3.6),
                         constrained_layout=True, squeeze=False, sharey=True)
for i, b in enumerate(BLOCKS):
    ax = axes[0, i]
    x = CURVE[b]
    series = [(np.vstack([o[1] for o in obs[b]]).T, sh.INK, 2.0, obs_label, 0.14)]
    for r in RES:
        series.append((np.hstack(mod[r][b]["c"]), sh.RES_COLORS.get(r, sh.INK), 1.6, labels[r], 0.10))
    xmax = 0.0
    for arr, col, lw, lab, alpha in series:
        lo, md, hi = np.nanpercentile(arr, [25, 50, 75], axis=1)
        ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0, step="pre")
        ax.plot(x, md, color=col, lw=lw, label=lab, drawstyle="steps-pre")
        nz = np.where(hi > 0.005)[0]
        if nz.size:
            xmax = max(xmax, x[nz[-1]])
    step = TIP * 60.0 / b
    ax.set_xlim(0, np.ceil((xmax + 2 * step) / 5.0) * 5.0)
    ax.set_xlabel(f"Rate threshold, {b}-min blocks (mm/h); steps of {step:.2f} mm/h = one tip")
    top = ax.secondary_xaxis("top", functions=(lambda v, b=b: v * b / 60.0, lambda v, b=b: v * 60.0 / b))
    top.set_xlabel(f"Depth per {b}-min block (mm)")
    if i == 0:
        ax.set_ylabel("Share of storm total at or above rate")
    ax.set_ylim(0, 1)
    sh.panel_title(ax, i, f"{b}-min blocks")
axes[0, 0].legend(loc="upper right")
sh.save_png(fig, os.path.join(outdir, "hf_contribution.png"))
plt.close(fig)

for b in BLOCKS:
    fig, axes = plt.subplots(2, 3, figsize=sh.fig_size(10.5, 6.0), constrained_layout=True)
    groups = ["ASOS"] + RES
    cols = [sh.INK] + [sh.RES_COLORS.get(r, sh.INK) for r in RES]
    names = ["ASOS"] + [labels[r].replace("E3SM ", "") for r in RES]
    for i, (ax, (k, title)) in enumerate(zip(axes.flat, METRICS)):
        data = [obs_vals(b, k)] + [pooled(r, b, k) for r in RES]
        data = [x[np.isfinite(x)] for x in data]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color="white", lw=1.6))
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
        for j, col in enumerate(cols):
            for part in ("whiskers", "caps"):
                for ln in bp[part][2 * j: 2 * j + 2]:
                    ln.set_color(col)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(names)
        sh.panel_title(ax, i, title)
    sh.save_png(fig, os.path.join(outdir, f"hf_metrics_{b}min.png"))
    plt.close(fig)
