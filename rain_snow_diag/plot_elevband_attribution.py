"""Is the extra snow at finer resolution a terrain effect?

Same decomposition as plot_snowfrac_decomposition.py, but binned by model
surface elevation instead of temperature:

  dF = sum dshare(z) * f_ref(z)     elevation sampling (precip falls higher up)
     + sum share_ref(z) * df(z)     phase at the same elevation (colder/warmer air there)
     + interaction

Terrain explains the change if the elevation-sampling term carries dF and
precip-weighted wet bulb at a given elevation matches across meshes.

a) share of precip by elevation band
b) area-mean precip depth by band (orographic enhancement vs more area)
c) precip-weighted wet bulb by band (thermodynamics at fixed elevation)
d) elevation decomposition of dF vs ref_resolution, per event and pooled

Profiles (a-c) pool all events. Land, PRECT >= summary.min_rate_mmhr.
Output: plots/multi_event/elevband_attribution.png
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["elev_attribution"]
z0, z1, dz = po["bins_m"]
edges = np.arange(z0, z1 + 1e-9, dz)
centers = 0.5 * (edges[:-1] + edges[1:])
nb = len(centers)
ref = so["ref_resolution"]


def band_sums(d):
    """Per elevation band (end bins catch everything): precip*area, snow*area,
    Tw*precip*area, land area, precip depth*area (mm*sr)."""
    land = rc.land_mask(d, opts)
    p, s, w, ok, _ = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
    ib = np.clip(np.digitize(d["zsfc"], edges) - 1, 0, nb - 1)
    ibt = np.broadcast_to(ib, p.shape)
    tw = np.where(np.isfinite(d["tw"]), d["tw"], 0.0)
    okt = ok & np.isfinite(d["tw"])
    out = np.zeros((5, nb))
    out[0] = np.bincount(ibt[ok], (p * w)[ok], nb)
    out[1] = np.bincount(ibt[ok], (s * w)[ok], nb)
    out[2] = np.bincount(ibt[okt], (tw * p * w)[okt], nb)
    out[3] = np.bincount(ib[land], d["area"][land], nb)
    out[4] = out[0] * 3.0                        # 3-hr means -> mm, area weighted
    out_tw_w = np.bincount(ibt[okt], (p * w)[okt], nb)
    return out, out_tw_w


def decompose(a, b):
    """a, b: (P, S) per band. Returns terms in fraction units."""
    sa, sb = a[0] / a[0].sum(), b[0] / b[0].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.where(a[0] > 0, a[1] / a[0], np.nan)
        fb = np.where(b[0] > 0, b[1] / b[0], np.nan)
    fa = np.nan_to_num(np.where(np.isfinite(fa), fa, fb))
    fb = np.nan_to_num(np.where(np.isfinite(fb), fb, fa))
    return {"sampling": np.sum((sb - sa) * fa), "phase": np.sum(sa * (fb - fa)),
            "interaction": np.sum((sb - sa) * (fb - fa)), "total": np.sum(sb * fb) - np.sum(sa * fa),
            "z_ref": np.sum(sa * centers), "z": np.sum(sb * centers)}


sums, tww, labels, colors = {}, {}, {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    ev = cfg["event_label"]
    print(f"=== {ev}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        sums.setdefault(ev, {})[res], tww.setdefault(ev, {})[res] = band_sums(d)
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))
        colors[res] = rc.case_color(case)
if not sums:
    raise SystemExit("no built events")

res_order = [r for r in rc.RES_COLORS if r in labels]
others = [r for r in res_order if r != ref]
events = [e for e in sums if ref in sums[e]]
pooled = {r: sum(sums[e][r] for e in sums if r in sums[e]) for r in res_order}
pooled_tww = {r: sum(tww[e][r] for e in tww if r in tww[e]) for r in res_order}

# ── printed tables ──
print(f"\nPooled by elevation band (land, PRECT >= {so['min_rate_mmhr']} mm/hr)")
print(f"{'band (m)':<12s}" + "".join(f"{r + ' area%':>12s}{r + ' P%':>10s}{r + ' Tw':>9s}" for r in res_order))
for k in range(nb):
    lab = f"{edges[k]:.0f}-{edges[k + 1]:.0f}" if k < nb - 1 else f">{edges[k]:.0f}"
    row = f"{lab:<12s}"
    for r in res_order:
        A, P = pooled[r][3], pooled[r][0]
        tw_k = pooled[r][2][k] / pooled_tww[r][k] if pooled_tww[r][k] > 0 else np.nan
        row += f"{100 * A[k] / A.sum():12.1f}{100 * P[k] / P.sum():10.1f}{tw_k:9.2f}"
    print(row)

print(f"\n{'event':<10s} {'res':<7s} {'z_P ref':>8s} {'z_P':>7s} {'dF':>6s} {'z-sampl':>8s} "
      f"{'z-phase':>8s} {'inter':>6s}  (m; percentage points)")
dec = {}
for name, blk in [(rc.event_short(e), sums[e]) for e in events] + [("All events", pooled)]:
    for r in others:
        if r not in blk:
            continue
        dc = decompose(blk[ref], blk[r])
        dec[(name, r)] = dc
        print(f"{name:<10s} {r:<7s} {dc['z_ref']:8.0f} {dc['z']:7.0f} {100 * dc['total']:6.1f} "
              f"{100 * dc['sampling']:8.1f} {100 * dc['phase']:8.1f} {100 * dc['interaction']:6.1f}")

# ── figure ──
fig = plt.figure(figsize=rc.fig_size(10, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.12)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1], sharey=axa)
axc = fig.add_subplot(gs[0, 2], sharey=axa)
axd = fig.add_subplot(gs[1, :])
y = centers / 1000
for r in res_order:
    P, A, D = pooled[r][0], pooled[r][3], pooled[r][4]
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(A > 0, D / A, np.nan)
        twb = np.where(pooled_tww[r] > 0, pooled[r][2] / pooled_tww[r], np.nan)
    kw = dict(color=colors[r], marker="o", ms=2.5, label=labels[r])
    axa.plot(100 * P / P.sum(), y, **kw)
    axb.plot(depth, y, **kw)
    axc.plot(twb, y, **kw)
axc.axvline(0, color=rc.MUTED, lw=0.6)
axa.set_xlabel("% of precip")
axb.set_xlabel("precip depth, all events summed (mm)")
axc.set_xlabel("precip-weighted wet bulb (C)")
axa.set_ylabel("surface elevation (km)")
plt.setp(axb.get_yticklabels(), visible=False)
plt.setp(axc.get_yticklabels(), visible=False)
rc.panel_title(axa, 0, "precip share by elevation")
rc.panel_title(axb, 1, "precip depth by elevation")
rc.panel_title(axc, 2, "wet bulb where it precipitates")
axa.legend(loc="upper right")

names = [rc.event_short(e) for e in events] + ["All events"]
x = np.arange(len(names))
nw = len(others) * 2
width = 0.8 / nw
for i, name in enumerate(names):
    for j, r in enumerate(others):
        if (name, r) not in dec:
            continue
        dc = dec[(name, r)]
        xs = x[i] - 0.4 + width * (2 * j + 0.5)
        axd.bar(xs, 100 * dc["sampling"], width * 0.92, color=colors[r])
        axd.bar(xs + width, 100 * dc["phase"], width * 0.92, color=colors[r], alpha=0.35,
                hatch="///", edgecolor=colors[r], lw=0)
        axd.plot(xs + width / 2, 100 * dc["total"], "D", color=rc.INK, ms=5, mec="white", zorder=4)
axd.axhline(0, color=rc.MUTED, lw=0.8)
axd.axvline(len(names) - 1.5, color=rc.GRID, lw=1.2)
axd.set_xticks(x)
axd.set_xticklabels(names)
axd.grid(axis="x", visible=False)
axd.set_ylabel("change in snow fraction (percentage points)")
rc.panel_title(axd, 3, f"snow-fraction change vs {labels[ref]}, split by elevation")
handles, hl = [], []
for r in others:
    handles += [Patch(color=colors[r]), Patch(facecolor=colors[r], alpha=0.35, hatch="///",
                                               edgecolor=colors[r], lw=0)]
    hl += [f"{labels[r]}: elevation sampling", f"{labels[r]}: phase at same elevation"]
handles.append(Line2D([], [], marker="D", color=rc.INK, lw=0, ms=5, mec="white"))
hl.append("total change")
axd.legend(handles, hl, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7)
rc.save_png(fig, rc.plot_path(opts, "multi_event", "elevband_attribution.png"))
