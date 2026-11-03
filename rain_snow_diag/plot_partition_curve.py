"""Snow fraction vs near-surface temperature, one line per resolution.

Columns: wet-bulb (Tw) and 2-m air temperature (TREFHT).
Top row: share of total precip falling in each temperature bin.
Bottom row: precip-weighted snow fraction per bin; T50 (50% crossing) printed
and marked. Uses every 3-hr land sample with PRECT >= min_rate_mmhr.
Output: plots/EVENT/partition_curve.png and, if pool_events,
plots/multi_event/partition_curve_all_events.png.
"""
import numpy as np

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
po = opts["plots"]["partition_curve"]
edges = np.arange(po["temp_min_c"], po["temp_max_c"] + 1e-9, po["bin_c"])
centers = 0.5 * (edges[:-1] + edges[1:])
nb = len(centers)
TEMPS = [("tw", "wet-bulb temperature (C)"), ("trefht", "2-m air temperature (C)")]


def accumulate(d):
    land = rc.land_mask(d, opts)
    if po.get("elev_range_m"):
        lo, hi = po["elev_range_m"]
        land &= (d["zsfc"] >= lo) & (d["zsfc"] <= hi)
    p = d["prect"][:, land]
    s = d["snow"][:, land]
    w = np.broadcast_to(d["area"][land], p.shape)
    out = {}
    for key, _ in TEMPS:
        t = d[key][:, land]
        ok = np.isfinite(p) & np.isfinite(t) & (p >= po["min_rate_mmhr"])
        ib = np.digitize(t[ok], edges) - 1
        inb = (ib >= 0) & (ib < nb)
        ib = ib[inb]
        pw = (p * w)[ok][inb]
        sw = (s * w)[ok][inb]
        out[key] = np.array([np.bincount(ib, pw, nb), np.bincount(ib, sw, nb),
                             np.bincount(ib, minlength=nb).astype(float)])
    return out


def t50(frac):
    for i in range(len(frac) - 1):
        a, b = frac[i], frac[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a >= 0.5 > b:
            return centers[i] + (a - 0.5) / (a - b) * (centers[i + 1] - centers[i])
    return np.nan


def draw(sums, title, path):
    """sums: {res: (label, color, {key: [P, S, N]})}"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    for res, (lab, col, acc) in sums.items():
        for j, (key, xlabel) in enumerate(TEMPS):
            P, S, N = acc[key]
            frac = np.where((N >= po["min_samples"]) & (P > 0), S / np.where(P > 0, P, 1), np.nan)
            axes[0, j].plot(centers, 100 * P / P.sum(), color=col, label=lab)
            axes[1, j].plot(centers, 100 * frac, color=col, marker="o", ms=3, label=lab)
            t = t50(frac)
            if np.isfinite(t):
                axes[1, j].axvline(t, color=col, lw=0.8, ls=":")
            print(f"  {lab:<16s} {key:<7s} T50 = {t:5.2f} C   samples {int(N.sum())}")
            axes[1, j].set_xlabel(xlabel)
    for j in range(2):
        axes[0, j].set_ylabel("% of precip")
        axes[1, j].set_ylabel("snow fraction (%)")
        axes[1, j].set_ylim(-3, 103)
        axes[1, j].axhline(50, color=rc.MUTED, lw=0.6)
        axes[1, j].axvline(0, color=rc.MUTED, lw=0.6)
    axes[1, 0].legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    rc.save_png(fig, path)
    plt.close(fig)


pooled = {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    label = cfg["event_label"]
    print(f"=== {label}")
    cases = rc.built_cases(opts, cfg)
    if not cases:
        continue
    sums = {}
    for case, d in cases:
        res = case["resolution"]
        acc = accumulate(d)
        sums[res] = (d["label"], rc.case_color(case), acc)
        if res not in pooled:
            pooled[res] = (d["label"], rc.case_color(case),
                           {k: np.zeros((3, nb)) for k, _ in TEMPS})
        for k, _ in TEMPS:
            pooled[res][2][k] += acc[k]
    draw(sums, f"{label}: snow fraction vs temperature, land, PRECT >= {po['min_rate_mmhr']} mm/hr",
         rc.plot_path(opts, label, "partition_curve.png"))

if po.get("pool_events") and pooled:
    print("=== all events pooled")
    order = {r: i for i, r in enumerate(rc.RES_COLORS)}
    pooled = dict(sorted(pooled.items(), key=lambda kv: order.get(kv[0], 99)))
    draw(pooled, "All events: snow fraction vs temperature (pooled samples)",
         rc.plot_path(opts, "multi_event", "partition_curve_all_events.png"))
