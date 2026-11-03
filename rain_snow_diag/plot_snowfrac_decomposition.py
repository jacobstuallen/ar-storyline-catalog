"""Figure 2: why does event snow fraction change with resolution?

Event snow fraction F = sum_T share(T) * f(T), where share(T) is the fraction
of precip falling at temperature T and f(T) the snow fraction there. Against
the reference resolution (ref_resolution):

  dF = sum dshare * f_ref        temperature sampling (where/when it precipitates)
     + sum share_ref * df        phase at the same temperature (the curve)
     + sum dshare * df           interaction

Bins cover every sample (end bins catch the tails), so the terms sum exactly
to dF. Bins where only one run has precip borrow that run's f, which puts
them entirely in the sampling term.

Output: plots/multi_event/snowfrac_decomposition.png and a printed table.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rs_common as rc

plt = rc.style()
opts = rc.load_options()
so = opts["plots"]["summary"]
po = opts["plots"]["decomposition"]
key = po["temp_var"]
lo, hi, dw = po["wide_bins_c"]
edges = np.arange(lo, hi + 1e-9, dw)
ref = so["ref_resolution"]
TERMS = [("sampling", "temperature sampling", "#6250d6"),
         ("phase", "phase at same temperature", "#eda100"),
         ("interaction", "interaction", "#a3a29a")]


def share_frac(sums):
    P, S = sums[0], sums[1]
    share = P / P.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(P > 0, S / P, np.nan)
    return share, f


def decompose(a, b):
    """a = reference sums (3, nb), b = other. Returns dict of terms + F's."""
    sa, fa = share_frac(a)
    sb, fb = share_frac(b)
    fa = np.where(np.isfinite(fa), fa, np.nan_to_num(fb))
    fb = np.where(np.isfinite(fb), fb, fa)
    fa, fb = np.nan_to_num(fa), np.nan_to_num(fb)
    out = {"sampling": np.sum((sb - sa) * fa), "phase": np.sum(sa * (fb - fa)),
           "interaction": np.sum((sb - sa) * (fb - fa)),
           "F_ref": np.sum(sa * fa), "F": np.sum(sb * fb)}
    out["total"] = out["F"] - out["F_ref"]
    return out


sums, labels = {}, {}
for cfg_name in opts["events"]:
    cfg = rc.load_event(opts, cfg_name)
    ev = cfg["event_label"]
    print(f"=== {ev}")
    for case, d in rc.built_cases(opts, cfg):
        res = case["resolution"]
        p, s, w, ok, day = rc.samples(d, opts, so["min_rate_mmhr"], so.get("elev_range_m"))
        sums.setdefault(ev, {})[res] = rc.binned(d[key], p, s, w, ok, day, edges, clip=True).sum(axis=0)
        labels.setdefault(res, cfg.get("resolution_labels", {}).get(res, d["label"]))

others = [r for r in rc.RES_COLORS if r in labels and r != ref]
events = [e for e in sums if ref in sums[e]]
if not events or not others:
    raise SystemExit(f"need {ref} and at least one other resolution in some event")
skipped = [e for e in sums if ref not in sums[e]]
if skipped:
    print(f"  no {ref} case, left out: {skipped}")

names = [rc.event_short(e) for e in events] + ["All events"]
fig, axes = plt.subplots(1, len(others), figsize=(1.2 + 1.3 * len(names) * len(others), 4.2),
                         sharey=True, squeeze=False)
print(f"\n{'event':<10s} {'res':<7s} {'F_ref':>6s} {'F':>6s} {'dF':>6s} {'sampl':>6s} {'phase':>6s} {'inter':>6s}  (percentage points)")
for ax, r in zip(axes[0], others):
    x = np.arange(len(names))
    for i, name in enumerate(names):
        if name == "All events":
            ev_list = [e for e in events if r in sums[e]]
            if not ev_list:
                continue
            a = sum(sums[e][ref] for e in ev_list)
            b = sum(sums[e][r] for e in ev_list)
        else:
            e = events[i]
            if r not in sums[e]:
                continue
            a, b = sums[e][ref], sums[e][r]
        dc = decompose(a, b)
        print(f"{name:<10s} {r:<7s} {100 * dc['F_ref']:6.1f} {100 * dc['F']:6.1f} {100 * dc['total']:6.1f} "
              f"{100 * dc['sampling']:6.1f} {100 * dc['phase']:6.1f} {100 * dc['interaction']:6.1f}")
        width = 0.24
        for k, (t, _, c) in enumerate(TERMS):
            ax.bar(x[i] + (k - 1) * width, 100 * dc[t], width * 0.9, color=c)
        ax.plot(x[i], 100 * dc["total"], "D", color=rc.INK, ms=6, mec="white", zorder=4)
    ax.axhline(0, color=rc.MUTED, lw=0.8)
    ax.axvline(len(names) - 1.5, color=rc.GRID, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(f"{labels[r]} minus {labels[ref]}", loc="left")
    ax.grid(axis="x", visible=False)
axes[0, 0].set_ylabel("change in event snow fraction (percentage points)")
handles = [Patch(color=c) for _, _, c in TERMS] + [
    Line2D([], [], marker="D", color=rc.INK, lw=0, ms=6, mec="white")]
fig.legend(handles, [n for _, n, _ in TERMS] + ["total change"], loc="upper left",
           ncol=4, bbox_to_anchor=(0.01, 1.02))
fig.suptitle(f"Snow-fraction change vs {labels[ref]}, binned by {'wet-bulb' if key == 'tw' else '2-m'} "
             f"temperature; land, PRECT >= {so['min_rate_mmhr']} mm/hr", x=0.01, ha="left", y=1.09)
fig.tight_layout()
rc.save_png(fig, rc.plot_path(opts, "multi_event", f"snowfrac_decomposition_{key}.png"))
