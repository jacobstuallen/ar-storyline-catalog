"""
plot_budget_summary.py

Cross-event, cross-resolution summary of the drying ratio DR = P / F_in over
summary.window (budget.yaml). One figure per box:

  (a) F_in vs P, one point per event and resolution, lines of constant DR
  (b) DR by resolution, one line per event; bars = +/- |R| / F_in
  (c) supply vs efficiency: dln P = dln F_in + dln DR relative to
      summary.reference_resolution, in percent
  (d) budget closure: R / F_in by event and resolution

With two or more boxes, also a box comparison:
  (a) dln DR relative to the reference resolution, one bar per box
  (b, ...) for each box marked `inside: <outer box>` in budget.yaml, the share
      of the outer box's precipitation (mass) that falls in the inner box

Needs compute_budget.py first.

Output: plots/multi_event/<box>/budget_summary_<window>.png and .csv,
        plots/multi_event/budget_box_compare_<window>.png

Run:  python plot_budget_summary.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import budget_io as bio

settings = bio.load_settings()
order = settings["resolution_order"]
labels = settings["resolution_labels"]
colors = settings["resolution_colors"]
opt = settings["summary"]
WINDOW = opt["window"]
REF = opt["reference_resolution"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BOX_STYLES = [dict(color="0.45"), dict(color="white", edgecolor="0.3", hatch="////"),
              dict(color="0.8"), dict(color="white", edgecolor="0.3", hatch="....")]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
multi_dir = os.path.join(settings["plot_dir"], "multi_event")

# ---- collect window budgets for every box
rows = []
events_all = []
for cfg_name in settings["events"]:
    ev = bio.load_event(settings, cfg_name)
    if WINDOW not in ev["windows"]:
        print(f"{ev['label']}: no {WINDOW} window, skipping")
        continue
    a, b = ev["windows"][WINDOW]
    for box in settings["boxes"]:
        for res in order:
            path = bio.case_nc_path(settings, box["name"], ev["label"], res)
            if not os.path.exists(path):
                continue
            d = bio.read_case_nc(path)
            wb = bio.window_budget(d, a, b)
            if np.isnan(wb["P"]) or np.isnan(wb["F_in"]):
                print(f"{ev['label']} {res} [{box['name']}]: {WINDOW} window not covered")
                continue
            rows.append({"box": box["name"], "event": ev["label"], "resolution": res,
                         "start": a, "end": b, "box_area_km2": d["box_area_m2"] / 1e6,
                         **{k: wb[k] for k in ["P", "F_in", "F_out", "E", "dW", "R", "DR",
                                               "R_over_Fin"]}})
            if ev["label"] not in events_all:
                events_all.append(ev["label"])

if not rows:
    raise SystemExit("No budget results; run compute_budget.py first.")


def get(box, ev, res, key):
    for r in rows:
        if r["box"] == box and r["event"] == ev and r["resolution"] == res:
            return r[key]
    return np.nan


marker = {ev: MARKERS[i % len(MARKERS)] for i, ev in enumerate(events_all)}

for box in settings["boxes"]:
    bname = box["name"]
    brows = [r for r in rows if r["box"] == bname]
    if not brows:
        continue
    events = [e for e in events_all if any(r["event"] == e for r in brows)]
    out_dir = os.path.join(multi_dir, bname)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"budget_summary_{WINDOW}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys()))
        w.writeheader()
        w.writerows(brows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # (a) F_in vs P with constant-DR lines
    ax = axes[0, 0]
    fmax = 1.08 * max(r["F_in"] for r in brows)
    pmax = 1.15 * max(r["P"] for r in brows)
    for dr in opt["dr_lines"]:
        x = np.array([0, fmax])
        ax.plot(x, dr * x, color="0.82", lw=0.8, zorder=0)
        xe = min(fmax, pmax / dr)
        if xe > 0.15 * fmax:
            ax.text(0.97 * xe, dr * 0.97 * xe, f"DR {dr:g} ", color="0.5", fontsize=7,
                    ha="right", va="bottom")
    for ev in events:
        xs = [get(bname, ev, r, "F_in") for r in order]
        ys = [get(bname, ev, r, "P") for r in order]
        ax.plot(xs, ys, color="0.7", lw=0.8, zorder=1)
        for res, xv, yv in zip(order, xs, ys):
            if not np.isnan(xv):
                ax.scatter(xv, yv, marker=marker[ev], s=55, color=colors[res],
                           edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, pmax)
    ax.set_xlabel("vapour inflow F$_{in}$ (mm, box mean)")
    ax.set_ylabel("precipitation P (mm, box mean)")
    ax.set_title("(a) Supply and precipitation", loc="left")
    h_res = [plt.Line2D([], [], ls="", marker="o", color=colors[r], label=labels[r]) for r in order]
    h_ev = [plt.Line2D([], [], ls="", marker=marker[e], color="0.3", label=e) for e in events]
    ax.legend(handles=h_res + h_ev, frameon=False, fontsize=7, loc="upper left")

    # (b) DR by resolution
    ax = axes[0, 1]
    xpos = np.arange(len(order))
    for ev in events:
        dr = np.array([get(bname, ev, r, "DR") for r in order])
        err = np.abs([get(bname, ev, r, "R_over_Fin") for r in order])
        ax.plot(xpos, dr, color="0.55", lw=1, zorder=1)
        for k, res in enumerate(order):
            if not np.isnan(dr[k]):
                ax.errorbar(xpos[k], dr[k], yerr=err[k], color=colors[res], capsize=3, lw=1,
                            zorder=2)
                ax.scatter(xpos[k], dr[k], marker=marker[ev], s=55, color=colors[res],
                           edgecolor="white", linewidth=1.2, zorder=3)
        last = np.where(~np.isnan(dr))[0]
        if len(last):
            ax.text(xpos[last[-1]] + 0.08, dr[last[-1]], ev.split("_")[0], fontsize=7,
                    va="center", color="0.3")
    ax.set_xticks(xpos, [labels[r] for r in order])
    ax.set_xlim(-0.3, len(order) - 0.5)
    ax.set_ylabel("drying ratio DR = P / F$_{in}$")
    ax.set_title("(b) Drying ratio by resolution (bars: $\\pm$|R| / F$_{in}$)", loc="left")
    ax.grid(axis="y", color="0.92", lw=0.5)

    # (c) attribution relative to the reference mesh
    ax = axes[1, 0]
    groups = [(ev, res) for ev in events for res in order
              if res != REF and not np.isnan(get(bname, ev, res, "P"))
              and not np.isnan(get(bname, ev, REF, "P"))]
    if groups:
        x = np.arange(len(groups))
        sup = np.array([100 * np.log(get(bname, e, r, "F_in") / get(bname, e, REF, "F_in"))
                        for e, r in groups])
        eff = np.array([100 * np.log(get(bname, e, r, "DR") / get(bname, e, REF, "DR"))
                        for e, r in groups])
        ax.bar(x - 0.19, sup, 0.36, color="0.45", label="supply  $\\Delta$ln F$_{in}$")
        ax.bar(x + 0.19, eff, 0.36, color="white", edgecolor="0.3", hatch="////",
               label="efficiency  $\\Delta$ln DR")
        ax.scatter(x, sup + eff, marker="D", color="black", s=28, zorder=3,
                   label="total  $\\Delta$ln P")
        ax.set_xticks(x, [f"{e.split('_')[0]}\n{labels[r]}" for e, r in groups], fontsize=7)
        for k in range(1, len(groups)):
            if groups[k][0] != groups[k - 1][0]:
                ax.axvline(k - 0.5, color="0.85", lw=0.6)
        ax.legend(frameon=False, fontsize=7)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel(f"change relative to {labels.get(REF, REF)} (%, 100 $\\times$ $\\Delta$ln)")
    ax.set_title("(c) Supply vs efficiency contributions to the change in P", loc="left")

    # (d) closure
    ax = axes[1, 1]
    width = 0.8 / len(order)
    for k, res in enumerate(order):
        vals = [100 * get(bname, ev, res, "R_over_Fin") for ev in events]
        ax.bar(np.arange(len(events)) + (k - (len(order) - 1) / 2) * width, vals, width * 0.92,
               color=colors[res], label=labels[res])
    ax.set_xticks(np.arange(len(events)), [e.split("_")[0] for e in events])
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("R / F$_{in}$ (%)")
    ax.set_title("(d) Budget residual (condensate flux + sampling)", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="0.92", lw=0.5)

    fig.suptitle(f"Regional drying ratio, box '{bname}', {WINDOW} window", fontsize=10)
    png = os.path.join(out_dir, f"budget_summary_{WINDOW}.png")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    for r in brows:
        print(f"  {r['event']:12s} {r['resolution']:6s} P {r['P']:6.1f}  F_in {r['F_in']:7.1f}  "
              f"DR {r['DR']:.3f}  R/F_in {100 * r['R_over_Fin']:5.1f}%")

# ---- box comparison
boxes = [b["name"] for b in settings["boxes"] if any(r["box"] == b["name"] for r in rows)]
pairs = [(b["name"], b["inside"]) for b in settings["boxes"]
         if b.get("inside") in boxes and b["name"] in boxes]
if len(boxes) >= 2:
    fig, axes = plt.subplots(1, 1 + len(pairs), figsize=(7 + 5 * len(pairs), 4.8),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]

    # (a) dln DR relative to the reference mesh, one bar per box
    ax = axes[0]
    groups = [(ev, res) for ev in events_all for res in order if res != REF]
    x = np.arange(len(groups))
    width = 0.8 / len(boxes)
    for k, bname in enumerate(boxes):
        vals = [100 * np.log(get(bname, e, r, "DR") / get(bname, e, REF, "DR")) for e, r in groups]
        ax.bar(x + (k - (len(boxes) - 1) / 2) * width, vals, width * 0.92,
               label=bname, **BOX_STYLES[k % len(BOX_STYLES)])
    ax.set_xticks(x, [f"{e.split('_')[0]}\n{labels[r]}" for e, r in groups], fontsize=7)
    for k in range(1, len(groups)):
        if groups[k][0] != groups[k - 1][0]:
            ax.axvline(k - 0.5, color="0.85", lw=0.6)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel(f"$\\Delta$ln DR relative to {labels.get(REF, REF)} (%)")
    ax.set_title("(a) Change in drying ratio with resolution, by box", loc="left")
    ax.legend(frameon=False, fontsize=7)

    # (b, ...) share of the outer box's precipitation mass falling in the inner box
    width = 0.8 / len(order)
    for j, (inner, outer) in enumerate(pairs):
        ax = axes[1 + j]
        for k, res in enumerate(order):
            share = [100 * get(inner, e, res, "P") * get(inner, e, res, "box_area_km2") /
                     (get(outer, e, res, "P") * get(outer, e, res, "box_area_km2"))
                     for e in events_all]
            ax.bar(np.arange(len(events_all)) + (k - (len(order) - 1) / 2) * width, share,
                   width * 0.92, color=colors[res], label=labels[res])
        ax.set_xticks(np.arange(len(events_all)), [e.split("_")[0] for e in events_all])
        ax.set_ylabel(f"% of '{outer}' precipitation")
        ax.set_title(f"({chr(98 + j)}) Share of '{outer}' precipitation in '{inner}'",
                     loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", color="0.92", lw=0.5)

    fig.suptitle(f"Box comparison, {WINDOW} window", fontsize=10)
    png = os.path.join(multi_dir, f"budget_box_compare_{WINDOW}.png")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
