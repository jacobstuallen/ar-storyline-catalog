"""
plot_budget_timeseries.py

Check figure, one per event: box-mean budget terms as 3-hourly interval means
(mm/h) for each resolution. Panels: F_in, F_out, P, E, dW/dt (vapour +
condensate), residual R. Storm window shaded light, day window darker.
Also prints window totals and writes them to plots/<event_label>/<box>/budget_totals.csv.

Needs compute_budget.py first.

Output: plots/<event_label>/<box>/budget_timeseries.png

Run:  python plot_budget_timeseries.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import budget_io as bio

settings = bio.load_settings()
order = settings["resolution_order"]
labels = settings["resolution_labels"]
colors = settings["resolution_colors"]

PANELS = [("F_in", "Vapour inflow  F_in"), ("F_out", "Vapour outflow  F_out"),
          ("P", "Precipitation  P"), ("E", "Evaporation  E"),
          ("dW", "Storage change  dW/dt"), ("R", "Residual  R")]
TERMS = ["P", "F_in", "F_out", "E", "dW_vap", "dW_cond", "dW", "R", "DR", "R_over_Fin"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

for cfg_name in settings["events"]:
    ev = bio.load_event(settings, cfg_name)
    for box in settings["boxes"]:
        cases = []
        for res in order:
            path = bio.case_nc_path(settings, box["name"], ev["label"], res)
            if os.path.exists(path):
                cases.append((res, bio.read_case_nc(path)))
        if not cases:
            print(f"{ev['label']} [{box['name']}]: no budget files, skipping")
            continue

        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
        for (key, title), ax in zip(PANELS, axes.flat):
            for wname, alpha in [("storm", 0.10), ("day", 0.18)]:
                if wname in ev["windows"]:
                    ax.axvspan(*ev["windows"][wname], color="0.5", alpha=alpha, lw=0)
            for res, d in cases:
                r = bio.interval_rates(d)
                x = list(d["p_lo"]) + [d["p_hi"][-1]]
                y = np.append(r[key], r[key][-1])
                ax.step(x, y, where="post", color=colors[res], lw=1.5, label=labels.get(res, res))
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_ylabel("mm h$^{-1}$ (box mean)")
            ax.grid(axis="y", color="0.9", lw=0.5)
        axes[0, 0].legend(frameon=False, fontsize=8)
        for ax in axes[-1]:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.suptitle(f"{ev['label']}: water budget for box '{box['name']}' "
                     f"(shaded: storm window, darker: day window)", fontsize=10)

        out_dir = bio.plot_dir(settings, ev["label"], box["name"])
        png = os.path.join(out_dir, "budget_timeseries.png")
        fig.savefig(png, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"{ev['label']} [{box['name']}]: wrote {png}")

        # Window totals (mm, box mean)
        rows = []
        print(f"  {'window':6s} {'mesh':8s} " + " ".join(f"{t:>9s}" for t in TERMS))
        for wname, (a, b) in ev["windows"].items():
            for res, d in cases:
                wb = bio.window_budget(d, a, b)
                rows.append({"event": ev["label"], "window": wname, "resolution": res,
                             "start": a, "end": b, **{t: wb[t] for t in TERMS},
                             **{k: v for k, v in wb.items() if k.startswith("F_")},
                             "p_source": d["p_source"]})
                print(f"  {wname:6s} {res:8s} " + " ".join(f"{wb[t]:9.3f}" for t in TERMS))
        with open(os.path.join(out_dir, "budget_totals.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
