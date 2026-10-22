"""
plot_sf_resolutions.py  (block 3)

Time-mean S2(r) for every resolution of one event at one level. Solid =
model field, dashed = its ERA5 nudging target. Ticks on the x axis mark
each mesh's median column spacing. Reads the output of
compute_structure_functions.py.

Run:  python plot_sf_resolutions.py
"""

import os

import numpy as np

import process_io as pio
import storyline_plot as sp
import matplotlib.pyplot as plt

EVENT_YEAR = pio.event_year(2019)     # run_process.py sets this per event
LEVEL_HPA = 500
FIELD = "U"
TARGET = "Target_U"       # None to leave the target out
R_ANCHOR_KM = 100

pcfg = pio.load_process_cfg()
cfg = pio.find_event(pcfg, EVENT_YEAR)

fig, ax = plt.subplots(figsize=(6.5, 5))
ref = None
for case in cfg['cases']:
    path = pio.sf_path(pcfg, cfg['event_label'], case['resolution'])
    if not os.path.isfile(path):
        print(f"  SKIP {case['resolution']}: no {path}")
        continue
    sf = pio.read_sf(path)
    r, color = sf['r'], sp.case_color(case)
    s2, lev = pio.sf_mean(sf, FIELD, LEVEL_HPA)
    ax.plot(*pio.finite_bins(r, s2), color=color, lw=1.8, marker='o', ms=3,
            label=f"{case['label']} {FIELD}")
    if TARGET:
        s2t, _ = pio.sf_mean(sf, TARGET, LEVEL_HPA)
        ax.plot(*pio.finite_bins(r, s2t), color=color, lw=1.2, ls='--', marker='o',
                ms=2, mfc='white', label=f"{case['label']} {TARGET}")
    dx = float(sf['attrs']['grid_spacing_km'])
    ax.axvline(dx, ymax=0.06, color=color, lw=2)
    if ref is None:
        ref = (r, s2)

if ref is None:
    print("SKIP: no structure-function files for this event; run compute_structure_functions.py.")
    raise SystemExit(0)

r, s2 = ref
r0, s0 = pio.anchor(r, s2, R_ANCHOR_KM)
for p, lab in ((2 / 3, r"$r^{2/3}$"), (2, r"$r^{2}$")):
    ax.plot(r, s0 * (r / r0) ** p, color='0.6', lw=0.9, ls=':')
    ax.text(r[-1], s0 * (r[-1] / r0) ** p, lab, fontsize=8, color='0.4', va='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Separation r (km)")
ax.set_ylabel(r"$S_2(r)$ (m$^2$ s$^{-2}$)")
ax.set_title(f"{cfg['event_label']}  {FIELD}  lev {lev:.0f} hPa, NORCAL", fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()

sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', cfg['event_label'])}/"
                 f"sf_resolutions_{FIELD}_{LEVEL_HPA}.png")
