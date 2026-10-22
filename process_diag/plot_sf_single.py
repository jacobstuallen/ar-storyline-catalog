"""
plot_sf_single.py  (block 2)

Time-mean structure function S2(r) of the model wind and its ERA5 nudging
target, one event, one resolution, one level. Reads the output of
compute_structure_functions.py.

Reference slopes: r^(2/3) matches a k^-5/3 spectrum, r^2 a k^-3 spectrum
(or a field that is smooth at that scale). Below ~100-150 km Target_U is
ERA5 interpolated to the model columns, so its small-r values reflect the
interpolation, not ERA5 dynamics.

Run:  python plot_sf_single.py
"""

import os

import numpy as np

import process_io as pio
import storyline_plot as sp
import matplotlib.pyplot as plt

EVENT_YEAR = pio.event_year(2019)     # run_process.py sets this per event
RESOLUTION = "30x16"
LEVEL_HPA = 500
FIELDS = ["U", "Target_U"]
STYLES = {"U": dict(color="black", ls="-"), "Target_U": dict(color="black", ls="--"),
          "V": dict(color="tab:blue", ls="-"), "Target_V": dict(color="tab:blue", ls="--")}
R_ANCHOR_KM = 100         # where the reference slopes touch the first field's curve

pcfg = pio.load_process_cfg()
cfg = pio.find_event(pcfg, EVENT_YEAR)
path = pio.sf_path(pcfg, cfg['event_label'], RESOLUTION)
if not os.path.isfile(path):
    print(f"SKIP: no {path}")
    raise SystemExit(0)
sf = pio.read_sf(path)
r = sf['r']
dx = float(sf['attrs']['grid_spacing_km'])

fig, ax = plt.subplots(figsize=(6, 5))
for name in FIELDS:
    s2, lev = pio.sf_mean(sf, name, LEVEL_HPA)
    ax.plot(*pio.finite_bins(r, s2), lw=1.8, marker='o', ms=3, label=name,
            **STYLES.get(name, {}))

s2_ref, lev = pio.sf_mean(sf, FIELDS[0], LEVEL_HPA)
r0, s0 = pio.anchor(r, s2_ref, R_ANCHOR_KM)
for p, lab in ((2 / 3, r"$r^{2/3}$ ($k^{-5/3}$)"), (2, r"$r^{2}$ ($k^{-3}$)")):
    ax.plot(r, s0 * (r / r0) ** p, color='0.6', lw=0.9, ls=':')
    ax.text(r[-1], s0 * (r[-1] / r0) ** p, lab, fontsize=8, color='0.4',
            va='center', ha='left')

ax.axvline(dx, color='0.5', lw=0.8)
ax.text(dx, 0.02, f" $\\Delta x$ = {dx:.1f} km", transform=ax.get_xaxis_transform(),
        fontsize=8, color='0.4')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Separation r (km)")
ax.set_ylabel(r"$S_2(r)$ (m$^2$ s$^{-2}$)")
ax.set_title(f"{cfg['event_label']}  {sf['attrs']['label']}  lev {lev:.0f} hPa, NORCAL",
             fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.legend(frameon=False)
fig.tight_layout()

sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', cfg['event_label'])}/"
                 f"sf_{RESOLUTION}_{'_'.join(FIELDS)}_{LEVEL_HPA}.png")
