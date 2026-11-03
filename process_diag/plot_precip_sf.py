"""
plot_precip_sf.py  (block 4)

Time-mean S2(r) of precipitation for every mesh and obs product of one
event. One figure per mask group in process.yaml, one panel per mask
and kind. Ticks on the x axis mark each product's median
point spacing. Reads the output of compute_precip_structure_functions.py.

Run:  python plot_precip_sf.py
"""

import os

from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt

import process_io as pio
import storyline_plot as sp

EVENT_YEAR = pio.event_year(2019)     # run_process.py sets this per event
NORMALIZE_KM = None       # e.g. 150 to divide each curve by its value there (compares shape only)
KIND_TITLES = {"raw": "", "anom": "Event mean removed, "}
NCOLS = 2

pcfg = pio.load_process_cfg()
opt = pcfg['precip_structure_function']
cfg = pio.find_event(pcfg, EVENT_YEAR)
obs_styles = (cfg.get('obs', {}) or {}).get('styles', {})

lines = []      # (psf, label, style)
for case in cfg['cases']:
    path = pio.psf_path(pcfg, cfg['event_label'], case['resolution'])
    if os.path.isfile(path):
        lines.append((pio.read_psf(path), case['label'],
                      dict(color=sp.case_color(case), ls='-', lw=1.8)))
for name in opt['obs']:
    path = pio.psf_path(pcfg, cfg['event_label'], name)
    if os.path.isfile(path):
        st = obs_styles.get(name, {})
        psf = pio.read_psf(path)
        lines.append((psf, psf['attrs']['label'],
                      dict(color=st.get('color', 'black'), ls=st.get('ls', '-'), lw=1.4)))

if not lines:
    print("SKIP: no precip structure-function files for this event; "
          "run compute_precip_structure_functions.py.")
    raise SystemExit(0)

masks, kinds = lines[0][0]['masks'], lines[0][0]['kinds']
A = int(lines[0][0]['attrs']['accum_hours'])
for group, gmasks in pio.mask_groups(opt, masks).items():
    panels = [(m, k) for m in gmasks for k in kinds]
    fig, axes = pio.panel_grid(len(panels), NCOLS)
    for n, ((mask, kind), ax) in enumerate(zip(panels, axes)):
        for psf, label, style in lines:
            if mask not in psf['masks']:
                continue
            r, s2 = pio.finite_bins(psf['r'], pio.psf_mean(psf, mask, kind))
            if len(r) == 0:
                continue
            if NORMALIZE_KM:
                s2 = s2 / s2[int(np.argmin(np.abs(r - NORMALIZE_KM)))]
            ax.plot(r, s2, marker='o', ms=2.5, label=label, **style)
            ax.axvline(float(psf['attrs']['grid_spacing_km']), ymax=0.06,
                       color=style['color'], lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        title = KIND_TITLES.get(kind, kind + ", ") + pio.mask_title(mask, opt)
        ax.set_title(f"{ascii_lowercase[n]}) {title}", loc='left', fontsize=10)
        if n >= len(panels) - NCOLS:
            ax.set_xlabel("Separation r (km)")
        if n % NCOLS == 0:
            ax.set_ylabel(rf"$S_2 / S_2({NORMALIZE_KM}\,$km$)$" if NORMALIZE_KM else
                          rf"$S_2(r)$ (mm$^2$, {A}-h depth)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{cfg['event_label']}  precipitation S2, {opt['region']}", fontsize=10)
    fig.tight_layout()
    name = f"precip_sf_resolutions_{group}" + (f"_norm{NORMALIZE_KM}" if NORMALIZE_KM else "")
    sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', cfg['event_label'])}/{name}.png")
