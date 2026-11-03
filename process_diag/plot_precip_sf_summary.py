"""
plot_precip_sf_summary.py  (block 4, all events)

Precipitation S2(r) summarized across every event in process.yaml. For
each product (mesh or obs), each event's time-mean S2 is divided by its
own value at NORMALIZE_KM, so storms of different intensity can be
combined. Line = geometric mean across events, band = event min to max.
One figure per mask group in process.yaml, one panel per mask and kind. n in the legend is the
number of events with that product (AORC starts 1979, Stage IV 2002).
Reads the output of compute_precip_structure_functions.py.

Run:  python plot_precip_sf_summary.py
"""

import os
import warnings

from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt

import process_io as pio
import storyline_plot as sp

NORMALIZE_KM = 100        # None = no normalization (geometric mean of the raw S2)
SHOW_BAND = True          # event min-max band
KIND_TITLES = {"raw": "", "anom": "Event mean removed, "}
NCOLS = 2

pcfg = pio.load_process_cfg()
opt = pcfg['precip_structure_function']
events = pio.load_events(pcfg)

# products in plot order: meshes of the first event, then obs
products = []
for case in events[0]['cases']:
    products.append((case['resolution'], case['label'],
                     dict(color=sp.case_color(case), ls='-', lw=1.8)))
obs_styles = (events[0].get('obs', {}) or {}).get('styles', {})
for name in opt['obs']:
    st = obs_styles.get(name, {})
    products.append((name, None, dict(color=st.get('color', 'black'), ls=st.get('ls', '-'), lw=1.4)))

# psf[product] = list of files, one per event that has it
psf = {p: [] for p, _, _ in products}
for cfg in events:
    for p in psf:
        path = pio.psf_path(pcfg, cfg['event_label'], p)
        if os.path.isfile(path):
            psf[p].append(pio.read_psf(path))
for p, files in psf.items():
    print(f"  {p}: {len(files)} events")
if not any(psf.values()):
    print("SKIP: no precip structure-function files; run compute_precip_structure_functions.py.")
    raise SystemExit(0)

first = next(f[0] for f in psf.values() if f)
masks, kinds, r = first['masks'], first['kinds'], first['r']
A = int(first['attrs']['accum_hours'])


def event_curves(files, mask, kind):
    """(n_events, n_bins) time-mean S2, each normalized at NORMALIZE_KM."""
    out = []
    for f in files:
        s2 = pio.psf_mean(f, mask, kind)
        if NORMALIZE_KM:
            ok = np.isfinite(s2)
            if not ok.any():
                continue
            k = np.where(ok)[0][np.argmin(np.abs(r[ok] - NORMALIZE_KM))]
            s2 = s2 / s2[k]
        out.append(s2)
    return np.array(out)


for group, gmasks in pio.mask_groups(opt, masks).items():
    panels = [(m, k) for m in gmasks for k in kinds]
    fig, axes = pio.panel_grid(len(panels), NCOLS)
    for n, ((mask, kind), ax) in enumerate(zip(panels, axes)):
        for p, label, style in products:
            files = [f for f in psf[p] if mask in f['masks']]
            if not files:
                continue
            c = event_curves(files, mask, kind)
            if len(c) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                gmean = np.exp(np.nanmean(np.log(c), axis=0))
                lo, hi = np.nanmin(c, axis=0), np.nanmax(c, axis=0)
            ok = np.isfinite(gmean)
            name = label or files[0]['attrs']['label']
            ax.plot(r[ok], gmean[ok], marker='o', ms=2.5, label=f"{name} (n={len(c)})", **style)
            if SHOW_BAND and len(c) > 1:
                ax.fill_between(r[ok], lo[ok], hi[ok], color=style['color'], alpha=0.15, lw=0)
            dx = np.median([float(f['attrs']['grid_spacing_km']) for f in files])
            ax.axvline(dx, ymax=0.06, color=style['color'], lw=2)
        if NORMALIZE_KM:
            ax.axvline(NORMALIZE_KM, color='0.7', lw=0.8, ls=':')
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
    fig.tight_layout()
    name = f"precip_sf_summary_{group}" + (f"_norm{NORMALIZE_KM}" if NORMALIZE_KM else "")
    sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', 'summary')}/{name}.png")
