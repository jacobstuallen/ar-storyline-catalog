"""
plot_precip_ratio_dx.py  (block 4, all events)

Fraction of reference variance captured, R(r) = S2n_product / S2n_reference,
against separation in multiples of each product's grid spacing. If the
meshes' curves collapse, each resolves precipitation variance to the same
number of grid spacings.

Per event: each time-mean S2 has empty interior bins filled (log-log
interpolation) and is normalized at NORMALIZE_KM, then the ratio is taken
within the event. Line = geometric mean across events, band = event range.
Thick lines: reference = finest mesh (all events). Thin dashed: reference
= Stage IV (events from 2002 on). One figure per mask group in process.yaml
(elevation bands, slope classes), one panel per mask and kind.
Reads the output of compute_precip_structure_functions.py.

Run:  python plot_precip_ratio_dx.py
"""

import os
import warnings
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

import process_io as pio
import storyline_plot as sp

NORMALIZE_KM = 100
X_UNITS = "dx_nominal"    # "dx_nominal" (14 / 7 / 3.5 km), "dx_pg2" (column spacing) or "km"
NOMINAL_DX_KM = {"30x8": 14.0, "30x16": 7.0, "30x32": 3.5}
# reference: (products, line style)
COMPARISONS = {
    "30x32":   (["30x8", "30x16"], dict(ls='-', lw=1.8, alpha_band=0.18)),
    "StageIV": (["30x8", "30x16", "30x32"], dict(ls='--', lw=1.1, alpha_band=0.0)),
}
REF_LABELS = {"30x32": "vs E3SM ~3.5 km", "StageIV": "vs Stage IV"}
KIND_TITLES = {"raw": "", "anom": "event mean removed, "}
NCOLS = 2
R_LINES = [1.0, 0.8, 0.5]
YLIM = (0.2, 1.6)
X_TICKS = {"dx_nominal": [1, 2, 3, 5, 7, 10, 20], "dx_pg2": [1, 2, 3, 5, 7, 10, 20],
           "km": [5, 10, 20, 50, 100]}

pcfg = pio.load_process_cfg()
for REGION, opt in pio.precip_regions(pcfg):
    print(f"\n##### region {REGION}: {opt['region']}")
    events = pio.load_events(pcfg)
    colors = {c['resolution']: sp.case_color(c) for c in events[0]['cases']}
    labels = {c['resolution']: c['label'] for c in events[0]['cases']}

    files = {}      # (event, product) -> psf
    for cfg in events:
        for p in set(NOMINAL_DX_KM) | set(COMPARISONS):
            path = pio.psf_path(pcfg, cfg['event_label'], p, REGION)
            if os.path.isfile(path):
                files[(cfg['event_label'], p)] = pio.read_psf(path)
    if not files:
        print(f"SKIP {REGION}: no precip structure-function files; run compute_precip_structure_functions.py.")
        continue

    first = next(iter(files.values()))
    kinds, r = first['kinds'], np.asarray(first['r'])


    def x_scale(prod):
        """km per x unit for this product."""
        if X_UNITS == "km":
            return 1.0
        if X_UNITS == "dx_nominal":
            return NOMINAL_DX_KM[prod]
        return float(np.median([f['attrs']['grid_spacing_km'] for (e, p), f in files.items() if p == prod]))


    def ratios(ref, prod, mask, kind):
        """(n_events, n_bins) R for every event with both files and the mask."""
        out = []
        for cfg in events:
            ev = cfg['event_label']
            if (ev, ref) not in files or (ev, prod) not in files:
                continue
            fp, fr = files[(ev, prod)], files[(ev, ref)]
            if mask not in fp['masks'] or mask not in fr['masks']:
                continue
            a = pio.filled_normalized(fp, mask, kind, NORMALIZE_KM)
            b = pio.filled_normalized(fr, mask, kind, NORMALIZE_KM)
            if a is not None and b is not None:
                out.append(a / b)
        return np.array(out)


    for group, masks in pio.mask_groups(opt, first['masks']).items():
        panels = [(m, k) for m in masks for k in kinds]
        fig, axes = pio.panel_grid(len(panels), NCOLS, sharex=True, sharey=True)
        for n, ((mask, kind), ax) in enumerate(zip(panels, axes)):
            for ref, (prods, st) in COMPARISONS.items():
                for prod in prods:
                    R = ratios(ref, prod, mask, kind)
                    if len(R) == 0:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        g = np.exp(np.nanmean(np.log(R), axis=0))
                        lo, hi = np.nanmin(R, axis=0), np.nanmax(R, axis=0)
                    ok = np.isfinite(g) & (r <= NORMALIZE_KM * 1.0001)
                    x = r[ok] / x_scale(prod)
                    ax.plot(x, g[ok], color=colors.get(prod, 'grey'), ls=st['ls'], lw=st['lw'],
                            marker='o', ms=2.5 if st['ls'] == '-' else 0,
                            label=f"{labels.get(prod, prod)} {REF_LABELS[ref]} (n={len(R)})")
                    if st['alpha_band'] and len(R) > 1:
                        ax.fill_between(x, lo[ok], hi[ok], color=colors.get(prod, 'grey'),
                                        alpha=st['alpha_band'], lw=0)
            for y in R_LINES:
                ax.axhline(y, color='0.6', lw=0.8, ls='-' if y == 1 else ':')
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(X_TICKS[X_UNITS]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(*YLIM)
            ax.grid(alpha=0.3, which='both')
            title = KIND_TITLES.get(kind, kind + ", ") + pio.mask_title(mask, opt)
            ax.set_title(f"{ascii_lowercase[n]}) {title[0].upper() + title[1:]}", loc='left', fontsize=10)
            if n >= len(panels) - NCOLS:
                ax.set_xlabel({"dx_nominal": r"r / $\Delta x$ (nominal: 14, 7, 3.5 km)",
                               "dx_pg2": r"r / $\Delta x_{pg2}$", "km": "Separation r (km)"}[X_UNITS])
            if n % NCOLS == 0:
                ax.set_ylabel(f"R = fraction of reference S2\n(each normalized at {NORMALIZE_KM} km)")
        axes[0].legend(frameon=False, fontsize=7.5, loc='lower right')
        fig.tight_layout()
        sp.save_png(fig, f"{pio.out_dir(pcfg, 'plot_dir', 'summary', REGION)}/"
                         f"precip_ratio_{group}_{X_UNITS}_norm{NORMALIZE_KM}.png")
