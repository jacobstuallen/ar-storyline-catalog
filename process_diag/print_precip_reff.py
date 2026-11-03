"""
print_precip_reff.py  (block 4, all events)

Effective resolution and fraction of variance captured, from the
precipitation structure functions of every event in process.yaml.

For each event, product and reference (same event, same mask, same kind):
  1. Fill empty interior bins (log-log interpolation; separations near the
     grid scale are discrete, so products fill different bins), then
     normalize each time-mean S2 by its own value at NORMALIZE_KM. This removes amplitude
     (intensity) differences, leaving the scale dependence.
  2. R(r) = S2n_product / S2n_reference on the bins both have.
  3. r_eff = where R first drops below THRESHOLD, moving from NORMALIZE_KM
     toward smaller r (log-log interpolation between bins). If R stays
     above it down to the product's smallest bin, r_eff is reported as
     "< r_min" (an upper bound).
  4. Fraction captured at each scale in SCALES_KM = R there (log-log
     interpolation). NaN where either curve has no value.
r_eff is given in km, in pg2 column spacings (Δx from the files, median
nearest-neighbour distance) and, for the meshes, in nominal dynamics-grid
spacings (NOMINAL_DX_KM).

Prints the median and range across events for the first NORMALIZE_KM, and
writes every event, normalization, threshold, mask and kind to
output/precip_structure_functions/reff_<region>.csv.
Reads the output of compute_precip_structure_functions.py.

Run:  python print_precip_reff.py
"""

import csv
import os

import numpy as np

import process_io as pio

NORMALIZE_KM = [100, 50, 200]   # first = main table; the rest go to the CSV and the sensitivity table
THRESHOLDS = [0.5, 0.8]
SCALES_KM = [10, 20, 50]
# reference: products compared against it
COMPARISONS = {
    "30x32":   ["30x8", "30x16"],
    "StageIV": ["30x8", "30x16", "30x32", "AORC"],
}
NOMINAL_DX_KM = {"30x8": 14.0, "30x16": 7.0, "30x32": 3.5}

pcfg = pio.load_process_cfg()
opt = pcfg['precip_structure_function']
events = pio.load_events(pcfg)


def normalized(psf, mask, kind, r_km):
    """(r, normalized S2 with empty interior bins filled); None if empty."""
    s = pio.filled_normalized(psf, mask, kind, r_km)
    return None if s is None else (np.asarray(psf['r']), s)


def loglog_interp(x, xs, ys):
    """y at x by linear interpolation in log-log space; NaN outside xs."""
    if len(xs) < 2 or x < xs.min() or x > xs.max():
        return np.nan
    o = np.argsort(xs)
    return float(np.exp(np.interp(np.log(x), np.log(xs[o]), np.log(ys[o]))))


def r_eff(r, R, r_norm, thr):
    """(r_eff km, is_upper_bound). Walks from r_norm toward small r."""
    sel = np.isfinite(R) & (R > 0) & (r <= r_norm * 1.0001)
    rr, RR = r[sel][::-1], R[sel][::-1]          # descending r
    if len(rr) == 0:
        return np.nan, False
    for k in range(len(rr)):
        if RR[k] < thr:
            if k == 0:
                return float(rr[0]), False
            # crossing between rr[k-1] (above) and rr[k] (below), log-log
            x0, x1 = np.log(rr[k - 1]), np.log(rr[k])
            y0, y1 = np.log(RR[k - 1]), np.log(RR[k])
            return float(np.exp(x0 + (np.log(thr) - y0) * (x1 - x0) / (y1 - y0))), False
    return float(rr[-1]), True


rows = []
for cfg in events:
    ev = cfg['event_label']
    files = {}
    for p in {p for v in COMPARISONS.values() for p in v} | set(COMPARISONS):
        path = pio.psf_path(pcfg, ev, p)
        if os.path.isfile(path):
            files[p] = pio.read_psf(path)
    for ref, prods in COMPARISONS.items():
        if ref not in files:
            continue
        fr = files[ref]
        for prod in prods:
            if prod not in files:
                continue
            fp = files[prod]
            if not np.allclose(fp['r'], fr['r']):
                print(f"  SKIP {ev} {prod} vs {ref}: different r bins")
                continue
            dx = float(fp['attrs']['grid_spacing_km'])
            for mask in fp['masks']:
                for kind in fp['kinds']:
                    for rn in NORMALIZE_KM:
                        a, b = normalized(fp, mask, kind, rn), normalized(fr, mask, kind, rn)
                        if a is None or b is None:
                            continue
                        r = a[0]
                        R = a[1] / b[1]
                        ok = np.isfinite(R) & (R > 0)
                        frac = {s: loglog_interp(s, r[ok], R[ok]) for s in SCALES_KM}
                        for thr in THRESHOLDS:
                            re, ub = r_eff(r, R, rn, thr)
                            rows.append(dict(
                                event=ev, reference=ref, product=prod, mask=mask, kind=kind,
                                normalize_km=rn, threshold=thr, r_eff_km=re, upper_bound=ub,
                                dx_pg2_km=dx, r_eff_dx_pg2=re / dx,
                                r_eff_dx_nominal=re / NOMINAL_DX_KM[prod] if prod in NOMINAL_DX_KM else np.nan,
                                **{f"frac_{s}km": frac[s] for s in SCALES_KM}))

if not rows:
    print("SKIP: no matching precip structure-function files; run compute_precip_structure_functions.py.")
    raise SystemExit(0)

region = "_".join(f"{v:g}" for v in opt['region'])
out = os.path.join(pio.out_dir(pcfg, 'data_dir', 'precip_structure_functions'), f"reff_{region}.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"Written: {out}  ({len(rows)} rows)")


def summary(vals):
    """'median [min-max]' over finite values."""
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return "n/a"
    return f"{np.median(v):5.1f} [{v.min():.1f}-{v.max():.1f}]"


def fsummary(vals):
    """Same as summary, two decimals (for fractions)."""
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return "n/a"
    return f"{np.median(v):.2f} [{v.min():.2f}-{v.max():.2f}]"


def pick(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


main = NORMALIZE_KM[0]
for mask in list(dict.fromkeys(r['mask'] for r in rows)):
    for kind in list(dict.fromkeys(r['kind'] for r in rows)):
        print(f"\n=== {mask}, {kind}, normalized at {main} km "
              f"(median [min-max] across events)"
              f"\n    bN = in N events R never fell below the threshold, so r_eff there is the"
              f" product's smallest bin (an upper bound)")
        for ref, prods in COMPARISONS.items():
            hdr = f"  vs {ref}:"
            print(f"{hdr:14} {'n':>2}  " + "  ".join(
                f"{'r_eff(R<' + str(t) + ') km':>20} {'/dx_pg2':>16} {'/dx_nom':>16}" for t in THRESHOLDS))
            for prod in prods:
                sel = pick(reference=ref, product=prod, mask=mask, kind=kind, normalize_km=main)
                if not sel:
                    continue
                cells = []
                for t in THRESHOLDS:
                    s = [r for r in sel if r['threshold'] == t]
                    nb = sum(r['upper_bound'] for r in s)
                    flag = f"b{nb}" if nb else "  "
                    cells.append(f"{flag}{summary([r['r_eff_km'] for r in s]):>18} "
                                 f"{summary([r['r_eff_dx_pg2'] for r in s]):>16} "
                                 f"{summary([r['r_eff_dx_nominal'] for r in s]):>16}")
                n = len({r['event'] for r in sel})
                print(f"    {prod:10} {n:>2}  " + "  ".join(cells))
            print(f"{'':14} {'':>2}  fraction of reference variance captured (R):")
            for prod in prods:
                sel = [r for r in pick(reference=ref, product=prod, mask=mask, kind=kind,
                                       normalize_km=main) if r['threshold'] == THRESHOLDS[0]]
                if sel:
                    print(f"    {prod:10} {'':>2}  " + "  ".join(
                        f"{s:>3} km: {fsummary([r[f'frac_{s}km'] for r in sel]):>17}"
                        for s in SCALES_KM))

# sensitivity of r_eff (first threshold) to the normalization scale, first mask and kind
mask0, kind0 = rows[0]['mask'], rows[0]['kind']
print(f"\n=== Sensitivity: median r_eff (km, R < {THRESHOLDS[0]}) vs normalization scale, {mask0}, {kind0}")
print(f"  {'reference':10} {'product':10} " + " ".join(f"{str(n) + ' km':>10}" for n in NORMALIZE_KM))
for ref, prods in COMPARISONS.items():
    for prod in prods:
        cells = []
        for rn in NORMALIZE_KM:
            v = [r['r_eff_km'] for r in pick(reference=ref, product=prod, mask=mask0, kind=kind0,
                                             normalize_km=rn, threshold=THRESHOLDS[0])]
            v = [x for x in v if np.isfinite(x)]
            cells.append(f"{np.median(v):10.1f}" if v else f"{'n/a':>10}")
        print(f"  {ref:10} {prod:10} " + " ".join(cells))
