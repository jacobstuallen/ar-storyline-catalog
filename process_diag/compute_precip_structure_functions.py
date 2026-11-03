"""
compute_precip_structure_functions.py  (block 4, compute step)

S2(r) of precipitation depth per period (hourly by default) for every E3SM
mesh and hourly obs product, inside precip_structure_function.region.

  kinds:  raw   the field as is
          anom  each point's event-mean field removed (mean over the window)
  masks:  from process.yaml (all land, land below an elevation limit). All
          products use one reference elevation/land field, so a mask covers
          the same area for every product.

Native points only: E3SM columns and obs grid cells, no regridding. Points
missing any period are dropped, so raw and anom use the same pairs. One
netCDF per event and product:
output/precip_structure_functions/psf_<event_label>_<product>.nc,
S2 (mask, kind, time, bin). Always overwrites.

Run:  python compute_precip_structure_functions.py
"""

import numpy as np

import process_io as pio
import storyline_io as sio

KINDS = ["raw", "anom"]

pcfg = pio.load_process_cfg()
opt = pcfg['precip_structure_function']
bbox = opt['region']
A = opt['accum_hours']
edges = pio.r_edges(opt)
n_bins = len(edges) - 1
mask_names = list(opt['masks'])


def compute(lat, lon, vals, r_exact_km):
    """S2 (mask, kind, time, bin), npairs (mask, bin), grid spacing, n points."""
    rows = np.isfinite(vals).any(axis=1)                  # periods with data
    valid = np.isfinite(vals[rows]).all(axis=0)           # points with every period
    masks = pio.mask_points(lat, lon, surf, opt['masks'])
    use = valid & np.logical_or.reduce(list(masks.values()))
    lat, lon, vals = lat[use], lon[use], vals[:, use]
    masks = {k: m[use] for k, m in masks.items()}
    dx = pio.grid_spacing_km(lat, lon)
    print(f"    {use.sum()} points, median spacing {dx:.1f} km, "
          + ", ".join(f"{k}: {m.sum()}" for k, m in masks.items()))

    i, j, b = pio.build_pairs(lat, lon, edges, r_exact_km, opt['n_random_pairs'])
    fields = {"raw": vals, "anom": vals - np.nanmean(vals[rows], axis=0)}
    s2 = np.full((len(mask_names), len(KINDS), len(vals), n_bins), np.nan, dtype=np.float32)
    npairs = np.zeros((len(mask_names), n_bins))
    for m, name in enumerate(mask_names):
        for k, kind in enumerate(KINDS):
            f_all = np.where(masks[name], fields[kind], np.nan)
            for t in np.where(rows)[0]:
                s, cnt = pio.structure_function(f_all[t], i, j, b, n_bins)
                s[cnt < opt['min_pairs']] = np.nan
                s2[m, k, t] = s
                npairs[m] = cnt
    return s2, npairs, dx, int(use.sum())


for cfg in pio.load_events(pcfg):
    try:
        t0, t1 = pio.analysis_window(cfg, pcfg['window'])
    except KeyError as e:
        print(f"SKIP {e}")
        continue
    print(f"\n=== {cfg['event_label']}: {t0} to {t1}, {A}-hour periods")
    try:
        surf = pio.surface_fields(cfg, opt, t0, t1, bbox)
    except (KeyError, FileNotFoundError) as e:
        print(f"SKIP {cfg['event_label']}: {e}")
        continue
    ends = pio.period_ends(t0, t1, A)
    base = {'event_label': cfg['event_label'], 'region': str(bbox), 'window': f"{t0} to {t1}",
            'accum_hours': A, 'surface_resolution': opt['surface']['resolution'],
            'masks': str(opt['masks'])}

    products = []
    for case in cfg['cases']:
        products.append((case['resolution'], case['label'], 'e3sm', case))
    available = sio.obs_products(cfg, t0, t1)
    for name in opt['obs']:
        if name in available:
            products.append((name, sio.obs_label(name), 'obs', name))

    for product, label, source, ref in products:
        print(f"  {product}:")
        if source == 'e3sm':
            got = pio.read_e3sm_precip_periods(ref, opt['stream'], opt['variable'],
                                               t0, t1, bbox, A)
            if got is None:
                print(f"  SKIP {product}: no {opt['stream']} records")
                continue
            cols, vals = got
            lat, lon, r_exact = cols.lat, cols.lon, opt['r_exact_km']
            extra = {'case': ref['name'], 'stream': opt['stream']}
        else:
            o = opt['obs_options'].get(ref, {})
            try:
                lat, lon, vals = pio.read_obs_precip_periods(cfg, ref, t0, t1, bbox, A,
                                                             o.get('coarsen', 1))
            except (FileNotFoundError, ValueError) as e:
                print(f"  SKIP {product}: {e}")
                continue
            r_exact = o.get('r_exact_km', opt['r_exact_km'])
            extra = {'coarsen': o.get('coarsen', 1)}

        s2, npairs, dx, npts = compute(lat, lon, vals, r_exact)
        out = pio.psf_path(pcfg, cfg['event_label'], product)
        pio.write_psf(out, s2, npairs, edges, ends, mask_names, KINDS,
                      dict(base, product=product, label=label, source=source,
                           grid_spacing_km=dx, n_points=npts, **extra))
        print(f"  Written: {out}")
