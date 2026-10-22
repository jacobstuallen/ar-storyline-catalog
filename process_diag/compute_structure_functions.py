"""
compute_structure_functions.py  (blocks 2-3, compute step)

S2(r) on the native columns inside the analysis box, for every event,
resolution, field and level in process.yaml. One netCDF per event and
resolution: output/structure_functions/sf_<event_label>_<resolution>.nc,
S2 (field, level, time, bin). Always overwrites.

Run:  python compute_structure_functions.py     (or sbatch submit_process.sbatch)
"""

import numpy as np

import process_io as pio

pcfg = pio.load_process_cfg()
opt = pcfg['structure_function']
edges = pio.r_edges(opt)
n_bins = len(edges) - 1
fields = opt['fields']

for cfg in pio.load_events(pcfg):
    try:
        t0, t1 = pio.analysis_window(cfg, pcfg['window'])
    except KeyError as e:
        print(f"SKIP {e}")
        continue
    print(f"\n=== {cfg['event_label']}: {t0} to {t1}")

    for case in cfg['cases']:
        recs = pio.records(case, pcfg['stream'], t0, t1, opt['every_hours'])
        if not pio.check_coverage(recs, t0, t1, case['name']):
            continue
        cols = pio.Columns(recs[0][0], pcfg['region'])
        lev_idx = cols.level_index(opt['levels_hpa'])
        dx = pio.grid_spacing_km(cols.lat, cols.lon)
        print(f"  {case['resolution']}: {cols.n} columns, median spacing {dx:.1f} km, "
              f"{len(recs)} records, levels {[round(float(cols.lev[k]), 1) for k in lev_idx]} hPa")

        i, j, b = pio.build_pairs(cols.lat, cols.lon, edges, opt['r_exact_km'],
                                  opt['n_random_pairs'])
        npairs = np.bincount(b, minlength=n_bins).astype(float)
        sparse = npairs < opt['min_pairs']

        s2 = np.full((len(fields), len(lev_idx), len(recs), n_bins), np.nan, dtype=np.float32)
        for n, (path, rec, t) in enumerate(recs):
            for f, name in enumerate(fields):
                vals = cols.read(path, rec, name, lev_idx)          # (level, ncol)
                for m in range(len(lev_idx)):
                    s, _ = pio.structure_function(vals[m], i, j, b, n_bins)
                    s[sparse] = np.nan
                    s2[f, m, n] = s

        out = pio.sf_path(pcfg, cfg['event_label'], case['resolution'])
        pio.write_sf(out, s2, npairs, edges, [r[2] for r in recs], fields,
                     opt['levels_hpa'], cols.lev[lev_idx],
                     {'case': case['name'], 'resolution': case['resolution'],
                      'label': case['label'], 'event_label': cfg['event_label'],
                      'grid_spacing_km': dx, 'n_columns': cols.n,
                      'region': str(pcfg['region']), 'stream': pcfg['stream'],
                      'window': f"{t0} to {t1}"})
        print(f"  Written: {out}")
