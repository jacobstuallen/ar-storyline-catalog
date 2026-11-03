# Process diagnostics

Cross-resolution science comparisons. Case paths, dates and storm windows come
from the `../storyline_diag/config.*.yaml` files; shared settings are in
`process.yaml`.

## Run (from process_diag/)
Everything, all events (compute, then every plot):
```
mkdir -p slurm_logs && sbatch submit_process.sbatch
```
Pieces by hand:
```
python compute_structure_functions.py     # all events and resolutions in process.yaml
python plot_precip_masks.py               # block 4 check first: elevation/slope maps and mask classes
python compute_precip_structure_functions.py   # block 4 compute: meshes + AORC/Stage IV
python run_process.py                     # every plot script for every event
python plot_nudge_profile.py              # block 1, one event: Nudge_U and U - Target_U profiles
python plot_sf_single.py                  # block 2, one event: U vs Target_U, one resolution
python plot_sf_resolutions.py             # block 3, one event: all resolutions
python plot_precip_sf.py                  # block 4, one event: precip S2, masks x raw/anomaly
python plot_precip_sf_summary.py          # block 4, all events: normalized S2, geometric mean + event range
python plot_precip_ratio_dx.py            # block 4, all events: R(r) vs r/dx, meshes vs 30x32 and Stage IV
python print_precip_reff.py               # block 4, all events: r_eff (km, dx) and variance fraction vs 30x32 and Stage IV
```
Run by hand, a plot script uses the default year at the top (`pio.event_year(2019)`);
`run_process.py` overrides it per event. Resolution and level are variables at the top
of each script. Scripts with nothing to plot print `SKIP` and exit cleanly.

Output: `plots/<event_label>/*.png` (300 dpi), `output/structure_functions/*.nc`,
`output/precip_structure_functions/*.nc`,
logs in `logs/<year>/` with a pass/skip/fail table at the end of the run.

## Method
Second-order structure function S2(r) = mean (f_i - f_j)^2 over column pairs at
separation r, on the native unstructured columns inside the box (no regridding).
k^-5/3 corresponds to r^(2/3), k^-3 to r^2. Pairs: all pairs below
`r_exact_km`, random pairs above. Bins with fewer than `min_pairs` are NaN.

Precipitation (block 4): S2 of depth per `accum_hours` period (hourly by default).
E3SM from the averaged h5 PRECT, records assigned to periods by interval midpoint
(20 / 6.7 / 3.3 min records for 30x8 / 30x16 / 30x32); obs hourly sums. Obs grid
cells are treated as points, no regridding. `raw` and optionally `anom` (each point's
event-mean field removed; `kinds:`). Masks are land plus elevation bands and terrain-slope
classes ([min, max) in `masks:`), from one reference mesh's PHIS and OCNFRAC (`surface:`),
so every product covers the same area. Slope = least-squares plane through the columns
within `slope_radius_km`. Plots make one figure per `mask_groups:` entry. The compute step
prints pair counts per mask near 50 and 100 km: narrow bands (high elevation, steep) can
fall under `min_pairs` at the coarsest mesh. Points missing any period
are dropped. AORC at ~1 km has very many pairs: keep its `r_exact_km` small or
set `coarsen`.

r_eff (print_precip_reff.py): each event's S2 is normalized at NORMALIZE_KM, then
R(r) = S2n_product / S2n_reference within the same event. Empty interior bins are filled by
log-log interpolation first (separations near the grid scale are discrete, so two products
often fill different bins); no extrapolation past a curve's own range. r_eff is where R first falls
below the threshold moving toward small r (log-log interpolation). If it never does, the
product's smallest bin is reported and flagged as an upper bound. The variance fraction
at a scale is R there.

Levels are the nearest model `lev` (reference pressure at PS = 1000 hPa), not
interpolated to pressure.

## Files
- `process.yaml` events, box, window, block options
- `process_io.py` record index, column reader, pair builder, S2, netCDF I/O
- `plot_nudge_profile.py`, `compute_structure_functions.py`, `plot_sf_single.py`, `plot_sf_resolutions.py`
- `plot_precip_masks.py`, `compute_precip_structure_functions.py`, `plot_precip_sf.py`, `plot_precip_sf_summary.py` (runs once, not per event; output in `plots/summary/`), `plot_precip_ratio_dx.py`, `print_precip_reff.py` (table + `output/precip_structure_functions/reff_<region>.csv`)
- `run_process.py` runs the plot scripts for every event (SCRIPTS and EVENTS lists at the top)
- `submit_process.sbatch` compute + run_process.py
