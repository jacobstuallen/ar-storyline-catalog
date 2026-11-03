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
python compute_precip_structure_functions.py   # block 4 compute: meshes + AORC/Stage IV
python run_process.py                     # every plot script for every event
python plot_nudge_profile.py              # block 1, one event: Nudge_U and U - Target_U profiles
python plot_sf_single.py                  # block 2, one event: U vs Target_U, one resolution
python plot_sf_resolutions.py             # block 3, one event: all resolutions
python plot_precip_sf.py                  # block 4, one event: precip S2, masks x raw/anomaly
python plot_precip_sf_summary.py          # block 4, all events: normalized S2, geometric mean + event range
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
cells are treated as points, no regridding. `raw` and `anom` (each point's
event-mean field removed). Masks from one reference mesh's PHIS and OCNFRAC
(`surface:`), so every product covers the same area. Points missing any period
are dropped. AORC at ~1 km has very many pairs: keep its `r_exact_km` small or
set `coarsen`.

Levels are the nearest model `lev` (reference pressure at PS = 1000 hPa), not
interpolated to pressure.

## Files
- `process.yaml` events, box, window, block options
- `process_io.py` record index, column reader, pair builder, S2, netCDF I/O
- `plot_nudge_profile.py`, `compute_structure_functions.py`, `plot_sf_single.py`, `plot_sf_resolutions.py`
- `compute_precip_structure_functions.py`, `plot_precip_sf.py`, `plot_precip_sf_summary.py` (runs once, not per event; output in `plots/summary/`)
- `run_process.py` runs the plot scripts for every event (SCRIPTS and EVENTS lists at the top)
- `submit_process.sbatch` compute + run_process.py
