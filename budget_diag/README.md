# budget_diag

Regional water budget and drying ratio (DR = P / F_in) for a coastal box,
per storyline event and resolution. Self-contained: no imports from
storyline_diag. Case paths, dates and windows come from the event configs
in this directory (`config.<year>.WUS.G-NDG.yaml`, copied from
storyline_diag). Box, streams and plot options are in `budget.yaml`.

## Setup (once, from the repo root)
```
tar xzf budget_diag.tar.gz
cp storyline_diag/config.*.WUS.G-NDG.yaml budget_diag/
```
The copies can drift from storyline_diag. If case paths or windows change
there, copy again (or point `event_config_dir` in budget.yaml back to
`../storyline_diag`).

## Run (from budget_diag/)
```
python compute_budget.py          # reads E3SM h0 + h1, writes output/budget/*.nc
python plot_budget_timeseries.py  # check: budget terms per event (and budget_totals.csv)
python plot_budget_map.py         # check: IVT maps + boundary inflow profile per event
python plot_budget_summary.py     # all events and resolutions in one figure (+ csv)
```
Every script loops over the events and over the boxes in `budget.yaml`.
Output (300 dpi):
- `output/budget/<box>/budget_<event>_<res>.nc`
- `plots/<event_label>/<box>/budget_timeseries.png`, `budget_map.png`, `budget_totals.csv`
- `plots/multi_event/<box>/budget_summary_<window>.png` and `.csv`
- `plots/multi_event/budget_box_compare_<window>.png` (two or more boxes)

## Boxes
- `offshore_to_crest`: ~150 km of ocean to the Coast Range crest, 38.2-39.6 N.
- `coast_to_crest`: Sonoma / southern Mendocino, ~15 km offshore to the crest
  (`inside: offshore_to_crest`).
- `norcal_coast`: Sonoma to Humboldt (38.2-41.5 N), ~15 km offshore to the
  crest, then the Mendocino/Humboldt county lines where there is no single crest.
- `sierra_west`: northern/central Sierra windward slope (37.0-39.9 N), from the
  Central Valley axis to the crest.
The box comparison shows the change in DR with resolution for every box, and
for each box marked `inside:` the share of the outer box's precipitation that
falls in it. All vertices are first guesses; check them on budget_map.png.

## Budget
P = F_in - F_out + E - dW + R (box means, mm), with
W = TMQ + TMCLDLIQ + TMCLDICE + TMRAINQM, E = QFLX, P = time-mean PRECT (h1).
F_in, F_out = line integral of max(-Q.n, 0), max(Q.n, 0), Q = (TUQ, TVQ).
R holds the condensate flux across the box edges, 3-hourly sampling error in
TUQ/TVQ/QFLX, interpolation error and model non-conservation.

## Method
- Boundary: samples every `box.sample_km` along the edges; TUQ/TVQ
  interpolated linearly (Delaunay) from native columns. Outward normals per sample.
- Area terms: native columns with centres inside the box, weighted by `area`.
- Window integrals: trapezoid for instantaneous h0 terms, exact for h1
  interval means; window edges between records by linear interpolation of
  the cumulative integral. Windows can change without rerunning compute.
- If h1 has no time-mean PRECT, h0 PRECT (instantaneous) is used and the
  output file records it (`p_source`).

## Files
- `budget.yaml`: events, box polygon and face names, streams, colors, summary options
- `budget_io.py`: config, file index, box geometry, interpolation, window integrals, netCDF I/O
- `compute_budget.py`, `plot_budget_timeseries.py`, `plot_budget_map.py`, `plot_budget_summary.py`

## Before trusting results
- All box vertices (coastlines, crests, valley axis) are from memory; check them on budget_map.png.
- Check the printed column spacing per box: a box reaching outside the refined region
  shows a larger spacing at 30x8.
- Tested on synthetic data with an analytic budget (closes to 0.0 mm), not yet on NERSC output.
