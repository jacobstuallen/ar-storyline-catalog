# sub_hourly_precip

Short-duration (20 min to 72 h) precipitation in the storyline runs: how it
changes with resolution (30x8, 30x16, 30x32) and how it compares with ASOS
1-minute gauges. Sits next to `storyline_diag/` and reads its event configs
(`cases`, `time`, `observed_comparison.storm_window`, `event_label`) and
`defaults.yaml` (`resolution_labels`). No storyline_diag Python is imported.
Plan and rationale: project doc `claude/short_duration_plan.md`.

## Run

```
python fetch_asos.py            # LOGIN NODE (needs network): data/asos/stations.csv, data/asos/EVENT/ID.csv
python fetch_ar_storms.py       # LOGIN NODE: AR3+ storms from the CW3E catalog + ASOS for each (see below)
python build_blocks_nc.py       # data/EVENT/blocks_RES.nc
python plot_depth_duration.py
python plot_asos_compare.py
python plot_hf_characteristics.py
```

Before `fetch_ar_storms.py`: download the CW3E event CSV for each coastal
point in `ar_catalog.files` (default 38.5N, the Sonoma coast) from
https://cw3e.ucsd.edu/Projects/ARCatalog/catalog.html ("Download here" under
AR Event Information) into `data/ar_catalog/`, set the file names, and check
the printed first rows against `ar_catalog.columns` (start/end column numbers
are a guess; CW3E documents column 6 as the AR scale). Cite CW3E and notify
J. Cordeira if used in the paper.

or, after the fetch, `mkdir -p slurm_logs && sbatch submit_sub_hourly.sbatch`.
All options are in `sub_hourly.yaml`. Build and fetch skip existing files
unless their `overwrite` is true.

## Outputs

| File | What |
| --- | --- |
| `plots/EVENT/depth_duration.png` | a) wettest land cell, b) area-weighted q0.99 of per-cell max depth, c) (b) relative to 30x8, all vs duration. Solid = native mesh, dashed = finer mesh aggregated onto 30x8 cells. |
| `plots/multi_event/depth_duration_ratio.png` | Panel (c) for every event, one panel per finer mesh. |
| `plots/EVENT/asos_compare.png` | Per station (wettest first): cumulative depth, depth-duration with the neighbourhood cell range, and the 30 wettest 20-min blocks, station vs nearest land cell. |

| `plots/multi_event/hf_contribution.png` | Share of the storm total falling at or above each rate, at 20- and 60-min blocks, evaluated on the gauge tip ladder (0.76 / 0.25 mm/h steps); top axis = depth per block. ASOS AR3+ climatology (median, IQR) vs each mesh (land cells <= 500 m, pooled over CW3E-matched storms). |
| `plots/multi_event/hf_metrics_NNmin.png` | Boxplots: share of total at >= 10 and >= 20 mm/h, share in the wettest 1 h and 3 h, fraction of blocks wet, peak block / mean wet block. |

Scripts print the numbers behind each figure, including a pooled median
model/station max-depth ratio by duration at the end of `plot_asos_compare.py`.

Figures: no suptitles, lettered panels, < 12 x 12 in, titles 10 pt, labels
and ticks 8 pt, legends 7 pt, png at 300 dpi.

## Definitions

- **Blocks.** Everything is cut into `block_minutes` (20) blocks from the
  window start, stamped at the block end. 20 min is the shortest interval all
  meshes share: h5 is written every 2 x dtime, i.e. 20 / 6.7 / 3.3 min.
  h5 records go to the block holding their time_bnds midpoint; a block not
  fully covered is NaN.
- **Window.** `window: storm` uses each config's `observed_comparison.storm_window`.
- **Max depth.** Largest running sum over d / 20 consecutive blocks.
- **On 30x8 cells.** Each fine cell is assigned to the nearest 30x8 cell centre
  and averaged by area (a Voronoi approximation to conservative remapping).
  Only 30x8 cells with >= 80% of their area covered are kept. The block series
  are aggregated first, then the maxima are taken.
- **ASOS.** IEM copy of NCEI DSI-6406. Precip in inches per minute, taken
  as falling in the minute ending at `valid`. A block needs >= 90% of its
  minutes reported (blocks are not rescaled). Running windows may contain up
  to 5% missing blocks, counted as zero, so station maxima are lower bounds.
- **Neighbourhood.** Land cells within `compare.radius_km` of the station and
  within `compare.elev_tol_m` of its elevation (model elevation PHIS/g).

- **High-frequency metrics.** Each compares a site with itself (share of its
  own storm total, its own wet blocks), so storm placement and timing don't
  enter. Wet block = >= 0.254 mm (one 0.01-in tip). 60-min blocks are
  fixed clock-aligned sums of three 20-min blocks; wettest-N-hour shares use
  running windows.
- **Model storm windows.** Same CW3E selection and padding as the gauges,
  applied to every AR storm fully inside each run (after 24 h spin-up), from
  `blocks_run_RES.nc` (whole run, written by the build step). An event with
  none falls back to its strongest overlapping catalog event (2019: AR2).
  Model blocks are rounded to whole 0.254 mm tips with the remainder carried
  forward, as in a tipping bucket.
- **AR3+ storms.** CW3E events with AR scale >= 3 at the listed coastal
  points, 2000 to Sep 2023, merged when they overlap. Station window = AR
  start - 6 h to AR end + 12 h. A station-storm counts if >= 25 mm and
  <= 10% of blocks missing.

## Limits

- 30x8 sub-hourly values are two model timesteps per block; treat their shape
  below ~1 h as numerical.
- dtime shrinks with grid spacing (600 / 200 / 100 s), so resolution and
  timestep effects are not separable here.
- Gauge = point, cell = 5-20 km area mean. Station above the cell range at
  short durations is expected even for a perfect model.
- ASOS stations are low-elevation airports: they test frontal/band intensity,
  not the windward-slope signal. No ASOS 1-minute data before 2000 (1964,
  1986 are skipped).
- ASOS heated tipping buckets undercatch at high rates and in wind.
- No ASOS 1-minute data for the 2024 storm: NCEI's archive is empty from
  midday 12 Nov to 24 Nov 2024 at every station checked.
- High-frequency metrics are still scale-dependent: area means are smoother
  than points, so a perfect model sits below the gauges on high-rate shares.
  The test is which mesh moves toward the ASOS distribution.
- Tape layout (h4, h5) is from the 2019 v2 runs; verify for other events.
- Tested on synthetic files with the EAM h5/h4 schema and IEM CSV format,
  not on NERSC output or real IEM downloads.
