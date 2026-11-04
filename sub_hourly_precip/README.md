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
python build_blocks_nc.py       # data/EVENT/blocks_RES.nc
python plot_depth_duration.py
python plot_asos_compare.py
```

or, after the fetch, `mkdir -p slurm_logs && sbatch submit_sub_hourly.sbatch`.
All options are in `sub_hourly.yaml`. Build and fetch skip existing files
unless their `overwrite` is true.

## Outputs

| File | What |
| --- | --- |
| `plots/EVENT/depth_duration.png` | a) wettest land cell, b) area-weighted q0.99 of per-cell max depth, c) (b) relative to 30x8, all vs duration. Solid = native mesh, dashed = finer mesh aggregated onto 30x8 cells. |
| `plots/multi_event/depth_duration_ratio.png` | Panel (c) for every event, one panel per finer mesh. |
| `plots/EVENT/asos_compare.png` | Per station (wettest first): cumulative depth, depth-duration with the neighbourhood cell range, and the 30 wettest 20-min blocks, station vs nearest land cell. |

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
- Tape layout (h4, h5) is from the 2019 v2 runs; verify for other events.
- Tested on synthetic files with the EAM h5/h4 schema and IEM CSV format,
  not on NERSC output or real IEM downloads.
