# Storyline diagnostic plots

## Run
```
python prefetch.py      # once, on a login node: boundary zips, Atlas 14, Natural Earth
./submit.sh             # all 5 events (array job), then the multi-event plots
./submit.sh 1,4         # only EVENTS[1] and EVENTS[4]
```
ERA5 frame diagnostics (hours) stay separate: `sbatch submit_era5_diagnostics.sbatch`.

One plot by hand (default event is config.1986):
```
python plot_max_precip.py
STORYLINE_CONFIG=config.2024.WUS.G-NDG.yaml python plot_max_precip.py
python run_diagnostics.py          # everything, serially
```

Output: `plots/<event_label>/*.png` (300 dpi), multi-event in `plots/multi_event/`.
Logs: `logs/<event_label>/<script>.log`; the SLURM log ends with a pass/fail table.

## Where options live
| What | Where |
|---|---|
| Event cases, dates, storm/day windows | `config.<year>.WUS.G-NDG.yaml` |
| Data paths, labels, regions, locations, every plot's options | `defaults.yaml` (`data:`, `resolution_labels:`, `plots:`) |
| Per-event override of any plot option | same key under `plots:` in the event config |
| Which scripts/events run | lists at the top of `run_diagnostics.py` |
| Multi-event county timeseries | `county_timeseries.yaml` |

`vmax`-style options set to `null` are computed from the data.

## Files
- `storyline_config.py` config merge, `load_event()`, `event_window()`, `plot_opts()`
- `storyline_io.py` E3SM/AORC/Stage IV readers, depth integrals, case coverage checks
- `storyline_plot.py` style, `save_png`, map/polygon helpers, cached downloads
- `obs_compare.py` storm/day-window helpers for the two observed-comparison scripts
- `build_comparison_nc.py` then `plot_multicase_overlay.py`, `plot_accum_precip.py`,
  `plot_max_precip.py`, `plot_county_accum.py`, `plot_huc10_accum.py`,
  `plot_huc10_24hr.py`, `print_basin_totals.py`, `plot_idf_overlay.py`, `plot_ivt_ar.py`
- `plot_county_timeseries.py` (multi-event)

## Conventions
- Region bbox `[min_lat, min_lon, max_lat, max_lon]`; map extent `[min_lon, max_lon, min_lat, max_lat]`.
- E3SM PRECT is instantaneous: depths are trapezoidal integrals.
- AORC/Stage IV hours are stamped at the hour's end; window (t0, t1] (flip `OBS_STAMP_AT_END` in storyline_io.py if wrong).
- A case whose output doesn't cover the window is skipped with a printed reason. A script with no usable case exits with an error.
- No AORC before 1979: AORC panels show "n/a", E3SM still plotted.
