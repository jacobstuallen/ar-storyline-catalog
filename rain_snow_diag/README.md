# rain_snow_diag

Rain-snow transition in the storyline runs and how it changes with
resolution (30x8, 30x16, 30x32). Sits next to `storyline_diag/` and reads its
event configs (`cases`, `time`, `event_label`) and `defaults.yaml`
(`resolution_labels`). No storyline_diag Python is imported.

## Run

```
python build_phase_nc.py            # step 1, writes data/EVENT/phase_RES.nc
python plot_elevband_timeseries.py
python plot_partition_curve.py
python plot_snowline_maps.py
python plot_transect.py             # reads h2 directly for the chosen times
```

or `mkdir -p slurm_logs && sbatch submit_rain_snow.sbatch`. All options are in
`rain_snow.yaml`. Build skips existing files unless `build.overwrite: true`.

## Outputs (plots/EVENT/)

| File | What |
| --- | --- |
| `elevband_timeseries.png` | Row per resolution. Area-mean rain (solid) and snow (dashed) per elevation band; box-median freezing level (IQR) and snow line. |
| `partition_curve.png` | Snow fraction and precip share vs wet-bulb and 2-m T, line per resolution; T50 printed. Pooled version in `plots/multi_event/`. |
| `snowline_maps.png` | Rain fraction per AR pulse, 50% contour, snow line in titles. |
| `rain_on_snow_maps.png` | Rain on snowpack per pulse. |
| `transect_NAME_YYYYMMDDHH.png` | Height cross-sections: T with 0 C, along-transect wind, RH. |
| `transect_locator.png` | Transect lines over terrain. |

## Definitions

- Snow = PRECSL + PRECSC, rain = PRECT - snow (h1, 3-hr means), mm/hr.
- Wet bulb from TREFHT, QREFHT, PS (h3) with Stull (2011).
- Freezing level: height of the 0 C crossing in T/Z3 (h2, 6-hourly);
  `highest` crossing by default. Columns frozen to the ground get the surface
  elevation.
- Snow line: elevation where area-weighted snow fraction of precipitating land
  cells, binned by 100 m of model elevation, first reaches 50% going uphill.
- Elevation is model elevation (PHIS/g). Hypsometry differs by mesh; the
  elevband script prints cell count and area per band.
- Pulses: wet spells in land-mean precip of the first built case (same pulses
  for every resolution), or set them in `plots.snowline_maps.pulses`.

## Limits

- No 3D cloud liquid, rain, ice or OMEGA on any tape. Add CLDLIQ, RAINQM,
  CLDICE, OMEGA to fincl to put them on the transect.
- Tape layout (h0-h4) is from the 2019 v2 runs. If an event differs, change
  `streams:`.
- Tested on synthetic files with the EAM schema, not on NERSC output.
