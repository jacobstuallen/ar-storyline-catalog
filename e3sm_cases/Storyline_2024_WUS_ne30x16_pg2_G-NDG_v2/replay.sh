./case.setup

./xmlchange RUN_STARTDATE=2024-11-14

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./xmlchange STOP_N=14

./xmlchange REST_N=7

./xmlchange JOB_WALLCLOCK_TIME=08:00:00

./case.submit

./check_case

