./case.setup

./xmlchange RUN_STARTDATE=2005-12-18

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./xmlchange STOP_N=21

./xmlchange REST_N=7

./case.submit

./case.submit

./case.submit

./case.submit

./case.submit

./case.submit

./xmlchange REST_N=1

./case.submit

