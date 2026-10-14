./case.setup

./xmlchange RUN_STARTDATE=2024-11-14

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./case.submit

./xmlchange --append --subgroup case.run BATCH_COMMAND_FLAGS=" --exclude=nid006154"

./xmlchange CONTINUE_RUN=FALSE

./case.submit

