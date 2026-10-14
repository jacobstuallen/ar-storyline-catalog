./case.setup

./xmlchange RUN_STARTDATE=2005-12-18

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./xmlchange STOP_N=21

./xmlchange REST_N=7

./case.submit

./xmlchange JOB_WALLCLOCK_TIME=08:00:00

./case.submit

./xmlchange REST_N=7

./xmlchange JOB_WALLCLOCK_TIME=12:00:00

./case.submit

./xmlchange NTASKS=-16,NTASKS_ESP=1,NTASKS_IAC=1

./xmlchange ATM_NCPL=432

./check_case

./check_case

./xmlchange LND_NCPL=432

./check_case

./check_case

./case.submit

