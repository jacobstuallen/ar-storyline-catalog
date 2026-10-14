./case.setup

./xmlchange RUN_STARTDATE=1986-02-04

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./case.submit

./xmlchange CAM_CONFIG_OPTS="-mach pm-cpu -phys default -nlev 80 -clubb_sgs -microphys p3 -chem chemuci_linozv3_mam5_vbs -rain_evap_to_coarse_aero -vbs -bc_dep_to_snow_updates"

./xmlchange EPS_FRAC=1.0e-01

./xmlchange EPS_AAREA=1.0e-01

./xmlchange EPS_AMASK=1.0e-01

./xmlchange EPS_AGRID=1.0e-01

./xmlchange EPS_OAREA=1.0e-01

./xmlchange EPS_OMASK=1.0e-01

./xmlchange EPS_OGRID=1.0e-01

./xmlchange RUN_STARTDATE=1986-02-04

./xmlchange REST_N=6

./check_case

./case.submit

