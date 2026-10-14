#!/bin/bash

set -e

# Created 2026-10-03 07:53:30

CASEDIR="/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cases/Storyline_2019_WUS_ne30x16_pg2_G-NDG_v2"

/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cime/scripts/create_newcase --case "${CASEDIR}" --compset F2010-CICE --res SnowEater_WUS_30x16_pg2 --mach pm-cpu --project m2637 --compiler=gnu

cd "${CASEDIR}"

./xmlchange NTASKS=-16,NTASKS_ESP=1,NTASKS_IAC=1

./xmlchange ATM_NCPL=432

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange EPS_FRAC=1.0e-01

./xmlchange EPS_AAREA=1.0e-01

./xmlchange EPS_AMASK=1.0e-01

./xmlchange EPS_AGRID=1.0e-01

./xmlchange EPS_OAREA=1.0e-01

./xmlchange EPS_OMASK=1.0e-01

./xmlchange EPS_OGRID=1.0e-01

./xmlchange RUN_STARTDATE=2019-02-19

./xmlchange STOP_N=13

./xmlchange JOB_WALLCLOCK_TIME=12:00:00

./case.setup

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./case.build

./xmlchange REST_N=6

./case.submit

./xmlchange CAM_CONFIG_OPTS="-mach pm-cpu -phys default -nlev 80 -clubb_sgs -microphys p3 -chem chemuci_linozv3_mam5_vbs -rain_evap_to_coarse_aero -vbs -bc_dep_to_snow_updates"

./case.build --clean-all

./case.build

./check_case

./case.submit

