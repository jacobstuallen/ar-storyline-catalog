#!/bin/bash

set -e

# Created 2026-10-03 07:53:21

CASEDIR="/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cases/Storyline_2019_WUS_ne30x8_pg2_G-NDG_v2"

/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cime/scripts/create_newcase --case "${CASEDIR}" --compset F2010-CICE --res SnowEater_WUS_30x8_pg2 --mach pm-cpu --project m2637 --compiler=gnu

cd "${CASEDIR}"

./xmlchange CAM_CONFIG_OPTS="-mach pm-cpu -phys default -nlev 80 -clubb_sgs -microphys p3 -chem chemuci_linozv3_mam5_vbs -rain_evap_to_coarse_aero -vbs -bc_dep_to_snow_updates"

./xmlchange NTASKS=-8,NTASKS_ESP=1,NTASKS_IAC=1

./xmlchange RUN_STARTDATE=2019-02-19

./xmlchange ATM_NCPL=144

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

./case.build

./case.setup

./case.setup

./case.build

./xmlchange STOP_OPTION=ndays

./xmlchange STOP_N=14

./xmlchange JOB_WALLCLOCK_TIME=8:00:00

./xmlchange REST_N

./xmlchange REST_N=7

./preview_namelists

./case.submit

./case.submit

./xmlchange EPS_AGRID=0.1,EPS_AAREA=0.1,EPS_AMASK=0.1

./case.submit

./case.submit

./xmlchange CONTINUE_RUN

./case.submit

