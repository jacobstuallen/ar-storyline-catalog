#!/bin/bash

set -e

# Created 2026-10-03 07:53:08

CASEDIR="/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cases/Storyline_2019_WUS_ne30_pg2_G-NDG_v2"

/pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cime/scripts/create_newcase --case "${CASEDIR}" --compset F2010-CICE --res ne30_g16 --mach pm-cpu --project m2637 --compiler=gnu

cd "${CASEDIR}"

./xmlchange NTASKS=-8,NTASKS_ESP=1,NTASKS_IAC=1

./xmlchange NTASKS=-8,NTASKS_ESP=1,NTASKS_IAC=1

./xmlchange CAM_CONFIG_OPTS="-mach pm-cpu -phys default -nlev 80 -clubb_sgs -microphys p3 -chem chemuci_linozv3_mam5_vbs -rain_evap_to_coarse_aero -vbs -bc_dep_to_snow_updates"

./xmlchange EPS_FRAC=1.0e-01

./xmlchange EPS_AAREA=1.0e-01

./xmlchange EPS_AMASK=1.0e-01

./xmlchange EPS_AGRID=1.0e-01

./xmlchange EPS_OAREA=1.0e-01

./xmlchange EPS_OMASK=1.0e-01

./xmlchange EPS_OGRID=1.0e-01

./xmlchange RUN_STARTDATE=2019-02-19

./xmlchange SSTICE_YEAR_START=1982,SSTICE_YEAR_END=2024,SSTICE_YEAR_ALIGN=1982

./xmlchange SSTICE_DATA_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/sst_reynolds_daily_180x360_1982_2024_timeseries.nc

./xmlchange SSTICE_GRID_FILENAME=/pscratch/sd/j/jsallen/betacast/grids/domains/domain.ocn.180x360.nc

