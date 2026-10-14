# This file is for user convenience only and is not used by the model
# Changes to this file will be ignored and overwritten
# Changes to the environment should be made in env_mach_specific.xml
# Run ./case.setup --reset to regenerate this file
. /opt/cray/pe/lmod/lmod/init/sh
module unload cray-hdf5-parallel cray-netcdf-hdf5parallel cray-parallel-netcdf cray-netcdf cray-hdf5 PrgEnv-gnu PrgEnv-intel PrgEnv-nvidia PrgEnv-cray PrgEnv-aocc gcc-native intel intel-oneapi nvidia aocc cudatoolkit climate-utils cray-libsci matlab craype-accel-nvidia80 craype-accel-host perftools-base perftools darshan
module load PrgEnv-gnu/8.5.0 gcc-native/13.2 cray-libsci/24.07.0 craype-accel-host craype/2.7.32 cray-mpich/8.1.30 cray-hdf5-parallel/1.14.3.9 cray-netcdf-hdf5parallel/4.9.0.13 cray-parallel-netcdf/1.12.3.13 cmake/3.30.2
export MPICH_SMP_SINGLE_COPY_MODE=CMA
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_MPIIO_DVS_MAXNODES=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export PERL5LIB=/global/cfs/cdirs/e3sm/perl/lib/perl5-only-switch
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=kdreg2
export MPICH_COLL_SYNC=MPI_Bcast
export NETCDF_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.13/gnu/12.3
export PNETCDF_PATH=/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3
export GATOR_INITIAL_MB=4000MB
export ADIOS2_ROOT=/global/cfs/cdirs/e3sm/3rdparty/adios2/2.9.1/cray-mpich-8.1.25/gcc-11.2.0
export BLA_VENDOR=Generic
export Albany_ROOT=/global/common/software/e3sm/albany/2024.03.26/gcc/11.2.0
export Trilinos_ROOT=/global/common/software/e3sm/trilinos/15.1.1/gcc/11.2.0
export MOAB_ROOT=/global/cfs/cdirs/e3sm/software/moab/gnu
