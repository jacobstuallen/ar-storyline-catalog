
# This file is auto-generated, do not edit. If you want to change
# sharedlib flags, you can edit the cmake_macros in this case. You
# can change flags for specific sharedlibs only by checking COMP_NAME.

CMAKE_CXX_FLAGS := 
CMAKE_CXX_FLAGS_DEBUG :=  -g -Wall -fbacktrace
CMAKE_CXX_FLAGS_RELEASE :=  -O
CMAKE_C_FLAGS := -mcmodel=medium
CMAKE_C_FLAGS_DEBUG :=  -g -Wall -fbacktrace -fcheck=bounds -ffpe-trap=invalid,zero,overflow
CMAKE_C_FLAGS_RELEASE :=  -O -O2 -g
CMAKE_EXE_LINKER_FLAGS :=  -static-libstdc++
CMAKE_Fortran_FLAGS := -mcmodel=medium -fconvert=big-endian -ffree-line-length-none -ffixed-line-length-none -fallow-argument-mismatch
CMAKE_Fortran_FLAGS_DEBUG :=  -g -Wall -fbacktrace -fcheck=bounds,pointer -ffpe-trap=invalid,zero,overflow
CMAKE_Fortran_FLAGS_RELEASE :=  -O -O2 -g
CMAKE_Fortran_FORMAT_FIXED_FLAG :=  -ffixed-form
CMAKE_Fortran_FORMAT_FREE_FLAG :=  -ffree-form
CMAKE_PARENT_LIST_FILE := /pscratch/sd/j/jsallen/E3SM-v3.0.2-fresh/cases/Storyline_2024_WUS_ne30x16_pg2_G-NDG_v2/Macros.cmake
CONFIG_ARGS :=  --host=cray
CPPDEFS := $(CPPDEFS)  -DFORTRANUNDERSCORE -DNO_R16 -DCPRGNU
CPPDEFS_DEBUG :=  -DYAKL_DEBUG
E3SM_LINK_WITH_FORTRAN := FALSE
HAS_F2008_CONTIGUOUS := FALSE
KOKKOS_OPTIONS := -DKokkos_ENABLE_SERIAL=On
MPICC := cc
MPICXX := CC
MPIFC := ftn
SCC := gcc
SCXX := g++
SFC := gfortran

ifeq "$(COMP_NAME)" "driver"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "eam"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "elm"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "cice"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "docn"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "srof"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "sglc"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "swav"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "siac"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "sesp"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "mct"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "pio2"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "gptl"
  CMAKE_MATCH_0 := 
  CPPDEFS := $(CPPDEFS)  -DFORTRANUNDERSCORE -DNO_R16 -DCPRGNU -DHAVE_NANOTIME -DBIT64 -DHAVE_SLASHPROC -DHAVE_GETTIMEOFDAY
endif
ifeq "$(COMP_NAME)" "csm_share"
  CMAKE_C_FLAGS := -mcmodel=medium -std=c99
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "csm_share_cpl7"
  CMAKE_MATCH_0 := 
endif
ifeq "$(COMP_NAME)" "mpi-serial"
  CMAKE_MATCH_0 := 
endif
