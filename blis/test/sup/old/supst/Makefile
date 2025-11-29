#!/bin/bash
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2019, Advanced Micro Devices, Inc.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# Makefile
#
# Field G. Van Zee
#
# Makefile for standalone BLIS test drivers.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all all-st all-mt \
        blis blis-st blis-mt \
        clean cleanx



#
# --- Determine makefile fragment location -------------------------------------
#

# Comments:
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
ifneq ($(strip $(BLIS_INSTALL_PATH)),)
LIB_PATH   := $(BLIS_INSTALL_PATH)/lib
INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
SHARE_PATH := $(BLIS_INSTALL_PATH)/share/blis
else
DIST_PATH  := ../..
LIB_PATH    = ../../lib/$(CONFIG_NAME)
INC_PATH    = ../../include/$(CONFIG_NAME)
SHARE_PATH := ../..
endif



#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk



#
# --- BLAS and LAPACK implementations ------------------------------------------
#

# BLIS library and header path. This is simply wherever it was installed.
#BLIS_LIB_PATH  := $(INSTALL_PREFIX)/lib
#BLIS_INC_PATH  := $(INSTALL_PREFIX)/include/blis

# BLIS library.
#BLIS_LIB       := $(BLIS_LIB_PATH)/libblis.a

# BLAS library path(s). This is where the BLAS libraries reside.
HOME_LIB_PATH  := $(HOME)/flame/lib
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64

# netlib BLAS
NETLIB_LIB     := $(HOME_LIB_PATH)/libblas.a

# OpenBLAS
OPENBLAS_LIB   := $(HOME_LIB_PATH)/libopenblas.a
OPENBLASP_LIB  := $(HOME_LIB_PATH)/libopenblasp.a

# BLASFEO
BLASFEO_LIB    := $(HOME_LIB_PATH)/libblasfeo.a

# libxsmm
LIBXSMM_LIB    := $(HOME_LIB_PATH)/libxsmm.a -ldl \
                  $(NETLIB_LIB) -lgfortran

# ATLAS
ATLAS_LIB      := $(HOME_LIB_PATH)/libf77blas.a \
                  $(HOME_LIB_PATH)/libatlas.a

# Eigen
EIGEN_INC      := $(HOME)/flame/eigen/include/eigen3
EIGEN_LIB      := $(HOME_LIB_PATH)/libeigen_blas_static.a
EIGENP_LIB     := $(EIGEN_LIB)

# MKL
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_sequential \
                  -lpthread -lm -ldl
MKLP_LIB       := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_gnu_thread \
                  -lpthread -lm -ldl -fopenmp
                  #-L$(ICC_LIB_PATH) \
                  #-lgomp

VENDOR_LIB     := $(MKL_LIB)
VENDORP_LIB    := $(MKLP_LIB)


#
# --- Problem size definitions -------------------------------------------------
#

# Single core
PS_BEGIN_3L := 2
PS_MAX_3L   := 400
PS_INC_3L   := 2

PS_BEGIN_2L := 4
PS_MAX_2L   := 800
PS_INC_2L   := 4

PS_BEGIN_1L := 32
PS_MAX_1L   := 6400
PS_INC_1L   := 32


#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(sort $(patsubst $(TEST_SRC_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(TEST_SRC_PATH)/*.c)))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-frame-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS.
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Define a set of CFLAGS for use with C++ and Eigen.
CXXFLAGS       := $(subst -std=c99,-std=c++11,$(CFLAGS))
CXXFLAGS       += -I$(EIGEN_INC)

# Create a copy of CXXFLAGS without -fopenmp in order to disable multithreading.
CXXFLAGS_ST    := -march=native $(subst -fopenmp,,$(CXXFLAGS))
CXXFLAGS_MT    := -march=native $(CXXFLAGS)

# Single or multithreaded string.
STR_ST   := -DTHR_STR=\"st\"
STR_MT   := -DTHR_STR=\"mt\"

# Number of trials per problem size.
N_TRIALS := -DN_TRIALS=3

# Problem size specification.
PDEF_ST_1L := -DP_BEGIN=$(PS_BEGIN_1L) -DP_MAX=$(PS_MAX_1L) -DP_INC=$(PS_INC_1L)
PDEF_ST_2L := -DP_BEGIN=$(PS_BEGIN_2L) -DP_MAX=$(PS_MAX_2L) -DP_INC=$(PS_INC_2L)
PDEF_ST_3L := -DP_BEGIN=$(PS_BEGIN_3L) -DP_MAX=$(PS_MAX_3L) -DP_INC=$(PS_INC_3L)

ifeq ($(E),1)
ERRCHK := -DERROR_CHECK
else
ERRCHK := -DNO_ERROR_CHECK
endif

# Enumerate possible datatypes and computation precisions.
#dts := s d c z
DTS := d

TRANS := n_n \
         n_t \
         t_n \
         t_t

# While BLIS supports all combinations of row and column storage for matrices
# C, A, and B, the alternatives mostly only support CBLAS APIs, which inherently
# support only "all row-storage" or "all column-storage". Thus, we disable the
# building of those other drivers so that compilation/linking completes sooner.
#STORS := r_r_r \
#         r_r_c \
#         r_c_r \
#         r_c_c \
#         c_r_r \
#         c_r_c \
#         c_c_r \
#         c_c_c
STORS := r_r_r \
         c_c_c


SHAPES := l_l_s \
          l_s_l \
          s_l_l \
          s_s_l \
          s_l_s \
          l_s_s \
          l_l_l

SMS := 6
SNS := 8
SKS := 4


#
# --- Function definitions -----------------------------------------------------
#

# A function to strip the underscores from a list of strings.
stripu = $(subst _,,$(1))

# Various functions that help us construct the datatype combinations and then
# extract the needed datatype strings and C preprocessor define flags.
get-1of2 = $(word 1,$(subst _, ,$(1)))
get-2of2 = $(word 2,$(subst _, ,$(1)))

get-1of3 = $(word 1,$(subst _, ,$(1)))
get-2of3 = $(word 2,$(subst _, ,$(1)))
get-3of3 = $(word 3,$(subst _, ,$(1)))

# A function to return the correct PDEFS_ST variable given the shape string.
get-pdefs = $(strip $(subst l_l_l,$(PDEF_ST_3L), \
                    $(subst l_l_s,$(PDEF_ST_2L), \
                    $(subst l_s_l,$(PDEF_ST_2L), \
                    $(subst s_l_l,$(PDEF_ST_2L), \
                    $(subst s_s_l,$(PDEF_ST_1L), \
                    $(subst s_l_s,$(PDEF_ST_1L), \
                    $(subst l_s_s,$(PDEF_ST_1L),$(1)))))))))

# Datatype defs.
get-dt-cpp = $(strip \
             $(if $(findstring s,$(1)),-DDT=BLIS_FLOAT    -DIS_FLOAT,\
             $(if $(findstring d,$(1)),-DDT=BLIS_DOUBLE   -DIS_DOUBLE,\
             $(if $(findstring c,$(1)),-DDT=BLIS_SCOMPLEX -DIS_SCOMPLEX,\
                                       -DDT=BLIS_DCOMPLEX -DIS_DCOMPLEX))))

# Transpose defs.
get-tra-defs-a = $(strip $(subst n,-DTRANSA=BLIS_NO_TRANSPOSE -DA_NOTRANS, \
                         $(subst t,-DTRANSA=BLIS_TRANSPOSE    -DA_TRANS,$(call get-1of2,$(1)))))
get-tra-defs-b = $(strip $(subst n,-DTRANSB=BLIS_NO_TRANSPOSE -DB_NOTRANS, \
                         $(subst t,-DTRANSB=BLIS_TRANSPOSE    -DB_TRANS,$(call get-2of2,$(1)))))
get-tra-defs = $(call get-tra-defs-a,$(1)) $(call get-tra-defs-b,$(1))

# Storage defs.
get-sto-uch-a = $(strip $(subst r,R, \
                        $(subst c,C,$(call get-1of3,$(1)))))
get-sto-uch-b = $(strip $(subst r,R, \
                        $(subst c,C,$(call get-2of3,$(1)))))
get-sto-uch-c = $(strip $(subst r,R, \
                        $(subst c,C,$(call get-3of3,$(1)))))
get-sto-defs = $(strip \
                 -DSTOR3=BLIS_$(call get-sto-uch-a,$(1))$(call get-sto-uch-b,$(1))$(call get-sto-uch-c,$(1)) \
                 -DA_STOR_$(call get-sto-uch-a,$(1)) \
                 -DB_STOR_$(call get-sto-uch-b,$(1)) \
                 -DC_STOR_$(call get-sto-uch-c,$(1)))

# Dimension defs.
get-shape-defs-cm = $(if $(findstring l,$(1)),-DM_DIM=-1,-DM_DIM=$(2))
get-shape-defs-cn = $(if $(findstring l,$(1)),-DN_DIM=-1,-DN_DIM=$(2))
get-shape-defs-ck = $(if $(findstring l,$(1)),-DK_DIM=-1,-DK_DIM=$(2))
get-shape-defs-m  = $(call get-shape-defs-cm,$(call get-1of3,$(1)),$(2))
get-shape-defs-n  = $(call get-shape-defs-cn,$(call get-2of3,$(1)),$(2))
get-shape-defs-k  = $(call get-shape-defs-ck,$(call get-3of3,$(1)),$(2))

# arguments: 1: shape (w/ underscores)  2: smallm  3: smalln  4: smallk
get-shape-defs    = $(strip $(call get-shape-defs-m,$(1),$(2)) \
                            $(call get-shape-defs-n,$(1),$(3)) \
                            $(call get-shape-defs-k,$(1),$(4)))

#$(error l_l_s 6 8 4 = $(call get-shape-defs,l_l_s,6,8,4))

# Shape-dimension string.
get-shape-str-ch  = $(if $(findstring l,$(1)),p,$(2))
get-shape-str-m   = $(call get-shape-str-ch,$(call get-1of3,$(1)),$(2))
get-shape-str-n   = $(call get-shape-str-ch,$(call get-2of3,$(1)),$(2))
get-shape-str-k   = $(call get-shape-str-ch,$(call get-3of3,$(1)),$(2))

# arguments: 1: shape (w/ underscores)  2: smallm  3: smalln  4: smallk
get-shape-dim-str = m$(call get-shape-str-m,$(1),$(2))n$(call get-shape-str-n,$(1),$(3))k$(call get-shape-str-k,$(1),$(4))

# Implementation defs.
# Define a function to return the appropriate -DSTR= and -D[BLIS|BLAS] flags.
get-imp-defs = $(strip $(subst  blissup,-DSTR=\"$(1)\" -DBLIS -DSUP, \
                       $(subst blislpab,-DSTR=\"$(1)\" -DBLIS, \
                       $(subst    eigen,-DSTR=\"$(1)\" -DEIGEN, \
                       $(subst openblas,-DSTR=\"$(1)\" -DCBLAS, \
                       $(subst  blasfeo,-DSTR=\"$(1)\" -DCBLAS, \
                       $(subst  libxsmm,-DSTR=\"$(1)\" -DBLAS -DXSMM, \
                       $(subst   vendor,-DSTR=\"$(1)\" -DCBLAS,$(1)))))))))

TRANS0  = $(call stripu,$(TRANS))
STORS0  = $(call stripu,$(STORS))

# Limit BLAS and Eigen to only using all row-stored, or all column-stored matrices.
# Also, limit libxsmm to using all column-stored matrices since it does not offer
# CBLAS interfaces.
BSTORS0 = rrr ccc
ESTORS0 = rrr ccc
XSTORS0 = ccc


#
# --- Object and binary file definitons ----------------------------------------
#

get-st-objs = $(foreach dt,$(1),$(foreach tr,$(2),$(foreach st,$(3),$(foreach sh,$(4),$(foreach sm,$(5),$(foreach sn,$(6),$(foreach sk,$(7),test_$(dt)gemm_$(tr)_$(st)_$(call get-shape-dim-str,$(sh),$(sm),$(sn),$(sk))_$(8)_st.o)))))))

# Build a list of object files and binaries for each single-threaded
# implementation using the get-st-objs() function defined above.
BLISSUP_ST_OBJS  := $(call get-st-objs,$(DTS),$(TRANS0),$(STORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),blissup)
BLISSUP_ST_BINS  := $(patsubst %.o,%.x,$(BLISSUP_ST_OBJS))

BLISLPAB_ST_OBJS := $(call get-st-objs,$(DTS),$(TRANS0),$(STORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),blislpab)
BLISLPAB_ST_BINS := $(patsubst %.o,%.x,$(BLISLPAB_ST_OBJS))

EIGEN_ST_OBJS    := $(call get-st-objs,$(DTS),$(TRANS0),$(ESTORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),eigen)
EIGEN_ST_BINS    := $(patsubst %.o,%.x,$(EIGEN_ST_OBJS))

OPENBLAS_ST_OBJS := $(call get-st-objs,$(DTS),$(TRANS0),$(BSTORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),openblas)
OPENBLAS_ST_BINS := $(patsubst %.o,%.x,$(OPENBLAS_ST_OBJS))

BLASFEO_ST_OBJS  := $(call get-st-objs,$(DTS),$(TRANS0),$(BSTORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),blasfeo)
BLASFEO_ST_BINS  := $(patsubst %.o,%.x,$(BLASFEO_ST_OBJS))

LIBXSMM_ST_OBJS  := $(call get-st-objs,$(DTS),$(TRANS0),$(XSTORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),libxsmm)
LIBXSMM_ST_BINS  := $(patsubst %.o,%.x,$(LIBXSMM_ST_OBJS))

VENDOR_ST_OBJS   := $(call get-st-objs,$(DTS),$(TRANS0),$(BSTORS0),$(SHAPES),$(SMS),$(SNS),$(SKS),vendor)
VENDOR_ST_BINS   := $(patsubst %.o,%.x,$(VENDOR_ST_OBJS))

#$(error "objs = $(EIGEN_ST_BINS)" )

# Mark the object files as intermediate so that make will remove them
# automatically after building the binaries on which they depend.
.INTERMEDIATE: $(BLISSUP_ST_OBJS) \
               $(BLISLPAB_ST_OBJS) \
               $(EIGEN_ST_OBJS) \
               $(OPENBLAS_ST_OBJS) \
               $(BLASFEO_ST_OBJS) \
               $(LIBXSMM_ST_OBJS) \
               $(VENDOR_ST_OBJS)


#
# --- Targets/rules ------------------------------------------------------------
#

all:         st

blissup:     blissup-st
blislpab:    blislpab-st
eigen:       eigen-st
openblas:    openblas-st
blasfeo:     blasfeo-st
libxsmm:     libxsmm-st
vendor:      vendor-st

st:          blissup-st blislpab-st \
             eigen-st openblas-st blasfeo-st libxsmm-st vendor-st
blis:        blissup-st blislpab-st

blissup-st:  $(BLISSUP_ST_BINS)
blislpab-st: $(BLISLPAB_ST_BINS)
eigen-st:    $(EIGEN_ST_BINS)
openblas-st: $(OPENBLAS_ST_BINS)
blasfeo-st:  $(BLASFEO_ST_BINS)
libxsmm-st:  $(LIBXSMM_ST_BINS)
vendor-st:   $(VENDOR_ST_BINS)


# --Object file rules --

# Define the implementations for which we will instantiate compilation rules.
BIMPLS := blissup blislpab openblas blasfeo libxsmm vendor
EIMPLS := eigen

#      1     2  3   4 567  8
# test_dgemm_nn_rrr_mpn6kp_blissup_st.x

# Define the function that will be used to instantiate compilation rules
# for the various implementations.
define make-st-rule
test_$(1)gemm_$(call stripu,$(2))_$(call stripu,$(3))_$(call get-shape-dim-str,$(4),$(5),$(6),$(7))_$(8)_st.o: test_gemm.c Makefile
	$(CC) $(CFLAGS) $(ERRCHK) $(N_TRIALS) $(call get-pdefs,$(4)) $(call get-dt-cpp,$(1)) $(call get-tra-defs,$(2)) $(call get-sto-defs,$(3)) $(call get-shape-defs,$(4),$(5),$(6),$(7)) $(call get-imp-defs,$(8)) $(STR_ST) -c $$< -o $$@
endef

# Instantiate the rule function make-st-rule() for each BLIS/BLAS/CBLAS
# implementation.
$(foreach dt,$(DTS), \
$(foreach tr,$(TRANS), \
$(foreach st,$(STORS), \
$(foreach sh,$(SHAPES), \
$(foreach sm,$(SMS), \
$(foreach sn,$(SNS), \
$(foreach sk,$(SKS), \
$(foreach impl,$(BIMPLS), \
$(eval $(call make-st-rule,$(dt),$(tr),$(st),$(sh),$(sm),$(sn),$(sk),$(impl)))))))))))

# Define the function that will be used to instantiate compilation rules
# for the various implementations.
define make-eigst-rule
test_$(1)gemm_$(call stripu,$(2))_$(call stripu,$(3))_$(call get-shape-dim-str,$(4),$(5),$(6),$(7))_$(8)_st.o: test_gemm.c Makefile
	$(CXX) $(CXXFLAGS_ST) $(ERRCHK) $(N_TRIALS) $(call get-pdefs,$(4)) $(call get-dt-cpp,$(1)) $(call get-tra-defs,$(2)) $(call get-sto-defs,$(3)) $(call get-shape-defs,$(4),$(5),$(6),$(7)) $(call get-imp-defs,$(8)) $(STR_ST) -c $$< -o $$@
endef

# Instantiate the rule function make-st-rule() for each Eigen implementation.
$(foreach dt,$(DTS), \
$(foreach tr,$(TRANS), \
$(foreach st,$(STORS), \
$(foreach sh,$(SHAPES), \
$(foreach sm,$(SMS), \
$(foreach sn,$(SNS), \
$(foreach sk,$(SKS), \
$(foreach impl,$(EIMPLS), \
$(eval $(call make-eigst-rule,$(dt),$(tr),$(st),$(sh),$(sm),$(sn),$(sk),$(impl)))))))))))


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_blissup_st.x:  test_%_blissup_st.o  $(LIBBLIS_LINK)
	$(CC) $(strip $<                       $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_blislpab_st.x: test_%_blislpab_st.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                       $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_eigen_st.x:    test_%_eigen_st.o    $(LIBBLIS_LINK)
	$(CXX) $(strip $<                      $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_openblas_st.x: test_%_openblas_st.o $(LIBBLIS_LINK)
	$(CC) $(strip $<  $(OPENBLAS_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_blasfeo_st.x:  test_%_blasfeo_st.o  $(LIBBLIS_LINK)
	$(CC) $(strip $<  $(BLASFEO_LIB)       $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_libxsmm_st.x:  test_%_libxsmm_st.o  $(LIBBLIS_LINK)
	$(CC) $(strip $<  $(LIBXSMM_LIB)       $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_vendor_st.x:   test_%_vendor_st.o   $(LIBBLIS_LINK)
	$(CC) $(strip $<  $(VENDOR_LIB)        $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.x *.o

