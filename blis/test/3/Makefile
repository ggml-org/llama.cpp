#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2018, Advanced Micro Devices, Inc.
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

.PHONY: all \
        check-env check-env-mk check-lib \
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
# --- BLAS implementations -----------------------------------------------------
#

# BLAS library path(s). This is where the BLAS libraries reside.
HOME_LIB_PATH  := $(HOME)/flame/lib

# OpenBLAS
OPENBLAS_LIB   := $(HOME_LIB_PATH)/libopenblas.a
OPENBLASP_LIB  := $(HOME_LIB_PATH)/libopenblasp.a

# ATLAS
#ATLAS_LIB      := $(HOME_LIB_PATH)/libf77blas.a \
#                  $(HOME_LIB_PATH)/libatlas.a

# Eigen
EIGEN_INC      := $(HOME)/flame/eigen/include/eigen3
EIGEN_LIB      := $(HOME_LIB_PATH)/libeigen_blas_static.a
EIGENP_LIB     := $(EIGEN_LIB)

# MKL
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_sequential \
                  -lpthread -lm -ldl
#MKLP_LIB       := -L$(MKL_LIB_PATH) \
#                  -lmkl_intel_thread \
#                  -lmkl_core \
#                  -lmkl_intel_ilp64 \
#                  -L$(ICC_LIB_PATH) \
#                  -liomp5
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
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(sort $(patsubst $(TEST_SRC_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(TEST_SRC_PATH)/*.c)))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS.
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Define a set of CFLAGS for use with C++ and Eigen.
CXXFLAGS       := $(subst -std=c99,-std=c++11,$(CFLAGS))
CXXFLAGS       += -I$(EIGEN_INC)

# Create a copy of CXXFLAGS without -fopenmp in order to disable multithreading.
CXXFLAGS_ST    := -march=native $(subst -fopenmp,,$(CXXFLAGS))
CXXFLAGS_MT    := -march=native $(CXXFLAGS)


# Which library?
DEF_BLI  := -DBLIS
DEF_BLA  := -DBLAS
DEF_EIG  := -DEIGEN

# Implementation string.
STR_BLI  := -DIMPL_STR=\"blis\"
STR_OBL  := -DIMPL_STR=\"openblas\"
STR_EIG  := -DIMPL_STR=\"eigen\"
STR_VEN  := -DIMPL_STR=\"vendor\"

# Single or multithreaded string.
STR_ST   := -DTHR_STR=\"st\"
STR_MT   := -DTHR_STR=\"mt\"



#
# --- Targets/rules ------------------------------------------------------------
#

all:      all-st

all-st:   blis-st openblas-st mkl-st eigen-st
all-mt:   blis-mt openblas-mt mkl-mt eigen-mt

blis:     blis-st
openblas: openblas-st
eigen:    eigen-st
vendor:   vendor-st
mkl:      mkl-st

# Define the datatypes, operations, and implementations.
OPS    := gemm hemm herk trmm trsm
BIMPLS := blis openblas vendor
EIMPLS := eigen

# Define a function to construct object filenames from the operations
# given an implementation.
get-st-objs = $(foreach op,$(OPS),test_$(op)_$(1)_st.o)
get-mt-objs = $(foreach op,$(OPS),test_$(op)_$(1)_mt.o)

# Construct object and binary names for single-threaded and multithreaded
# files for BLIS, OpenBLAS, Eigen, and a vendor library (e.g. MKL).
BLIS_ST_OBJS     := $(call get-st-objs,blis)
BLIS_ST_BINS     := $(patsubst %.o,%.x,$(BLIS_ST_OBJS))

BLIS_MT_OBJS     := $(call get-mt-objs,blis)
BLIS_MT_BINS     := $(patsubst %.o,%.x,$(BLIS_MT_OBJS))

OPENBLAS_ST_OBJS := $(call get-st-objs,openblas)
OPENBLAS_ST_BINS := $(patsubst %.o,%.x,$(OPENBLAS_ST_OBJS))

OPENBLAS_MT_OBJS := $(call get-mt-objs,openblas)
OPENBLAS_MT_BINS := $(patsubst %.o,%.x,$(OPENBLAS_MT_OBJS))

EIGEN_ST_OBJS    := $(call get-st-objs,eigen)
EIGEN_ST_BINS    := $(patsubst %.o,%.x,$(EIGEN_ST_OBJS))

EIGEN_MT_OBJS    := $(call get-mt-objs,eigen)
EIGEN_MT_BINS    := $(patsubst %.o,%.x,$(EIGEN_MT_OBJS))

VENDOR_ST_OBJS   := $(call get-st-objs,vendor)
VENDOR_ST_BINS   := $(patsubst %.o,%.x,$(VENDOR_ST_OBJS))

VENDOR_MT_OBJS   := $(call get-mt-objs,vendor)
VENDOR_MT_BINS   := $(patsubst %.o,%.x,$(VENDOR_MT_OBJS))

# List other miscellaneous object files
UTIL_OBJS        := test_utils.o
UTIL_HDRS        := test_utils.h

# Define some targets associated with the above object/binary files.
blis-st:     check-env $(BLIS_ST_BINS)
blis-mt:     check-env $(BLIS_MT_BINS)
openblas-st: check-env $(OPENBLAS_ST_BINS)
openblas-mt: check-env $(OPENBLAS_MT_BINS)
eigen-st:    check-env $(EIGEN_ST_BINS)
eigen-mt:    check-env $(EIGEN_MT_BINS)
vendor-st:   check-env $(VENDOR_ST_BINS)
vendor-mt:   check-env $(VENDOR_MT_BINS)
mkl-st:      vendor-st
mkl-mt:      vendor-mt
armpl-st:    vendor-st
armpl-mt:    vendor-mt

# Mark the object files as intermediate so that make will remove them
# automatically after building the binaries on which they depend.
.INTERMEDIATE: $(BLIS_ST_OBJS)     $(BLIS_MT_OBJS)
.INTERMEDIATE: $(OPENBLAS_ST_OBJS) $(OPENBLAS_MT_OBJS)
.INTERMEDIATE: $(EIGEN_ST_OBJS)    $(EIGEN_MT_OBJS)
.INTERMEDIATE: $(VENDOR_ST_OBJS)   $(VENDOR_MT_OBJS)
.INTERMEDIATE: $(UTIL_OBJS)


# -- Object file rules --

# A function to return other cpp macros that help the test driver
# identify the implementation.
get-bl-cpp = $(strip \
             $(if $(findstring     blis,$(1)),$(STR_BLI) $(DEF_BLI),\
             $(if $(findstring openblas,$(1)),$(STR_OBL) $(DEF_BLA),\
             $(if $(and $(findstring eigen,$(1)),\
                        $(findstring  gemm,$(2))),\
                                              $(STR_EIG) $(DEF_EIG),\
             $(if       $(findstring eigen,$(1)),\
                                              $(STR_EIG) $(DEF_BLA),\
                                              $(STR_VEN) $(DEF_BLA))))))

# Rules for miscellaneous files.
test_utils.o: test_utils.c test_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

# Rules for BLIS and BLAS libraries.
define make-st-rule
test_$(1)_$(2)_st.o: test_$(op).c Makefile
	$(CC) $(CFLAGS) $(call get-bl-cpp,$(2),$(1)) $(STR_ST) -c $$< -o $$@
endef

define make-mt-rule
test_$(1)_$(2)_mt.o: test_$(op).c Makefile
	$(CC) $(CFLAGS) $(call get-bl-cpp,$(2),$(1)) $(STR_MT) -c $$< -o $$@
endef

$(foreach op,$(OPS), \
$(foreach im,$(BIMPLS),$(eval $(call make-st-rule,$(op),$(im)))))

$(foreach op,$(OPS), \
$(foreach im,$(BIMPLS),$(eval $(call make-mt-rule,$(op),$(im)))))

# Rules for Eigen.
# NOTE: Eigen determines single- vs. multithreadedness at compile time.
define make-eigst-rule
test_$(1)_$(2)_st.o: test_$(op).c Makefile
	$(CXX) $(CXXFLAGS_ST) $(call get-bl-cpp,$(2),$(1)) $(STR_ST) -c $$< -o $$@
endef

define make-eigmt-rule
test_$(1)_$(2)_mt.o: test_$(op).c Makefile
	$(CXX) $(CXXFLAGS_MT) $(call get-bl-cpp,$(2),$(1)) $(STR_MT) -c $$< -o $$@
endef

$(foreach op,$(OPS), \
$(foreach im,$(EIMPLS),$(eval $(call make-eigst-rule,$(op),$(im)))))

$(foreach op,$(OPS), \
$(foreach im,$(EIMPLS),$(eval $(call make-eigmt-rule,$(op),$(im)))))


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

# Combine the miscellaneous objects with libblis for conciseness (since all
# driver binaries depend on these objects).
COMMON_OBJS := $(UTIL_OBJS) $(LIBBLIS_LINK)

test_%_blis_st.x:     test_%_blis_st.o     $(COMMON_OBJS)
	$(CC) $(strip $<                    $(COMMON_OBJS) $(LDFLAGS) -o $@)

test_%_blis_mt.x:     test_%_blis_mt.o     $(COMMON_OBJS)
	$(CC) $(strip $<                    $(COMMON_OBJS) $(LDFLAGS) -o $@)


test_%_openblas_st.x: test_%_openblas_st.o $(COMMON_OBJS)
	$(CC) $(strip $<   $(OPENBLAS_LIB)  $(COMMON_OBJS) $(LDFLAGS) -o $@)

test_%_openblas_mt.x: test_%_openblas_mt.o $(COMMON_OBJS)
	$(CC) $(strip $<   $(OPENBLASP_LIB) $(COMMON_OBJS) $(LDFLAGS) -o $@)


test_%_eigen_st.x:    test_%_eigen_st.o    $(COMMON_OBJS)
	$(CXX) $(strip $<  $(EIGEN_LIB)     $(COMMON_OBJS) $(LDFLAGS) -o $@)

test_%_eigen_mt.x:    test_%_eigen_mt.o    $(COMMON_OBJS)
	$(CXX) $(strip $<  $(EIGENP_LIB)    $(COMMON_OBJS) $(LDFLAGS) -o $@)


test_%_vendor_st.x:   test_%_vendor_st.o   $(COMMON_OBJS)
	$(CC) $(strip $<   $(VENDOR_LIB)    $(COMMON_OBJS) $(LDFLAGS) -o $@)

test_%_vendor_mt.x:   test_%_vendor_mt.o   $(COMMON_OBJS)
	$(CC) $(strip $<   $(VENDORP_LIB)   $(COMMON_OBJS) $(LDFLAGS) -o $@)


# -- Environment check rules --

check-env: check-lib

check-env-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-lib: check-env-mk
ifeq ($(wildcard $(LIBBLIS_LINK)),)
	$(error Cannot proceed: BLIS library not yet built! Run make first)
endif


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

