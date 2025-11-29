#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2017 - 2020, Advanced Micro Devices, Inc. All rights reserved.
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
        blis openblas atlas mkl \
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
DIST_PATH  := ..
LIB_PATH    = ../lib/$(CONFIG_NAME)
INC_PATH    = ../include/$(CONFIG_NAME)
SHARE_PATH := ..
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
BLAS_LIB_PATH  := $(HOME)/flame/lib
#MKL_LIB_PATH   := /opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64
#MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64

# OpenBLAS
OPENBLAS_LIB   := $(BLAS_LIB_PATH)/libopenblas.a

# MKL
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_sequential \
                  -lpthread -lm -ldl


#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(patsubst $(TEST_SRC_PATH)/%.c, \
                             $(TEST_OBJ_PATH)/%.o, \
                             $(wildcard $(TEST_SRC_PATH)/*.c))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)



#
# --- Targets/rules ------------------------------------------------------------
#

# Define the operations we will test.
TEST_OPS := dotv axpyv axpbyv\
            gemv ger hemv her her2 trmv trsv \
            gemm gemm3m gemm_batch hemm herk her2k trmm trsm

# Optionally test gemmt, which some libraries might not implement.
ifeq ($(BUILD_GEMMT),yes)
TEST_OPS := $(TEST_OPS) gemmt
endif

# Define a function to create the executable names.
test-bins = $(foreach op, $(TEST_OPS), test_$(op)_$(1).x)

# Create the list of executables for each implementation.
TEST_BINS_BLIS     := $(call test-bins,blis)
TEST_BINS_OPENBLAS := $(call test-bins,openblas)
TEST_BINS_MKL      := $(call test-bins,mkl)


all: blis openblas mkl

blis: check-env $(TEST_BINS_BLIS)

openblas: check-env $(TEST_BINS_OPENBLAS)

mkl: check-env $(TEST_BINS_MKL)


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@


test_%_openblas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"openblas\" -c $< -o $@

test_%_mkl.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mkl\" -c $< -o $@

test_%_blis.o: test_%.c
	$(CC) $(CFLAGS) -DBLIS -c $< -o $@


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_openblas.x: test_%_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(OPENBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mkl.x: test_%_mkl.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(MKL_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_blis.x: test_%_blis.o $(LIBBLIS_LINK)
	$(LINKER) $<                             $(LIBBLIS_LINK) $(LDFLAGS) -o $@


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

