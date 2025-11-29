#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
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
BLAS_LIB_PATH  := $(HOME)/flame/lib
#MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64/
MKL_LIB_PATH   := ${MKLROOT}/lib/intel64

# OpenBLAS, ATLAS, and MKL libraries.
#BLAS_LIB       := $(LIB_PATH)/libblas.a
#BLAS_LIB       := $(LIB_PATH)/libgoto.a
#BLAS_LIB       := $(LIB_PATH)/libgoto2.a
OBLAS_LIB       := $(BLAS_LIB_PATH)/libopenblas.a
ABLAS_LIB       := $(BLAS_LIB_PATH)/libf77blas.a \
                   $(BLAS_LIB_PATH)/libatlas.a
#MBLAS_LIB       := -L$(MKL_LIB_PATH) \
#                   -lmkl_sequential \
#                   -lmkl_core \
#                   -lmkl_intel_lp64
#MBLAS_LIB      := -Wl,--start-group \
#                  $(MKL_LIB_PATH)/libmkl_sequential.a \
#                  $(MKL_LIB_PATH)/libmkl_core.a \
#                  $(MKL_LIB_PATH)/libmkl_intel_ilp64.a \
#                  -Wl,--end-group \
#                  -lpthread -lm
MBLAS_LIB       := -L$(MKL_LIB_PATH) \
                   -lmkl_intel_lp64 \
                   -lmkl_core \
                   -lmkl_sequential \
                   -lpthread -lm -ldl



#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

## Gather all local source files.
TEST_SIZES_SRC := test_size.c

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

MAKE_BLIS      := yes
MAKE_OPENBLAS  := yes
MAKE_MKL       := yes
MAKE_ATLAS     := no
MAKE_DCOMPLEX  := yes

ifeq ($(MAKE_BLIS),yes)
TEST_BINS += test_blis01 \
             test_blis02 \
             test_blis03 \
             test_blis04 \
             test_blis05 \
             test_blis06
ifeq ($(MAKE_DCOMPLEX),yes)
TEST_BINS += test_blis07 \
             test_blis08 \
             test_blis09 \
             test_blis10 \
             test_blis11 \
             test_blis12
endif
endif
ifeq ($(MAKE_OPENBLAS),yes)
TEST_BINS += test_oblas01 \
             test_oblas02 \
             test_oblas03 \
             test_oblas04 \
             test_oblas05 \
             test_oblas06
ifeq ($(MAKE_DCOMPLEX),yes)
TEST_BINS += test_oblas07 \
             test_oblas08 \
             test_oblas09 \
             test_oblas10 \
             test_oblas11 \
             test_oblas12
endif
endif
ifeq ($(MAKE_ATLAS),yes)
TEST_BINS += test_ablas01 \
             test_ablas02 \
             test_ablas03 \
             test_ablas04 \
             test_ablas05 \
             test_ablas06
ifeq ($(MAKE_DCOMPLEX),yes)
TEST_BINS += test_ablas07 \
             test_ablas08 \
             test_ablas09 \
             test_ablas10 \
             test_ablas11 \
             test_ablas12
endif
endif
ifeq ($(MAKE_MKL),yes)
TEST_BINS += test_mblas01 \
             test_mblas02 \
             test_mblas03 \
             test_mblas04 \
             test_mblas05 \
             test_mblas06
ifeq ($(MAKE_DCOMPLEX),yes)
TEST_BINS += test_mblas07 \
             test_mblas08 \
             test_mblas09 \
             test_mblas10 \
             test_mblas11 \
             test_mblas12
endif
endif


# --Object file rules --

all: $(TEST_BINS)

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@


# -- Executable file rules --

# BLIS rules
test_blis01:
	$(CC) $(CFLAGS) -DNBLIS=1  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis02:
	$(CC) $(CFLAGS) -DNBLIS=2  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis03:
	$(CC) $(CFLAGS) -DNBLIS=3  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis04:
	$(CC) $(CFLAGS) -DNBLIS=4  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis05:
	$(CC) $(CFLAGS) -DNBLIS=5  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis06:
	$(CC) $(CFLAGS) -DNBLIS=6  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis07:
	$(CC) $(CFLAGS) -DNBLIS=7  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis08:
	$(CC) $(CFLAGS) -DNBLIS=8  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis09:
	$(CC) $(CFLAGS) -DNBLIS=9  $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis10:
	$(CC) $(CFLAGS) -DNBLIS=10 $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis11:
	$(CC) $(CFLAGS) -DNBLIS=11 $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_blis12:
	$(CC) $(CFLAGS) -DNBLIS=12 $(TEST_SIZES_SRC)              $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x

# OpenBLAS rules
test_oblas01:
	$(CC) $(CFLAGS) -DNBLAS=1  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas02:
	$(CC) $(CFLAGS) -DNBLAS=2  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas03:
	$(CC) $(CFLAGS) -DNBLAS=3  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas04:
	$(CC) $(CFLAGS) -DNBLAS=4  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas05:
	$(CC) $(CFLAGS) -DNBLAS=5  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas06:
	$(CC) $(CFLAGS) -DNBLAS=6  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas07:
	$(CC) $(CFLAGS) -DNBLAS=7  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas08:
	$(CC) $(CFLAGS) -DNBLAS=8  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas09:
	$(CC) $(CFLAGS) -DNBLAS=9  $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas10:
	$(CC) $(CFLAGS) -DNBLAS=10 $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas11:
	$(CC) $(CFLAGS) -DNBLAS=11 $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_oblas12:
	$(CC) $(CFLAGS) -DNBLAS=12 $(TEST_SIZES_SRC) $(OBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x

# ATLAS BLAS rules
test_ablas01:
	$(CC) $(CFLAGS) -DNBLAS=1  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas02:
	$(CC) $(CFLAGS) -DNBLAS=2  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas03:
	$(CC) $(CFLAGS) -DNBLAS=3  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas04:
	$(CC) $(CFLAGS) -DNBLAS=4  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas05:
	$(CC) $(CFLAGS) -DNBLAS=5  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas06:
	$(CC) $(CFLAGS) -DNBLAS=6  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas07:
	$(CC) $(CFLAGS) -DNBLAS=7  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas08:
	$(CC) $(CFLAGS) -DNBLAS=8  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas09:
	$(CC) $(CFLAGS) -DNBLAS=9  $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas10:
	$(CC) $(CFLAGS) -DNBLAS=10 $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas11:
	$(CC) $(CFLAGS) -DNBLAS=11 $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_ablas12:
	$(CC) $(CFLAGS) -DNBLAS=12 $(TEST_SIZES_SRC) $(ABLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x

# MKL BLAS rules
test_mblas01:
	$(CC) $(CFLAGS) -DNBLAS=1  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas02:
	$(CC) $(CFLAGS) -DNBLAS=2  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas03:
	$(CC) $(CFLAGS) -DNBLAS=3  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas04:
	$(CC) $(CFLAGS) -DNBLAS=4  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas05:
	$(CC) $(CFLAGS) -DNBLAS=5  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas06:
	$(CC) $(CFLAGS) -DNBLAS=6  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas07:
	$(CC) $(CFLAGS) -DNBLAS=7  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas08:
	$(CC) $(CFLAGS) -DNBLAS=8  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas09:
	$(CC) $(CFLAGS) -DNBLAS=9  $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas10:
	$(CC) $(CFLAGS) -DNBLAS=10 $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas11:
	$(CC) $(CFLAGS) -DNBLAS=11 $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x
test_mblas12:
	$(CC) $(CFLAGS) -DNBLAS=12 $(TEST_SIZES_SRC) $(MBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@.x


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.x

