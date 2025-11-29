#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2017 - 2019, Advanced Micro Devices, Inc.
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
        blis \
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


#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
CPP_SRC_PATH  := ../cpp/
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(patsubst $(TEST_SRC_PATH)/%.c, \
                             $(TEST_OBJ_PATH)/%.o, \
                             $(wildcard $(TEST_SRC_PATH)/*.c))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

CXX	= g++

# Use the CFLAGS for the configuration family.
override CFLAGS         += $(call get-sandbox-cxxflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
#CFLAGS         = -O0 -g -Wall 
#CFLAGS         += -I$(INC_PATH)
override CFLAGS         += -I$(TEST_SRC_PATH)
override CFLAGS      += -I$(CPP_SRC_PATH)

LINKER = $(CXX)

# Locate the libblis library to which we will link.
LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)



#
# --- Targets/rules ------------------------------------------------------------
#

# Complete list of possible targets when defining 'all':
#
#   blis 
#
all: blis 


blis: test_asum_blis.x \
	test_axpy_blis.x \
	test_copy_blis.x \
	test_dot_blis.x \
	test_dotc_blis.x \
	test_gbmv_blis.x \
	test_gemm_blis.x \
	test_gemv_blis.x \
	test_ger_blis.x \
	test_gerc_blis.x \
	test_geru_blis.x \
	test_hemm_blis.x \
	test_hemv_blis.x \
	test_her2_blis.x \
	test_her_blis.x \
	test_herk_blis.x \
	test_hpr2_blis.x \
	test_hpr_blis.x \
	test_nrm2_blis.x \
	test_rot_blis.x \
	test_rotg_blis.x \
	test_rotm_blis.x \
	test_rotmg_blis.x \
	test_scal_blis.x \
	test_sdsdot_blis.x \
	test_spr2_blis.x \
	test_spr_blis.x \
	test_swap_blis.x \
	test_symm_blis.x \
	test_syr2_blis.x \
	test_syr2k_blis.x \
	test_syr_blis.x \
	test_syrk_blis.x \
	test_tbmv_blis.x \
	test_tbsv_blis.x \
	test_tpmv_blis.x \
	test_tpsv_blis.x \
	test_trmm_blis.x \
	test_trsm_blis.x \
	test_trsv_blis.x 
 


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.cc
	$(CXX) $(CFLAGS) -c $< -o $@

test_%_blis.o: test_%.cc
	@$(CXX) $(CFLAGS) -c $< -o $@


# -- Executable file rules --

test_%_blis.x: test_%_blis.o $(LIBBLIS_LINK)
	@$(LINKER) $^       $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	./$@

# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

