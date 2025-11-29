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
# --- Problem size definitions -------------------------------------------------
#

# Single core (single-threaded)
PS_BEGIN := 48
PS_MAX   := 2400
PS_INC   := 48

# Single-socket (multithreaded)
P1_BEGIN := 96
P1_MAX   := 4800
P1_INC   := 96

# Dual-socket (multithreaded)
P2_BEGIN := 144
P2_MAX   := 7200
P2_INC   := 144


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
BLI_DEF  := -DBLIS
BLA_DEF  := -DBLAS
EIG_DEF  := -DEIGEN

# Complex implementation type
D1M      := -DIND=BLIS_1M
DNAT     := -DIND=BLIS_NAT

# Implementation string
STR_1M   := -DSTR=\"1m_blis\"
STR_NAT  := -DSTR=\"asm_blis\"
STR_OBL  := -DSTR=\"openblas\"
STR_EIG  := -DSTR=\"eigen\"
STR_VEN  := -DSTR=\"vendor\"

# Single or multithreaded string
STR_ST   := -DTHR_STR=\"st\"
STR_1S   := -DTHR_STR=\"1s\"
STR_2S   := -DTHR_STR=\"2s\"

# Problem size specification
PDEF_ST  := -DP_BEGIN=$(PS_BEGIN)  -DP_INC=$(PS_INC)  -DP_MAX=$(PS_MAX)
PDEF_1S  := -DP_BEGIN=$(P1_BEGIN) -DP_INC=$(P1_INC) -DP_MAX=$(P1_MAX)
PDEF_2S  := -DP_BEGIN=$(P2_BEGIN) -DP_INC=$(P2_INC) -DP_MAX=$(P2_MAX)



#
# --- Targets/rules ------------------------------------------------------------
#

all:        all-st all-1s all-2s
blis:       blis-st blis-1s blis-2s
openblas:   openblas-st openblas-1s openblas-2s
eigen:      eigen-st eigen-1s eigen-2s
vendor:     vendor-st vendor-1s vendor-2s
mkl:        vendor
armpl:      vendor

all-st:     blis-st openblas-st mkl-st
all-1s:     blis-1s openblas-1s mkl-1s
all-2s:     blis-2s openblas-2s mkl-2s

blis-st:    blis-nat-st blis-1m-st
blis-1s:    blis-nat-1s blis-1m-1s
blis-2s:    blis-nat-2s blis-1m-2s

#blis-ind:   blis-ind-st blis-ind-mt
blis-nat:   blis-nat-st  blis-nat-1s  blis-nat-2s
blis-1m:    blis-1m-st   blis-1m-1s   blis-1m-2s

# Define the datatypes, operations, and implementations.
DTS    := s d c z
OPS    := gemm
BIMPLS := asm_blis 1m_blis openblas vendor
EIMPLS := eigen

# Define functions to construct object filenames from the datatypes and
# operations given an implementation. We define one function for single-
# threaded, single-socket, and dual-socket filenames.
get-st-objs = $(foreach dt,$(DTS),$(foreach op,$(OPS),test_$(dt)$(op)_$(PS_MAX)_$(1)_st.o))
get-1s-objs = $(foreach dt,$(DTS),$(foreach op,$(OPS),test_$(dt)$(op)_$(P1_MAX)_$(1)_1s.o))
get-2s-objs = $(foreach dt,$(DTS),$(foreach op,$(OPS),test_$(dt)$(op)_$(P2_MAX)_$(1)_2s.o))

# Construct object and binary names for single-threaded, single-socket, and
# dual-socket files for BLIS, OpenBLAS, and a vendor library (e.g. MKL).
BLIS_1M_ST_OBJS := $(call get-st-objs,1m_blis)
BLIS_1M_ST_BINS := $(patsubst %.o,%.x,$(BLIS_1M_ST_OBJS))
BLIS_1M_1S_OBJS := $(call get-1s-objs,1m_blis)
BLIS_1M_1S_BINS := $(patsubst %.o,%.x,$(BLIS_1M_1S_OBJS))
BLIS_1M_2S_OBJS := $(call get-2s-objs,1m_blis)
BLIS_1M_2S_BINS := $(patsubst %.o,%.x,$(BLIS_1M_2S_OBJS))

BLIS_NAT_ST_OBJS := $(call get-st-objs,asm_blis)
BLIS_NAT_ST_BINS := $(patsubst %.o,%.x,$(BLIS_NAT_ST_OBJS))
BLIS_NAT_1S_OBJS := $(call get-1s-objs,asm_blis)
BLIS_NAT_1S_BINS := $(patsubst %.o,%.x,$(BLIS_NAT_1S_OBJS))
BLIS_NAT_2S_OBJS := $(call get-2s-objs,asm_blis)
BLIS_NAT_2S_BINS := $(patsubst %.o,%.x,$(BLIS_NAT_2S_OBJS))

OPENBLAS_ST_OBJS := $(call get-st-objs,openblas)
OPENBLAS_ST_BINS := $(patsubst %.o,%.x,$(OPENBLAS_ST_OBJS))
OPENBLAS_1S_OBJS := $(call get-1s-objs,openblas)
OPENBLAS_1S_BINS := $(patsubst %.o,%.x,$(OPENBLAS_1S_OBJS))
OPENBLAS_2S_OBJS := $(call get-2s-objs,openblas)
OPENBLAS_2S_BINS := $(patsubst %.o,%.x,$(OPENBLAS_2S_OBJS))

EIGEN_ST_OBJS    := $(call get-st-objs,eigen)
EIGEN_ST_BINS    := $(patsubst %.o,%.x,$(EIGEN_ST_OBJS))
EIGEN_1S_OBJS    := $(call get-1s-objs,eigen)
EIGEN_1S_BINS    := $(patsubst %.o,%.x,$(EIGEN_1S_OBJS))
EIGEN_2S_OBJS    := $(call get-2s-objs,eigen)
EIGEN_2S_BINS    := $(patsubst %.o,%.x,$(EIGEN_2S_OBJS))

VENDOR_ST_OBJS   := $(call get-st-objs,vendor)
VENDOR_ST_BINS   := $(patsubst %.o,%.x,$(VENDOR_ST_OBJS))
VENDOR_1S_OBJS   := $(call get-1s-objs,vendor)
VENDOR_1S_BINS   := $(patsubst %.o,%.x,$(VENDOR_1S_OBJS))
VENDOR_2S_OBJS   := $(call get-2s-objs,vendor)
VENDOR_2S_BINS   := $(patsubst %.o,%.x,$(VENDOR_2S_OBJS))

# Define some targets associated with the above object/binary files.
blis-nat-st: $(BLIS_NAT_ST_BINS)
blis-nat-1s: $(BLIS_NAT_1S_BINS)
blis-nat-2s: $(BLIS_NAT_2S_BINS)

blis-1m-st: $(BLIS_1M_ST_BINS)
blis-1m-1s: $(BLIS_1M_1S_BINS)
blis-1m-2s: $(BLIS_1M_2S_BINS)

openblas-st: $(OPENBLAS_ST_BINS)
openblas-1s: $(OPENBLAS_1S_BINS)
openblas-2s: $(OPENBLAS_2S_BINS)

eigen-st: $(EIGEN_ST_BINS)
eigen-1s: $(EIGEN_1S_BINS)
eigen-2s: $(EIGEN_2S_BINS)

vendor-st: $(VENDOR_ST_BINS)
vendor-1s: $(VENDOR_1S_BINS)
vendor-2s: $(VENDOR_2S_BINS)

mkl-st: vendor-st
mkl-1s: vendor-1s
mkl-2s: vendor-2s

armpl-st: vendor-st
armpl-1s: vendor-1s
armpl-2s: vendor-2s

# Mark the object files as intermediate so that make will remove them
# automatically after building the binaries on which they depend.
.INTERMEDIATE: $(BLIS_NAT_ST_OBJS) $(BLIS_NAT_1S_OBJS) $(BLIS_NAT_2S_OBJS)
.INTERMEDIATE: $(BLIS_1M_ST_OBJS)  $(BLIS_1M_1S_OBJS)  $(BLIS_1M_2S_OBJS)
.INTERMEDIATE: $(OPENBLAS_ST_OBJS) $(OPENBLAS_1S_OBJS) $(OPENBLAS_2S_OBJS)
.INTERMEDIATE: $(EIGEN_ST_OBJS)    $(EIGEN_1S_OBJS)    $(EIGEN_2S_OBJS)
.INTERMEDIATE: $(VENDOR_ST_OBJS)   $(VENDOR_1S_OBJS)   $(VENDOR_2S_OBJS)


# --Object file rules --

#$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
#	$(CC) $(CFLAGS) -c $< -o $@

# A function to return the datatype cpp macro def from the datatype
# character.
get-dt-cpp = $(strip \
             $(if $(findstring s,$(1)),-DDT=BLIS_FLOAT    -DIS_FLOAT,\
             $(if $(findstring d,$(1)),-DDT=BLIS_DOUBLE   -DIS_DOUBLE,\
             $(if $(findstring c,$(1)),-DDT=BLIS_SCOMPLEX -DIS_SCOMPLEX,\
                                       -DDT=BLIS_DCOMPLEX -DIS_DCOMPLEX))))

get-in-cpp = $(strip \
             $(if $(findstring   1m_blis,$(1)),-DIND=BLIS_1M,\
                                               -DIND=BLIS_NAT))

# A function to return other cpp macros that help the test driver
# identify the implementation.
#get-bl-cpp = $(strip \
#             $(if $(findstring     blis,$(1)),$(STR_NAT) $(BLI_DEF),\
#             $(if $(findstring openblas,$(1)),$(STR_OBL) $(BLA_DEF),\
#             $(if $(findstring    eigen,$(1)),$(STR_EIG) $(EIG_DEF),\
#                                              $(STR_VEN) $(BLA_DEF)))))

get-bl-cpp = $(strip \
             $(if $(findstring   1m_blis,$(1)),$(STR_1M) $(BLI_DEF),\
             $(if $(findstring  asm_blis,$(1)),$(STR_NAT) $(BLI_DEF),\
             $(if $(findstring  openblas,$(1)),$(STR_OBL) $(BLA_DEF),\
             $(if $(and $(findstring eigen,$(1)),\
                        $(findstring  gemm,$(2))),\
                                              $(STR_EIG) $(EIG_DEF),\
             $(if       $(findstring eigen,$(1)),\
                                              $(STR_EIG) $(BLA_DEF),\
                                              $(STR_VEN) $(BLA_DEF)))))))


# Rules for BLIS and BLAS libraries.
define make-st-rule
test_$(1)$(2)_$(PS_MAX)_$(3)_st.o: test_$(op).c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(call get-in-cpp,$(3)) $(STR_ST) -c $$< -o $$@
endef

define make-1s-rule
test_$(1)$(2)_$(P1_MAX)_$(3)_1s.o: test_$(op).c Makefile
	$(CC) $(CFLAGS) $(PDEF_1S) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(call get-in-cpp,$(3)) $(STR_1S) -c $$< -o $$@
endef

define make-2s-rule
test_$(1)$(2)_$(P2_MAX)_$(3)_2s.o: test_$(op).c Makefile
	$(CC) $(CFLAGS) $(PDEF_2S) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(call get-in-cpp,$(3)) $(STR_2S) -c $$< -o $$@
endef

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(BIMPLS),$(eval $(call make-st-rule,$(dt),$(op),$(im))))))

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(BIMPLS),$(eval $(call make-1s-rule,$(dt),$(op),$(im))))))

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(BIMPLS),$(eval $(call make-2s-rule,$(dt),$(op),$(im))))))

# Rules for Eigen.
define make-eigst-rule
test_$(1)$(2)_$(PS_MAX)_$(3)_st.o: test_$(op).c Makefile
	$(CXX) $(CXXFLAGS_ST) $(PDEF_ST) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(DNAT) $(STR_ST) -c $$< -o $$@
endef

define make-eig1s-rule
test_$(1)$(2)_$(P1_MAX)_$(3)_1s.o: test_$(op).c Makefile
	$(CXX) $(CXXFLAGS_MT) $(PDEF_1S) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(DNAT) $(STR_1S) -c $$< -o $$@
endef

define make-eig2s-rule
test_$(1)$(2)_$(P2_MAX)_$(3)_2s.o: test_$(op).c Makefile
	$(CXX) $(CXXFLAGS_MT) $(PDEF_2S) $(call get-dt-cpp,$(1)) $(call get-bl-cpp,$(3),$(2)) $(DNAT) $(STR_2S) -c $$< -o $$@
endef

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(EIMPLS),$(eval $(call make-eigst-rule,$(dt),$(op),$(im))))))

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(EIMPLS),$(eval $(call make-eig1s-rule,$(dt),$(op),$(im))))))

$(foreach dt,$(DTS), \
$(foreach op,$(OPS), \
$(foreach im,$(EIMPLS),$(eval $(call make-eig2s-rule,$(dt),$(op),$(im))))))


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_$(PS_MAX)_1m_blis_st.x: test_%_$(PS_MAX)_1m_blis_st.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P1_MAX)_1m_blis_1s.x: test_%_$(P1_MAX)_1m_blis_1s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P2_MAX)_1m_blis_2s.x: test_%_$(P2_MAX)_1m_blis_2s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


test_%_$(PS_MAX)_asm_blis_st.x: test_%_$(PS_MAX)_asm_blis_st.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P1_MAX)_asm_blis_1s.x: test_%_$(P1_MAX)_asm_blis_1s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P2_MAX)_asm_blis_2s.x: test_%_$(P2_MAX)_asm_blis_2s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<                    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


test_%_$(PS_MAX)_openblas_st.x: test_%_$(PS_MAX)_openblas_st.o $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(OPENBLAS_LIB)  $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P1_MAX)_openblas_1s.x: test_%_$(P1_MAX)_openblas_1s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(OPENBLASP_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P2_MAX)_openblas_2s.x: test_%_$(P2_MAX)_openblas_2s.o $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(OPENBLASP_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


test_%_$(PS_MAX)_eigen_st.x:    test_%_$(PS_MAX)_eigen_st.o    $(LIBBLIS_LINK)
	$(CXX) $(strip $<  $(EIGEN_LIB)     $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P1_MAX)_eigen_1s.x:    test_%_$(P1_MAX)_eigen_1s.o    $(LIBBLIS_LINK)
	$(CXX) $(strip $<  $(EIGENP_LIB)    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P2_MAX)_eigen_2s.x:    test_%_$(P2_MAX)_eigen_2s.o    $(LIBBLIS_LINK)
	$(CXX) $(strip $<  $(EIGENP_LIB)    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


test_%_$(PS_MAX)_vendor_st.x:   test_%_$(PS_MAX)_vendor_st.o   $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(VENDOR_LIB)    $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P1_MAX)_vendor_1s.x:   test_%_$(P1_MAX)_vendor_1s.o   $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(VENDORP_LIB)   $(LIBBLIS_LINK) $(LDFLAGS) -o $@)

test_%_$(P2_MAX)_vendor_2s.x:   test_%_$(P2_MAX)_vendor_2s.o   $(LIBBLIS_LINK)
	$(CC) $(strip $<   $(VENDORP_LIB)   $(LIBBLIS_LINK) $(LDFLAGS) -o $@)


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

