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

.PHONY: all all-st all-mt \
        blis blis-st blis-mt \
        blis-nat blis-nat-st blis-nat-mt \
        openblas openblas-st openblas-mt \
        mkl mkl-st mkl-mt \
        blis-gemm-st blis-gemm-mt \
        blis-gemm-nat-st blis-gemm-nat-mt \
        openblas-gemm-st openblas-gemm-mt \
        mkl-gemm-st mkl-gemm-mt \
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
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)


# Which library?
BLI_DEF  := -DBLIS
BLA_DEF  := -DBLAS

# Single or multithreaded string
STR_ST   := -DTHR_STR=\"st\"
STR_MT   := -DTHR_STR=\"mt\"

# Problem size specification
PDEF_ST  := -DP_BEGIN=40 \
            -DP_MAX=2000 \
            -DP_INC=40

PDEF_MT  := -DP_BEGIN=160 \
            -DP_MAX=8000 \
            -DP_INC=160

# Enumerate possible datatypes and computation precisions.
dts := s d c z
prs := s d

# Various functions that help us construct the datatype combinations and then
# extract the needed datatype strings and C preprocessor define flags.
get-char-c = $(word 1,$(subst _, ,$(1)))
get-char-a = $(word 2,$(subst _, ,$(1)))
get-char-b = $(word 3,$(subst _, ,$(1)))
get-char-x = $(word 4,$(subst _, ,$(1)))
get-cstr   = $(call get-char-c,$(1))$(call get-char-a,$(1))$(call get-char-b,$(1))$(call get-char-x,$(1))

get-cdef-a = $(strip $(subst s,-DDTA=BLIS_FLOAT, \
                     $(subst d,-DDTA=BLIS_DOUBLE, \
                     $(subst c,-DDTA=BLIS_SCOMPLEX, \
                     $(subst z,-DDTA=BLIS_DCOMPLEX,$(call get-char-a,$(1)))))))
get-cdef-b = $(strip $(subst s,-DDTB=BLIS_FLOAT, \
                     $(subst d,-DDTB=BLIS_DOUBLE, \
                     $(subst c,-DDTB=BLIS_SCOMPLEX, \
                     $(subst z,-DDTB=BLIS_DCOMPLEX,$(call get-char-b,$(1)))))))
get-cdef-c = $(strip $(subst s,-DDTC=BLIS_FLOAT, \
                     $(subst d,-DDTC=BLIS_DOUBLE, \
                     $(subst c,-DDTC=BLIS_SCOMPLEX, \
                     $(subst z,-DDTC=BLIS_DCOMPLEX,$(call get-char-c,$(1)))))))
get-cdef-x = $(strip $(subst s,-DDTX=BLIS_FLOAT, \
                     $(subst d,-DDTX=BLIS_DOUBLE,$(call get-char-x,$(1)))))
get-cdefs  = $(call get-cdef-c,$(1)) $(call get-cdef-a,$(1)) $(call get-cdef-b,$(1)) $(call get-cdef-x,$(1))

# Define a function to return the appropriate -DSTR= and -D[BLIS|BLAS] flags.
get-idefs = $(strip $(subst intern,-DSTR=\"$(1)\" -DBLIS, \
                    $(subst ad_hoc,-DSTR=\"$(1)\" -DBLAS, \
                    $(subst    mkl,-DSTR=\"$(1)\" -DBLAS,$(1)))))

# Enumerate all possible datatype combinations.
DT_CODES := $(foreach dt0,$(dts),$(foreach dt1,$(dts),$(foreach dt2,$(dts),$(foreach pr,$(prs),$(dt0)_$(dt1)_$(dt2)_$(pr)))))

# Build a list of the datatype strings.
DT_COMBOS := $(foreach code,$(DT_CODES),$(call get-cstr,$(code)))

# Build a list of BLIS, OpenBLAS, and MKL executables.
INTERN_OBJS_ST := $(foreach combo,$(DT_COMBOS),test_$(combo)gemm_intern_st.o)
INTERN_BINS_ST := $(patsubst %.o,%.x,$(INTERN_OBJS_ST))
AD_HOC_OBJS_ST := $(foreach combo,$(DT_COMBOS),test_$(combo)gemm_ad_hoc_st.o)
AD_HOC_BINS_ST := $(patsubst %.o,%.x,$(AD_HOC_OBJS_ST))

INTERN_OBJS_MT := $(foreach combo,$(DT_COMBOS),test_$(combo)gemm_intern_mt.o)
INTERN_BINS_MT := $(patsubst %.o,%.x,$(INTERN_OBJS_MT))
AD_HOC_OBJS_MT := $(foreach combo,$(DT_COMBOS),test_$(combo)gemm_ad_hoc_mt.o)
AD_HOC_BINS_MT := $(patsubst %.o,%.x,$(AD_HOC_OBJS_MT))



#
# --- Targets/rules ------------------------------------------------------------
#

all:       st

st:        intern-st ad_hoc-st
mt:        intern-mt ad_hoc-mt

intern-st: $(INTERN_BINS_ST)
ad_hoc-st: $(AD_HOC_BINS_ST)
intern-mt: $(INTERN_BINS_MT)
ad_hoc-mt: $(AD_HOC_BINS_MT)

#blis:        test_ssssgemm_asm_blis_st.x \
#             test_sssdgemm_asm_blis_st.x \
#             test_ssdsgemm_asm_blis_st.x \
#             test_sdssgemm_asm_blis_st.x \
#             test_dsssgemm_asm_blis_st.x \
#             test_dddsgemm_asm_blis_st.x \
#             test_ddddgemm_asm_blis_st.x


# --Object file rules --

# Define the function that will be used to instantiate compilation rules
# for the various implementations.
define make-st-rule
test_$(call get-cstr,$(1))gemm_$(2)_st.o: test_gemm.c Makefile
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) $(PDEF_ST) $(call get-cdefs,$(1)) $(call get-idefs,$(2)) $(STR_ST) -c $$< -o $$@
else
	@echo "Compiling $$@"
	@$(CC) $(CFLAGS) $(PDEF_ST) $(call get-cdefs,$(1)) $(call get-idefs,$(2)) $(STR_ST) -c $$< -o $$@
endif
endef

define make-mt-rule
test_$(call get-cstr,$(1))gemm_$(2)_mt.o: test_gemm.c Makefile
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) $(PDEF_MT) $(call get-cdefs,$(1)) $(call get-idefs,$(2)) $(STR_MT) -c $$< -o $$@
else
	@echo "Compiling $$@"
	@$(CC) $(CFLAGS) $(PDEF_MT) $(call get-cdefs,$(1)) $(call get-idefs,$(2)) $(STR_MT) -c $$< -o $$@
endif
endef


# Define the implementations for which we will instantiate compilation rules.
IMPLS := intern ad_hoc

# Instantiate the rule function make-st-rule() and make-mt-rule for each
# implementation in IMPLS and each of the datatype "codes" in DT_CODES.
$(foreach impl,$(IMPLS), \
$(foreach code,$(DT_CODES),$(eval $(call make-st-rule,$(code),$(impl)))))

$(foreach impl,$(IMPLS), \
$(foreach code,$(DT_CODES),$(eval $(call make-mt-rule,$(code),$(impl)))))


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_ad_hoc_st.x: test_%_ad_hoc_st.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER)  $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	$(RM_F) $<
else
	@@echo "Linking $@ to '$(LIBBLIS_LINK)'"
	@$(LINKER) $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	@$(RM_F) $<
endif

test_%_ad_hoc_mt.x: test_%_ad_hoc_mt.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER)  $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	$(RM_F) $<
else
	@@echo "Linking $@ to '$(LIBBLIS_LINK)'"
	@$(LINKER) $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	@$(RM_F) $<
endif


test_%_intern_st.x: test_%_intern_st.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER)  $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	$(RM_F) $<
else
	@@echo "Linking $@ to '$(LIBBLIS_LINK)'"
	@$(LINKER) $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	@$(RM_F) $<
endif

test_%_intern_mt.x: test_%_intern_mt.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER)  $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	$(RM_F) $<
else
	@@echo "Linking $@ to '$(LIBBLIS_LINK)'"
	@$(LINKER) $<  $(LIBBLIS_LINK) $(LDFLAGS) -o $@
	@$(RM_F) $<
endif


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

cleanout:
	- $(RM_F) *.m


