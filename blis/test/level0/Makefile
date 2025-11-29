#!/bin/bash
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2025, Southern Methodist University
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
        test-ranges \
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
DIST_PATH   := ../..
CONFIG_NAME := $(shell grep -E "CONFIG_NAME *:=" $(DIST_PATH)/config.mk | sed 's/.*:= *//')
LIB_PATH     = $(DIST_PATH)/lib/$(CONFIG_NAME)
INC_PATH     = $(DIST_PATH)/include/$(CONFIG_NAME)
SHARE_PATH  := $(DIST_PATH)
endif


#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk


#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
SRC_SUFFIXES   := c cxx cpp
TEST_OBJS      := $(foreach suf, \
                            $(SRC_SUFFIXES), \
                            $(sort $(patsubst $(TEST_SRC_PATH)/%.$(suf), \
                                              $(TEST_OBJ_PATH)/%.o, \
                                              $(wildcard $(TEST_SRC_PATH)/*.$(suf)))))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)
CXXINCFLAGS    := -I$(INC_PATH)
CXXLANGFLAGS   := -std=c++17

# Use the CFLAGS for the configuration family.
CFLAGS         := $(call get-frame-cflags-for,$(CONFIG_NAME))
CXXFLAGS       := $(call get-frame-cxxflags-for,$(CONFIG_NAME))

# Add installed and local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)
CXXFLAGS       += -I$(TEST_SRC_PATH) -DENABLE_INFO

HDR_SUFFIXES   := h hpp
HEADERS        := $(foreach suf, $(HDR_SUFFIXES), $(wildcard $(TEST_SRC_PATH)/*.$(suf))) $(INC_PATH)/blis.h


#
# --- Targets/rules ------------------------------------------------------------
#

all: test

test: test_l0.x


# -- Source file rules --

test_%.o: test_%.c Makefile $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

test_%.o: test_%.cpp Makefile $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_%.o: test_%.cxx Makefile $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_l0.x: $(TEST_OBJS) $(LIBBLIS_LINK)
	$(CXX) $(TEST_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

