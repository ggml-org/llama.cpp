#
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
# Makefile for BLIS testsuite.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all bin clean \
        check-env check-env-mk check-env-fragments check-env-make-defs \
        run run-amd64 run-x86 run-arm



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
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := src
TEST_OBJ_PATH  := obj

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

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Binary executable name.
TESTSUITE_BIN  := test_libblis.x



#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: check-env bin

bin: check-env $(TESTSUITE_BIN)


# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-config-mk

check-env-config-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
    $(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(MAKE_DEFS_MK_PRESENT),no)
    $(error Cannot proceed: make_defs.mk not detected! Invalid configuration)
endif


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(CFLAGS) -c $< -o $@
endif


# -- Executable file rules --

$(TESTSUITE_BIN): $(TEST_OBJS) $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $(TEST_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(LIBBLIS_LINK) $(LDFLAGS)'"
	@$(LINKER) $(TEST_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
endif


# -- Test run/check rules --

run: $(TESTSUITE_BIN)
ifeq ($(ENABLE_VERBOSE),yes)
	./$(TESTSUITE_BIN) > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@./$(TESTSUITE_BIN) > $(TESTSUITE_OUT_FILE)
endif

run-fast: $(TESTSUITE_BIN)
ifeq ($(ENABLE_VERBOSE),yes)
	./$(TESTSUITE_BIN) -g $(TESTSUITE_FAST_GEN) -o $(TESTSUITE_FAST_OPS) > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) (fast) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@./$(TESTSUITE_BIN) -g $(TESTSUITE_FAST_GEN) -o $(TESTSUITE_FAST_OPS) > $(TESTSUITE_OUT_FILE)
endif

check: run
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK) $(TESTSUITE_OUT_FILE)
endif

check-fast: run-fast
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK) $(TESTSUITE_OUT_FILE)
endif


# -- Clean rules --

clean:
	- $(RM_F) $(TEST_OBJS) $(TESTSUITE_BIN)

