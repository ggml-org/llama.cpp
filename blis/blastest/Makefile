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
# Makefile for BLAS test drivers.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all f2c bin \
        clean cleanf2c cleanobj cleanbin cleanout \
        check-env check-env-mk check-env-fragments check-env-make-defs \
        run check



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

TEST_OBJ_PATH  := obj

F2C_LIB        := libf2c.a

F2C_PATH       := f2c
DRIVER_PATH    := src
BLIS_H_PATH    := $(BUILD_PATH)/$(BASE_INC_PATH)

INPUT_DIR      := input

# Gather all local object files.
F2C_OBJS       := $(sort $(patsubst $(F2C_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(F2C_PATH)/*.c)))

DRIVER_OBJS    := $(sort $(patsubst $(DRIVER_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(DRIVER_PATH)/*.c)))

# Extract base names for each test driver file.
DRIVER_BASES   := $(basename $(notdir $(DRIVER_OBJS)))

# Binary executable names.
DRIVER_BINS    := $(addsuffix .x,$(DRIVER_BASES))

# Binary run-rule names
DRIVER_BINS_R  := $(addprefix run-,$(DRIVER_BASES))

# Filter level-1, level-2, and level-3 names to different variables.
DRIVER1_BASES  := $(filter %1,$(DRIVER_BASES))
DRIVER2_BASES  := $(filter %2,$(DRIVER_BASES))
DRIVER3_BASES  := $(filter %3,$(DRIVER_BASES))

# The location of the script that checks the BLAS test output.
#BLASTEST_CHECK := $(DIST_PATH)/$(BUILD_DIR)/check-blastest.sh

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Suppress warnings about uninitialized functions, add local header
# paths and the path to blis.h to CFLAGS.
CFLAGS         += -Wno-maybe-uninitialized -Wno-parentheses -Wfatal-errors \
                  -I$(F2C_PATH) \
                  -I$(INC_PATH) -DHAVE_BLIS_H

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Override the location of the check-blastest.sh script.
#BLASTEST_CHECK := ./check-blastest.sh

TESTSUITE_WRAPPER       ?=

#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: check-env f2c bin

f2c: check-env $(F2C_LIB)

bin: check-env $(DRIVER_BINS)


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(F2C_PATH)/%.c
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(CFLAGS) -c $< -o $@
endif

$(TEST_OBJ_PATH)/%.o: $(DRIVER_PATH)/%.c
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(CFLAGS) -c $< -o $@
endif


# -- libf2c library archive rule --

$(F2C_LIB): $(F2C_OBJS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif


# -- Executable file rules --

# first argument: the base name of the BLAS test driver.
define make-blat-rule
$(1).x: $(TEST_OBJ_PATH)/$(1).o $(F2C_LIB) $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $(TEST_OBJ_PATH)/$(1).o $(F2C_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $$@
else
	@echo "Linking $$@ against '$(F2C_LIB) $(LIBBLIS_LINK) $(LDFLAGS)'"
	@$(LINKER) $(TEST_OBJ_PATH)/$(1).o $(F2C_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $$@
endif
endef

# Instantiate the rule above for each driver file.
$(foreach name, $(DRIVER_BASES), $(eval $(call make-blat-rule,$(name))))


# -- Test run rules --

run: $(DRIVER_BINS_R)

# A rule to run ?blat1.x driver files.
define make-run-blat1-rule
run-$(1): $(1).x
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(1).x > out.$(1)
else
	@echo "Running $(1).x > 'out.$(1)'"
	@$(TESTSUITE_WRAPPER) ./$(1).x > out.$(1)
endif
endef

# Instantiate the rule above for each level-1 driver file.
$(foreach name, $(DRIVER1_BASES), $(eval $(call make-run-blat1-rule,$(name))))

# A rule to run ?blat2.x and ?blat3.x driver files.
define make-run-blat23-rule
run-$(1): $(1).x
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(1).x < $(INPUT_DIR)/$(1).in
else
	@echo "Running $(1).x < '$(INPUT_DIR)/$(1).in' (output to 'out.$(1)')"
	@$(TESTSUITE_WRAPPER) ./$(1).x < $(INPUT_DIR)/$(1).in
endif
endef

# Instantiate the rule above for each level-2 driver file.
$(foreach name, $(DRIVER2_BASES), $(eval $(call make-run-blat23-rule,$(name))))

# Instantiate the rule above for each level-3 driver file.
$(foreach name, $(DRIVER3_BASES), $(eval $(call make-run-blat23-rule,$(name))))

check: run
ifeq ($(ENABLE_VERBOSE),yes)
	- $(BLASTEST_CHECK)
else
	@- $(BLASTEST_CHECK)
endif


# -- Clean rules --

cleanf2c:
	- $(RM_F) $(F2C_OBJS) $(F2C_LIB)

cleanobj:
	- $(RM_F) $(DRIVER_OBJS)

cleanbin:
	- $(RM_F) $(DRIVER_BINS)

cleanout:
	- $(RM_F) $(addprefix out.,$(DRIVER_BASES))

clean: cleanf2c cleanobj cleanbin cleanout

