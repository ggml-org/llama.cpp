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
# Makefile for BLIS typed API example code.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all bin clean run



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

# Detect whether we actually got the configuration file. If we didn't, then
# we may be building from an installed version and need BLIS_INSTALL_PATH.
ifeq ($(strip $(COMMON_MK_INCLUDED)),yes)
COMMON_MK_PRESENT := yes
else
COMMON_MK_PRESENT := no
$(error common.mk not found, you may need to specify BLIS_INSTALL_PATH)
endif



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

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Binary executable name.
TEST_BINS      := 00level1v.x \
                  01level1m.x \
                  02level1m_diag.x \
				  03level2.x \
				  04level3.x \
				  05util.x



#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: bin

bin: $(TEST_BINS)


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

%.x: %.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(LIBBLIS_LINK) $(LDFLAGS)'"
	@$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@
endif

# -- Test run rules --

#run: $(TEST_BIN)
#	./$(TEST_BIN)

# -- Clean rules --

clean:
	- $(RM_F) $(TEST_OBJS) $(TEST_BINS)

