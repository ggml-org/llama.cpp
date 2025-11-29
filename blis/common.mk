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

# Only include this block of code once
ifndef COMMON_MK_INCLUDED
COMMON_MK_INCLUDED := yes



#
# --- CFLAGS storage functions -------------------------------------------------
#

# Define a function that stores the value of a variable to a different
# variable containing a specified suffix (corresponding to a configuration).
define store-var-for
$(strip $(1)).$(strip $(2)) := $($(strip $(1)))
endef

# Define a function similar to store-var-for, except that appends instead
# of overwriting.
define append-var-for
$(strip $(1)).$(strip $(2)) += $($(strip $(1)))
endef

# Define a function that stores the value of all of the variables in a
# make_defs.mk file to other variables with the configuration (the
# argument $(1)) added as a suffix. This function is called once from
# each make_defs.mk. Also, add the configuration to CONFIGS_INCL.
define store-make-defs
$(eval $(call store-var-for,CC,         $(1)))
$(eval $(call store-var-for,CC_VENDOR,  $(1)))
$(eval $(call store-var-for,CPPROCFLAGS,$(1)))
$(eval $(call store-var-for,CLANGFLAGS, $(1)))
$(eval $(call store-var-for,CMISCFLAGS, $(1)))
$(eval $(call store-var-for,CPICFLAGS,  $(1)))
$(eval $(call store-var-for,CWARNFLAGS, $(1)))
$(eval $(call store-var-for,CDBGFLAGS,  $(1)))
$(eval $(call store-var-for,COPTFLAGS,  $(1)))
$(eval $(call store-var-for,CKOPTFLAGS, $(1)))
$(eval $(call store-var-for,CKVECFLAGS, $(1)))
$(eval $(call store-var-for,CROPTFLAGS, $(1)))
$(eval $(call store-var-for,CRVECFLAGS, $(1)))
CONFIGS_INCL += $(1)
endef

# Define a function that retreives the value of a variable for a
# given configuration.
define load-var-for
$($(strip $(1)).$(strip $(2)))
endef



#
# --- CFLAGS query functions ---------------------------------------------------
#

# Define some functions that return the appropriate CFLAGS for a given
# configuration. This assumes that the make_defs.mk files have already been
# included, which results in those values having been stored to
# configuration-qualified variables.

get-noopt-cflags-for     = $(strip $(CFLAGS_PRESET) \
                                   $(call load-var-for,CDBGFLAGS,$(1)) \
                                   $(call load-var-for,CWARNFLAGS,$(1)) \
                                   $(call load-var-for,CPICFLAGS,$(1)) \
                                   $(call load-var-for,CMISCFLAGS,$(1)) \
                                   $(call load-var-for,CLANGFLAGS,$(1)) \
                                   $(call load-var-for,CPPROCFLAGS,$(1)) \
                                   $(CTHREADFLAGS) \
                                   $(CINCFLAGS) \
                            )

get-noopt-cxxflags-for   = $(strip $(CXXFLAGS_PRESET) \
                                   $(call load-var-for,CDBGFLAGS,$(1)) \
                                   $(call load-var-for,CWARNFLAGS,$(1)) \
                                   $(call load-var-for,CPICFLAGS,$(1)) \
                                   $(call load-var-for,CMISCFLAGS,$(1)) \
                                   $(call load-var-for,CPPROCFLAGS,$(1)) \
                                   $(CXXLANGFLAGS) \
                                   $(CTHREADFLAGS) \
                                   $(CXXTHREADFLAGS) \
                                   $(CINCFLAGS) \
                            )

get-refinit-cflags-for   = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                                   -DBLIS_IN_REF_KERNEL=1 \
                                   -include $(CONFIG_PATH)/$(1)/bli_kernel_defs_$(1).h \
                            )

get-refinit-cxxflags-for = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                                   -DBLIS_IN_REF_KERNEL=1 \
                                   -include $(CONFIG_PATH)/$(1)/bli_kernel_defs_$(1).h \
                            )

get-refkern-cflags-for   = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,CROPTFLAGS,$(1)) \
                                   $(call load-var-for,CRVECFLAGS,$(1)) \
                                   $(COMPSIMDFLAGS) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                                   -DBLIS_IN_REF_KERNEL=1 \
                                   -include $(CONFIG_PATH)/$(1)/bli_kernel_defs_$(1).h \
                            )

get-refkern-cxxflags-for = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,CROPTFLAGS,$(1)) \
                                   $(call load-var-for,CRVECFLAGS,$(1)) \
                                   $(COMPSIMDFLAGS) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                                   -DBLIS_IN_REF_KERNEL=1 \
                                   -include $(CONFIG_PATH)/$(1)/bli_kernel_defs_$(1).h \
                            )

get-config-cflags-for    = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

get-config-cxxflags-for  = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

get-frame-cflags-for     = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

get-frame-cxxflags-for   = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(BUILD_ASANFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

get-kernel-cflags-for    = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,CKOPTFLAGS,$(1)) \
                                   $(call load-var-for,CKVECFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

get-kernel-cxxflags-for  = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,CKOPTFLAGS,$(1)) \
                                   $(call load-var-for,CKVECFLAGS,$(1)) \
                                   -DBLIS_CNAME=$(1) \
                                   -DBLIS_CNAME_UPPER=$(shell echo $(1) | tr a-z A-Z) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

# When compiling addons, we use flags similar to those of general framework
# source. This ensures that the same code can be linked and run across various
# sub-configurations.
get-addon-c99flags-for   = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(CADDONINCFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )
get-addon-cxxflags-for   = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(CADDONINCFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )
# When compiling addon kernels, we use flags similar to those of kernels
# flags, except we also include the addon header paths.
get-addon-kernel-c99flags-for = $(strip $(call get-noopt-cflags-for,$(1)) \
                                        $(call load-var-for,CKOPTFLAGS,$(1)) \
                                        $(call load-var-for,CKVECFLAGS,$(1)) \
                                        $(CADDONINCFLAGS) \
                                        $(BUILD_CPPFLAGS) \
                                        $(BUILD_SYMFLAGS) \
                                 )

# When compiling sandboxes, we use flags similar to those of general framework
# source. This ensures that the same code can be linked and run across various
# sub-configurations. (NOTE: If we ever switch to using refkernel or kernel
# flags, we should prevent enabling sandboxes for umbrella families by verifying
# that config_list == config_name if --enable-sandbox is given. THIS ALSO
# APPLIES TO ADDONS ABOVE.)
get-sandbox-c99flags-for = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(CSANDINCFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )
get-sandbox-cxxflags-for = $(strip $(call get-noopt-cxxflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                                   $(CSANDINCFLAGS) \
                                   $(BUILD_CPPFLAGS) \
                                   $(BUILD_SYMFLAGS) \
                            )

# Define a separate function that will return appropriate flags for use by
# applications that want to use the same basic flags as those used when BLIS
# was compiled. (NOTE: This is the same as the $(get-frame-cflags-for ...)
# function, except that it omits a few variables that contain flags exclusively
# for use when BLIS is being compiled/built:
# - BUILD_CPPFLAGS, which contains a cpp macro that confirms that BLIS
#   is being built;
# - BUILD_SYMFLAGS, which contains symbol export flags that are only
#   needed when a shared library is being compiled/linked; and
# - BUILD_ASANFLAGS, which contains a flag that causes the compiler to
#   insert instrumentation for memory error detection.
get-user-cflags-for      = $(strip $(call get-noopt-cflags-for,$(1)) \
                                   $(call load-var-for,COPTFLAGS,$(1)) \
                            )

# Define functions that return messages appropriate for each non-verbose line
# of compilation output.
get-noopt-text            = "(CFLAGS for no optimization)"
get-refinit-text-for      = "('$(1)' CFLAGS for ref. kernel init)"
get-refinit-cxxtext-for   = "('$(1)' CXXFLAGS for ref. kernel init)"
get-refkern-text-for      = "('$(1)' CFLAGS for ref. kernels)"
get-refkern-cxxtext-for   = "('$(1)' CXXFLAGS for ref. kernels)"
get-config-text-for       = "('$(1)' CFLAGS for config code)"
get-config-cxxtext-for    = "('$(1)' CXXFLAGS for config code)"
get-frame-text-for        = "('$(1)' CFLAGS for framework code)"
get-frame-cxxtext-for     = "('$(1)' CXXFLAGS for framework code)"
get-kernel-text-for       = "('$(1)' CFLAGS for kernels)"
get-kernel-cxxtext-for    = "('$(1)' CXXFLAGS for kernels)"
get-addon-c99text-for     = "('$(1)' CFLAGS for addons)"
get-addon-cxxtext-for     = "('$(1)' CXXFLAGS for addons)"
get-addon-kernel-text-for = "('$(1)' CFLAGS for addon kernels)"
get-sandbox-c99text-for   = "('$(1)' CFLAGS for sandboxes)"
get-sandbox-cxxtext-for   = "('$(1)' CXXFLAGS for sandboxes)"



#
# --- Miscellaneous helper functions -------------------------------------------
#

# Define functions that filters a list of filepaths $(1) that contain (or
# omit) an arbitrary substring $(2).
files-that-contain      = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),$(f),)))
files-that-dont-contain = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),,$(f))))

# Define a function that removes duplicate strings *without* using the sort
# function.
rm-dups = $(if $1,$(firstword $1) $(call rm-dups,$(filter-out $(firstword $1),$1)))



#
# --- Include makefile configuration file --------------------------------------
#

# Use the current directory as the default path to the root directory for
# makefile fragments (and the configuration family's make_defs.mk), but
# allow the includer to override this value if it needs to point to an
# installation directory.
ifeq ($(strip $(SHARE_PATH)),)
SHARE_PATH        := .
endif

# Define the name of the configuration file.
CONFIG_MK_FILE     := config.mk

# Identify the base path for the root directory for makefile fragments (and
# the configuration family's make_defs.mk). We define this path in terms of
# SHARE_PATH, which gets a default value above (which is what happens for the
# top-level Makefile). If SHARE_PATH is specified by the Makefile prior to
# including common.mk, that path is used instead. This allows Makefiles for
# example code and test drivers to reference an installed prefix directory
# for situations when the build directory no longer exists.
BASE_SHARE_PATH    := $(SHARE_PATH)

# Include the configuration file.
-include $(BASE_SHARE_PATH)/$(CONFIG_MK_FILE)



#
# --- Handle 'make clean' and friends without config.mk ------------------------
#

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
IS_CONFIGURED     := yes
else
CONFIG_MK_PRESENT := no
IS_CONFIGURED     := no
endif

# If we didn't get config.mk, then we need to set some basic variables so
# that make will function without error for things like 'make clean'.
ifeq ($(IS_CONFIGURED),no)

# If this makefile fragment is being run and there is no config.mk present,
# then it's probably safe to assume that the user is currently located in the
# source distribution.
DIST_PATH := .

# Even though they won't be used explicitly, it appears that setting these
# INSTALL_* variables to something sane (that is, not allowing them default
# to the empty string) is necessary to prevent make from hanging, likely
# because the statements that define UNINSTALL_LIBS and UNINSTALL_HEADERS,
# when evaluated, result in running 'find' on the root directory--definitely
# something we would like to avoid.
INSTALL_LIBDIR   := $(HOME)/blis/lib
INSTALL_INCDIR   := $(HOME)/blis/include
INSTALL_SHAREDIR := $(HOME)/blis/share

endif



#
# --- Primary makefile variable definitions ------------------------------------
#

# Construct the architecture-version string, which will be used to name the
# library upon installation.
VERS_CONF          := $(VERSION)-$(CONFIG_NAME)

# All makefile fragments in the tree will have this name.
FRAGMENT_MK        := .fragment.mk

# Locations of important files.
BUILD_DIR          := build
CONFIG_DIR         := config
ifeq ($(FRAME_DIR),)
FRAME_DIR          := frame
endif
REFKERN_DIR        := ref_kernels
KERNELS_DIR        := kernels
ADDON_DIR          := addon
SANDBOX_DIR        := sandbox
OBJ_DIR            := obj
LIB_DIR            := lib
INCLUDE_DIR        := include
BLASTEST_DIR       := blastest
TESTSUITE_DIR      := testsuite

VEND_DIR           := vendor
VEND_CPP_DIR       := $(VEND_DIR)/cpp
VEND_TESTCPP_DIR   := $(VEND_DIR)/testcpp

# The filename suffix for reference kernels.
REFNM              := ref

# Source suffixes.
CONFIG_SRC_SUFS    := c
KERNELS_SRC_SUFS   := c s S
ifneq ($(findstring hpx,$(THREADING_MODEL)),)
FRAME_SRC_SUFS     := c cpp
else
FRAME_SRC_SUFS     := c
endif

ADDON_C99_SUFS     := c
ADDON_CXX_SUFS     := cc cpp cxx
ADDON_SRC_SUFS     := $(ADDON_C99_SUFS) $(ADDON_CXX_SUFS)

SANDBOX_C99_SUFS   := c
SANDBOX_CXX_SUFS   := cc cpp cxx
SANDBOX_SRC_SUFS   := $(SANDBOX_C99_SUFS) $(SANDBOX_CXX_SUFS)

# Header suffixes.
FRAME_H99_SUFS     := h
FRAME_HDR_SUFS     := $(FRAME_H99_SUFS)

ADDON_H99_SUFS     := h
ADDON_HXX_SUFS     := hh hpp hxx
ADDON_HDR_SUFS     := $(ADDON_H99_SUFS) $(ADDON_HXX_SUFS)

SANDBOX_H99_SUFS   := h
SANDBOX_HXX_SUFS   := hh hpp hxx
SANDBOX_HDR_SUFS   := $(SANDBOX_H99_SUFS) $(SANDBOX_HXX_SUFS)

# Combine all header suffixes and remove duplicates via sort().
ALL_HDR_SUFS       := $(sort $(FRAME_HDR_SUFS) \
                             $(ADDON_HDR_SUFS) \
                             $(SANDBOX_HDR_SUFS) )

ALL_H99_SUFS       := $(sort $(FRAME_H99_SUFS) \
                             $(ADDON_H99_SUFS) \
                             $(SANDBOX_H99_SUFS) )

# The names of scripts that check output from the BLAS test drivers and
# BLIS test suite.
BLASTEST_CHECK     := check-blastest.sh
TESTSUITE_CHECK    := check-blistest.sh

# The names of the testsuite input/configuration files.
TESTSUITE_CONF_GEN := input.general
TESTSUITE_CONF_OPS := input.operations
TESTSUITE_FAST_GEN := input.general.fast
TESTSUITE_FAST_OPS := input.operations.fast
TESTSUITE_MIXD_GEN := input.general.mixed
TESTSUITE_MIXD_OPS := input.operations.mixed
TESTSUITE_SALT_GEN := input.general.salt
TESTSUITE_SALT_OPS := input.operations.salt
TESTSUITE_OUT_FILE := output.testsuite

# CHANGELOG file.
CHANGELOG          := CHANGELOG

# Something for OS X so that echo -n works as expected.
SHELL              := bash

# Construct paths to the four primary directories of source code:
# the config directory, general framework code, reference kernel code,
# and optimized kernel code. Also process paths for addon and sandbox
# directories.
CONFIG_PATH        := $(DIST_PATH)/$(CONFIG_DIR)
FRAME_PATH         := $(DIST_PATH)/$(FRAME_DIR)
REFKERN_PATH       := $(DIST_PATH)/$(REFKERN_DIR)
KERNELS_PATH       := $(DIST_PATH)/$(KERNELS_DIR)
ADDON_PATH         := $(DIST_PATH)/$(ADDON_DIR)
SANDBOX_PATH       := $(DIST_PATH)/$(SANDBOX_DIR)
BUILD_PATH         := $(DIST_PATH)/$(BUILD_DIR)

# Construct paths to some optional C++ template headers contributed by AMD.
VEND_CPP_PATH      := $(DIST_PATH)/$(VEND_CPP_DIR)
VEND_TESTCPP_PATH  := $(DIST_PATH)/$(VEND_TESTCPP_DIR)

# Construct paths to the makefile fragments for the four primary directories
# of source code: the config directory, general framework code, reference
# kernel code, and optimized kernel code.
CONFIG_FRAG_PATH   := ./obj/$(CONFIG_NAME)/$(CONFIG_DIR)
FRAME_FRAG_PATH    := ./obj/$(CONFIG_NAME)/$(FRAME_DIR)
REFKERN_FRAG_PATH  := ./obj/$(CONFIG_NAME)/$(REFKERN_DIR)
KERNELS_FRAG_PATH  := ./obj/$(CONFIG_NAME)/$(KERNELS_DIR)
ADDON_FRAG_PATH    := ./obj/$(CONFIG_NAME)/$(ADDON_DIR)
SANDBOX_FRAG_PATH  := ./obj/$(CONFIG_NAME)/$(SANDBOX_DIR)



#
# --- Library name and local paths ---------------------------------------------
#

# Use lib/CONFIG_NAME as the default path to the local header files, but
# allow the includer to override this value if it needs to point to an
# installation directory.
ifeq ($(strip $(LIB_PATH)),)
LIB_PATH           := $(LIB_DIR)/$(CONFIG_NAME)
endif

# Identify the base path for the intermediate library directory. We define
# this path in terms of LIB_PATH, which gets a default value above (which is
# what happens for the top-level Makefile). If LIB_PATH is specified by the
# Makefile prior to including common.mk, that path is used instead. This
# allows Makefiles for example code and test drivers to reference an installed
# prefix directory for situations when the build directory no longer exists.
BASE_LIB_PATH      := $(LIB_PATH)

# The base name of the BLIS library that we will build.
LIBBLIS            := libblis

# The shared (dynamic) library file suffix is different for Linux and OS X.
ifeq ($(OS_NAME),Darwin)
SHLIB_EXT          := dylib
else ifeq ($(IS_WIN),yes)
ifeq ($(IS_MSVC),no)
SHLIB_EXT          := dll.a
else
SHLIB_EXT          := lib
endif
else
SHLIB_EXT          := so
endif

# Note: These names will be modified later to include the configuration and
# version strings.
LIBBLIS_A          := $(LIBBLIS).a
LIBBLIS_SO         := $(LIBBLIS).$(SHLIB_EXT)

# Append the base library path to the library names.
LIBBLIS_A_PATH     := $(BASE_LIB_PATH)/$(LIBBLIS_A)
LIBBLIS_SO_PATH    := $(BASE_LIB_PATH)/$(LIBBLIS_SO)

# Create a filepath to a local symlink to the soname--that is, the same as
# LIBBLIS_SO_PATH except with the .so major version number. Since the shared
# library lists its soname as 'libblis.so.n', where n is the .so major version
# number, a symlink in BASE_LIB_PATH is needed so that ld can find the local
# shared library when the testsuite is run via 'make test' or 'make check'.

ifeq ($(OS_NAME),Darwin)
# OS X shared library extensions.
LIBBLIS_SO_MAJ_EXT := $(SO_MAJOR).$(SHLIB_EXT)
LIBBLIS_SO_MMB_EXT := $(SO_MMB).$(SHLIB_EXT)
else ifeq ($(IS_WIN),yes)
# Windows shared library extension.
LIBBLIS_SO_MAJ_EXT := $(SO_MAJOR).dll
LIBBLIS_SO_MMB_EXT :=
else
# Linux shared library extensions.
LIBBLIS_SO_MAJ_EXT := $(SHLIB_EXT).$(SO_MAJOR)
LIBBLIS_SO_MMB_EXT := $(SHLIB_EXT).$(SO_MMB)
endif
LIBBLIS_SONAME         := $(LIBBLIS).$(LIBBLIS_SO_MAJ_EXT)
LIBBLIS_SO_MAJ_PATH    := $(BASE_LIB_PATH)/$(LIBBLIS_SONAME)

# Construct the output path when building a shared library.
# NOTE: This code and the code immediately above is a little curious and
# perhaps could be refactored (carefully).
ifeq ($(IS_WIN),yes)
LIBBLIS_SO_OUTPUT_NAME := $(LIBBLIS_SO_MAJ_PATH)
else
LIBBLIS_SO_OUTPUT_NAME := $(LIBBLIS_SO_PATH)
endif



#
# --- Utility program definitions ----------------------------------------------
#

SH         := /bin/sh
MV         := mv
MKDIR      := mkdir -p
RM_F       := rm -f
RM_RF      := rm -rf
SYMLINK    := ln -sf
FIND       := find
GREP       := grep
EGREP      := grep -E
XARGS      := xargs
INSTALL    := install -c
DEVNULL    := /dev/null

# Script for creating a monolithic header file.
#FLATTEN_H  := $(DIST_PATH)/build/flatten-headers.sh
FLATTEN_H  := $(PYTHON) $(DIST_PATH)/build/flatten-headers.py

# Default archiver flags.
ARFLAGS    := cr

# Used to refresh CHANGELOG.
GIT        := git
GIT_LOG    := $(GIT) log --decorate

# Define the locations of a script to generate a list of shared library symbols
# within BLIS as well as the symbol file itself.
GEN_SYMS   := $(BUILD_PATH)/gen-libblis-symbols.sh
SYM_FILE   := $(BUILD_PATH)/libblis-symbols.def



#
# --- Default linker definitions -----------------------------------------------
#

# NOTE: This section needs to reside before the inclusion of make_defs.mk
# files (just below), as most configurations' make_defs.mk don't tinker
# with things like LDFLAGS, but some do (or may), in which case they can
# manually override whatever they need.

# Define the external libraries we may potentially need at link-time.
ifeq ($(IS_MSVC),yes)
LIBM       :=
else
LIBM       := -lm
endif
LIBMEMKIND := -lmemkind

# Default linker flags.
# NOTE: -lpthread is needed unconditionally because BLIS uses pthread_once()
# to initialize itself in a thread-safe manner. The one exception to this
# rule: if --disable-system is given at configure-time, LIBPTHREAD is empty.
LDFLAGS    := $(LDFLAGS_PRESET) $(LIBM) $(LIBPTHREAD)

# Add libmemkind to the link-time flags, if it was enabled at configure-time.
ifeq ($(MK_ENABLE_MEMKIND),yes)
LDFLAGS    += $(LIBMEMKIND)
endif

# Never use libm with Intel compilers.
ifeq ($(CC_VENDOR),icc)
LDFLAGS    := $(filter-out $(LIBM),$(LDFLAGS))
endif

# Never use libmemkind with Intel SDE.
ifeq ($(DEBUG_TYPE),sde)
LDFLAGS    := $(filter-out $(LIBMEMKIND),$(LDFLAGS))
endif

# If AddressSanitizer is enabled, add the compiler flag to LDFLAGS.
ifeq ($(MK_ENABLE_ASAN),yes)
LDFLAGS    += -fsanitize=address
endif

# Specify the shared library's 'soname' field.
# NOTE: The flag for creating shared objects is different for Linux and OS X.
ifeq ($(OS_NAME),Darwin)
# OS X shared library link flags.
SOFLAGS    := -dynamiclib
ifeq ($(MK_ENABLE_RPATH),yes)
SOFLAGS    += -Wl,-install_name,@rpath/$(LIBBLIS_SONAME)
else
SOFLAGS    += -Wl,-install_name,$(libdir)/$(LIBBLIS_SONAME)
endif
else
SOFLAGS    := -shared
ifeq ($(IS_WIN),yes)
# Windows shared library link flags.
ifeq ($(IS_MSVC),yes)
SOFLAGS    += -Wl,-implib:$(BASE_LIB_PATH)/$(LIBBLIS).lib
else
SOFLAGS    += -Wl,--out-implib,$(BASE_LIB_PATH)/$(LIBBLIS).dll.a
endif
else
# Linux shared library link flags.
SOFLAGS    += -Wl,-soname,$(LIBBLIS_SONAME)
endif
endif

# Decide which library to link to for things like the testsuite and BLIS test
# drivers. We default to the static library, unless only the shared library was
# enabled, in which case we use the shared library.
LIBBLIS_L      := $(LIBBLIS_A)
LIBBLIS_LINK   := $(LIBBLIS_A_PATH)
ifeq ($(MK_ENABLE_SHARED),yes)
ifeq ($(MK_ENABLE_STATIC),no)
LIBBLIS_L      := $(LIBBLIS_SO)
LIBBLIS_LINK   := $(LIBBLIS_SO_PATH)
ifeq ($(IS_WIN),no)
# For Linux and OS X: set rpath property of shared object.
ifeq ($(OS_NAME),Darwin)
# rpath for test_libblis.x
LDFLAGS        += -Wl,-rpath,@executable_path/$(BASE_LIB_PATH)
# rpath for BLAS tests
LDFLAGS        += -Wl,-rpath,@executable_path/../../../$(BASE_LIB_PATH)
else
# rpath for test_libblis.x
LDFLAGS        += -Wl,-rpath,'$$ORIGIN/$(BASE_LIB_PATH)'
# rpath for BLAS tests
LDFLAGS        += -Wl,-rpath,'$$ORIGIN/../../../$(BASE_LIB_PATH)'
endif
endif
endif
# On windows, use the shared library even if static is created.
ifeq ($(IS_WIN),yes)
LIBBLIS_L      := $(LIBBLIS_SO)
LIBBLIS_LINK   := $(LIBBLIS_SO_PATH)
endif
endif



#
# --- Include makefile definitions file ----------------------------------------
#

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE     := make_defs.mk

# Assemble a list of all configuration family members, including the
# configuration family name itself. Note that sort() will remove duplicates
# for situations where CONFIG_NAME is present in CONFIG_LIST, such as would
# be the case for singleton families.
CONFIG_LIST_FAM    := $(sort $(strip $(CONFIG_LIST) $(CONFIG_NAME)))

# Construct the paths to the makefile definitions files, each of which
# resides in a separate configuration sub-directory. We use CONFIG_LIST_FAM
# since we might need the makefile definitions associated with the
# configuration family (if it is an umbrella family).
# NOTE: We use the prefix $(BASE_SHARE_PATH)/$(CONFIG_DIR)/ instead of
# $(CONFIG_PATH) so that make_defs.mk can be found when it is installed,
# provided the caller defined SHARE_PATH to that install directory.
CONFIG_PATHS       := $(addprefix $(BASE_SHARE_PATH)/$(CONFIG_DIR)/, \
                                  $(CONFIG_LIST_FAM))
MAKE_DEFS_MK_PATHS := $(addsuffix /$(MAKE_DEFS_FILE), $(CONFIG_PATHS))

# Initialize the list of included (found) configurations to empty.
CONFIGS_INCL       :=

# Include the makefile definitions files implied by the list of configurations.
-include $(MAKE_DEFS_MK_PATHS)

# Detect whether we actually got all of the make definitions files. If
# we didn't, then maybe a configuration is mislabeled or missing. The
# check-env-make-defs target checks ALL_MAKE_DEFS_MK_PRESENT and outputs
# an error message if it is set to 'no'.
# NOTE: We use CONFIG_LIST_FAM as the expected list of configurations.
# This combines CONFIG_NAME with CONFIG_LIST. The inclusion of CONFIG_NAME
# is needed for situations where the configuration family is an umbrella
# family (e.g. 'intel64'), since families have separate make_def.mk files.
CONFIGS_EXPECTED := $(CONFIG_LIST_FAM)
ifeq ($(sort $(strip $(CONFIGS_INCL))), \
      $(sort $(strip $(CONFIGS_EXPECTED))))
ALL_MAKE_DEFS_MK_PRESENT := yes
else
ALL_MAKE_DEFS_MK_PRESENT := no
endif



#
# --- Configuration-agnostic flags ---------------------------------------------
#

# --- Linker program ---

# Use whatever compiler was chosen. A C++ compiler must be used if HPX is enabled.
ifneq ($(findstring hpx,$(THREADING_MODEL)),)
LINKER     := $(CXX)
else
LINKER     := $(CC)
endif

# --- Warning flags ---

CWARNFLAGS :=

# Do not allow functions with implicit definitions to be called
ifneq ($(CC_VENDOR),ibm)
CWARNFLAGS += -Werror=implicit-function-declaration
endif

# Disable unused function warnings and stop compiling on first error for
# all compilers that accept such options: gcc, clang, and icc.
ifneq ($(CC_VENDOR),ibm)
ifneq ($(CC_VENDOR),NVIDIA)
CWARNFLAGS += -Wall -Wno-unused-function -Wfatal-errors
else
CWARNFLAGS += -Wall -Wno-unused-function
endif
endif

# Disable tautological comparision warnings in clang.
ifeq ($(CC_VENDOR),clang)
CWARNFLAGS += -Wno-tautological-compare -Wno-pass-failed
endif

# Disable some warnings in nvc
ifeq ($(CC_VENDOR),NVIDIA)
CWARNFLAGS += --diag_suppress unsigned_compare_with_zero \
			  --diag_suppress cast_to_qualified_type \
			  --diag_suppress bad_initializer_type \
			  --diag_suppress used_before_set \
			  --diag_suppress mixed_enum_type \
			  --diag_suppress incompatible_assignment_operands
endif

# Disable other annoying warnings.
ifeq ($(CC_VENDOR),clang)
CWARNFLAGS +=
else
ifeq ($(CC_VENDOR),gcc)
# The '-Wno-maybe-uninitialized' option makes me nervous. Let's temporarily
# disable for now. -FGVZ
#CWARNFLAGS += -Wno-maybe-uninitialized -Wno-comment
CWARNFLAGS += -Wno-comment
endif
endif

$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CWARNFLAGS,$(c))))

# --- Position-independent code flags (shared libraries only) ---

# Note: Avoid -fPIC flags for Windows builds since all code is position-
# independent.
ifeq ($(IS_MSVC),yes)
CPICFLAGS :=
endif
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call store-var-for,CPICFLAGS,$(c))))

# --- Symbol exporting flags (shared libraries only) ---

ifeq ($(MK_ENABLE_SHARED),yes)

# NOTE: These flags are only applied when building BLIS and not used by
# applications that import BLIS compilation flags via the
# $(get-user-cflags-for ...) function.

# Determine default export behavior / visibility of symbols for gcc.
ifeq ($(CC_VENDOR),gcc)
ifeq ($(IS_WIN),yes)
ifeq ($(EXPORT_SHARED),all)
BUILD_SYMFLAGS := -Wl,--export-all-symbols, -Wl,--enable-auto-import
else # ifeq ($(EXPORT_SHARED),public)
BUILD_SYMFLAGS := -Wl,--exclude-all-symbols
endif
else # ifeq ($(IS_WIN),no)
ifeq ($(EXPORT_SHARED),all)
# Export all symbols by default.
BUILD_SYMFLAGS := -fvisibility=default
else # ifeq ($(EXPORT_SHARED),public)
# Hide all symbols by default and export only those that have been annotated
# as needing to be exported.
BUILD_SYMFLAGS := -fvisibility=hidden
endif
endif
endif

# Determine default export behavior / visibility of symbols for icc.
# NOTE: The Windows branches have been omitted since we currently make no
# effort to support Windows builds via icc (only gcc/clang via AppVeyor).
ifeq ($(CC_VENDOR),icc)
ifeq ($(EXPORT_SHARED),all)
# Export all symbols by default.
BUILD_SYMFLAGS := -fvisibility=default
else # ifeq ($(EXPORT_SHARED),public)
# Hide all symbols by default and export only those that have been annotated
# as needing to be exported.
BUILD_SYMFLAGS := -fvisibility=hidden
endif
endif

# Determine default export behavior / visibility of symbols for clang.
ifeq ($(CC_VENDOR),clang)
ifeq ($(IS_WIN),yes)
ifeq ($(IS_MSVC),no)
# This is a clang build targetting MinGW-w64 env
ifeq ($(EXPORT_SHARED),all)
BUILD_SYMFLAGS := -Wl,--export-all-symbols, -Wl,--enable-auto-import
else # ifeq ($(EXPORT_SHARED),all)
BUILD_SYMFLAGS := -Wl,--exclude-all-symbols
endif
endif # ifeq ($(IS_MSVC),no)
ifeq ($(EXPORT_SHARED),all)
# NOTE: clang on Windows does not appear to support exporting all symbols
# by default, and therefore we ignore the value of EXPORT_SHARED.
BUILD_SYMFLAGS :=
else # ifeq ($(EXPORT_SHARED),public)
# NOTE: The default behavior of clang on Windows is to hide all symbols
# and only export functions and other declarations that have beenannotated
# as needing to be exported.
BUILD_SYMFLAGS :=
endif
else # ifeq ($(IS_WIN),no)
ifeq ($(EXPORT_SHARED),all)
# Export all symbols by default.
BUILD_SYMFLAGS := -fvisibility=default
else # ifeq ($(EXPORT_SHARED),public)
# Hide all symbols by default and export only those that have been annotated
# as needing to be exported.
BUILD_SYMFLAGS := -fvisibility=hidden
endif
endif
endif

else #ifeq ($(MK_ENABLE_SHARED),no)

# Don't modify CPICFLAGS for the various configuration family members.
# Don't use any special symbol export flags.
BUILD_SYMFLAGS :=

endif

# --- Language flags ---

# Enable C99.
CLANGFLAGS := -std=c99
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CLANGFLAGS,$(c))))

# Enable C++11, or C++17 if HPX threading is enabled.
# If building a plugin, do not set any default C++ standard.
ifeq ($(PLUGIN_NAME),)
ifneq ($(findstring hpx,$(THREADING_MODEL)),)
CXXLANGFLAGS := -std=c++17
else
CXXLANGFLAGS := -std=c++11
endif
else
CXXLANGFLAGS :=
endif

# --- C Preprocessor flags ---

# Enable clock_gettime() in time.h.
CPPROCFLAGS := -D_POSIX_C_SOURCE=200112L
# Enable ip_mreq on macOS which is needed for ASIO which is needed for HPX.
ifeq ($(OS_NAME),Darwin)
CPPROCFLAGS += -D_DARWIN_C_SOURCE
endif
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CPPROCFLAGS,$(c))))

# --- AddressSanitizer flags ---

ifeq ($(MK_ENABLE_ASAN),yes)
BUILD_ASANFLAGS := -fsanitize=address
else
BUILD_ASANFLAGS :=
endif

# --- Threading flags ---

# NOTE: We don't have to explicitly omit -pthread when --disable-system is given
# since that option forces --enable-threading=single, and thus -pthread never
# gets added to begin with.

CTHREADFLAGS :=
CXXTHREADFLAGS :=

ifeq ($(CC_VENDOR),gcc)
#ifneq ($(findstring auto,$(THREADING_MODEL)),)
#THREADING_MODEL := openmp
#endif
ifneq ($(findstring openmp,$(THREADING_MODEL)),)
CTHREADFLAGS += -fopenmp
LDFLAGS      += -fopenmp
endif
ifneq ($(findstring pthreads,$(THREADING_MODEL)),)
CTHREADFLAGS += -pthread
LDFLAGS      += $(LIBPTHREAD)
endif
endif

ifeq ($(CC_VENDOR),icc)
#ifneq ($(findstring auto,$(THREADING_MODEL)),)
#THREADING_MODEL := openmp
#endif
ifneq ($(findstring openmp,$(THREADING_MODEL)),)
CTHREADFLAGS += -fopenmp
LDFLAGS      += -fopenmp
endif
ifneq ($(findstring pthreads,$(THREADING_MODEL)),)
CTHREADFLAGS += -pthread
LDFLAGS      += $(LIBPTHREAD)
endif
endif

ifeq ($(CC_VENDOR),clang)
#ifneq ($(findstring auto,$(THREADING_MODEL)),)
#THREADING_MODEL := pthreads
#endif
ifneq ($(findstring openmp,$(THREADING_MODEL)),)
CTHREADFLAGS += -fopenmp
LDFLAGS      += -fopenmp
endif
ifneq ($(findstring pthreads,$(THREADING_MODEL)),)
CTHREADFLAGS += -pthread
LDFLAGS      += $(LIBPTHREAD)
endif
endif

# Threading flags for HPX.
ifneq ($(findstring hpx,$(THREADING_MODEL)),)
HPX_CXXFLAGS := $(shell pkg-config --cflags hpx_component)
HPX_LDFLAGS  := $(filter-out -shared,$(shell pkg-config --libs hpx_component))
CTHREADFLAGS += $(filter-out -std=%,$(HPX_CXXFLAGS))
LDFLAGS      += $(HPX_LDFLAGS)
ifeq ($(OS_NAME),Darwin)
RPATH_PREFIX := -Wl,-rpath,
LDFLAGS      += $(patsubst -L%,$(RPATH_PREFIX)%,$(filter -L%,$(HPX_LDFLAGS)))
endif
endif

# --- #pragma omp simd flags (used for reference kernels only) ---

ifeq ($(PRAGMA_OMP_SIMD),yes)
ifeq ($(CC_VENDOR),gcc)
COMPSIMDFLAGS := -fopenmp-simd
else
ifeq ($(CC_VENDOR),clang)
COMPSIMDFLAGS := -fopenmp-simd
else
ifeq ($(CC_VENDOR),icc)
COMPSIMDFLAGS := -qopenmp-simd
endif
endif
endif
else # ifeq ($(PRAGMA_OMP_SIMD),no)
COMPSIMDFLAGS :=
endif



#
# --- Adjust verbosity level manually using make V=[0,1] -----------------------
#

ifeq ($(V),1)
ENABLE_VERBOSE := yes
BLIS_ENABLE_TEST_OUTPUT := yes
endif

ifeq ($(V),0)
ENABLE_VERBOSE := no
BLIS_ENABLE_TEST_OUTPUT := no
endif



#
# --- Append OS-specific libraries to LDFLAGS ----------------------------------
#

ifeq ($(OS_NAME),Linux)
# Exclude -lrt on Android by detecting Bionic.
# printf *must* be used here rather than echo -e
BIONIC := $(findstring bionic,$(shell printf "\#ifdef __BIONIC__\nbionic\n\#endif" | $(CC) -E -))
ifeq (,$(BIONIC))
LDFLAGS += -lrt
endif
endif



#
# --- LDFLAGS cleanup ----------------------------------------------------------
#



#
# --- Include makefile fragments -----------------------------------------------
#

# Initialize our list of directory paths to makefile fragments with the empty
# list. This variable will accumulate all of the directory paths in which
# makefile fragments reside.
FRAGMENT_DIR_PATHS :=

# Initialize our makefile variables that source code files will be accumulated
# into by the makefile fragments. This initialization is very important! These
# variables will end up with weird contents if we don't initialize them to
# empty prior to recursively including the makefile fragments.
MK_CONFIG_SRC      :=
MK_KERNELS_SRC     :=
MK_REFKERN_SRC     :=
MK_FRAME_SRC       :=
MK_ADDON_SRC       :=
MK_SANDBOX_SRC     :=

# -- config --

# Construct paths to each of the sub-configurations specified in the
# configuration list. Note that we use CONFIG_LIST_FAM, which already
# has CONFIG_NAME included (with duplicates removed).
CONFIG_PATHS       := $(addprefix $(CONFIG_FRAG_PATH)/, $(CONFIG_LIST_FAM))

# This variable is used by the include statements as they recursively include
# one another. For the 'config' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_SRC_PATH    := $(CONFIG_PATH)
PARENT_PATH        := $(CONFIG_FRAG_PATH)

# Recursively include the makefile fragments in each of the sub-configuration
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(CONFIG_PATHS))

# -- kernels --

# Construct paths to each of the kernel sets required by the sub-configurations
# in the configuration list.
KERNEL_PATHS       := $(addprefix $(KERNELS_FRAG_PATH)/, $(KERNEL_LIST))

# This variable is used by the include statements as they recursively include
# one another. For the 'kernels' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_SRC_PATH    := $(KERNELS_PATH)
PARENT_PATH        := $(KERNELS_FRAG_PATH)

# Recursively include the makefile fragments in each of the kernels sub-
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(KERNEL_PATHS))

# -- ref_kernels --
# -- frame --

# This variable is used by the include statements as they recursively include
# one another. For the framework and reference kernel source trees (ie: the
# 'frame' and 'ref_kernels' directories), we initialize it to the top-level
# directory since that is its parent.
PARENT_SRC_PATH    := $(DIST_PATH)
PARENT_PATH        := $(OBJ_DIR)/$(CONFIG_NAME)

# Recursively include all the makefile fragments in the directories for the
# reference kernels and portable framework.
-include $(addsuffix /$(FRAGMENT_MK), $(REFKERN_FRAG_PATH))
-include $(addsuffix /$(FRAGMENT_MK), $(FRAME_FRAG_PATH))

# -- addon --

# Construct paths to each addon.
# NOTE: If $(ADDON_LIST) is empty (because no addon was enabled at configure-
# time) then $(ADDON_PATHS) will also be empty, which will cause no fragments
# to be included.
ADDON_PATHS        := $(addprefix $(ADDON_FRAG_PATH)/, $(ADDON_LIST))

# This variable is used by the include statements as they recursively include
# one another. For the 'addons' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_SRC_PATH    := $(ADDON_PATH)
PARENT_PATH        := $(ADDON_FRAG_PATH)

# Recursively include the makefile fragments in each of the addons sub-
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(ADDON_PATHS))

# -- sandbox --

# Construct paths to each sandbox. (At present, there can be only one.)
# NOTE: If $(SANDBOX) is empty (because no sandbox was enabled at configure-
# time) then $(SANDBOX_PATHS) will also be empty, which will cause no
# fragments to be included.
SANDBOX_PATHS      := $(addprefix $(SANDBOX_FRAG_PATH)/, $(SANDBOX))

# This variable is used by the include statements as they recursively include
# one another. For the 'sandbox' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_SRC_PATH    := $(SANDBOX_PATH)
PARENT_PATH        := $(SANDBOX_FRAG_PATH)

# Recursively include the makefile fragments in the sandbox sub-directory.
-include $(addsuffix /$(FRAGMENT_MK), $(SANDBOX_PATHS))

# -- post-processing --

# Create a list of the makefile fragments using the variable into which each
# of the above include statements accumulated their directory paths.
MAKEFILE_FRAGMENTS := $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_DIR_PATHS))

# Detect whether we actually got any makefile fragments. If we didn't, then it
# is likely that the user has not yet generated them (via configure).
ifeq ($(strip $(MAKEFILE_FRAGMENTS)),)
MAKEFILE_FRAGMENTS_PRESENT := no
else
MAKEFILE_FRAGMENTS_PRESENT := yes
endif


#
# --- Important sets of header files and paths ---------------------------------
#

# Define a function that will expand all of the directory paths given in $(1)
# to actual filepaths using the list of suffixes provided in $(2).
get-filepaths = $(strip $(foreach path, $(1), \
                            $(foreach suf, $(2), \
                                $(wildcard $(path)/*.$(suf)) \
                 )       )   )

# Define a function that will expand all of the directory paths given in $(1)
# to actual filepaths using the list of suffixes provided in $(2), taking only
# the first expansion from each directory with at least one file matching
# the current suffix. Finally, strip the filenames from all resulting files,
# returning only the directory paths.
get-dirpaths  = $(dir $(foreach path, $(1), \
                          $(firstword \
                              $(foreach suf, $(2), \
                                  $(wildcard $(path)/*.$(suf)) \
                 )     )   )   )

# We'll use three directory lists. The first is a list of all of the directories
# in which makefile fragments were generated, plus the current directory. (The
# current directory is needed so we include bli_config.h and bli_addon.h in the
# processing of header files.) The second and third are subsets of the first
# that begins with the addon and sandbox root paths, respectively.
ALLFRAG_DIR_PATHS := . $(FRAGMENT_DIR_PATHS)
ADDON_DIR_PATHS   := $(filter $(ADDON_PATH)/%,$(ALLFRAG_DIR_PATHS))
SANDBOX_DIR_PATHS := $(filter $(SANDBOX_PATH)/%,$(ALLFRAG_DIR_PATHS))

ALL_H99_FILES     := $(call get-filepaths,$(ALLFRAG_DIR_PATHS),$(ALL_H99_SUFS))
FRAME_H99_FILES   := $(filter-out $(ADDON_PATH)/%, \
                        $(filter-out $(SANDBOX_PATH)/%, \
                                    $(ALL_H99_FILES) \
                      )  )

ALL_H99_DIRPATHS     := $(call get-dirpaths,$(ALLFRAG_DIR_PATHS),$(ALL_H99_SUFS))

ADDON_H99_FILES      := $(call get-filepaths,$(ADDON_DIR_PATHS),$(ADDON_H99_SUFS))
ADDON_HXX_FILES      := $(call get-filepaths,$(ADDON_DIR_PATHS),$(ADDON_HXX_SUFS))
ADDON_HDR_DIRPATHS   := $(call get-dirpaths,$(ADDON_DIR_PATHS),$(ALL_HDR_SUFS))

SANDBOX_H99_FILES    := $(call get-filepaths,$(SANDBOX_DIR_PATHS),$(SANDBOX_H99_SUFS))
SANDBOX_HXX_FILES    := $(call get-filepaths,$(SANDBOX_DIR_PATHS),$(SANDBOX_HXX_SUFS))
SANDBOX_HDR_DIRPATHS := $(call get-dirpaths,$(SANDBOX_DIR_PATHS),$(ALL_HDR_SUFS))



#
# --- blis.h header definitions ------------------------------------------------
#

# Use include/CONFIG_NAME as the default path to the local header files, but
# allow the includer to override this value if it needs to point to an
# installation directory.
ifeq ($(strip $(INC_PATH)),)
INC_PATH        := $(INCLUDE_DIR)/$(CONFIG_NAME)
endif

# Identify the base path for the intermediate include directory. We define
# this path in terms of INC_PATH, which gets a default value above (which is
# what happens for the top-level Makefile). If INC_PATH is specified by the
# Makefile prior to including common.mk, that path is used instead. This
# allows Makefiles for example code and test drivers to reference an installed
# prefix directory for situations when the build directory no longer exists.
BASE_INC_PATH   := $(INC_PATH)

# Isolate the path to blis.h by filtering the file from the list of framework
# header files.
BLIS_H          := blis.h
BLIS_H_SRC_PATH := $(filter %/$(BLIS_H), $(FRAME_H99_FILES))

# Construct the path to what will be the intermediate flattened/monolithic
# blis.h file.
BLIS_H_FLAT     := $(BASE_INC_PATH)/$(BLIS_H)

# Construct the path to the helper blis.h file that will reside one directory
# up from the installed copy of blis.h.
HELP_BLIS_H_PATH := $(BUILD_DIR)/$(BLIS_H)


#
# --- cblas.h header definitions -----------------------------------------------
#

# Isolate the path to cblas.h by filtering the file from the list of framework
# header files, and then strip the filename to obtain the directory in which
# cblas.h resides.
CBLAS_H          := cblas.h
CBLAS_H_SRC_PATH := $(filter %/$(CBLAS_H), $(FRAME_H99_FILES))
CBLAS_H_DIRPATH  := $(dir $(CBLAS_H_SRC_PATH))

# Construct the path to what will be the intermediate flattened/monolithic
# cblas.h file.
CBLAS_H_FLAT      := $(BASE_INC_PATH)/$(CBLAS_H)

# Construct the path to the helper cblas.h file that will reside one directory
# up from the installed copy of cblas.h.
HELP_CBLAS_H_PATH := $(BUILD_DIR)/$(CBLAS_H)


#
# --- Compiler include path definitions ----------------------------------------
#

# Obtain a list of header files #included inside of the bli_cntx_ref.c file.
# Due to the way that bli_cntx_ref.c uses headers and macros, paths to these
# files will be needed when compiling bli_cntx_ref.c with the monolithic header.
ifeq ($(strip $(SHARE_PATH)),.)
REF_KER_SRC     := $(DIST_PATH)/$(REFKERN_DIR)/bli_cntx_ref.c
#
# NOTE: A redirect to /dev/null has been added to the grep command below because
# as of version 3.8, grep outputs warnings when encountering stray backslashes
# in regular expressions [1]. Versions older than 3.8 not only do not complain,
# but actually seem to *require* the backslash, perhaps because of the way we
# are invoking grep via GNU make's shell command. WHEN DEBUGGING ANYTHING
# INVOLVING THE MAKE VARIABLE BELOW, PLEASE CONSIDER TEMPORARILY REMOVING THE
# REDIRECT TO /dev/null SO THAT YOU SEE ANY MESSAGES SENT TO STANDARD ERROR.
#
# [1] https://lists.gnu.org/archive/html/info-gnu/2022-09/msg00001.html
#
REF_KER_HEADERS := $(shell $(GREP) "\#include" $(REF_KER_SRC) 2> $(DEVNULL) | sed -e "s/\#include [\"<]\([a-zA-Z0-9\_\.\/\-]*\)[\">].*/\1/g" | $(GREP) -v $(BLIS_H))
endif

# Match each header found above with the path to that header, and then strip
# leading, trailing, and internal whitespace.
REF_KER_H_PATHS := $(call rm-dups,$(strip \
                                  $(foreach header, $(REF_KER_HEADERS), \
                                      $(dir $(filter %/$(header), \
                                                     $(FRAME_H99_FILES))))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler. Then add frame/include since it's needed when compiling source
# files that #include bli_oapi_ba.h or bli_oapi_ex.h.
REF_KER_I_PATHS := $(strip $(patsubst %, -I%, $(REF_KER_H_PATHS)))
REF_KER_I_PATHS += -I$(DIST_PATH)/frame/include

# Prefix the paths above with the base include path.
# NOTE: We no longer need every header path in the source tree since we
# now #include the monolithic/flattened blis.h instead.
CINCFLAGS       := -I$(BASE_INC_PATH) $(REF_KER_I_PATHS)

# If CBLAS is enabled, we also include the path to the cblas.h directory so
# that the compiler will be able to find cblas.h as the CBLAS source code is
# being compiled.
ifeq ($(MK_ENABLE_CBLAS),yes)
CINCFLAGS       += -I$(CBLAS_H_DIRPATH)
endif

# Obtain a list of header paths in the configured addons. Then add -I to each
# header path.
CADDONINCFLAGS  := $(strip $(patsubst %, -I%, $(ADDON_HDR_DIRPATHS)))

# Obtain a list of header paths in the configured sandbox. Then add -I to each
# header path.
CSANDINCFLAGS   := $(strip $(patsubst %, -I%, $(SANDBOX_HDR_DIRPATHS)))


#
# --- BLIS configuration header definitions ------------------------------------
#

# These files were created by configure, but we need to define them here so we
# can remove them as part of the clean targets.
BLIS_ADDON_H    := ./bli_addon.h
BLIS_CONFIG_H   := ./bli_config.h


#
# --- Special preprocessor macro definitions -----------------------------------
#

# Define a C preprocessor flag that is *only* defined when BLIS is being
# compiled. (In other words, an application that #includes blis.h will not
# get this cpp macro.)
BUILD_CPPFLAGS := -DBLIS_IS_BUILDING_LIBRARY


#
# --- configure file location --------------------------------------------------
#

CONFIGURE_FILE := $(DIST_PATH)/configure


# end of ifndef COMMON_MK_INCLUDED conditional block
endif
