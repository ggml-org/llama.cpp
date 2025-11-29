#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2022, Advanced Micro Devices, Inc.
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
# Top-level makefile for libflame linear algebra library.
#
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all \
        libs libblis \
        check-env check-env-mk check-env-fragments check-env-make-defs \
        flat-header flat-cblas-header \
        test \
        testblas blastest-f2c blastest-bin blastest-run \
        testsuite testsuite-bin \
        testsuite-run testsuite-run-fast testsuite-run-md testsuite-run-salt \
        testblis testblis-fast testblis-md testblis-salt \
        check checkblas \
        checkblis checkblis-fast checkblis-md checkblis-salt \
        install-headers install-helper-headers install-libs install-lib-symlinks \
        showconfig \
        clean cleanmk cleanh cleanlib distclean \
        cleantest cleanblastest cleanblistest \
        changelog \
        symbols \
        install uninstall uninstall-old \
        uninstall-libs uninstall-lib-symlinks uninstall-headers \
        uninstall-old-libs uninstall-lib-symlinks uninstall-old-headers

#
# --- Determine makefile fragment location -------------------------------------
#

# Comments:
# - We don't need to define DIST_PATH, LIB_PATH, INC_PATH, or SHARE_PATH since
#   the defaults in common.mk (and config.mk) are designed to work with the
#   top-level Makefile.
#DIST_PATH  := .
#LIB_PATH    = ./lib/$(CONFIG_NAME)
#INC_PATH    = ./include/$(CONFIG_NAME)
#SHARE_PATH := .


#
# --- Include common makefile definitions --------------------------------------
#

# Define the name of the common makefile.
COMMON_MK_FILE    := common.mk

# Include the configuration file.
-include $(COMMON_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(COMMON_MK_INCLUDED)),yes)
COMMON_MK_PRESENT := yes
else
COMMON_MK_PRESENT := no
endif



#
# --- Main target variable definitions -----------------------------------------
#

# --- Object file paths ---

# Construct the base object file path for the current configuration.
BASE_OBJ_PATH          := ./$(OBJ_DIR)/$(CONFIG_NAME)

# Construct base object file paths corresponding to the four locations
# of source code.
BASE_OBJ_CONFIG_PATH   := $(BASE_OBJ_PATH)/$(CONFIG_DIR)
BASE_OBJ_FRAME_PATH    := $(BASE_OBJ_PATH)/$(FRAME_DIR)
BASE_OBJ_REFKERN_PATH  := $(BASE_OBJ_PATH)/$(REFKERN_DIR)
BASE_OBJ_KERNELS_PATH  := $(BASE_OBJ_PATH)/$(KERNELS_DIR)
BASE_OBJ_ADDON_PATH    := $(BASE_OBJ_PATH)/$(ADDON_DIR)
BASE_OBJ_SANDBOX_PATH  := $(BASE_OBJ_PATH)/$(SANDBOX_DIR)

# --- Define install target names for static libraries ---

LIBBLIS_A_INST            := $(INSTALL_LIBDIR)/$(LIBBLIS_A)

# --- Define install target names for shared libraries ---

LIBBLIS_SO_INST           := $(INSTALL_LIBDIR)/$(LIBBLIS_SO)
LIBBLIS_SO_MAJ_INST       := $(INSTALL_LIBDIR)/$(LIBBLIS_SONAME)

ifeq ($(IS_WIN),yes)
# The 'install' target does not create symlinks for Windows builds, so we don't
# bother defining LIBBLIS_SO_MMB_INST.
LIBBLIS_SO_MMB_INST       :=
else
LIBBLIS_SO_MMB_INST       := $(INSTALL_LIBDIR)/$(LIBBLIS).$(LIBBLIS_SO_MMB_EXT)
endif

# --- Determine which libraries to build ---

MK_LIBS                   :=
MK_LIBS_INST              :=
MK_LIBS_SYML              :=

ifeq ($(MK_ENABLE_STATIC),yes)
MK_LIBS                   += $(LIBBLIS_A_PATH)
MK_LIBS_INST              += $(LIBBLIS_A_INST)
MK_LIBS_SYML              +=
endif
ifeq ($(MK_ENABLE_SHARED),yes)
MK_LIBS                   += $(LIBBLIS_SO_PATH) \
                             $(LIBBLIS_SO_MAJ_PATH)
MK_LIBS_INST              += $(LIBBLIS_SO_MMB_INST)
MK_LIBS_SYML              += $(LIBBLIS_SO_INST) \
                             $(LIBBLIS_SO_MAJ_INST)
endif

# Strip leading, internal, and trailing whitespace.
MK_LIBS_INST              := $(strip $(MK_LIBS_INST))
MK_LIBS_SYML              := $(strip $(MK_LIBS_SYML))

# --- Define install directory for headers ---

# Set the path to the subdirectory of the include installation directory.
MK_INCL_DIR_INST          := $(INSTALL_INCDIR)/blis

# --- Define install directory for public makefile fragments ---

# Set the path to the subdirectory of the share installation directory.
MK_SHARE_DIR_INST         := $(INSTALL_SHAREDIR)/blis

PC_SHARE_DIR_INST         := $(INSTALL_SHAREDIR)/pkgconfig


#
# --- Library object definitions -----------------------------------------------
#

# In this section, we will isolate the relevant source code filepaths and
# convert them to lists of object filepaths. Relevant source code falls into
# four categories: configuration source; architecture-specific kernel source;
# reference kernel source; and general framework source.

# $(call gen-obj-paths-from-src file_exts, src_files, base_src_path, base_obj_path)
gen-obj-paths-from-src = $(foreach ch, $(1), \
                             $(patsubst $(3)/%.$(ch), \
                                        $(4)/%.o, \
                                        $(filter %.$(ch), $(2)) ) )

# Generate object file paths for source code found in the sub-configuration
# directories.
MK_CONFIG_OBJS      := $(call gen-obj-paths-from-src,$(CONFIG_SRC_SUFS),$(MK_CONFIG_SRC),$(CONFIG_PATH),$(BASE_OBJ_CONFIG_PATH))

# Generate object file paths for architecture-specific kernel source code.
# We target only .c, .s, and .S files. Note that MK_KERNELS_SRC is already
# limited to the kernel source corresponding to the kernel sets in
# KERNEL_LIST. This is because the configure script only propogated makefile
# fragments into those specific kernel subdirectories.
MK_KERNELS_OBJS     := $(call gen-obj-paths-from-src,$(KERNELS_SRC_SUFS),$(MK_KERNELS_SRC),$(KERNELS_PATH),$(BASE_OBJ_KERNELS_PATH))

# Generate object file paths for reference kernels, with one set of object
# files for each sub-configuration in CONFIG_LIST. Note that due to the
# nuances of naming the reference kernel files, we can't use the function
# gen-obj-paths-from-src as we do above and below.
MK_REFKERN_C        := $(filter %.c, $(MK_REFKERN_SRC))
MK_REFKERN_OBJS     := $(foreach arch, $(CONFIG_LIST), \
                           $(patsubst $(REFKERN_PATH)/%_$(REFNM).c, \
                                 $(BASE_OBJ_REFKERN_PATH)/$(arch)/%_$(arch)_$(REFNM).o, \
                                 $(MK_REFKERN_C) \
                            ) \
                        )

# Generate object file paths for all of the portable framework source code.
MK_FRAME_OBJS       := $(call gen-obj-paths-from-src,$(FRAME_SRC_SUFS),$(MK_FRAME_SRC),$(FRAME_PATH),$(BASE_OBJ_FRAME_PATH))

# Generate object file paths for the addon source code. If one or more addons
# were not enabled a configure-time, these variable will we empty.
# NOTE: We separate the source and objects into kernel and non-kernel lists.
MK_ADDON_KERS_SRC   := $(foreach addon, $(ADDON_LIST), \
                           $(filter $(ADDON_PATH)/$(addon)/$(KERNELS_DIR)/%, \
                                    $(MK_ADDON_SRC)) \
                        )
MK_ADDON_OTHER_SRC  := $(foreach addon, $(ADDON_LIST), \
                           $(filter-out $(ADDON_PATH)/$(addon)/$(KERNELS_DIR)/%, \
                                        $(MK_ADDON_SRC)) \
                        )
MK_ADDON_KERS_OBJS  := $(call gen-obj-paths-from-src,$(ADDON_SRC_SUFS),$(MK_ADDON_KERS_SRC),$(ADDON_PATH),$(BASE_OBJ_ADDON_PATH))
MK_ADDON_OTHER_OBJS := $(call gen-obj-paths-from-src,$(ADDON_SRC_SUFS),$(MK_ADDON_OTHER_SRC),$(ADDON_PATH),$(BASE_OBJ_ADDON_PATH))
MK_ADDON_OBJS       := $(MK_ADDON_KERS_OBJS) $(MK_ADDON_OTHER_OBJS)

# Generate object file paths for the sandbox source code. If a sandbox was not
# enabled a configure-time, this variable will we empty.
MK_SANDBOX_OBJS     := $(call gen-obj-paths-from-src,$(SANDBOX_SRC_SUFS),$(MK_SANDBOX_SRC),$(SANDBOX_PATH),$(BASE_OBJ_SANDBOX_PATH))

# AMD has chosen to introduce AOCL-specific optimizations to certain BLIS
# framework files that are otherwise intended to remain generic. Upstream
# developers of vanilla BLIS have agreed to integrate some of these
# optimizations, but in a way that keeps the AOCL-specific code segregated
# in separate files containing the suffix '_amd'. For example, the BLAS
# compatibility layer in vanilla BLIS contains a generic file named
# 'bla_gemm.c'. AMD's version of this file is named 'bla_gemm_amd.c'.
# Only one or the other is ever built and included in libblis. Currently,
# these files are chosen automatically based on the target configuration.
ifeq ($(ENABLE_AMD_FRAME_TWEAKS),yes)
# Build is being done for AMD platforms; remove the objects which DO NOT have
# an "_amd" suffix.
MK_FRAME_AMD_OBJS  := $(filter $(BASE_OBJ_FRAME_PATH)/%amd.o, $(MK_FRAME_OBJS))
FILES_TO_REMOVE := $(subst _amd.o,.o, $(MK_FRAME_AMD_OBJS))
MK_FRAME_OBJS := $(filter-out $(FILES_TO_REMOVE), $(MK_FRAME_OBJS))
else
# Build is being done for non-AMD platforms; remove the objects which DO have
# an "_amd" suffix.
MK_FRAME_AMD_OBJS  := $(filter $(BASE_OBJ_FRAME_PATH)/%amd.o, $(MK_FRAME_OBJS))
MK_FRAME_OBJS := $(filter-out $(MK_FRAME_AMD_OBJS), $(MK_FRAME_OBJS))
endif

# Combine all of the object files into some readily-accessible variables.
MK_BLIS_OBJS        := $(MK_CONFIG_OBJS) \
                       $(MK_KERNELS_OBJS) \
                       $(MK_REFKERN_OBJS) \
                       $(MK_FRAME_OBJS) \
                       $(MK_ADDON_OBJS) \
                       $(MK_SANDBOX_OBJS)

# Optionally filter out the BLAS and CBLAS compatibility layer object files.
# This is not actually necessary, since each affected file is guarded by C
# preprocessor macros, but it but prevents "empty" object files from being
# added into the library (and reduces compilation time).
BASE_OBJ_BLAS_PATH  := $(BASE_OBJ_FRAME_PATH)/compat
BASE_OBJ_CBLAS_PATH := $(BASE_OBJ_FRAME_PATH)/compat/cblas
ifeq ($(MK_ENABLE_CBLAS),no)
MK_BLIS_OBJS        := $(filter-out $(BASE_OBJ_CBLAS_PATH)/%.o, $(MK_BLIS_OBJS) )
endif
ifeq ($(MK_ENABLE_BLAS),no)
MK_BLIS_OBJS        := $(filter-out $(BASE_OBJ_BLAS_PATH)/%.o,  $(MK_BLIS_OBJS) )
endif



#
# --- Monolithic header definitions --------------------------------------------
#

# Define lists of headers to create/install. The default is to only
# create/install blis.h.
HEADERS_TO_BUILD        := $(BLIS_H_FLAT)
HEADERS_TO_INSTALL      := $(BLIS_H_FLAT)

# If CBLAS is enabled, we also create/install cblas.h. This allows the user to
# continue using #include "cblas.h" in their application, if they wish. (NOTE:
# The user can also access CBLAS definitions and function prototypes by
# #include'ing "blis.h".)
ifeq ($(MK_ENABLE_CBLAS),yes)
HEADERS_TO_BUILD   += $(CBLAS_H_FLAT)
HEADERS_TO_INSTALL += $(CBLAS_H_FLAT)
endif

# If requested, include AMD's C++ template header files in the list of headers
# to install.
ifeq ($(INSTALL_HH),yes)
HEADERS_TO_INSTALL += $(wildcard $(VEND_CPP_PATH)/*.hh)
endif

# Determine that paths of the installed header files
HEADERS_INSTALLED := $(foreach hdr,$(HEADERS_TO_INSTALL),$(MK_INCL_DIR_INST)/$(notdir $(hdr)))

# Define a list of so-called helper headers to install. These helper headers
# are very simple headers that go one directory up from INCDIR/blis (which
# by default is PREFIX/include/blis, where PREFIX is the install prefix). The
# default is to only install the blis.h helper header.
HELP_HEADERS_TO_INSTALL := $(HELP_BLIS_H_PATH)
HELP_HEADERS_INSTALLED  := $(INSTALL_INCDIR)/$(BLIS_H)

# If CBLAS is enabled, we also install the cblas.h helper header.
ifeq ($(MK_ENABLE_CBLAS),yes)
HELP_HEADERS_TO_INSTALL += $(HELP_CBLAS_H_PATH)
HELP_HEADERS_INSTALLED  += $(INSTALL_INCDIR)/$(CBLAS_H)
endif



#
# --- public makefile fragment definitions -------------------------------------
#

# Define a list of makefile fragments to install.
FRAGS_TO_INSTALL := $(CONFIG_MK_FILE) \
                    $(COMMON_MK_FILE) \
                    $(DIST_PATH)/build/gen-make-frags/gen-make-frag.sh \
                    $(DIST_PATH)/build/gen-make-frags/fragment.mk \
                    $(DIST_PATH)/build/gen-make-frags/ignore_list \
                    $(DIST_PATH)/build/gen-make-frags/special_list \
                    $(DIST_PATH)/build/gen-make-frags/suffix_list \
                    $(DIST_PATH)/build/flatten-headers.py \
                    $(DIST_PATH)/build/mirror-tree.sh \
                    $(DIST_PATH)/config_registry \
                    $(DIST_PATH)/build/detect/iset/avx.s \
                    $(DIST_PATH)/build/detect/iset/avx512dq.s \
                    $(DIST_PATH)/build/detect/iset/avx512f.s \
                    $(DIST_PATH)/build/detect/iset/fma3.s \
                    $(DIST_PATH)/build/detect/iset/fma4.s

# Define a list of plugin makefile fragments to install.
PLUGIN_FRAGS_TO_INSTALL := $(DIST_PATH)/build/plugin/bli_plugin_init_ref.c \
                           $(DIST_PATH)/build/plugin/bli_plugin_init_zen3.c \
                           $(DIST_PATH)/build/plugin/bli_plugin_register.c \
                           $(DIST_PATH)/build/plugin/my_kernel_1_ref.c \
                           $(DIST_PATH)/build/plugin/my_kernel_2_ref.c \
                           $(DIST_PATH)/build/plugin/my_kernel_1_zen3.c \
                           $(DIST_PATH)/build/plugin/bli_plugin.h.in \
                           $(DIST_PATH)/build/plugin/config.mk.in \
                           $(DIST_PATH)/build/plugin/Makefile

PC_IN_FILE  := blis.pc.in
PC_OUT_FILE := blis.pc


#
# --- BLAS test drivers definitions --------------------------------------------
#

# The location of the BLAS test suite's input files.
BLASTEST_INPUT_PATH    := $(DIST_PATH)/$(BLASTEST_DIR)/input

# The location of the BLAS test suite object directory.
BASE_OBJ_BLASTEST_PATH := $(BASE_OBJ_PATH)/$(BLASTEST_DIR)

# The locations of the BLAS test suite source code (f2c and drivers).
BLASTEST_F2C_SRC_PATH  := $(DIST_PATH)/$(BLASTEST_DIR)/f2c
BLASTEST_DRV_SRC_PATH  := $(DIST_PATH)/$(BLASTEST_DIR)/src

# The paths to object files we will create (f2c and drivers).
BLASTEST_F2C_OBJS      := $(sort \
                          $(patsubst $(BLASTEST_F2C_SRC_PATH)/%.c, \
                                     $(BASE_OBJ_BLASTEST_PATH)/%.o, \
                                     $(wildcard $(BLASTEST_F2C_SRC_PATH)/*.c)) \
                           )

BLASTEST_DRV_OBJS      := $(sort \
                          $(patsubst $(BLASTEST_DRV_SRC_PATH)/%.c, \
                                     $(BASE_OBJ_BLASTEST_PATH)/%.o, \
                                     $(wildcard $(BLASTEST_DRV_SRC_PATH)/*.c)) \
                           )

# libf2c name and location.
BLASTEST_F2C_LIB_NAME  := libf2c.a
BLASTEST_F2C_LIB       := $(BASE_OBJ_BLASTEST_PATH)/$(BLASTEST_F2C_LIB_NAME)

# The base names of each driver source file (ie: filename minus suffix).
BLASTEST_DRV_BASES     := $(basename $(notdir $(BLASTEST_DRV_OBJS)))

# The binary executable driver names.
BLASTEST_DRV_BINS      := $(addsuffix .x,$(BLASTEST_DRV_BASES))
BLASTEST_DRV_BIN_PATHS := $(addprefix $(BASE_OBJ_BLASTEST_PATH)/,$(BLASTEST_DRV_BINS))

# Binary executable driver "run-" names
BLASTEST_DRV_BINS_R    := $(addprefix run-,$(BLASTEST_DRV_BASES))

# Filter level-1, level-2, and level-3 names to different variables.
BLASTEST_DRV1_BASES    := $(filter %1,$(BLASTEST_DRV_BASES))
BLASTEST_DRV2_BASES    := $(filter %2,$(BLASTEST_DRV_BASES))
BLASTEST_DRV3_BASES    := $(filter %3,$(BLASTEST_DRV_BASES))

# Define some CFLAGS that we'll only use when compiling BLAS test suite
# files.
BLAT_CFLAGS            := -Wno-parentheses \
                          -I$(BLASTEST_F2C_SRC_PATH) \
                          -I. -DHAVE_BLIS_H

# Suppress warnings about possibly uninitialized variables for the BLAS
# test driver code (as output from f2c), which is riddled with such
# variables, but only if the option to do so is supported.
ifeq ($(CC_VENDOR),gcc)
BLAT_CFLAGS            += -Wno-maybe-uninitialized
endif

# The location of the script that checks the BLAS test output.
BLASTEST_CHECK_PATH    := $(DIST_PATH)/$(BLASTEST_DIR)/$(BLASTEST_CHECK)


#
# --- BLIS testsuite definitions -----------------------------------------------
#

# The location of the test suite's general and operations-specific
# input/configuration files.
TESTSUITE_CONF_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_GEN)
TESTSUITE_CONF_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_OPS)
TESTSUITE_FAST_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_FAST_GEN)
TESTSUITE_FAST_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_FAST_OPS)
TESTSUITE_MIXD_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_MIXD_GEN)
TESTSUITE_MIXD_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_MIXD_OPS)
TESTSUITE_SALT_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_SALT_GEN)
TESTSUITE_SALT_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_SALT_OPS)

# The locations of the test suite source directory and the local object
# directory.
TESTSUITE_SRC_PATH      := $(DIST_PATH)/$(TESTSUITE_DIR)/src
BASE_OBJ_TESTSUITE_PATH := $(BASE_OBJ_PATH)/$(TESTSUITE_DIR)

# Convert source file paths to object file paths by replacing the base source
# directories with the base object directories, and also replacing the source
# file suffix (eg: '.c') with '.o'.
MK_TESTSUITE_OBJS       := $(sort \
                           $(patsubst $(TESTSUITE_SRC_PATH)/%.c, \
                                      $(BASE_OBJ_TESTSUITE_PATH)/%.o, \
                                      $(wildcard $(TESTSUITE_SRC_PATH)/*.c)) \
                            )

# The test suite binary executable filename.
# NOTE: The TESTSUITE_WRAPPER variable defaults to the empty string if it
# is not already set, in which case it has no effect lateron when the
# testsuite binary is executed via lines such as
#
#   $(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) ... > $(TESTSUITE_OUT_FILE)
#
# The reason TESTSUITE_WRAPPER is employed in this way is so that some
# unusual environments (e.g. ARM) can run the testsuite through some other
# binary. See .travis.yml for details on how the variable is employed in
# practice.
TESTSUITE_BIN           := test_$(LIBBLIS).x
TESTSUITE_WRAPPER       ?=

# The location of the script that checks the BLIS testsuite output.
TESTSUITE_CHECK_PATH    := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CHECK)



#
# --- Uninstall definitions ----------------------------------------------------
#

ifeq ($(IS_CONFIGURED),yes)

# These shell commands gather the filepaths to any library in the current
# LIBDIR that might be left over from an old installation. We start with
# including nothing for static libraries, since older static libraries are
# always overwritten by newer ones. Then we add shared libraries, which are
# named with three .so version numbers.
UNINSTALL_OLD_LIBS    :=

UNINSTALL_OLD_LIBS    += $(filter-out $(INSTALL_LIBDIR)/$(LIBBLIS).$(LIBBLIS_SO_MMB_EXT),$(wildcard $(INSTALL_LIBDIR)/$(LIBBLIS_SO).?.?.?))

# These shell commands gather the filepaths to any library symlink in the
# current LIBDIR that might be left over from an old installation. We start
# with symlinks named using the .so major version number.
UNINSTALL_OLD_SYML    := $(filter-out $(INSTALL_LIBDIR)/$(LIBBLIS_SO).$(SO_MAJOR),$(wildcard $(INSTALL_LIBDIR)/$(LIBBLIS_SO).?))

# We also prepare to uninstall older-style symlinks whose names contain the
# BLIS version number and configuration family.
UNINSTALL_OLD_SYML    += $(wildcard $(INSTALL_LIBDIR)/$(LIBBLIS)-*.a)
UNINSTALL_OLD_SYML    += $(wildcard $(INSTALL_LIBDIR)/$(LIBBLIS)-*.$(SHLIB_EXT))

# This shell command grabs all files named "*.h" that are not blis.h or cblas.h
# in the installation directory. We consider this set of headers to be "old" and
# eligible for removal upon running of the uninstall-old-headers target.
UNINSTALL_OLD_HEADERS := $(filter-out %/$(BLIS_H),$(filter-out %/$(CBLAS_H),$(wildcard $(INSTALL_INCDIR)/blis/*.h)))

endif # IS_CONFIGURED


#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: libs

libs: libblis

test: checkblis checkblas

check: checkblis-fast checkblas

install: libs install-libs install-lib-symlinks install-headers install-share

uninstall: uninstall-libs uninstall-lib-symlinks uninstall-headers uninstall-share

uninstall-old: uninstall-old-libs uninstall-old-symlinks uninstall-old-headers

clean: cleanh cleanlib


# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-mk

check-env-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-fragments: check-env-mk
ifeq ($(MAKEFILE_FRAGMENTS_PRESENT),no)
	$(error Cannot proceed: makefile fragments not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(ALL_MAKE_DEFS_MK_PRESENT),no)
	$(error Cannot proceed: Some make_defs.mk files not found or mislabeled!)
endif


# --- Shared/dynamic libblis symbol file creation/refresh ---

symbols: check-env $(SYM_FILE)

$(SYM_FILE): $(HEADERS_TO_INSTALL)
ifeq ($(ENABLE_VERBOSE),yes)
	$(GEN_SYMS) > $(SYM_FILE)
else
	@echo "Updating $(SYM_FILE)"
	@$(GEN_SYMS) > $(SYM_FILE)
endif


# --- Consolidated blis.h header creation ---

flat-header: check-env $(BLIS_H_FLAT)

$(BLIS_H_FLAT): $(ALL_H99_FILES)
ifeq ($(ENABLE_VERBOSE),yes)
	$(FLATTEN_H) -l -v1 $(BLIS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(ALL_H99_DIRPATHS)"
else
	@echo -n "Generating monolithic blis.h"
	@$(FLATTEN_H) -l -v1 $(BLIS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(ALL_H99_DIRPATHS)"
	@echo "Generated $@"
endif

# --- Consolidated cblas.h header creation ---

flat-cblas-header: check-env $(CBLAS_H_FLAT)

# Note that the flattened blis.h is a prerequisite of flattening cblas.h. This
# is done so that the two headers are built sequentially even when using
# 'make -j[n]'. Otherwise, the output from the two processes can become
# interleaved, which looks awkward/confusing.
$(CBLAS_H_FLAT): $(FRAME_H99_FILES) $(BLIS_H_FLAT)
ifeq ($(ENABLE_VERBOSE),yes)
	$(FLATTEN_H) -l -v1 $(CBLAS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(ALL_H99_DIRPATHS)"
else
	@echo -n "Generating monolithic cblas.h"
	@$(FLATTEN_H) -l -v1 $(CBLAS_H_SRC_PATH) $@ "./$(INCLUDE_DIR)" "$(ALL_H99_DIRPATHS)"
	@echo "Generated $@"
endif


# --- General source code / object code rules ---

# FGVZ: Add support for compiling .s and .S files in 'config'/'kernels'
# directories.
#  - May want to add an extra foreach loop around function eval/call.

# first argument: a configuration name from config_list, used to look up the
# CFLAGS to use during compilation.
define make-config-rule
$(BASE_OBJ_CONFIG_PATH)/$(1)/%.o: $(CONFIG_PATH)/$(1)/%.c $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-config-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-config-text-for,$(1))
	@$(CC) $(call get-config-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
define make-frame-rule
$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.c $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-frame-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-text-for,$(1))
	@$(CC) $(call get-frame-cflags-for,$(1)) -c $$< -o $$@
endif

ifneq ($(findstring hpx,$(THREADING_MODEL)),)
$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.cpp $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-frame-cxxflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-cxxtext-for,$(1))
	@$(CXX) $(call get-frame-cxxflags-for,$(1)) -c $$< -o $$@
endif
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
define make-refinit-rule
$(BASE_OBJ_REFKERN_PATH)/$(1)/bli_cntx_$(1)_ref.o: $(REFKERN_PATH)/bli_cntx_ref.c $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-refinit-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refinit-text-for,$(1))
	@$(CC) $(call get-refinit-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
define make-refkern-rule
$(BASE_OBJ_REFKERN_PATH)/$(1)/%_$(1)_ref.o: $(REFKERN_PATH)/%_ref.c $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-refkern-cflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refkern-text-for,$(1))
	@$(CC) $(call get-refkern-cflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
# second argument: the configuration whose CFLAGS we should use in compilation.
# third argument: the kernel file suffix being considered.
define make-kernels-rule
$(BASE_OBJ_KERNELS_PATH)/$(1)/%.o: $(KERNELS_PATH)/$(1)/%.$(3) $(HEADERS_TO_BUILD) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-kernel-cflags-for,$(2)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-kernel-text-for,$(2))
	@$(CC) $(call get-kernel-cflags-for,$(2)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
# second argument: the C99 addon file suffix being considered.
define make-c99-addon-rule
$(BASE_OBJ_ADDON_PATH)/%.o: $(ADDON_PATH)/%.$(2) $(HEADERS_TO_BUILD) $(ADDON_H99_FILES) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-addon-c99flags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-addon-c99text-for,$(1))
	@$(CC) $(call get-addon-c99flags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
# second argument: the C99 addon file suffix being considered.
# third argument: the name of the addon being considered.
define make-c99-addon-kers-rule
$(BASE_OBJ_ADDON_PATH)/$(3)/$(KERNELS_DIR)/%.o: $(ADDON_PATH)/$(3)/$(KERNELS_DIR)/%.$(2) $(HEADERS_TO_BUILD) $(ADDON_H99_FILES) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-addon-kernel-c99flags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-addon-kernel-text-for,$(1))
	@$(CC) $(call get-addon-kernel-c99flags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
# second argument: the C++ addon file suffix being considered.
define make-cxx-addon-rule
$(BASE_OBJ_ADDON_PATH)/%.o: $(ADDON_PATH)/%.$(2) $(HEADERS_TO_BUILD) $(ADDON_HXX_FILES) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-addon-cxxflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-addon-cxxtext-for,$(1))
	@$(CXX) $(call get-addon-cxxflags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
# second argument: the C99 sandbox file suffix being considered.
define make-c99-sandbox-rule
$(BASE_OBJ_SANDBOX_PATH)/%.o: $(SANDBOX_PATH)/%.$(2) $(HEADERS_TO_BUILD) $(SANDBOX_H99_FILES) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-sandbox-c99flags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-sandbox-c99text-for,$(1))
	@$(CC) $(call get-sandbox-c99flags-for,$(1)) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
# second argument: the C++ sandbox file suffix being considered.
define make-cxx-sandbox-rule
$(BASE_OBJ_SANDBOX_PATH)/%.o: $(SANDBOX_PATH)/%.$(2) $(HEADERS_TO_BUILD) $(SANDBOX_HXX_FILES) $(MAKE_DEFS_MK_PATHS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-sandbox-cxxflags-for,$(1)) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-sandbox-cxxtext-for,$(1))
	@$(CXX) $(call get-sandbox-cxxflags-for,$(1)) -c $$< -o $$@
endif
endef

# Define functions to choose the correct sub-configuration name for the
# given kernel set. This function is called when instantiating the
# make-kernels-rule.
get-config-for-kset = $(lastword $(subst :, ,$(filter $(1):%,$(KCONFIG_MAP))))

# Instantiate the build rule for files in the configuration directory for
# each of the sub-configurations in CONFIG_LIST with the CFLAGS designated
# for that sub-configuration.
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-config-rule,$(conf))))

# Instantiate the build rule for framework files. Use the CFLAGS for the
# configuration family, which exists in the directory whose name is equal to
# CONFIG_NAME. Note that this doesn't need to be in a loop since we expect
# CONFIG_NAME to only ever contain a single name. (BTW: If CONFIG_NAME refers
# to a singleton family, then CONFIG_LIST contains CONFIG_NAME as its only
# item.)
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-frame-rule,$(conf))))

# Instantiate the build rule for reference kernel initialization and
# reference kernels for each of the sub-configurations in CONFIG_LIST with
# the CFLAGS designated for that sub-configuration.
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-refinit-rule,$(conf))))
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-refkern-rule,$(conf))))

# Instantiate the build rule for optimized kernels for each of the kernel
# sets in KERNEL_LIST with the CFLAGS designated for the sub-configuration
# specified by the KCONFIG_MAP.
$(foreach suf, $(KERNELS_SRC_SUFS), \
$(foreach kset, $(KERNEL_LIST), $(eval $(call make-kernels-rule,$(kset),$(call get-config-for-kset,$(kset)),$(suf)))))

# Instantiate the build rule for C addon files. Use the CFLAGS for the
# configuration family.
$(foreach suf, $(ADDON_C99_SUFS), \
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-c99-addon-rule,$(conf),$(suf)))))

# Instantiate the build rule for C addon/kernels files. Use the CFLAGS for the
# configuration family.
$(foreach addon, $(ADDON_LIST), \
$(foreach suf, $(ADDON_C99_SUFS), \
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-c99-addon-kers-rule,$(conf),$(suf),$(addon))))))

# Instantiate the build rule for C++ addon files. Use the CFLAGS for the
# configuration family.
$(foreach suf, $(ADDON_CXX_SUFS), \
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-cxx-addon-rule,$(conf),$(suf)))))

# Instantiate the build rule for C sandbox files. Use the CFLAGS for the
# configuration family.
$(foreach suf, $(SANDBOX_C99_SUFS), \
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-c99-sandbox-rule,$(conf),$(suf)))))

# Instantiate the build rule for C++ sandbox files. Use the CXXFLAGS for the
# configuration family.
$(foreach suf, $(SANDBOX_CXX_SUFS), \
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-cxx-sandbox-rule,$(conf),$(suf)))))


# --- All-purpose library rule (static and shared) ---

libblis: check-env $(MK_LIBS)


# --- Static library archiver rules ---

$(LIBBLIS_A_PATH): $(MK_BLIS_OBJS)
ifeq ($(ENABLE_VERBOSE),yes)
ifeq ($(ARG_MAX_HACK),yes)
	$(file > $@.in,$^)
	$(AR) $(ARFLAGS) $@ @$@.in
	$(RM_F) $@.in
	$(RANLIB) $@
else
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
endif
else # ifeq ($(ENABLE_VERBOSE),no)
ifeq ($(ARG_MAX_HACK),yes)
	@echo "Archiving $@"
	@$(file > $@.in,$^)
	@$(AR) $(ARFLAGS) $@ @$@.in
	@$(RM_F) $@.in
	@$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif
endif


# --- Shared library linker rules ---

$(LIBBLIS_SO_PATH): $(MK_BLIS_OBJS)
ifeq ($(ENABLE_VERBOSE),yes)
ifeq ($(ARG_MAX_HACK),yes)
	$(file > $@.in,$^)
	$(LINKER) $(SOFLAGS) -o $(LIBBLIS_SO_OUTPUT_NAME) @$@.in $(LDFLAGS)
	$(RM_F) $@.in
else
	$(LINKER) $(SOFLAGS) -o $(LIBBLIS_SO_OUTPUT_NAME) $^ $(LDFLAGS)
endif
else # ifeq ($(ENABLE_VERBOSE),no)
ifeq ($(ARG_MAX_HACK),yes)
	@echo "Dynamically linking $@"
	@$(file > $@.in,$^)
	@$(LINKER) $(SOFLAGS) -o $(LIBBLIS_SO_OUTPUT_NAME) @$@.in $(LDFLAGS)
	@$(RM_F) $@.in
else
	@echo "Dynamically linking $@"
	@$(LINKER) $(SOFLAGS) -o $(LIBBLIS_SO_OUTPUT_NAME) $^ $(LDFLAGS)
endif
endif

# Local symlink for shared library.
# NOTE: We use a '.loc' suffix to avoid filename collisions in case this
# rule is executed concurrently with the install-lib-symlinks rule, which
# also creates symlinks in the current directory (before installing them).
# NOTE: We don't create any symlinks during Windows builds.
$(LIBBLIS_SO_MAJ_PATH): $(LIBBLIS_SO_PATH)
ifeq ($(IS_WIN),no)
ifeq ($(ENABLE_VERBOSE),yes)
	$(SYMLINK) $(<F) $(@F).loc
	$(MV) $(@F).loc $(BASE_LIB_PATH)/$(@F)
else # ifeq ($(ENABLE_VERBOSE),no)
	@echo "Creating symlink $@"
	@$(SYMLINK) $(<F) $(@F).loc
	@$(MV) $(@F).loc $(BASE_LIB_PATH)/$(@F)
endif
endif


# --- BLAS test suite rules ---

testblas: blastest-run

blastest-f2c: check-env $(BLASTEST_F2C_LIB)

blastest-bin: check-env blastest-f2c $(BLASTEST_DRV_BIN_PATHS)

blastest-run: $(BLASTEST_DRV_BINS_R)

# f2c object file rule.
$(BASE_OBJ_BLASTEST_PATH)/%.o: $(BLASTEST_F2C_SRC_PATH)/%.c $(HEADERS_TO_BUILD)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) $(BLAT_CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) $(BLAT_CFLAGS) -c $< -o $@
endif

# driver object file rule.
$(BASE_OBJ_BLASTEST_PATH)/%.o: $(BLASTEST_DRV_SRC_PATH)/%.c $(HEADERS_TO_BUILD)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) $(BLAT_CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) $(BLAT_CFLAGS) -c $< -o $@
endif

# libf2c library archive rule.
$(BLASTEST_F2C_LIB): $(BLASTEST_F2C_OBJS)
ifeq ($(ENABLE_VERBOSE),yes)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif

$(BASE_OBJ_BLASTEST_PATH)/%.x: $(BASE_OBJ_BLASTEST_PATH)/%.o $(BLASTEST_F2C_LIB) $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $< $(BLASTEST_F2C_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(notdir $(BLASTEST_F2C_LIB)) $(LIBBLIS_LINK) "$(LDFLAGS)"'"
	@$(LINKER) $< $(BLASTEST_F2C_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
endif

# A rule to run ?blat1.x driver files.
define make-run-blat1-rule
run-$(1): $(BASE_OBJ_BLASTEST_PATH)/$(1).x
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) $(BASE_OBJ_BLASTEST_PATH)/$(1).x > out.$(1)
else
	@echo "Running $(1).x > 'out.$(1)'"
	@$(TESTSUITE_WRAPPER) $(BASE_OBJ_BLASTEST_PATH)/$(1).x > out.$(1)
endif
endef

# Instantiate the rule above for each level-1 driver file.
$(foreach name, $(BLASTEST_DRV1_BASES), $(eval $(call make-run-blat1-rule,$(name))))

# A rule to run ?blat2.x and ?blat3.x driver files.
define make-run-blat23-rule
run-$(1): $(BASE_OBJ_BLASTEST_PATH)/$(1).x
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) $(BASE_OBJ_BLASTEST_PATH)/$(1).x < $(BLASTEST_INPUT_PATH)/$(1).in
else
	@echo "Running $(1).x < '$(BLASTEST_INPUT_PATH)/$(1).in' (output to 'out.$(1)')"
	@$(TESTSUITE_WRAPPER) $(BASE_OBJ_BLASTEST_PATH)/$(1).x < $(BLASTEST_INPUT_PATH)/$(1).in
endif
endef

# Instantiate the rule above for each level-2 driver file.
$(foreach name, $(BLASTEST_DRV2_BASES), $(eval $(call make-run-blat23-rule,$(name))))

# Instantiate the rule above for each level-3 driver file.
$(foreach name, $(BLASTEST_DRV3_BASES), $(eval $(call make-run-blat23-rule,$(name))))

# Check the results of the BLAS test suite drivers.
checkblas: blastest-run
ifeq ($(ENABLE_VERBOSE),yes)
	- $(BLASTEST_CHECK_PATH)
else
	@- $(BLASTEST_CHECK_PATH)
endif

# --- BLIS test suite rules ---

testblis: testsuite

testblis-fast: testsuite-run-fast

testblis-md: testsuite-run-md

testblis-salt: testsuite-run-salt

testsuite: testsuite-run

testsuite-bin: check-env $(TESTSUITE_BIN)

# Object file rule.
$(BASE_OBJ_TESTSUITE_PATH)/%.o: $(TESTSUITE_SRC_PATH)/%.c $(HEADERS_TO_BUILD)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(call get-user-cflags-for,$(CONFIG_NAME)) -c $< -o $@
endif

# Testsuite binary rule.
$(TESTSUITE_BIN): $(MK_TESTSUITE_OBJS) $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $(MK_TESTSUITE_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(LIBBLIS_LINK) "$(LDFLAGS)"'"
	@$(LINKER) $(MK_TESTSUITE_OBJS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@
endif

# A rule to run the testsuite using the normal input.* files.
testsuite-run: testsuite-bin
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                   -o $(TESTSUITE_CONF_OPS_PATH) \
	                    > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                    -o $(TESTSUITE_CONF_OPS_PATH) \
	                     > $(TESTSUITE_OUT_FILE)
endif

# A rule to run the testsuite using the input.*.fast files, which
# run a set of tests designed to finish much more quickly.
testsuite-run-fast: testsuite-bin
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_FAST_GEN_PATH) \
	                   -o $(TESTSUITE_FAST_OPS_PATH) \
	                    > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) (fast) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_FAST_GEN_PATH) \
	                    -o $(TESTSUITE_FAST_OPS_PATH) \
	                     > $(TESTSUITE_OUT_FILE)
endif

# A rule to run the testsuite using the input.*.md files, which
# run a set of tests designed to only exercise mixed-datatype gemm.
testsuite-run-md: testsuite-bin
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_MIXD_GEN_PATH) \
	                   -o $(TESTSUITE_MIXD_OPS_PATH) \
	                    > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) (mixed dt) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_MIXD_GEN_PATH) \
	                    -o $(TESTSUITE_MIXD_OPS_PATH) \
	                     > $(TESTSUITE_OUT_FILE)
endif

# A rule to run the testsuite using the input.*.salt files, which
# simulates application-level threading across operation tests.
testsuite-run-salt: testsuite-bin
ifeq ($(ENABLE_VERBOSE),yes)
	$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_SALT_GEN_PATH) \
	                   -o $(TESTSUITE_SALT_OPS_PATH) \
	                    > $(TESTSUITE_OUT_FILE)

else
	@echo "Running $(TESTSUITE_BIN) (salt) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@$(TESTSUITE_WRAPPER) ./$(TESTSUITE_BIN) -g $(TESTSUITE_SALT_GEN_PATH) \
	                    -o $(TESTSUITE_SALT_OPS_PATH) \
	                     > $(TESTSUITE_OUT_FILE)
endif

# Check the results of the BLIS testsuite.
checkblis: testsuite-run
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
endif

# Check the results of the BLIS testsuite (fast).
checkblis-fast: testsuite-run-fast
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
endif

# Check the results of the BLIS testsuite (mixed-datatype).
checkblis-md: testsuite-run-md
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
endif

# Check the results of the BLIS testsuite (salt).
checkblis-salt: testsuite-run-salt
ifeq ($(ENABLE_VERBOSE),yes)
	- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
else
	@- $(TESTSUITE_CHECK_PATH) $(TESTSUITE_OUT_FILE)
endif


# --- AMD's C++ template header test rules ---

# NOTE: The targets below won't work as intended for an out-of-tree build,
# and so it's disabled for now.

#testcpp: testvendcpp

# Recursively run the test for AMD's C++ template header.
#testvendcpp:
#	$(MAKE) -C $(VEND_TESTCPP_PATH)


# --- Install header rules ---

install-headers: check-env $(HEADERS_INSTALLED) install-helper-headers

# Rule for installing main headers.
$(MK_INCL_DIR_INST)/%.h: $(BASE_INC_PATH)/%.h $(CONFIG_MK_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(dir $(@))
	$(INSTALL) -m 0644 $(<) $(dir $(@))
else
	@$(MKDIR) $(dir $(@))
	@echo "Installing $(notdir $(<)) into $(dir $(@))/"
	@$(INSTALL) -m 0644 $(<) $(dir $(@))
endif

$(MK_INCL_DIR_INST)/%.hh: $(VEND_CPP_PATH)/%.hh $(CONFIG_MK_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(dir $(@))
	$(INSTALL) -m 0644 $(<) $(dir $(@))
else
	@$(MKDIR) $(dir $(@))
	@echo "Installing $(notdir $(<)) into $(dir $(@))/"
	@$(INSTALL) -m 0644 $(<) $(dir $(@))
endif

install-helper-headers: check-env $(HELP_HEADERS_INSTALLED)

# A rule to install a helper header file.
define make-helper-header-rule
$(INSTALL_INCDIR)/$(notdir $(1)): $(BUILD_PATH)/$(notdir $(1)) $(CONFIG_MK_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(INSTALL_INCDIR)
	$(INSTALL) -m 0644 $$(<) $$(@)
else
	@$(MKDIR) $(INSTALL_INCDIR)
	@echo "Installing $$(@F) helper header into $(INSTALL_INCDIR)/"
	@$(INSTALL) -m 0644 $$(<) $$(@)
endif
endef

# Instantiate the rule above for each helper header file to install.
$(foreach h, $(HELP_HEADERS_TO_INSTALL), $(eval $(call make-helper-header-rule,$(h))))

# --- Install share rules ---

install-share: check-env $(MK_SHARE_DIR_INST) $(PC_SHARE_DIR_INST)

$(MK_SHARE_DIR_INST): $(CONFIGURE_FILE) $(FRAGS_TO_INSTALL) $(PLUGIN_FRAGS_TO_INSTALL) $(CONFIG_DIR)/$(CONFIG_NAME)/$(MAKE_DEFS_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(@)
	$(MKDIR) $(@)/plugin
	$(INSTALL) -m 0755 $(filter %.sh,$(FRAGS_TO_INSTALL)) $(@)
	$(INSTALL) -m 0644 $(filter-out %.sh,$(FRAGS_TO_INSTALL)) $(@)
	$(INSTALL) -m 0644 $(PLUGIN_FRAGS_TO_INSTALL) $(@)/plugin
	$(INSTALL) -m 0755 $(CONFIGURE_FILE) $(@)/configure-plugin
#	$(MKDIR) -p $(@)/$(CONFIG_DIR)/$(CONFIG_NAME)
#	$(INSTALL) -m 0644 $(CONFIG_DIR)/$(CONFIG_NAME)/$(MAKE_DEFS_FILE) \
#	              $(@)/$(CONFIG_DIR)/$(CONFIG_NAME)
	for THIS_CONFIG in $(FULL_CONFIG_LIST); do \
		$(MKDIR) -p $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
		$(INSTALL) -m 0644 $(CONFIG_DIR)/$$THIS_CONFIG/$(MAKE_DEFS_FILE) \
		              $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
		$(INSTALL) -m 0644 $(CONFIG_DIR)/$$THIS_CONFIG/bli_kernel_defs_$$THIS_CONFIG.h \
		              $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
	done
else
	@$(MKDIR) $(@)
	@$(MKDIR) $(@)/plugin
	@echo "Installing $(notdir $(FRAGS_TO_INSTALL)) into $(@)/"
	@$(INSTALL) -m 0755 $(filter %.sh,$(FRAGS_TO_INSTALL)) $(@)
	@$(INSTALL) -m 0644 $(filter-out %.sh,$(FRAGS_TO_INSTALL)) $(@)
	@echo "Installing $(notdir $(PLUGIN_FRAGS_TO_INSTALL)) into $(@)/plugin/"
	@$(INSTALL) -m 0644 $(PLUGIN_FRAGS_TO_INSTALL) $(@)/plugin
	@echo "Installing $(CONFIGURE_FILE) into $(@)/configure-plugin"
	@$(INSTALL) -m 0755 $(CONFIGURE_FILE) $(@)/configure-plugin
#	@$(MKDIR) -p $(@)/$(CONFIG_DIR)/$(CONFIG_NAME)#\
#	@echo "Installing $(CONFIG_DIR)/$(CONFIG_NAME)/$(MAKE_DEFS_FILE) into $(@)/$(CONFIG_DIR)/$(CONFIG_NAME)"
#	@$(INSTALL) -m 0644 $(CONFIG_DIR)/$(CONFIG_NAME)/$(MAKE_DEFS_FILE) \
#	               $(@)/$(CONFIG_DIR)/$(CONFIG_NAME)/
	@for THIS_CONFIG in $(FULL_CONFIG_LIST); do \
		$(MKDIR) -p $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
		echo "Installing $(CONFIG_DIR)/$$THIS_CONFIG/$(MAKE_DEFS_FILE) into $(@)/$(CONFIG_DIR)/$$THIS_CONFIG"; \
		$(INSTALL) -m 0644 $(CONFIG_DIR)/$$THIS_CONFIG/$(MAKE_DEFS_FILE) \
		              $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
		echo "Installing $(CONFIG_DIR)/$$THIS_CONFIG/bli_kernel_defs_$$THIS_CONFIG.h into $(@)/$(CONFIG_DIR)/$$THIS_CONFIG"; \
		$(INSTALL) -m 0644 $(CONFIG_DIR)/$$THIS_CONFIG/bli_kernel_defs_$$THIS_CONFIG.h \
		              $(@)/$(CONFIG_DIR)/$$THIS_CONFIG; \
	done
endif

$(PC_SHARE_DIR_INST): $(PC_IN_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(@)
	$(shell cat "$(PC_IN_FILE)" \
	| sed -e "s#@PACKAGE_VERSION@#$(VERSION)#g" \
	| sed -e "s#@prefix@#$(prefix)#g" \
	| sed -e "s#@exec_prefix@#$(exec_prefix)#g" \
	| sed -e "s#@libdir@#$(libdir)#g" \
	| sed -e "s#@includedir@#$(includedir)#g" \
	| sed -e "s#@LDFLAGS@#$(LDFLAGS)#g" \
	> "$(PC_OUT_FILE)" )
	$(INSTALL) -m 0644 $(PC_OUT_FILE) $(@)
else
	@$(MKDIR) $(@)
	@echo "Installing $(PC_OUT_FILE) into $(@)/"
	@$(shell cat "$(PC_IN_FILE)" \
	| sed -e "s#@PACKAGE_VERSION@#$(VERSION)#g" \
	| sed -e "s#@prefix@#$(prefix)#g" \
	| sed -e "s#@exec_prefix@#$(exec_prefix)#g" \
	| sed -e "s#@libdir@#$(libdir)#g" \
	| sed -e "s#@includedir@#$(includedir)#g" \
	| sed -e "s#@LDFLAGS@#$(LDFLAGS)#g" \
	> "$(PC_OUT_FILE)" )
	@$(INSTALL) -m 0644 $(PC_OUT_FILE) $(@)
endif

# --- Install library rules ---

install-libs: check-env $(MK_LIBS_INST)

# Install static library.
$(INSTALL_LIBDIR)/%.a: $(BASE_LIB_PATH)/%.a $(CONFIG_MK_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_LIBDIR)/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $< $@
endif

# Install shared library containing .so major, minor, and build versions.
# Note: Installation rules for Windows does not include major, minor, and
# build version numbers.
ifeq ($(IS_WIN),no)

# Linux/OSX library (.so OR .dylib) installation rules.
$(INSTALL_LIBDIR)/%.$(LIBBLIS_SO_MMB_EXT): $(BASE_LIB_PATH)/%.$(SHLIB_EXT) $(CONFIG_MK_FILE)
ifeq ($(ENABLE_VERBOSE),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0755 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_LIBDIR)/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0755 $< $@
endif

else # ifeq ($(IS_WIN),yes)

# Windows library (.dll and .lib) installation rules.
$(INSTALL_LIBDIR)/%.$(SHLIB_EXT): $(BASE_LIB_PATH)/%.$(SHLIB_EXT)
ifeq ($(ENABLE_VERBOSE),yes)
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $(BASE_LIB_PATH)/$(@F) $@
else
	@echo "Installing $(@F) into $(INSTALL_LIBDIR)/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $(BASE_LIB_PATH)/$(@F) $@
endif

$(INSTALL_LIBDIR)/%.$(LIBBLIS_SO_MAJ_EXT): $(BASE_LIB_PATH)/%.$(LIBBLIS_SO_MAJ_EXT)
ifeq ($(ENABLE_VERBOSE),yes)
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $(BASE_LIB_PATH)/$(@F) $@
else
	@echo "Installing $(@F) into $(INSTALL_LIBDIR)/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $(BASE_LIB_PATH)/$(@F) $@
endif

endif # ifeq ($(IS_WIN),no)

# --- Install-symlinks rules ---

install-lib-symlinks: check-env $(MK_LIBS_SYML)

# Note: Symlinks are not installed on Windows.
ifeq ($(IS_WIN),no)

# Install generic shared library symlink.
$(INSTALL_LIBDIR)/%.$(SHLIB_EXT): $(INSTALL_LIBDIR)/%.$(LIBBLIS_SO_MMB_EXT)
ifeq ($(ENABLE_VERBOSE),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_LIBDIR)/
else
	@echo "Installing symlink $(@F) into $(INSTALL_LIBDIR)/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_LIBDIR)/
endif

# Install shared library symlink containing only .so major version.
$(INSTALL_LIBDIR)/%.$(LIBBLIS_SO_MAJ_EXT): $(INSTALL_LIBDIR)/%.$(LIBBLIS_SO_MMB_EXT)
ifeq ($(ENABLE_VERBOSE),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_LIBDIR)/
else
	@echo "Installing symlink $(@F) into $(INSTALL_LIBDIR)/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_LIBDIR)/
endif

endif # ifeq ($(IS_WIN),no)

# --- Query current configuration ---

showconfig: check-env
	@echo "configuration family:       $(CONFIG_NAME)"
	@echo "sub-configurations:         $(CONFIG_LIST)"
	@echo "requisite kernels sets:     $(KERNEL_LIST)"
	@echo "kernel-to-config map:       $(KCONFIG_MAP)"
	@echo "-------------------------"
	@echo "BLIS version string:        $(VERSION)"
	@echo ".so major version:          $(SO_MAJOR)"
	@echo ".so minor.build vers:       $(SO_MINORB)"
	@echo "install libdir:             $(INSTALL_LIBDIR)"
	@echo "install includedir:         $(INSTALL_INCDIR)"
	@echo "install sharedir:           $(INSTALL_SHAREDIR)"
	@echo "debugging status:           $(DEBUG_TYPE)"
	@echo "enable AddressSanitizer?    $(MK_ENABLE_ASAN)"
	@echo "enabled threading model(s): $(THREADING_MODEL)"
	@echo "enable BLAS API?            $(MK_ENABLE_BLAS)"
	@echo "enable CBLAS API?           $(MK_ENABLE_CBLAS)"
	@echo "build static library?       $(MK_ENABLE_STATIC)"
	@echo "build shared library?       $(MK_ENABLE_SHARED)"
	@echo "ARG_MAX hack enabled?       $(ARG_MAX_HACK)"


# --- Clean rules ---

cleanmk:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(FIND) $(CONFIG_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(FRAME_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(REFKERN_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(KERNELS_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
ifneq ($(ADDON_LIST),)
	- $(FIND) $(ADDON_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif
ifneq ($(SANDBOX),)
	- $(FIND) $(SANDBOX_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif
else
	@echo "Removing makefile fragments from $(CONFIG_FRAG_PATH)"
	@- $(FIND) $(CONFIG_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(FRAME_FRAG_PATH)"
	@- $(FIND) $(FRAME_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(REFKERN_FRAG_PATH)"
	@- $(FIND) $(REFKERN_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(KERNELS_FRAG_PATH)"
	@- $(FIND) $(KERNELS_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
ifneq ($(ADDON_LIST),)
	@echo "Removing makefile fragments from $(ADDON_FRAG_PATH)"
	@- $(FIND) $(ADDON_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif
ifneq ($(SANDBOX),)
	@echo "Removing makefile fragments from $(SANDBOX_FRAG_PATH)"
	@- $(FIND) $(SANDBOX_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif
endif
endif

cleanh:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	$(RM_F) $(BLIS_H_FLAT)
	$(RM_F) $(CBLAS_H_FLAT)
else
	@echo "Removing flattened header files from $(BASE_INC_PATH)"
	@$(RM_F) $(BLIS_H_FLAT)
	@$(RM_F) $(CBLAS_H_FLAT)
endif
endif

cleanlib:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	- $(RM_F) $(LIBBLIS_A_PATH)
	- $(RM_F) $(LIBBLIS_SO_PATH)
else
	@echo "Removing object files from $(BASE_OBJ_PATH)"
	@- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing libraries from $(BASE_LIB_PATH)"
	@- $(RM_F) $(LIBBLIS_A_PATH)
	@- $(RM_F) $(LIBBLIS_SO_PATH)
endif
endif

cleantest: cleanblastest cleanblistest

ifeq ($(BUILDING_OOT),no)
cleanblastest: cleanblastesttop cleanblastestdir
else
cleanblastest: cleanblastesttop
endif

cleanblastesttop:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(BLASTEST_F2C_OBJS) $(BLASTEST_DRV_OBJS)
	- $(RM_F) $(BLASTEST_F2C_LIB)
	- $(RM_F) $(BLASTEST_DRV_BIN_PATHS)
	- $(RM_F) $(addprefix out.,$(BLASTEST_DRV_BASES))
else
	@echo "Removing object files from $(BASE_OBJ_BLASTEST_PATH)"
	@- $(RM_F) $(BLASTEST_F2C_OBJS) $(BLASTEST_DRV_OBJS)
	@echo "Removing libf2c.a from $(BASE_OBJ_BLASTEST_PATH)"
	@- $(RM_F) $(BLASTEST_F2C_LIB)
	@echo "Removing binaries from $(BASE_OBJ_BLASTEST_PATH)"
	@- $(RM_F) $(BLASTEST_DRV_BIN_PATHS)
	@echo "Removing driver output files 'out.*'"
	@- $(RM_F) $(addprefix out.,$(BLASTEST_DRV_BASES))
endif # ENABLE_VERBOSE
endif # IS_CONFIGURED

cleanblastestdir:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(FIND) $(BLASTEST_DIR)/$(OBJ_DIR) -name "*.o" | $(XARGS) $(RM_F)
	- $(FIND) $(BLASTEST_DIR) -name "*.x" | $(XARGS) $(RM_F)
	- $(RM_F) $(BLASTEST_DIR)/$(BLASTEST_F2C_LIB_NAME)
	- $(RM_F) $(addprefix $(BLASTEST_DIR)/out.,$(BLASTEST_DRV_BASES))
else
	@echo "Removing object files from ./$(BLASTEST_DIR)/$(OBJ_DIR)"
	@- $(FIND) $(BLASTEST_DIR)/$(OBJ_DIR) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing libf2c.a from ./$(BLASTEST_DIR)"
	@- $(RM_F) $(BLASTEST_DIR)/$(BLASTEST_F2C_LIB_NAME)
	@echo "Removing binaries from ./$(BLASTEST_DIR)"
	@- $(FIND) $(BLASTEST_DIR) -name "*.x" | $(XARGS) $(RM_F)
	@echo "Removing driver output files 'out.*' from ./$(BLASTEST_DIR)"
	@- $(RM_F) $(addprefix $(BLASTEST_DIR)/out.,$(BLASTEST_DRV_BASES))
endif # ENABLE_VERBOSE
endif # IS_CONFIGURED

ifeq ($(BUILDING_OOT),no)
cleanblistest: cleanblistesttop cleanblistestdir
else
cleanblistest: cleanblistesttop
endif

cleanblistesttop:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(MK_TESTSUITE_OBJS)
	- $(RM_F) $(TESTSUITE_BIN)
	- $(RM_F) $(TESTSUITE_OUT_FILE)
else
	@echo "Removing object files from $(BASE_OBJ_TESTSUITE_PATH)"
	@- $(RM_F) $(MK_TESTSUITE_OBJS)
	@echo "Removing binary $(TESTSUITE_BIN)"
	@- $(RM_F) $(TESTSUITE_BIN)
	@echo "Removing $(TESTSUITE_OUT_FILE)"
	@- $(RM_F) $(TESTSUITE_OUT_FILE)
endif # ENABLE_VERBOSE
endif # IS_CONFIGURED

cleanblistestdir:
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(FIND) $(TESTSUITE_DIR)/$(OBJ_DIR) -name "*.o" | $(XARGS) $(RM_F)
	- $(RM_F) $(TESTSUITE_DIR)/$(TESTSUITE_BIN)
#	- $(MAKE) -C $(VEND_TESTCPP_DIR) clean
else
	@echo "Removing object files from $(TESTSUITE_DIR)/$(OBJ_DIR)"
	@- $(FIND) $(TESTSUITE_DIR)/$(OBJ_DIR) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing binary $(TESTSUITE_DIR)/$(TESTSUITE_BIN)"
	@- $(RM_F) $(TESTSUITE_DIR)/$(TESTSUITE_BIN)
#	@$(MAKE) -C $(VEND_TESTCPP_DIR) clean
endif # ENABLE_VERBOSE
endif # IS_CONFIGURED

distclean: cleanmk cleanh cleanlib cleantest
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(BLIS_ADDON_H)
	- $(RM_F) $(BLIS_CONFIG_H)
	- $(RM_F) $(CONFIG_MK_FILE)
	- $(RM_F) $(PC_OUT_FILE)
	- $(RM_RF) $(OBJ_DIR)
	- $(RM_RF) $(LIB_DIR)
	- $(RM_RF) $(INCLUDE_DIR)
else
	@echo "Removing $(BLIS_ADDON_H)"
	@$(RM_F) $(BLIS_ADDON_H)
	@echo "Removing $(BLIS_CONFIG_H)"
	@$(RM_F) $(BLIS_CONFIG_H)
	@echo "Removing $(CONFIG_MK_FILE)"
	@- $(RM_F) $(CONFIG_MK_FILE)
	@echo "Removing $(PC_OUT_FILE)"
	@- $(RM_F) $(PC_OUT_FILE)
	@echo "Removing $(OBJ_DIR)"
	@- $(RM_RF) $(OBJ_DIR)
	@echo "Removing $(LIB_DIR)"
	@- $(RM_RF) $(LIB_DIR)
	@echo "Removing $(INCLUDE_DIR)"
	@- $(RM_RF) $(INCLUDE_DIR)
endif
endif


# --- CHANGELOG rules ---

changelog:
	@echo "Updating '$(DIST_PATH)/$(CHANGELOG)' via '$(GIT_LOG)'"
	@$(GIT_LOG) > $(DIST_PATH)/$(CHANGELOG)


# --- Uninstall rules ---

# NOTE: We can't write these uninstall rules directly in terms of targets
# $(MK_LIBS_VERS_CONF_INST), $(MK_LIBS_INST), and $(MK_INCL_DIR_INST)
# because those targets are already defined in terms of rules that *build*
# those products.

uninstall-libs: check-env
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(MK_LIBS_INST)
else
	@echo "Uninstalling libraries $(notdir $(MK_LIBS_INST)) from $(dir $(firstword $(MK_LIBS_INST)))"
	@- $(RM_F) $(MK_LIBS_INST)
endif

uninstall-lib-symlinks: check-env
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(MK_LIBS_SYML)
else
	@echo "Uninstalling symlinks $(notdir $(MK_LIBS_SYML)) from $(dir $(firstword $(MK_LIBS_SYML)))"
	@- $(RM_F) $(MK_LIBS_SYML)
endif

uninstall-headers: check-env
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_RF) $(MK_INCL_DIR_INST)
	- $(RM_RF) $(HELP_HEADERS_INSTALLED)
else
	@echo "Uninstalling directory '$(notdir $(MK_INCL_DIR_INST))' from $(dir $(MK_INCL_DIR_INST))"
	@- $(RM_RF) $(MK_INCL_DIR_INST)
	@echo "Uninstalling $(notdir $(HELP_HEADERS_INSTALLED)) from $(dir $(INSTALL_INCDIR))"
	@- $(RM_RF) $(HELP_HEADERS_INSTALLED)
endif

uninstall-share: check-env
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_RF) $(MK_SHARE_DIR_INST)
else
	@echo "Uninstalling directory '$(notdir $(MK_SHARE_DIR_INST))' from $(dir $(MK_SHARE_DIR_INST))"
	@- $(RM_RF) $(MK_SHARE_DIR_INST)
endif

# --- Uninstall old rules ---

uninstall-old-libs: $(UNINSTALL_OLD_LIBS) check-env

uninstall-old-symlinks: $(UNINSTALL_OLD_SYML) check-env

uninstall-old-headers: $(UNINSTALL_OLD_HEADERS) check-env

$(UNINSTALL_OLD_LIBS) $(UNINSTALL_OLD_SYML) $(UNINSTALL_OLD_HEADERS): check-env
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $@
else
	@echo "Uninstalling $(@F) from $(@D)/"
	@- $(RM_F) $@
endif

