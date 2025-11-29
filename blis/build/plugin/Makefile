#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2022, Advanced Micro Devices, Inc.
#  Copyright (C) 2023, Southern Methodist University
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
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all \
        plugin \
        showconfig \
        clean cleanmk cleanlib distclean \
        check-env check-env-make-defs check-env-fragments check-env-mk


#
# --- Include config makefile definitions --------------------------------------
#

# Define the name of the config makefile.
CONFIG_MK_FILE := config.mk

# Include the configuration file.
-include $(CONFIG_MK_FILE)


#
# --- Include common makefile definitions --------------------------------------
#

INC_PATH           := $(includedir)/blis

# Define the name of the common makefile.
COMMON_MK_FILE := $(sharedir)/blis/common.mk

# Include the configuration file.
include $(COMMON_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(COMMON_MK_INCLUDED)),yes)
COMMON_MK_PRESENT := yes
else
COMMON_MK_PRESENT := no
endif

# Source suffixes.
CONFIG_SRC_SUFS   := c cxx cpp
KERNELS_SRC_SUFS  := c cxx cpp s S
REFKERN_SRC_SUFS  := c cxx cpp
FRAME_SRC_SUFS    := c cxx cpp

# Make sure the plugin path is included when searching for headers (e.g. bli_plugin_<name>.h).
CINCFLAGS         += -I$(DIST_PATH)

PLUGIN_A_PATH     := $(BASE_LIB_PATH)/libblis_$(PLUGIN_NAME).a
PLUGIN_SO_PATH    := $(BASE_LIB_PATH)/libblis_$(PLUGIN_NAME).$(SHLIB_EXT)

# Specify the shared library's 'soname' field.
# NOTE: The flag for creating shared objects is different for Linux and OS X.
LDFLAGS    += -L$(libdir) -lblis
ifeq ($(OS_NAME),Darwin)
# OS X shared library link flags.
SOFLAGS    := -dynamiclib
else
SOFLAGS    := -shared
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

# --- Determine which libraries to build ---

MK_LIBS                   :=

ifeq ($(MK_ENABLE_STATIC),yes)
MK_LIBS                   += $(PLUGIN_A_PATH)
endif
ifeq ($(MK_ENABLE_SHARED),yes)
MK_LIBS                   += $(PLUGIN_SO_PATH)
endif

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
MK_REFKERN_OBJS     := $(foreach suf, $(REFKERN_SRC_SUFS), \
                           $(foreach arch, $(CONFIG_LIST), \
                               $(patsubst $(REFKERN_PATH)/%_$(REFNM).$(suf), \
                                     $(BASE_OBJ_REFKERN_PATH)/$(arch)/%_$(arch)_$(REFNM).o, \
                                     $(filter %.$(suf), $(MK_REFKERN_SRC)) \
                                ) \
                            ) \
                        )

# Generate object file paths for all of the portable framework source code.
MK_FRAME_OBJS       := $(call gen-obj-paths-from-src,$(FRAME_SRC_SUFS),$(MK_FRAME_SRC),$(FRAME_PATH),$(BASE_OBJ_FRAME_PATH))

# Combine all of the object files into some readily-accessible variables.
MK_PLUGIN_OBJS      := $(MK_CONFIG_OBJS) \
                       $(MK_KERNELS_OBJS) \
                       $(MK_REFKERN_OBJS) \
                       $(MK_FRAME_OBJS)


#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: libs

libs: plugin

clean: cleanlib


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


# --- General source code / object code rules ---

# FGVZ: Add support for compiling .s and .S files in 'config'/'kernels'
# directories.
#  - May want to add an extra foreach loop around function eval/call.

# first argument: a configuration name from config_list, used to look up the
# CFLAGS to use during compilation.
define make-config-rule
$(BASE_OBJ_CONFIG_PATH)/$(1)/%.o: $(CONFIG_PATH)/$(1)/%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-config-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-config-text-for,$(1))
	@$(CC) $(call get-config-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_CONFIG_PATH)/$(1)/%.o: $(CONFIG_PATH)/$(1)/%.cxx $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-config-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-config-cxxtext-for,$(1))
	@$(CXX) $(call get-config-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_CONFIG_PATH)/$(1)/%.o: $(CONFIG_PATH)/$(1)/%.cpp $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-config-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-config-cxxtext-for,$(1))
	@$(CXX) $(call get-config-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
# The 'trailing' % is important so that these are technically pattern rules and the appropriate one can be
# selected based on the suffix of bli_cntx_ref.
define make-refinit-rule
$(BASE_OBJ_REFKERN_PATH)/$(1)/bli_cntx_$(1)_ref%.o: $(REFKERN_PATH)/bli_cntx_ref%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-refinit-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refinit-text-for,$(1))
	@$(CC) $(call get-refinit-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_REFKERN_PATH)/$(1)/bli_cntx_$(1)_ref%.o: $(REFKERN_PATH)/bli_cntx_ref%.cpp $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-refinit-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refinit-cxxtext-for,$(1))
	@$(CXX) $(call get-refinit-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_REFKERN_PATH)/$(1)/bli_cntx_$(1)_ref%.o: $(REFKERN_PATH)/bli_cntx_ref%.cxx $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-refinit-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refinit-cxxtext-for,$(1))
	@$(CXX) $(call get-refinit-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
define make-refkern-rule
$(BASE_OBJ_REFKERN_PATH)/$(1)/%_$(1)_ref.o: $(REFKERN_PATH)/%_ref.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-refkern-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refkern-text-for,$(1))
	@$(CC) $(call get-refkern-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_REFKERN_PATH)/$(1)/%_$(1)_ref.o: $(REFKERN_PATH)/%_ref.cpp $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-refkern-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refkern-cxxtext-for,$(1))
	@$(CXX) $(call get-refkern-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_REFKERN_PATH)/$(1)/%_$(1)_ref.o: $(REFKERN_PATH)/%_ref.cxx $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-refkern-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-refkern-cxxtext-for,$(1))
	@$(CXX) $(call get-refkern-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
endef

# first argument: a configuration name from the union of config_list and
# config_name, used to look up the CFLAGS to use during compilation.
define make-frame-rule
$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.c $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-frame-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-text-for,$(1))
	@$(CC) $(call get-frame-cflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.cxx $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-frame-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-cxxtext-for,$(1))
	@$(CXX) $(call get-frame-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif

$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.cpp $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-frame-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-frame-cxxtext-for,$(1))
	@$(CXX) $(call get-frame-cxxflags-for,$(1)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
endef

# first argument: a kernel set (name) being targeted (e.g. haswell).
# second argument: the configuration whose CFLAGS we should use in compilation.
# third argument: the kernel file suffix being considered.
define make-kernels-rule
$(BASE_OBJ_KERNELS_PATH)/$(1)/%.o: $(KERNELS_PATH)/$(1)/%.$(3) $(BLIS_H_FLAT) $(MAKE_DEFS_MK_PATHS)
	@mkdir -p $$(dir $$@)
ifeq ($(3),$(filter cxx cpp,$(3)))
ifeq ($(ENABLE_VERBOSE),yes)
	$(CXX) $(call get-kernel-cxxflags-for,$(2)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-kernel-cxxtext-for,$(2))
	@$(CXX) $(call get-kernel-cxxflags-for,$(2)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
else
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(call get-kernel-cflags-for,$(2)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
else
	@echo "Compiling $$@" $(call get-kernel-text-for,$(2))
	@$(CC) $(call get-kernel-cflags-for,$(2)) -DBLIS_PNAME=$(PLUGIN_NAME) -c $$< -o $$@
endif
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

# Instantiate the build rule for reference kernel initialization and
# reference kernels for each of the sub-configurations in CONFIG_LIST with
# the CFLAGS designated for that sub-configuration.
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-refinit-rule,$(conf))))
$(foreach conf, $(CONFIG_LIST), $(eval $(call make-refkern-rule,$(conf))))

# Instantiate the build rule for framework files. Use the CFLAGS for the
# configuration family, which exists in the directory whose name is equal to
# CONFIG_NAME. Note that this doesn't need to be in a loop since we expect
# CONFIG_NAME to only ever contain a single name. (BTW: If CONFIG_NAME refers
# to a singleton family, then CONFIG_LIST contains CONFIG_NAME as its only
# item.)
$(foreach conf, $(CONFIG_NAME), $(eval $(call make-frame-rule,$(conf))))

# Instantiate the build rule for optimized kernels for each of the kernel
# sets in KERNEL_LIST with the CFLAGS designated for the sub-configuration
# specified by the KCONFIG_MAP.
$(foreach suf, $(KERNELS_SRC_SUFS), \
$(foreach kset, $(KERNEL_LIST), $(eval $(call make-kernels-rule,$(kset),$(call get-config-for-kset,$(kset)),$(suf)))))


# --- All-purpose library rule (static and shared) ---

plugin: check-env $(MK_LIBS)


# --- Static library archiver rules ---

$(PLUGIN_A_PATH): $(MK_PLUGIN_OBJS)
	@mkdir -p $(dir $@)
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

$(PLUGIN_SO_PATH): $(MK_PLUGIN_OBJS)
	@mkdir -p $(dir $@)
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
	- $(FIND) $(REFKERN_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(KERNELS_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
else
	@echo "Removing makefile fragments from $(CONFIG_FRAG_PATH)"
	@- $(FIND) $(CONFIG_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(REFKERN_FRAG_PATH)"
	@- $(FIND) $(REFKERN_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(KERNELS_FRAG_PATH)"
	@- $(FIND) $(KERNELS_FRAG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
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

distclean: cleanmk cleanlib
ifeq ($(IS_CONFIGURED),yes)
ifeq ($(ENABLE_VERBOSE),yes)
	- $(RM_F) $(CONFIG_MK_FILE)
	- $(RM_RF) $(OBJ_DIR)
	- $(RM_RF) $(LIB_DIR)
else
	@echo "Removing $(CONFIG_MK_FILE)"
	@- $(RM_F) $(CONFIG_MK_FILE)
	@echo "Removing $(OBJ_DIR)"
	@- $(RM_RF) $(OBJ_DIR)
	@echo "Removing $(LIB_DIR)"
	@- $(RM_RF) $(LIB_DIR)
endif
endif

