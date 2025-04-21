# check all setup prerequisites if the command goal is not clean
ifneq ($(MAKECMDGOALS),clean)
ifndef QNN_INCLUDE
$(info "INFO: Qnn include not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
endif
ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path is not set. QNN include paths must be set to obtain BE headers necessary to compile the package")
endif
ifndef QNN_TARGET_LIB
$(info "INFO: Qnn target not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_TARGET_LIB := $(QNN_SDK_ROOT)/lib/aarch64-android
endif
ifeq ($(wildcard $(QNN_TARGET_LIB)),)
ifeq ($(MAKECMDGOALS),htp_aarch64)
$(error "ERROR: QNN_TARGET_LIB is needed to compile package for aarch64")
else ifeq ($(MAKECMDGOALS),all)
$(info "WARNING:QNN_TARGET_LIB may need to be defined to compile packages")
endif
endif

ifndef HEXAGON_SDK_ROOT
$(error "ERROR: HEXAGON_SDK_ROOT is not set. Hexagon-SDK path must be set to the latest hexagon-sdk-x.y.z")
endif

ifeq ($(wildcard $(HEXAGON_SDK_ROOT)),)
$(error "ERROR: HEXAGON_SDK_ROOT is not set correctly. Please set HEXAGON_SDK_ROOT to latest hexagon-sdk-X.Y.Z path")
endif

HEXAGON_SDK_BASE := $(dir $(HEXAGON_SDK_ROOT))

$(info "HEXAGON_SDK_ROOT is [${HEXAGON_SDK_ROOT}]")
# Users should note that the tools version may change between hexagon sdk versions
# Following combination of SDK and Tool version is supported
# fix the sdk root for new versions
HEXAGON_SDK_ROOT_V68 := $(HEXAGON_SDK_ROOT)
HEXAGON_SDK_ROOT_V69 := $(HEXAGON_SDK_ROOT)
HEXAGON_SDK_ROOT_V73 := $(HEXAGON_SDK_ROOT)
HEXAGON_SDK_ROOT_V75 := $(HEXAGON_SDK_ROOT)
HEXAGON_SDK_ROOT_V79 := $(HEXAGON_SDK_ROOT)

#Updated to point to latest sdk to match with libQnnHtp.so
HEXAGON_SDK_ROOT_X86 := $(HEXAGON_SDK_ROOT)
HEXAGON_TOOLS_VERSION_V68 := 8.7.06
HEXAGON_TOOLS_VERSION_V69 := 8.7.06
HEXAGON_TOOLS_VERSION_V73 := 8.7.06
HEXAGON_TOOLS_VERSION_V75 := 8.7.06
HEXAGON_TOOLS_VERSION_V79 := 8.7.06

#Updated to point to latest sdk to match with libQnnHtp.so
HEXAGON_TOOLS_VERSION_X86 := 8.7.06

ifndef ANDROID_NDK_ROOT
ifeq ($(MAKECMDGOALS),htp_aarch64)
$(error "ERROR: ANDROID_NDK_ROOT is not set. Android NDK path must be set to compile package for aarch64")
else ifeq ($(MAKECMDGOALS),all)
$(info "WARNING: ANDROID_NDK_ROOT is not set. Android NDK path must be set to compile package for aarch64")
endif
endif

ifndef PACKAGE_NAME
export
PACKAGE_NAME := $(notdir $(shell pwd))
$(info "INFO: No package name defined. Using current directory name: $(PACKAGE_NAME) as the package name")
endif

WORK := build
SRC_DIR := src
OP_SRC_DIR := src/ops
OP_INCLUDE_DIR := ./include
OP_INCLUDES =  #$(wildcard $(OP_INCLUDE_DIR)/*.h) user defined if any op specific headers are needed, add -I to common flags
LIBRARY_NAME := libQnn$(PACKAGE_NAME).so
SUPPORTED_TARGETS = x86_64-linux-clang hexagon-v68 hexagon-v69 hexagon-v73 hexagon-v75 hexagon-v79 aarch64-android


COMMON_CXX_FLAGS = -std=c++17 -I$(QNN_INCLUDE) -fPIC -Wall -Wreorder -Wno-missing-braces -Wno-unused-function
COMMON_CXX_FLAGS += -Werror -Wno-format -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++
COMMON_CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

X86_LIBNATIVE_RELEASE_DIR := $(HEXAGON_SDK_ROOT_X86)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_X86)/Tools

# Ensure hexagon sdk tool version can be retrieved
ifeq ($(wildcard $(X86_LIBNATIVE_RELEASE_DIR)/.),)
$(error "Cannot retrieve hexagon tools from: $(X86_LIBNATIVE_RELEASE_DIR).  \
         \
         Please check that hexagon tools version is correct. Expected: $(HEXAGON_TOOLS_VERSION_X86)")
endif

#Check tools for hexagon_v68 are present.
ifeq ($(MAKECMDGOALS),htp_v68)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V68)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V68 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V68)")
endif
endif

ifeq ($(MAKECMDGOALS),htp_v69)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V69)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V69 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V69)")
endif
endif

ifeq ($(MAKECMDGOALS),htp_v73)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V73)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V73 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V73)")
endif
endif

ifeq ($(MAKECMDGOALS),htp_v75)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V75)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V75 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V75)")
endif
endif

#Check tools for hexagon_v79 are present.
ifeq ($(MAKECMDGOALS),htp_v79)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V79)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V79 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V79)")
endif
endif



endif
OP_SOURCES = $(wildcard $(OP_SRC_DIR)/*.cpp)
OTHER_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
HFILES = $(wildcard $(QNN_INCLUDE)/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/core/*.h)
OP_OBJS = $(patsubst $(SRC_DIR)/%,%,$(patsubst %.cpp,%.o,$(OP_SOURCES)))
OTHER_OBJS =  $(patsubst $(SRC_DIR)/%,%,$(patsubst %.cpp,%.o,$(OTHER_SOURCES)))

#======= Assembly ========
OP_SOURCES_ASM_X86 += $(wildcard $(OP_SRC_DIR)/x86_asm/*.S)
OP_OBJS_ASM_X86 += $(subst /x86_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_X86))))
OP_SOURCES_ASM_V68 += $(wildcard $(OP_SRC_DIR)/v68_asm/*.S)
OP_OBJS_ASM_V68 += $(subst /v68_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_V68))))
OP_SOURCES_ASM_V69 += $(wildcard $(OP_SRC_DIR)/v69_asm/*.S)
OP_OBJS_ASM_V69 += $(subst /v69_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_V69))))
OP_SOURCES_ASM_V73 += $(wildcard $(OP_SRC_DIR)/v73_asm/*.S)
OP_OBJS_ASM_V73 += $(subst /v73_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_V73))))
OP_SOURCES_ASM_V75 += $(wildcard $(OP_SRC_DIR)/v75_asm/*.S)
OP_OBJS_ASM_V75 += $(subst /v75_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_V75))))
OP_SOURCES_ASM_V79 += $(wildcard $(OP_SRC_DIR)/v79_asm/*.S)
OP_OBJS_ASM_V79 += $(subst /v79_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_V79))))

OP_SOURCES_ASM_ANDROID += $(wildcard $(OP_SRC_DIR)/android_asm/*.S)
OP_OBJS_ASM_ANDROID += $(subst /android_asm/,/,$(patsubst $(SRC_DIR)/%,%,$(patsubst %.S,%.o,$(OP_SOURCES_ASM_ANDROID))))


all: htp_v68 htp_x86 htp_aarch64

#============================================================================================================
# Setup compiler, compiler instructions and linker for x86
X86_CXX ?= clang++-9
# Checking if clang++-9 is present. If not switch to clang++
ifeq ($(shell $(X86_CXX) -v 2>&1 | grep -c "clang version"), 0)
  X86_CXX := clang++
endif
X86_LDFLAGS:= -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread -L$(QNN_SDK_ROOT)/lib/x86_64-linux-clang -lHtpPrepare
X86_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX
X86_CXX_FLAGS = $(COMMON_CXX_FLAGS) $(X86_C_FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof
linux_objs =
#============================================================================================================
# Setup compiler, compiler instructions and linker for hexagon
HEXAGON_CXX_FLAGS := $(COMMON_CXX_FLAGS) -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -O2 -Wno-reorder -DPREPARE_DISABLED

HEXAGON_CXX_FLAGS_V68 := $(HEXAGON_CXX_FLAGS) -mv68 -I$(HEXAGON_SDK_ROOT_V68)/rtos/qurt/computev68/include/qurt -I$(HEXAGON_SDK_ROOT_V68)/rtos/qurt/computev68/include/posix -I$(HEXAGON_SDK_ROOT_V68)/incs -I$(HEXAGON_SDK_ROOT_V68)/incs/stddef
HEXAGON_CXX_FLAGS_V69 := $(HEXAGON_CXX_FLAGS) -mv69 -I$(HEXAGON_SDK_ROOT_V69)/rtos/qurt/computev69/include/qurt -I$(HEXAGON_SDK_ROOT_V69)/rtos/qurt/computev69/include/posix -I$(HEXAGON_SDK_ROOT_V69)/incs -I$(HEXAGON_SDK_ROOT_V69)/incs/stddef
HEXAGON_CXX_FLAGS_V73 := $(HEXAGON_CXX_FLAGS) -mv73 -I$(HEXAGON_SDK_ROOT_V73)/rtos/qurt/computev73/include/qurt -I$(HEXAGON_SDK_ROOT_V73)/rtos/qurt/computev73/include/posix -I$(HEXAGON_SDK_ROOT_V73)/incs -I$(HEXAGON_SDK_ROOT_V73)/incs/stddef
HEXAGON_CXX_FLAGS_V75 := $(HEXAGON_CXX_FLAGS) -mv75 -I$(HEXAGON_SDK_ROOT_V75)/rtos/qurt/computev75/include/qurt -I$(HEXAGON_SDK_ROOT_V75)/rtos/qurt/computev75/include/posix -I$(HEXAGON_SDK_ROOT_V75)/incs -I$(HEXAGON_SDK_ROOT_V75)/incs/stddef
HEXAGON_CXX_FLAGS_V79 := $(HEXAGON_CXX_FLAGS) -mv79 -I$(HEXAGON_SDK_ROOT_V79)/rtos/qurt/computev79/include/qurt -I$(HEXAGON_SDK_ROOT_V79)/rtos/qurt/computev79/include/posix -I$(HEXAGON_SDK_ROOT_V79)/incs -I$(HEXAGON_SDK_ROOT_V79)/incs/stddef


HEXAGON_CXX_V68 := $(HEXAGON_SDK_ROOT_V68)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V68)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V69 := $(HEXAGON_SDK_ROOT_V69)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V69)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V73 := $(HEXAGON_SDK_ROOT_V73)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V73)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V75 := $(HEXAGON_SDK_ROOT_V75)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V75)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V79 := $(HEXAGON_SDK_ROOT_V79)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V79)/Tools/bin/hexagon-clang++


HEX_LDFLAGS =
hexagon_objs =
#============================================================================================================
# Setup compiler, compiler instructions and linker for aarch64
AARCH64_C__FLAGS = -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX -DANDROID
AARCH64_CXX_FLAGS = $(COMMON_CXX_FLAGS) $(AARCH64_C__FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof  -Wno-unused-variable -Wno-unused-parameter -Wno-missing-braces -Wno-sign-compare -Wno-unused-private-field -Wno-unused-variable -Wno-ignored-qualifiers -Wno-missing-field-initializers
ARM_CLANG_OPTS =--target=aarch64-none-linux-android21 --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot -stdlib=libc++ -static-libstdc++
AARCH64_CXX = $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ $(ARM_CLANG_OPTS)
AARCH64_LDFLAGS = -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare
aarch64_objs =
#============================================================================================================
# Setup targets and goals

htp_x86: X86_BUILD

htp_v68: HEXAGON_BUILD_V68

htp_v69: HEXAGON_BUILD_V69

htp_v73: HEXAGON_BUILD_V73

htp_v75: HEXAGON_BUILD_V75

htp_v79: HEXAGON_BUILD_V79



htp_aarch64: AARCH64_BUILD

AARCH64_BUILD: $(WORK)/aarch64-android/$(LIBRARY_NAME)

HEXAGON_BUILD_V68:  $(WORK)/hexagon-v68/$(LIBRARY_NAME)

HEXAGON_BUILD_V69:  $(WORK)/hexagon-v69/$(LIBRARY_NAME)

HEXAGON_BUILD_V73:  $(WORK)/hexagon-v73/$(LIBRARY_NAME)

HEXAGON_BUILD_V75:  $(WORK)/hexagon-v75/$(LIBRARY_NAME)

HEXAGON_BUILD_V79:  $(WORK)/hexagon-v79/$(LIBRARY_NAME)



X86_BUILD: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)


define build_objs =
ifneq ($(filter $(2),$(SUPPORTED_TARGETS)),)
$(2)_objs += $(foreach x,$(1),$(WORK)/$(2)/$(x))
else
$$(error "Unknown target option provided: $(2): Supported targets are: $(SUPPORTED_TARGETS)")
endif
endef

$(eval $(call build_objs,$(OTHER_OBJS),x86_64-linux-clang))
$(eval $(call build_objs,$(OP_OBJS),x86_64-linux-clang))
$(eval $(call build_objs,$(OP_OBJS_ASM_X86),x86_64-linux-clang))
$(eval $(call build_objs,$(OTHER_OBJS),hexagon-v68))
$(eval $(call build_objs,$(OP_OBJS),hexagon-v68))
$(eval $(call build_objs,$(OP_OBJS_ASM_V68),hexagon-v68))
$(eval $(call build_objs,$(OTHER_OBJS),hexagon-v69))
$(eval $(call build_objs,$(OP_OBJS),hexagon-v69))
$(eval $(call build_objs,$(OP_OBJS_ASM_V69),hexagon-v69))
$(eval $(call build_objs,$(OTHER_OBJS),hexagon-v73))
$(eval $(call build_objs,$(OP_OBJS),hexagon-v73))
$(eval $(call build_objs,$(OP_OBJS_ASM_V73),hexagon-v73))
$(eval $(call build_objs,$(OTHER_OBJS),hexagon-v75))
$(eval $(call build_objs,$(OP_OBJS),hexagon-v75))
$(eval $(call build_objs,$(OP_OBJS_ASM_V75),hexagon-v75))
$(eval $(call build_objs,$(OTHER_OBJS),hexagon-v79))
$(eval $(call build_objs,$(OP_OBJS),hexagon-v79))
$(eval $(call build_objs,$(OP_OBJS_ASM_V75),hexagon-v79))

$(eval $(call build_objs,$(OTHER_OBJS),aarch64-android))
$(eval $(call build_objs,$(OP_OBJS),aarch64-android))
$(eval $(call build_objs,$(OP_OBJS_ASM_ANDROID),aarch64-android))

# x86
$(WORK)/x86_64-linux-clang $(WORK)/hexagon-v68 $(WORK)/hexagon-v69 $(WORK)/hexagon-v73 $(WORK)/hexagon-v75 $(WORK)/hexagon-v79 $(WORK)/aarch64-android:
	@mkdir -p $@/ops

$(WORK)/x86_64-linux-clang/%.o: $(SRC_DIR)/%.cpp | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/ops/%.o: $(OP_SRC_DIR)/x86_asm/%.S | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(x86_64-linux-clang_objs) | $(HFILES)
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(X86_LDFLAGS)

# v68
$(WORK)/hexagon-v68/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v68
	$(HEXAGON_CXX_V68) $(HEXAGON_CXX_FLAGS_V68) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v68/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/hexagon-v68
	$(HEXAGON_CXX_V68) $(HEXAGON_CXX_FLAGS_V68) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v68/ops/%.o: $(OP_SRC_DIR)/v68_asm/%.S | $(WORK)/hexagon-v68
	$(HEXAGON_CXX_V68) $(HEXAGON_CXX_FLAGS_V68) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v68/$(LIBRARY_NAME): $(hexagon-v68_objs) | $(HFILES)
	$(HEXAGON_CXX_V68) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

# v69
$(WORK)/hexagon-v69/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v69
	$(HEXAGON_CXX_V69) $(HEXAGON_CXX_FLAGS_V69) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v69/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/hexagon-v69
	$(HEXAGON_CXX_V69) $(HEXAGON_CXX_FLAGS_V69) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v69/ops/%.o: $(OP_SRC_DIR)/v69_asm/%.S | $(WORK)/hexagon-v69
	$(HEXAGON_CXX_V69) $(HEXAGON_CXX_FLAGS_V69) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v69/$(LIBRARY_NAME): $(hexagon-v69_objs) | $(HFILES)
	$(HEXAGON_CXX_V69) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

# v73
$(WORK)/hexagon-v73/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v73
	$(HEXAGON_CXX_V73) $(HEXAGON_CXX_FLAGS_V73) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v73/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/hexagon-v73
	$(HEXAGON_CXX_V73) $(HEXAGON_CXX_FLAGS_V73) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v73/ops/%.o: $(OP_SRC_DIR)/v73_asm/%.S | $(WORK)/hexagon-v73
	$(HEXAGON_CXX_V73) $(HEXAGON_CXX_FLAGS_V73) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v73/$(LIBRARY_NAME): $(hexagon-v73_objs) | $(HFILES)
	$(HEXAGON_CXX_V73) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

#v75
$(WORK)/hexagon-v75/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v75
	$(HEXAGON_CXX_V75) $(HEXAGON_CXX_FLAGS_V75) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v75/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/hexagon-v75
	$(HEXAGON_CXX_V75) $(HEXAGON_CXX_FLAGS_V75) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v75/ops/%.o: $(OP_SRC_DIR)/v75_asm/%.S | $(WORK)/hexagon-v75
	$(HEXAGON_CXX_V75) $(HEXAGON_CXX_FLAGS_V75) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v75/$(LIBRARY_NAME): $(hexagon-v75_objs) | $(HFILES)
	$(HEXAGON_CXX_V75) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

#v79
$(WORK)/hexagon-v79/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v79
	$(HEXAGON_CXX_V79) $(HEXAGON_CXX_FLAGS_V79) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v79/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/hexagon-v79
	$(HEXAGON_CXX_V79) $(HEXAGON_CXX_FLAGS_V79) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v79/ops/%.o: $(OP_SRC_DIR)/v79_asm/%.S | $(WORK)/hexagon-v79
	$(HEXAGON_CXX_V79) $(HEXAGON_CXX_FLAGS_V79) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v79/$(LIBRARY_NAME): $(hexagon-v79_objs) | $(HFILES)
	$(HEXAGON_CXX_V79) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)



# aarch64
$(WORK)/aarch64-android/%.o: $(SRC_DIR)/%.cpp | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/aarch64-android/ops/%.o: $(OP_SRC_DIR)/%.cpp | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/aarch64-android/ops/%.o: $(OP_SRC_DIR)/android_asm/%.S | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(aarch64-android_objs) | $(HFILES)
	$(AARCH64_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(AARCH64_LDFLAGS)

clean:
	-rm -rf $(WORK)

.PHONY: all clean
