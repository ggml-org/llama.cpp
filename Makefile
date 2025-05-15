ifndef LLAMA_MAKEFILE
$(error The Makefile build is deprecated. Use the CMake build instead. For more details, see https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
endif

# Define the default target now so that it is always the first target
BUILD_TARGETS = \
	libllava.a \
	llama-batched \
	llama-batched-bench \
	llama-bench \
	llama-cli \
	llama-convert-llama2c-to-ggml \
	llama-embedding \
	llama-eval-callback \
	llama-export-lora \
	llama-gbnf-validator \
	llama-gguf \
	llama-gguf-hash \
	llama-gguf-split \
	llama-gritlm \
	llama-imatrix \
	llama-infill \
	llama-llava-cli \
	llama-minicpmv-cli\
	llama-qwen2vl-cli\
	llama-lookahead \
	llama-lookup \
	llama-lookup-create \
	llama-lookup-merge \
	llama-lookup-stats \
	llama-parallel \
	llama-passkey \
	llama-perplexity \
	llama-q8dot \
	llama-quantize \
	llama-quantize-stats \
	llama-retrieval \
	llama-save-load-state \
	llama-server \
	llama-simple \
	llama-simple-chat \
	llama-run \
	llama-speculative \
	llama-tokenize \
	llama-vdot \
	llama-cvector-generator \
	llama-gen-docs \
	tests/test-c.o

# Binaries only useful for tests
TEST_TARGETS = \
	tests/test-arg-parser \
	tests/test-autorelease \
	tests/test-backend-ops \
	tests/test-chat \
	tests/test-chat-template \
	tests/test-double-float \
	tests/test-grammar-integration \
	tests/test-grammar-parser \
	tests/test-json-schema-to-grammar \
	tests/test-llama-grammar \
	tests/test-log \
	tests/test-model-load-cancel \
	tests/test-quantize-fns \
	tests/test-quantize-perf \
	tests/test-rope \
	tests/test-sampling \
	tests/test-tokenizer-0 \
	tests/test-tokenizer-1-bpe \
	tests/test-tokenizer-1-spm
#	tests/test-opt \

# Legacy build targets that were renamed in #7809, but should still be removed when the project is cleaned
LEGACY_TARGETS_CLEAN = main quantize quantize-stats perplexity imatrix embedding vdot q8dot convert-llama2c-to-ggml \
	simple batched batched-bench save-load-state server gguf gguf-split eval-callback llama-bench libllava.a llava-cli baby-llama \
	retrieval speculative infill tokenize parallel export-lora lookahead lookup passkey gritlm

# Legacy build targets that were renamed in #7809, but we want to build binaries that for them that output a deprecation warning if people try to use them.
#  We don't want to clutter things too much, so we only build replacements for the most commonly used binaries.
LEGACY_TARGETS_BUILD = main quantize perplexity embedding server

# Deprecation aliases
ifdef LLAMA_CUBLAS
$(error LLAMA_CUBLAS is removed. Use GGML_CUDA instead.)
endif

ifdef LLAMA_CUDA
GGML_CUDA := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_KOMPUTE
GGML_KOMPUTE := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_METAL
GGML_METAL := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_RPC
GGML_RPC := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_SYCL
GGML_SYCL := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_SYCL_F16
GGML_SYCL_F16 := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_OPENBLAS
GGML_OPENBLAS := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_OPENBLAS64
GGML_OPENBLAS64 := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_BLIS
GGML_BLIS := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_NO_LLAMAFILE
GGML_NO_LLAMAFILE := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_NO_ACCELERATE
GGML_NO_ACCELERATE := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_NO_OPENMP
GGML_NO_OPENMP := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_NO_METAL
GGML_NO_METAL := 1
DEPRECATE_WARNING := 1
endif

ifdef LLAMA_DISABLE_LOGS
REMOVE_WARNING := 1
endif

ifdef LLAMA_SERVER_VERBOSE
REMOVE_WARNING := 1
endif

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

# In GNU make default CXX is g++ instead of c++.  Let's fix that so that users
# of non-gcc compilers don't have to provide g++ alias or wrapper.
DEFCC  := cc
DEFCXX := c++
ifeq ($(origin CC),default)
CC  := $(DEFCC)
endif
ifeq ($(origin CXX),default)
CXX := $(DEFCXX)
endif

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifndef GGML_NO_METAL
		GGML_METAL := 1
	endif

	GGML_NO_OPENMP := 1

	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

ifdef GGML_METAL
	GGML_METAL_EMBED_LIBRARY := 1
endif

ifdef GGML_RPC
	BUILD_TARGETS += rpc-server
endif

ifdef GGML_VULKAN
	BUILD_TARGETS += vulkan-shaders-gen
endif

default: $(BUILD_TARGETS) $(LEGACY_TARGETS_BUILD)

test: $(TEST_TARGETS)
	@failures=0; \
	for test_target in $(TEST_TARGETS); do \
		if [ "$$test_target" = "tests/test-tokenizer-0" ]; then \
			./$$test_target $(CURDIR)/models/ggml-vocab-llama-spm.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-llama-bpe.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-phi-3.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-falcon.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-bert-bge.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-starcoder.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-gpt-2.gguf; \
			./$$test_target $(CURDIR)/models/ggml-vocab-refact.gguf; \
		elif [ "$$test_target" = "tests/test-tokenizer-1-spm" ]; then \
			continue; \
		elif [ "$$test_target" = "tests/test-tokenizer-1-bpe" ]; then \
			continue; \
		else \
			echo "Running test $$test_target..."; \
			./$$test_target; \
		fi; \
		if [ $$? -ne 0 ]; then \
			printf 'Test %s FAILED!\n\n' $$test_target; \
			failures=$$(( failures + 1 )); \
		else \
			printf 'Test %s passed.\n\n' $$test_target; \
		fi; \
	done; \
	if [ $$failures -gt 0 ]; then \
		printf '\n%s tests failed.\n' $$failures; \
		exit 1; \
	fi
	@echo 'All tests passed.'

all: $(BUILD_TARGETS) $(TEST_TARGETS) $(LEGACY_TARGETS_BUILD)

ifdef RISCV_CROSS_COMPILE
CC	:= riscv64-unknown-linux-gnu-gcc
CXX	:= riscv64-unknown-linux-gnu-g++
endif

#
# Compile flags
#

# keep standard at C11 and C++17
MK_CPPFLAGS  = -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -DGGML_USE_CPU
MK_CFLAGS    = -std=c11   -fPIC
MK_CXXFLAGS  = -std=c++17 -fPIC
MK_NVCCFLAGS = -std=c++17

ifdef LLAMA_NO_CCACHE
GGML_NO_CCACHE := 1
DEPRECATE_WARNING := 1
endif

ifndef GGML_NO_CCACHE
CCACHE := $(shell which ccache)
ifdef CCACHE
export CCACHE_SLOPPINESS = time_macros
$(info I ccache found, compilation results will be cached. Disable with GGML_NO_CCACHE.)
CC    := $(CCACHE) $(CC)
CXX   := $(CCACHE) $(CXX)
else
$(info I ccache not found. Consider installing it for faster compilation.)
endif # CCACHE
endif # GGML_NO_CCACHE

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
MK_CPPFLAGS += -D_XOPEN_SOURCE=600

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -U_XOPEN_SOURCE -D_XOPEN_SOURCE=700
endif

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
ifeq ($(UNAME_S),Linux)
	MK_CPPFLAGS += -D_GNU_SOURCE
	MK_LDFLAGS  += -ldl
endif

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
ifeq ($(UNAME_S),Darwin)
	MK_CPPFLAGS += -D_DARWIN_C_SOURCE
endif
ifeq ($(UNAME_S),DragonFly)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
ifeq ($(UNAME_S),FreeBSD)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif
ifeq ($(UNAME_S),NetBSD)
	MK_CPPFLAGS += -D_NETBSD_SOURCE
endif
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -D_BSD_SOURCE
endif

ifdef GGML_SCHED_MAX_COPIES
	MK_CPPFLAGS += -DGGML_SCHED_MAX_COPIES=$(GGML_SCHED_MAX_COPIES)
endif

ifdef LLAMA_DEBUG
	MK_CFLAGS    += -O0 -g
	MK_CXXFLAGS  += -O0 -g
	MK_LDFLAGS   += -g
	MK_NVCCFLAGS += -O0 -g

	ifeq ($(UNAME_S),Linux)
		MK_CPPFLAGS += -D_GLIBCXX_ASSERTIONS
	endif
else
	MK_CPPFLAGS   += -DNDEBUG
	MK_CFLAGS     += -O3 -g
	MK_CXXFLAGS   += -O3 -g
	MK_NVCCFLAGS  += -O3 -g
endif

ifdef LLAMA_SANITIZE_THREAD
	MK_CFLAGS   += -fsanitize=thread -g
	MK_CXXFLAGS += -fsanitize=thread -g
	MK_LDFLAGS  += -fsanitize=thread -g
endif

ifdef LLAMA_SANITIZE_ADDRESS
	MK_CFLAGS   += -fsanitize=address -fno-omit-frame-pointer -g
	MK_CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
	MK_LDFLAGS  += -fsanitize=address -fno-omit-frame-pointer -g
endif

ifdef LLAMA_SANITIZE_UNDEFINED
	MK_CFLAGS   += -fsanitize=undefined -g
	MK_CXXFLAGS += -fsanitize=undefined -g
	MK_LDFLAGS  += -fsanitize=undefined -g
endif

ifdef LLAMA_SERVER_SSL
	MK_CPPFLAGS += -DCPPHTTPLIB_OPENSSL_SUPPORT
	MK_LDFLAGS += -lssl -lcrypto
endif

ifndef GGML_NO_CPU_AARCH64
	MK_CPPFLAGS += -DGGML_USE_CPU_AARCH64
endif

# warnings
WARN_FLAGS = \
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wcast-qual \
	-Wno-unused-function

MK_CFLAGS += \
	$(WARN_FLAGS) \
	-Wshadow \
	-Wstrict-prototypes \
	-Wpointer-arith \
	-Wmissing-prototypes \
	-Werror=implicit-int \
	-Werror=implicit-function-declaration

MK_CXXFLAGS += \
	$(WARN_FLAGS) \
	-Wmissing-declarations \
	-Wmissing-noreturn

ifeq ($(LLAMA_FATAL_WARNINGS),1)
	MK_CFLAGS   += -Werror
	MK_CXXFLAGS += -Werror
endif

# this version of Apple ld64 is buggy
ifneq '' '$(findstring dyld-1015.7,$(shell $(CC) $(LDFLAGS) -Wl,-v 2>&1))'
	MK_CPPFLAGS += -DHAVE_BUGGY_APPLE_LINKER
endif

# OS specific
# TODO: support Windows
ifneq '' '$(filter $(UNAME_S),Linux Darwin FreeBSD NetBSD OpenBSD Haiku)'
	MK_CFLAGS   += -pthread
	MK_CXXFLAGS += -pthread
endif

# detect Windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
endif

# library name prefix
ifneq ($(_WIN32),1)
	LIB_PRE := lib
endif

# Dynamic Shared Object extension
ifneq ($(_WIN32),1)
	DSO_EXT := .so
else
	DSO_EXT := .dll
endif

# Windows Sockets 2 (Winsock) for network-capable apps
ifeq ($(_WIN32),1)
	LWINSOCK2 := -lws2_32
endif

ifdef LLAMA_GPROF
	MK_CFLAGS   += -pg
	MK_CXXFLAGS += -pg
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue

ifndef RISCV_CROSS_COMPILE

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	MK_CFLAGS     += -march=native -mtune=native
	HOST_CXXFLAGS += -march=native -mtune=native

	# Usage AMX build test
	#MK_CFLAGS     += -march=graniterapids -mtune=graniterapids
	#HOST_CXXFLAGS += -march=graniterapids -mtune=graniterapids

	# Usage AVX-only
	#MK_CFLAGS   += -mfma -mf16c -mavx
	#MK_CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#MK_CFLAGS   += -mssse3
	#MK_CXXFLAGS += -mssse3
endif

ifneq '' '$(findstring mingw,$(shell $(CC) -dumpmachine))'
	# The stack is only 16-byte aligned on Windows, so don't let gcc emit aligned moves.
	# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412
	# https://github.com/ggml-org/llama.cpp/issues/2922
	MK_CFLAGS   += -Xassembler -muse-unaligned-vector-move
	MK_CXXFLAGS += -Xassembler -muse-unaligned-vector-move

	# Target Windows 8 for PrefetchVirtualMemory
	MK_CPPFLAGS += -D_WIN32_WINNT=0x602
endif

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	# Nvidia Jetson
	MK_CFLAGS   += -mcpu=native
	MK_CXXFLAGS += -mcpu=native
	JETSON_RELEASE_INFO = $(shell jetson_release)
	ifdef JETSON_RELEASE_INFO
		ifneq ($(filter TX2%,$(JETSON_RELEASE_INFO)),)
			JETSON_EOL_MODULE_DETECT = 1
			CC = aarch64-unknown-linux-gnu-gcc
			cxx = aarch64-unknown-linux-gnu-g++
		endif
	endif
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	MK_CFLAGS   += -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		MK_CFLAGS   += -mcpu=power9
		MK_CXXFLAGS += -mcpu=power9
	endif
endif

ifneq ($(filter ppc64le%,$(UNAME_M)),)
	MK_CFLAGS   += -mcpu=powerpc64le
	MK_CXXFLAGS += -mcpu=powerpc64le
	CUDA_POWER_ARCH = 1
endif

ifneq ($(filter loongarch64%,$(UNAME_M)),)
	MK_CFLAGS   += -mlasx
	MK_CXXFLAGS += -mlasx
endif

ifneq ($(filter riscv64%,$(UNAME_M)),)
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

else # RISC-V CROSS COMPILATION
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

ifndef GGML_NO_ACCELERATE
	# Mac OS - include Accelerate framework.
	# `-framework Accelerate` works both with Apple Silicon and Mac Intel
	ifeq ($(UNAME_S),Darwin)
		MK_CPPFLAGS  += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DGGML_BLAS_USE_ACCELERATE
		MK_CPPFLAGS  += -DACCELERATE_NEW_LAPACK
		MK_CPPFLAGS  += -DACCELERATE_LAPACK_ILP64
		MK_LDFLAGS   += -framework Accelerate
		OBJ_GGML_EXT += ggml/src/ggml-blas/ggml-blas.o
	endif
endif # GGML_NO_ACCELERATE

ifndef GGML_NO_OPENMP
	MK_CPPFLAGS += -DGGML_USE_OPENMP
	MK_CFLAGS   += -fopenmp
	MK_CXXFLAGS += -fopenmp
endif # GGML_NO_OPENMP

ifdef GGML_OPENBLAS
	MK_CPPFLAGS  += -DGGML_USE_BLAS $(shell pkg-config --cflags-only-I openblas)
	MK_CFLAGS    += $(shell pkg-config --cflags-only-other openblas)
	MK_LDFLAGS   += $(shell pkg-config --libs openblas)
	OBJ_GGML_EXT += ggml/src/ggml-blas/ggml-blas.o
endif # GGML_OPENBLAS

ifdef GGML_OPENBLAS64
	MK_CPPFLAGS  += -DGGML_USE_BLAS $(shell pkg-config --cflags-only-I openblas64)
	MK_CFLAGS    += $(shell pkg-config --cflags-only-other openblas64)
	MK_LDFLAGS   += $(shell pkg-config --libs openblas64)
	OBJ_GGML_EXT += ggml/src/ggml-blas/ggml-blas.o
endif # GGML_OPENBLAS64

ifdef GGML_BLIS
	MK_CPPFLAGS  += -DGGML_USE_BLAS -DGGML_BLAS_USE_BLIS -I/usr/local/include/blis -I/usr/include/blis
	MK_LDFLAGS   += -lblis -L/usr/local/lib
	OBJ_GGML_EXT += ggml/src/ggml-blas/ggml-blas.o
endif # GGML_BLIS

ifdef GGML_NVPL
	MK_CPPFLAGS  += -DGGML_USE_BLAS -DGGML_BLAS_USE_NVPL -DNVPL_ILP64 -I/usr/local/include/nvpl_blas -I/usr/include/nvpl_blas
	MK_LDFLAGS   += -L/usr/local/lib -lnvpl_blas_core -lnvpl_blas_ilp64_gomp
	OBJ_GGML_EXT += ggml/src/ggml-blas/ggml-blas.o
endif # GGML_NVPL

ifndef GGML_NO_LLAMAFILE
	MK_CPPFLAGS  += -DGGML_USE_LLAMAFILE
	OBJ_GGML_EXT += ggml/src/ggml-cpu/llamafile/sgemm.o
endif

ifndef GGML_NO_AMX
	MK_CPPFLAGS += -DGGML_USE_AMX
	OBJ_GGML_EXT += ggml/src/ggml-cpu/amx/amx.o ggml/src/ggml-cpu/amx/mmq.o
endif

# only necessary for the CPU backend files
MK_CPPFLAGS += -Iggml/src/ggml-cpu

ifdef GGML_RPC
	MK_CPPFLAGS  += -DGGML_USE_RPC
	OBJ_GGML_EXT += ggml/src/ggml-rpc.o
endif # GGML_RPC

OBJ_CUDA_TMPL      = $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-mma*.cu))
OBJ_CUDA_TMPL     += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/mmq*.cu))

ifdef GGML_CUDA_FA_ALL_QUANTS
	OBJ_CUDA_TMPL += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*.cu))
else
	OBJ_CUDA_TMPL += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu))
	OBJ_CUDA_TMPL += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu))
	OBJ_CUDA_TMPL += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu))
endif # GGML_CUDA_FA_ALL_QUANTS

ifdef GGML_CUDA
	ifneq ('', '$(wildcard /opt/cuda)')
		CUDA_PATH ?= /opt/cuda
	else
		CUDA_PATH ?= /usr/local/cuda
	endif

	MK_CPPFLAGS  += -DGGML_USE_CUDA -DGGML_CUDA_USE_GRAPHS -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/$(UNAME_M)-linux/include
	MK_LDFLAGS   += -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L$(CUDA_PATH)/lib64 -L/usr/lib64 -L$(CUDA_PATH)/targets/$(UNAME_M)-linux/lib -L$(CUDA_PATH)/lib64/stubs -L/usr/lib/wsl/lib
	MK_NVCCFLAGS += -use_fast_math

	OBJ_GGML_EXT += ggml/src/ggml-cuda/ggml-cuda.o
	OBJ_GGML_EXT += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	OBJ_GGML_EXT += $(OBJ_CUDA_TMPL)

ifdef LLAMA_FATAL_WARNINGS
	MK_NVCCFLAGS += -Werror all-warnings
endif # LLAMA_FATAL_WARNINGS

ifndef JETSON_EOL_MODULE_DETECT
	MK_NVCCFLAGS += --forward-unknown-to-host-compiler
endif # JETSON_EOL_MODULE_DETECT

ifdef LLAMA_DEBUG
	MK_NVCCFLAGS += -lineinfo
endif # LLAMA_DEBUG

ifdef GGML_CUDA_DEBUG
	MK_NVCCFLAGS += --device-debug
endif # GGML_CUDA_DEBUG

ifdef GGML_CUDA_NVCC
	NVCC = $(CCACHE) $(GGML_CUDA_NVCC)
else
	NVCC = $(CCACHE) nvcc
endif # GGML_CUDA_NVCC

ifdef CUDA_DOCKER_ARCH
	MK_NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else ifndef CUDA_POWER_ARCH
	MK_NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH

ifdef GGML_CUDA_FORCE_MMQ
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_MMQ
endif # GGML_CUDA_FORCE_MMQ

ifdef GGML_CUDA_FORCE_CUBLAS
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_CUBLAS
endif # GGML_CUDA_FORCE_CUBLAS

ifdef GGML_CUDA_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # GGML_CUDA_F16

ifdef GGML_CUDA_DMMV_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # GGML_CUDA_DMMV_F16

ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=$(GGML_CUDA_PEER_MAX_BATCH_SIZE)
else
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
endif # GGML_CUDA_PEER_MAX_BATCH_SIZE

ifdef GGML_CUDA_NO_PEER_COPY
	MK_NVCCFLAGS += -DGGML_CUDA_NO_PEER_COPY
endif # GGML_CUDA_NO_PEER_COPY

ifdef GGML_CUDA_CCBIN
	MK_NVCCFLAGS += -ccbin $(GGML_CUDA_CCBIN)
endif # GGML_CUDA_CCBIN

ifdef GGML_CUDA_NO_FA
	MK_NVCCFLAGS += -DGGML_CUDA_NO_FA
endif # GGML_CUDA_NO_FA

ifdef GGML_CUDA_FA_ALL_QUANTS
	MK_NVCCFLAGS += -DGGML_CUDA_FA_ALL_QUANTS
endif # GGML_CUDA_FA_ALL_QUANTS

ifdef JETSON_EOL_MODULE_DETECT
define NVCC_COMPILE
	$(NVCC) -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_CUDA -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/aarch64-linux/include -std=c++11 -O3 $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
else
define NVCC_COMPILE
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
endif # JETSON_EOL_MODULE_DETECT

ggml/src/ggml-cuda/%.o: \
	ggml/src/ggml-cuda/%.cu \
	ggml/include/ggml.h \
	ggml/src/ggml-common.h \
	ggml/src/ggml-cuda/common.cuh
	$(NVCC_COMPILE)

ggml/src/ggml-cuda/ggml-cuda.o: \
	ggml/src/ggml-cuda/ggml-cuda.cu \
	ggml/include/ggml-cuda.h \
	ggml/include/ggml.h \
	ggml/include/ggml-backend.h \
	ggml/src/ggml-backend-impl.h \
	ggml/src/ggml-common.h \
	$(wildcard ggml/src/ggml-cuda/*.cuh)
	$(NVCC_COMPILE)
endif # GGML_CUDA

ifdef GGML_VULKAN
	MK_CPPFLAGS  += -DGGML_USE_VULKAN
	MK_LDFLAGS   += $(shell pkg-config --libs vulkan)
	OBJ_GGML_EXT += ggml/src/ggml-vulkan.o ggml/src/ggml-vulkan-shaders.o

ifdef GGML_VULKAN_CHECK_RESULTS
	MK_CPPFLAGS  += -DGGML_VULKAN_CHECK_RESULTS
endif

ifdef GGML_VULKAN_DEBUG
	MK_CPPFLAGS  += -DGGML_VULKAN_DEBUG
endif

ifdef GGML_VULKAN_MEMORY_DEBUG
	MK_CPPFLAGS  += -DGGML_VULKAN_MEMORY_DEBUG
endif

ifdef GGML_VULKAN_PERF
	MK_CPPFLAGS  += -DGGML_VULKAN_PERF
endif

ifdef GGML_VULKAN_VALIDATE
	MK_CPPFLAGS  += -DGGML_VULKAN_VALIDATE
endif

ifdef GGML_VULKAN_RUN_TESTS
	MK_CPPFLAGS  += -DGGML_VULKAN_RUN_TESTS
endif

GLSLC_CMD  = glslc
_ggml_vk_genshaders_cmd = $(shell pwd)/vulkan-shaders-gen
_ggml_vk_header = ggml/src/ggml-vulkan-shaders.hpp
_ggml_vk_source = ggml/src/ggml-vulkan-shaders.cpp
_ggml_vk_input_dir = ggml/src/ggml-vulkan/vulkan-shaders
_ggml_vk_shader_deps = $(echo $(_ggml_vk_input_dir)/*.comp)

ggml/src/ggml-vulkan.o: ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/include/ggml-vulkan.h $(_ggml_vk_header) $(_ggml_vk_source)
	$(CXX) $(CXXFLAGS) $(shell pkg-config --cflags vulkan) -c $< -o $@

$(_ggml_vk_header): $(_ggml_vk_source)

$(_ggml_vk_source): $(_ggml_vk_shader_deps) vulkan-shaders-gen
	$(_ggml_vk_genshaders_cmd) \
		--glslc      $(GLSLC_CMD) \
		--input-dir  $(_ggml_vk_input_dir) \
		--target-hpp $(_ggml_vk_header) \
		--target-cpp $(_ggml_vk_source)

vulkan-shaders-gen: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(LDFLAGS) ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp

endif # GGML_VULKAN

ifdef GGML_HIP
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH      ?= /usr
		AMDGPU_TARGETS ?= $(shell $(shell which amdgpu-arch))
	else
		ROCM_PATH	?= /opt/rocm
		AMDGPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
	endif

	MK_CPPFLAGS += -DGGML_USE_HIP -DGGML_USE_CUDA

	MK_LDFLAGS += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	MK_LDFLAGS += -L$(ROCM_PATH)/lib64 -Wl,-rpath=$(ROCM_PATH)/lib64
	MK_LDFLAGS += -lhipblas -lamdhip64 -lrocblas

	HIPCC ?= $(CCACHE) $(ROCM_PATH)/bin/hipcc

	HIPFLAGS += $(addprefix --offload-arch=,$(AMDGPU_TARGETS))

ifdef GGML_CUDA_FORCE_MMQ
	HIPFLAGS += -DGGML_CUDA_FORCE_MMQ
endif # GGML_CUDA_FORCE_MMQ

ifdef GGML_CUDA_FORCE_CUBLAS
	HIPFLAGS += -DGGML_CUDA_FORCE_CUBLAS
endif # GGML_CUDA_FORCE_CUBLAS

ifdef GGML_CUDA_NO_PEER_COPY
	HIPFLAGS += -DGGML_CUDA_NO_PEER_COPY
endif # GGML_CUDA_NO_PEER_COPY

ifdef GGML_CUDA_NO_FA
	HIPFLAGS += -DGGML_CUDA_NO_FA
endif # GGML_CUDA_NO_FA

	OBJ_GGML_EXT += ggml/src/ggml-cuda/ggml-cuda.o
	OBJ_GGML_EXT += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	OBJ_GGML_EXT += $(OBJ_CUDA_TMPL)

ggml/src/ggml-cuda/ggml-cuda.o: \
	ggml/src/ggml-cuda/ggml-cuda.cu \
	ggml/include/ggml-cuda.h \
	ggml/include/ggml.h \
	ggml/include/ggml-backend.h \
	ggml/src/ggml-backend-impl.h \
	ggml/src/ggml-common.h \
	$(wildcard ggml/src/ggml-cuda/*.cuh)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<

ggml/src/ggml-cuda/%.o: \
	ggml/src/ggml-cuda/%.cu \
	ggml/include/ggml.h \
	ggml/src/ggml-common.h \
	ggml/src/ggml-cuda/common.cuh
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<
endif # GGML_HIP

ifdef GGML_MUSA
	ifeq ($(wildcard /opt/musa),)
		MUSA_PATH ?= /usr/local/musa
	else
		MUSA_PATH ?= /opt/musa
	endif
	MUSA_ARCHITECTURES ?= 21;22;31

	MK_CPPFLAGS += -DGGML_USE_MUSA -DGGML_USE_CUDA
	MK_LDFLAGS += -L$(MUSA_PATH)/lib -Wl,-rpath=$(MUSA_PATH)/lib
	MK_LDFLAGS += -lmusa -lmusart -lmublas

	ifndef GGML_NO_OPENMP
		# For Ubuntu Focal
		MK_CPPFLAGS += -I/usr/lib/llvm-10/include/openmp
		MK_LDFLAGS  += -L/usr/lib/llvm-10/lib
		# For Ubuntu Jammy
		MK_CPPFLAGS += -I/usr/lib/llvm-14/lib/clang/14.0.0/include
		MK_LDFLAGS  += -L/usr/lib/llvm-14/lib
	endif # GGML_NO_OPENMP

	CC  := $(MUSA_PATH)/bin/clang
	CXX := $(MUSA_PATH)/bin/clang++
	MCC := $(CCACHE) $(MUSA_PATH)/bin/mcc

	MUSAFLAGS  = -fsigned-char -x musa -mtgpu
	MUSAFLAGS += $(foreach arch,$(subst ;, ,$(MUSA_ARCHITECTURES)),--cuda-gpu-arch=mp_$(arch))

ifdef GGML_CUDA_FORCE_MMQ
	MUSAFLAGS += -DGGML_CUDA_FORCE_MMQ
endif # GGML_CUDA_FORCE_MMQ

ifdef GGML_CUDA_FORCE_CUBLAS
	MUSAFLAGS += -DGGML_CUDA_FORCE_CUBLAS
endif # GGML_CUDA_FORCE_CUBLAS

ifdef GGML_CUDA_F16
	MUSAFLAGS += -DGGML_CUDA_F16
endif # GGML_CUDA_F16

ifdef GGML_CUDA_DMMV_F16
	MUSAFLAGS += -DGGML_CUDA_F16
endif # GGML_CUDA_DMMV_F16

ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
	MUSAFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=$(GGML_CUDA_PEER_MAX_BATCH_SIZE)
else
	MUSAFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
endif # GGML_CUDA_PEER_MAX_BATCH_SIZE

ifdef GGML_CUDA_NO_PEER_COPY
	MUSAFLAGS += -DGGML_CUDA_NO_PEER_COPY
endif # GGML_CUDA_NO_PEER_COPY

ifdef GGML_CUDA_NO_FA
	MUSAFLAGS += -DGGML_CUDA_NO_FA
endif # GGML_CUDA_NO_FA

ifdef GGML_CUDA_FA_ALL_QUANTS
	MUSAFLAGS += -DGGML_CUDA_FA_ALL_QUANTS
endif # GGML_CUDA_FA_ALL_QUANTS

	OBJ_GGML_EXT += ggml/src/ggml-cuda/ggml-cuda.o
	OBJ_GGML_EXT += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	OBJ_GGML_EXT += $(OBJ_CUDA_TMPL)

ggml/src/ggml-cuda/ggml-cuda.o: \
	ggml/src/ggml-cuda/ggml-cuda.cu \
	ggml/include/ggml-cuda.h \
	ggml/include/ggml.h \
	ggml/include/ggml-backend.h \
	ggml/src/ggml-backend-impl.h \
	ggml/src/ggml-common.h \
	$(wildcard ggml/src/ggml-cuda/*.cuh)
	$(MCC) $(CXXFLAGS) $(MUSAFLAGS) -c -o $@ $<

ggml/src/ggml-cuda/%.o: \
	ggml/src/ggml-cuda/%.cu \
	ggml/include/ggml.h \
	ggml/src/ggml-common.h \
	ggml/src/ggml-cuda/common.cuh
	$(MCC) $(CXXFLAGS) $(MUSAFLAGS) -c -o $@ $<
endif # GGML_MUSA

ifdef GGML_METAL
	MK_CPPFLAGS  += -DGGML_USE_METAL
	MK_LDFLAGS   += -framework Foundation -framework Metal -framework MetalKit
	OBJ_GGML_EXT += ggml/src/ggml-metal/ggml-metal.o

ifdef GGML_METAL_USE_BF16
	MK_CPPFLAGS += -DGGML_METAL_USE_BF16
endif # GGML_METAL_USE_BF16
ifdef GGML_METAL_NDEBUG
	MK_CPPFLAGS += -DGGML_METAL_NDEBUG
endif
ifdef GGML_METAL_EMBED_LIBRARY
	MK_CPPFLAGS  += -DGGML_METAL_EMBED_LIBRARY
	OBJ_GGML_EXT += ggml/src/ggml-metal-embed.o
endif
endif # GGML_METAL

ifdef GGML_METAL
ggml/src/ggml-metal/ggml-metal.o: \
	ggml/src/ggml-metal/ggml-metal.m \
	ggml/src/ggml-metal/ggml-metal-impl.h \
	ggml/include/ggml-metal.h \
	ggml/include/ggml.h
	$(CC) $(CFLAGS) -c $< -o $@

ifdef GGML_METAL_EMBED_LIBRARY
ggml/src/ggml-metal-embed.o: \
	ggml/src/ggml-metal/ggml-metal.metal \
	ggml/src/ggml-metal/ggml-metal-impl.h \
	ggml/src/ggml-common.h
	@echo "Embedding Metal library"
	@sed -e '/__embed_ggml-common.h__/r      ggml/src/ggml-common.h'                -e '/__embed_ggml-common.h__/d'      < ggml/src/ggml-metal/ggml-metal.metal           > ggml/src/ggml-metal/ggml-metal-embed.metal.tmp
	@sed -e '/#include "ggml-metal-impl.h"/r ggml/src/ggml-metal/ggml-metal-impl.h' -e '/#include "ggml-metal-impl.h"/d' < ggml/src/ggml-metal/ggml-metal-embed.metal.tmp > ggml/src/ggml-metal/ggml-metal-embed.metal
	$(eval TEMP_ASSEMBLY=$(shell mktemp -d))
	@echo ".section __DATA, __ggml_metallib"                       >  $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	@echo ".globl _ggml_metallib_start"                            >> $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	@echo "_ggml_metallib_start:"                                  >> $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	@echo ".incbin \"ggml/src/ggml-metal/ggml-metal-embed.metal\"" >> $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	@echo ".globl _ggml_metallib_end"                              >> $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	@echo "_ggml_metallib_end:"                                    >> $(TEMP_ASSEMBLY)/ggml-metal-embed.s
	$(CC) $(CFLAGS) -c $(TEMP_ASSEMBLY)/ggml-metal-embed.s -o $@
	@rm -f ${TEMP_ASSEMBLY}/ggml-metal-embed.s
	@rmdir ${TEMP_ASSEMBLY}
endif
endif # GGML_METAL

DIR_GGML = ggml
DIR_LLAMA = src
DIR_COMMON = common

OBJ_GGML = \
	$(DIR_GGML)/src/ggml.o \
	$(DIR_GGML)/src/ggml-alloc.o \
	$(DIR_GGML)/src/ggml-backend.o \
	$(DIR_GGML)/src/ggml-backend-reg.o \
	$(DIR_GGML)/src/ggml-opt.o \
	$(DIR_GGML)/src/ggml-quants.o \
	$(DIR_GGML)/src/ggml-threading.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu_cpp.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu-aarch64.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu-hbm.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu-quants.o \
	$(DIR_GGML)/src/ggml-cpu/ggml-cpu-traits.o \
	$(OBJ_GGML_EXT)

OBJ_LLAMA = \
	$(DIR_LLAMA)/llama.o \
	$(DIR_LLAMA)/llama-vocab.o \
	$(DIR_LLAMA)/llama-grammar.o \
	$(DIR_LLAMA)/llama-sampling.o \
	$(DIR_LLAMA)/unicode.o \
	$(DIR_LLAMA)/unicode-data.o

OBJ_COMMON = \
	$(DIR_COMMON)/common.o \
	$(DIR_COMMON)/arg.o \
	$(DIR_COMMON)/log.o \
	$(DIR_COMMON)/console.o \
	$(DIR_COMMON)/ngram-cache.o \
	$(DIR_COMMON)/sampling.o \
	$(DIR_COMMON)/speculative.o \
	$(DIR_COMMON)/chat.o \
	$(DIR_COMMON)/build-info.o \
	$(DIR_COMMON)/json-schema-to-grammar.o

OBJ_ALL = $(OBJ_GGML) $(OBJ_LLAMA) $(OBJ_COMMON)

LIB_GGML   = $(LIB_PRE)ggml$(DSO_EXT)
LIB_GGML_S = $(LIB_PRE)ggml.a

LIB_LLAMA   = $(LIB_PRE)llama$(DSO_EXT)
LIB_LLAMA_S = $(LIB_PRE)llama.a

LIB_COMMON   = $(LIB_PRE)common$(DSO_EXT)
LIB_COMMON_S = $(LIB_PRE)common.a

LIB_ALL   = $(LIB_GGML)   $(LIB_LLAMA)   $(LIB_COMMON)
LIB_ALL_S = $(LIB_GGML_S) $(LIB_LLAMA_S) $(LIB_COMMON_S)

GF_CC := $(CC)
include scripts/get-flags.mk

# combine build flags with cmdline overrides
override CPPFLAGS  := $(MK_CPPFLAGS) $(CPPFLAGS)
override CFLAGS    := $(CPPFLAGS) $(MK_CFLAGS) $(GF_CFLAGS) $(CFLAGS)
BASE_CXXFLAGS      := $(MK_CXXFLAGS) $(CXXFLAGS)
override CXXFLAGS  := $(BASE_CXXFLAGS) $(HOST_CXXFLAGS) $(GF_CXXFLAGS) $(CPPFLAGS)
override NVCCFLAGS := $(MK_NVCCFLAGS) $(NVCCFLAGS)
override LDFLAGS   := $(MK_LDFLAGS) $(LDFLAGS)

# identify CUDA host compiler
ifdef GGML_CUDA
GF_CC := $(NVCC) $(NVCCFLAGS) 2>/dev/null .c -Xcompiler
include scripts/get-flags.mk
CUDA_CXXFLAGS := $(BASE_CXXFLAGS) $(GF_CXXFLAGS) -Wno-pedantic
endif

ifdef LLAMA_CURL
override CXXFLAGS := $(CXXFLAGS) -DLLAMA_USE_CURL
override LDFLAGS  := $(LDFLAGS) -lcurl
endif

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:   $(UNAME_S))
$(info I UNAME_P:   $(UNAME_P))
$(info I UNAME_M:   $(UNAME_M))
$(info I CFLAGS:    $(CFLAGS))
$(info I CXXFLAGS:  $(CXXFLAGS))
$(info I NVCCFLAGS: $(NVCCFLAGS))
$(info I LDFLAGS:   $(LDFLAGS))
$(info I CC:        $(shell $(CC)   --version | head -n 1))
$(info I CXX:       $(shell $(CXX)  --version | head -n 1))
ifdef GGML_CUDA
$(info I NVCC:      $(shell $(NVCC) --version | tail -n 1))
CUDA_VERSION := $(shell $(NVCC) --version | grep -oP 'release (\K[0-9]+\.[0-9])')
ifeq ($(shell awk -v "v=$(CUDA_VERSION)" 'BEGIN { print (v < 11.7) }'),1)

ifndef CUDA_DOCKER_ARCH
ifndef CUDA_POWER_ARCH
$(error I ERROR: For CUDA versions < 11.7 a target CUDA architecture must be explicitly provided via environment variable CUDA_DOCKER_ARCH, e.g. by running "export CUDA_DOCKER_ARCH=compute_XX" on Unix-like systems, where XX is the minimum compute capability that the code needs to run on. A list with compute capabilities can be found here: https://developer.nvidia.com/cuda-gpus )
endif # CUDA_POWER_ARCH
endif # CUDA_DOCKER_ARCH

endif # eq ($(shell echo "$(CUDA_VERSION) < 11.7" | bc),1)
endif # GGML_CUDA
$(info )

ifdef DEPRECATE_WARNING
$(info !!! DEPRECATION WARNING !!!)
$(info The following LLAMA_ options are deprecated and will be removed in the future. Use the GGML_ prefix instead)
$(info   - LLAMA_CUDA)
$(info   - LLAMA_METAL)
$(info   - LLAMA_METAL_EMBED_LIBRARY)
$(info   - LLAMA_OPENMP)
$(info   - LLAMA_RPC)
$(info   - LLAMA_SYCL)
$(info   - LLAMA_SYCL_F16)
$(info   - LLAMA_OPENBLAS)
$(info   - LLAMA_OPENBLAS64)
$(info   - LLAMA_BLIS)
$(info   - LLAMA_NO_LLAMAFILE)
$(info   - LLAMA_NO_ACCELERATE)
$(info   - LLAMA_NO_OPENMP)
$(info   - LLAMA_NO_METAL)
$(info   - LLAMA_NO_CCACHE)
$(info )
endif

ifdef REMOVE_WARNING
$(info !!! REMOVAL WARNING !!!)
$(info The following LLAMA_ options have been removed and are no longer supported)
$(info   - LLAMA_DISABLE_LOGS   (https://github.com/ggml-org/llama.cpp/pull/9418))
$(info   - LLAMA_SERVER_VERBOSE (https://github.com/ggml-org/llama.cpp/pull/9418))
$(info )
endif

#
# Build libraries
#

# Libraries
LIB_GGML   = libggml.so
LIB_GGML_S = libggml.a

LIB_LLAMA   = libllama.so
LIB_LLAMA_S = libllama.a

LIB_COMMON   = libcommon.so
LIB_COMMON_S = libcommon.a

# Targets
BUILD_TARGETS += $(LIB_GGML) $(LIB_GGML_S) $(LIB_LLAMA) $(LIB_LLAMA_S) $(LIB_COMMON) $(LIB_COMMON_S)

# Dependency files
DEP_FILES = $(OBJ_GGML:.o=.d) $(OBJ_LLAMA:.o=.d) $(OBJ_COMMON:.o=.d)

# Default target
all: $(BUILD_TARGETS)

# force c++ build for source file that have same name as c file
# Note: need this exception because `ggml-cpu.c` and `ggml-cpu.cpp` both produce the same obj/dep files
$(DIR_GGML)/%_cpp.o: $(DIR_GGML)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Rules for building object files
$(DIR_GGML)/%.o: $(DIR_GGML)/%.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(DIR_GGML)/%.o: $(DIR_GGML)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(DIR_LLAMA)/%.o: $(DIR_LLAMA)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(DIR_COMMON)/%.o: $(DIR_COMMON)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Rules for building libraries
$(LIB_GGML): $(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

$(LIB_GGML_S): $(OBJ_GGML)
	ar rcs $(LIB_GGML_S) $^

$(LIB_LLAMA): $(OBJ_LLAMA) $(LIB_GGML)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

$(LIB_LLAMA_S): $(OBJ_LLAMA)
	ar rcs $(LIB_LLAMA_S) $^

$(LIB_COMMON): $(OBJ_COMMON) $(LIB_LLAMA) $(LIB_GGML)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

$(LIB_COMMON_S): $(OBJ_COMMON)
	ar rcs $(LIB_COMMON_S) $^

# Include dependency files
-include $(DEP_FILES)

# Clean generated server assets
clean-server-assets:
	find tools/server -type f -name "*.js.hpp"   -delete
	find tools/server -type f -name "*.mjs.hpp"  -delete
	find tools/server -type f -name "*.css.hpp"  -delete
	find tools/server -type f -name "*.html.hpp" -delete

# Clean rule
clean: clean-server-assets
	rm -vrf $(BUILD_TARGETS) $(TEST_TARGETS)
	rm -rvf *.a *.dll *.so *.dot
	find ggml src common tests examples pocs -type f -name "*.o" -delete
	find ggml src common tests examples pocs -type f -name "*.d" -delete

#
# Examples
#

# $< is the first prerequisite, i.e. the source file.
# Explicitly compile this to an object file so that it can be cached with ccache.
# The source file is then filtered out from $^ (the list of all prerequisites) and the object file is added instead.

# Helper function that replaces .c, .cpp, and .cu file endings with .o:
GET_OBJ_FILE = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(1))))

llama-cli: tools/main/main.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./llama-cli -h for help.  ===='
	@echo

llama-run: tools/run/run.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-simple: examples/simple/simple.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-simple-chat: examples/simple-chat/simple-chat.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-tokenize: tools/tokenize/tokenize.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-batched: examples/batched/batched.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-batched-bench: tools/batched-bench/batched-bench.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-quantize: tools/quantize/quantize.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-quantize-stats: tools/quantize-stats/quantize-stats.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-perplexity: tools/perplexity/perplexity.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-imatrix: tools/imatrix/imatrix.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-embedding: examples/embedding/embedding.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gritlm: examples/gritlm/gritlm.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-save-load-state: examples/save-load-state/save-load-state.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gguf: examples/gguf/gguf.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

examples/gguf-hash/deps/sha1/sha1.o: \
	examples/gguf-hash/deps/sha1/sha1.c
	$(CC) $(CFLAGS) -Iexamples/gguf-hash/deps -c $< -o $@

examples/gguf-hash/deps/xxhash/xxhash.o: \
	examples/gguf-hash/deps/xxhash/xxhash.c
	$(CC) $(CFLAGS) -Iexamples/gguf-hash/deps -c $< -o $@

examples/gguf-hash/deps/sha256/sha256.o: \
	examples/gguf-hash/deps/sha256/sha256.c
	$(CC) $(CFLAGS) -Iexamples/gguf-hash/deps -c $< -o $@

llama-gguf-hash: examples/gguf-hash/gguf-hash.cpp examples/gguf-hash/deps/sha1/sha1.o examples/gguf-hash/deps/xxhash/xxhash.o examples/gguf-hash/deps/sha256/sha256.o\
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -Iexamples/gguf-hash/deps -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gguf-split: tools/gguf-split/gguf-split.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-eval-callback: examples/eval-callback/eval-callback.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-cvector-generator: tools/cvector-generator/cvector-generator.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-convert-llama2c-to-ggml: examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-bench: tools/llama-bench/llama-bench.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-export-lora: tools/export-lora/export-lora.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-retrieval: examples/retrieval/retrieval.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-speculative: examples/speculative/speculative.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-parallel: examples/parallel/parallel.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookahead: examples/lookahead/lookahead.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup: examples/lookup/lookup.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-create: examples/lookup/lookup-create.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-merge: examples/lookup/lookup-merge.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-stats: examples/lookup/lookup-stats.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-passkey: examples/passkey/passkey.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gbnf-validator: examples/gbnf-validator/gbnf-validator.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

ifdef GGML_RPC
rpc-server: tools/rpc/rpc-server.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
endif # GGML_RPC

llama-server: \
	tools/server/server.cpp \
	tools/server/utils.hpp \
	tools/server/httplib.h \
	tools/server/index.html.hpp \
	tools/server/loading.html.hpp \
	common/chat.cpp \
	common/chat.h \
	common/chat-template.hpp \
	common/json.hpp \
	common/minja.hpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp $<,$^) -Itools/server $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS) $(LWINSOCK2)

# Portable equivalent of `cd tools/server/public && xxd -i $(notdir $<) ../$(notdir $<).hpp`:
tools/server/%.hpp: tools/server/public/% FORCE Makefile
	@( export NAME=$(subst .,_,$(subst -,_,$(notdir $<))) && \
		echo "unsigned char $${NAME}[] = {" && \
		cat $< | od -v -t x1 -An | sed -E 's/([0-9a-fA-F]+)/0x\1, /g' && \
		echo "};" && \
		echo "unsigned int $${NAME}_len = $(shell cat $< | wc -c );" \
	) > $@

llama-gen-docs: examples/gen-docs/gen-docs.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

libllava.a: tools/mtmd/llava.cpp \
	tools/mtmd/llava.h \
	tools/mtmd/clip.cpp \
	tools/mtmd/clip.h \
	common/stb_image.h \
	common/base64.hpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -static -fPIC -c $< -o $@ -Wno-cast-qual

llama-llava-cli: tools/mtmd/llava-cli.cpp \
	tools/mtmd/llava.cpp \
	tools/mtmd/llava.h \
	tools/mtmd/clip.cpp \
	tools/mtmd/clip.h \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) $< $(filter-out %.h $<,$^) -o $@ $(LDFLAGS) -Wno-cast-qual

llama-minicpmv-cli: tools/mtmd/minicpmv-cli.cpp \
	tools/mtmd/llava.cpp \
	tools/mtmd/llava.h \
	tools/mtmd/clip.cpp \
	tools/mtmd/clip.h \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) $< $(filter-out %.h $<,$^) -o $@ $(LDFLAGS) -Wno-cast-qual

llama-qwen2vl-cli: tools/mtmd/qwen2vl-cli.cpp \
	tools/mtmd/llava.cpp \
	tools/mtmd/llava.h \
	tools/mtmd/clip.cpp \
	tools/mtmd/clip.h \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) $< $(filter-out %.h $<,$^) -o $@ $(LDFLAGS) -Wno-cast-qual

ifeq ($(UNAME_S),Darwin)
swift: examples/batched.swift
	(cd examples/batched.swift; make build)
endif

common/build-info.cpp: $(wildcard .git/index) scripts/build-info.sh
	@sh scripts/build-info.sh "$(CC)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
	else \
		rm $@.tmp; \
	fi

common/build-info.o: common/build-info.cpp
	$(CXX) $(CXXFLAGS) -c $(filter-out %.h,$^) -o $@

#
# Tests
#

tests: $(TEST_TARGETS)

tests/test-arg-parser: tests/test-arg-parser.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-llama-grammar: tests/test-llama-grammar.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-log: tests/test-log.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-grammar-parser: tests/test-grammar-parser.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-grammar-integration: tests/test-grammar-integration.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-double-float: tests/test-double-float.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-json-schema-to-grammar: tests/test-json-schema-to-grammar.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -Itools/server -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-chat: tests/test-chat.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -Itools/server -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-opt: tests/test-opt.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-quantize-fns: tests/test-quantize-fns.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-quantize-perf: tests/test-quantize-perf.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-sampling: tests/test-sampling.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-0: tests/test-tokenizer-0.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-1-bpe: tests/test-tokenizer-1-bpe.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-1-spm: tests/test-tokenizer-1-spm.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-rope: tests/test-rope.cpp ggml/src/ggml.o \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-c.o: tests/test-c.c include/llama.h
	$(CC) $(CFLAGS) -c $(filter-out %.h,$^) -o $@

tests/test-backend-ops: tests/test-backend-ops.cpp \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-model-load-cancel: tests/test-model-load-cancel.cpp tests/get-model.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-autorelease: tests/test-autorelease.cpp tests/get-model.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-chat-template: tests/test-chat-template.cpp \
	$(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

#
# PoCs
#

llama-vdot: pocs/vdot/vdot.cpp ggml/src/ggml.o \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-q8dot: pocs/vdot/q8dot.cpp ggml/src/ggml.o \
	$(OBJ_GGML)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

#
# Deprecated binaries that we want to keep around long enough for people to migrate to the new filenames, then these can be removed.
#
# Mark legacy binary targets as .PHONY so that they are always checked.
.PHONY: FORCE main quantize perplexity embedding server

# Define the object file target
examples/deprecation-warning/deprecation-warning.o: examples/deprecation-warning/deprecation-warning.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# NOTE: We currently will always build the deprecation-warning `main` and `server` binaries to help users migrate.
#  Eventually we will want to remove these target from building all the time.
main: examples/deprecation-warning/deprecation-warning.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "NOTICE: The 'main' binary is deprecated. Please use 'llama-cli' instead."

server: examples/deprecation-warning/deprecation-warning.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "NOTICE: The 'server' binary is deprecated. Please use 'llama-server' instead."

quantize: examples/deprecation-warning/deprecation-warning.o
ifneq (,$(wildcard quantize))
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "#########"
	@echo "WARNING: The 'quantize' binary is deprecated. Please use 'llama-quantize' instead."
	@echo "  Remove the 'quantize' binary to remove this warning."
	@echo "#########"
endif

perplexity: examples/deprecation-warning/deprecation-warning.o
ifneq (,$(wildcard perplexity))
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "#########"
	@echo "WARNING: The 'perplexity' binary is deprecated. Please use 'llama-perplexity' instead."
	@echo "  Remove the 'perplexity' binary to remove this warning."
	@echo "#########"
endif

embedding: examples/deprecation-warning/deprecation-warning.o
ifneq (,$(wildcard embedding))
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "#########"
	@echo "WARNING: The 'embedding' binary is deprecated. Please use 'llama-embedding' instead."
	@echo "  Remove the 'embedding' binary to remove this warning."
	@echo "#########"
endif
