# Define the default target now so that it is always the first target
BUILD_TARGETS = \
	libllava.a \
	llama-baby-llama \
	llama-batched \
	llama-batched-bench \
	llama-bench \
	llama-benchmark-matmult \
	llama-cli \
	llama-convert-llama2c-to-ggml \
	llama-embedding \
	llama-eval-callback \
	llama-export-lora \
	llama-finetune \
	llama-gbnf-validator \
	llama-gguf \
	llama-gguf-split \
	llama-gritlm \
	llama-imatrix \
	llama-infill \
	llama-llava-cli \
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
	llama-speculative \
	llama-tokenize \
	llama-train-text-from-scratch \
	llama-vdot \
	llama-cvector-generator \
	tests/test-c.o

# Binaries only useful for tests
TEST_TARGETS = \
	tests/test-autorelease \
	tests/test-backend-ops \
	tests/test-double-float \
	tests/test-grad0 \
	tests/test-grammar-integration \
	tests/test-grammar-parser \
	tests/test-json-schema-to-grammar \
	tests/test-llama-grammar \
	tests/test-model-load-cancel \
	tests/test-opt \
	tests/test-quantize-fns \
	tests/test-quantize-perf \
	tests/test-rope \
	tests/test-sampling \
	tests/test-tokenizer-0 \
	tests/test-tokenizer-1-bpe \
	tests/test-tokenizer-1-spm

# Code coverage output files
COV_TARGETS = *.gcno tests/*.gcno *.gcda tests/*.gcda *.gcov tests/*.gcov lcov-report gcovr-report

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
	ifndef LLAMA_NO_METAL
		LLAMA_METAL := 1
	endif

	LLAMA_NO_OPENMP := 1

	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

ifdef LLAMA_RPC
	BUILD_TARGETS += rpc-server
endif

default: $(BUILD_TARGETS)

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

all: $(BUILD_TARGETS) $(TEST_TARGETS)

coverage: ## Run code coverage
	gcov -pb tests/*.cpp

lcov-report: coverage ## Generate lcov report
	mkdir -p lcov-report
	lcov --capture --directory . --output-file lcov-report/coverage.info
	genhtml lcov-report/coverage.info --output-directory lcov-report

gcovr-report: coverage ## Generate gcovr report
	mkdir -p gcovr-report
	gcovr --root . --html --html-details --output gcovr-report/coverage.html

ifdef RISCV_CROSS_COMPILE
CC	:= riscv64-unknown-linux-gnu-gcc
CXX	:= riscv64-unknown-linux-gnu-g++
endif

#
# Compile flags
#

# keep standard at C11 and C++11
MK_CPPFLAGS  = -I. -Icommon
MK_CFLAGS    = -std=c11   -fPIC
MK_CXXFLAGS  = -std=c++11 -fPIC
MK_NVCCFLAGS = -std=c++11

# -Ofast tends to produce faster code, but may not be available for some compilers.
ifdef LLAMA_FAST
MK_CFLAGS     += -Ofast
HOST_CXXFLAGS += -Ofast
ifndef LLAMA_DEBUG
MK_NVCCFLAGS  += -O3
endif # LLAMA_DEBUG
else
MK_CFLAGS     += -O3
MK_CXXFLAGS   += -O3
ifndef LLAMA_DEBUG
MK_NVCCFLAGS  += -O3
endif # LLAMA_DEBUG
endif # LLAMA_FAST

ifndef LLAMA_NO_CCACHE
CCACHE := $(shell which ccache)
ifdef CCACHE
export CCACHE_SLOPPINESS = time_macros
$(info I ccache found, compilation results will be cached. Disable with LLAMA_NO_CCACHE.)
CC    := $(CCACHE) $(CC)
CXX   := $(CCACHE) $(CXX)
else
$(info I ccache not found. Consider installing it for faster compilation.)
endif # CCACHE
endif # LLAMA_NO_CCACHE

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

ifdef LLAMA_SCHED_MAX_COPIES
	MK_CPPFLAGS += -DGGML_SCHED_MAX_COPIES=$(LLAMA_SCHED_MAX_COPIES)
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
	MK_CPPFLAGS += -DNDEBUG
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

ifdef LLAMA_SERVER_VERBOSE
	MK_CPPFLAGS += -DSERVER_VERBOSE=$(LLAMA_SERVER_VERBOSE)
endif

ifdef LLAMA_SERVER_SSL
	MK_CPPFLAGS += -DCPPHTTPLIB_OPENSSL_SUPPORT
	MK_LDFLAGS += -lssl -lcrypto
endif

ifdef LLAMA_CODE_COVERAGE
	MK_CXXFLAGS += -fprofile-arcs -ftest-coverage -dumpbase ''
endif

ifdef LLAMA_DISABLE_LOGS
	MK_CPPFLAGS += -DLOG_DISABLE_LOGS
endif # LLAMA_DISABLE_LOGS

# warnings
WARN_FLAGS    = -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
MK_CFLAGS    += $(WARN_FLAGS) -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int \
				-Werror=implicit-function-declaration
MK_CXXFLAGS  += $(WARN_FLAGS) -Wmissing-declarations -Wmissing-noreturn

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

ifndef RISCV

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	MK_CFLAGS     += -march=native -mtune=native
	HOST_CXXFLAGS += -march=native -mtune=native

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
	# https://github.com/ggerganov/llama.cpp/issues/2922
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

else
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

ifndef LLAMA_NO_ACCELERATE
	# Mac OS - include Accelerate framework.
	# `-framework Accelerate` works both with Apple Silicon and Mac Intel
	ifeq ($(UNAME_S),Darwin)
		MK_CPPFLAGS += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS
		MK_CPPFLAGS += -DACCELERATE_NEW_LAPACK
		MK_CPPFLAGS += -DACCELERATE_LAPACK_ILP64
		MK_LDFLAGS  += -framework Accelerate
		OBJS        += ggml-blas.o
	endif
endif # LLAMA_NO_ACCELERATE

ifndef LLAMA_NO_OPENMP
	MK_CPPFLAGS += -DGGML_USE_OPENMP
	MK_CFLAGS   += -fopenmp
	MK_CXXFLAGS += -fopenmp
endif # LLAMA_NO_OPENMP

ifdef LLAMA_OPENBLAS
	MK_CPPFLAGS += -DGGML_USE_BLAS $(shell pkg-config --cflags-only-I openblas)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas)
	OBJS        += ggml-blas.o
endif # LLAMA_OPENBLAS

ifdef LLAMA_OPENBLAS64
	MK_CPPFLAGS += -DGGML_USE_BLAS $(shell pkg-config --cflags-only-I openblas64)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas64)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas64)
	OBJS        += ggml-blas.o
endif # LLAMA_OPENBLAS64

ifdef LLAMA_BLIS
	MK_CPPFLAGS += -DGGML_USE_BLAS -I/usr/local/include/blis -I/usr/include/blis
	MK_LDFLAGS  += -lblis -L/usr/local/lib
	OBJS        += ggml-blas.o
endif # LLAMA_BLIS

ifndef LLAMA_NO_LLAMAFILE
	MK_CPPFLAGS += -DGGML_USE_LLAMAFILE
	OBJS        += sgemm.o
endif

ifdef LLAMA_RPC
	MK_CPPFLAGS   += -DGGML_USE_RPC
	OBJS          += ggml-rpc.o
endif # LLAMA_RPC

ifdef LLAMA_CUBLAS
# LLAMA_CUBLAS is deprecated and will be removed in the future
	LLAMA_CUDA := 1
endif

OBJS_CUDA_TEMP_INST      = $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/fattn-wmma*.cu))
OBJS_CUDA_TEMP_INST     += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/mmq*.cu))
ifdef LLAMA_CUDA_FA_ALL_QUANTS
	OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/fattn-vec*.cu))
else
	OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu))
	OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu))
	OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/template-instances/fattn-vec*f16-f16.cu))
endif # LLAMA_CUDA_FA_ALL_QUANTS

ifdef LLAMA_CUDA
	ifneq ('', '$(wildcard /opt/cuda)')
		CUDA_PATH ?= /opt/cuda
	else
		CUDA_PATH ?= /usr/local/cuda
	endif
	MK_CPPFLAGS  += -DGGML_USE_CUDA -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/$(UNAME_M)-linux/include -DGGML_CUDA_USE_GRAPHS
	MK_LDFLAGS   += -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L$(CUDA_PATH)/lib64 -L/usr/lib64 -L$(CUDA_PATH)/targets/$(UNAME_M)-linux/lib -L$(CUDA_PATH)/lib64/stubs -L/usr/lib/wsl/lib
	OBJS         += ggml-cuda.o
	OBJS         += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/*.cu))
	OBJS         += $(OBJS_CUDA_TEMP_INST)
	MK_NVCCFLAGS += -use_fast_math
ifdef LLAMA_FATAL_WARNINGS
	MK_NVCCFLAGS += -Werror all-warnings
endif # LLAMA_FATAL_WARNINGS
ifndef JETSON_EOL_MODULE_DETECT
	MK_NVCCFLAGS += --forward-unknown-to-host-compiler
endif # JETSON_EOL_MODULE_DETECT
ifdef LLAMA_DEBUG
	MK_NVCCFLAGS += -lineinfo
endif # LLAMA_DEBUG
ifdef LLAMA_CUDA_DEBUG
	MK_NVCCFLAGS += --device-debug
endif # LLAMA_CUDA_DEBUG
ifdef LLAMA_CUDA_NVCC
	NVCC = $(CCACHE) $(LLAMA_CUDA_NVCC)
else
	NVCC = $(CCACHE) nvcc
endif #LLAMA_CUDA_NVCC
ifdef CUDA_DOCKER_ARCH
	MK_NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else ifndef CUDA_POWER_ARCH
	MK_NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH
ifdef LLAMA_CUDA_FORCE_DMMV
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
ifdef LLAMA_CUDA_FORCE_MMQ
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_MMQ
endif # LLAMA_CUDA_FORCE_MMQ
ifdef LLAMA_CUDA_FORCE_CUBLAS
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_CUBLAS
endif # LLAMA_CUDA_FORCE_CUBLAS
ifdef LLAMA_CUDA_DMMV_X
	MK_NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
else
	MK_NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # LLAMA_CUDA_DMMV_X
ifdef LLAMA_CUDA_MMV_Y
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
else ifdef LLAMA_CUDA_DMMV_Y
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_DMMV_Y) # for backwards compatibility
else
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # LLAMA_CUDA_MMV_Y
ifdef LLAMA_CUDA_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_F16
ifdef LLAMA_CUDA_DMMV_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_DMMV_F16
ifdef LLAMA_CUDA_KQUANTS_ITER
	MK_NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
else
	MK_NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
ifdef LLAMA_CUDA_PEER_MAX_BATCH_SIZE
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=$(LLAMA_CUDA_PEER_MAX_BATCH_SIZE)
else
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
endif # LLAMA_CUDA_PEER_MAX_BATCH_SIZE
ifdef LLAMA_CUDA_NO_PEER_COPY
	MK_NVCCFLAGS += -DGGML_CUDA_NO_PEER_COPY
endif # LLAMA_CUDA_NO_PEER_COPY
ifdef LLAMA_CUDA_CCBIN
	MK_NVCCFLAGS += -ccbin $(LLAMA_CUDA_CCBIN)
endif # LLAMA_CUDA_CCBIN
ifdef LLAMA_CUDA_FA_ALL_QUANTS
	MK_NVCCFLAGS += -DGGML_CUDA_FA_ALL_QUANTS
endif # LLAMA_CUDA_FA_ALL_QUANTS

ifdef JETSON_EOL_MODULE_DETECT
define NVCC_COMPILE
	$(NVCC) -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_CUDA -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/aarch64-linux/include -std=c++11 -O3 $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
else
define NVCC_COMPILE
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
endif # JETSON_EOL_MODULE_DETECT

ggml-cuda/%.o: ggml-cuda/%.cu ggml.h ggml-common.h ggml-cuda/common.cuh
	$(NVCC_COMPILE)

ggml-cuda.o: ggml-cuda.cu ggml-cuda.h ggml.h ggml-backend.h ggml-backend-impl.h ggml-common.h $(wildcard ggml-cuda/*.cuh)
	$(NVCC_COMPILE)
endif # LLAMA_CUDA

ifdef LLAMA_VULKAN
	MK_CPPFLAGS  += -DGGML_USE_VULKAN
	MK_LDFLAGS += -lvulkan
	OBJS    += ggml-vulkan.o

ifdef LLAMA_VULKAN_CHECK_RESULTS
	MK_CPPFLAGS  += -DGGML_VULKAN_CHECK_RESULTS
endif

ifdef LLAMA_VULKAN_DEBUG
	MK_CPPFLAGS  += -DGGML_VULKAN_DEBUG
endif

ifdef LLAMA_VULKAN_MEMORY_DEBUG
	MK_CPPFLAGS  += -DGGML_VULKAN_MEMORY_DEBUG
endif

ifdef LLAMA_VULKAN_VALIDATE
	MK_CPPFLAGS  += -DGGML_VULKAN_VALIDATE
endif

ifdef LLAMA_VULKAN_RUN_TESTS
	MK_CPPFLAGS  += -DGGML_VULKAN_RUN_TESTS
endif

ggml-vulkan.o: ggml-vulkan.cpp ggml-vulkan.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # LLAMA_VULKAN

ifdef LLAMA_HIPBLAS
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH	?= /usr
		AMDGPU_TARGETS ?= $(shell $(shell which amdgpu-arch))
	else
		ROCM_PATH	?= /opt/rocm
		AMDGPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
	endif
	HIPCC                   ?= $(CCACHE) $(ROCM_PATH)/bin/hipcc
	LLAMA_CUDA_DMMV_X       ?= 32
	LLAMA_CUDA_MMV_Y        ?= 1
	LLAMA_CUDA_KQUANTS_ITER ?= 2
	MK_CPPFLAGS += -DGGML_USE_HIPBLAS -DGGML_USE_CUDA
ifdef LLAMA_HIP_UMA
	MK_CPPFLAGS += -DGGML_HIP_UMA
endif # LLAMA_HIP_UMA
	MK_LDFLAGS  += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	MK_LDFLAGS  += -L$(ROCM_PATH)/lib64 -Wl,-rpath=$(ROCM_PATH)/lib64
	MK_LDFLAGS	+= -lhipblas -lamdhip64 -lrocblas
	HIPFLAGS    += $(addprefix --offload-arch=,$(AMDGPU_TARGETS))
	HIPFLAGS    += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
	HIPFLAGS    += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
	HIPFLAGS    += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
ifdef LLAMA_CUDA_FORCE_DMMV
	HIPFLAGS 	+= -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
ifdef LLAMA_CUDA_NO_PEER_COPY
	HIPFLAGS 	+= -DGGML_CUDA_NO_PEER_COPY
endif # LLAMA_CUDA_NO_PEER_COPY
	OBJS        += ggml-cuda.o
	OBJS        += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/*.cu))
	OBJS        += $(OBJS_CUDA_TEMP_INST)

ggml-cuda.o: ggml-cuda.cu ggml-cuda.h ggml.h ggml-backend.h ggml-backend-impl.h ggml-common.h $(wildcard ggml-cuda/*.cuh)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<

ggml-cuda/%.o: ggml-cuda/%.cu ggml.h ggml-common.h ggml-cuda/common.cuh
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<

endif # LLAMA_HIPBLAS

ifdef LLAMA_METAL
	MK_CPPFLAGS += -DGGML_USE_METAL
	MK_LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit
	OBJS		+= ggml-metal.o
ifdef LLAMA_METAL_NDEBUG
	MK_CPPFLAGS += -DGGML_METAL_NDEBUG
endif
ifdef LLAMA_METAL_EMBED_LIBRARY
	MK_CPPFLAGS += -DGGML_METAL_EMBED_LIBRARY
	OBJS        += ggml-metal-embed.o
endif
endif # LLAMA_METAL

ifdef LLAMA_METAL
ggml-metal.o: ggml-metal.m ggml-metal.h ggml.h
	$(CC) $(CFLAGS) -c $< -o $@

ifdef LLAMA_METAL_EMBED_LIBRARY
ggml-metal-embed.o: ggml-metal.metal ggml-common.h
	@echo "Embedding Metal library"
	@sed -e '/#include "ggml-common.h"/r ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml-metal.metal > ggml-metal-embed.metal
	$(eval TEMP_ASSEMBLY=$(shell mktemp))
	@echo ".section __DATA, __ggml_metallib"   >  $(TEMP_ASSEMBLY)
	@echo ".globl _ggml_metallib_start"        >> $(TEMP_ASSEMBLY)
	@echo "_ggml_metallib_start:"              >> $(TEMP_ASSEMBLY)
	@echo ".incbin \"ggml-metal-embed.metal\"" >> $(TEMP_ASSEMBLY)
	@echo ".globl _ggml_metallib_end"          >> $(TEMP_ASSEMBLY)
	@echo "_ggml_metallib_end:"                >> $(TEMP_ASSEMBLY)
	@$(AS) $(TEMP_ASSEMBLY) -o $@
	@rm -f ${TEMP_ASSEMBLY}
endif
endif # LLAMA_METAL

OBJS += ggml-alloc.o ggml-backend.o ggml-quants.o unicode.o unicode-data.o
COMMON_H_DEPS = common/common.h common/sampling.h common/log.h llama.h
COMMON_DEPS   = common.o sampling.o grammar-parser.o build-info.o json-schema-to-grammar.o

ifndef LLAMA_NO_LLAMAFILE
sgemm.o: sgemm.cpp sgemm.h ggml.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

ifdef LLAMA_RPC
ggml-rpc.o: ggml-rpc.cpp ggml-rpc.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

rpc-server.o: examples/rpc/rpc-server.cpp ggml-rpc.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

rpc-server: rpc-server.o ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
endif # LLAMA_RPC

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
ifdef LLAMA_CUDA
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
ifdef LLAMA_CUDA
$(info I NVCC:      $(shell $(NVCC) --version | tail -n 1))
CUDA_VERSION := $(shell $(NVCC) --version | grep -oP 'release (\K[0-9]+\.[0-9])')
ifeq ($(shell awk -v "v=$(CUDA_VERSION)" 'BEGIN { print (v < 11.7) }'),1)
ifndef CUDA_DOCKER_ARCH
ifndef CUDA_POWER_ARCH
$(error I ERROR: For CUDA versions < 11.7 a target CUDA architecture must be explicitly provided via environment variable CUDA_DOCKER_ARCH, e.g. by running "export CUDA_DOCKER_ARCH=compute_XX" on Unix-like systems, where XX is the minimum compute capability that the code needs to run on. A list with compute capabilities can be found here: https://developer.nvidia.com/cuda-gpus )
endif # CUDA_POWER_ARCH
endif # CUDA_DOCKER_ARCH
endif # eq ($(shell echo "$(CUDA_VERSION) < 11.7" | bc),1)
endif # LLAMA_CUDA
$(info )

ifdef LLAMA_CUBLAS
$(info !!!!)
$(info LLAMA_CUBLAS is deprecated and will be removed in the future. Use LLAMA_CUDA instead.)
$(info !!!!)
$(info )
endif

#
# Build library
#

ggml.o: ggml.c ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS)   -c $< -o $@

ggml-alloc.o: ggml-alloc.c ggml.h ggml-alloc.h
	$(CC)  $(CFLAGS)   -c $< -o $@

ggml-backend.o: ggml-backend.c ggml.h ggml-backend.h
	$(CC)  $(CFLAGS)   -c $< -o $@

ggml-quants.o: ggml-quants.c ggml.h ggml-quants.h ggml-common.h
	$(CC) $(CFLAGS)    -c $< -o $@

ggml-blas.o: ggml-blas.cpp ggml-blas.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

unicode.o: unicode.cpp unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

unicode-data.o: unicode-data.cpp unicode-data.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

llama.o: llama.cpp unicode.h ggml.h ggml-alloc.h ggml-backend.h ggml-cuda.h ggml-metal.h llama.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

common.o: common/common.cpp $(COMMON_H_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

sampling.o: common/sampling.cpp $(COMMON_H_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

console.o: common/console.cpp common/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

grammar-parser.o: common/grammar-parser.cpp common/grammar-parser.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

json-schema-to-grammar.o: common/json-schema-to-grammar.cpp common/json-schema-to-grammar.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

train.o: common/train.cpp common/train.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

ngram-cache.o: common/ngram-cache.cpp common/ngram-cache.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libllama.so: llama.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

libllama.a: llama.o ggml.o $(OBJS) $(COMMON_DEPS)
	ar rcs libllama.a llama.o ggml.o $(OBJS) $(COMMON_DEPS)

clean:
	rm -vrf *.o tests/*.o *.so *.a *.dll common/build-info.cpp *.dot $(COV_TARGETS) $(BUILD_TARGETS) $(TEST_TARGETS)
	rm -vrf ggml-cuda/*.o
	rm -vrf ggml-cuda/template-instances/*.o
	find examples pocs -type f -name "*.o" -delete

#
# Examples
#

# $< is the first prerequisite, i.e. the source file.
# Explicitly compile this to an object file so that it can be cached with ccache.
# The source file is then filtered out from $^ (the list of all prerequisites) and the object file is added instead.

# Helper function that replaces .c, .cpp, and .cu file endings with .o:
GET_OBJ_FILE = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(1))))

llama-cli: examples/main/main.cpp                                  ggml.o llama.o $(COMMON_DEPS) console.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./llama-cli -h for help.  ===='
	@echo

llama-infill: examples/infill/infill.cpp                            ggml.o llama.o $(COMMON_DEPS) console.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-simple: examples/simple/simple.cpp                            ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-tokenize: examples/tokenize/tokenize.cpp                      ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-batched: examples/batched/batched.cpp                         ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-batched-bench: examples/batched-bench/batched-bench.cpp       build-info.o ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-quantize: examples/quantize/quantize.cpp                      ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-quantize-stats: examples/quantize-stats/quantize-stats.cpp    build-info.o ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-perplexity: examples/perplexity/perplexity.cpp                ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-imatrix: examples/imatrix/imatrix.cpp                         ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-embedding: examples/embedding/embedding.cpp                   ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gritlm: examples/gritlm/gritlm.cpp                         ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-save-load-state: examples/save-load-state/save-load-state.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-server: examples/server/server.cpp examples/server/utils.hpp examples/server/httplib.h common/json.hpp examples/server/colorthemes.css.hpp examples/server/style.css.hpp examples/server/theme-beeninorder.css.hpp examples/server/theme-ketivah.css.hpp examples/server/theme-mangotango.css.hpp examples/server/theme-playground.css.hpp examples/server/theme-polarnight.css.hpp examples/server/theme-snowstorm.css.hpp examples/server/index.html.hpp examples/server/index-new.html.hpp examples/server/index.js.hpp examples/server/completion.js.hpp examples/server/system-prompts.js.hpp examples/server/prompt-formats.js.hpp examples/server/json-schema-to-grammar.mjs.hpp common/stb_image.h ggml.o llama.o $(COMMON_DEPS) grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp $<,$^) -Iexamples/server $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS) $(LWINSOCK2)

# Portable equivalent of `cd examples/server/public && xxd -i $(notdir $<) ../$(notdir $<).hpp`:
examples/server/%.hpp: examples/server/public/% Makefile
	@( export NAME=$(subst .,_,$(subst -,_,$(notdir $<))) && \
		echo "unsigned char $${NAME}[] = {" && \
		cat $< | od -v -t x1 -An | sed -E 's/([0-9a-fA-F]+)/0x\1, /g' && \
		echo "};" && \
		echo "unsigned int $${NAME}_len = $(shell cat $< | wc -c );" \
	) > $@

llama-gguf: examples/gguf/gguf.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gguf-split: examples/gguf-split/gguf-split.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-eval-callback: examples/eval-callback/eval-callback.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-cvector-generator: examples/cvector-generator/cvector-generator.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-train-text-from-scratch: examples/train-text-from-scratch/train-text-from-scratch.cpp ggml.o llama.o $(COMMON_DEPS) train.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-convert-llama2c-to-ggml: examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-bench: examples/llama-bench/llama-bench.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

libllava.a: examples/llava/llava.cpp examples/llava/llava.h examples/llava/clip.cpp examples/llava/clip.h common/stb_image.h common/base64.hpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -static -fPIC -c $< -o $@ -Wno-cast-qual

llama-llava-cli: examples/llava/llava-cli.cpp examples/llava/clip.h examples/llava/clip.cpp examples/llava/llava.h examples/llava/llava.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) -c examples/llava/clip.cpp  -o $(call GET_OBJ_FILE, examples/llava/clip.cpp) -Wno-cast-qual
	$(CXX) $(CXXFLAGS) -c examples/llava/llava.cpp -o $(call GET_OBJ_FILE, examples/llava/llava.cpp)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $< examples/llava/clip.cpp examples/llava/llava.cpp,$^) $(call GET_OBJ_FILE, $<) $(call GET_OBJ_FILE, examples/llava/clip.cpp) $(call GET_OBJ_FILE, examples/llava/llava.cpp) -o $@ $(LDFLAGS)

llama-baby-llama: examples/baby-llama/baby-llama.cpp ggml.o llama.o $(COMMON_DEPS) train.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-finetune: examples/finetune/finetune.cpp ggml.o llama.o $(COMMON_DEPS) train.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-export-lora: examples/export-lora/export-lora.cpp ggml.o common/common.h $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-retrieval: examples/retrieval/retrieval.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-speculative: examples/speculative/speculative.cpp ggml.o llama.o $(COMMON_DEPS) grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-parallel: examples/parallel/parallel.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookahead: examples/lookahead/lookahead.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup: examples/lookup/lookup.cpp ggml.o llama.o ngram-cache.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-create: examples/lookup/lookup-create.cpp ggml.o llama.o ngram-cache.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-merge: examples/lookup/lookup-merge.cpp ggml.o llama.o ngram-cache.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-lookup-stats: examples/lookup/lookup-stats.cpp ggml.o llama.o ngram-cache.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-passkey: examples/passkey/passkey.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-gbnf-validator: examples/gbnf-validator/gbnf-validator.cpp ggml.o llama.o $(COMMON_DEPS) grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

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

build-info.o: common/build-info.cpp
	$(CXX) $(CXXFLAGS) -c $(filter-out %.h,$^) -o $@

#
# Tests
#

tests: $(TEST_TARGETS)

llama-benchmark-matmult: examples/benchmark/benchmark-matmult.cpp build-info.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

run-benchmark-matmult: llama-benchmark-matmult
	./$@

.PHONY: run-benchmark-matmult swift

llama-vdot: pocs/vdot/vdot.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

llama-q8dot: pocs/vdot/q8dot.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-llama-grammar: tests/test-llama-grammar.cpp ggml.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-grammar-parser: tests/test-grammar-parser.cpp ggml.o llama.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-grammar-integration: tests/test-grammar-integration.cpp json-schema-to-grammar.o ggml.o llama.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-double-float: tests/test-double-float.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-json-schema-to-grammar: tests/test-json-schema-to-grammar.cpp json-schema-to-grammar.o ggml.o llama.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -Iexamples/server -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-grad0: tests/test-grad0.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-opt: tests/test-opt.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-quantize-fns: tests/test-quantize-fns.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-quantize-perf: tests/test-quantize-perf.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-sampling: tests/test-sampling.cpp ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-0: tests/test-tokenizer-0.cpp ggml.o llama.o $(COMMON_DEPS) console.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-1-bpe: tests/test-tokenizer-1-bpe.cpp ggml.o llama.o $(COMMON_DEPS) console.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-tokenizer-1-spm: tests/test-tokenizer-1-spm.cpp ggml.o llama.o $(COMMON_DEPS) console.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-rope: tests/test-rope.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-c.o: tests/test-c.c llama.h
	$(CC) $(CFLAGS) -c $(filter-out %.h,$^) -o $@

tests/test-backend-ops: tests/test-backend-ops.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-model-load-cancel: tests/test-model-load-cancel.cpp ggml.o llama.o tests/get-model.cpp $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-autorelease: tests/test-autorelease.cpp ggml.o llama.o tests/get-model.cpp $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)

tests/test-chat-template: tests/test-chat-template.cpp ggml.o llama.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(filter-out %.h $<,$^) $(call GET_OBJ_FILE, $<) -o $@ $(LDFLAGS)
