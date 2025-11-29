#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
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


# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := zen3
#CONFIGS_INCL   += $(THIS_CONFIG)

#
# --- Determine the C compiler and related flags ---
#

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.
CPPROCFLAGS    :=
CMISCFLAGS     :=
CPICFLAGS      := -fPIC
CWARNFLAGS     :=

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := -O0
else
COPTFLAGS      := -O3
endif

# Flags specific to optimized and reference kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
CKOPTFLAGS         := $(COPTFLAGS) -fomit-frame-pointer
CROPTFLAGS         := $(CKOPTFLAGS)
CKVECFLAGS         := -mavx2 -mfma
CRVECFLAGS         := $(CKVECFLAGS)
ifeq ($(CC_VENDOR),gcc)
  ifeq ($(GCC_OT_9_1_0),yes)  # gcc versions older than 9.1.
    CVECFLAGS_VER  := -march=znver1 -mno-avx256-split-unaligned-store
  else
  ifeq ($(GCC_OT_10_3_0),yes) # gcc versions 9.1 or newer, but older than 10.3.
    CVECFLAGS_VER  := -march=znver2
  else                        # gcc versions 10.1 or newer.
    CVECFLAGS_VER  := -march=znver3
  endif
  endif
  CKVECFLAGS       += -mfpmath=sse
  CRVECFLAGS       += -funsafe-math-optimizations -ffp-contract=fast
else
ifeq ($(CC_VENDOR),clang)
  ifeq ($(CLANG_OT_9_0_0),yes)  # clang versions older than 9.0.
    CVECFLAGS_VER  := -march=znver1
  else
  ifeq ($(CLANG_OT_12_0_0),yes) # clang versions 9.0 or newer, but older than 12.0.
    CVECFLAGS_VER  := -march=znver2
  else
  ifeq ($(OS_NAME),Darwin)      # clang version 12.0 on OSX lacks znver3 support
    CVECFLAGS_VER  := -march=znver2
  else                          # clang versions 12.0 or newer.
    CVECFLAGS_VER  := -march=znver3
  endif
  endif
  endif
  CKVECFLAGS       += -mfpmath=sse
  CRVECFLAGS       += -funsafe-math-optimizations -ffp-contract=fast
else
ifeq ($(CC_VENDOR),aocc)
  ifeq ($(AOCC_OT_2_0_0),yes)   # aocc versions older than 2.0.
    CVECFLAGS_VER  := -march=znver1
  else
  ifeq ($(AOCC_OT_3_0_0),yes)   # aocc versions 2.0 or newer, but older than 3.0.
    CVECFLAGS_VER  := -march=znver2
  else                          # aocc versions 3.0 or newer.
    CVECFLAGS_VER  := -march=znver3
  endif
  endif
  CKVECFLAGS       += -mfpmath=sse
  CRVECFLAGS       += -funsafe-math-optimizations -ffp-contract=fast
ifeq ($(CC_VENDOR),NVIDIA)
  CVECFLAGS_VER    := -march=znver3
  CRVECFLAGS       += -fast
else
  $(error gcc, clang, nvc or aocc is required for this configuration.)
endif
endif
endif
endif
CKVECFLAGS         += $(CVECFLAGS_VER)
CRVECFLAGS         += $(CVECFLAGS_VER)

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))
