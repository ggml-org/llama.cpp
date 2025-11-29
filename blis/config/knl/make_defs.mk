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


# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := knl
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
COPTFLAGS      := -O2
endif

ifeq ($(DEBUG_TYPE),sde)
# Unconditionally disable use of libmemkind in Intel SDE.
# Note: The BLIS_DISABLE_MEMKIND macro definition will override
# (undefine) the BLIS_ENABLE_MEMKIND macro definition.
CPPROCFLAGS    += -DBLIS_DISABLE_MEMKIND
# This value is normally set by configure and communicated to make via
# config.mk, however, the make_defs.mk files (this file) get included
# after config.mk, so this definition will override that earlier
# definition.
MK_ENABLE_MEMKIND := no
endif

# Flags specific to optimized kernels.
CKOPTFLAGS     := $(COPTFLAGS) -O3
ifeq ($(CC_VENDOR),gcc)
CKVECFLAGS     := -mavx512f -mavx512pf -mfpmath=sse -march=knl
else
ifeq ($(CC_VENDOR),icc)
CKVECFLAGS     := -xMIC-AVX512
else
ifeq ($(CC_VENDOR),clang)
CKVECFLAGS     := -mavx512f -mavx512pf -mfpmath=sse -march=knl
else
$(error gcc, icc, or clang is required for this configuration.)
endif
endif
endif

# The assembler on OS X won't recognize AVX512 without help.
ifneq ($(CC_VENDOR),icc)
ifeq ($(OS_NAME),Darwin)
CKVECFLAGS     += -Wa,-march=knl
endif
endif

# Flags specific to reference kernels.
# Note: We use AVX2 for reference kernels instead of AVX-512.
CROPTFLAGS     := $(CKOPTFLAGS)
ifeq ($(CC_VENDOR),gcc)
CRVECFLAGS     := -march=knl -mno-avx512f -mno-avx512pf -mno-avx512er -mno-avx512cd -funsafe-math-optimizations -ffp-contract=fast
else
ifeq ($(CC_VENDOR),icc)
CRVECFLAGS     := -xMIC-AVX512
else
ifeq ($(CC_VENDOR),clang)
CRVECFLAGS     := -march=knl -mno-avx512f -mno-avx512pf -mno-avx512er -mno-avx512cd -funsafe-math-optimizations -ffp-contract=fast
else
$(error gcc, icc, or clang is required for this configuration.)
endif
endif
endif

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))

