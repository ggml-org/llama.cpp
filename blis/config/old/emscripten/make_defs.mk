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

# Only include this block of code once.
ifndef MAKE_DEFS_MK_INCLUDED
MAKE_DEFS_MK_INCLUDED := yes



#
# --- Development tools definitions --------------------------------------------
#

# --- Determine the C compiler and related flags ---
CC             := emcc
CC_VENDOR      := emcc
# Enable IEEE Standard 1003.1-2004 (POSIX.1d). 
# NOTE: This is needed to enable posix_memalign().
CPPROCFLAGS    := -D_POSIX_C_SOURCE=200112L
CMISCFLAGS     := -std=c99
CPICFLAGS      := -fPIC
CDBGFLAGS      := #-g4
CWARNFLAGS     := -Wall -Wno-unused-function -Wfatal-errors
COPTFLAGS      := -O2
CKOPTFLAGS     := -O3
CKVECFLAGS     :=

# --- Determine the archiver and related flags ---
AR             := emar
RANLIB         := emranlib
ARFLAGS        := cr

# --- Determine the linker and related flags ---
LINKER         := $(CC)
SOFLAGS        := -shared
LDFLAGS        := -O3 -s TOTAL_MEMORY=67108864 -s FORCE_ALIGNED_MEMORY=1 -s PRECISE_F32=2 -s GC_SUPPORT=0 

# --- Determine JS interpreter ---
JSINT          := node

# end of ifndef MAKE_DEFS_MK_INCLUDED conditional block
endif
