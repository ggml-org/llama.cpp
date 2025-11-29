/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_ARCH_CONFIG_H
#define BLIS_ARCH_CONFIG_H

//
// -- Context initialization prototypes ----------------------------------------
//

#undef GENTCONF
#define GENTCONF( CONFIG, config ) CNTX_INIT_PROTS( config )

INSERT_GENTCONF

//
// -- Architecture family-specific headers -------------------------------------
//

// -- x86_64 families --

#ifdef BLIS_FAMILY_INTEL64
#include "bli_family_intel64.h"
#endif
#ifdef BLIS_FAMILY_AMD64
#include "bli_family_amd64.h"
#endif
#ifdef BLIS_FAMILY_AMD64_LEGACY
#include "bli_family_amd64_legacy.h"
#endif
#ifdef BLIS_FAMILY_X86_64
#include "bli_family_x86_64.h"
#endif

// -- Intel64 architectures --

#ifdef BLIS_FAMILY_SKX
#include "bli_family_skx.h"
#endif
#ifdef BLIS_FAMILY_KNL
#include "bli_family_knl.h"
#endif
#ifdef BLIS_FAMILY_KNC
#include "bli_family_knc.h"
#endif
#ifdef BLIS_FAMILY_HASWELL
#include "bli_family_haswell.h"
#endif
#ifdef BLIS_FAMILY_SANDYBRIDGE
#include "bli_family_sandybridge.h"
#endif
#ifdef BLIS_FAMILY_PENRYN
#include "bli_family_penryn.h"
#endif

// -- AMD64 architectures --

#ifdef BLIS_FAMILY_ZEN3
#include "bli_family_zen3.h"
#endif
#ifdef BLIS_FAMILY_ZEN2
#include "bli_family_zen2.h"
#endif
#ifdef BLIS_FAMILY_ZEN
#include "bli_family_zen.h"
#endif
#ifdef BLIS_FAMILY_EXCAVATOR
#include "bli_family_excavator.h"
#endif
#ifdef BLIS_FAMILY_STEAMROLLER
#include "bli_family_steamroller.h"
#endif
#ifdef BLIS_FAMILY_PILEDRIVER
#include "bli_family_piledriver.h"
#endif
#ifdef BLIS_FAMILY_BULLDOZER
#include "bli_family_bulldozer.h"
#endif

// -- ARM families --
#ifdef BLIS_FAMILY_ARM64
#include "bli_family_arm64.h"
#endif
#ifdef BLIS_FAMILY_ARM32
#include "bli_family_arm32.h"
#endif

// -- ARM architectures --

// ARM-SVE
#ifdef BLIS_FAMILY_ARMSVE
#include "bli_family_armsve.h"
#endif
#ifdef BLIS_FAMILY_A64FX
#include "bli_family_a64fx.h"
#endif
// ARM-NEON (4x128)
#ifdef BLIS_FAMILY_ALTRAMAX
#include "bli_family_altramax.h"
#endif
#ifdef BLIS_FAMILY_ALTRA
#include "bli_family_altra.h"
#endif
#ifdef BLIS_FAMILY_FIRESTORM
#include "bli_family_firestorm.h"
#endif
// ARM-NEON (2x128)
#ifdef BLIS_FAMILY_THUNDERX2
#include "bli_family_thunderx2.h"
#endif
#ifdef BLIS_FAMILY_CORTEXA57
#include "bli_family_cortexa57.h"
#endif
#ifdef BLIS_FAMILY_CORTEXA53
#include "bli_family_cortexa53.h"
#endif
// ARM 32-bit (vintage)
#ifdef BLIS_FAMILY_CORTEXA15
#include "bli_family_cortexa15.h"
#endif
#ifdef BLIS_FAMILY_CORTEXA9
#include "bli_family_cortexa9.h"
#endif

// -- IBM Power families --
#ifdef BLIS_FAMILY_POWER
#include "bli_family_power.h"
#endif

// -- IBM Power architectures --

#ifdef BLIS_FAMILY_POWER10
#include "bli_family_power10.h"
#endif
#ifdef BLIS_FAMILY_POWER9
#include "bli_family_power9.h"
#endif
#ifdef BLIS_FAMILY_POWER7
#include "bli_family_power7.h"
#endif

// -- IBM BG/Q --

#ifdef BLIS_FAMILY_BGQ
#include "bli_family_bgq.h"
#endif

// -- SiFive families --

#ifdef BLIS_FAMILY_SIFIVE_RVV
#include "bli_family_sifive_rvv.h"
#endif
#ifdef BLIS_FAMILY_SIFIVE_X280
#include "bli_family_sifive_x280.h"
#endif

// -- Generic --

#ifdef BLIS_FAMILY_GENERIC
#include "bli_family_generic.h"
#endif


//
// -- kernel set prototypes ----------------------------------------------------
//

// -- Intel64 architectures --
#ifdef BLIS_KERNELS_SKX
#include "bli_kernels_skx.h"
#endif
#ifdef BLIS_KERNELS_KNL
#include "bli_kernels_knl.h"
#endif
#ifdef BLIS_KERNELS_KNC
#include "bli_kernels_knc.h"
#endif
#ifdef BLIS_KERNELS_HASWELL
#include "bli_kernels_haswell.h"
#endif
#ifdef BLIS_KERNELS_SANDYBRIDGE
#include "bli_kernels_sandybridge.h"
#endif
#ifdef BLIS_KERNELS_PENRYN
#include "bli_kernels_penryn.h"
#endif

// -- AMD64 architectures --

#ifdef BLIS_KERNELS_ZEN2
#include "bli_kernels_zen2.h"
#endif
#ifdef BLIS_KERNELS_ZEN
#include "bli_kernels_zen.h"
#endif
//#ifdef BLIS_KERNELS_EXCAVATOR
//#include "bli_kernels_excavator.h"
//#endif
//#ifdef BLIS_KERNELS_STEAMROLLER
//#include "bli_kernels_steamroller.h"
//#endif
#ifdef BLIS_KERNELS_PILEDRIVER
#include "bli_kernels_piledriver.h"
#endif
#ifdef BLIS_KERNELS_BULLDOZER
#include "bli_kernels_bulldozer.h"
#endif

// -- ARM architectures --

#ifdef BLIS_KERNELS_ARMSVE
#include "bli_kernels_armsve.h"
#endif
#ifdef BLIS_KERNELS_ARMV8A
#include "bli_kernels_armv8a.h"
#endif
#ifdef BLIS_KERNELS_ARMV7A
#include "bli_kernels_armv7a.h"
#endif

// -- IBM Power --

#ifdef BLIS_KERNELS_POWER10
#include "bli_kernels_power10.h"
#endif
#ifdef BLIS_KERNELS_POWER9
#include "bli_kernels_power9.h"
#endif
#ifdef BLIS_KERNELS_POWER7
#include "bli_kernels_power7.h"
#endif

// -- IBM BG/Q --

#ifdef BLIS_KERNELS_BGQ
#include "bli_kernels_bgq.h"
#endif

// -- RISC-V --

#ifdef BLIS_KERNELS_RVI
#include "bli_kernels_rvi.h"
#endif
#ifdef BLIS_KERNELS_RVIV
#include "bli_kernels_rviv.h"
#endif

// -- SiFive RISC-V architectures --

#ifdef BLIS_KERNELS_SIFIVE_RVV
#include "bli_kernels_sifive_rvv.h"
#endif
#ifdef BLIS_KERNELS_SIFIVE_X280
#include "bli_kernels_sifive_x280.h"
#endif


#endif

