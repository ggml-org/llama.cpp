/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Southern Methodist University

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

#ifndef BLIS_GENTCONF_MACRO_DEFS_H
#define BLIS_GENTCONF_MACRO_DEFS_H


//
// -- MACROS TO INSERT CONFIGURATION-SPECIFIC MACROS ---------------------------
//


// -- configuration-specific macros which are conditionally-enabled --

// -- Intel architectures ------------------------------------------------------

#ifdef BLIS_CONFIG_SKX
#define INSERT_GENTCONF_SKX GENTCONF( SKX, skx )
#else
#define INSERT_GENTCONF_SKX
#endif
#ifdef BLIS_CONFIG_KNL
#define INSERT_GENTCONF_KNL GENTCONF( KNL, knl )
#else
#define INSERT_GENTCONF_KNL
#endif
#ifdef BLIS_CONFIG_KNC
#define INSERT_GENTCONF_KNC GENTCONF( KNC, knc )
#else
#define INSERT_GENTCONF_KNC
#endif
#ifdef BLIS_CONFIG_HASWELL
#define INSERT_GENTCONF_HASWELL GENTCONF( HASWELL, haswell )
#else
#define INSERT_GENTCONF_HASWELL
#endif
#ifdef BLIS_CONFIG_SANDYBRIDGE
#define INSERT_GENTCONF_SANDYBRIDGE GENTCONF( SANDYBRIDGE, sandybridge )
#else
#define INSERT_GENTCONF_SANDYBRIDGE
#endif
#ifdef BLIS_CONFIG_PENRYN
#define INSERT_GENTCONF_PENRYN GENTCONF( PENRYN, penryn )
#else
#define INSERT_GENTCONF_PENRYN
#endif

// -- AMD architectures --------------------------------------------------------

#ifdef BLIS_CONFIG_ZEN3
#define INSERT_GENTCONF_ZEN3 GENTCONF( ZEN3, zen3 )
#else
#define INSERT_GENTCONF_ZEN3
#endif
#ifdef BLIS_CONFIG_ZEN2
#define INSERT_GENTCONF_ZEN2 GENTCONF( ZEN2, zen2 )
#else
#define INSERT_GENTCONF_ZEN2
#endif
#ifdef BLIS_CONFIG_ZEN
#define INSERT_GENTCONF_ZEN GENTCONF( ZEN, zen )
#else
#define INSERT_GENTCONF_ZEN
#endif
#ifdef BLIS_CONFIG_EXCAVATOR
#define INSERT_GENTCONF_EXCAVATOR GENTCONF( EXCAVATOR, excavator )
#else
#define INSERT_GENTCONF_EXCAVATOR
#endif
#ifdef BLIS_CONFIG_STEAMROLLER
#define INSERT_GENTCONF_STEAMROLLER GENTCONF( STEAMROLLER, steamroller )
#else
#define INSERT_GENTCONF_STEAMROLLER
#endif
#ifdef BLIS_CONFIG_PILEDRIVER
#define INSERT_GENTCONF_PILEDRIVER GENTCONF( PILEDRIVER, piledriver )
#else
#define INSERT_GENTCONF_PILEDRIVER
#endif
#ifdef BLIS_CONFIG_BULLDOZER
#define INSERT_GENTCONF_BULLDOZER GENTCONF( BULLDOZER, bulldozer )
#else
#define INSERT_GENTCONF_BULLDOZER
#endif

// -- ARM architectures --------------------------------------------------------

// -- ARM-SVE --
#ifdef BLIS_CONFIG_ARMSVE
#define INSERT_GENTCONF_ARMSVE GENTCONF( ARMSVE, armsve )
#else
#define INSERT_GENTCONF_ARMSVE
#endif
#ifdef BLIS_CONFIG_A64FX
#define INSERT_GENTCONF_A64FX GENTCONF( A64FX, a64fx )
#else
#define INSERT_GENTCONF_A64FX
#endif

// -- ARM-NEON (4 pipes x 128-bit vectors) --
#ifdef BLIS_CONFIG_ALTRA
#define INSERT_GENTCONF_ALTRA GENTCONF( ALTRA, altra )
#else
#define INSERT_GENTCONF_ALTRA
#endif
#ifdef BLIS_CONFIG_ALTRAMAX
#define INSERT_GENTCONF_ALTRAMAX GENTCONF( ALTRAMAX, altramax )
#else
#define INSERT_GENTCONF_ALTRAMAX
#endif
#ifdef BLIS_CONFIG_FIRESTORM
#define INSERT_GENTCONF_FIRESTORM GENTCONF( FIRESTORM, firestorm )
#else
#define INSERT_GENTCONF_FIRESTORM
#endif

// -- ARM (2 pipes x 128-bit vectors) --
#ifdef BLIS_CONFIG_THUNDERX2
#define INSERT_GENTCONF_THUNDERX2 GENTCONF( THUNDERX2, thunderx2 )
#else
#define INSERT_GENTCONF_THUNDERX2
#endif
#ifdef BLIS_CONFIG_CORTEXA57
#define INSERT_GENTCONF_CORTEXA57 GENTCONF( CORTEXA57, cortexa57 )
#else
#define INSERT_GENTCONF_CORTEXA57
#endif
#ifdef BLIS_CONFIG_CORTEXA53
#define INSERT_GENTCONF_CORTEXA53 GENTCONF( CORTEXA53, cortexa53 )
#else
#define INSERT_GENTCONF_CORTEXA53
#endif

		// -- ARM (older 32-bit microarchitectures) --
#ifdef BLIS_CONFIG_CORTEXA15
#define INSERT_GENTCONF_CORTEXA15 GENTCONF( CORTEXA15, cortexa15 )
#else
#define INSERT_GENTCONF_CORTEXA15
#endif
#ifdef BLIS_CONFIG_CORTEXA9
#define INSERT_GENTCONF_CORTEXA9 GENTCONF( CORTEXA9, cortexa9 )
#else
#define INSERT_GENTCONF_CORTEXA9
#endif

		// -- IBM architectures ------------------------------------------------

#ifdef BLIS_CONFIG_POWER10
#define INSERT_GENTCONF_POWER10 GENTCONF( POWER10, power10 )
#else
#define INSERT_GENTCONF_POWER10
#endif
#ifdef BLIS_CONFIG_POWER9
#define INSERT_GENTCONF_POWER9 GENTCONF( POWER9, power9 )
#else
#define INSERT_GENTCONF_POWER9
#endif
#ifdef BLIS_CONFIG_POWER7
#define INSERT_GENTCONF_POWER7 GENTCONF( POWER7, power7 )
#else
#define INSERT_GENTCONF_POWER7
#endif
#ifdef BLIS_CONFIG_BGQ
#define INSERT_GENTCONF_BGQ GENTCONF( BGQ, bgq )
#else
#define INSERT_GENTCONF_BGQ
#endif

// -- RISC-V architectures ----------------------------------------------------

#ifdef BLIS_CONFIG_RV32I
#define INSERT_GENTCONF_RV32I GENTCONF( RV32I, rv32i )
#else
#define INSERT_GENTCONF_RV32I
#endif
#ifdef BLIS_CONFIG_RV64I
#define INSERT_GENTCONF_RV64I GENTCONF( RV64I, rv64i )
#else
#define INSERT_GENTCONF_RV64I
#endif
#ifdef BLIS_CONFIG_RV32IV
#define INSERT_GENTCONF_RV32IV GENTCONF( RV32IV, rv32iv )
#else
#define INSERT_GENTCONF_RV32IV
#endif
#ifdef BLIS_CONFIG_RV64IV
#define INSERT_GENTCONF_RV64IV GENTCONF( RV64IV, rv64iv )
#else
#define INSERT_GENTCONF_RV64IV
#endif

// -- SiFive architectures ----------------------------------------------------

#ifdef BLIS_CONFIG_SIFIVE_RVV
#define INSERT_GENTCONF_SIFIVE_RVV GENTCONF( SIFIVE_RVV, sifive_rvv )
#else
#define INSERT_GENTCONF_SIFIVE_RVV
#endif
#ifdef BLIS_CONFIG_SIFIVE_X280
#define INSERT_GENTCONF_SIFIVE_X280 GENTCONF( SIFIVE_X280, sifive_x280 )
#else
#define INSERT_GENTCONF_SIFIVE_X280
#endif

// -- Generic architectures ----------------------------------------------------

#ifdef BLIS_CONFIG_GENERIC
#define INSERT_GENTCONF_GENERIC GENTCONF( GENERIC, generic )
#else
#define INSERT_GENTCONF_GENERIC
#endif


// -- configuration-specific macro --

#define INSERT_GENTCONF \
\
INSERT_GENTCONF_SKX \
INSERT_GENTCONF_KNL \
INSERT_GENTCONF_KNC \
INSERT_GENTCONF_HASWELL \
INSERT_GENTCONF_SANDYBRIDGE \
INSERT_GENTCONF_PENRYN \
\
INSERT_GENTCONF_ZEN3 \
INSERT_GENTCONF_ZEN2 \
INSERT_GENTCONF_ZEN \
INSERT_GENTCONF_EXCAVATOR \
INSERT_GENTCONF_STEAMROLLER \
INSERT_GENTCONF_PILEDRIVER \
INSERT_GENTCONF_BULLDOZER \
\
INSERT_GENTCONF_ARMSVE \
INSERT_GENTCONF_A64FX \
\
INSERT_GENTCONF_ALTRAMAX \
INSERT_GENTCONF_ALTRA \
INSERT_GENTCONF_FIRESTORM \
\
INSERT_GENTCONF_THUNDERX2 \
INSERT_GENTCONF_CORTEXA57 \
INSERT_GENTCONF_CORTEXA53 \
\
INSERT_GENTCONF_CORTEXA15 \
INSERT_GENTCONF_CORTEXA9 \
\
INSERT_GENTCONF_POWER10 \
INSERT_GENTCONF_POWER9 \
INSERT_GENTCONF_POWER7 \
INSERT_GENTCONF_BGQ \
\
INSERT_GENTCONF_RV32I \
INSERT_GENTCONF_RV64I \
INSERT_GENTCONF_RV32IV \
INSERT_GENTCONF_RV64IV \
\
INSERT_GENTCONF_SIFIVE_RVV \
INSERT_GENTCONF_SIFIVE_X280 \
\
INSERT_GENTCONF_GENERIC


#endif
