/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#ifndef BLIS_THREAD_H
#define BLIS_THREAD_H

// Include thread communicator (thrcomm_t) object definitions and prototypes.
#include "bli_thrcomm.h"

// Include thread info (thrinfo_t) object definitions and prototypes.
#include "bli_thrinfo.h"

// Thread lanuch prototypes. Must go before including implementation headers.
typedef void (*thread_func_t)( thrcomm_t* gl_comm, dim_t tid, const void* params );

// Include threading implementations.
#include "bli_thread_openmp.h"
#include "bli_thread_pthreads.h"
#include "bli_thread_hpx.h"
#include "bli_thread_single.h"

// Initialization-related prototypes.
int bli_thread_init( void );
int bli_thread_finalize( void );

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS void bli_thread_launch
     (
             timpl_t       ti,
             dim_t         nt,
             thread_func_t func,
       const void*         params
     );

// -----------------------------------------------------------------------------

// Factorization and partitioning prototypes
typedef struct bli_prime_factors_s
{
    dim_t n;
    dim_t sqrt_n;
    dim_t f;
} bli_prime_factors_t;

void bli_prime_factorization( dim_t n, bli_prime_factors_t* factors );

dim_t bli_next_prime_factor( bli_prime_factors_t* factors );
bool  bli_is_prime( dim_t n );

void bli_thread_partition_2x2
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     );
void bli_thread_partition_2x2_slow
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     );
void bli_thread_partition_2x2_fast
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     );

// -----------------------------------------------------------------------------

dim_t bli_gcd( dim_t x, dim_t y );
dim_t bli_lcm( dim_t x, dim_t y );
dim_t bli_ipow( dim_t base, dim_t power );

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS dim_t   bli_thread_get_jc_nt( void );
BLIS_EXPORT_BLIS dim_t   bli_thread_get_pc_nt( void );
BLIS_EXPORT_BLIS dim_t   bli_thread_get_ic_nt( void );
BLIS_EXPORT_BLIS dim_t   bli_thread_get_jr_nt( void );
BLIS_EXPORT_BLIS dim_t   bli_thread_get_ir_nt( void );
BLIS_EXPORT_BLIS dim_t   bli_thread_get_num_threads( void );
BLIS_EXPORT_BLIS timpl_t bli_thread_get_thread_impl( void );
BLIS_EXPORT_BLIS const char* bli_thread_get_thread_impl_str( timpl_t ti );

BLIS_EXPORT_BLIS void    bli_thread_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir );
BLIS_EXPORT_BLIS void    bli_thread_set_num_threads( dim_t value );
BLIS_EXPORT_BLIS void    bli_thread_set_thread_impl( timpl_t ti );
BLIS_EXPORT_BLIS void    bli_thread_reset( void );


#endif
