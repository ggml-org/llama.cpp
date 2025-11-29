/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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

#ifndef BLIS_GKS_H
#define BLIS_GKS_H

int                            bli_gks_init( void );
int                            bli_gks_finalize( void );

void                           bli_gks_init_index( void );

BLIS_EXPORT_BLIS const cntx_t* bli_gks_lookup_id( arch_t id );
void                           bli_gks_register_cntx( arch_t id, void_fp nat_fp, void_fp ref_fp );

BLIS_EXPORT_BLIS const cntx_t* bli_gks_query_cntx( void );
const cntx_t*                  bli_gks_query_cntx_noinit( void );
const cntx_t*                  bli_gks_query_cntx_impl( void );

BLIS_EXPORT_BLIS void          bli_gks_init_ref_cntx( cntx_t* cntx );

bool                           bli_gks_cntx_ukr_is_ref( num_t dt, ukr_t ukr_id, const cntx_t* cntx );
bool                           bli_gks_cntx_ukr2_is_ref( num_t dt1, num_t dt2, ukr_t ukr_id, const cntx_t* cntx );

BLIS_EXPORT_BLIS const char*   bli_gks_l3_ukr_impl_string( ukr_t ukr, ind_t method, num_t dt );
BLIS_EXPORT_BLIS kimpl_t       bli_gks_l3_ukr_impl_type( ukr_t ukr, ind_t method, num_t dt );

//char*                          bli_gks_l3_ukr_avail_impl_string( ukr_t ukr, num_t dt );

BLIS_EXPORT_BLIS err_t bli_gks_register_blksz( kerid_t* bs_id );

BLIS_EXPORT_BLIS err_t bli_gks_register_ukr( kerid_t* ukr_id );

BLIS_EXPORT_BLIS err_t bli_gks_register_ukr2( kerid_t* ukr_id );

BLIS_EXPORT_BLIS err_t bli_gks_register_ukr_pref( kerid_t* ukr_pref_id );

#endif

