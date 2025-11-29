/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_L1M_KER_PARAMS_H
#define BLIS_L1M_KER_PARAMS_H


//
// -- Level-1m kernel function types -------------------------------------------
//

// packm

// NOTE: This is the function type for the structure-aware "kernel".

#define packm_params \
\
             struc_t strucc, \
             diag_t  diagc, \
             uplo_t  uploc, \
             conj_t  conjc, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   panel_dim, \
             dim_t   panel_len, \
             dim_t   panel_dim_max, \
             dim_t   panel_len_max, \
             dim_t   panel_dim_off, \
             dim_t   panel_len_off, \
             dim_t   panel_bcast, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params  \


// packm_cxk (packm microkernel)

#define packm_cxk_params \
\
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   cdim_max, \
             dim_t   cdim_bcast, \
             dim_t   n, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const void*   params  \


// unpackm_cxk kernel

#define unpackm_cxk_params \
\
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   cdim_bcast, \
             dim_t   n, \
       const void*   kappa, \
       const void*   p,             inc_t ldp, \
             void*   a, inc_t inca, inc_t lda, \
       const void*   params  \


// packm_cxc_diag kernel

#define packm_cxc_diag_params \
\
             struc_t struca, \
             diag_t  diaga, \
             uplo_t  uploa, \
             conj_t  conja, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   cdim, \
             dim_t   cdim_max, \
             dim_t   cdim_bcast, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const void*   params  \


#endif

