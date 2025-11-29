/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#ifndef BLIS_THREAD_RANGE_TLB_H
#define BLIS_THREAD_RANGE_TLB_H

#if 0
dim_t bli_thread_range_tlb
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
#endif
dim_t bli_thread_range_tlb_l
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_u
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_d
     (
       const dim_t  nt,
       const dim_t  tid,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );

// ---

dim_t bli_thread_range_tlb_trmm_ll
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_trmm_lu
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_trmm_lx_impl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
#if 0
dim_t bli_thread_range_tlb_trmm_r
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
#endif

// ---

dim_t bli_thread_range_tlb_trmm_rl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_trmm_ru
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     );
dim_t bli_thread_range_tlb_trmm_rl_impl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p,
             inc_t* j_en_p,
             inc_t* i_en_p
     );

#endif
