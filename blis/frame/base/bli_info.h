/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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


// -- General library information ----------------------------------------------

BLIS_EXPORT_BLIS const char* bli_info_get_version_str( void );
BLIS_EXPORT_BLIS const char* bli_info_get_int_type_size_str( void );


// -- General configuration-related --------------------------------------------

BLIS_EXPORT_BLIS gint_t bli_info_get_int_type_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_num_fp_types( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_max_type_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_page_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_simd_num_registers( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_simd_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_simd_align_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_stack_buf_max_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_stack_buf_align_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_heap_addr_align_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_heap_stride_align_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_align_size_a( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_align_size_b( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_align_size_c( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_align_size_gen( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_offset_size_a( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_offset_size_b( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_offset_size_c( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_pool_addr_offset_size_gen( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_stay_auto_init( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_blas( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_cblas( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_blas_int_type_size( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_pba_pools( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_sba_pools( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_threading( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_openmp( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_pthreads( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_hpx( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_openmp_as_default( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_pthreads_as_default( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_hpx_as_default( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_thread_jrir_slab( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_thread_jrir_rr( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_thread_jrir_tlb( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_tls( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_memkind( void );
BLIS_EXPORT_BLIS gint_t bli_info_get_enable_sandbox( void );


// -- Kernel implementation-related --------------------------------------------


// -- Level-3 kernel definitions --

BLIS_EXPORT_BLIS const char* bli_info_get_gemm_ukr_impl_string( ind_t method, num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_gemmtrsm_l_ukr_impl_string( ind_t method, num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_gemmtrsm_u_ukr_impl_string( ind_t method, num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_trsm_l_ukr_impl_string( ind_t method, num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_trsm_u_ukr_impl_string( ind_t method, num_t dt );


// -- BLIS implementation query (level-3) --------------------------------------

BLIS_EXPORT_BLIS const char* bli_info_get_gemm_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_gemmt_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_hemm_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_herk_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_her2k_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_symm_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_syrk_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_syr2k_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_trmm_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_trmm3_impl_string( num_t dt );
BLIS_EXPORT_BLIS const char* bli_info_get_trsm_impl_string( num_t dt );

