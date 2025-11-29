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

#include "blis.h"


// -- General library information ----------------------------------------------

// This string gets defined via -D on the command line when BLIS is compiled.
// This string is (or rather, should be) only used here.
static const char* bli_version_str       = BLIS_VERSION_STRING;
static const char* bli_int_type_size_str = STRINGIFY_INT( BLIS_INT_TYPE_SIZE );

const char* bli_info_get_version_str( void )          { return bli_version_str; }
const char* bli_info_get_int_type_size_str( void )    { return bli_int_type_size_str; }


// -- General configuration-related --------------------------------------------

gint_t bli_info_get_int_type_size( void )             { return BLIS_INT_TYPE_SIZE; }
gint_t bli_info_get_num_fp_types( void )              { return BLIS_NUM_FP_TYPES; }
gint_t bli_info_get_max_type_size( void )             { return BLIS_MAX_TYPE_SIZE; }
gint_t bli_info_get_page_size( void )                 { return BLIS_PAGE_SIZE; }
gint_t bli_info_get_simd_num_registers( void )        { return BLIS_SIMD_MAX_NUM_REGISTERS; }
gint_t bli_info_get_simd_size( void )                 { return BLIS_SIMD_MAX_SIZE; }
gint_t bli_info_get_simd_align_size( void )           { return BLIS_SIMD_ALIGN_SIZE; }
gint_t bli_info_get_stack_buf_max_size( void )        { return BLIS_STACK_BUF_MAX_SIZE; }
gint_t bli_info_get_stack_buf_align_size( void )      { return BLIS_STACK_BUF_ALIGN_SIZE; }
gint_t bli_info_get_heap_addr_align_size( void )      { return BLIS_HEAP_ADDR_ALIGN_SIZE; }
gint_t bli_info_get_heap_stride_align_size( void )    { return BLIS_HEAP_STRIDE_ALIGN_SIZE; }
gint_t bli_info_get_pool_addr_align_size_a( void )    { return BLIS_POOL_ADDR_ALIGN_SIZE_A; }
gint_t bli_info_get_pool_addr_align_size_b( void )    { return BLIS_POOL_ADDR_ALIGN_SIZE_B; }
gint_t bli_info_get_pool_addr_align_size_c( void )    { return BLIS_POOL_ADDR_ALIGN_SIZE_C; }
gint_t bli_info_get_pool_addr_align_size_gen( void )  { return BLIS_POOL_ADDR_ALIGN_SIZE_GEN; }
gint_t bli_info_get_pool_addr_offset_size_a( void )   { return BLIS_POOL_ADDR_OFFSET_SIZE_A; }
gint_t bli_info_get_pool_addr_offset_size_b( void )   { return BLIS_POOL_ADDR_OFFSET_SIZE_B; }
gint_t bli_info_get_pool_addr_offset_size_c( void )   { return BLIS_POOL_ADDR_OFFSET_SIZE_C; }
gint_t bli_info_get_pool_addr_offset_size_gen( void ) { return BLIS_POOL_ADDR_OFFSET_SIZE_GEN; }
gint_t bli_info_get_enable_blas( void )
{
#ifdef BLIS_ENABLE_BLAS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_cblas( void )
{
#ifdef BLIS_ENABLE_CBLAS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_blas_int_type_size( void ) { return BLIS_BLAS_INT_TYPE_SIZE; }
gint_t bli_info_get_enable_pba_pools( void )
{
#ifdef BLIS_ENABLE_PBA_POOLS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_sba_pools( void )
{
#ifdef BLIS_ENABLE_SBA_POOLS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_threading( void )
{
	if ( bli_info_get_enable_openmp()   ||
	     bli_info_get_enable_pthreads() ||
	     bli_info_get_enable_hpx() ) return 1;
	else                             return 0;
}
gint_t bli_info_get_enable_openmp( void )
{
#ifdef BLIS_ENABLE_OPENMP
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_pthreads( void )
{
#ifdef BLIS_ENABLE_PTHREADS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_hpx( void )
{
#ifdef BLIS_ENABLE_HPX
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_openmp_as_default( void )
{
#ifdef BLIS_ENABLE_OPENMP_AS_DEFAULT
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_pthreads_as_default( void )
{
#ifdef BLIS_ENABLE_PTHREADS_AS_DEFAULT
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_hpx_as_default( void )
{
#ifdef BLIS_ENABLE_HPX_AS_DEFAULT
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_thread_jrir_slab( void )
{
#ifdef BLIS_ENABLE_JRIR_SLAB
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_thread_jrir_rr( void )
{
#ifdef BLIS_ENABLE_JRIR_RR
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_thread_jrir_tlb( void )
{
#ifdef BLIS_ENABLE_JRIR_TLB
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_tls( void )
{
#ifdef BLIS_ENABLE_TLS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_memkind( void )
{
#ifdef BLIS_ENABLE_MEMKIND
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_sandbox( void )
{
#ifdef BLIS_ENABLE_SANDBOX
	return 1;
#else
	return 0;
#endif
}


// -- Kernel implementation-related --------------------------------------------


// -- Level-3 kernel definitions --

const char* bli_info_get_gemm_ukr_impl_string( ind_t method, num_t dt )
{ bli_init_once(); return bli_gks_l3_ukr_impl_string( BLIS_GEMM_UKR,       method, dt ); }
const char* bli_info_get_gemmtrsm_l_ukr_impl_string( ind_t method, num_t dt )
{ bli_init_once(); return bli_gks_l3_ukr_impl_string( BLIS_GEMMTRSM_L_UKR, method, dt ); }
const char* bli_info_get_gemmtrsm_u_ukr_impl_string( ind_t method, num_t dt )
{ bli_init_once(); return bli_gks_l3_ukr_impl_string( BLIS_GEMMTRSM_U_UKR, method, dt ); }
const char* bli_info_get_trsm_l_ukr_impl_string( ind_t method, num_t dt )
{ bli_init_once(); return bli_gks_l3_ukr_impl_string( BLIS_TRSM_L_UKR,     method, dt ); }
const char* bli_info_get_trsm_u_ukr_impl_string( ind_t method, num_t dt )
{ bli_init_once(); return bli_gks_l3_ukr_impl_string( BLIS_TRSM_U_UKR,     method, dt ); }


// -- BLIS implementation query (level-3) --------------------------------------

const char* bli_info_get_gemm_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_GEMM,  dt ); }
const char* bli_info_get_gemmt_impl_string( num_t dt ) { return bli_ind_oper_get_avail_impl_string( BLIS_GEMMT, dt ); }
const char* bli_info_get_hemm_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_HEMM,  dt ); }
const char* bli_info_get_herk_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_GEMMT, dt ); }
const char* bli_info_get_her2k_impl_string( num_t dt ) { return bli_ind_oper_get_avail_impl_string( BLIS_GEMMT, dt ); }
const char* bli_info_get_symm_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_SYMM,  dt ); }
const char* bli_info_get_syrk_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_GEMMT, dt ); }
const char* bli_info_get_syr2k_impl_string( num_t dt ) { return bli_ind_oper_get_avail_impl_string( BLIS_GEMMT, dt ); }
const char* bli_info_get_trmm_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_TRMM,  dt ); }
const char* bli_info_get_trmm3_impl_string( num_t dt ) { return bli_ind_oper_get_avail_impl_string( BLIS_TRMM3, dt ); }
const char* bli_info_get_trsm_impl_string( num_t dt )  { return bli_ind_oper_get_avail_impl_string( BLIS_TRSM,  dt ); }

