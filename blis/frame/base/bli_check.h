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


BLIS_EXPORT_BLIS err_t bli_check_error_code_helper( gint_t code, const char* file, guint_t line );

err_t bli_check_valid_error_level( errlev_t level );

err_t bli_check_null_pointer( const void* ptr );

err_t bli_check_valid_side( side_t side );
err_t bli_check_valid_uplo( uplo_t uplo );
err_t bli_check_valid_trans( trans_t trans );
err_t bli_check_valid_diag( diag_t diag );
err_t bli_check_nonunit_diag( const obj_t* a );

err_t bli_check_valid_datatype( num_t dt );
err_t bli_check_object_valid_datatype( const obj_t* a );
err_t bli_check_noninteger_datatype( num_t dt );
err_t bli_check_noninteger_object( const obj_t* a );
err_t bli_check_nonconstant_datatype( num_t dt );
err_t bli_check_nonconstant_object( const obj_t* a );
err_t bli_check_floating_datatype( num_t dt );
err_t bli_check_floating_object( const obj_t* a );
err_t bli_check_real_datatype( num_t dt );
err_t bli_check_real_object( const obj_t* a );
err_t bli_check_integer_datatype( num_t dt );
err_t bli_check_integer_object( const obj_t* a );
err_t bli_check_consistent_datatypes( num_t dt_a, num_t dt_b );
err_t bli_check_consistent_object_datatypes( const obj_t* a, const obj_t* b );
err_t bli_check_datatype_real_proj_of( num_t dt_c, num_t dt_r );
err_t bli_check_object_real_proj_of( const obj_t* c, const obj_t* r );
err_t bli_check_real_valued_object( const obj_t* a );
err_t bli_check_consistent_precisions( num_t dt_a, num_t dt_b );
err_t bli_check_consistent_object_precisions( const obj_t* a, const obj_t* b );

err_t bli_check_conformal_dims( const obj_t* a, const obj_t* b );
err_t bli_check_level3_dims( const obj_t* a, const obj_t* b, const obj_t* c );
err_t bli_check_scalar_object( const obj_t* a );
err_t bli_check_vector_object( const obj_t* a );
err_t bli_check_matrix_object( const obj_t* a );
err_t bli_check_equal_vector_lengths( const obj_t* x, const obj_t* y );
err_t bli_check_square_object( const obj_t* a );
err_t bli_check_object_length_equals( const obj_t* a, dim_t m );
err_t bli_check_object_width_equals( const obj_t* a, dim_t n );
err_t bli_check_vector_dim_equals( const obj_t* a, dim_t n );
err_t bli_check_object_diag_offset_equals( const obj_t* a, doff_t offset );

err_t bli_check_matrix_strides( dim_t m, dim_t n, inc_t rs, inc_t cs, inc_t is );

err_t bli_check_general_object( const obj_t* a );
err_t bli_check_hermitian_object( const obj_t* a );
err_t bli_check_symmetric_object( const obj_t* a );
err_t bli_check_triangular_object( const obj_t* a );
err_t bli_check_object_struc( const obj_t* a, struc_t struc );

err_t bli_check_upper_or_lower_object( const obj_t* a );

err_t bli_check_valid_3x1_subpart( subpart_t part );
err_t bli_check_valid_1x3_subpart( subpart_t part );
err_t bli_check_valid_3x3_subpart( subpart_t part );

err_t bli_check_valid_cntl( const void* cntl );

err_t bli_check_packm_schema_on_unpack( const obj_t* a );
err_t bli_check_packv_schema_on_unpack( const obj_t* a );

err_t bli_check_object_buffer( const obj_t* a );

err_t bli_check_valid_malloc_buf( const void* ptr );

err_t bli_check_valid_packbuf( packbuf_t buf_type );
err_t bli_check_if_exhausted_pool( const pool_t* pool );
err_t bli_check_sufficient_stack_buf_size( const cntx_t* cntx );
err_t bli_check_alignment_is_power_of_two( size_t align_size );
err_t bli_check_alignment_is_mult_of_ptr_size( size_t align_size );

err_t bli_check_object_alias_of( const obj_t* a, const obj_t* b );

err_t bli_check_valid_arch_id( arch_t id );
err_t bli_check_initialized_gks_cntx( const cntx_t* cntx );

err_t bli_check_valid_mc_mod_mult( const blksz_t* mc, const blksz_t* mr );
err_t bli_check_valid_nc_mod_mult( const blksz_t* nc, const blksz_t* nr );
err_t bli_check_valid_kc_mod_mult( const blksz_t* kc, const blksz_t* kr );

err_t bli_check_valid_mr_even( const blksz_t* mr, const mbool_t* row_pref );
err_t bli_check_valid_nr_even( const blksz_t* mr, const mbool_t* row_pref );

