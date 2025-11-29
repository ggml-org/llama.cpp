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

void bli_obj_create_check(       num_t  dt,
                                 dim_t  m,
                                 dim_t  n,
                                 inc_t  rs,
                                 inc_t  cs,
                           const obj_t* obj );

void bli_obj_create_without_buffer_check(       num_t  dt,
                                                dim_t  m,
                                                dim_t  n,
                                          const obj_t* obj );

void bli_obj_alloc_buffer_check(       inc_t  rs,
                                       inc_t  cs,
                                       inc_t  is,
                                 const obj_t* obj );

void bli_obj_attach_buffer_check( const void*  p,
                                        inc_t  rs,
                                        inc_t  cs,
                                        inc_t  is,
                                  const obj_t* obj );

void bli_obj_create_scalar_check( num_t dt, const obj_t* obj );

void bli_obj_free_check( const obj_t* obj );

void bli_obj_create_const_check( double value, const obj_t* obj );

void bli_obj_create_const_copy_of_check( const obj_t* a, const obj_t* b );

void bli_dt_size_check( num_t dt );

void bli_dt_string_check( num_t dt );

void bli_dt_union_check( num_t dt1, num_t dt2 );

void bli_obj_print_check( const char* label, const obj_t* obj );

