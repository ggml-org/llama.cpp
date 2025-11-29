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

#include "bli_obj_check.h"

BLIS_EXPORT_BLIS void bli_obj_create
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       inc_t  rs,
       inc_t  cs,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_create_with_attached_buffer
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       void*  p,
       inc_t  rs,
       inc_t  cs,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_create_without_buffer
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_alloc_buffer
     (
       inc_t  rs,
       inc_t  cs,
       inc_t  is,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_attach_buffer
     (
       void*  p,
       inc_t  rs,
       inc_t  cs,
       inc_t  is,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_create_1x1
     (
       num_t  dt,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_create_1x1_with_attached_buffer
     (
       num_t  dt,
       void*  p,
       obj_t* obj
     );

BLIS_EXPORT_BLIS void bli_obj_create_conf_to
     (
       const obj_t* s,
             obj_t* d
     );

BLIS_EXPORT_BLIS void bli_obj_free
     (
       obj_t* obj
     );

void bli_adjust_strides
     (
       dim_t  m,
       dim_t  n,
       siz_t  elem_size,
       inc_t* rs,
       inc_t* cs,
       inc_t* is
     );

BLIS_EXPORT_BLIS siz_t bli_dt_size
     (
       num_t dt
     );

BLIS_EXPORT_BLIS const char* bli_dt_string
     (
       num_t dt
     );

BLIS_EXPORT_BLIS dim_t bli_align_dim_to_mult
     (
       dim_t dim,
       dim_t dim_mult,
       bool  round_up
     );

BLIS_EXPORT_BLIS dim_t bli_align_dim_to_size
     (
       dim_t dim,
       siz_t elem_size,
       siz_t align_size
     );

BLIS_EXPORT_BLIS dim_t bli_align_ptr_to_size
     (
       const void*  p,
             size_t align_size
     );

BLIS_EXPORT_BLIS void bli_obj_print
     (
       const char*  label,
       const obj_t* obj
     );

