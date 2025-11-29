/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Southern Methodist University

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

#include "syrk_diagonal_ref.h"
#include "complex_math.hpp"

typedef void (*syrk_diag_ref_vft)
    (
        uplo_t uplo,
        dim_t m,
        dim_t k,
        void* alpha,
        void* a, inc_t rs_a, inc_t cs_a,
        void* d, inc_t incd,
        void* beta,
        void* c, inc_t rs_c, inc_t cs_c
    );

template <typename T>
void syrk_diag_ref
    (
        uplo_t uplo,
        dim_t m,
        dim_t k,
        void* alpha,
        void* a, inc_t rs_a, inc_t cs_a,
        void* d, inc_t incd,
        void* beta,
        void* c, inc_t rs_c, inc_t cs_c
    )
{
    auto alpha_cast = *( T* )alpha;
    auto beta_cast  = *( T* )beta;
    auto a_cast     = ( T* )a;
    auto d_cast     = ( T* )d;
    auto c_cast     = ( T* )c;

    for ( dim_t i = 0; i < m; i++ )
    {
        dim_t j_min = uplo == BLIS_UPPER ? i : 0;
        dim_t j_max = uplo == BLIS_UPPER ? m : i+1;

        for ( dim_t j = j_min; j < j_max; j++ )
        {
            auto ada = convert<T>(0.0);

            for ( dim_t p = 0; p < k; p++ )
            {
                ada += a_cast[ i*rs_a + p*cs_a ] *
                       d_cast[          p*incd ] *
                       a_cast[ j*rs_a + p*cs_a ];
            }

            if ( beta_cast == convert<T>(0.0) )
            {
                c_cast[ i*rs_c + j*cs_c ] = alpha_cast * ada;
            }
            else
            {
                c_cast[ i*rs_c + j*cs_c ] = alpha_cast * ada +
                                             beta_cast * c_cast[ i*rs_c + j*cs_c ];
            }
        }
    }
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &syrk_diag_ref<ctype>;

INSERT_GENTFUNC_BASIC0(syrk_diag_ref);

static syrk_diag_ref_vft GENARRAY( syrk_diag_ref_impl, syrk_diag_ref );

void syrk_diag_ref( obj_t* alpha, obj_t* a, obj_t* d, obj_t* beta, obj_t* c )
{
    num_t dt = bli_obj_dt( a );

    dim_t m = bli_obj_length_after_trans( a );
    dim_t k = bli_obj_width_after_trans( a );

    inc_t rs_a = bli_obj_row_stride( a );
    inc_t cs_a = bli_obj_col_stride( a );
    inc_t rs_c = bli_obj_row_stride( c );
    inc_t cs_c = bli_obj_col_stride( c );
    inc_t incd = bli_obj_row_stride( d );

    if ( bli_obj_has_trans( a ) )
        bli_swap_incs( &rs_a, &cs_a );

    if ( bli_obj_has_trans( c ) )
        bli_swap_incs( &rs_c, &cs_c );

    syrk_diag_ref_impl[ dt ]
    (
      bli_obj_uplo( c ),
      m, k,
      bli_obj_buffer_for_1x1( dt, alpha ),
      bli_obj_buffer_at_off( a ), rs_a, cs_a,
      bli_obj_buffer_at_off( d ), incd,
      bli_obj_buffer_for_1x1( dt, beta ),
      bli_obj_buffer_at_off( c ), rs_c, cs_c
    );
}

