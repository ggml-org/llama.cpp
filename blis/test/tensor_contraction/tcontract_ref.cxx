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

#include "tcontract_ref.hpp"

template <typename T>
void tcontract_ref( const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c )
{
    auto alpha_cast = *( T* )alpha;
    auto beta_cast  = *( T* )beta;
    auto a_cast     = ( T* )a;
    auto b_cast     = ( T* )b;
    auto c_cast     = ( T* )c;

    for_each(m.size(), m.data(), a_cast, rs_a.data(), c_cast, rs_c.data(),
    [&]
    {
        for_each(n.size(), n.data(), b_cast, cs_b.data(), c_cast, cs_c.data(),
        [&]
        {
            auto ab = convert<T>(0.0);

            for_each(k.size(), k.data(), a_cast, cs_a.data(), b_cast, rs_b.data(),
            [&]
            {
                ab += (*a_cast) * (*b_cast);
            });

            if ( beta_cast == convert<T>(0.0) )
            {
                *c_cast = alpha_cast * ab;
            }
            else
            {
                *c_cast = alpha_cast * ab + beta_cast * (*c_cast);
            }
        });

        assert(b_cast == b);
    });

    assert(a_cast == a);
    assert(c_cast == c);
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &tcontract_ref<ctype>;

INSERT_GENTFUNC_BASIC0(tcontract_ref);

static decltype(&tcontract_ref<void>) GENARRAY( tcontract_ref_impl, tcontract_ref );

void tcontract_ref( num_t dt, const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c )
{
    tcontract_ref_impl[ dt ]
    (
      m, n, k,
      alpha, a, rs_a, cs_a,
             b, rs_b, cs_b,
       beta, c, rs_c, cs_c
    );
}

