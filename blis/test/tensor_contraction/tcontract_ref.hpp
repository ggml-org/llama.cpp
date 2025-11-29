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

#include "blis.h"
#include "complex_math.hpp"

#include <vector>
#include <array>
#include <cassert>

inline void increment(inc_t, gint_t) {}

template <typename T, typename... Args>
void increment(inc_t n, gint_t i, T& off, const inc_t* s, Args&... args)
{
    off += s[i]*n;
    increment(n, i, args...);
}

template <typename Body, typename... Args>
void for_each_impl(gint_t ndim, const dim_t* n,
                   dim_t off, dim_t len,
                   Body& body,
                   Args&... args)
{
    std::array<dim_t,8> i = {};
    assert( ndim <= i.size() );

    if ( off )
    {
        for ( gint_t k = 0; k < ndim; k++ )
        {
            i[k] = off % n[k];
            off /= n[k];
            increment(i[k], k, args...);
        }
    }

    for ( dim_t pos = 0; pos < len; pos++ )
    {
        body();

        for ( gint_t k = 0; k < ndim; k++ )
        {
            if ( i[k] == n[k]-1 )
            {
                increment(-i[k], k, args...);
                i[k] = 0;
            }
            else
            {
                increment(1, k, args...);
                i[k]++;
                break;
            }
        }
    }
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              dim_t off, dim_t len,
              T& a, const inc_t* s_a,
              Body&& body)
{
    for_each_impl( ndim, n, off, len, body, a, s_a );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              dim_t off, dim_t len,
              T& a, const inc_t* s_a,
              T& b, const inc_t* s_b,
              Body&& body)
{
    for_each_impl( ndim, n, off, len, body, a, s_a, b, s_b );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              T& a, const inc_t* s_a,
              Body&& body)
{
    dim_t len = 1;
    for ( gint_t i = 0;i < ndim;i++ ) len *= n[i];
    for_each_impl( ndim, n, 0, len, body, a, s_a );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              T& a, const inc_t* s_a,
              T& b, const inc_t* s_b,
              Body&& body)
{
    dim_t len = 1;
    for ( gint_t i = 0;i < ndim;i++ ) len *= n[i];
    for_each_impl( ndim, n, 0, len, body, a, s_a, b, s_b );
}

void tcontract_ref( num_t dt, const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c );
