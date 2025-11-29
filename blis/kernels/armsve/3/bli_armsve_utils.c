/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Forschunszentrum Juelich
   Copyright (C) 2020, The University of Tokyo

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

dim_t bli_vl_bytes_armsve(void)
{ \
    uint64_t vl = 0;
    __asm__ (
      " mov  x0, xzr   \n\t"
      " incb x0        \n\t"
      " mov  %[vl], x0 \n\t"
    : [vl] "=r" (vl)
    :
    : "x0"
     );
    return vl;
}


#define EXPANDMAC_BLKSZ_ARMSVE(ch, S_Data) \
void PASTEMAC(ch, _blksz_armsve) (dim_t *m_r_, dim_t *n_r_, \
                                  dim_t *k_c_, dim_t *m_c_, dim_t *n_c_) \
{ \
    dim_t W_L1 = bli_env_get_var("BLIS_SVE_W_L1", W_L1_SVE_DEFAULT); \
    dim_t N_L1 = bli_env_get_var("BLIS_SVE_N_L1", N_L1_SVE_DEFAULT); \
    dim_t C_L1 = bli_env_get_var("BLIS_SVE_C_L1", C_L1_SVE_DEFAULT); \
    dim_t W_L2 = bli_env_get_var("BLIS_SVE_W_L2", W_L2_SVE_DEFAULT); \
    dim_t N_L2 = bli_env_get_var("BLIS_SVE_N_L2", N_L2_SVE_DEFAULT); \
    dim_t C_L2 = bli_env_get_var("BLIS_SVE_C_L2", C_L2_SVE_DEFAULT); \
    dim_t W_L3 = bli_env_get_var("BLIS_SVE_W_L3", W_L3_SVE_DEFAULT); \
    dim_t N_L3 = bli_env_get_var("BLIS_SVE_N_L3", N_L3_SVE_DEFAULT); \
    dim_t C_L3 = bli_env_get_var("BLIS_SVE_C_L3", C_L3_SVE_DEFAULT); \
\
    dim_t vl_b = bli_vl_bytes_armsve(); \
    dim_t vl = vl_b / S_Data; \
    dim_t m_r = 2 * vl; \
    dim_t n_r = 10; \
\
    dim_t k_c = (dim_t)( floor((W_L1 - 1.0)/(1.0 + (double)n_r/m_r)) * N_L1 * C_L1 ) \
        / (n_r * S_Data); \
\
    dim_t C_Ac = W_L2 - 1 - ceil( (2.0 * k_c * n_r * S_Data)/(C_L2 * N_L2) ); \
    dim_t m_c = C_Ac * (N_L2 * C_L2)/(k_c * S_Data); \
    m_c -= m_c % m_r; \
\
    dim_t C_Bc = W_L3 - 1 - ceil( (2.0 * k_c * m_c * S_Data)/(C_L3 * N_L3) ); \
    dim_t n_c = C_Bc * (N_L3 * C_L3)/(k_c * S_Data); \
    n_c -= n_c % n_r; \
\
    /* Ensure non-zero block sizes. */ \
    m_c = bli_max(m_c, m_r); \
    n_c = bli_max(n_c, n_r); \
    k_c = bli_max(k_c, 128); \
\
    *m_r_ = m_r; \
    *n_r_ = n_r; \
    *k_c_ = k_c; \
    *m_c_ = m_c; \
    *n_c_ = n_c; \
}

EXPANDMAC_BLKSZ_ARMSVE( s, 4 )
EXPANDMAC_BLKSZ_ARMSVE( d, 8 )
EXPANDMAC_BLKSZ_ARMSVE( c, 8  )
EXPANDMAC_BLKSZ_ARMSVE( z, 16 )

