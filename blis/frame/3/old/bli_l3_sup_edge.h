/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

static
void bli_dgemmsup_ker_edge_dispatcher
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx,
       const dim_t         num_mr,
       const dim_t         num_nr,
       dim_t*     restrict mrs,
       dim_t*     restrict nrs,
       dgemmsup_ker_ft*    kmap
     )
{
	#if 1

	// outer loop = mr; inner loop = nr

	dim_t            n_left = n0;
	double* restrict cj     = c;
	double* restrict bj     = b;

	for ( dim_t j = 0; n_left != 0; ++j )
	{
		const dim_t nr_cur = nrs[ j ];

		if ( nr_cur <= n_left )
		{
			dim_t            m_left = m0;
			double* restrict cij    = cj;
			double* restrict ai     = a;

			for ( dim_t i = 0; m_left != 0; ++i )
			{
				const dim_t mr_cur = mrs[ i ];

				if ( mr_cur <= m_left )
				{
					dgemmsup_ker_ft ker_fp = kmap[ i*num_nr + j*1 ];

					ker_fp
					(
					  conja, conjb, mr_cur, nr_cur, k0,
					  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					  beta, cij, rs_c0, cs_c0, data, cntx
					);

					cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
				}
			}

			cj += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
	}

	#else

	// outer loop = nr; inner loop = mr

	dim_t            m_left = m0;
	double* restrict ci     = c;
	double* restrict ai     = a;

	for ( dim_t i = 0; m_left != 0; ++i )
	{
		const dim_t mr_cur = mrs[ i ];

		if ( mr_cur <= m_left )
		{
			dim_t            n_left = n0;
			double* restrict cij    = ci;
			double* restrict bj     = b;

			for ( dim_t j = 0; n_left != 0; ++j )
			{
				const dim_t nr_cur = nrs[ j ];

				if ( nr_cur <= n_left )
				{
					dgemmsup_ker_ft ker_fp = kmap[ i*num_nr + j*1 ];

					ker_fp
					(
					  conja, conjb, mr_cur, nr_cur, k0,
					  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					  beta, cij, rs_c0, cs_c0, data, cntx
					);

					cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;

				}
			}

			ci += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
	}
	#endif
}

