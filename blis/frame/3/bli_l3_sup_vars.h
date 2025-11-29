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


//
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             trans_t    trans, \
       const obj_t*     alpha, \
       const obj_t*     a, \
       const obj_t*     b, \
       const obj_t*     beta, \
       const obj_t*     c, \
             stor3_t    eff_id, \
       const cntx_t*    cntx, \
       const rntm_t*    rntm, \
             thrinfo_t* thread  \
     );

GENPROT( gemmsup_ref_var1 )
GENPROT( gemmsup_ref_var2 )

GENPROT( gemmsup_ref_var1n )
GENPROT( gemmsup_ref_var2m )


//
// Prototype BLAS-like interfaces with void pointer operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t     conja, \
       conj_t     conjb, \
       dim_t      m, \
       dim_t      n, \
       dim_t      k, \
       void*      alpha, \
       void*      a, inc_t rs_a, inc_t cs_a, \
       void*      b, inc_t rs_b, inc_t cs_b, \
       void*      beta, \
       void*      c, inc_t rs_c, inc_t cs_c, \
       stor3_t    eff_id, \
       cntx_t*    cntx, \
       rntm_t*    rntm, \
       thrinfo_t* thread  \
     );

INSERT_GENTPROT_BASIC( gemmsup_ref_var1 )
INSERT_GENTPROT_BASIC( gemmsup_ref_var2 )

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemmsup_ref_var1n2m_opt_cases
     (
             num_t    dt,
             trans_t* trans,
             bool     packa,
             bool     packb,
             stor3_t* eff_id,
       const cntx_t*  cntx
     )
{
	const bool row_pref = bli_cntx_ukr_prefers_rows_dt( dt, bli_stor3_ukr( *eff_id ), cntx );

	// Handle row- and column-preferrential kernels separately.
	if ( row_pref )
	{
		if      ( packa && packb )
		{
			if      ( *eff_id == BLIS_RRC )
			{
				// Since C is already row-stored, we can use BLIS_RRR kernel instead.
				*eff_id = BLIS_RRR;
			}
			else if ( *eff_id == BLIS_CRC )
			{
				// BLIS_RRC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*eff_id = BLIS_CCC; // BLIS_RRR when transposed below.
			}
			else if ( *eff_id == BLIS_CRR )
			{
				// Induce a transpose to make C row-stored.
				// BLIS_RCC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_CCC; // BLIS_RRR when transposed below.
			}
		}
		else if ( packb )
		{
			if      ( *eff_id == BLIS_RRC )
			{
				// Since C is already row-stored, we can use BLIS_RRR kernel instead.
				*eff_id = BLIS_RRR;
			}
			else if ( *eff_id == BLIS_CRC )
			{
				// BLIS_RRC when transposed below (with packa instead of packb).
				// No transformation is beneficial here.
			}
			else if ( *eff_id == BLIS_RCC )
			{
				// C is already row-stored; cancel transposition and use BLIS_RCR
				// kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_RCR;
			}
			#if 0
			// This transformation performs poorly. Theory: packing A (formerly B)
			// when eff_id == BLIS_RCC (formerly BLIS_CRR) to row storage is slow
			// and kills the performance?
			else if ( eff_id == BLIS_CRR )
			{
				trans = bli_trans_toggled( trans );
				eff_id = BLIS_CRC; // BLIS_RRC when transposed below.
			}
			#endif
		}
		else if ( packa )
		{
			if      ( *eff_id == BLIS_CRR )
			{
				// Induce a transpose to make C row-stored.
				// BLIS_RCC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_CCR; // BLIS_RCR when transposed below.
			}
		}
	}
	else
	{
		//bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
		printf( "libblis: sup var1n2m_opt_cases not yet implemented for column-preferential kernels.\n" );
		bli_abort();
	}
}

