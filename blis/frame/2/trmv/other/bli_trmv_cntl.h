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

struct trmv_s
{
	impl_t             impl_type;
	varnum_t           var_num;
	bszid_t            bszid;
	struct packm_s*    sub_packm_a11;
	struct packv_s*    sub_packv_x1;
	struct gemv_s*     sub_gemv_rp;
	struct gemv_s*     sub_gemv_cp;
	struct trmv_s*     sub_trmv;
	struct unpackv_s*  sub_unpackv_x1;
};
typedef struct trmv_s trmv_t;

#define bli_cntl_sub_trmv( cntl )      cntl->sub_trmv

void     bli_trmv_cntl_init( void );
void     bli_trmv_cntl_finalize( void );
trmv_t* bli_trmv_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  bszid_t    bszid,
                                  packm_t*   sub_packm_a11,
                                  packv_t*   sub_packv_x1,
                                  gemv_t*    sub_gemv_rp,
                                  gemv_t*    sub_gemv_cp,
                                  trmv_t*    sub_trmv,
                                  unpackv_t* sub_unpackv_x1 );
void bli_trmv_cntl_obj_init( trmv_t*    cntl,
                             impl_t     impl_type,
                             varnum_t   var_num,
                             bszid_t    bszid,
                             packm_t*   sub_packm_a11,
                             packv_t*   sub_packv_x1,
                             gemv_t*    sub_gemv_rp,
                             gemv_t*    sub_gemv_cp,
                             trmv_t*    sub_trmv,
                             unpackv_t* sub_unpackv_x1 );

