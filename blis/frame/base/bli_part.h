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

#include "bli_part_check.h"

// -- Matrix partitioning ------------------------------------------------------

BLIS_EXPORT_BLIS void bli_acquire_mpart
     (
             dim_t  i,
             dim_t  j,
             dim_t  m,
             dim_t  n,
       const obj_t* obj,
             obj_t* sub_obj
     );

#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC( opname ) \
     ( \
             subpart_t req_part, \
             dim_t     i, \
             dim_t     b, \
       const obj_t*    obj, \
             obj_t*    sub_obj \
     );

GENPROT( acquire_mpart_t2b )
GENPROT( acquire_mpart_b2t )
GENPROT( acquire_mpart_l2r )
GENPROT( acquire_mpart_r2l )
GENPROT( acquire_mpart_tl2br )
GENPROT( acquire_mpart_br2tl )


#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC( opname ) \
     ( \
             dir_t     direct, \
             subpart_t req_part, \
             dim_t     i, \
             dim_t     b, \
       const obj_t*    obj, \
             obj_t*    sub_obj \
     );

GENPROT( acquire_mpart_mdim )
GENPROT( acquire_mpart_ndim )
GENPROT( acquire_mpart_mndim )


// -- Vector partitioning ------------------------------------------------------

#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC( opname ) \
     ( \
             subpart_t req_part, \
             dim_t     i, \
             dim_t     b, \
       const obj_t*    obj, \
             obj_t*    sub_obj \
     );

GENPROT( acquire_vpart_f2b )
GENPROT( acquire_vpart_b2f )

// -- Scalar acquisition -------------------------------------------------------

BLIS_EXPORT_BLIS void bli_acquire_mij
     (
             dim_t  i,
             dim_t  j,
       const obj_t* obj,
             obj_t* sub_obj
     );

BLIS_EXPORT_BLIS void bli_acquire_vi
     (
             dim_t  i,
       const obj_t* obj,
             obj_t* sub_obj
     );

