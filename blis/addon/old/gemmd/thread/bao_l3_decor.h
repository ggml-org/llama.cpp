/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

// Level-3 internal function type.
typedef void (*l3aoint_ft)
     (
       obj_t*     alpha,
       obj_t*     a,
       obj_t*     b,
       obj_t*     d,
       obj_t*     beta,
       obj_t*     c,
       cntx_t*    cntx,
       rntm_t*    rntm,
       thrinfo_t* thread
     );

// Level-3 thread decorator function type.
typedef void (*l3ao_decor_ft)
     (
       l3aoint_ft func,
       opid_t   family,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       obj_t*   d,
       obj_t*   beta,
       obj_t*   c,
       cntx_t*  cntx,
       rntm_t*  rntm
     );

// Level-3 thread decorator prototype.
void bao_l3_thread_decorator
     (
       l3aoint_ft func,
       opid_t   family,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       obj_t*   d,
       obj_t*   beta,
       obj_t*   c,
       cntx_t*  cntx,
       rntm_t*  rntm
     );

void bao_l3_thread_decorator_check
     (
       rntm_t* rntm
     );

// Include definitions specific to the method of multithreading.
#include "bao_l3_decor_single.h"
#include "bao_l3_decor_openmp.h"
#include "bao_l3_decor_pthreads.h"

