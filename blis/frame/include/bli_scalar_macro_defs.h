/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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

#ifndef BLIS_SCALAR_MACRO_DEFS_H
#define BLIS_SCALAR_MACRO_DEFS_H


#include "bli_assigns.h"
#include "bli_complex_terms.h"
#include "bli_constants.h"
#include "bli_declinits.h"

// -- Assignment/Accessor macros --

// NOTE: This macro is defined first since some of the other scalar macros
// use it to abstract away the method used to assign complex values (ie:
// whether fields of a struct are set directly or whether native C99
// assignment is used).

#include "bli_tsets.h"    // sets both real and imaginary components

// NOTE: This macro also needs to be defined early on since it determines
// how real and imaginary components are accessed (ie: whether the fields
// of a struct are read directly or whether native C99 functions are used.)

#include "bli_tgets.h"

// -- Scalar macros --

#include "bli_tabsq2s.h"
#include "bli_tabval2s.h"
#include "bli_tadd3s.h"
#include "bli_tadds.h"
#include "bli_taxpbys.h"
#include "bli_taxpys.h"
#include "bli_tconjs.h"
#include "bli_tcopycjs.h"
#include "bli_tcopynzs.h"
#include "bli_tcopys.h"
#include "bli_tdots.h"
#include "bli_teqs.h"
#include "bli_tfprints.h"
#include "bli_tinverts.h"
#include "bli_tinvscals.h"
#include "bli_tneg2s.h"
#include "bli_trandnp2s.h"
#include "bli_trands.h"
#include "bli_tscalcjs.h"
#include "bli_tscal2s.h"
#include "bli_tscals.h"
#include "bli_tsets.h"
#include "bli_tsqrt2s.h"
#include "bli_tsubs.h"
#include "bli_tswaps.h"
#include "bli_txpbys.h"


#endif
