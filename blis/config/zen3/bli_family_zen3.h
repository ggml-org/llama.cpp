/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLI_FAMILY_ZEN3_
#define BLI_FAMILY_ZEN3_

// By default, it is effective to parallelize the outer loops.
// Setting these macros to 1 will force JR and IR inner loops
// to be not paralleized.
//

#define BLIS_THREAD_MAX_IR      1
#define BLIS_THREAD_MAX_JR      1


// To enable framework optimizations for zen3 platform
// All zen3 specific code should be included in this macro
#define BLIS_CONFIG_ZEN3

// To enable framework optimizations for zen3 platform
// All zen3 specific code should be included in this macro
#define BLIS_CONFIG_ZEN3

//#define BLIS_ENABLE_SMALL_MATRIX
//#define BLIS_ENABLE_SMALL_MATRIX_TRSM


// This will select the threshold below which small matrix code will be called.
#define BLIS_SMALL_MATRIX_THRES        700
#define BLIS_SMALL_M_RECT_MATRIX_THRES 160
#define BLIS_SMALL_K_RECT_MATRIX_THRES 128

#define BLIS_SMALL_MATRIX_THRES_TRSM   32768 //128(128+128) => m*(m+n)
#define BLIS_SMALL_MATRIX_A_THRES_TRSM  128

#define BLIS_SMALL_MATRIX_A_THRES_M_GEMMT 96
#define BLIS_SMALL_MATRIX_A_THRES_N_GEMMT 128

#define BLIS_ENABLE_SMALL_MATRIX_ROME
#define BLIS_SMALL_MATRIX_THRES_ROME       400

#define D_BLIS_SMALL_MATRIX_THRES_TRSM_ALXB_ROME 80
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_ALXB_ROME_ROW_PANEL_M 40
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_ALXB_ROME_COLUMN_PANEL_M 1000
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_ALXB_ROME_COLUMN_PANEL_N 10

#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XAUB_ROME 150
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XAUB_ROME_ROW_PANEL_M 5
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XAUB_ROME_COLUMN_PANEL_N 130

#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME 120
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME_ROW_PANEL_M 10
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME_ROW_PANEL_N 1200
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME_SQUARE_M 30
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME_SQUARE_N 280
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALTB_ROME_COLUMN_PANEL_N 100

#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALB_ROME 110
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XALB_ROME_COL_PANEL_N 30

#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XAUTB_ROME 120
#define D_BLIS_SMALL_MATRIX_THRES_TRSM_XAUTB_ROME_COL_PANEL_N 50
  
#endif
