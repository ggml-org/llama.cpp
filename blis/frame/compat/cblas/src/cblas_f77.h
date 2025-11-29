/*
   cblas_f77.h
   Written by Keita Teranishi

   Updated by Jeff Horner
   Merged cblas_f77.h and cblas_fortran_header.h

   (Heavily hacked down from the original)
*/

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

#ifndef CBLAS_F77_H
#define CBLAS_F77_H

/*
 * Level 1 BLAS
 */
#define F77_xerbla     xerbla_
#define F77_srotg      srotg_
#define F77_srotmg     srotmg_
#define F77_srot       srot_
#define F77_srotm      srotm_
#define F77_drotg      drotg_
#define F77_drotmg     drotmg_
#define F77_drot       drot_
#define F77_drotm      drotm_
#define F77_sswap      sswap_
#define F77_scopy      scopy_
#define F77_saxpy      saxpy_
#define F77_isamax_sub isamaxsub_
#define F77_dswap      dswap_
#define F77_dcopy      dcopy_
#define F77_daxpy      daxpy_
#define F77_idamax_sub idamaxsub_
#define F77_cswap      cswap_
#define F77_ccopy      ccopy_
#define F77_caxpy      caxpy_
#define F77_icamax_sub icamaxsub_
#define F77_zswap      zswap_
#define F77_zcopy      zcopy_
#define F77_zaxpy      zaxpy_
#define F77_izamax_sub izamaxsub_
#define F77_sdot_sub   sdotsub_
#define F77_ddot_sub   ddotsub_
#define F77_dsdot_sub  dsdotsub_
#define F77_sscal      sscal_
#define F77_dscal      dscal_
#define F77_cscal      cscal_
#define F77_zscal      zscal_
#define F77_csscal     csscal_
#define F77_zdscal     zdscal_
#define F77_cdotu_sub  cdotusub_
#define F77_cdotc_sub  cdotcsub_
#define F77_zdotu_sub  zdotusub_
#define F77_zdotc_sub  zdotcsub_
#define F77_snrm2_sub  snrm2sub_
#define F77_sasum_sub  sasumsub_
#define F77_dnrm2_sub  dnrm2sub_
#define F77_dasum_sub  dasumsub_
#define F77_scnrm2_sub scnrm2sub_
#define F77_scasum_sub scasumsub_
#define F77_dznrm2_sub dznrm2sub_
#define F77_dzasum_sub dzasumsub_
#define F77_sdsdot_sub sdsdotsub_
/*
* Level 2 BLAS
*/
#define F77_ssymv ssymv_
#define F77_ssbmv ssbmv_
#define F77_sspmv sspmv_
#define F77_sger  sger_
#define F77_ssyr  ssyr_
#define F77_sspr  sspr_
#define F77_ssyr2 ssyr2_
#define F77_sspr2 sspr2_
#define F77_dsymv dsymv_
#define F77_dsbmv dsbmv_
#define F77_dspmv dspmv_
#define F77_dger  dger_
#define F77_dsyr  dsyr_
#define F77_dspr  dspr_
#define F77_dsyr2 dsyr2_
#define F77_dspr2 dspr2_
#define F77_chemv chemv_
#define F77_chbmv chbmv_
#define F77_chpmv chpmv_
#define F77_cgeru cgeru_
#define F77_cgerc cgerc_
#define F77_cher  cher_
#define F77_chpr  chpr_
#define F77_cher2 cher2_
#define F77_chpr2 chpr2_
#define F77_zhemv zhemv_
#define F77_zhbmv zhbmv_
#define F77_zhpmv zhpmv_
#define F77_zgeru zgeru_
#define F77_zgerc zgerc_
#define F77_zher  zher_
#define F77_zhpr  zhpr_
#define F77_zher2 zher2_
#define F77_zhpr2 zhpr2_
#define F77_sgemv sgemv_
#define F77_sgbmv sgbmv_
#define F77_strmv strmv_
#define F77_stbmv stbmv_
#define F77_stpmv stpmv_
#define F77_strsv strsv_
#define F77_stbsv stbsv_
#define F77_stpsv stpsv_
#define F77_dgemv dgemv_
#define F77_dgbmv dgbmv_
#define F77_dtrmv dtrmv_
#define F77_dtbmv dtbmv_
#define F77_dtpmv dtpmv_
#define F77_dtrsv dtrsv_
#define F77_dtbsv dtbsv_
#define F77_dtpsv dtpsv_
#define F77_cgemv cgemv_
#define F77_cgbmv cgbmv_
#define F77_ctrmv ctrmv_
#define F77_ctbmv ctbmv_
#define F77_ctpmv ctpmv_
#define F77_ctrsv ctrsv_
#define F77_ctbsv ctbsv_
#define F77_ctpsv ctpsv_
#define F77_zgemv zgemv_
#define F77_zgbmv zgbmv_
#define F77_ztrmv ztrmv_
#define F77_ztbmv ztbmv_
#define F77_ztpmv ztpmv_
#define F77_ztrsv ztrsv_
#define F77_ztbsv ztbsv_
#define F77_ztpsv ztpsv_
/*
* Level 3 BLAS
*/
#define F77_chemm  chemm_
#define F77_cherk  cherk_
#define F77_cher2k cher2k_
#define F77_zhemm  zhemm_
#define F77_zherk  zherk_
#define F77_zher2k zher2k_
#define F77_sgemm  sgemm_
#define F77_ssymm  ssymm_
#define F77_ssyrk  ssyrk_
#define F77_ssyr2k ssyr2k_
#define F77_strmm  strmm_
#define F77_strsm  strsm_
#define F77_dgemm  dgemm_
#define F77_dsymm  dsymm_
#define F77_dsyrk  dsyrk_
#define F77_dsyr2k dsyr2k_
#define F77_dtrmm  dtrmm_
#define F77_dtrsm  dtrsm_
#define F77_cgemm  cgemm_
#define F77_csymm  csymm_
#define F77_csyrk  csyrk_
#define F77_csyr2k csyr2k_
#define F77_ctrmm  ctrmm_
#define F77_ctrsm  ctrsm_
#define F77_zgemm  zgemm_
#define F77_zsymm  zsymm_
#define F77_zsyrk  zsyrk_
#define F77_zsyr2k zsyr2k_
#define F77_ztrmm  ztrmm_
#define F77_ztrsm  ztrsm_
/*
* BLAS extensions
*/
#define F77_saxpby  saxpby_
#define F77_daxpby  daxpby_
#define F77_caxpby  caxpby_
#define F77_zaxpby  zaxpby_

#define F77_sgemmt  sgemmt_
#define F77_dgemmt  dgemmt_
#define F77_cgemmt  cgemmt_
#define F77_zgemmt  zgemmt_

#define F77_sgemm_batch  sgemm_batch_
#define F77_dgemm_batch  dgemm_batch_
#define F77_cgemm_batch  cgemm_batch_
#define F77_zgemm_batch  zgemm_batch_

#define F77_cgemm3m    cgemm3m_
#define F77_zgemm3m    zgemm3m_


#endif /*  CBLAS_F77_H */
