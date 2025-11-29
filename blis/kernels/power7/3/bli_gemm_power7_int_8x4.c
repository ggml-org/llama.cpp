/*

   (C) Copyright IBM Corporation 2013

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

#ifdef UTEST
#include "blis_utest.h"
#else
#include "blis.h"
#endif
#include <altivec.h>

#define COLMAJ_INDEX(row,col,ld) ((col*ld)+row)
#define ROWMAJ_INDEX(row,col,ld) ((row*ld)+col)
#define BLIS_INDEX(row,col,rs,cs) ((row*rs)+(col*cs))

/*
 * Perform
 *   c = beta * c + alpha * a * b
 * where
 *   alpha & beta are scalars
 *   c is mr x nr in blis-format, (col-stride & row-stride)
 *   a is mr x k in packed col-maj format (leading dim is mr)
 *   b is k x nr in packed row-maj format (leading dim is nr)
 */
void bli_sgemm_power7_int_8x4
     (
              dim_t      m,
              dim_t      n,
              dim_t      k,
        const void*      alpha0,
        const void*      a0,
        const void*      b0,
        const void*      beta0,
              void*      c0, inc_t rs_c, inc_t cs_c,
        const auxinfo_t* data,
        const cntx_t*    cntx
     )
{
#if 1 || defined(UTEST)
    const long MR = BLIS_DEFAULT_MR_S, NR = BLIS_DEFAULT_NR_S;
    const long LDA = MR, LDB = NR;
    long i, j, kk;
    float c00;

    const float* alpha = alpha0;
    const float* a     = a0;
    const float* b     = b0;
    const float* beta  = beta0;
          float* c     = c0;

    for (i=0; i < m; i++) {
        for (j=0; j < n; j++) {
            c00 = c[BLIS_INDEX(i,j,rs_c,cs_c)] * *beta;
            for (kk=0; kk < k; kk++)
                c00 += *alpha * (a[COLMAJ_INDEX(i,kk,LDA)] * b[ROWMAJ_INDEX(kk,j,LDB)]);
            c[BLIS_INDEX(i,j,rs_c,cs_c)] = c00;
        }
    }
#else
    //BLIS_SGEMM_UKERNEL_REF(k, alpha, a, b, beta, c, rs_c, cs_c, data);
#endif
}

/*
 * Perform
 *   c = beta * c + alpha * a * b
 * where
 *   alpha & beta are scalars
 *   c is mr x nr in blis-format, (col-stride & row-stride)
 *   a is mr x k in packed col-maj format (leading dim is mr)
 *   b is k x nr in packed row-maj format (leading dim is nr)
 */
void bli_dgemm_power7_int_8x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha0,
       const void*      a0,
       const void*      b0,
       const void*      beta0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    const double* alpha = alpha0;
    const double* a     = a0;
    const double* b     = b0;
    const double* beta  = beta0;
          double* c     = c0;

    if ( cs_c == 1 )
    {
        // Optimized code for case where C rows are contiguous (i.e. C is row-major)

        vector double vzero = vec_splats( 0.0 );

        vector double vc00_01 = vzero;
        vector double vc02_03 = vzero;
        vector double vc10_11 = vzero;
        vector double vc12_13 = vzero;
        vector double vc20_21 = vzero;
        vector double vc22_23 = vzero;
        vector double vc30_31 = vzero;
        vector double vc32_33 = vzero;
        vector double vc40_41 = vzero;
        vector double vc42_43 = vzero;
        vector double vc50_51 = vzero;
        vector double vc52_53 = vzero;
        vector double vc60_61 = vzero;
        vector double vc62_63 = vzero;
        vector double vc70_71 = vzero;
        vector double vc72_73 = vzero;

        unsigned long long pa = (unsigned long long)a;
        unsigned long long pb = (unsigned long long)b;

#if 0
        unsigned long long d1 = 1*sizeof(double);
        unsigned long long d2 = 2*sizeof(double);
        unsigned long long d3 = 3*sizeof(double);
        unsigned long long d4 = 4*sizeof(double);
        unsigned long long d6 = 6*sizeof(double);
#else
        // ppc64 linux abi: r14-r31   Nonvolatile registers used for local variables
        register unsigned long long d1 __asm ("r21") = 1*sizeof(double);
        register unsigned long long d2 __asm ("r22") = 2*sizeof(double);
        register unsigned long long d3 __asm ("r23") = 3*sizeof(double);
        register unsigned long long d4 __asm ("r24") = 4*sizeof(double);
        register unsigned long long d5 __asm ("r25") = 5*sizeof(double);
        register unsigned long long d6 __asm ("r26") = 6*sizeof(double);
        register unsigned long long d7 __asm ("r27") = 7*sizeof(double);

        __asm__ volatile (";" : "=r" (d1) : "r" (d1) );
        __asm__ volatile (";" : "=r" (d2) : "r" (d2) );
        __asm__ volatile (";" : "=r" (d3) : "r" (d3) );
        __asm__ volatile (";" : "=r" (d4) : "r" (d4) );
        __asm__ volatile (";" : "=r" (d5) : "r" (d5) );
        __asm__ volatile (";" : "=r" (d6) : "r" (d6) );
        __asm__ volatile (";" : "=r" (d7) : "r" (d7) );
#endif

        int kk;
        for (kk=k; kk > 0; kk--) {
            vector double va00 = vec_splats( *(double *)( pa+0 ) );
            vector double va10 = vec_splats( *(double *)( pa+d1 ) );
            vector double va20 = vec_splats( *(double *)( pa+d2 ) );
            vector double va30 = vec_splats( *(double *)( pa+d3 ) );
            vector double va40 = vec_splats( *(double *)( pa+d4 ) );
            vector double va50 = vec_splats( *(double *)( pa+d5 ) );
            vector double va60 = vec_splats( *(double *)( pa+d6 ) );
            vector double va70 = vec_splats( *(double *)( pa+d7 ) );
            pa += 8*sizeof(double);

            vector double vb00_01 = *(vector double *)( pb+0 );
            vector double vb02_03 = *(vector double *)( pb+d2 );
            pb += 4*sizeof(double);

            vc00_01 = vec_madd(va00, vb00_01, vc00_01);
            vc02_03 = vec_madd(va00, vb02_03, vc02_03);
            vc10_11 = vec_madd(va10, vb00_01, vc10_11);
            vc12_13 = vec_madd(va10, vb02_03, vc12_13);
            vc20_21 = vec_madd(va20, vb00_01, vc20_21);
            vc22_23 = vec_madd(va20, vb02_03, vc22_23);
            vc30_31 = vec_madd(va30, vb00_01, vc30_31);
            vc32_33 = vec_madd(va30, vb02_03, vc32_33);
            vc40_41 = vec_madd(va40, vb00_01, vc40_41);
            vc42_43 = vec_madd(va40, vb02_03, vc42_43);
            vc50_51 = vec_madd(va50, vb00_01, vc50_51);
            vc52_53 = vec_madd(va50, vb02_03, vc52_53);
            vc60_61 = vec_madd(va60, vb00_01, vc60_61);
            vc62_63 = vec_madd(va60, vb02_03, vc62_63);
            vc70_71 = vec_madd(va70, vb00_01, vc70_71);
            vc72_73 = vec_madd(va70, vb02_03, vc72_73);
        }

        vector double valpha = vec_splats( *alpha );
        vector double vbeta  = (vector double) { *beta, *beta };

        vector double *pc = (vector double *)c;

        vc00_01 = vec_mul(valpha, vc00_01);
        vc02_03 = vec_mul(valpha, vc02_03);
        pc[0] = vec_madd( pc[0], vbeta, vc00_01);
        pc[1] = vec_madd( pc[1], vbeta, vc02_03);
        pc += rs_c/2;

        vc10_11 = vec_mul(valpha, vc10_11);
        vc12_13 = vec_mul(valpha, vc12_13);
        pc[0] = vec_madd( pc[0], vbeta, vc10_11);
        pc[1] = vec_madd( pc[1], vbeta, vc12_13);
        pc += rs_c/2;

        vc20_21 = vec_mul(valpha, vc20_21);
        vc22_23 = vec_mul(valpha, vc22_23);
        pc[0] = vec_madd( pc[0], vbeta, vc20_21);
        pc[1] = vec_madd( pc[1], vbeta, vc22_23);
        pc += rs_c/2;

        vc30_31 = vec_mul(valpha, vc30_31);
        vc32_33 = vec_mul(valpha, vc32_33);
        pc[0] = vec_madd( pc[0], vbeta, vc30_31);
        pc[1] = vec_madd( pc[1], vbeta, vc32_33);
        pc += rs_c/2;

        vc40_41 = vec_mul(valpha, vc40_41);
        vc42_43 = vec_mul(valpha, vc42_43);
        pc[0] = vec_madd( pc[0], vbeta, vc40_41);
        pc[1] = vec_madd( pc[1], vbeta, vc42_43);
        pc += rs_c/2;

        vc50_51 = vec_mul(valpha, vc50_51);
        vc52_53 = vec_mul(valpha, vc52_53);
        pc[0] = vec_madd( pc[0], vbeta, vc50_51);
        pc[1] = vec_madd( pc[1], vbeta, vc52_53);
        pc += rs_c/2;

        vc60_61 = vec_mul(valpha, vc60_61);
        vc62_63 = vec_mul(valpha, vc62_63);
        pc[0] = vec_madd( pc[0], vbeta, vc60_61);
        pc[1] = vec_madd( pc[1], vbeta, vc62_63);
        pc += rs_c/2;

        vc70_71 = vec_mul(valpha, vc70_71);
        vc72_73 = vec_mul(valpha, vc72_73);
        pc[0] = vec_madd( pc[0], vbeta, vc70_71);
        pc[1] = vec_madd( pc[1], vbeta, vc72_73);
        pc += rs_c/2;
    }
    else
    {
        GEMM_UKR_SETUP_CT( d, 8, 4, false );

        // Optimized code for case where C columns are contiguous (column-major C)
        vector double vzero = vec_splats( 0.0 );

        vector double vc00_10 = vzero;
        vector double vc20_30 = vzero;
        vector double vc40_50 = vzero;
        vector double vc60_70 = vzero;
        vector double vc01_11 = vzero;
        vector double vc21_31 = vzero;
        vector double vc41_51 = vzero;
        vector double vc61_71 = vzero;
        vector double vc02_12 = vzero;
        vector double vc22_32 = vzero;
        vector double vc42_52 = vzero;
        vector double vc62_72 = vzero;
        vector double vc03_13 = vzero;
        vector double vc23_33 = vzero;
        vector double vc43_53 = vzero;
        vector double vc63_73 = vzero;

        unsigned long long pa = (unsigned long long)a;
        unsigned long long pb = (unsigned long long)b;

#if 0
        unsigned long long d1 = 1*sizeof(double);
        unsigned long long d2 = 2*sizeof(double);
        unsigned long long d3 = 3*sizeof(double);
        unsigned long long d4 = 4*sizeof(double);
        unsigned long long d6 = 6*sizeof(double);
#else
        // ppc64 linux abi: r14-r31   Nonvolatile registers used for local variables
        register unsigned long long d1 __asm ("r21") = 1*sizeof(double);
        register unsigned long long d2 __asm ("r22") = 2*sizeof(double);
        register unsigned long long d3 __asm ("r23") = 3*sizeof(double);
        register unsigned long long d4 __asm ("r24") = 4*sizeof(double);
        register unsigned long long d6 __asm ("r26") = 6*sizeof(double);

        __asm__ volatile (";" : "=r" (d1) : "r" (d1) );
        __asm__ volatile (";" : "=r" (d2) : "r" (d2) );
        __asm__ volatile (";" : "=r" (d3) : "r" (d3) );
        __asm__ volatile (";" : "=r" (d4) : "r" (d4) );
        __asm__ volatile (";" : "=r" (d6) : "r" (d6) );
#endif

        int kk;
        for (kk=k; kk > 1; kk-=2) {

            vector double va00_10 = *(vector double *)( pa+0 );
            vector double va20_30 = *(vector double *)( pa+d2 );
            vector double va40_50 = *(vector double *)( pa+d4 );
            vector double va60_70 = *(vector double *)( pa+d6 );
            pa += 8*sizeof(double);

            vector double vb00 = vec_splats( *(double *)( pb+0 ) );
            vector double vb01 = vec_splats( *(double *)( pb+d1 ) );
            vector double vb02 = vec_splats( *(double *)( pb+d2 ) );
            vector double vb03 = vec_splats( *(double *)( pb+d3 ) );
            pb += 4*sizeof(double);

            vc00_10 = vec_madd(va00_10, vb00, vc00_10);
            vc20_30 = vec_madd(va20_30, vb00, vc20_30);
            vc40_50 = vec_madd(va40_50, vb00, vc40_50);
            vc60_70 = vec_madd(va60_70, vb00, vc60_70);
            vc01_11 = vec_madd(va00_10, vb01, vc01_11);
            vc21_31 = vec_madd(va20_30, vb01, vc21_31);
            vc41_51 = vec_madd(va40_50, vb01, vc41_51);
            vc61_71 = vec_madd(va60_70, vb01, vc61_71);
            vc02_12 = vec_madd(va00_10, vb02, vc02_12);
            vc22_32 = vec_madd(va20_30, vb02, vc22_32);
            vc42_52 = vec_madd(va40_50, vb02, vc42_52);
            vc62_72 = vec_madd(va60_70, vb02, vc62_72);
            vc03_13 = vec_madd(va00_10, vb03, vc03_13);
            vc23_33 = vec_madd(va20_30, vb03, vc23_33);
            vc43_53 = vec_madd(va40_50, vb03, vc43_53);
            vc63_73 = vec_madd(va60_70, vb03, vc63_73);

            va00_10 = *(vector double *)( pa+0 );
            va20_30 = *(vector double *)( pa+d2 );
            va40_50 = *(vector double *)( pa+d4 );
            va60_70 = *(vector double *)( pa+d6 );
            pa += 8*sizeof(double);

            vb00 = vec_splats( *(double *)( pb+0 ) );
            vb01 = vec_splats( *(double *)( pb+d1 ) );
            vb02 = vec_splats( *(double *)( pb+d2 ) );
            vb03 = vec_splats( *(double *)( pb+d3 ) );
            pb += 4*sizeof(double);

            vc00_10 = vec_madd(va00_10, vb00, vc00_10);
            vc20_30 = vec_madd(va20_30, vb00, vc20_30);
            vc40_50 = vec_madd(va40_50, vb00, vc40_50);
            vc60_70 = vec_madd(va60_70, vb00, vc60_70);
            vc01_11 = vec_madd(va00_10, vb01, vc01_11);
            vc21_31 = vec_madd(va20_30, vb01, vc21_31);
            vc41_51 = vec_madd(va40_50, vb01, vc41_51);
            vc61_71 = vec_madd(va60_70, vb01, vc61_71);
            vc02_12 = vec_madd(va00_10, vb02, vc02_12);
            vc22_32 = vec_madd(va20_30, vb02, vc22_32);
            vc42_52 = vec_madd(va40_50, vb02, vc42_52);
            vc62_72 = vec_madd(va60_70, vb02, vc62_72);
            vc03_13 = vec_madd(va00_10, vb03, vc03_13);
            vc23_33 = vec_madd(va20_30, vb03, vc23_33);
            vc43_53 = vec_madd(va40_50, vb03, vc43_53);
            vc63_73 = vec_madd(va60_70, vb03, vc63_73);
        }

        for (kk=kk; kk > 0; kk--) {

            vector double va00_10 = *(vector double *)( pa+0 );
            vector double va20_30 = *(vector double *)( pa+d2 );
            vector double va40_50 = *(vector double *)( pa+d4 );
            vector double va60_70 = *(vector double *)( pa+d6 );
            pa += 8*sizeof(double);

            vector double vb00 = vec_splats( *(double *)( pb+0 ) );
            vector double vb01 = vec_splats( *(double *)( pb+d1 ) );
            vector double vb02 = vec_splats( *(double *)( pb+d2 ) );
            vector double vb03 = vec_splats( *(double *)( pb+d3 ) );
            pb += 4*sizeof(double);

            vc00_10 = vec_madd(va00_10, vb00, vc00_10);
            vc20_30 = vec_madd(va20_30, vb00, vc20_30);
            vc40_50 = vec_madd(va40_50, vb00, vc40_50);
            vc60_70 = vec_madd(va60_70, vb00, vc60_70);
            vc01_11 = vec_madd(va00_10, vb01, vc01_11);
            vc21_31 = vec_madd(va20_30, vb01, vc21_31);
            vc41_51 = vec_madd(va40_50, vb01, vc41_51);
            vc61_71 = vec_madd(va60_70, vb01, vc61_71);
            vc02_12 = vec_madd(va00_10, vb02, vc02_12);
            vc22_32 = vec_madd(va20_30, vb02, vc22_32);
            vc42_52 = vec_madd(va40_50, vb02, vc42_52);
            vc62_72 = vec_madd(va60_70, vb02, vc62_72);
            vc03_13 = vec_madd(va00_10, vb03, vc03_13);
            vc23_33 = vec_madd(va20_30, vb03, vc23_33);
            vc43_53 = vec_madd(va40_50, vb03, vc43_53);
            vc63_73 = vec_madd(va60_70, vb03, vc63_73);
        }

        // The following code is dependent on rs_c == 1

        vector double valpha = vec_splats( *alpha );
        vector double vbeta  = (vector double) { *beta, *beta };

        vector double *pc = (vector double *)c;

        vc00_10 = vec_mul(valpha, vc00_10);
        vc20_30 = vec_mul(valpha, vc20_30);
        vc40_50 = vec_mul(valpha, vc40_50);
        vc60_70 = vec_mul(valpha, vc60_70);

        pc[0] = vec_madd( pc[0], vbeta, vc00_10);
        pc[1] = vec_madd( pc[1], vbeta, vc20_30);
        pc[2] = vec_madd( pc[2], vbeta, vc40_50);
        pc[3] = vec_madd( pc[3], vbeta, vc60_70);
        pc += cs_c/2;

        vc01_11 = vec_mul(valpha, vc01_11);
        vc21_31 = vec_mul(valpha, vc21_31);
        vc41_51 = vec_mul(valpha, vc41_51);
        vc61_71 = vec_mul(valpha, vc61_71);

        pc[0] = vec_madd( pc[0], vbeta, vc01_11);
        pc[1] = vec_madd( pc[1], vbeta, vc21_31);
        pc[2] = vec_madd( pc[2], vbeta, vc41_51);
        pc[3] = vec_madd( pc[3], vbeta, vc61_71);
        pc += cs_c/2;

        vc02_12 = vec_mul(valpha, vc02_12);
        vc22_32 = vec_mul(valpha, vc22_32);
        vc42_52 = vec_mul(valpha, vc42_52);
        vc62_72 = vec_mul(valpha, vc62_72);

        pc[0] = vec_madd( pc[0], vbeta, vc02_12);
        pc[1] = vec_madd( pc[1], vbeta, vc22_32);
        pc[2] = vec_madd( pc[2], vbeta, vc42_52);
        pc[3] = vec_madd( pc[3], vbeta, vc62_72);
        pc += cs_c/2;

        vc03_13 = vec_mul(valpha, vc03_13);
        vc23_33 = vec_mul(valpha, vc23_33);
        vc43_53 = vec_mul(valpha, vc43_53);
        vc63_73 = vec_mul(valpha, vc63_73);

        pc[0] = vec_madd( pc[0], vbeta, vc03_13);
        pc[1] = vec_madd( pc[1], vbeta, vc23_33);
        pc[2] = vec_madd( pc[2], vbeta, vc43_53);
        pc[3] = vec_madd( pc[3], vbeta, vc63_73);

        GEMM_UKR_FLUSH_CT( d );
    }
}

/*
 * Perform
 *   c = beta * c + alpha * a * b
 * where
 *   alpha & beta are scalars
 *   c is mr x nr in blis-format, (col-stride & row-stride)
 *   a is mr x k in packed col-maj format (leading dim is mr)
 *   b is k x nr in packed row-maj format (leading dim is nr)
 */
void bli_cgemm_power7_int_8x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha0,
       const void*      a0,
       const void*      b0,
       const void*      beta0,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
#if 1 || defined(UTEST)
    const long MR = BLIS_DEFAULT_MR_C, NR = BLIS_DEFAULT_NR_C;
    const long LDA = MR, LDB = NR;
    int i, j, kk;
    scomplex c00;

    const scomplex* alpha = alpha0;
    const scomplex* a     = a0;
    const scomplex* b     = b0;
    const scomplex* beta  = beta0;
          scomplex* c     = c0;

    for (i=0; i < m; i++) {
        for (j=0; j < n; j++) {
            scomplex tmpc, tmpa, tmpb, tmp;
            //c00 = c[BLIS_INDEX(i,j,rs_c,cs_c)] * *beta;
            tmpc = c[BLIS_INDEX(i,j,rs_c,cs_c)];
            c00.real = tmpc.real * (*beta).real - tmpc.imag * (*beta).imag;
            c00.imag = tmpc.real * (*beta).imag + tmpc.imag * (*beta).real;
            for (kk=0; kk < k; kk++) {
                //c00 += *alpha * (a[COLMAJ_INDEX(i,kk,LDA)] * b[ROWMAJ_INDEX(kk,j,LDB)]);
                tmpa = a[COLMAJ_INDEX(i,kk,LDA)];
                tmpb = b[ROWMAJ_INDEX(kk,j,LDB)];
                tmp.real = tmpa.real * tmpb.real - tmpa.imag * tmpb.imag;
                tmp.imag = tmpa.real * tmpb.imag + tmpa.imag * tmpb.real;
                c00.real += (*alpha).real * tmp.real - (*alpha).imag * tmp.imag;
                c00.imag += (*alpha).real * tmp.imag + (*alpha).imag * tmp.real;
            }
            c[BLIS_INDEX(i,j,rs_c,cs_c)] = c00;
        }
    }
#else
    //BLIS_CGEMM_UKERNEL_REF(k, alpha, a, b, beta, c, rs_c, cs_c, data);
#endif
}

/*
 * Perform
 *   c = beta * c + alpha * a * b
 * where
 *   alpha & beta are scalars
 *   c is mr x nr in blis-format, (col-stride & row-stride)
 *   a is mr x k in packed col-maj format (leading dim is mr)
 *   b is k x nr in packed row-maj format (leading dim is nr)
 */
void bli_zgemm_power7_int_8x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha0,
       const void*      a0,
       const void*      b0,
       const void*      beta0,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
#if 1 || defined(UTEST)
    const long MR = BLIS_DEFAULT_MR_Z, NR = BLIS_DEFAULT_NR_Z;
    const long LDA = MR, LDB = NR;
    int i, j, kk;
    dcomplex c00;

    const dcomplex* alpha = alpha0;
    const dcomplex* a     = a0;
    const dcomplex* b     = b0;
    const dcomplex* beta  = beta0;
          dcomplex* c     = c0;

    for (i=0; i < m; i++) {
        for (j=0; j < n; j++) {
            dcomplex tmpc, tmpa, tmpb, tmp;
            //c00 = c[BLIS_INDEX(i,j,rs_c,cs_c)] * *beta;
            tmpc = c[BLIS_INDEX(i,j,rs_c,cs_c)];
            c00.real = tmpc.real * (*beta).real - tmpc.imag * (*beta).imag;
            c00.imag = tmpc.real * (*beta).imag + tmpc.imag * (*beta).real;
            for (kk=0; kk < k; kk++) {
                //c00 += *alpha * (a[COLMAJ_INDEX(i,kk,LDA)] * b[ROWMAJ_INDEX(kk,j,LDB)]);
                tmpa = a[COLMAJ_INDEX(i,kk,LDA)];
                tmpb = b[ROWMAJ_INDEX(kk,j,LDB)];
                tmp.real = tmpa.real * tmpb.real - tmpa.imag * tmpb.imag;
                tmp.imag = tmpa.real * tmpb.imag + tmpa.imag * tmpb.real;
                c00.real += (*alpha).real * tmp.real - (*alpha).imag * tmp.imag;
                c00.imag += (*alpha).real * tmp.imag + (*alpha).imag * tmp.real;
            }
            c[BLIS_INDEX(i,j,rs_c,cs_c)] = c00;
        }
    }
#else
    //BLIS_ZGEMM_UKERNEL_REF(k, alpha, a, b, beta, c, rs_c, cs_c, data);
#endif
}

