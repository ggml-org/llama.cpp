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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#undef restrict
#include <complex.h>
#include <assert.h>


/*
 * Here is dgemm kernel for QPX.
 * Instruction mix was divined by a statement in an email from John Gunnels when asked about the peak performance with a single thread:
 * "Achievable peak can either be:
 * 1) 12.8 GF 8 FMAs cycle * 1.6 GHz
 * 2) 8.53 GF Takes intoo account the instruction mix in DGEMM and the fact that you can only do an FMA or a load/store in a single cycle with just one thread
 * 3) 7.58 GF (2) + the fact that we can only issue 8 instructions in 9 cycles with one thread"
 *
 * Which I have taken to mean: 8.53 GFLOPS implies on average 5.33 flops/cycle.
 * I know the kernel John uses is 8x8, so 16 flops per loop iteration.
 * Thus there must be 24 total instructions per iteration because 16/24 = 5.33.
 *
 * Here, we have 6 loads per iteration. These are executed on a different pipeline from FMAs so
 * we could (maybe) theoretically hit 100% of peak with this instruction mix
*/

void bli_dgemm_bgq_int_8x8
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

    GEMM_UKR_SETUP_CT_ANY( d, 8, 8, false );

    //Registers for storing C.
    //4 4x4 subblocks of C, c00, c01, c10, c11
    //4 registers per subblock: a, b, c, d
    //There is an excel file that details which register ends up storing what
    vector4double c00a = vec_splats( 0.0 );
    vector4double c00b = vec_splats( 0.0 );
    vector4double c00c = vec_splats( 0.0 );
    vector4double c00d = vec_splats( 0.0 );

    vector4double c01a = vec_splats( 0.0 );
    vector4double c01b = vec_splats( 0.0 );
    vector4double c01c = vec_splats( 0.0 );
    vector4double c01d = vec_splats( 0.0 );

    vector4double c10a = vec_splats( 0.0 );
    vector4double c10b = vec_splats( 0.0 );
    vector4double c10c = vec_splats( 0.0 );
    vector4double c10d = vec_splats( 0.0 );

    vector4double c11a = vec_splats( 0.0 );
    vector4double c11b = vec_splats( 0.0 );
    vector4double c11c = vec_splats( 0.0 );
    vector4double c11d = vec_splats( 0.0 );

    vector4double b0a, b1a;
    vector4double b0b, b1b;
    vector4double a0, a1;

    for( dim_t i = 0; i < k; i++ )
    {
        b0a = vec_ld2a( 0 * sizeof(double), &b[8*i] );
        b0b = vec_ld2a( 2 * sizeof(double), &b[8*i] );
        b1a = vec_ld2a( 4 * sizeof(double), &b[8*i] );
        b1b = vec_ld2a( 6 * sizeof(double), &b[8*i] );

        a0  = vec_lda ( 0 * sizeof(double), &a[8*i] );
        a1  = vec_lda ( 4 * sizeof(double), &a[8*i] );

        c00a    = vec_xmadd ( b0a, a0, c00a );
        c00b    = vec_xxmadd( a0, b0a, c00b );
        c00c    = vec_xmadd ( b0b, a0, c00c );
        c00d    = vec_xxmadd( a0, b0b, c00d );

        c01a    = vec_xmadd ( b1a, a0, c01a );
        c01b    = vec_xxmadd( a0, b1a, c01b );
        c01c    = vec_xmadd ( b1b, a0, c01c );
        c01d    = vec_xxmadd( a0, b1b, c01d );

        c10a    = vec_xmadd ( b0a, a1, c10a );
        c10b    = vec_xxmadd( a1, b0a, c10b );
        c10c    = vec_xmadd ( b0b, a1, c10c );
        c10d    = vec_xxmadd( a1, b0b, c10d );

        c11a    = vec_xmadd ( b1a, a1, c11a );
        c11b    = vec_xxmadd( a1, b1a, c11b );
        c11c    = vec_xmadd ( b1b, a1, c11c );
        c11d    = vec_xxmadd( a1, b1b, c11d );
    }

    // Create patterns for permuting Cb and Cd
    vector4double pattern = vec_gpci( 01032 );

    vector4double AB;
    vector4double C = vec_splats( 0.0 );
    vector4double betav  = vec_lds( 0, ( double* )beta );
    vector4double alphav = vec_lds( 0, ( double* )alpha );
    double ct;

    //Macro to update 4 elements of C in a column.
    //REG is the register holding those 4 elements
    //ADDR is the address to write them to
    //OFFSET is the number of rows from ADDR to write to
#define UPDATE( REG, ADDR, OFFSET )     \
{                                       \
    ct = *(ADDR + (OFFSET + 0) * rs_c); \
    C = vec_insert( ct, C, 0 );         \
    ct = *(ADDR + (OFFSET + 1) * rs_c); \
    C = vec_insert( ct, C, 1 );         \
    ct = *(ADDR + (OFFSET + 2) * rs_c); \
    C = vec_insert( ct, C, 2 );         \
    ct = *(ADDR + (OFFSET + 3) * rs_c); \
    C = vec_insert( ct, C, 3 );         \
                                        \
    AB = vec_mul( REG, alphav );        \
    AB = vec_madd( C, betav, AB);       \
                                        \
    ct = vec_extract( AB, 0 );          \
    *(ADDR + (OFFSET + 0) * rs_c) = ct; \
    ct = vec_extract( AB, 1 );          \
    *(ADDR + (OFFSET + 1) * rs_c) = ct; \
    ct = vec_extract( AB, 2 );          \
    *(ADDR + (OFFSET + 2) * rs_c) = ct; \
    ct = vec_extract( AB, 3 );          \
    *(ADDR + (OFFSET + 3) * rs_c) = ct; \
}
    //Update c00 and c10 sub-blocks
    UPDATE( c00a, c, 0 );
    UPDATE( c10a, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00b, c00b, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10b, pattern );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    UPDATE( c00c, c, 0 );
    UPDATE( c10c, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00d, c00d, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10d, c10d, pattern );
    UPDATE( AB, c, 4 );

    //Update c01 and c11 sub-blocks
    c = c + cs_c;
    UPDATE( c01a, c, 0 );
    UPDATE( c11a, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01b, c01b, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11b, c11b, pattern );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    UPDATE( c01c, c, 0 );
    UPDATE( c11c, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01d, c01d, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11d, c11d, pattern );
    UPDATE( AB, c, 4 );

    GEMM_UKR_FLUSH_CT( d );
}

void printvec(vector4double v)
{
    double a = vec_extract(v, 0);
    double b = vec_extract(v, 1);
    double c = vec_extract(v, 2);
    double d = vec_extract(v, 3);
    printf("%4.3f\t%4.3f\t%4.3f\t%4.3f\n", a, b, c, d);
}

void bli_zgemm_bgq_int_4x4
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
	const dcomplex* alpha = alpha0;
	const dcomplex* a     = a0;
	const dcomplex* b     = b0;
	const dcomplex* beta  = beta0;
	      dcomplex* c     = c0;

    GEMM_UKR_SETUP_CT_ANY( z, 4, 4, false );

    const double* a_d = ( double* )a;
    const double* b_d = ( double* )b;
          double* c_d = ( double* )c;

    //Registers for storing C.
    //2 2x4 subblocks of C, c0, and c1
    //Each sub-block has 4 columns, 0, 1, 2, 3
    //Each column has 2 partial sum, a and b, and contains 2 complex numbers.
    vector4double c00a = vec_splats( 0.0 );
    vector4double c00b = vec_splats( 0.0 );
    vector4double c01a = vec_splats( 0.0 );
    vector4double c01b = vec_splats( 0.0 );
    vector4double c02a = vec_splats( 0.0 );
    vector4double c02b = vec_splats( 0.0 );
    vector4double c03a = vec_splats( 0.0 );
    vector4double c03b = vec_splats( 0.0 );

    vector4double c10a = vec_splats( 0.0 );
    vector4double c10b = vec_splats( 0.0 );
    vector4double c11a = vec_splats( 0.0 );
    vector4double c11b = vec_splats( 0.0 );
    vector4double c12a = vec_splats( 0.0 );
    vector4double c12b = vec_splats( 0.0 );
    vector4double c13a = vec_splats( 0.0 );
    vector4double c13b = vec_splats( 0.0 );


    vector4double b0, b1, b2, b3;
    vector4double a0, a1;

    for( dim_t i = 0; i < k; i++ )
    {

        b0 = vec_ld2a( 0 * sizeof(double), &b_d[8*i] );
        b1 = vec_ld2a( 2 * sizeof(double), &b_d[8*i] );
        b2 = vec_ld2a( 4 * sizeof(double), &b_d[8*i] );
        b3 = vec_ld2a( 6 * sizeof(double), &b_d[8*i] );

        a0 = vec_lda ( 0 * sizeof(double), &a_d[8*i] );
        a1 = vec_lda ( 4 * sizeof(double), &a_d[8*i] );

        c00a    = vec_xmadd ( b0, a0, c00a );
        c00b    = vec_xxcpnmadd( a0, b0, c00b );
        c01a    = vec_xmadd ( b1, a0, c01a );
        c01b    = vec_xxcpnmadd( a0, b1, c01b );

        c02a    = vec_xmadd ( b2, a0, c02a );
        c02b    = vec_xxcpnmadd( a0, b2, c02b );
        c03a    = vec_xmadd ( b3, a0, c03a );
        c03b    = vec_xxcpnmadd( a0, b3, c03b );


        c10a    = vec_xmadd ( b0, a1, c10a );
        c10b    = vec_xxcpnmadd( a1, b0, c10b );
        c11a    = vec_xmadd ( b1, a1, c11a );
        c11b    = vec_xxcpnmadd( a1, b1, c11b );

        c12a    = vec_xmadd ( b2, a1, c12a );
        c12b    = vec_xxcpnmadd( a1, b2, c12b );
        c13a    = vec_xmadd ( b3, a1, c13a );
        c13b    = vec_xxcpnmadd( a1, b3, c13b );

    }

    // Create patterns for permuting the "b" parts of each vector
    vector4double pattern = vec_gpci( 01032 );
    vector4double zed = vec_splats( 0.0 );

    vector4double AB;
    vector4double C = vec_splats( 0.0 );
    vector4double C1 = vec_splats( 0.0 );
    vector4double C2 = vec_splats( 0.0 );

    double alphar = bli_zreal( *alpha );
    double alphai = bli_zimag( *alpha );
    double betar  = bli_zreal( *beta );
    double betai  = bli_zimag( *beta );
    vector4double alphav = vec_splats( 0.0 );
    vector4double betav = vec_splats( 0.0 );
    alphav = vec_insert( alphar, alphav, 0);
    alphav = vec_insert( alphai, alphav, 1);
    alphav = vec_insert( alphar, alphav, 2);
    alphav = vec_insert( alphai, alphav, 3);
    betav = vec_insert( betar, betav, 0);
    betav = vec_insert( betai, betav, 1);
    betav = vec_insert( betar, betav, 2);
    betav = vec_insert( betai, betav, 3);
    double ct;


    //Macro to update 2 elements of C in a column.
    //REG1 is the register holding the first partial sum of those 2 elements
    //REG2 is the register holding the second partial sum of those 2 elements
    //ADDR is the address to write them to
    //OFFSET is the number of rows from ADDR to write to
#define ZUPDATE( REG1, REG2, ADDR, OFFSET )     \
{                                               \
    ct = *(ADDR + (OFFSET + 0) * rs_c);         \
    C = vec_insert( ct, C, 0 );                 \
    ct = *(ADDR + (OFFSET + 0) * rs_c + 1);     \
    C = vec_insert( ct, C, 1 );                 \
    ct = *(ADDR + (OFFSET + 2) * rs_c);         \
    C = vec_insert( ct, C, 2 );                 \
    ct = *(ADDR + (OFFSET + 2) * rs_c + 1);     \
    C = vec_insert( ct, C, 3 );                 \
                                                \
    AB = vec_sub(REG1, REG2 ); \
                                                \
    /* Scale by alpha */                        \
    REG1 = vec_xmadd( alphav, AB, zed );        \
    REG2 = vec_xxcpnmadd( AB, alphav, zed );     \
    AB = vec_sub(REG1, REG2 ); \
                                                \
                                                \
    /* Scale by beta */                         \
    REG1 = vec_xmadd( betav, C, zed );          \
    REG2 = vec_xxcpnmadd( C, betav, zed );       \
    C = vec_sub(REG1, REG2 ); \
                                                \
    /* Add AB to C */                           \
    C    = vec_add( AB, C );                    \
                                                \
    ct = vec_extract( C, 0 );                  \
    *(ADDR + (OFFSET + 0) * rs_c) = ct;         \
    ct = vec_extract( C, 1 );                  \
    *(ADDR + (OFFSET + 0) * rs_c + 1) = ct;     \
    ct = vec_extract( C, 2 );                  \
    *(ADDR + (OFFSET + 2) * rs_c) = ct;         \
    ct = vec_extract( C, 3 );                  \
    *(ADDR + (OFFSET + 2) * rs_c + 1) = ct;     \
}


    ZUPDATE( c00a, c00b, c_d, 0 );
    ZUPDATE( c10a, c10b, c_d, 4 );
    c_d += 2*cs_c;
    ZUPDATE( c01a, c01b, c_d, 0 );
    ZUPDATE( c11a, c11b, c_d, 4 );
    c_d += 2*cs_c;
    ZUPDATE( c02a, c02b, c_d, 0 );
    ZUPDATE( c12a, c12b, c_d, 4 );
    c_d += 2*cs_c;
    ZUPDATE( c03a, c03b, c_d, 0 );
    ZUPDATE( c13a, c13b, c_d, 4 );

    GEMM_UKR_FLUSH_CT( z );
}
