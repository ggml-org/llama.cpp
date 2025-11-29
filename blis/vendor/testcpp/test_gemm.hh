/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * test_gemm.hh
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#ifndef TEST_GEMM_HH
#define TEST_GEMM_HH

#include <math.h>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define B( i, j )     B[ (j)*ldb + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]

template< typename T >
int computeError(
        int    ldc,
        int    ldc_ref,
        int    m,
        int    n,
        T *C,
        T *C_ref
        )
{
    int    i, j;
    int ret = 0;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if (  C( i, j ) != C_ref( i, j ) ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                ret = 1;
                break;
            }
        }
    }
    return ret;

}

/*
 *
 *
 */
template <typename T>
void bl_dgemm_printmatrix(
        T *A,
        int    lda,
        int    m,
        int    n
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
		cout<< A[j * lda + i]<<" ";
        }
        printf("\n");
    }
    printf("\n");
}

#endif
