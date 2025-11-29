/*

   BLISPP
   C++ test driver for BLIS CPP gemm routine and reference blis gemm routine.

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

#include <complex>
#include <iostream>
#include "blis.hh"
#include "test.hh"

using namespace blis;
using namespace std;
//#define PRINT
#define ALPHA -1.0
#define BETA -1.0
#define M 3
#define N 4

template< typename T >
void test_gbmv(  ) 
{
//    int    i, j, p;
    T alpha, beta;
    int m,n;
   int KL = 1;
   int KU = 1;
   int lda = 4;
   T A[] = { 0.423f, -0.143f, -0.182f, -0.076f, -0.855f, 0.599f, 0.389f, -0.473f, 0.493f, -0.902f, -0.889f, -0.256f, 0.112f, 0.128f, -0.277f, -0.777f };
   T X[] = { 0.488f, 0.029f, -0.633f, 0.84f };
   int incX = -1;
   T Y[] = { 0.874f, 0.322f, -0.477f };
   int incY = -1;
   T Y_ref[] = { -0.656261f, 0.19575f, 0.055905f }; 
    alpha = ALPHA;
    beta = BETA;
    m = M;
    n = N;


#ifdef PRINT
        printmatrix(A, lda ,m,n,(char *) "A");
        printvector(Y, m, (char *)"m");
#endif
	blis::gbmv(
	    CblasColMajor,
	    CblasNoTrans,
            m,
            n,KL,KU,
            alpha,
            A,
            lda,
            X,
	    incX,
	    beta,
            Y,
            incY
            );

#ifdef PRINT
         printvector(Y, m,(char *)"Y blis:gbmv");
         printvector(Y_ref, m, (char *) "Y_ref blis:gbmv" );

#endif

    if(computeErrorV(incY,incY, m, Y, Y_ref )==1)
	    printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__ );
    else
	    printf("%s TEST PASS\n" , __PRETTY_FUNCTION__ );

}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
   test_gbmv<double>( );
   test_gbmv<float>( );
   test_gbmv<complex<float>>( );
   test_gbmv<complex<double>>( );
    return 0;

}
