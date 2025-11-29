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
//#define PRINT
#define K 1
#define N 3
/*
 * Test application assumes matrices to be column major, non-transposed
 */

template< typename T >
void test_tbsv(  ) 
{
 int n,k,lda;
 
 k = K;
 n = N;
 lda = n;

 T A[] = { -0.681f, 0.209f, 0.436f, -0.369f, 0.786f, -0.84f, 0.86f, -0.233f, 0.734f };
 T X[] = { -0.305f, 0.61f, -0.831f };
 int incX = -1;
 T X_ref[] = { 0.524539f, -0.961964f, 1.22026f };


#ifdef PRINT
    printmatrix(A, lda ,n,n,(char *)"A");
    printvector(X, n,(char *) "X");
#endif
	blis::tbsv(
	    CblasColMajor,
	    CblasLower,
	    CblasNoTrans,
	    CblasNonUnit,
            n,
	    k,
            A,
            lda,
            X,
            incX
            );

#ifdef PRINT
    printvector(X, n, (char *)"X blis::tbsv\n");
    printvector(X_ref, n,(char *) "X_ref blis::tbsv output");

#endif

  if(computeErrorV(1,1, n, X, X_ref )==1)
	    printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else
	    printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);  


}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_tbsv<double>( );
    test_tbsv<float>( );
    test_tbsv<complex<float>>( );
    test_tbsv<complex<double>>( );
    return 0;

}	
