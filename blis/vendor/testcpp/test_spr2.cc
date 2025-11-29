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
#define ALPHA -1.0f
#define N 2

/*
 * Test application assumes matrices to be column major, non-transposed
 */

template< typename T >
void test_spr2(  )
{
   int n;	
   int incX = -1;
   int incY = -1;
   T alpha;
    
   alpha = ALPHA;
   n = N;

   T A[] = { 0.493f, -0.175f, -0.831f };
   T X[] = { -0.163f, 0.489f };
   T Y[] = { 0.154f, 0.769f };
   T A_ref[]= { -0.259082f, -0.124959f, -0.780796f };
     
    
    
#ifdef PRINT
   printf("Matrix A\n");
   printmatrix(A, incX, n,n,(char *)"A");
   printf("Vector X \n");
   printvector(X, n, (char *)"X");
#endif
	blis::spr2(
	    CblasColMajor,
	    CblasLower,
            n,
	    alpha,
            X,
            incX,
            Y,
	    incY,
            A
            );

#ifdef PRINT
     printf("Matrix A after blis:spr2\n");
     printmatrix (A,1 ,n, n,(char *)"A");
     printf("A_ref \n");
     printmatrix(A_ref, 1, n,n,(char *)"A_ref output");
#endif

     if(computeErrorM(1, 1, n, n, A, A_ref )==1)
             printf("%s TEST FAIL\n" ,__PRETTY_FUNCTION__);
     else
             printf("%s TEST PASS\n" , __PRETTY_FUNCTION__); 


}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
	test_spr2<double>( );
	test_spr2<float>( );
    	return 0;

}
