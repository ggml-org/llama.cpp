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
#define N 2

/*
 * Test application assumes matrices to be column major, non-transposed
 */

template< typename T >
void test_hpr(  )
{
int n;
real_type<T> alpha;
int incX = -1;
	
alpha = 1.0;
n = N;
	
        
T A[4];
  A[0] = { 0.265, 0.362};
  A[1] = {-0.855, 0.035};
  A[2] = {0.136, 0.133 };
  A[3] = { 0.00, 0.00};

T X[2];
  X[0] = { -0.278, -0.686};
  X[1] = {-0.736, -0.918 };

T A_ref[4];
  A_ref[0] = { 1.64942, 0.0};
  A_ref[1] = {-0.020644, 0.284692};
  A_ref[2] = {0.68388, 0.0 };
  A_ref[3] = {0.00, 0.00 };
  

   
#ifdef PRINT
   printmatrix(A, n,n, n,(char *) "A");
   printvector(X, n, (char *)"X");
#endif
	blis::hpr(
	    CblasColMajor,
	    CblasLower,
            n,
	    alpha,
            X,
            incX,
            A
            );

#ifdef PRINT
     printmatrix(A, n , n, n,(char *)"A blis:hpr\n");
     
     printmatrix(A_ref, n, n, n,(char *)"A_ref output\n");
#endif

     if(computeErrorM(n, n, n, n, A, A_ref )==1)
             printf("%s TEST FAIL\n" ,__PRETTY_FUNCTION__);
     else
             printf("%s TEST PASS\n" , __PRETTY_FUNCTION__); 


}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
   test_hpr<complex<float>>( );
   test_hpr<complex<double>>( );
    return 0;

}
