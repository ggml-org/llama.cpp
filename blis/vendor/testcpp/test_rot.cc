/*

   BLISPP
   C++ test driver for BLIS CPP rot routine and reference blis rot routine.

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
#define N 1

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T>
void test_rot()
{

    T c, s;
    T X[N], X_ref[N];
    T Y[N], Y_ref[N];
    int n;
    int incx, incy;
    
    n = N;
    incx = 1;
    incy = 1;
    if(is_same<T , float>::value){
        c = -1.0f;
        s = 0.0f;
        X[0] = { -0.314f };
        Y[0] = { -0.406f };
        X_ref[0] = { 0.314f };
        Y_ref[0] = { 0.406f }; 
    }else{
        c = -1;
        s = 0;
        X[0] = { -0.176  };
        Y[0] = { -0.165  };
        X_ref[0] = { 0.176 };
        Y_ref[0] = { 0.165 };
    }

#ifdef PRINT
    printvector(X, n, (char *)"Before blis::rot\nVector X");
    printvector(Y, n, (char *)"Vector Y");
#endif
    blis::rot<T>( N, X, incx, Y, incy, c, s);
#ifdef PRINT
    printvector(X, n, (char *)"After blis::rot\nVector X");
    printvector(Y, n, (char *) "Vector Y");
    printvector(X, n, (char *) "Expected Output from blis::rot\nVector X");
    printvector(Y, n, (char *)"Vector Y");
#endif

    if((computeErrorV(incx, incx , n, X, X_ref )==1) || (computeErrorV(incy, incy , n, Y, Y_ref )==1))
         printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else
         printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);

}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_rot<float>( );
    test_rot<double>( );
    return 0;

}
