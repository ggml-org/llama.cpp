/*

   BLISPP
   C++ test driver for BLIS CPP nrm2 routine and reference blis nrm2 routine.

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
#define ALPHA 0.5

#define TOLERANCE          0.0000001
/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T>
void test_nrm2()
{

    T X[N];
    T nrm2, nrm2_ref;
    int n;
    int incx;

    n = N;
    incx = 1;

    if(is_same<T , float>::value)
    {
        X[0] =  0.14f;
        X[1] =  -0.632f;
        nrm2_ref = 0.647320631527f;
    }
    else if(is_same<T , double>::value)
    {
        X[0] =  0.696;
        X[1] =  -0.804;
        nrm2_ref = 1.06340584915;
    }

#ifdef PRINT
    printvector(X, n,(char *) "Vector X after blis::nrm2");
#endif
    nrm2 = blis::nrm2<T>(
            n,
            X,
            incx
            );
#ifdef PRINT
    printf("Norm of a Vector %E  \n", nrm2);
    printf("Ref Norm of a Vector %E  \n", nrm2_ref);
#endif

    if (fabs(nrm2 - nrm2_ref) > TOLERANCE) 
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_nrm2<float>( );
    test_nrm2<double>( );
    return 0;

}
