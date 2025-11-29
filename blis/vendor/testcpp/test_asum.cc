/*

   BLISPP
   C++ test driver for BLIS CPP asum routine and reference blis asum routine.

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
#define N 6
#define ALPHA 0.5

template< typename T, typename TR>
void ref_asum(int64_t n,
              T *X,
              TR *asum
              )
{
    obj_t obj_x;
    obj_t obj_asum;
    num_t dt, dtR;

    if(is_same<T , float>::value)
        dt = BLIS_FLOAT;
    else if(is_same<T , double>::value)
        dt = BLIS_DOUBLE;
    else if(is_same<T , complex<float>>::value)
        dt = BLIS_SCOMPLEX;
    else if(is_same<T , complex<double>>::value)
        dt = BLIS_DCOMPLEX;

    if(is_same<TR , float>::value)
        dtR = BLIS_FLOAT;
    else if(is_same<TR , double>::value)
        dtR = BLIS_DOUBLE;

    bli_obj_create_with_attached_buffer( dt, n, 1, X, 1, n,&obj_x );
    bli_obj_create_with_attached_buffer( dtR, 1, 1, asum, 1, 1,&obj_asum );

    bli_asumv(&obj_x, &obj_asum);

}
template< typename T, typename TR>
void test_asum()
{

    T *X, *X_ref;
    TR asum, asum_ref;
    int n;
    int incx;

    n = N;
    incx = 1;
    srand (time(NULL));
    allocate_init_buffer(X , n , 1);
    copy_buffer(X, X_ref , n ,1);

#ifdef PRINT
    printvector(X, n,(char *) "X");
#endif

    asum = blis::asum<T>(
            n,
            X,
            incx
            );

#ifdef PRINT
    cout<< "Sum of all values in Vector X: " << asum << "\n";
#endif

    ref_asum(n, X_ref, &asum_ref ); 

#ifdef PRINT
    cout<< "Ref Sum of all values in Vector X: " << asum_ref << "\n";
#endif
    if(computeErrorV(incx, incx, 1, &asum, &asum_ref )==1)
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);

    delete[]( X     );
    delete[]( X_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_asum<float, float>( );
    test_asum<double, double>( );
    test_asum<std::complex<float>, float>( );
    test_asum<std::complex<double>, double>( );
    return 0;

}
