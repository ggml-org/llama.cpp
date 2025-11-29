/*

   BLISPP
   C++ test driver for BLIS CPP rotmg routine and reference blis rotmg routine.

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

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T>
void test_rotmg()
{
    T d1, d2, b1, b2;
    T d1_ref, d2_ref, b1_ref;
    T h[5] = { -999.0f, -999.1f, -999.2f, -999.3f, -999.4f };
    T h_ref[5] = {-1.0f, 0.0f, 0.0f, 0.0f,0.0f};
    T h_double[5] = { -999.0, -999.1, -999.2, -999.3, -999.4 };
    T h_ref_double[5] = { 1, 0, 0, 0};

   if(is_same<T , float>::value)
   {
        d1 = -1630.28519312f;
        d2 = 44320.1964703f;
        b1 = 1274.7681352f;
        b2 = 0.983006912864f;
        d1_ref= 0.0f;
        d2_ref= 0.0f;
        b1_ref= 0.0f;
   }else{
        d1 = -49.1978123005;
        d2 = 0.228703451277;
        b1 = 1.8901039144;
        b2 = 7081.47754386;
        d1_ref= 0;
        d2_ref= 0;
        b1_ref= 0;
   }

#ifdef PRINT
    cout<< "Before blis::rotmg \nd1 Value : " << d1 << "\n" ;
    cout<< "d2 Value : " << d2 << "\n" ;
    cout<< "b1 Value : " << b1 << "\n" ;
    printvector(h, 5,(char *) "param");
#endif
    if(is_same<T , float>::value)
    {
        blis::rotmg<T>( 
                &d1,
                &d2,
                &b1,
                b2,
                h
                );
    }else{
        blis::rotmg<T>( 
         &d1,
         &d2,
         &b1,
         b2,
         h_double
         );
    }

#ifdef PRINT
    cout<< "After blis::rotmg \nd1 Value : " << d1 << "\n" ;
    cout<< "d2 Value : " << d2 << "\n" ;
    cout<< "b1 Value : " << b1 << "\n" ;
    printvector(h, 5,(char *) "param");
#endif

#ifdef PRINT
    cout<< "Expected Output from blis::rotmg \nd1 Value : " << d1_ref << "\n" ;
    cout<< "d2 Value : " << d2_ref << "\n" ;
    cout<< "b1 Value : " << b1_ref << "\n" ;
    printvector(h_ref, 5,(char *) "param");
#endif
    if( (d1 != d1_ref ) || (d2 != d2_ref ) || (b1 != b1_ref ) )
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else if(is_same<T , float>::value){
        if(computeErrorV(1, 1 , 5, h, h_ref )==1) 
             printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
        else
             printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);
    }else if(is_same<T , float>::value){
        if(computeErrorV(1, 1 , 5, h_double, h_ref_double )==1)
            printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
        else
            printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);
    }else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);

}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_rotmg<float>( );
    test_rotmg<double>( );
    return 0;

}
