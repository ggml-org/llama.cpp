/*

   BLISPP
   C++ test driver for BLIS CPP rotg routine and reference blis rotg routine.

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
void test_rotg()
{

    T a, b, c, s;
    T a_ref, b_ref, c_ref, s_ref;

   if(is_same<T , float>::value)
   {
      a = 1.0f;
      b = 1.0f;
      a_ref =  1.41421356237f;
      b_ref =  1.41421356237f;
      c_ref =  0.707106781187f;
      s_ref =  0.707106781187f;
   }else{
      a = 1;
      b = 0;
      a_ref = 1;
      b_ref = 0;
      c_ref = 1;
      s_ref = 0;
   }

#ifdef PRINT
        cout<< "Before blis::rotg \na Value : " << a << "\n" ;
        cout<< "b Value : " << b << "\n" ;
#endif
    blis::rotg<T>( 
            &a,
            &b,
            &c,
            &s
            );

#ifdef PRINT
        cout<< "After blis::rotg \na Value : " << a << "\n" ;
        cout<< "b Value : " << b << "\n" ;
        cout<< "c Value : " << c << "\n" ;
        cout<< "s Value : " << s << "\n" ;
#endif

#ifdef PRINT
        cout<< "Expected Output\na Value : " << a_ref << "\n" ;
        cout<< "b Value : " << b_ref << "\n" ;
        cout<< "c Value : " << c_ref << "\n" ;
        cout<< "s Value : " << s_ref << "\n" ;
#endif
     if( (a != a_ref ) || (b != b_ref ) || (c != c_ref ) || (s != s_ref ))
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
     else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);

}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_rotg<float>( );
    test_rotg<double>( );
    return 0;

}
