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
 * test.hh
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


#ifndef TEST_HH
#define TEST_HH

#include <math.h>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define A_ref( i, j )     A_ref[ (j)*lda_ref + (i) ]

#define B( i, j )     B[ (j)*ldb + (i) ]
#define B_ref( i, j ) B_ref[ (j)*ldb_ref + (i) ]

#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]

#define X( i )        X[ incx + (i) ]
#define X_ref( i, j ) X_ref[ (j)*incx_ref + (i) 

#define Y( i )        Y[ incy + (i) ]
#define Y_ref( i )   Y_ref[ incy_ref + (i) ]\

// Allocate memory and initialise memory with random values
void allocate_init_buffer(int *aIn, int m, int n)
{
  aIn =  new int [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((int) rand() / ((int) RAND_MAX / 2.0)) - 1.0;
  }
}

void allocate_init_buffer(float *&aIn, int m, int n)
{
  aIn =  new float [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
  }
}

void allocate_init_buffer(double *&aIn, int m, int n)
{
  aIn =  new double [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
  }
}
void allocate_init_buffer(complex<float> *&aIn, int m, int n)
{
  aIn =  new complex<float> [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     float real = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
     float imag = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
     aIn[i] = {real,imag};
  }
}
void allocate_init_buffer(complex<double> *&aIn, int m, int n)
{
  aIn =  new complex<double> [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     double real = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
     double imag = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
     aIn[i] = {real,imag};
  }
}

template< typename T >
void copy_buffer(T *aSrc, T *&aDest, int m, int n)
{
  aDest =  new T [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aDest[i] = aSrc[i];
  }
}

template< typename T >
int computeErrorM(
        int    lda,
        int    lda_ref,
        int    m,
        int    n,
        T *A,
        T *A_ref
        )
{

    int    i, j;
    int ret = 0;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
	     if ( (fabs (A( i, j )) - fabs( A_ref( i, j ))) > 0.0000001 )  {
                cout << A(i,j) << A_ref(i,j);
                ret = 1;
                break;
            }
        }
    }
    return ret;

}



 template< typename T >
 int computeErrorV(
         int    incy,
         int    incy_ref,
         int    n,
         T *Y,
         T *Y_ref
         )
 {
     int    i;
     int ret = 0;
     for ( i = 0; i < n; i ++ ) {
           if ( (fabs( Y_ref[ i ]) - fabs(Y[ i ] ) ) > 0.00001)  {
		  cout  << Y[i] <<  Y_ref[i];
                 ret = 1;
                 break;
             }
         }

       return ret;

 }

/*
 *printing matix and vector
 *
 */

template <typename T>
void printmatrix(
        T *A,
        int    lda,
        int    m,
        int    n,
        char *func_str
        )
{
    int    i, j;
    cout << func_str <<"\n";
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
		cout<< A[j * lda + i]<<" ";
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
void printvector(
         T *X,
         int    m,
        char *func_str
         )
 {
     int    i;
    cout << func_str <<"\n";
     for ( i = 0; i < m; i ++ ) {
                  cout<< X[i]<<" ";
                  cout<<"\n";
          }
          printf("\n");

 } 


#endif
