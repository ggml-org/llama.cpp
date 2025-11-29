/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "blis.h"

#ifdef BLIS_ENABLE_BLAS

/* ctpmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,tpmv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_scomplex *ap, bla_scomplex *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer i__1, i__2, i__3, i__4, i__5;
    bla_scomplex q__1, q__2, q__3;

    /* Builtin functions */
    //void bla_r_cnjg(bla_scomplex *, bla_scomplex *);

    /* Local variables */
    bla_integer info;
    bla_scomplex temp;
    bla_integer i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTPMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := conjg( A' )*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX          array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans,
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF77(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(diag,
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF77(xerbla)("CTPMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTEF77(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTEF77(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x:= A*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f) {
			i__2 = j;
			bli_tsets( c,c, (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
			k = kk;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = k;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__5]) - bli_cimag(temp) * bli_cimag(ap[i__5])), (bli_creal(temp) * bli_cimag(ap[i__5]) + bli_cimag(temp) * bli_creal(ap[i__5])), q__2 );
			    bli_tsets( c,c, (bli_creal(x[i__4]) + bli_creal(q__2)), (bli_cimag(x[i__4]) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__3] );
			    ++k;
/* L10: */
			}
			if (nounit) {
			    i__2 = j;
			    i__3 = j;
			    i__4 = kk + j - 1;
			    bli_tsets( c,c, (bli_creal(x[i__3]) * bli_creal(ap[i__4]) - bli_cimag(x[i__3]) * bli_cimag(ap[i__4])), (bli_creal(x[i__3]) * bli_cimag(ap[i__4]) + bli_cimag(x[i__3]) * bli_creal(ap[i__4])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			}
		    }
		    kk += j;
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f) {
			i__2 = jx;
			bli_tsets( c,c, (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
			ix = kx;
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = k;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__5]) - bli_cimag(temp) * bli_cimag(ap[i__5])), (bli_creal(temp) * bli_cimag(ap[i__5]) + bli_cimag(temp) * bli_creal(ap[i__5])), q__2 );
			    bli_tsets( c,c, (bli_creal(x[i__4]) + bli_creal(q__2)), (bli_cimag(x[i__4]) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__3] );
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__2 = jx;
			    i__3 = jx;
			    i__4 = kk + j - 1;
			    bli_tsets( c,c, (bli_creal(x[i__3]) * bli_creal(ap[i__4]) - bli_cimag(x[i__3]) * bli_cimag(ap[i__4])), (bli_creal(x[i__3]) * bli_cimag(ap[i__4]) + bli_cimag(x[i__3]) * bli_creal(ap[i__4])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			}
		    }
		    jx += *incx;
		    kk += j;
/* L40: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (bli_creal(x[i__1]) != 0.f || bli_cimag(x[i__1]) != 0.f) {
			i__1 = j;
			bli_tsets( c,c, (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
			k = kk;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = k;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__4]) - bli_cimag(temp) * bli_cimag(ap[i__4])), (bli_creal(temp) * bli_cimag(ap[i__4]) + bli_cimag(temp) * bli_creal(ap[i__4])), q__2 );
			    bli_tsets( c,c, (bli_creal(x[i__3]) + bli_creal(q__2)), (bli_cimag(x[i__3]) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			    --k;
/* L50: */
			}
			if (nounit) {
			    i__1 = j;
			    i__2 = j;
			    i__3 = kk - *n + j;
			    bli_tsets( c,c, (bli_creal(x[i__2]) * bli_creal(ap[i__3]) - bli_cimag(x[i__2]) * bli_cimag(ap[i__3])), (bli_creal(x[i__2]) * bli_cimag(ap[i__3]) + bli_cimag(x[i__2]) * bli_creal(ap[i__3])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__1] );
			}
		    }
		    kk -= *n - j + 1;
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (bli_creal(x[i__1]) != 0.f || bli_cimag(x[i__1]) != 0.f) {
			i__1 = jx;
			bli_tsets( c,c, (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
			ix = kx;
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = k;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__4]) - bli_cimag(temp) * bli_cimag(ap[i__4])), (bli_creal(temp) * bli_cimag(ap[i__4]) + bli_cimag(temp) * bli_creal(ap[i__4])), q__2 );
			    bli_tsets( c,c, (bli_creal(x[i__3]) + bli_creal(q__2)), (bli_cimag(x[i__3]) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__1 = jx;
			    i__2 = jx;
			    i__3 = kk - *n + j;
			    bli_tsets( c,c, (bli_creal(x[i__2]) * bli_creal(ap[i__3]) - bli_cimag(x[i__2]) * bli_cimag(ap[i__3])), (bli_creal(x[i__2]) * bli_cimag(ap[i__3]) + bli_cimag(x[i__2]) * bli_creal(ap[i__3])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), x[i__1] );
			}
		    }
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    bli_tsets( c,c, (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
		    k = kk - 1;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__1]) - bli_cimag(temp) * bli_cimag(ap[i__1])), (bli_creal(temp) * bli_cimag(ap[i__1]) + bli_cimag(temp) * bli_creal(ap[i__1])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = k;
			    i__2 = i__;
			    bli_tsets( c,c, (bli_creal(ap[i__1]) * bli_creal(x[i__2]) - bli_cimag(ap[i__1]) * bli_cimag(x[i__2])), (bli_creal(ap[i__1]) * bli_cimag(x[i__2]) + bli_cimag(ap[i__1]) * bli_creal(x[i__2])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    --k;
/* L90: */
			}
		    } else {
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk]);
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(q__2) - bli_cimag(temp) * bli_cimag(q__2)), (bli_creal(temp) * bli_cimag(q__2) + bli_cimag(temp) * bli_creal(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__1 = i__;
			    bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__1]) - bli_cimag(q__3) * bli_cimag(x[i__1])), (bli_creal(q__3) * bli_cimag(x[i__1]) + bli_cimag(q__3) * bli_creal(x[i__1])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    --k;
/* L100: */
			}
		    }
		    i__1 = j;
		    bli_tsets( c,c, (bli_creal(temp)), (bli_cimag(temp)), x[i__1] );
		    kk -= j;
/* L110: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    bli_tsets( c,c, (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__1]) - bli_cimag(temp) * bli_cimag(ap[i__1])), (bli_creal(temp) * bli_cimag(ap[i__1]) + bli_cimag(temp) * bli_creal(ap[i__1])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    i__2 = k;
			    i__3 = ix;
			    bli_tsets( c,c, (bli_creal(ap[i__2]) * bli_creal(x[i__3]) - bli_cimag(ap[i__2]) * bli_cimag(x[i__3])), (bli_creal(ap[i__2]) * bli_cimag(x[i__3]) + bli_cimag(ap[i__2]) * bli_creal(x[i__3])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L120: */
			}
		    } else {
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk]);
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(q__2) - bli_cimag(temp) * bli_cimag(q__2)), (bli_creal(temp) * bli_cimag(q__2) + bli_cimag(temp) * bli_creal(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__2 = ix;
			    bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__2]) - bli_cimag(q__3) * bli_cimag(x[i__2])), (bli_creal(q__3) * bli_cimag(x[i__2]) + bli_cimag(q__3) * bli_creal(x[i__2])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L130: */
			}
		    }
		    i__1 = jx;
		    bli_tsets( c,c, (bli_creal(temp)), (bli_cimag(temp)), x[i__1] );
		    jx -= *incx;
		    kk -= j;
/* L140: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    bli_tsets( c,c, (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
		    k = kk + 1;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__2]) - bli_cimag(temp) * bli_cimag(ap[i__2])), (bli_creal(temp) * bli_cimag(ap[i__2]) + bli_cimag(temp) * bli_creal(ap[i__2])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = k;
			    i__4 = i__;
			    bli_tsets( c,c, (bli_creal(ap[i__3]) * bli_creal(x[i__4]) - bli_cimag(ap[i__3]) * bli_cimag(x[i__4])), (bli_creal(ap[i__3]) * bli_cimag(x[i__4]) + bli_cimag(ap[i__3]) * bli_creal(x[i__4])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ++k;
/* L150: */
			}
		    } else {
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk]);
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(q__2) - bli_cimag(temp) * bli_cimag(q__2)), (bli_creal(temp) * bli_cimag(q__2) + bli_cimag(temp) * bli_creal(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__3 = i__;
			    bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__3]) - bli_cimag(q__3) * bli_cimag(x[i__3])), (bli_creal(q__3) * bli_cimag(x[i__3]) + bli_cimag(q__3) * bli_creal(x[i__3])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ++k;
/* L160: */
			}
		    }
		    i__2 = j;
		    bli_tsets( c,c, (bli_creal(temp)), (bli_cimag(temp)), x[i__2] );
		    kk += *n - j + 1;
/* L170: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    bli_tsets( c,c, (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(ap[i__2]) - bli_cimag(temp) * bli_cimag(ap[i__2])), (bli_creal(temp) * bli_cimag(ap[i__2]) + bli_cimag(temp) * bli_creal(ap[i__2])), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    i__3 = k;
			    i__4 = ix;
			    bli_tsets( c,c, (bli_creal(ap[i__3]) * bli_creal(x[i__4]) - bli_cimag(ap[i__3]) * bli_cimag(x[i__4])), (bli_creal(ap[i__3]) * bli_cimag(x[i__4]) + bli_cimag(ap[i__3]) * bli_creal(x[i__4])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L180: */
			}
		    } else {
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk]);
			    bli_tsets( c,c, (bli_creal(temp) * bli_creal(q__2) - bli_cimag(temp) * bli_cimag(q__2)), (bli_creal(temp) * bli_cimag(q__2) + bli_cimag(temp) * bli_creal(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__3 = ix;
			    bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__3]) - bli_cimag(q__3) * bli_cimag(x[i__3])), (bli_creal(q__3) * bli_cimag(x[i__3]) + bli_cimag(q__3) * bli_creal(x[i__3])), q__2 );
			    bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L190: */
			}
		    }
		    i__2 = jx;
		    bli_tsets( c,c, (bli_creal(temp)), (bli_cimag(temp)), x[i__2] );
		    jx += *incx;
		    kk += *n - j + 1;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of CTPMV . */

} /* ctpmv_ */

/* dtpmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,tpmv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_double *ap, bla_double *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer i__1, i__2;

    /* Local variables */
    bla_integer info;
    bla_double temp;
    bla_integer i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTPMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := A'*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - DOUBLE PRECISION array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans,
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF77(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(diag,
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF77(xerbla)("DTPMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTEF77(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x:= A*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.) {
			temp = x[j];
			k = kk;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[i__] += temp * ap[k];
			    ++k;
/* L10: */
			}
			if (nounit) {
			    x[j] *= ap[kk + j - 1];
			}
		    }
		    kk += j;
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    x[ix] += temp * ap[k];
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    x[jx] *= ap[kk + j - 1];
			}
		    }
		    jx += *incx;
		    kk += j;
/* L40: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.) {
			temp = x[j];
			k = kk;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[i__] += temp * ap[k];
			    --k;
/* L50: */
			}
			if (nounit) {
			    x[j] *= ap[kk - *n + j];
			}
		    }
		    kk -= *n - j + 1;
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    x[ix] += temp * ap[k];
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    x[jx] *= ap[kk - *n + j];
			}
		    }
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    if (nounit) {
			temp *= ap[kk];
		    }
		    k = kk - 1;
		    for (i__ = j - 1; i__ >= 1; --i__) {
			temp += ap[k] * x[i__];
			--k;
/* L90: */
		    }
		    x[j] = temp;
		    kk -= j;
/* L100: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= ap[kk];
		    }
		    i__1 = kk - j + 1;
		    for (k = kk - 1; k >= i__1; --k) {
			ix -= *incx;
			temp += ap[k] * x[ix];
/* L110: */
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    kk -= j;
/* L120: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    if (nounit) {
			temp *= ap[kk];
		    }
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			temp += ap[k] * x[i__];
			++k;
/* L130: */
		    }
		    x[j] = temp;
		    kk += *n - j + 1;
/* L140: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= ap[kk];
		    }
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			temp += ap[k] * x[ix];
/* L150: */
		    }
		    x[jx] = temp;
		    jx += *incx;
		    kk += *n - j + 1;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTPMV . */

} /* dtpmv_ */

/* stpmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,tpmv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_real *ap, bla_real *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer i__1, i__2;

    /* Local variables */
    bla_integer info;
    bla_real temp;
    bla_integer i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  STPMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := A'*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - REAL             array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - REAL             array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans,
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF77(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(diag,
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF77(xerbla)("STPMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTEF77(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x:= A*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			k = kk;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[i__] += temp * ap[k];
			    ++k;
/* L10: */
			}
			if (nounit) {
			    x[j] *= ap[kk + j - 1];
			}
		    }
		    kk += j;
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    x[ix] += temp * ap[k];
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    x[jx] *= ap[kk + j - 1];
			}
		    }
		    jx += *incx;
		    kk += j;
/* L40: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			k = kk;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[i__] += temp * ap[k];
			    --k;
/* L50: */
			}
			if (nounit) {
			    x[j] *= ap[kk - *n + j];
			}
		    }
		    kk -= *n - j + 1;
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    x[ix] += temp * ap[k];
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    x[jx] *= ap[kk - *n + j];
			}
		    }
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    if (nounit) {
			temp *= ap[kk];
		    }
		    k = kk - 1;
		    for (i__ = j - 1; i__ >= 1; --i__) {
			temp += ap[k] * x[i__];
			--k;
/* L90: */
		    }
		    x[j] = temp;
		    kk -= j;
/* L100: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= ap[kk];
		    }
		    i__1 = kk - j + 1;
		    for (k = kk - 1; k >= i__1; --k) {
			ix -= *incx;
			temp += ap[k] * x[ix];
/* L110: */
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    kk -= j;
/* L120: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    if (nounit) {
			temp *= ap[kk];
		    }
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			temp += ap[k] * x[i__];
			++k;
/* L130: */
		    }
		    x[j] = temp;
		    kk += *n - j + 1;
/* L140: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= ap[kk];
		    }
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			temp += ap[k] * x[ix];
/* L150: */
		    }
		    x[jx] = temp;
		    jx += *incx;
		    kk += *n - j + 1;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of STPMV . */

} /* stpmv_ */

/* ztpmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,tpmv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_dcomplex *ap, bla_dcomplex *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer i__1, i__2, i__3, i__4, i__5;
    bla_dcomplex z__1, z__2, z__3;

    /* Builtin functions */
    //void bla_d_cnjg(bla_dcomplex *, bla_dcomplex *);

    /* Local variables */
    bla_integer info;
    bla_dcomplex temp;
    bla_integer i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTPMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := conjg( A' )*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX*16       array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans,
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF77(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(diag,
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF77(xerbla)("ZTPMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTEF77(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTEF77(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x:= A*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0.) {
			i__2 = j;
			bli_tsets( z,z, (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
			k = kk;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = k;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__5]) - bli_zimag(temp) * bli_zimag(ap[i__5])), (bli_zreal(temp) * bli_zimag(ap[i__5]) + bli_zimag(temp) * bli_zreal(ap[i__5])), z__2 );
			    bli_tsets( z,z, (bli_zreal(x[i__4]) + bli_zreal(z__2)), (bli_zimag(x[i__4]) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__3] );
			    ++k;
/* L10: */
			}
			if (nounit) {
			    i__2 = j;
			    i__3 = j;
			    i__4 = kk + j - 1;
			    bli_tsets( z,z, (bli_zreal(x[i__3]) * bli_zreal(ap[i__4]) - bli_zimag(x[i__3]) * bli_zimag(ap[i__4])), (bli_zreal(x[i__3]) * bli_zimag(ap[i__4]) + bli_zimag(x[i__3]) * bli_zreal(ap[i__4])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			}
		    }
		    kk += j;
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0.) {
			i__2 = jx;
			bli_tsets( z,z, (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
			ix = kx;
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = k;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__5]) - bli_zimag(temp) * bli_zimag(ap[i__5])), (bli_zreal(temp) * bli_zimag(ap[i__5]) + bli_zimag(temp) * bli_zreal(ap[i__5])), z__2 );
			    bli_tsets( z,z, (bli_zreal(x[i__4]) + bli_zreal(z__2)), (bli_zimag(x[i__4]) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__3] );
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__2 = jx;
			    i__3 = jx;
			    i__4 = kk + j - 1;
			    bli_tsets( z,z, (bli_zreal(x[i__3]) * bli_zreal(ap[i__4]) - bli_zimag(x[i__3]) * bli_zimag(ap[i__4])), (bli_zreal(x[i__3]) * bli_zimag(ap[i__4]) + bli_zimag(x[i__3]) * bli_zreal(ap[i__4])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			}
		    }
		    jx += *incx;
		    kk += j;
/* L40: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (bli_zreal(x[i__1]) != 0. || bli_zimag(x[i__1]) != 0.) {
			i__1 = j;
			bli_tsets( z,z, (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
			k = kk;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = k;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__4]) - bli_zimag(temp) * bli_zimag(ap[i__4])), (bli_zreal(temp) * bli_zimag(ap[i__4]) + bli_zimag(temp) * bli_zreal(ap[i__4])), z__2 );
			    bli_tsets( z,z, (bli_zreal(x[i__3]) + bli_zreal(z__2)), (bli_zimag(x[i__3]) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			    --k;
/* L50: */
			}
			if (nounit) {
			    i__1 = j;
			    i__2 = j;
			    i__3 = kk - *n + j;
			    bli_tsets( z,z, (bli_zreal(x[i__2]) * bli_zreal(ap[i__3]) - bli_zimag(x[i__2]) * bli_zimag(ap[i__3])), (bli_zreal(x[i__2]) * bli_zimag(ap[i__3]) + bli_zimag(x[i__2]) * bli_zreal(ap[i__3])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__1] );
			}
		    }
		    kk -= *n - j + 1;
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (bli_zreal(x[i__1]) != 0. || bli_zimag(x[i__1]) != 0.) {
			i__1 = jx;
			bli_tsets( z,z, (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
			ix = kx;
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = k;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__4]) - bli_zimag(temp) * bli_zimag(ap[i__4])), (bli_zreal(temp) * bli_zimag(ap[i__4]) + bli_zimag(temp) * bli_zreal(ap[i__4])), z__2 );
			    bli_tsets( z,z, (bli_zreal(x[i__3]) + bli_zreal(z__2)), (bli_zimag(x[i__3]) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__1 = jx;
			    i__2 = jx;
			    i__3 = kk - *n + j;
			    bli_tsets( z,z, (bli_zreal(x[i__2]) * bli_zreal(ap[i__3]) - bli_zimag(x[i__2]) * bli_zimag(ap[i__3])), (bli_zreal(x[i__2]) * bli_zimag(ap[i__3]) + bli_zimag(x[i__2]) * bli_zreal(ap[i__3])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__1] );
			}
		    }
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

	if (PASTEF77(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    bli_tsets( z,z, (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
		    k = kk - 1;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__1]) - bli_zimag(temp) * bli_zimag(ap[i__1])), (bli_zreal(temp) * bli_zimag(ap[i__1]) + bli_zimag(temp) * bli_zreal(ap[i__1])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = k;
			    i__2 = i__;
			    bli_tsets( z,z, (bli_zreal(ap[i__1]) * bli_zreal(x[i__2]) - bli_zimag(ap[i__1]) * bli_zimag(x[i__2])), (bli_zreal(ap[i__1]) * bli_zimag(x[i__2]) + bli_zimag(ap[i__1]) * bli_zreal(x[i__2])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    --k;
/* L90: */
			}
		    } else {
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk]);
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(z__2) - bli_zimag(temp) * bli_zimag(z__2)), (bli_zreal(temp) * bli_zimag(z__2) + bli_zimag(temp) * bli_zreal(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__1 = i__;
			    bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__1]) - bli_zimag(z__3) * bli_zimag(x[i__1])), (bli_zreal(z__3) * bli_zimag(x[i__1]) + bli_zimag(z__3) * bli_zreal(x[i__1])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    --k;
/* L100: */
			}
		    }
		    i__1 = j;
		    bli_tsets( z,z, (bli_zreal(temp)), (bli_zimag(temp)), x[i__1] );
		    kk -= j;
/* L110: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    bli_tsets( z,z, (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__1]) - bli_zimag(temp) * bli_zimag(ap[i__1])), (bli_zreal(temp) * bli_zimag(ap[i__1]) + bli_zimag(temp) * bli_zreal(ap[i__1])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    i__2 = k;
			    i__3 = ix;
			    bli_tsets( z,z, (bli_zreal(ap[i__2]) * bli_zreal(x[i__3]) - bli_zimag(ap[i__2]) * bli_zimag(x[i__3])), (bli_zreal(ap[i__2]) * bli_zimag(x[i__3]) + bli_zimag(ap[i__2]) * bli_zreal(x[i__3])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L120: */
			}
		    } else {
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk]);
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(z__2) - bli_zimag(temp) * bli_zimag(z__2)), (bli_zreal(temp) * bli_zimag(z__2) + bli_zimag(temp) * bli_zreal(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__2 = ix;
			    bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__2]) - bli_zimag(z__3) * bli_zimag(x[i__2])), (bli_zreal(z__3) * bli_zimag(x[i__2]) + bli_zimag(z__3) * bli_zreal(x[i__2])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L130: */
			}
		    }
		    i__1 = jx;
		    bli_tsets( z,z, (bli_zreal(temp)), (bli_zimag(temp)), x[i__1] );
		    jx -= *incx;
		    kk -= j;
/* L140: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    bli_tsets( z,z, (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
		    k = kk + 1;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__2]) - bli_zimag(temp) * bli_zimag(ap[i__2])), (bli_zreal(temp) * bli_zimag(ap[i__2]) + bli_zimag(temp) * bli_zreal(ap[i__2])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = k;
			    i__4 = i__;
			    bli_tsets( z,z, (bli_zreal(ap[i__3]) * bli_zreal(x[i__4]) - bli_zimag(ap[i__3]) * bli_zimag(x[i__4])), (bli_zreal(ap[i__3]) * bli_zimag(x[i__4]) + bli_zimag(ap[i__3]) * bli_zreal(x[i__4])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ++k;
/* L150: */
			}
		    } else {
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk]);
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(z__2) - bli_zimag(temp) * bli_zimag(z__2)), (bli_zreal(temp) * bli_zimag(z__2) + bli_zimag(temp) * bli_zreal(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__3 = i__;
			    bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__3]) - bli_zimag(z__3) * bli_zimag(x[i__3])), (bli_zreal(z__3) * bli_zimag(x[i__3]) + bli_zimag(z__3) * bli_zreal(x[i__3])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ++k;
/* L160: */
			}
		    }
		    i__2 = j;
		    bli_tsets( z,z, (bli_zreal(temp)), (bli_zimag(temp)), x[i__2] );
		    kk += *n - j + 1;
/* L170: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    bli_tsets( z,z, (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(ap[i__2]) - bli_zimag(temp) * bli_zimag(ap[i__2])), (bli_zreal(temp) * bli_zimag(ap[i__2]) + bli_zimag(temp) * bli_zreal(ap[i__2])), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    i__3 = k;
			    i__4 = ix;
			    bli_tsets( z,z, (bli_zreal(ap[i__3]) * bli_zreal(x[i__4]) - bli_zimag(ap[i__3]) * bli_zimag(x[i__4])), (bli_zreal(ap[i__3]) * bli_zimag(x[i__4]) + bli_zimag(ap[i__3]) * bli_zreal(x[i__4])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L180: */
			}
		    } else {
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk]);
			    bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(z__2) - bli_zimag(temp) * bli_zimag(z__2)), (bli_zreal(temp) * bli_zimag(z__2) + bli_zimag(temp) * bli_zreal(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__3 = ix;
			    bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__3]) - bli_zimag(z__3) * bli_zimag(x[i__3])), (bli_zreal(z__3) * bli_zimag(x[i__3]) + bli_zimag(z__3) * bli_zreal(x[i__3])), z__2 );
			    bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L190: */
			}
		    }
		    i__2 = jx;
		    bli_tsets( z,z, (bli_zreal(temp)), (bli_zimag(temp)), x[i__2] );
		    jx += *incx;
		    kk += *n - j + 1;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of ZTPMV . */

} /* ztpmv_ */

#endif

