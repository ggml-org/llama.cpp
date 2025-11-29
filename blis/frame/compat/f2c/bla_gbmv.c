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

/* cgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,gbmv)(const bla_character *trans, const bla_integer *m, const bla_integer *n, const bla_integer *kl, const bla_integer *ku, const bla_scomplex *alpha, const bla_scomplex *a, const bla_integer *lda, const bla_scomplex *x, const bla_integer *incx, const bla_scomplex *beta, bla_scomplex *y, const bla_integer *incy)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    bla_scomplex q__1, q__2, q__3;

    /* Builtin functions */
    //void bla_r_cnjg(bla_scomplex *, bla_scomplex *);

    /* Local variables */
    bla_integer info;
    bla_scomplex temp;
    bla_integer lenx, leny, i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer ix, iy, jx, jy, kx, ky;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj;
    bla_integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or */

/*     y := alpha*conjg( A' )*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX         . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */


/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	PASTEF77(xerbla)("CGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (bli_creal(*alpha) == 0.f && bli_cimag(*alpha) == 0.f && (bli_creal(*beta)
	    == 1.f && bli_cimag(*beta) == 0.f))) {
	return 0;
    }

    noconj = PASTEF77(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (bli_creal(*beta) != 1.f || bli_cimag(*beta) != 0.f) {
	if (*incy == 1) {
	    if (bli_creal(*beta) == 0.f && bli_cimag(*beta) == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    bli_tsets( c,c, (0.f), (0.f), y[i__2] );
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    bli_tsets( c,c, (bli_creal(*beta) * bli_creal(y[i__3]) - bli_cimag(*beta) * bli_cimag(y[i__3])), (bli_creal(*beta) * bli_cimag(y[i__3]) + bli_cimag(*beta) * bli_creal(y[i__3])), q__1 );
		    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__2] );
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (bli_creal(*beta) == 0.f && bli_cimag(*beta) == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    bli_tsets( c,c, (0.f), (0.f), y[i__2] );
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    bli_tsets( c,c, (bli_creal(*beta) * bli_creal(y[i__3]) - bli_cimag(*beta) * bli_cimag(y[i__3])), (bli_creal(*beta) * bli_cimag(y[i__3]) + bli_cimag(*beta) * bli_creal(y[i__3])), q__1 );
		    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__2] );
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (bli_creal(*alpha) == 0.f && bli_cimag(*alpha) == 0.f) {
	return 0;
    }
    kup1 = *ku + 1;
    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f) {
		    i__2 = jx;
		    bli_tsets( c,c, (bli_creal(*alpha) * bli_creal(x[i__2]) - bli_cimag(*alpha) * bli_cimag(x[i__2])), (bli_creal(*alpha) * bli_cimag(x[i__2]) + bli_cimag(*alpha) * bli_creal(x[i__2])), q__1 );
		    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			i__2 = i__;
			i__3 = i__;
			i__5 = k + i__ + j * a_dim1;
			bli_tsets( c,c, (bli_creal(temp) * bli_creal(a[i__5]) - bli_cimag(temp) * bli_cimag(a[i__5])), (bli_creal(temp) * bli_cimag(a[i__5]) + bli_cimag(temp) * bli_creal(a[i__5])), q__2 );
			bli_tsets( c,c, (bli_creal(y[i__3]) + bli_creal(q__2)), (bli_cimag(y[i__3]) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__2] );
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		if (bli_creal(x[i__4]) != 0.f || bli_cimag(x[i__4]) != 0.f) {
		    i__4 = jx;
		    bli_tsets( c,c, (bli_creal(*alpha) * bli_creal(x[i__4]) - bli_cimag(*alpha) * bli_cimag(x[i__4])), (bli_creal(*alpha) * bli_cimag(x[i__4]) + bli_cimag(*alpha) * bli_creal(x[i__4])), q__1 );
		    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = iy;
			i__2 = iy;
			i__5 = k + i__ + j * a_dim1;
			bli_tsets( c,c, (bli_creal(temp) * bli_creal(a[i__5]) - bli_cimag(temp) * bli_cimag(a[i__5])), (bli_creal(temp) * bli_cimag(a[i__5]) + bli_cimag(temp) * bli_creal(a[i__5])), q__2 );
			bli_tsets( c,c, (bli_creal(y[i__2]) + bli_creal(q__2)), (bli_cimag(y[i__2]) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__4] );
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		bli_tsets( c,c, (0.f), (0.f), temp );
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			i__3 = k + i__ + j * a_dim1;
			i__4 = i__;
			bli_tsets( c,c, (bli_creal(a[i__3]) * bli_creal(x[i__4]) - bli_cimag(a[i__3]) * bli_cimag(x[i__4])), (bli_creal(a[i__3]) * bli_cimag(x[i__4]) + bli_cimag(a[i__3]) * bli_creal(x[i__4])), q__2 );
			bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L90: */
		    }
		} else {
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			bla_r_cnjg(&q__3, &a[k + i__ + j * a_dim1]);
			i__2 = i__;
			bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__2]) - bli_cimag(q__3) * bli_cimag(x[i__2])), (bli_creal(q__3) * bli_cimag(x[i__2]) + bli_cimag(q__3) * bli_creal(x[i__2])), q__2 );
			bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L100: */
		    }
		}
		i__4 = jy;
		i__2 = jy;
		bli_tsets( c,c, (bli_creal(*alpha) * bli_creal(temp) - bli_cimag(*alpha) * bli_cimag(temp)), (bli_creal(*alpha) * bli_cimag(temp) + bli_cimag(*alpha) * bli_creal(temp)), q__2 );
		bli_tsets( c,c, (bli_creal(y[i__2]) + bli_creal(q__2)), (bli_cimag(y[i__2]) + bli_cimag(q__2)), q__1 );
		bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__4] );
		jy += *incy;
/* L110: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		bli_tsets( c,c, (0.f), (0.f), temp );
		ix = kx;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = k + i__ + j * a_dim1;
			i__2 = ix;
			bli_tsets( c,c, (bli_creal(a[i__4]) * bli_creal(x[i__2]) - bli_cimag(a[i__4]) * bli_cimag(x[i__2])), (bli_creal(a[i__4]) * bli_cimag(x[i__2]) + bli_cimag(a[i__4]) * bli_creal(x[i__2])), q__2 );
			bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			ix += *incx;
/* L120: */
		    }
		} else {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			bla_r_cnjg(&q__3, &a[k + i__ + j * a_dim1]);
			i__3 = ix;
			bli_tsets( c,c, (bli_creal(q__3) * bli_creal(x[i__3]) - bli_cimag(q__3) * bli_cimag(x[i__3])), (bli_creal(q__3) * bli_cimag(x[i__3]) + bli_cimag(q__3) * bli_creal(x[i__3])), q__2 );
			bli_tsets( c,c, (bli_creal(temp) + bli_creal(q__2)), (bli_cimag(temp) + bli_cimag(q__2)), q__1 );
			bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			ix += *incx;
/* L130: */
		    }
		}
		i__2 = jy;
		i__3 = jy;
		bli_tsets( c,c, (bli_creal(*alpha) * bli_creal(temp) - bli_cimag(*alpha) * bli_cimag(temp)), (bli_creal(*alpha) * bli_cimag(temp) + bli_cimag(*alpha) * bli_creal(temp)), q__2 );
		bli_tsets( c,c, (bli_creal(y[i__3]) + bli_creal(q__2)), (bli_cimag(y[i__3]) + bli_cimag(q__2)), q__1 );
		bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), y[i__2] );
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L140: */
	    }
	}
    }

    return 0;

/*     End of CGBMV . */

} /* cgbmv_ */

/* dgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,gbmv)(const bla_character *trans, const bla_integer *m, const bla_integer *n, const bla_integer *kl, const bla_integer *ku, const bla_double *alpha, const bla_double *a, const bla_integer *lda, const bla_double *x, const bla_integer *incx, const bla_double *beta, bla_double *y, const bla_integer *incy)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    bla_integer info;
    bla_double temp;
    bla_integer lenx, leny, i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer ix, iy, jx, jy, kx, ky;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	PASTEF77(xerbla)("DGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. && *beta == 1.)) {
	return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.) {
	return 0;
    }
    kup1 = *ku + 1;
    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			y[i__] += temp * a[k + i__ + j * a_dim1];
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			y[iy] += temp * a[k + i__ + j * a_dim1];
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		k = kup1 - j;
/* Computing MAX */
		i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__2 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L100: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		ix = kx;
		k = kup1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__4 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[ix];
		    ix += *incx;
/* L110: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L120: */
	    }
	}
    }

    return 0;

/*     End of DGBMV . */

} /* dgbmv_ */

/* sgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,gbmv)(const bla_character *trans, const bla_integer *m, const bla_integer *n, const bla_integer *kl, const bla_integer *ku, const bla_real *alpha, const bla_real *a, const bla_integer *lda, const bla_real *x, const bla_integer * incx, const bla_real *beta, bla_real *y, const bla_integer *incy)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    bla_integer info;
    bla_real temp;
    bla_integer lenx, leny, i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer ix, iy, jx, jy, kx, ky;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - REAL             array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - REAL            . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - REAL             array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	PASTEF77(xerbla)("SGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0.f && *beta == 1.f)) {
	return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (*beta != 1.f) {
	if (*incy == 1) {
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.f;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.f;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.f) {
	return 0;
    }
    kup1 = *ku + 1;
    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.f) {
		    temp = *alpha * x[jx];
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			y[i__] += temp * a[k + i__ + j * a_dim1];
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.f) {
		    temp = *alpha * x[jx];
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			y[iy] += temp * a[k + i__ + j * a_dim1];
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		k = kup1 - j;
/* Computing MAX */
		i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__2 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L100: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		ix = kx;
		k = kup1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__4 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[ix];
		    ix += *incx;
/* L110: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L120: */
	    }
	}
    }

    return 0;

/*     End of SGBMV . */

} /* sgbmv_ */

/* zgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,gbmv)(const bla_character *trans, const bla_integer *m, const bla_integer *n, const bla_integer *kl, const bla_integer *ku, const bla_dcomplex *alpha, const bla_dcomplex *a, const bla_integer *lda, const bla_dcomplex *x, const bla_integer *incx, const bla_dcomplex *beta, bla_dcomplex * y, const bla_integer *incy)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    bla_dcomplex z__1, z__2, z__3;

    /* Builtin functions */
    //void bla_d_cnjg(bla_dcomplex *, bla_dcomplex *);

    /* Local variables */
    bla_integer info;
    bla_dcomplex temp;
    bla_integer lenx, leny, i__, j, k;
    //extern bla_logical PASTEF77(lsame)(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer ix, iy, jx, jy, kx, ky;
    //extern /* Subroutine */ int PASTEF77(xerbla)(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj;
    bla_integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or */

/*     y := alpha*conjg( A' )*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX*16      . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX*16      . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */


/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! PASTEF77(lsame)(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	PASTEF77(xerbla)("ZGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (bli_zreal(*alpha) == 0. && bli_zimag(*alpha) == 0. && (bli_zreal(*beta) ==
	    1. && bli_zimag(*beta) == 0.))) {
	return 0;
    }

    noconj = PASTEF77(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (bli_zreal(*beta) != 1. || bli_zimag(*beta) != 0.) {
	if (*incy == 1) {
	    if (bli_zreal(*beta) == 0. && bli_zimag(*beta) == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    bli_tsets( z,z, (0.), (0.), y[i__2] );
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    bli_tsets( z,z, (bli_zreal(*beta) * bli_zreal(y[i__3]) - bli_zimag(*beta) * bli_zimag(y[i__3])), (bli_zreal(*beta) * bli_zimag(y[i__3]) + bli_zimag(*beta) * bli_zreal(y[i__3])), z__1 );
		    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__2] );
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (bli_zreal(*beta) == 0. && bli_zimag(*beta) == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    bli_tsets( z,z, (0.), (0.), y[i__2] );
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    bli_tsets( z,z, (bli_zreal(*beta) * bli_zreal(y[i__3]) - bli_zimag(*beta) * bli_zimag(y[i__3])), (bli_zreal(*beta) * bli_zimag(y[i__3]) + bli_zimag(*beta) * bli_zreal(y[i__3])), z__1 );
		    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__2] );
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (bli_zreal(*alpha) == 0. && bli_zimag(*alpha) == 0.) {
	return 0;
    }
    kup1 = *ku + 1;
    if (PASTEF77(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0.) {
		    i__2 = jx;
		    bli_tsets( z,z, (bli_zreal(*alpha) * bli_zreal(x[i__2]) - bli_zimag(*alpha) * bli_zimag(x[i__2])), (bli_zreal(*alpha) * bli_zimag(x[i__2]) + bli_zimag(*alpha) * bli_zreal(x[i__2])), z__1 );
		    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			i__2 = i__;
			i__3 = i__;
			i__5 = k + i__ + j * a_dim1;
			bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(a[i__5]) - bli_zimag(temp) * bli_zimag(a[i__5])), (bli_zreal(temp) * bli_zimag(a[i__5]) + bli_zimag(temp) * bli_zreal(a[i__5])), z__2 );
			bli_tsets( z,z, (bli_zreal(y[i__3]) + bli_zreal(z__2)), (bli_zimag(y[i__3]) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__2] );
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		if (bli_zreal(x[i__4]) != 0. || bli_zimag(x[i__4]) != 0.) {
		    i__4 = jx;
		    bli_tsets( z,z, (bli_zreal(*alpha) * bli_zreal(x[i__4]) - bli_zimag(*alpha) * bli_zimag(x[i__4])), (bli_zreal(*alpha) * bli_zimag(x[i__4]) + bli_zimag(*alpha) * bli_zreal(x[i__4])), z__1 );
		    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = iy;
			i__2 = iy;
			i__5 = k + i__ + j * a_dim1;
			bli_tsets( z,z, (bli_zreal(temp) * bli_zreal(a[i__5]) - bli_zimag(temp) * bli_zimag(a[i__5])), (bli_zreal(temp) * bli_zimag(a[i__5]) + bli_zimag(temp) * bli_zreal(a[i__5])), z__2 );
			bli_tsets( z,z, (bli_zreal(y[i__2]) + bli_zreal(z__2)), (bli_zimag(y[i__2]) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__4] );
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		bli_tsets( z,z, (0.), (0.), temp );
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			i__3 = k + i__ + j * a_dim1;
			i__4 = i__;
			bli_tsets( z,z, (bli_zreal(a[i__3]) * bli_zreal(x[i__4]) - bli_zimag(a[i__3]) * bli_zimag(x[i__4])), (bli_zreal(a[i__3]) * bli_zimag(x[i__4]) + bli_zimag(a[i__3]) * bli_zreal(x[i__4])), z__2 );
			bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L90: */
		    }
		} else {
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			bla_d_cnjg(&z__3, &a[k + i__ + j * a_dim1]);
			i__2 = i__;
			bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__2]) - bli_zimag(z__3) * bli_zimag(x[i__2])), (bli_zreal(z__3) * bli_zimag(x[i__2]) + bli_zimag(z__3) * bli_zreal(x[i__2])), z__2 );
			bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L100: */
		    }
		}
		i__4 = jy;
		i__2 = jy;
		bli_tsets( z,z, (bli_zreal(*alpha) * bli_zreal(temp) - bli_zimag(*alpha) * bli_zimag(temp)), (bli_zreal(*alpha) * bli_zimag(temp) + bli_zimag(*alpha) * bli_zreal(temp)), z__2 );
		bli_tsets( z,z, (bli_zreal(y[i__2]) + bli_zreal(z__2)), (bli_zimag(y[i__2]) + bli_zimag(z__2)), z__1 );
		bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__4] );
		jy += *incy;
/* L110: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		bli_tsets( z,z, (0.), (0.), temp );
		ix = kx;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = k + i__ + j * a_dim1;
			i__2 = ix;
			bli_tsets( z,z, (bli_zreal(a[i__4]) * bli_zreal(x[i__2]) - bli_zimag(a[i__4]) * bli_zimag(x[i__2])), (bli_zreal(a[i__4]) * bli_zimag(x[i__2]) + bli_zimag(a[i__4]) * bli_zreal(x[i__2])), z__2 );
			bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			ix += *incx;
/* L120: */
		    }
		} else {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			bla_d_cnjg(&z__3, &a[k + i__ + j * a_dim1]);
			i__3 = ix;
			bli_tsets( z,z, (bli_zreal(z__3) * bli_zreal(x[i__3]) - bli_zimag(z__3) * bli_zimag(x[i__3])), (bli_zreal(z__3) * bli_zimag(x[i__3]) + bli_zimag(z__3) * bli_zreal(x[i__3])), z__2 );
			bli_tsets( z,z, (bli_zreal(temp) + bli_zreal(z__2)), (bli_zimag(temp) + bli_zimag(z__2)), z__1 );
			bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			ix += *incx;
/* L130: */
		    }
		}
		i__2 = jy;
		i__3 = jy;
		bli_tsets( z,z, (bli_zreal(*alpha) * bli_zreal(temp) - bli_zimag(*alpha) * bli_zimag(temp)), (bli_zreal(*alpha) * bli_zimag(temp) + bli_zimag(*alpha) * bli_zreal(temp)), z__2 );
		bli_tsets( z,z, (bli_zreal(y[i__3]) + bli_zreal(z__2)), (bli_zimag(y[i__3]) + bli_zimag(z__2)), z__1 );
		bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), y[i__2] );
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L140: */
	    }
	}
    }

    return 0;

/*     End of ZGBMV . */

} /* zgbmv_ */

#endif

