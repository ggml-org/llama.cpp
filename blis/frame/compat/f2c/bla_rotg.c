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

/* srotg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Table of constant values */

static bla_real sc_b4 = 1.f;

/* Subroutine */ int PASTEF77(s,rotg)(bla_real *sa, bla_real *sb, bla_real *c__, bla_real *s)
{
    /* System generated locals */
    bla_real r__1, r__2;

    /* Builtin functions */
    //double sqrt(bla_double), bla_r_sign(bla_real *, bla_real *);

    /* Local variables */
    bla_real r__, scale, z__, roe;


/*     construct givens plane rotation. */
/*     jack dongarra, linpack, 3/11/78. */


    roe = *sb;
    if (bli_fabs(*sa) > bli_fabs(*sb)) {
	roe = *sa;
    }
    scale = bli_fabs(*sa) + bli_fabs(*sb);
    if (scale != 0.f) {
	goto L10;
    }
    *c__ = 1.f;
    *s = 0.f;
    r__ = 0.f;
    z__ = 0.f;
    goto L20;
L10:
/* Computing 2nd power */
    r__1 = *sa / scale;
/* Computing 2nd power */
    r__2 = *sb / scale;
    r__ = scale * sqrt(r__1 * r__1 + r__2 * r__2);
    r__ = bla_r_sign(&sc_b4, &roe) * r__;
    *c__ = *sa / r__;
    *s = *sb / r__;
    z__ = 1.f;
    if (bli_fabs(*sa) > bli_fabs(*sb)) {
	z__ = *s;
    }
    if (bli_fabs(*sb) >= bli_fabs(*sa) && *c__ != 0.f) {
	z__ = 1.f / *c__;
    }
L20:
    *sa = r__;
    *sb = z__;
    return 0;
} /* srotg_ */

/* drotg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Table of constant values */

static bla_double dc_b4 = 1.;

/* Subroutine */ int PASTEF77(d,rotg)(bla_double *da, bla_double *db, bla_double *c__, bla_double *s)
{
    /* System generated locals */
    bla_double d__1, d__2;

    /* Builtin functions */
    //double sqrt(bla_double), bla_d_sign(bla_double *, bla_double *);

    /* Local variables */
    bla_double r__, scale, z__, roe;


/*     construct givens plane rotation. */
/*     jack dongarra, linpack, 3/11/78. */


    roe = *db;
    if (bli_fabs(*da) > bli_fabs(*db)) {
	roe = *da;
    }
    scale = bli_fabs(*da) + bli_fabs(*db);
    if (scale != 0.) {
	goto L10;
    }
    *c__ = 1.;
    *s = 0.;
    r__ = 0.;
    z__ = 0.;
    goto L20;
L10:
/* Computing 2nd power */
    d__1 = *da / scale;
/* Computing 2nd power */
    d__2 = *db / scale;
    r__ = scale * sqrt(d__1 * d__1 + d__2 * d__2);
    r__ = bla_d_sign(&dc_b4, &roe) * r__;
    *c__ = *da / r__;
    *s = *db / r__;
    z__ = 1.;
    if (bli_fabs(*da) > bli_fabs(*db)) {
	z__ = *s;
    }
    if (bli_fabs(*db) >= bli_fabs(*da) && *c__ != 0.) {
	z__ = 1. / *c__;
    }
L20:
    *da = r__;
    *db = z__;
    return 0;
} /* drotg_ */

/* crotg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,rotg)(bla_scomplex *ca, bla_scomplex *cb, bla_real *c__, bla_scomplex *s)
{
    /* System generated locals */
    bla_real r__1, r__2;
    bla_scomplex q__1, q__2, q__3;

    /* Builtin functions */
    //double bla_c_abs(bla_scomplex *), sqrt(bla_double);
    //void bla_r_cnjg(bla_scomplex *, bla_scomplex *);

    /* Local variables */
    bla_real norm;
    bla_scomplex alpha;
    bla_real scale;

    if (bla_c_abs(ca) != 0.f) {
	goto L10;
    }
    *c__ = 0.f;
    bli_tsets( c,c, 1.f, 0.f, *s );
    bli_tsets( c,c, bli_creal(*cb), bli_cimag(*cb), *ca );
    goto L20;
L10:
    scale = bla_c_abs(ca) + bla_c_abs(cb);
    bli_tsets( c,c, (bli_creal(*ca) / scale), (bli_cimag(*ca) / scale), q__1 );
/* Computing 2nd power */
    r__1 = bla_c_abs(&q__1);
    bli_tsets( c,c, (bli_creal(*cb) / scale), (bli_cimag(*cb) / scale), q__2 );
/* Computing 2nd power */
    r__2 = bla_c_abs(&q__2);
    norm = scale * sqrt(r__1 * r__1 + r__2 * r__2);
    r__1 = bla_c_abs(ca);
    bli_tsets( c,c, (bli_creal(*ca) / r__1), (bli_cimag(*ca) / r__1), q__1 );
    bli_tsets( c,c, (bli_creal(q__1)), (bli_cimag(q__1)), alpha );
    *c__ = bla_c_abs(ca) / norm;
    bla_r_cnjg(&q__3, cb);
    bli_tsets( c,c, (bli_creal(alpha) * bli_creal(q__3) - bli_cimag(alpha) * bli_cimag(q__3)), (bli_creal(alpha) * bli_cimag(q__3) + bli_cimag(alpha) * bli_creal(q__3)), q__2 );
    bli_tsets( c,c, (bli_creal(q__2) / norm), (bli_cimag(q__2) / norm), q__1 );
    bli_tsets( c,c, bli_creal(q__1), bli_cimag(q__1), *s );
    bli_tsets( c,c, (norm * bli_creal(alpha)), (norm * bli_cimag(alpha)), q__1 );
    bli_tsets( c,c, bli_creal(q__1), bli_cimag(q__1), *ca );
L20:
    return 0;
} /* crotg_ */

/* zrotg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,rotg)(bla_dcomplex *ca, bla_dcomplex *cb, bla_double *c__, bla_dcomplex *s)
{
    /* System generated locals */
    bla_double d__1, d__2;
    bla_dcomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    //double bla_z_abs(bla_dcomplex *);
    //void bla_z_div(bla_dcomplex *, bla_dcomplex *, bla_dcomplex *);
    //double sqrt(bla_double);
    //void bla_d_cnjg(bla_dcomplex *, bla_dcomplex *);

    /* Local variables */
    bla_double norm;
    bla_dcomplex alpha;
    bla_double scale;

    if (bla_z_abs(ca) != 0.) {
	goto L10;
    }
    *c__ = 0.;
    bli_tsets( z,z, 1., 0., *s );
    bli_tsets( z,z, bli_zreal(*cb), bli_zimag(*cb), *ca );
    goto L20;
L10:
    scale = bla_z_abs(ca) + bla_z_abs(cb);
    bli_tsets( z,z, (scale), (0.), z__2 );
    bla_z_div(&z__1, ca, &z__2);
/* Computing 2nd power */
    d__1 = bla_z_abs(&z__1);
    bli_tsets( z,z, (scale), (0.), z__4 );
    bla_z_div(&z__3, cb, &z__4);
/* Computing 2nd power */
    d__2 = bla_z_abs(&z__3);
    norm = scale * sqrt(d__1 * d__1 + d__2 * d__2);
    d__1 = bla_z_abs(ca);
    bli_tsets( z,z, (bli_zreal(*ca) / d__1), (bli_zimag(*ca) / d__1), z__1 );
    bli_tsets( z,z, (bli_zreal(z__1)), (bli_zimag(z__1)), alpha );
    *c__ = bla_z_abs(ca) / norm;
    bla_d_cnjg(&z__3, cb);
    bli_tsets( z,z, (bli_zreal(alpha) * bli_zreal(z__3) - bli_zimag(alpha) * bli_zimag(z__3)), (bli_zreal(alpha) * bli_zimag(z__3) + bli_zimag(alpha) * bli_zreal(z__3)), z__2 );
    bli_tsets( z,z, (bli_zreal(z__2) / norm), (bli_zimag(z__2) / norm), z__1 );
    bli_tsets( z,z, bli_zreal(z__1), bli_zimag(z__1), *s );
    bli_tsets( z,z, (norm * bli_zreal(alpha)), (norm * bli_zimag(alpha)), z__1 );
    bli_tsets( z,z, bli_zreal(z__1), bli_zimag(z__1), *ca );
L20:
    return 0;
} /* zrotg_ */

#endif

