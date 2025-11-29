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

/* srotmg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,rotmg)(bla_real *sd1, bla_real *sd2, bla_real *sx1, const bla_real *sy1, bla_real *sparam)
{
    /* Initialized data */

    static bla_real zero = 0.f;
    static bla_real one = 1.f;
    static bla_real two = 2.f;
    static bla_real gam = 4096.f;
    static bla_real gamsq = 16777200.f;
    static bla_real rgamsq = 5.96046e-8f;

    /* Format strings */

    /* System generated locals */
    bla_real r__1;

    /* Local variables */
    bla_real sflag, stemp, su, sp1, sp2, sq2, sq1,
         sh11 = 0.f, sh21 = 0.f, sh12 = 0.f, sh22 = 0.f;
    bla_integer igo;

    /* Assigned format variables */


/*     CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS */
/*     THE SECOND COMPONENT OF THE 2-VECTOR  (SQRT(SD1)*SX1,SQRT(SD2)* */
/*     SY2)**T. */
/*     WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS.. */

/*     SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0 */

/*       (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0) */
/*     H=(          )    (          )    (          )    (          ) */
/*       (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0). */
/*     LOCATIONS 2-4 OF SPARAM CONTAIN SH11,SH21,SH12, AND SH22 */
/*     RESPECTIVELY. (VALUES OF 1.E0, -1.E0, OR 0.E0 IMPLIED BY THE */
/*     VALUE OF SPARAM(1) ARE NOT STORED IN SPARAM.) */

/*     THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE */
/*     INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE */
/*     OF SD1 AND SD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM. */


    /* Parameter adjustments */
    --sparam;

    /* Function Body */
    if (! (*sd1 < zero)) {
	goto L10;
    }
/*       GO ZERO-H-D-AND-SX1.. */
    goto L60;
L10:
/*     CASE-SD1-NONNEGATIVE */
    sp2 = *sd2 * *sy1;
    if (! (sp2 == zero)) {
	goto L20;
    }
    sflag = -two;
    goto L260;
/*     REGULAR-CASE.. */
L20:
    sp1 = *sd1 * *sx1;
    sq2 = sp2 * *sy1;
    sq1 = sp1 * *sx1;

    if (! (bli_fabs(sq1) > bli_fabs(sq2))) {
	goto L40;
    }
    sh21 = -(*sy1) / *sx1;
    sh12 = sp2 / sp1;

    su = one - sh12 * sh21;

    if (! (su <= zero)) {
	goto L30;
    }
/*         GO ZERO-H-D-AND-SX1.. */
    goto L60;
L30:
    sflag = zero;
    *sd1 /= su;
    *sd2 /= su;
    *sx1 *= su;
/*         GO SCALE-CHECK.. */
    goto L100;
L40:
    if (! (sq2 < zero)) {
	goto L50;
    }
/*         GO ZERO-H-D-AND-SX1.. */
    goto L60;
L50:
    sflag = one;
    sh11 = sp1 / sp2;
    sh22 = *sx1 / *sy1;
    su = one + sh11 * sh22;
    stemp = *sd2 / su;
    *sd2 = *sd1 / su;
    *sd1 = stemp;
    *sx1 = *sy1 * su;
/*         GO SCALE-CHECK */
    goto L100;
/*     PROCEDURE..ZERO-H-D-AND-SX1.. */
L60:
    sflag = -one;
    sh11 = zero;
    sh12 = zero;
    sh21 = zero;
    sh22 = zero;

    *sd1 = zero;
    *sd2 = zero;
    *sx1 = zero;
/*         RETURN.. */
    goto L220;
/*     PROCEDURE..FIX-H.. */
L70:
    if (! (sflag >= zero)) {
	goto L90;
    }

    if (! (sflag == zero)) {
	goto L80;
    }
    sh11 = one;
    sh22 = one;
    sflag = -one;
    goto L90;
L80:
    sh21 = -one;
    sh12 = one;
    sflag = -one;
L90:
    switch (igo) {
	case 0: goto L120;
	case 1: goto L150;
	case 2: goto L180;
	case 3: goto L210;
    }
/*     PROCEDURE..SCALE-CHECK */
L100:
L110:
    if (! (*sd1 <= rgamsq)) {
	goto L130;
    }
    if (*sd1 == zero) {
	goto L160;
    }
    igo = 0;
/*              FIX-H.. */
    goto L70;
L120:
/* Computing 2nd power */
    r__1 = gam;
    *sd1 *= r__1 * r__1;
    *sx1 /= gam;
    sh11 /= gam;
    sh12 /= gam;
    goto L110;
L130:
L140:
    if (! (*sd1 >= gamsq)) {
	goto L160;
    }
    igo = 1;
/*              FIX-H.. */
    goto L70;
L150:
/* Computing 2nd power */
    r__1 = gam;
    *sd1 /= r__1 * r__1;
    *sx1 *= gam;
    sh11 *= gam;
    sh12 *= gam;
    goto L140;
L160:
L170:
    if (! (bli_fabs(*sd2) <= rgamsq)) {
	goto L190;
    }
    if (*sd2 == zero) {
	goto L220;
    }
    igo = 2;
/*              FIX-H.. */
    goto L70;
L180:
/* Computing 2nd power */
    r__1 = gam;
    *sd2 *= r__1 * r__1;
    sh21 /= gam;
    sh22 /= gam;
    goto L170;
L190:
L200:
    if (! (bli_fabs(*sd2) >= gamsq)) {
	goto L220;
    }
    igo = 3;
/*              FIX-H.. */
    goto L70;
L210:
/* Computing 2nd power */
    r__1 = gam;
    *sd2 /= r__1 * r__1;
    sh21 *= gam;
    sh22 *= gam;
    goto L200;
L220:
    if (sflag < 0.f) {
	goto L250;
    } else if (sflag == 0) {
	goto L230;
    } else {
	goto L240;
    }
L230:
    sparam[3] = sh21;
    sparam[4] = sh12;
    goto L260;
L240:
    sparam[2] = sh11;
    sparam[5] = sh22;
    goto L260;
L250:
    sparam[2] = sh11;
    sparam[3] = sh21;
    sparam[4] = sh12;
    sparam[5] = sh22;
L260:
    sparam[1] = sflag;
    return 0;
} /* srotmg_ */

/* drotmg.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,rotmg)(bla_double *dd1, bla_double *dd2, bla_double *dx1, const bla_double *dy1, bla_double *dparam)
{
    /* Initialized data */

    static bla_double zero = 0.;
    static bla_double one = 1.;
    static bla_double two = 2.;
    static bla_double gam = 4096.;
    static bla_double gamsq = 16777216.;
    static bla_double rgamsq = 5.9604645e-8;

    /* Format strings */

    /* System generated locals */
    bla_double d__1;

    /* Local variables */
    bla_double dflag, dtemp, du, dp1, dp2, dq2, dq1,
               dh11 = 0.f, dh21 = 0.f, dh12 = 0.f, dh22 = 0.f;
    bla_integer igo;

    /* Assigned format variables */


/*     CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS */
/*     THE SECOND COMPONENT OF THE 2-VECTOR  (DSQRT(DD1)*DX1,DSQRT(DD2)* */
/*     DY2)**T. */
/*     WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS.. */

/*     DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0 */

/*       (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0) */
/*     H=(          )    (          )    (          )    (          ) */
/*       (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0). */
/*     LOCATIONS 2-4 OF DPARAM CONTAIN DH11, DH21, DH12, AND DH22 */
/*     RESPECTIVELY. (VALUES OF 1.D0, -1.D0, OR 0.D0 IMPLIED BY THE */
/*     VALUE OF DPARAM(1) ARE NOT STORED IN DPARAM.) */

/*     THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE */
/*     INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE */
/*     OF DD1 AND DD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM. */


    /* Parameter adjustments */
    --dparam;

    /* Function Body */
    if (! (*dd1 < zero)) {
	goto L10;
    }
/*       GO ZERO-H-D-AND-DX1.. */
    goto L60;
L10:
/*     CASE-DD1-NONNEGATIVE */
    dp2 = *dd2 * *dy1;
    if (! (dp2 == zero)) {
	goto L20;
    }
    dflag = -two;
    goto L260;
/*     REGULAR-CASE.. */
L20:
    dp1 = *dd1 * *dx1;
    dq2 = dp2 * *dy1;
    dq1 = dp1 * *dx1;

    if (! (bli_fabs(dq1) > bli_fabs(dq2))) {
	goto L40;
    }
    dh21 = -(*dy1) / *dx1;
    dh12 = dp2 / dp1;

    du = one - dh12 * dh21;

    if (! (du <= zero)) {
	goto L30;
    }
/*         GO ZERO-H-D-AND-DX1.. */
    goto L60;
L30:
    dflag = zero;
    *dd1 /= du;
    *dd2 /= du;
    *dx1 *= du;
/*         GO SCALE-CHECK.. */
    goto L100;
L40:
    if (! (dq2 < zero)) {
	goto L50;
    }
/*         GO ZERO-H-D-AND-DX1.. */
    goto L60;
L50:
    dflag = one;
    dh11 = dp1 / dp2;
    dh22 = *dx1 / *dy1;
    du = one + dh11 * dh22;
    dtemp = *dd2 / du;
    *dd2 = *dd1 / du;
    *dd1 = dtemp;
    *dx1 = *dy1 * du;
/*         GO SCALE-CHECK */
    goto L100;
/*     PROCEDURE..ZERO-H-D-AND-DX1.. */
L60:
    dflag = -one;
    dh11 = zero;
    dh12 = zero;
    dh21 = zero;
    dh22 = zero;

    *dd1 = zero;
    *dd2 = zero;
    *dx1 = zero;
/*         RETURN.. */
    goto L220;
/*     PROCEDURE..FIX-H.. */
L70:
    if (! (dflag >= zero)) {
	goto L90;
    }

    if (! (dflag == zero)) {
	goto L80;
    }
    dh11 = one;
    dh22 = one;
    dflag = -one;
    goto L90;
L80:
    dh21 = -one;
    dh12 = one;
    dflag = -one;
L90:
    switch (igo) {
	case 0: goto L120;
	case 1: goto L150;
	case 2: goto L180;
	case 3: goto L210;
    }
/*     PROCEDURE..SCALE-CHECK */
L100:
L110:
    if (! (*dd1 <= rgamsq)) {
	goto L130;
    }
    if (*dd1 == zero) {
	goto L160;
    }
    igo = 0;
/*              FIX-H.. */
    goto L70;
L120:
/* Computing 2nd power */
    d__1 = gam;
    *dd1 *= d__1 * d__1;
    *dx1 /= gam;
    dh11 /= gam;
    dh12 /= gam;
    goto L110;
L130:
L140:
    if (! (*dd1 >= gamsq)) {
	goto L160;
    }
    igo = 1;
/*              FIX-H.. */
    goto L70;
L150:
/* Computing 2nd power */
    d__1 = gam;
    *dd1 /= d__1 * d__1;
    *dx1 *= gam;
    dh11 *= gam;
    dh12 *= gam;
    goto L140;
L160:
L170:
    if (! (bli_fabs(*dd2) <= rgamsq)) {
	goto L190;
    }
    if (*dd2 == zero) {
	goto L220;
    }
    igo = 2;
/*              FIX-H.. */
    goto L70;
L180:
/* Computing 2nd power */
    d__1 = gam;
    *dd2 *= d__1 * d__1;
    dh21 /= gam;
    dh22 /= gam;
    goto L170;
L190:
L200:
    if (! (bli_fabs(*dd2) >= gamsq)) {
	goto L220;
    }
    igo = 3;
/*              FIX-H.. */
    goto L70;
L210:
/* Computing 2nd power */
    d__1 = gam;
    *dd2 /= d__1 * d__1;
    dh21 *= gam;
    dh22 *= gam;
    goto L200;
L220:
    if (dflag < 0.) {
	goto L250;
    } else if (dflag == 0) {
	goto L230;
    } else {
	goto L240;
    }
L230:
    dparam[3] = dh21;
    dparam[4] = dh12;
    goto L260;
L240:
    dparam[2] = dh11;
    dparam[5] = dh22;
    goto L260;
L250:
    dparam[2] = dh11;
    dparam[3] = dh21;
    dparam[4] = dh12;
    dparam[5] = dh22;
L260:
    dparam[1] = dflag;
    return 0;
} /* drotmg_ */

#endif

