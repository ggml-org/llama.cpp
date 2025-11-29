/* dblat1.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    integer icase, n, incx, incy;
    logical pass;
} combla_;

#define combla_1 combla_

/* Table of constant values */

static integer c__1 = 1;
static integer c__9 = 9;
static doublereal c_b35 = 1.;
static real c_b39 = .03125f;
static integer c__5 = 5;
static doublereal c_b63 = 0.;
static real c_b81 = 0.f;

/* > \brief \b DBLAT1 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM DBLAT1 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    Test program for the DOUBLE PRECISION Level 1 BLAS. */
/* > */
/* >    Based upon the original BLAS test routine together with: */
/* >    F06EAF Example Program Text */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date April 2012 */

/* > \ingroup double_blas_testing */

/*  ===================================================================== */
/* Main program */ int main(void)
{
#ifdef BLIS_ENABLE_HPX
    char* program = "dblat1";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static doublereal sfac = 9.765625e-4;

    /* Format strings */
    static char fmt_99999[] = "(\002 Real BLAS Test Program Results\002,/1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer ic;
    extern /* Subroutine */ int check0_(doublereal *), check1_(doublereal *),
	    check2_(doublereal *), check3_(doublereal *), header_(void);

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___4 = { 0, 6, 0, fmt_99998, 0 };



/*  -- Reference BLAS test routine (version 3.4.1) -- */
/*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. External Subroutines .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     .. Executable Statements .. */
    s_wsfe(&io___2);
    e_wsfe();
    for (ic = 1; ic <= 13; ++ic) {
	combla_1.icase = ic;
	header_();

/*        .. Initialize  PASS,  INCX,  and INCY for a new case. .. */
/*        .. the value 9999 for INCX or INCY will appear in the .. */
/*        .. detailed  output, if any, for cases  that do not involve .. */
/*        .. these parameters .. */

	combla_1.pass = TRUE_;
	combla_1.incx = 9999;
	combla_1.incy = 9999;
	if (combla_1.icase == 3 || combla_1.icase == 11) {
	    check0_(&sfac);
	} else if (combla_1.icase == 7 || combla_1.icase == 8 ||
		combla_1.icase == 9 || combla_1.icase == 10) {
	    check1_(&sfac);
	} else if (combla_1.icase == 1 || combla_1.icase == 2 ||
		combla_1.icase == 5 || combla_1.icase == 6 || combla_1.icase
		== 12 || combla_1.icase == 13) {
	    check2_(&sfac);
	} else if (combla_1.icase == 4) {
	    check3_(&sfac);
	}
/*        -- Print */
	if (combla_1.pass) {
	    s_wsfe(&io___4);
	    e_wsfe();
	}
/* L20: */
    }
    s_stop("", (ftnlen)0);

#ifdef BLIS_ENABLE_HPX
    return bli_thread_finalize_hpx();
#else
	// Return peacefully.
	return 0;
#endif
} /* main */

/* Subroutine */ int header_(void)
{
    /* Initialized data */

    static char l[6*13] = " DDOT " "DAXPY " "DROTG " " DROT " "DCOPY " "DSWA"
	    "P " "DNRM2 " "DASUM " "DSCAL " "IDAMAX" "DROTMG" "DROTM " "DSDOT "
	    ;

    /* Format strings */
    static char fmt_99999[] = "(/\002 Test of subprogram number\002,i3,12x,a"
	    "6)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Fortran I/O blocks */
    static cilist io___6 = { 0, 6, 0, fmt_99999, 0 };


/*     .. Parameters .. */
/*     .. Scalars in Common .. */
/*     .. Local Arrays .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     .. Executable Statements .. */
    s_wsfe(&io___6);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, l + (0 + (0 + (combla_1.icase - 1) * 6)), (ftnlen)6);
    e_wsfe();
    return 0;

} /* header_ */

/* Subroutine */ int check0_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal ds1[8] = { .8,.6,.8,-.6,.8,0.,1.,0. };
    static doublereal datrue[8] = { .5,.5,.5,-.5,-.5,0.,1.,1. };
    static doublereal dbtrue[8] = { 0.,.6,0.,-.6,0.,0.,1.,0. };
    static doublereal dab[36]	/* was [4][9] */ = { .1,.3,1.2,.2,.7,.2,.6,
	    4.2,0.,0.,0.,0.,4.,-1.,2.,4.,6e-10,.02,1e5,10.,4e10,.02,1e-5,10.,
	    2e-10,.04,1e5,10.,2e10,.04,1e-5,10.,4.,-2.,8.,4. };
    static doublereal dtrue[81]	/* was [9][9] */ = { 0.,0.,1.3,.2,0.,0.,0.,.5,
	    0.,0.,0.,4.5,4.2,1.,.5,0.,0.,0.,0.,0.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,
	    0.,4.,-1.,0.,0.,0.,0.,0.,.015,0.,10.,-1.,0.,-1e-4,0.,1.,0.,0.,
	    .06144,10.,-1.,4096.,-1e6,0.,1.,0.,0.,15.,10.,-1.,5e-5,0.,1.,0.,
	    0.,0.,15.,10.,-1.,5e5,-4096.,1.,.004096,0.,0.,7.,4.,0.,0.,-.5,
	    -.25,0. };
    static doublereal d12 = 4096.;
    static doublereal da1[8] = { .3,.4,-.3,-.4,-.3,0.,0.,1. };
    static doublereal db1[8] = { .4,.3,.4,.3,-.4,0.,1.,0. };
    static doublereal dc1[8] = { .6,.8,-.6,.8,.6,1.,0.,1. };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, k;
    doublereal sa, sb, sc, ss, dtemp[9];
    extern /* Subroutine */ int drotg_(doublereal *, doublereal *, doublereal
	    *, doublereal *), stest_(integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *), stest1_(doublereal *, doublereal *,
	    doublereal *, doublereal *), drotmg_(doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *);

    /* Fortran I/O blocks */
    static cilist io___23 = { 0, 6, 0, 0, 0 };


/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     INPUT FOR MODIFIED GIVENS */
/*    TRUE RESULTS FOR MODIFIED GIVENS */
/*                   4096 = 2 ** 12 */
    dtrue[0] = .092307692307692313;
    dtrue[1] = .27692307692307694;
    dtrue[6] = -.16666666666666666;
    dtrue[9] = .18666666666666668;
    dtrue[10] = .65333333333333332;
    dtrue[17] = .14285714285714285;
    dtrue[36] = d12 * d12 * 4.5e-10;
    dtrue[38] = 4e5 / (d12 * 3.);
    dtrue[41] = 1. / d12;
    dtrue[43] = 1e4 / (d12 * 3.);
    dtrue[45] = 4e10 / (d12 * 1.5 * d12);
    dtrue[46] = .013333333333333334;
    dtrue[52] = d12 * 5e-7;
    dtrue[54] = .026666666666666668;
    dtrue[55] = d12 * d12 * 1.3333333333333334e-10;
    dtrue[60] = -dtrue[41];
    dtrue[62] = 1e4 / d12;
    dtrue[63] = dtrue[54];
    dtrue[64] = 2e10 / (d12 * 1.5 * d12);
    dtrue[72] = 4.5714285714285712;
    dtrue[73] = -2.2857142857142856;
/*     .. Executable Statements .. */

/*     Compute true values which cannot be prestored */
/*     in decimal notation */

    dbtrue[0] = 1.6666666666666667;
    dbtrue[2] = -1.6666666666666667;
    dbtrue[4] = 1.6666666666666667;

    for (k = 1; k <= 8; ++k) {
/*        .. Set N=K for identification in output if any .. */
	combla_1.n = k;
	if (combla_1.icase == 3) {
/*           .. DROTG .. */
	    if (k > 8) {
		goto L40;
	    }
	    sa = da1[k - 1];
	    sb = db1[k - 1];
	    drotg_(&sa, &sb, &sc, &ss);
	    stest1_(&sa, &datrue[k - 1], &datrue[k - 1], sfac);
	    stest1_(&sb, &dbtrue[k - 1], &dbtrue[k - 1], sfac);
	    stest1_(&sc, &dc1[k - 1], &dc1[k - 1], sfac);
	    stest1_(&ss, &ds1[k - 1], &ds1[k - 1], sfac);
	} else if (combla_1.icase == 11) {
/*           .. DROTMG .. */
	    for (i__ = 1; i__ <= 4; ++i__) {
		dtemp[i__ - 1] = dab[i__ + (k << 2) - 5];
		dtemp[i__ + 3] = 0.f;
	    }
	    dtemp[8] = 0.f;
	    drotmg_(dtemp, &dtemp[1], &dtemp[2], &dtemp[3], &dtemp[4]);
	    stest_(&c__9, dtemp, &dtrue[k * 9 - 9], &dtrue[k * 9 - 9], sfac);
	} else {
	    s_wsle(&io___23);
	    do_lio(&c__9, &c__1, " Shouldn't be here in CHECK0", (ftnlen)28);
	    e_wsle();
	    s_stop("", (ftnlen)0);
	}
/* L20: */
    }
L40:
    return 0;
} /* check0_ */

/* Subroutine */ int check1_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal sa[10] = { .3,-1.,0.,1.,.3,.3,.3,.3,.3,.3 };
    static doublereal dv[80]	/* was [8][5][2] */ = { .1,2.,2.,2.,2.,2.,2.,
	    2.,.3,3.,3.,3.,3.,3.,3.,3.,.3,-.4,4.,4.,4.,4.,4.,4.,.2,-.6,.3,5.,
	    5.,5.,5.,5.,.1,-.3,.5,-.1,6.,6.,6.,6.,.1,8.,8.,8.,8.,8.,8.,8.,.3,
	    9.,9.,9.,9.,9.,9.,9.,.3,2.,-.4,2.,2.,2.,2.,2.,.2,3.,-.6,5.,.3,2.,
	    2.,2.,.1,4.,-.3,6.,-.5,7.,-.1,3. };
    static doublereal dtrue1[5] = { 0.,.3,.5,.7,.6 };
    static doublereal dtrue3[5] = { 0.,.3,.7,1.1,1. };
    static doublereal dtrue5[80]	/* was [8][5][2] */ = { .1,2.,2.,2.,
	    2.,2.,2.,2.,-.3,3.,3.,3.,3.,3.,3.,3.,0.,0.,4.,4.,4.,4.,4.,4.,.2,
	    -.6,.3,5.,5.,5.,5.,5.,.03,-.09,.15,-.03,6.,6.,6.,6.,.1,8.,8.,8.,
	    8.,8.,8.,8.,.09,9.,9.,9.,9.,9.,9.,9.,.09,2.,-.12,2.,2.,2.,2.,2.,
	    .06,3.,-.18,5.,.09,2.,2.,2.,.03,4.,-.09,6.,-.15,7.,-.03,3. };
    static integer itrue2[5] = { 0,1,2,2,3 };

    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__;
    doublereal sx[8];
    integer np1, len;
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *,
	    integer *);
    extern doublereal dasum_(integer *, doublereal *, integer *);
    doublereal stemp[1], strue[8];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *), itest1_(integer *, integer *),
	    stest1_(doublereal *, doublereal *, doublereal *, doublereal *);
    extern integer idamax_(integer *, doublereal *, integer *);

    /* Fortran I/O blocks */
    static cilist io___36 = { 0, 6, 0, 0, 0 };


/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     .. Executable Statements .. */
    for (combla_1.incx = 1; combla_1.incx <= 2; ++combla_1.incx) {
	for (np1 = 1; np1 <= 5; ++np1) {
	    combla_1.n = np1 - 1;
	    len = max(combla_1.n,1) << 1;
/*           .. Set vector arguments .. */
	    i__1 = len;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		sx[i__ - 1] = dv[i__ + (np1 + combla_1.incx * 5 << 3) - 49];
/* L20: */
	    }

	    if (combla_1.icase == 7) {
/*              .. DNRM2 .. */
		stemp[0] = dtrue1[np1 - 1];
		d__1 = dnrm2_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&d__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 8) {
/*              .. DASUM .. */
		stemp[0] = dtrue3[np1 - 1];
		d__1 = dasum_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&d__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 9) {
/*              .. DSCAL .. */
		dscal_(&combla_1.n, &sa[(combla_1.incx - 1) * 5 + np1 - 1],
			sx, &combla_1.incx);
		i__1 = len;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    strue[i__ - 1] = dtrue5[i__ + (np1 + combla_1.incx * 5 <<
			    3) - 49];
/* L40: */
		}
		stest_(&len, sx, strue, strue, sfac);
	    } else if (combla_1.icase == 10) {
/*              .. IDAMAX .. */
		i__1 = idamax_(&combla_1.n, sx, &combla_1.incx);
		itest1_(&i__1, &itrue2[np1 - 1]);
	    } else {
		s_wsle(&io___36);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK1", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L60: */
	}
/* L80: */
    }
    return 0;
} /* check1_ */

/* Subroutine */ int check2_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal sa = .3;
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static doublereal dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    static doublereal dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    static real sx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    static real sy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    static doublereal dt7[16]	/* was [4][4] */ = { 0.,.3,.21,.62,0.,.3,-.07,
	    .85,0.,.3,-.79,-.74,0.,.3,.33,1.27 };
    static doublereal dt8[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,0.,0.,0.,
	    .68,0.,0.,0.,0.,0.,0.,.68,-.87,0.,0.,0.,0.,0.,.68,-.87,.15,.94,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.35,-.9,.48,0.,
	    0.,0.,0.,.38,-.9,.57,.7,-.75,.2,.98,.5,0.,0.,0.,0.,0.,0.,.68,0.,
	    0.,0.,0.,0.,0.,.35,-.72,0.,0.,0.,0.,0.,.38,-.63,.15,.88,0.,0.,0.,
	    .5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.68,-.9,.33,0.,0.,0.,
	    0.,.68,-.9,.33,.7,-.75,.2,1.04 };
    static doublereal dt10x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,
	    0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,-.9,0.,0.,0.,0.,0.,.5,-.9,.3,.7,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.3,.1,.5,0.,0.,
	    0.,0.,.8,.1,-.6,.8,.3,-.3,.5,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,
	    0.,0.,-.9,.1,.5,0.,0.,0.,0.,.7,.1,.3,.8,-.9,-.3,.5,.6,0.,0.,0.,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,.3,0.,0.,0.,0.,0.,.5,.3,-.6,.8,0.,
	    0.,0. };
    static doublereal dt10y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,.1,0.,0.,0.,0.,0.,.6,.1,-.5,.8,
	    0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,-.5,-.9,.6,0.,
	    0.,0.,0.,-.4,-.9,.9,.7,-.5,.2,.6,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,
	    0.,0.,0.,-.5,.6,0.,0.,0.,0.,0.,-.4,.9,-.5,.6,0.,0.,0.,.5,0.,0.,0.,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,-.9,.1,0.,0.,0.,0.,.6,-.9,.1,.7,
	    -.5,.2,.8 };
    static doublereal ssize1[4] = { 0.,.3,1.6,3.2 };
    static doublereal ssize2[28]	/* was [14][2] */ = { 0.,0.,0.,0.,0.,
	    0.,0.,0.,0.,0.,0.,0.,0.,0.,1.17,1.17,1.17,1.17,1.17,1.17,1.17,
	    1.17,1.17,1.17,1.17,1.17,1.17,1.17 };
    static doublereal dpar[20]	/* was [5][4] */ = { -2.,0.,0.,0.,0.,-1.,2.,
	    -3.,-4.,5.,0.,0.,2.,-3.,0.,1.,5.,2.,0.,-4. };
    static struct {
	doublereal e_1[448];
	} equiv_3 = {{ .6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0.,
		.6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0., .6,
		0., 0., 0., 0., 0., 0., -.8, 0., 0., 0., 0., 0., 0., -.9, 0.,
		0., 0., 0., 0., 0., 3.5, 0., 0., 0., 0., 0., 0., .6, .1, 0.,
		0., 0., 0., 0., -.8, 3.8, 0., 0., 0., 0., 0., -.9, 2.8, 0.,
		0., 0., 0., 0., 3.5, -.4, 0., 0., 0., 0., 0., .6, .1, -.5, .8,
		 0., 0., 0., -.8, 3.8, -2.2, -1.2, 0., 0., 0., -.9, 2.8, -1.4,
		 -1.3, 0., 0., 0., 3.5, -.4, -2.2, 4.7, 0., 0., 0., .6, 0.,
		0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0., .6, 0., 0.,
		0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0.,
		0., 0., 0., -.8, 0., 0., 0., 0., 0., 0., -.9, 0., 0., 0., 0.,
		0., 0., 3.5, 0., 0., 0., 0., 0., 0., .6, .1, -.5, 0., 0., 0.,
		0., 0., .1, -3., 0., 0., 0., 0., -.3, .1, -2., 0., 0., 0., 0.,
		 3.3, .1, -2., 0., 0., 0., 0., .6, .1, -.5, .8, .9, -.3, -.4,
		-2., .1, 1.4, .8, .6, -.3, -2.8, -1.8, .1, 1.3, .8, 0., -.3,
		-1.9, 3.8, .1, -3.1, .8, 4.8, -.3, -1.5, .6, 0., 0., 0., 0.,
		0., 0., .6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0.,
		0., .6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0.,
		-.8, 0., 0., 0., 0., 0., 0., -.9, 0., 0., 0., 0., 0., 0., 3.5,
		 0., 0., 0., 0., 0., 0., .6, .1, -.5, 0., 0., 0., 0., 4.8, .1,
		 -3., 0., 0., 0., 0., 3.3, .1, -2., 0., 0., 0., 0., 2.1, .1,
		-2., 0., 0., 0., 0., .6, .1, -.5, .8, .9, -.3, -.4, -1.6, .1,
		-2.2, .8, 5.4, -.3, -2.8, -1.5, .1, -1.4, .8, 3.6, -.3, -1.9,
		3.7, .1, -2.2, .8, 3.6, -.3, -1.5, .6, 0., 0., 0., 0., 0., 0.,
		 .6, 0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0., .6,
		0., 0., 0., 0., 0., 0., .6, 0., 0., 0., 0., 0., 0., -.8, 0.,
		0., 0., 0., 0., 0., -.9, 0., 0., 0., 0., 0., 0., 3.5, 0., 0.,
		0., 0., 0., 0., .6, .1, 0., 0., 0., 0., 0., -.8, -1., 0., 0.,
		0., 0., 0., -.9, -.8, 0., 0., 0., 0., 0., 3.5, .8, 0., 0., 0.,
		 0., 0., .6, .1, -.5, .8, 0., 0., 0., -.8, -1., 1.4, -1.6, 0.,
		 0., 0., -.9, -.8, 1.3, -1.6, 0., 0., 0., 3.5, .8, -3.1, 4.8,
		0., 0., 0. }};

    static struct {
	doublereal e_1[448];
	} equiv_7 = {{ .5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0., 0., 0.,
		.5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .5,
		0., 0., 0., 0., 0., 0., .7, 0., 0., 0., 0., 0., 0., 1.7, 0.,
		0., 0., 0., 0., 0., -2.6, 0., 0., 0., 0., 0., 0., .5, -.9, 0.,
		 0., 0., 0., 0., .7, -4.8, 0., 0., 0., 0., 0., 1.7, -.7, 0.,
		0., 0., 0., 0., -2.6, 3.5, 0., 0., 0., 0., 0., .5, -.9, .3,
		.7, 0., 0., 0., .7, -4.8, 3., 1.1, 0., 0., 0., 1.7, -.7, -.7,
		2.3, 0., 0., 0., -2.6, 3.5, -.7, -3.6, 0., 0., 0., .5, 0., 0.,
		 0., 0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0.,
		0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0.,
		0., 0., .7, 0., 0., 0., 0., 0., 0., 1.7, 0., 0., 0., 0., 0.,
		0., -2.6, 0., 0., 0., 0., 0., 0., .5, -.9, .3, 0., 0., 0., 0.,
		 4., -.9, -.3, 0., 0., 0., 0., -.5, -.9, 1.5, 0., 0., 0., 0.,
		-1.5, -.9, -1.8, 0., 0., 0., 0., .5, -.9, .3, .7, -.6, .2, .8,
		 3.7, -.9, -1.2, .7, -1.5, .2, 2.2, -.3, -.9, 2.1, .7, -1.6,
		.2, 2., -1.6, -.9, -2.1, .7, 2.9, .2, -3.8, .5, 0., 0., 0.,
		0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0.,
		0., 0., .5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0., 0.,
		0., .7, 0., 0., 0., 0., 0., 0., 1.7, 0., 0., 0., 0., 0., 0.,
		-2.6, 0., 0., 0., 0., 0., 0., .5, -.9, 0., 0., 0., 0., 0., 4.,
		 -6.3, 0., 0., 0., 0., 0., -.5, .3, 0., 0., 0., 0., 0., -1.5,
		3., 0., 0., 0., 0., 0., .5, -.9, .3, .7, 0., 0., 0., 3.7,
		-7.2, 3., 1.7, 0., 0., 0., -.3, .9, -.7, 1.9, 0., 0., 0.,
		-1.6, 2.7, -.7, -3.4, 0., 0., 0., .5, 0., 0., 0., 0., 0., 0.,
		.5, 0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .5,
		0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0., 0., 0., .7, 0.,
		0., 0., 0., 0., 0., 1.7, 0., 0., 0., 0., 0., 0., -2.6, 0., 0.,
		 0., 0., 0., 0., .5, -.9, .3, 0., 0., 0., 0., .7, -.9, 1.2,
		0., 0., 0., 0., 1.7, -.9, .5, 0., 0., 0., 0., -2.6, -.9, -1.3,
		 0., 0., 0., 0., .5, -.9, .3, .7, -.6, .2, .8, .7, -.9, 1.2,
		.7, -1.5, .2, 1.6, 1.7, -.9, .5, .7, -1.6, .2, 2.4, -2.6, -.9,
		 -1.3, .7, 2.9, .2, -4. }};


    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;
    doublereal d__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, j;
    extern /* Subroutine */ int testdsdot_(real *, real *, real *, real *);
    integer ki, kn, mx, my;
    doublereal sx[7], sy[7];
    integer kni;
    doublereal stx[7], sty[7];
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *,
	    integer *);
    integer kpar, lenx, leny;
#define dt19x ((doublereal *)&equiv_3)
#define dt19y ((doublereal *)&equiv_7)
    doublereal dtemp[5];
#define dt19xa ((doublereal *)&equiv_3)
#define dt19xb ((doublereal *)&equiv_3 + 112)
#define dt19xc ((doublereal *)&equiv_3 + 224)
#define dt19xd ((doublereal *)&equiv_3 + 336)
#define dt19ya ((doublereal *)&equiv_7)
#define dt19yb ((doublereal *)&equiv_7 + 112)
#define dt19yc ((doublereal *)&equiv_7 + 224)
#define dt19yd ((doublereal *)&equiv_7 + 336)
    extern doublereal dsdot_(integer *, real *, integer *, real *, integer *);
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *,
	    doublereal *, integer *);
    integer ksize;
    extern /* Subroutine */ int daxpy_(integer *, doublereal *, doublereal *,
	    integer *, doublereal *, integer *), drotm_(integer *, doublereal
	    *, integer *, doublereal *, integer *, doublereal *), dswap_(
	    integer *, doublereal *, integer *, doublereal *, integer *);
    doublereal ssize[7];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *), stest1_(doublereal *, doublereal *,
	    doublereal *, doublereal *);

    /* Fortran I/O blocks */
    static cilist io___80 = { 0, 6, 0, 0, 0 };


/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/* **** FGVZ: We have to add separate REAL arrays for DSDOT() because */
/* ****       REAL() on an array argument does not translate via f2c. */

/*                         FOR DROTM */

/*                        TRUE X RESULTS F0R ROTATIONS DROTM */



/*                        TRUE Y RESULTS FOR ROTATIONS DROTM */




/*     .. Executable Statements .. */

    for (ki = 1; ki <= 4; ++ki) {
	combla_1.incx = incxs[ki - 1];
	combla_1.incy = incys[ki - 1];
	mx = abs(combla_1.incx);
	my = abs(combla_1.incy);

	for (kn = 1; kn <= 4; ++kn) {
	    combla_1.n = ns[kn - 1];
	    ksize = min(2,kn);
	    lenx = lens[kn + (mx << 2) - 5];
	    leny = lens[kn + (my << 2) - 5];
/*           .. Initialize all argument arrays .. */
	    for (i__ = 1; i__ <= 7; ++i__) {
		sx[i__ - 1] = dx1[i__ - 1];
		sy[i__ - 1] = dy1[i__ - 1];
/* **** FGVZ: We have to add a loop to initialize separate REAL arrays */
/* ****       for DSDOT() because REAL() on an array argument does not */
/* ****       translate via f2c. */
		sx1[i__ - 1] = dx1[i__ - 1];
		sy1[i__ - 1] = dy1[i__ - 1];
/* L20: */
	    }

	    if (combla_1.icase == 1) {
/*              .. DDOT .. */
		d__1 = ddot_(&combla_1.n, sx, &combla_1.incx, sy, &
			combla_1.incy);
		stest1_(&d__1, &dt7[kn + (ki << 2) - 5], &ssize1[kn - 1],
			sfac);
	    } else if (combla_1.icase == 2) {
/*              .. DAXPY .. */
		daxpy_(&combla_1.n, &sa, sx, &combla_1.incx, sy, &
			combla_1.incy);
		i__1 = leny;
		for (j = 1; j <= i__1; ++j) {
		    sty[j - 1] = dt8[j + (kn + (ki << 2)) * 7 - 36];
/* L40: */
		}
		stest_(&leny, sy, sty, &ssize2[ksize * 14 - 14], sfac);
	    } else if (combla_1.icase == 5) {
/*              .. DCOPY .. */
		for (i__ = 1; i__ <= 7; ++i__) {
		    sty[i__ - 1] = dt10y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L60: */
		}
		dcopy_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		stest_(&leny, sy, sty, ssize2, &c_b35);
	    } else if (combla_1.icase == 6) {
/*              .. DSWAP .. */
		dswap_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		for (i__ = 1; i__ <= 7; ++i__) {
		    stx[i__ - 1] = dt10x[i__ + (kn + (ki << 2)) * 7 - 36];
		    sty[i__ - 1] = dt10y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L80: */
		}
		stest_(&lenx, sx, stx, ssize2, &c_b35);
		stest_(&leny, sy, sty, ssize2, &c_b35);
	    } else if (combla_1.icase == 12) {
/*              .. DROTM .. */
		kni = kn + (ki - 1 << 2);
		for (kpar = 1; kpar <= 4; ++kpar) {
		    for (i__ = 1; i__ <= 7; ++i__) {
			sx[i__ - 1] = dx1[i__ - 1];
			sy[i__ - 1] = dy1[i__ - 1];
			stx[i__ - 1] = dt19x[i__ + (kpar + (kni << 2)) * 7 -
				36];
			sty[i__ - 1] = dt19y[i__ + (kpar + (kni << 2)) * 7 -
				36];
		    }

		    for (i__ = 1; i__ <= 5; ++i__) {
			dtemp[i__ - 1] = dpar[i__ + kpar * 5 - 6];
		    }

		    i__1 = lenx;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			ssize[i__ - 1] = stx[i__ - 1];
		    }
/*                   SEE REMARK ABOVE ABOUT DT11X(1,2,7) */
/*                       AND DT11X(5,3,8). */
		    if (kpar == 2 && kni == 7) {
			ssize[0] = 2.4;
		    }
		    if (kpar == 3 && kni == 8) {
			ssize[4] = 1.8;
		    }

		    drotm_(&combla_1.n, sx, &combla_1.incx, sy, &
			    combla_1.incy, dtemp);
		    stest_(&lenx, sx, stx, ssize, sfac);
		    stest_(&leny, sy, sty, sty, sfac);
		}
	    } else if (combla_1.icase == 13) {
/*              .. DSDOT .. */
/* ****       CALL TESTDSDOT(REAL(DSDOT(N,REAL(SX),INCX,REAL(SY),INCY)), */
		r__1 = (real) dsdot_(&combla_1.n, sx1, &combla_1.incx, sy1, &
			combla_1.incy);
		r__2 = (real) dt7[kn + (ki << 2) - 5];
		r__3 = (real) ssize1[kn - 1];
		testdsdot_(&r__1, &r__2, &r__3, &c_b39);
	    } else {
		s_wsle(&io___80);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK2", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L100: */
	}
/* L120: */
    }
    return 0;
} /* check2_ */

#undef dt19yd
#undef dt19yc
#undef dt19yb
#undef dt19ya
#undef dt19xd
#undef dt19xc
#undef dt19xb
#undef dt19xa
#undef dt19y
#undef dt19x


/* Subroutine */ int check3_(doublereal *sfac)
{
    /* Initialized data */

    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static doublereal dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    static doublereal dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    static doublereal sc = .8;
    static doublereal ss = .6;
    static doublereal dt9x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,0.,0.,0.,
	    .78,0.,0.,0.,0.,0.,0.,.78,-.46,0.,0.,0.,0.,0.,.78,-.46,-.22,1.06,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.66,.1,-.1,0.,
	    0.,0.,0.,.96,.1,-.76,.8,.9,-.3,-.02,.6,0.,0.,0.,0.,0.,0.,.78,0.,
	    0.,0.,0.,0.,0.,-.06,.1,-.1,0.,0.,0.,0.,.9,.1,-.22,.8,.18,-.3,-.02,
	    .6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.78,.26,0.,0.,0.,0.,0.,
	    .78,.26,-.76,1.12,0.,0.,0. };
    static doublereal dt9y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,0.,0.,0.,
	    .04,0.,0.,0.,0.,0.,0.,.04,-.78,0.,0.,0.,0.,0.,.04,-.78,.54,.08,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.7,-.9,-.12,0.,
	    0.,0.,0.,.64,-.9,-.3,.7,-.18,.2,.28,.5,0.,0.,0.,0.,0.,0.,.04,0.,
	    0.,0.,0.,0.,0.,.7,-1.08,0.,0.,0.,0.,0.,.64,-1.26,.54,.2,0.,0.,0.,
	    .5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.04,-.9,.18,0.,0.,0.,
	    0.,.04,-.9,.18,.7,-.18,.2,.16 };
    static doublereal ssize2[28]	/* was [14][2] */ = { 0.,0.,0.,0.,0.,
	    0.,0.,0.,0.,0.,0.,0.,0.,0.,1.17,1.17,1.17,1.17,1.17,1.17,1.17,
	    1.17,1.17,1.17,1.17,1.17,1.17,1.17 };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, k, ki, kn, mx, my;
    doublereal sx[7], sy[7], stx[7], sty[7];
    integer lenx, leny;
    doublereal mwpc[11];
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *);
    integer mwpn[11];
    doublereal mwps[11], mwpx[5], mwpy[5];
    integer ksize;
    doublereal copyx[5], copyy[5];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *);
    doublereal mwptx[55]	/* was [11][5] */, mwpty[55]	/* was [11][5]
	     */;
    integer mwpinx[11], mwpiny[11];
    doublereal mwpstx[5], mwpsty[5];

    /* Fortran I/O blocks */
    static cilist io___104 = { 0, 6, 0, 0, 0 };


/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     .. Executable Statements .. */

    for (ki = 1; ki <= 4; ++ki) {
	combla_1.incx = incxs[ki - 1];
	combla_1.incy = incys[ki - 1];
	mx = abs(combla_1.incx);
	my = abs(combla_1.incy);

	for (kn = 1; kn <= 4; ++kn) {
	    combla_1.n = ns[kn - 1];
	    ksize = min(2,kn);
	    lenx = lens[kn + (mx << 2) - 5];
	    leny = lens[kn + (my << 2) - 5];

	    if (combla_1.icase == 4) {
/*              .. DROT .. */
		for (i__ = 1; i__ <= 7; ++i__) {
		    sx[i__ - 1] = dx1[i__ - 1];
		    sy[i__ - 1] = dy1[i__ - 1];
		    stx[i__ - 1] = dt9x[i__ + (kn + (ki << 2)) * 7 - 36];
		    sty[i__ - 1] = dt9y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L20: */
		}
		drot_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy, &
			sc, &ss);
		stest_(&lenx, sx, stx, &ssize2[ksize * 14 - 14], sfac);
		stest_(&leny, sy, sty, &ssize2[ksize * 14 - 14], sfac);
	    } else {
		s_wsle(&io___104);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK3", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L40: */
	}
/* L60: */
    }

    mwpc[0] = 1.;
    for (i__ = 2; i__ <= 11; ++i__) {
	mwpc[i__ - 1] = 0.;
/* L80: */
    }
    mwps[0] = 0.;
    for (i__ = 2; i__ <= 6; ++i__) {
	mwps[i__ - 1] = 1.;
/* L100: */
    }
    for (i__ = 7; i__ <= 11; ++i__) {
	mwps[i__ - 1] = -1.;
/* L120: */
    }
    mwpinx[0] = 1;
    mwpinx[1] = 1;
    mwpinx[2] = 1;
    mwpinx[3] = -1;
    mwpinx[4] = 1;
    mwpinx[5] = -1;
    mwpinx[6] = 1;
    mwpinx[7] = 1;
    mwpinx[8] = -1;
    mwpinx[9] = 1;
    mwpinx[10] = -1;
    mwpiny[0] = 1;
    mwpiny[1] = 1;
    mwpiny[2] = -1;
    mwpiny[3] = -1;
    mwpiny[4] = 2;
    mwpiny[5] = 1;
    mwpiny[6] = 1;
    mwpiny[7] = -1;
    mwpiny[8] = -1;
    mwpiny[9] = 2;
    mwpiny[10] = 1;
    for (i__ = 1; i__ <= 11; ++i__) {
	mwpn[i__ - 1] = 5;
/* L140: */
    }
    mwpn[4] = 3;
    mwpn[9] = 3;
    for (i__ = 1; i__ <= 5; ++i__) {
	mwpx[i__ - 1] = (doublereal) i__;
	mwpy[i__ - 1] = (doublereal) i__;
	mwptx[i__ * 11 - 11] = (doublereal) i__;
	mwpty[i__ * 11 - 11] = (doublereal) i__;
	mwptx[i__ * 11 - 10] = (doublereal) i__;
	mwpty[i__ * 11 - 10] = (doublereal) (-i__);
	mwptx[i__ * 11 - 9] = (doublereal) (6 - i__);
	mwpty[i__ * 11 - 9] = (doublereal) (i__ - 6);
	mwptx[i__ * 11 - 8] = (doublereal) i__;
	mwpty[i__ * 11 - 8] = (doublereal) (-i__);
	mwptx[i__ * 11 - 6] = (doublereal) (6 - i__);
	mwpty[i__ * 11 - 6] = (doublereal) (i__ - 6);
	mwptx[i__ * 11 - 5] = (doublereal) (-i__);
	mwpty[i__ * 11 - 5] = (doublereal) i__;
	mwptx[i__ * 11 - 4] = (doublereal) (i__ - 6);
	mwpty[i__ * 11 - 4] = (doublereal) (6 - i__);
	mwptx[i__ * 11 - 3] = (doublereal) (-i__);
	mwpty[i__ * 11 - 3] = (doublereal) i__;
	mwptx[i__ * 11 - 1] = (doublereal) (i__ - 6);
	mwpty[i__ * 11 - 1] = (doublereal) (6 - i__);
/* L160: */
    }
    mwptx[4] = 1.;
    mwptx[15] = 3.;
    mwptx[26] = 5.;
    mwptx[37] = 4.;
    mwptx[48] = 5.;
    mwpty[4] = -1.;
    mwpty[15] = 2.;
    mwpty[26] = -2.;
    mwpty[37] = 4.;
    mwpty[48] = -3.;
    mwptx[9] = -1.;
    mwptx[20] = -3.;
    mwptx[31] = -5.;
    mwptx[42] = 4.;
    mwptx[53] = 5.;
    mwpty[9] = 1.;
    mwpty[20] = 2.;
    mwpty[31] = 2.;
    mwpty[42] = 4.;
    mwpty[53] = 3.;
    for (i__ = 1; i__ <= 11; ++i__) {
	combla_1.incx = mwpinx[i__ - 1];
	combla_1.incy = mwpiny[i__ - 1];
	for (k = 1; k <= 5; ++k) {
	    copyx[k - 1] = mwpx[k - 1];
	    copyy[k - 1] = mwpy[k - 1];
	    mwpstx[k - 1] = mwptx[i__ + k * 11 - 12];
	    mwpsty[k - 1] = mwpty[i__ + k * 11 - 12];
/* L180: */
	}
	drot_(&mwpn[i__ - 1], copyx, &combla_1.incx, copyy, &combla_1.incy, &
		mwpc[i__ - 1], &mwps[i__ - 1]);
	stest_(&c__5, copyx, mwpstx, mwpstx, sfac);
	stest_(&c__5, copyy, mwpsty, mwpsty, sfac);
/* L200: */
    }
    return 0;
} /* check3_ */

/* Subroutine */ int stest_(integer *len, doublereal *scomp, doublereal *
	strue, doublereal *ssize, doublereal *sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY  I                  "
	    "          \002,\002 COMP(I)                             TRUE(I) "
	    " DIFFERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,2i5,i3,2d36.8,2d12.4)";

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer i__;
    doublereal sd;
    extern double d_epsilon_(doublereal *);

    /* Fortran I/O blocks */
    static cilist io___121 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___122 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___123 = { 0, 6, 0, fmt_99997, 0 };


/*     ********************************* STEST ************************** */

/*     THIS SUBR COMPARES ARRAYS  SCOMP() AND STRUE() OF LENGTH LEN TO */
/*     SEE IF THE TERM BY TERM DIFFERENCES, MULTIPLIED BY SFAC, ARE */
/*     NEGLIGIBLE. */

/*     C. L. LAWSON, JPL, 1974 DEC 10 */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --ssize;
    --strue;
    --scomp;

    /* Function Body */
    i__1 = *len;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sd = scomp[i__] - strue[i__];
	if ((d__2 = *sfac * sd, abs(d__2)) <= (d__1 = ssize[i__], abs(d__1)) *
		 d_epsilon_(&c_b63)) {
	    goto L40;
	}

/*                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I). */

	if (! combla_1.pass) {
	    goto L20;
	}
/*                             PRINT FAIL MESSAGE AND HEADER. */
	combla_1.pass = FALSE_;
	s_wsfe(&io___121);
	e_wsfe();
	s_wsfe(&io___122);
	e_wsfe();
L20:
	s_wsfe(&io___123);
	do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&scomp[i__], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&strue[i__], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&sd, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ssize[i__], (ftnlen)sizeof(doublereal));
	e_wsfe();
L40:
	;
    }
    return 0;

} /* stest_ */

/* Subroutine */ int testdsdot_(real *scomp, real *strue, real *ssize, real *
	sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY                     "
	    "      \002,\002 COMP(I)                             TRUE(I)  DIF"
	    "FERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,1i5,i3,2e36.8,2e12.4)";

    /* System generated locals */
    real r__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    real sd;
    extern real s_epsilon_(real *);

    /* Fortran I/O blocks */
    static cilist io___125 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___126 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___127 = { 0, 6, 0, fmt_99997, 0 };


/*     ********************************* STEST ************************** */

/*     THIS SUBR COMPARES ARRAYS  SCOMP() AND STRUE() OF LENGTH LEN TO */
/*     SEE IF THE TERM BY TERM DIFFERENCES, MULTIPLIED BY SFAC, ARE */
/*     NEGLIGIBLE. */

/*     C. L. LAWSON, JPL, 1974 DEC 10 */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */

    sd = *scomp - *strue;
    if ((r__1 = *sfac * sd, abs(r__1)) <= abs(*ssize) * s_epsilon_(&c_b81)) {
	goto L40;
    }

/*                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I). */

    if (! combla_1.pass) {
	goto L20;
    }
/*                             PRINT FAIL MESSAGE AND HEADER. */
    combla_1.pass = FALSE_;
    s_wsfe(&io___125);
    e_wsfe();
    s_wsfe(&io___126);
    e_wsfe();
L20:
    s_wsfe(&io___127);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*scomp), (ftnlen)sizeof(real));
    do_fio(&c__1, (char *)&(*strue), (ftnlen)sizeof(real));
    do_fio(&c__1, (char *)&sd, (ftnlen)sizeof(real));
    do_fio(&c__1, (char *)&(*ssize), (ftnlen)sizeof(real));
    e_wsfe();
L40:
    return 0;

} /* testdsdot_ */

/* Subroutine */ int stest1_(doublereal *scomp1, doublereal *strue1,
	doublereal *ssize, doublereal *sfac)
{
    doublereal scomp[1], strue[1];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *);

/*     ************************* STEST1 ***************************** */

/*     THIS IS AN INTERFACE SUBROUTINE TO ACCOMODATE THE FORTRAN */
/*     REQUIREMENT THAT WHEN A DUMMY ARGUMENT IS AN ARRAY, THE */
/*     ACTUAL ARGUMENT MUST ALSO BE AN ARRAY OR AN ARRAY ELEMENT. */

/*     C.L. LAWSON, JPL, 1978 DEC 6 */

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --ssize;

    /* Function Body */
    scomp[0] = *scomp1;
    strue[0] = *strue1;
    stest_(&c__1, scomp, strue, &ssize[1], sfac);

    return 0;
} /* stest1_ */

doublereal sdiff_(doublereal *sa, doublereal *sb)
{
    /* System generated locals */
    doublereal ret_val;

/*     ********************************* SDIFF ************************** */
/*     COMPUTES DIFFERENCE OF TWO NUMBERS.  C. L. LAWSON, JPL 1974 FEB 15 */

/*     .. Scalar Arguments .. */
/*     .. Executable Statements .. */
    ret_val = *sa - *sb;
    return ret_val;
} /* sdiff_ */

/* Subroutine */ int itest1_(integer *icomp, integer *itrue)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY                     "
	    "          \002,\002 COMP                                TRUE    "
	    " DIFFERENCE\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,2i5,2i36,i12)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer id;

    /* Fortran I/O blocks */
    static cilist io___130 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___131 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___133 = { 0, 6, 0, fmt_99997, 0 };


/*     ********************************* ITEST1 ************************* */

/*     THIS SUBROUTINE COMPARES THE VARIABLES ICOMP AND ITRUE FOR */
/*     EQUALITY. */
/*     C. L. LAWSON, JPL, 1974 DEC 10 */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Local Scalars .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */

    if (*icomp == *itrue) {
	goto L40;
    }

/*                            HERE ICOMP IS NOT EQUAL TO ITRUE. */

    if (! combla_1.pass) {
	goto L20;
    }
/*                             PRINT FAIL MESSAGE AND HEADER. */
    combla_1.pass = FALSE_;
    s_wsfe(&io___130);
    e_wsfe();
    s_wsfe(&io___131);
    e_wsfe();
L20:
    id = *icomp - *itrue;
    s_wsfe(&io___133);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*icomp), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*itrue), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&id, (ftnlen)sizeof(integer));
    e_wsfe();
L40:
    return 0;

} /* itest1_ */

/* Main program alias */ int dblat1_ () { main (); return 0; }
