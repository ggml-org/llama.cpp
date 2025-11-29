/* sblat1.f -- translated by f2c (version 20100827).
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
static real c_b35 = 1.f;
static real c_b39 = .1f;
static integer c__5 = 5;
static real c_b63 = 0.f;

/* > \brief \b SBLAT1 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM SBLAT1 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    Test program for the REAL Level 1 BLAS. */
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

/* > \ingroup single_blas_testing */

/*  ===================================================================== */
/* Main program */ int main(void)
{
#ifdef BLIS_ENABLE_HPX
    char* program = "sblat1";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static real sfac = 9.765625e-4f;

    /* Format strings */
    static char fmt_99999[] = "(\002 Real BLAS Test Program Results\002,/1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer ic;
    extern /* Subroutine */ int check0_(real *), check1_(real *), check2_(
	    real *), check3_(real *), header_(void);

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

    static char l[6*13] = " SDOT " "SAXPY " "SROTG " " SROT " "SCOPY " "SSWA"
	    "P " "SNRM2 " "SASUM " "SSCAL " "ISAMAX" "SROTMG" "SROTM " "SDSDOT"
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

/* Subroutine */ int check0_(real *sfac)
{
    /* Initialized data */

    static real ds1[8] = { .8f,.6f,.8f,-.6f,.8f,0.f,1.f,0.f };
    static real datrue[8] = { .5f,.5f,.5f,-.5f,-.5f,0.f,1.f,1.f };
    static real dbtrue[8] = { 0.f,.6f,0.f,-.6f,0.f,0.f,1.f,0.f };
    static real dab[36]	/* was [4][9] */ = { .1f,.3f,1.2f,.2f,.7f,.2f,.6f,
	    4.2f,0.f,0.f,0.f,0.f,4.f,-1.f,2.f,4.f,6e-10f,.02f,1e5f,10.f,4e10f,
	    .02f,1e-5f,10.f,2e-10f,.04f,1e5f,10.f,2e10f,.04f,1e-5f,10.f,4.f,
	    -2.f,8.f,4.f };
    static real dtrue[81]	/* was [9][9] */ = { 0.f,0.f,1.3f,.2f,0.f,0.f,
	    0.f,.5f,0.f,0.f,0.f,4.5f,4.2f,1.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
	    -2.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,4.f,-1.f,0.f,0.f,0.f,0.f,0.f,
	    .015f,0.f,10.f,-1.f,0.f,-1e-4f,0.f,1.f,0.f,0.f,.06144f,10.f,-1.f,
	    4096.f,-1e6f,0.f,1.f,0.f,0.f,15.f,10.f,-1.f,5e-5f,0.f,1.f,0.f,0.f,
	    0.f,15.f,10.f,-1.f,5e5f,-4096.f,1.f,.004096f,0.f,0.f,7.f,4.f,0.f,
	    0.f,-.5f,-.25f,0.f };
    static real d12 = 4096.f;
    static real da1[8] = { .3f,.4f,-.3f,-.4f,-.3f,0.f,0.f,1.f };
    static real db1[8] = { .4f,.3f,.4f,.3f,-.4f,0.f,1.f,0.f };
    static real dc1[8] = { .6f,.8f,-.6f,.8f,.6f,1.f,0.f,1.f };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, k;
    real sa, sb, sc, ss, dtemp[9];
    extern /* Subroutine */ int srotg_(real *, real *, real *, real *),
	    stest_(integer *, real *, real *, real *, real *), stest1_(real *,
	     real *, real *, real *), srotmg_(real *, real *, real *, real *,
	    real *);

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
    dtrue[0] = .092307692307692313f;
    dtrue[1] = .27692307692307694f;
    dtrue[6] = -.16666666666666666f;
    dtrue[9] = .18666666666666668f;
    dtrue[10] = .65333333333333332f;
    dtrue[17] = .14285714285714285f;
    dtrue[36] = d12 * d12 * 4.5e-10f;
    dtrue[38] = 4e5f / (d12 * 3.f);
    dtrue[41] = 1.f / d12;
    dtrue[43] = 1e4f / (d12 * 3.f);
    dtrue[45] = 4e10f / (d12 * 1.5f * d12);
    dtrue[46] = .013333333333333334f;
    dtrue[52] = d12 * 5e-7f;
    dtrue[54] = .026666666666666668f;
    dtrue[55] = d12 * d12 * 1.3333333333333334e-10f;
    dtrue[60] = -dtrue[41];
    dtrue[62] = 1e4f / d12;
    dtrue[63] = dtrue[54];
    dtrue[64] = 2e10f / (d12 * 1.5f * d12);
    dtrue[72] = 4.5714285714285712f;
    dtrue[73] = -2.2857142857142856f;
/*     .. Executable Statements .. */

/*     Compute true values which cannot be prestored */
/*     in decimal notation */

    dbtrue[0] = 1.6666666666666667f;
    dbtrue[2] = -1.6666666666666667f;
    dbtrue[4] = 1.6666666666666667f;

    for (k = 1; k <= 8; ++k) {
/*        .. Set N=K for identification in output if any .. */
	combla_1.n = k;
	if (combla_1.icase == 3) {
/*           .. SROTG .. */
	    if (k > 8) {
		goto L40;
	    }
	    sa = da1[k - 1];
	    sb = db1[k - 1];
	    srotg_(&sa, &sb, &sc, &ss);
	    stest1_(&sa, &datrue[k - 1], &datrue[k - 1], sfac);
	    stest1_(&sb, &dbtrue[k - 1], &dbtrue[k - 1], sfac);
	    stest1_(&sc, &dc1[k - 1], &dc1[k - 1], sfac);
	    stest1_(&ss, &ds1[k - 1], &ds1[k - 1], sfac);
	} else if (combla_1.icase == 11) {
/*           .. SROTMG .. */
	    for (i__ = 1; i__ <= 4; ++i__) {
		dtemp[i__ - 1] = dab[i__ + (k << 2) - 5];
		dtemp[i__ + 3] = 0.f;
	    }
	    dtemp[8] = 0.f;
	    srotmg_(dtemp, &dtemp[1], &dtemp[2], &dtemp[3], &dtemp[4]);
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

/* Subroutine */ int check1_(real *sfac)
{
    /* Initialized data */

    static real sa[10] = { .3f,-1.f,0.f,1.f,.3f,.3f,.3f,.3f,.3f,.3f };
    static real dv[80]	/* was [8][5][2] */ = { .1f,2.f,2.f,2.f,2.f,2.f,2.f,
	    2.f,.3f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,.3f,-.4f,4.f,4.f,4.f,4.f,4.f,
	    4.f,.2f,-.6f,.3f,5.f,5.f,5.f,5.f,5.f,.1f,-.3f,.5f,-.1f,6.f,6.f,
	    6.f,6.f,.1f,8.f,8.f,8.f,8.f,8.f,8.f,8.f,.3f,9.f,9.f,9.f,9.f,9.f,
	    9.f,9.f,.3f,2.f,-.4f,2.f,2.f,2.f,2.f,2.f,.2f,3.f,-.6f,5.f,.3f,2.f,
	    2.f,2.f,.1f,4.f,-.3f,6.f,-.5f,7.f,-.1f,3.f };
    static real dtrue1[5] = { 0.f,.3f,.5f,.7f,.6f };
    static real dtrue3[5] = { 0.f,.3f,.7f,1.1f,1.f };
    static real dtrue5[80]	/* was [8][5][2] */ = { .1f,2.f,2.f,2.f,2.f,
	    2.f,2.f,2.f,-.3f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,0.f,0.f,4.f,4.f,4.f,
	    4.f,4.f,4.f,.2f,-.6f,.3f,5.f,5.f,5.f,5.f,5.f,.03f,-.09f,.15f,
	    -.03f,6.f,6.f,6.f,6.f,.1f,8.f,8.f,8.f,8.f,8.f,8.f,8.f,.09f,9.f,
	    9.f,9.f,9.f,9.f,9.f,9.f,.09f,2.f,-.12f,2.f,2.f,2.f,2.f,2.f,.06f,
	    3.f,-.18f,5.f,.09f,2.f,2.f,2.f,.03f,4.f,-.09f,6.f,-.15f,7.f,-.03f,
	    3.f };
    static integer itrue2[5] = { 0,1,2,2,3 };

    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__;
    real sx[8];
    integer np1, len;
    extern real snrm2_(integer *, real *, integer *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    real stemp[1];
    extern real sasum_(integer *, real *, integer *);
    real strue[8];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *,
	    real *), itest1_(integer *, integer *), stest1_(real *, real *,
	    real *, real *);
    extern integer isamax_(integer *, real *, integer *);

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
/*              .. SNRM2 .. */
		stemp[0] = dtrue1[np1 - 1];
		r__1 = snrm2_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&r__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 8) {
/*              .. SASUM .. */
		stemp[0] = dtrue3[np1 - 1];
		r__1 = sasum_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&r__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 9) {
/*              .. SSCAL .. */
		sscal_(&combla_1.n, &sa[(combla_1.incx - 1) * 5 + np1 - 1],
			sx, &combla_1.incx);
		i__1 = len;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    strue[i__ - 1] = dtrue5[i__ + (np1 + combla_1.incx * 5 <<
			    3) - 49];
/* L40: */
		}
		stest_(&len, sx, strue, strue, sfac);
	    } else if (combla_1.icase == 10) {
/*              .. ISAMAX .. */
		i__1 = isamax_(&combla_1.n, sx, &combla_1.incx);
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

/* Subroutine */ int check2_(real *sfac)
{
    /* Initialized data */

    static real sa = .3f;
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static real dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    static real dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    static real dt7[16]	/* was [4][4] */ = { 0.f,.3f,.21f,.62f,0.f,.3f,-.07f,
	    .85f,0.f,.3f,-.79f,-.74f,0.f,.3f,.33f,1.27f };
    static real st7b[16]	/* was [4][4] */ = { .1f,.4f,.31f,.72f,.1f,
	    .4f,.03f,.95f,.1f,.4f,-.69f,-.64f,.1f,.4f,.43f,1.37f };
    static real dt8[112]	/* was [7][4][4] */ = { .5f,0.f,0.f,0.f,0.f,
	    0.f,0.f,.68f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,-.87f,0.f,0.f,0.f,0.f,
	    0.f,.68f,-.87f,.15f,.94f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,
	    .68f,0.f,0.f,0.f,0.f,0.f,0.f,.35f,-.9f,.48f,0.f,0.f,0.f,0.f,.38f,
	    -.9f,.57f,.7f,-.75f,.2f,.98f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,0.f,
	    0.f,0.f,0.f,0.f,0.f,.35f,-.72f,0.f,0.f,0.f,0.f,0.f,.38f,-.63f,
	    .15f,.88f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,0.f,0.f,
	    0.f,0.f,0.f,0.f,.68f,-.9f,.33f,0.f,0.f,0.f,0.f,.68f,-.9f,.33f,.7f,
	    -.75f,.2f,1.04f };
    static real dt10x[112]	/* was [7][4][4] */ = { .6f,0.f,0.f,0.f,0.f,
	    0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,-.9f,0.f,0.f,0.f,0.f,0.f,
	    .5f,-.9f,.3f,.7f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,
	    0.f,0.f,0.f,0.f,0.f,.3f,.1f,.5f,0.f,0.f,0.f,0.f,.8f,.1f,-.6f,.8f,
	    .3f,-.3f,.5f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,
	    0.f,-.9f,.1f,.5f,0.f,0.f,0.f,0.f,.7f,.1f,.3f,.8f,-.9f,-.3f,.5f,
	    .6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,.3f,
	    0.f,0.f,0.f,0.f,0.f,.5f,.3f,-.6f,.8f,0.f,0.f,0.f };
    static real dt10y[112]	/* was [7][4][4] */ = { .5f,0.f,0.f,0.f,0.f,
	    0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,.1f,0.f,0.f,0.f,0.f,0.f,
	    .6f,.1f,-.5f,.8f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,
	    0.f,0.f,0.f,0.f,0.f,-.5f,-.9f,.6f,0.f,0.f,0.f,0.f,-.4f,-.9f,.9f,
	    .7f,-.5f,.2f,.6f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,
	    0.f,0.f,-.5f,.6f,0.f,0.f,0.f,0.f,0.f,-.4f,.9f,-.5f,.6f,0.f,0.f,
	    0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,
	    -.9f,.1f,0.f,0.f,0.f,0.f,.6f,-.9f,.1f,.7f,-.5f,.2f,.8f };
    static real ssize1[4] = { 0.f,.3f,1.6f,3.2f };
    static real ssize2[28]	/* was [14][2] */ = { 0.f,0.f,0.f,0.f,0.f,0.f,
	    0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.17f,1.17f,1.17f,1.17f,1.17f,
	    1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f };
    static real ssize3[4] = { .1f,.4f,1.7f,3.3f };
    static real dpar[20]	/* was [5][4] */ = { -2.f,0.f,0.f,0.f,0.f,
	    -1.f,2.f,-3.f,-4.f,5.f,0.f,0.f,2.f,-3.f,0.f,1.f,5.f,2.f,0.f,-4.f }
	    ;
    static struct {
	real e_1[448];
	} equiv_3 = {{ .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		-.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -.9f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, 3.5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, .1f, 0.f,
		0.f, 0.f, 0.f, 0.f, -.8f, 3.8f, 0.f, 0.f, 0.f, 0.f, 0.f, -.9f,
		 2.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.5f, -.4f, 0.f, 0.f, 0.f,
		0.f, 0.f, .6f, .1f, -.5f, .8f, 0.f, 0.f, 0.f, -.8f, 3.8f,
		-2.2f, -1.2f, 0.f, 0.f, 0.f, -.9f, 2.8f, -1.4f, -1.3f, 0.f,
		0.f, 0.f, 3.5f, -.4f, -2.2f, 4.7f, 0.f, 0.f, 0.f, .6f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		.6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -.8f, 0.f, 0.f,
		0.f, 0.f, 0.f, 0.f, -.9f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.5f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, .1f, -.5f, 0.f, 0.f, 0.f,
		0.f, 0.f, .1f, -3.f, 0.f, 0.f, 0.f, 0.f, -.3f, .1f, -2.f, 0.f,
		 0.f, 0.f, 0.f, 3.3f, .1f, -2.f, 0.f, 0.f, 0.f, 0.f, .6f, .1f,
		 -.5f, .8f, .9f, -.3f, -.4f, -2.f, .1f, 1.4f, .8f, .6f, -.3f,
		-2.8f, -1.8f, .1f, 1.3f, .8f, 0.f, -.3f, -1.9f, 3.8f, .1f,
		-3.1f, .8f, 4.8f, -.3f, -1.5f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, -.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		-.9f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.5f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, .6f, .1f, -.5f, 0.f, 0.f, 0.f, 0.f, 4.8f, .1f, -3.f,
		 0.f, 0.f, 0.f, 0.f, 3.3f, .1f, -2.f, 0.f, 0.f, 0.f, 0.f,
		2.1f, .1f, -2.f, 0.f, 0.f, 0.f, 0.f, .6f, .1f, -.5f, .8f, .9f,
		 -.3f, -.4f, -1.6f, .1f, -2.2f, .8f, 5.4f, -.3f, -2.8f, -1.5f,
		 .1f, -1.4f, .8f, 3.6f, -.3f, -1.9f, 3.7f, .1f, -2.2f, .8f,
		3.6f, -.3f, -1.5f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .6f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, -.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -.9f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, 3.5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		.6f, .1f, 0.f, 0.f, 0.f, 0.f, 0.f, -.8f, -1.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -.9f, -.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.5f, .8f, 0.f,
		 0.f, 0.f, 0.f, 0.f, .6f, .1f, -.5f, .8f, 0.f, 0.f, 0.f, -.8f,
		 -1.f, 1.4f, -1.6f, 0.f, 0.f, 0.f, -.9f, -.8f, 1.3f, -1.6f,
		0.f, 0.f, 0.f, 3.5f, .8f, -3.1f, 4.8f, 0.f, 0.f, 0.f }};

    static struct {
	real e_1[448];
	} equiv_7 = {{ .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		.7f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.7f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, -2.6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, -.9f, 0.f,
		 0.f, 0.f, 0.f, 0.f, .7f, -4.8f, 0.f, 0.f, 0.f, 0.f, 0.f,
		1.7f, -.7f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.6f, 3.5f, 0.f, 0.f,
		0.f, 0.f, 0.f, .5f, -.9f, .3f, .7f, 0.f, 0.f, 0.f, .7f, -4.8f,
		 3.f, 1.1f, 0.f, 0.f, 0.f, 1.7f, -.7f, -.7f, 2.3f, 0.f, 0.f,
		0.f, -2.6f, 3.5f, -.7f, -3.6f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f,
		0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .7f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, 1.7f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.6f, 0.f,
		 0.f, 0.f, 0.f, 0.f, 0.f, .5f, -.9f, .3f, 0.f, 0.f, 0.f, 0.f,
		4.f, -.9f, -.3f, 0.f, 0.f, 0.f, 0.f, -.5f, -.9f, 1.5f, 0.f,
		0.f, 0.f, 0.f, -1.5f, -.9f, -1.8f, 0.f, 0.f, 0.f, 0.f, .5f,
		-.9f, .3f, .7f, -.6f, .2f, .8f, 3.7f, -.9f, -1.2f, .7f, -1.5f,
		 .2f, 2.2f, -.3f, -.9f, 2.1f, .7f, -1.6f, .2f, 2.f, -1.6f,
		-.9f, -2.1f, .7f, 2.9f, .2f, -3.8f, .5f, 0.f, 0.f, 0.f, 0.f,
		0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f,
		0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .7f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, 1.7f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -2.6f, 0.f, 0.f, 0.f,
		 0.f, 0.f, 0.f, .5f, -.9f, 0.f, 0.f, 0.f, 0.f, 0.f, 4.f,
		-6.3f, 0.f, 0.f, 0.f, 0.f, 0.f, -.5f, .3f, 0.f, 0.f, 0.f, 0.f,
		 0.f, -1.5f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, -.9f, .3f,
		.7f, 0.f, 0.f, 0.f, 3.7f, -7.2f, 3.f, 1.7f, 0.f, 0.f, 0.f,
		-.3f, .9f, -.7f, 1.9f, 0.f, 0.f, 0.f, -1.6f, 2.7f, -.7f,
		-3.4f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f,
		0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f,
		0.f, .5f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, .5f, 0.f, 0.f, 0.f,
		0.f, 0.f, 0.f, .7f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.7f, 0.f,
		0.f, 0.f, 0.f, 0.f, 0.f, -2.6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
		.5f, -.9f, .3f, 0.f, 0.f, 0.f, 0.f, .7f, -.9f, 1.2f, 0.f, 0.f,
		 0.f, 0.f, 1.7f, -.9f, .5f, 0.f, 0.f, 0.f, 0.f, -2.6f, -.9f,
		-1.3f, 0.f, 0.f, 0.f, 0.f, .5f, -.9f, .3f, .7f, -.6f, .2f,
		.8f, .7f, -.9f, 1.2f, .7f, -1.5f, .2f, 1.6f, 1.7f, -.9f, .5f,
		.7f, -1.6f, .2f, 2.4f, -2.6f, -.9f, -1.3f, .7f, 2.9f, .2f,
		-4.f }};


    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, j, ki, kn, mx, my;
    real sx[7], sy[7];
    integer kni;
    real stx[7], sty[7];
    integer kpar, lenx, leny;
#define dt19x ((real *)&equiv_3)
#define dt19y ((real *)&equiv_7)
    extern real sdot_(integer *, real *, integer *, real *, integer *);
    real dtemp[5];
#define dt19xa ((real *)&equiv_3)
#define dt19xb ((real *)&equiv_3 + 112)
#define dt19xc ((real *)&equiv_3 + 224)
#define dt19xd ((real *)&equiv_3 + 336)
#define dt19ya ((real *)&equiv_7)
#define dt19yb ((real *)&equiv_7 + 112)
#define dt19yc ((real *)&equiv_7 + 224)
#define dt19yd ((real *)&equiv_7 + 336)
    integer ksize;
    real ssize[7];
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), sswap_(integer *, real *, integer *, real *, integer *
	    ), stest_(integer *, real *, real *, real *, real *), saxpy_(
	    integer *, real *, real *, integer *, real *, integer *), srotm_(
	    integer *, real *, integer *, real *, integer *, real *), stest1_(
	    real *, real *, real *, real *);
    extern real sdsdot_(integer *, real *, real *, integer *, real *, integer
	    *);

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
/* L20: */
	    }

	    if (combla_1.icase == 1) {
/*              .. SDOT .. */
		r__1 = sdot_(&combla_1.n, sx, &combla_1.incx, sy, &
			combla_1.incy);
		stest1_(&r__1, &dt7[kn + (ki << 2) - 5], &ssize1[kn - 1],
			sfac);
	    } else if (combla_1.icase == 2) {
/*              .. SAXPY .. */
		saxpy_(&combla_1.n, &sa, sx, &combla_1.incx, sy, &
			combla_1.incy);
		i__1 = leny;
		for (j = 1; j <= i__1; ++j) {
		    sty[j - 1] = dt8[j + (kn + (ki << 2)) * 7 - 36];
/* L40: */
		}
		stest_(&leny, sy, sty, &ssize2[ksize * 14 - 14], sfac);
	    } else if (combla_1.icase == 5) {
/*              .. SCOPY .. */
		for (i__ = 1; i__ <= 7; ++i__) {
		    sty[i__ - 1] = dt10y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L60: */
		}
		scopy_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		stest_(&leny, sy, sty, ssize2, &c_b35);
	    } else if (combla_1.icase == 6) {
/*              .. SSWAP .. */
		sswap_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		for (i__ = 1; i__ <= 7; ++i__) {
		    stx[i__ - 1] = dt10x[i__ + (kn + (ki << 2)) * 7 - 36];
		    sty[i__ - 1] = dt10y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L80: */
		}
		stest_(&lenx, sx, stx, ssize2, &c_b35);
		stest_(&leny, sy, sty, ssize2, &c_b35);
	    } else if (combla_1.icase == 12) {
/*              .. SROTM .. */
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
			ssize[0] = 2.4f;
		    }
		    if (kpar == 3 && kni == 8) {
			ssize[4] = 1.8f;
		    }

		    srotm_(&combla_1.n, sx, &combla_1.incx, sy, &
			    combla_1.incy, dtemp);
		    stest_(&lenx, sx, stx, ssize, sfac);
		    stest_(&leny, sy, sty, sty, sfac);
		}
	    } else if (combla_1.icase == 13) {
/*              .. SDSROT .. */
		r__1 = sdsdot_(&combla_1.n, &c_b39, sx, &combla_1.incx, sy, &
			combla_1.incy);
		stest1_(&r__1, &st7b[kn + (ki << 2) - 5], &ssize3[kn - 1],
			sfac);
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


/* Subroutine */ int check3_(real *sfac)
{
    /* Initialized data */

    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static real dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    static real dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    static real sc = .8f;
    static real ss = .6f;
    static real dt9x[112]	/* was [7][4][4] */ = { .6f,0.f,0.f,0.f,0.f,
	    0.f,0.f,.78f,0.f,0.f,0.f,0.f,0.f,0.f,.78f,-.46f,0.f,0.f,0.f,0.f,
	    0.f,.78f,-.46f,-.22f,1.06f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,
	    0.f,.78f,0.f,0.f,0.f,0.f,0.f,0.f,.66f,.1f,-.1f,0.f,0.f,0.f,0.f,
	    .96f,.1f,-.76f,.8f,.9f,-.3f,-.02f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,
	    .78f,0.f,0.f,0.f,0.f,0.f,0.f,-.06f,.1f,-.1f,0.f,0.f,0.f,0.f,.9f,
	    .1f,-.22f,.8f,.18f,-.3f,-.02f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.78f,
	    0.f,0.f,0.f,0.f,0.f,0.f,.78f,.26f,0.f,0.f,0.f,0.f,0.f,.78f,.26f,
	    -.76f,1.12f,0.f,0.f,0.f };
    static real dt9y[112]	/* was [7][4][4] */ = { .5f,0.f,0.f,0.f,0.f,
	    0.f,0.f,.04f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,-.78f,0.f,0.f,0.f,0.f,
	    0.f,.04f,-.78f,.54f,.08f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,
	    .04f,0.f,0.f,0.f,0.f,0.f,0.f,.7f,-.9f,-.12f,0.f,0.f,0.f,0.f,.64f,
	    -.9f,-.3f,.7f,-.18f,.2f,.28f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,0.f,
	    0.f,0.f,0.f,0.f,0.f,.7f,-1.08f,0.f,0.f,0.f,0.f,0.f,.64f,-1.26f,
	    .54f,.2f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,0.f,0.f,0.f,
	    0.f,0.f,0.f,.04f,-.9f,.18f,0.f,0.f,0.f,0.f,.04f,-.9f,.18f,.7f,
	    -.18f,.2f,.16f };
    static real ssize2[28]	/* was [14][2] */ = { 0.f,0.f,0.f,0.f,0.f,0.f,
	    0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.17f,1.17f,1.17f,1.17f,1.17f,
	    1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f,1.17f };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, k, ki, kn, mx, my;
    real sx[7], sy[7], stx[7], sty[7];
    integer lenx, leny;
    real mwpc[11];
    integer mwpn[11];
    real mwps[11];
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *);
    real mwpx[5], mwpy[5];
    integer ksize;
    real copyx[5], copyy[5];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *,
	    real *);
    real mwptx[55]	/* was [11][5] */, mwpty[55]	/* was [11][5] */;
    integer mwpinx[11], mwpiny[11];
    real mwpstx[5], mwpsty[5];

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
/*              .. SROT .. */
		for (i__ = 1; i__ <= 7; ++i__) {
		    sx[i__ - 1] = dx1[i__ - 1];
		    sy[i__ - 1] = dy1[i__ - 1];
		    stx[i__ - 1] = dt9x[i__ + (kn + (ki << 2)) * 7 - 36];
		    sty[i__ - 1] = dt9y[i__ + (kn + (ki << 2)) * 7 - 36];
/* L20: */
		}
		srot_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy, &
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

    mwpc[0] = 1.f;
    for (i__ = 2; i__ <= 11; ++i__) {
	mwpc[i__ - 1] = 0.f;
/* L80: */
    }
    mwps[0] = 0.f;
    for (i__ = 2; i__ <= 6; ++i__) {
	mwps[i__ - 1] = 1.f;
/* L100: */
    }
    for (i__ = 7; i__ <= 11; ++i__) {
	mwps[i__ - 1] = -1.f;
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
	mwpx[i__ - 1] = (real) i__;
	mwpy[i__ - 1] = (real) i__;
	mwptx[i__ * 11 - 11] = (real) i__;
	mwpty[i__ * 11 - 11] = (real) i__;
	mwptx[i__ * 11 - 10] = (real) i__;
	mwpty[i__ * 11 - 10] = (real) (-i__);
	mwptx[i__ * 11 - 9] = (real) (6 - i__);
	mwpty[i__ * 11 - 9] = (real) (i__ - 6);
	mwptx[i__ * 11 - 8] = (real) i__;
	mwpty[i__ * 11 - 8] = (real) (-i__);
	mwptx[i__ * 11 - 6] = (real) (6 - i__);
	mwpty[i__ * 11 - 6] = (real) (i__ - 6);
	mwptx[i__ * 11 - 5] = (real) (-i__);
	mwpty[i__ * 11 - 5] = (real) i__;
	mwptx[i__ * 11 - 4] = (real) (i__ - 6);
	mwpty[i__ * 11 - 4] = (real) (6 - i__);
	mwptx[i__ * 11 - 3] = (real) (-i__);
	mwpty[i__ * 11 - 3] = (real) i__;
	mwptx[i__ * 11 - 1] = (real) (i__ - 6);
	mwpty[i__ * 11 - 1] = (real) (6 - i__);
/* L160: */
    }
    mwptx[4] = 1.f;
    mwptx[15] = 3.f;
    mwptx[26] = 5.f;
    mwptx[37] = 4.f;
    mwptx[48] = 5.f;
    mwpty[4] = -1.f;
    mwpty[15] = 2.f;
    mwpty[26] = -2.f;
    mwpty[37] = 4.f;
    mwpty[48] = -3.f;
    mwptx[9] = -1.f;
    mwptx[20] = -3.f;
    mwptx[31] = -5.f;
    mwptx[42] = 4.f;
    mwptx[53] = 5.f;
    mwpty[9] = 1.f;
    mwpty[20] = 2.f;
    mwpty[31] = 2.f;
    mwpty[42] = 4.f;
    mwpty[53] = 3.f;
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
	srot_(&mwpn[i__ - 1], copyx, &combla_1.incx, copyy, &combla_1.incy, &
		mwpc[i__ - 1], &mwps[i__ - 1]);
	stest_(&c__5, copyx, mwpstx, mwpstx, sfac);
	stest_(&c__5, copyy, mwpsty, mwpsty, sfac);
/* L200: */
    }
    return 0;
} /* check3_ */

/* Subroutine */ int stest_(integer *len, real *scomp, real *strue, real *
	ssize, real *sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY  I                  "
	    "          \002,\002 COMP(I)                             TRUE(I) "
	    " DIFFERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,2i5,i3,2e36.8,2e12.4)";

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer i__;
    real sd;
    extern real s_epsilon_(real *);

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
	if ((r__2 = *sfac * sd, abs(r__2)) <= (r__1 = ssize[i__], abs(r__1)) *
		 s_epsilon_(&c_b63)) {
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
	do_fio(&c__1, (char *)&scomp[i__], (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&strue[i__], (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&sd, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&ssize[i__], (ftnlen)sizeof(real));
	e_wsfe();
L40:
	;
    }
    return 0;

} /* stest_ */

/* Subroutine */ int stest1_(real *scomp1, real *strue1, real *ssize, real *
	sfac)
{
    real scomp[1], strue[1];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *,
	    real *);

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

real sdiff_(real *sa, real *sb)
{
    /* System generated locals */
    real ret_val;

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
    static cilist io___126 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___127 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___129 = { 0, 6, 0, fmt_99997, 0 };


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
    s_wsfe(&io___126);
    e_wsfe();
    s_wsfe(&io___127);
    e_wsfe();
L20:
    id = *icomp - *itrue;
    s_wsfe(&io___129);
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

/* Main program alias */ int sblat1_ () { main (); return 0; }
