/* cblat1.f -- translated by f2c (version 20100827).
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
    integer icase, n, incx, incy, mode;
    logical pass;
} combla_;

#define combla_1 combla_

/* Table of constant values */

static integer c__1 = 1;
static integer c__9 = 9;
static integer c__5 = 5;
static real c_b43 = 1.f;
static real c_b52 = 0.f;

/* > \brief \b CBLAT1 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM CBLAT1 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    Test program for the COMPLEX Level 1 BLAS. */
/* >    Based upon the original BLAS test routine together with: */
/* > */
/* >    F06GAF Example Program Text */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date April 2012 */

/* > \ingroup complex_blas_testing */

/*  ===================================================================== */
/* Main program */ int main(void)
{
#ifdef BLIS_ENABLE_HPX
    char* program = "cblat1";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static real sfac = 9.765625e-4f;

    /* Format strings */
    static char fmt_99999[] = "(\002 Complex BLAS Test Program Results\002,/"
	    "1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer ic;
    extern /* Subroutine */ int check1_(real *), check2_(real *), header_(
	    void);

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
    for (ic = 1; ic <= 10; ++ic) {
	combla_1.icase = ic;
	header_();

/*        Initialize PASS, INCX, INCY, and MODE for a new case. */
/*        The value 9999 for INCX, INCY or MODE will appear in the */
/*        detailed  output, if any, for cases that do not involve */
/*        these parameters. */

	combla_1.pass = TRUE_;
	combla_1.incx = 9999;
	combla_1.incy = 9999;
	combla_1.mode = 9999;
	if (combla_1.icase <= 5) {
	    check2_(&sfac);
	} else if (combla_1.icase >= 6) {
	    check1_(&sfac);
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

    static char l[6*10] = "CDOTC " "CDOTU " "CAXPY " "CCOPY " "CSWAP " "SCNR"
	    "M2" "SCASUM" "CSCAL " "CSSCAL" "ICAMAX";

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

/* Subroutine */ int check1_(real *sfac)
{
    /* Initialized data */

    static real strue2[5] = { 0.f,.5f,.6f,.7f,.8f };
    static real strue4[5] = { 0.f,.7f,1.f,1.3f,1.6f };
    static complex ctrue5[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{-.16f,
	    -.37f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f}
	    ,{3.f,4.f},{-.17f,-.19f},{.13f,-.39f},{5.f,6.f},{5.f,6.f},{5.f,
	    6.f},{5.f,6.f},{5.f,6.f},{5.f,6.f},{.11f,-.03f},{-.17f,.46f},{
	    -.17f,-.19f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{
	    .19f,-.17f},{.2f,-.35f},{.35f,.2f},{.14f,.08f},{2.f,3.f},{2.f,3.f}
	    ,{2.f,3.f},{2.f,3.f},{.1f,.1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,
	    5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{-.16f,-.37f},{6.f,7.f},{6.f,
	    7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{-.17f,
	    -.19f},{8.f,9.f},{.13f,-.39f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,
	    5.f},{2.f,5.f},{.11f,-.03f},{3.f,6.f},{-.17f,.46f},{4.f,7.f},{
	    -.17f,-.19f},{7.f,2.f},{7.f,2.f},{7.f,2.f},{.19f,-.17f},{5.f,8.f},
	    {.2f,-.35f},{6.f,9.f},{.35f,.2f},{8.f,3.f},{.14f,.08f},{9.f,4.f} }
	    ;
    static complex ctrue6[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{.09f,
	    -.12f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f}
	    ,{3.f,4.f},{.03f,-.09f},{.15f,-.03f},{5.f,6.f},{5.f,6.f},{5.f,6.f}
	    ,{5.f,6.f},{5.f,6.f},{5.f,6.f},{.03f,.03f},{-.18f,.03f},{.03f,
	    -.09f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{.09f,
	    .03f},{.15f,0.f},{0.f,.15f},{0.f,.06f},{2.f,3.f},{2.f,3.f},{2.f,
	    3.f},{2.f,3.f},{.1f,.1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{
	    4.f,5.f},{4.f,5.f},{4.f,5.f},{.09f,-.12f},{6.f,7.f},{6.f,7.f},{
	    6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{.03f,-.09f},{
	    8.f,9.f},{.15f,-.03f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{
	    2.f,5.f},{.03f,.03f},{3.f,6.f},{-.18f,.03f},{4.f,7.f},{.03f,-.09f}
	    ,{7.f,2.f},{7.f,2.f},{7.f,2.f},{.09f,.03f},{5.f,8.f},{.15f,0.f},{
	    6.f,9.f},{0.f,.15f},{8.f,3.f},{0.f,.06f},{9.f,4.f} };
    static integer itrue3[5] = { 0,1,2,2,2 };
    static real sa = .3f;
    static complex ca = {.4f,-.7f};
    static complex cv[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{.3f,
	    -.4f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},
	    {3.f,4.f},{.1f,-.3f},{.5f,-.1f},{5.f,6.f},{5.f,6.f},{5.f,6.f},{
	    5.f,6.f},{5.f,6.f},{5.f,6.f},{.1f,.1f},{-.6f,.1f},{.1f,-.3f},{7.f,
	    8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{.3f,.1f},{.5f,0.f},{
	    0.f,.5f},{0.f,.2f},{2.f,3.f},{2.f,3.f},{2.f,3.f},{2.f,3.f},{.1f,
	    .1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{
	    4.f,5.f},{.3f,-.4f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,
	    7.f},{6.f,7.f},{6.f,7.f},{.1f,-.3f},{8.f,9.f},{.5f,-.1f},{2.f,5.f}
	    ,{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{.1f,.1f},{3.f,6.f},{
	    -.6f,.1f},{4.f,7.f},{.1f,-.3f},{7.f,2.f},{7.f,2.f},{7.f,2.f},{.3f,
	    .1f},{5.f,8.f},{.5f,0.f},{6.f,9.f},{0.f,.5f},{8.f,3.f},{0.f,.2f},{
	    9.f,4.f} };

    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1;
    complex q__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__;
    complex cx[8];
    integer np1, len;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), ctest_(integer *, complex *, complex *, complex *,
	    real *);
    complex mwpcs[5], mwpct[5];
    extern real scnrm2_(integer *, complex *, integer *);
    extern /* Subroutine */ int itest1_(integer *, integer *), stest1_(real *,
	     real *, real *, real *);
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *);
    extern real scasum_(integer *, complex *, integer *);

    /* Fortran I/O blocks */
    static cilist io___19 = { 0, 6, 0, 0, 0 };


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
		i__2 = i__ - 1;
		i__3 = i__ + (np1 + combla_1.incx * 5 << 3) - 49;
		cx[i__2].r = cv[i__3].r, cx[i__2].i = cv[i__3].i;
/* L20: */
	    }
	    if (combla_1.icase == 6) {
/*              .. SCNRM2 .. */
		r__1 = scnrm2_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&r__1, &strue2[np1 - 1], &strue2[np1 - 1], sfac);
	    } else if (combla_1.icase == 7) {
/*              .. SCASUM .. */
		r__1 = scasum_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&r__1, &strue4[np1 - 1], &strue4[np1 - 1], sfac);
	    } else if (combla_1.icase == 8) {
/*              .. CSCAL .. */
		cscal_(&combla_1.n, &ca, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue5[(np1 + combla_1.incx * 5 << 3) - 48],
			 &ctrue5[(np1 + combla_1.incx * 5 << 3) - 48], sfac);
	    } else if (combla_1.icase == 9) {
/*              .. CSSCAL .. */
		csscal_(&combla_1.n, &sa, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue6[(np1 + combla_1.incx * 5 << 3) - 48],
			 &ctrue6[(np1 + combla_1.incx * 5 << 3) - 48], sfac);
	    } else if (combla_1.icase == 10) {
/*              .. ICAMAX .. */
		i__1 = icamax_(&combla_1.n, cx, &combla_1.incx);
		itest1_(&i__1, &itrue3[np1 - 1]);
	    } else {
		s_wsle(&io___19);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK1", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }

/* L40: */
	}
/* L60: */
    }

    combla_1.incx = 1;
    if (combla_1.icase == 8) {
/*        CSCAL */
/*        Add a test for alpha equal to zero. */
	ca.r = 0.f, ca.i = 0.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0.f, mwpct[i__1].i = 0.f;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1.f, mwpcs[i__1].i = 1.f;
/* L80: */
	}
	cscal_(&c__5, &ca, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
    } else if (combla_1.icase == 9) {
/*        CSSCAL */
/*        Add a test for alpha equal to zero. */
	sa = 0.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0.f, mwpct[i__1].i = 0.f;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1.f, mwpcs[i__1].i = 1.f;
/* L100: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to one. */
	sa = 1.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpct[i__1].r = cx[i__2].r, mwpct[i__1].i = cx[i__2].i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpcs[i__1].r = cx[i__2].r, mwpcs[i__1].i = cx[i__2].i;
/* L120: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to minus one. */
	sa = -1.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    q__1.r = -cx[i__2].r, q__1.i = -cx[i__2].i;
	    mwpct[i__1].r = q__1.r, mwpct[i__1].i = q__1.i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    q__1.r = -cx[i__2].r, q__1.i = -cx[i__2].i;
	    mwpcs[i__1].r = q__1.r, mwpcs[i__1].i = q__1.i;
/* L140: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
    }
    return 0;
} /* check1_ */

/* Subroutine */ int check2_(real *sfac)
{
    /* Initialized data */

    static complex ca = {.4f,-.7f};
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static complex cx1[7] = { {.7f,-.8f},{-.4f,-.7f},{-.1f,-.9f},{.2f,-.8f},{
	    -.9f,-.4f},{.1f,.4f},{-.6f,.6f} };
    static complex cy1[7] = { {.6f,-.6f},{-.9f,.5f},{.7f,-.6f},{.1f,-.5f},{
	    -.1f,-.2f},{-.5f,-.3f},{.8f,-.7f} };
    static complex ct8[112]	/* was [7][4][4] */ = { {.6f,-.6f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,-1.41f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,
	    -1.41f},{-1.55f,.5f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{.32f,-1.41f},{-1.55f,.5f},{.03f,-.89f},{-.38f,-.96f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,-1.41f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.07f,-.89f},{-.9f,.5f},{
	    .42f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.78f,.06f},{
	    -.9f,.5f},{.06f,-.13f},{.1f,-.5f},{-.77f,-.49f},{-.5f,-.3f},{.52f,
	    -1.51f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.32f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{-.07f,-.89f},{-1.18f,-.31f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.78f,.06f},{-1.54f,.97f},{
	    .03f,-.89f},{-.18f,-1.31f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,
	    -.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {.32f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.32f,-1.41f},{-.9f,.5f},{.05f,-.6f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.32f,-1.41f},{-.9f,.5f},{.05f,-.6f},{.1f,
	    -.5f},{-.77f,-.49f},{-.5f,-.3f},{.32f,-1.16f} };
    static complex ct7[16]	/* was [4][4] */ = { {0.f,0.f},{-.06f,-.9f},{
	    .65f,-.47f},{-.34f,-1.22f},{0.f,0.f},{-.06f,-.9f},{-.59f,-1.46f},{
	    -1.04f,-.04f},{0.f,0.f},{-.06f,-.9f},{-.83f,.59f},{.07f,-.37f},{
	    0.f,0.f},{-.06f,-.9f},{-.76f,-1.15f},{-1.33f,-1.82f} };
    static complex ct6[16]	/* was [4][4] */ = { {0.f,0.f},{.9f,.06f},{
	    .91f,-.77f},{1.8f,-.1f},{0.f,0.f},{.9f,.06f},{1.45f,.74f},{.2f,
	    .9f},{0.f,0.f},{.9f,.06f},{-.55f,.23f},{.83f,-.39f},{0.f,0.f},{
	    .9f,.06f},{1.04f,.79f},{1.95f,1.22f} };
    static complex ct10x[112]	/* was [7][4][4] */ = { {.7f,-.8f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},
	    {-.9f,.5f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,
	    -.6f},{-.9f,.5f},{.7f,-.6f},{.1f,-.5f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.6f},{-.4f,-.7f},{.6f,-.6f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{.8f,-.7f},{-.4f,-.7f},{-.1f,-.2f},{.2f,
	    -.8f},{.7f,-.6f},{.1f,.4f},{.6f,-.6f},{.7f,-.8f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.9f,.5f},{
	    -.4f,-.7f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    .1f,-.5f},{-.4f,-.7f},{.7f,-.6f},{.2f,-.8f},{-.9f,.5f},{.1f,.4f},{
	    .6f,-.6f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.6f,-.6f},{.7f,-.6f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{.7f,-.6f},{-.1f,-.2f},{
	    .8f,-.7f},{0.f,0.f},{0.f,0.f},{0.f,0.f} };
    static complex ct10y[112]	/* was [7][4][4] */ = { {.6f,-.6f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},
	    {-.4f,-.7f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    .7f,-.8f},{-.4f,-.7f},{-.1f,-.9f},{.2f,-.8f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{-.1f,-.9f},{-.9f,.5f},{.7f,-.8f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{-.6f,.6f},{-.9f,.5f},{-.9f,-.4f},{
	    .1f,-.5f},{-.1f,-.9f},{-.5f,-.3f},{.7f,-.8f},{.6f,-.6f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.1f,-.9f}
	    ,{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    -.6f,.6f},{-.9f,-.4f},{-.1f,-.9f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.7f,-.8f},{-.9f,.5f},{-.4f,-.7f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{-.9f,.5f},{-.4f,-.7f},{
	    .1f,-.5f},{-.1f,-.9f},{-.5f,-.3f},{.2f,-.8f} };
    static complex csize1[4] = { {0.f,0.f},{.9f,.9f},{1.63f,1.73f},{2.9f,
	    2.78f} };
    static complex csize3[14] = { {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,
	    1.17f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,1.17f} };
    static complex csize2[14]	/* was [7][2] */ = { {0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{1.54f,1.54f},{1.54f,
	    1.54f},{1.54f,1.54f},{1.54f,1.54f},{1.54f,1.54f},{1.54f,1.54f},{
	    1.54f,1.54f} };

    /* System generated locals */
    integer i__1, i__2;
    complex q__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    integer i__, ki, kn;
    complex cx[7], cy[7];
    integer mx, my;
    complex cdot[1];
    integer lenx, leny;
    extern /* Complex */
#ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL
 void cdotc_(complex *,
#else
complex cdotc_(
#endif
 integer *, complex *, integer
	    *, complex *, integer *);
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *,
	    complex *, integer *);
    extern /* Complex */
#ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL
 void cdotu_(complex *,
#else
complex cdotu_(
#endif
 integer *, complex *, integer
	    *, complex *, integer *);
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *,
	    complex *, integer *), ctest_(integer *, complex *, complex *,
	    complex *, real *);
    integer ksize;
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *,
	    integer *, complex *, integer *);

    /* Fortran I/O blocks */
    static cilist io___48 = { 0, 6, 0, 0, 0 };


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
/*           .. initialize all argument arrays .. */
	    for (i__ = 1; i__ <= 7; ++i__) {
		i__1 = i__ - 1;
		i__2 = i__ - 1;
		cx[i__1].r = cx1[i__2].r, cx[i__1].i = cx1[i__2].i;
		i__1 = i__ - 1;
		i__2 = i__ - 1;
		cy[i__1].r = cy1[i__2].r, cy[i__1].i = cy1[i__2].i;
/* L20: */
	    }
	    if (combla_1.icase == 1) {
/*              .. CDOTC .. */

#ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL
		cdotc_(&q__1,
#else
		q__1 = cdotc_(
#endif
		 &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = q__1.r, cdot[0].i = q__1.i;
		ctest_(&c__1, cdot, &ct6[kn + (ki << 2) - 5], &csize1[kn - 1],
			 sfac);
	    } else if (combla_1.icase == 2) {
/*              .. CDOTU .. */

#ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL
		cdotu_(&q__1,
#else
		q__1 = cdotu_(
#endif
		 &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = q__1.r, cdot[0].i = q__1.i;
		ctest_(&c__1, cdot, &ct7[kn + (ki << 2) - 5], &csize1[kn - 1],
			 sfac);
	    } else if (combla_1.icase == 3) {
/*              .. CAXPY .. */
		caxpy_(&combla_1.n, &ca, cx, &combla_1.incx, cy, &
			combla_1.incy);
		ctest_(&leny, cy, &ct8[(kn + (ki << 2)) * 7 - 35], &csize2[
			ksize * 7 - 7], sfac);
	    } else if (combla_1.icase == 4) {
/*              .. CCOPY .. */
		ccopy_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
		ctest_(&leny, cy, &ct10y[(kn + (ki << 2)) * 7 - 35], csize3, &
			c_b43);
	    } else if (combla_1.icase == 5) {
/*              .. CSWAP .. */
		cswap_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
		ctest_(&lenx, cx, &ct10x[(kn + (ki << 2)) * 7 - 35], csize3, &
			c_b43);
		ctest_(&leny, cy, &ct10y[(kn + (ki << 2)) * 7 - 35], csize3, &
			c_b43);
	    } else {
		s_wsle(&io___48);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK2", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }

/* L40: */
	}
/* L60: */
    }
    return 0;
} /* check2_ */

/* Subroutine */ int stest_(integer *len, real *scomp, real *strue, real *
	ssize, real *sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY MODE  I             "
	    "               \002,\002 COMP(I)                             TRU"
	    "E(I)  DIFFERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,3i5,i3,2e36.8,2e12.4)";

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
    static cilist io___51 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___52 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___53 = { 0, 6, 0, fmt_99997, 0 };


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
		 s_epsilon_(&c_b52)) {
	    goto L40;
	}

/*                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I). */

	if (! combla_1.pass) {
	    goto L20;
	}
/*                             PRINT FAIL MESSAGE AND HEADER. */
	combla_1.pass = FALSE_;
	s_wsfe(&io___51);
	e_wsfe();
	s_wsfe(&io___52);
	e_wsfe();
L20:
	s_wsfe(&io___53);
	do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.mode, (ftnlen)sizeof(integer));
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

/* Subroutine */ int ctest_(integer *len, complex *ccomp, complex *ctrue,
	complex *csize, real *sfac)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    integer i__;
    real scomp[20], ssize[20], strue[20];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *,
	    real *);

/*     **************************** CTEST ***************************** */

/*     C.L. LAWSON, JPL, 1978 DEC 6 */

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    --csize;
    --ctrue;
    --ccomp;

    /* Function Body */
    i__1 = *len;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	scomp[(i__ << 1) - 2] = ccomp[i__2].r;
	scomp[(i__ << 1) - 1] = r_imag(&ccomp[i__]);
	i__2 = i__;
	strue[(i__ << 1) - 2] = ctrue[i__2].r;
	strue[(i__ << 1) - 1] = r_imag(&ctrue[i__]);
	i__2 = i__;
	ssize[(i__ << 1) - 2] = csize[i__2].r;
	ssize[(i__ << 1) - 1] = r_imag(&csize[i__]);
/* L20: */
    }

    i__1 = *len << 1;
    stest_(&i__1, scomp, strue, ssize, sfac);
    return 0;
} /* ctest_ */

/* Subroutine */ int itest1_(integer *icomp, integer *itrue)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY MODE                "
	    "               \002,\002 COMP                                TRU"
	    "E     DIFFERENCE\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,3i5,2i36,i12)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer id;

    /* Fortran I/O blocks */
    static cilist io___60 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___61 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___63 = { 0, 6, 0, fmt_99997, 0 };


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
    s_wsfe(&io___60);
    e_wsfe();
    s_wsfe(&io___61);
    e_wsfe();
L20:
    id = *icomp - *itrue;
    s_wsfe(&io___63);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.mode, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*icomp), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*itrue), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&id, (ftnlen)sizeof(integer));
    e_wsfe();
L40:
    return 0;

} /* itest1_ */

/* Main program alias */ int cblat1_ () { main (); return 0; }
