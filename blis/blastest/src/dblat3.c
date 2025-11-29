/* dblat3.f -- translated by f2c (version 20100827).
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

union {
    struct {
	integer infot, noutc;
	logical ok, lerr;
    } _1;
    struct {
	integer infot, nout;
	logical ok, lerr;
    } _2;
} infoc_;

#define infoc_1 (infoc_._1)
#define infoc_2 (infoc_._2)

struct {
    char srnamt[6];
} srnamc_;

#define srnamc_1 srnamc_

/* Table of constant values */

static integer c__9 = 9;
static integer c__1 = 1;
static integer c__3 = 3;
static integer c__8 = 8;
static integer c__5 = 5;
static integer c__65 = 65;
static integer c__7 = 7;
static doublereal c_b86 = 0.;
static doublereal c_b96 = 1.;
static logical c_true = TRUE_;
static logical c_false = FALSE_;
static integer c__0 = 0;
static integer c_n1 = -1;
static integer c__2 = 2;

/* > \brief \b DBLAT3 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM DBLAT3 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > Test program for the DOUBLE PRECISION Level 3 Blas. */
/* > */
/* > The program must be driven by a short data file. The first 14 records */
/* > of the file are read using list-directed input, the last 6 records */
/* > are read using the format ( A6, L2 ). An annotated example of a data */
/* > file can be obtained by deleting the first 3 characters from the */
/* > following 20 lines: */
/* > 'dblat3.out'      NAME OF SUMMARY OUTPUT FILE */
/* > 6                 UNIT NUMBER OF SUMMARY FILE */
/* > 'DBLAT3.SNAP'     NAME OF SNAPSHOT OUTPUT FILE */
/* > -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0) */
/* > F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD. */
/* > F        LOGICAL FLAG, T TO STOP ON FAILURES. */
/* > T        LOGICAL FLAG, T TO TEST ERROR EXITS. */
/* > 16.0     THRESHOLD VALUE OF TEST RATIO */
/* > 6                 NUMBER OF VALUES OF N */
/* > 0 1 2 3 5 9       VALUES OF N */
/* > 3                 NUMBER OF VALUES OF ALPHA */
/* > 0.0 1.0 0.7       VALUES OF ALPHA */
/* > 3                 NUMBER OF VALUES OF BETA */
/* > 0.0 1.0 1.3       VALUES OF BETA */
/* > DGEMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > DSYMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > DTRMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > DTRSM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > DSYRK  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > DSYR2K T PUT F FOR NO TEST. SAME COLUMNS. */
/* > */
/* > Further Details */
/* > =============== */
/* > */
/* > See: */
/* > */
/* >    Dongarra J. J., Du Croz J. J., Duff I. S. and Hammarling S. */
/* >    A Set of Level 3 Basic Linear Algebra Subprograms. */
/* > */
/* >    Technical Memorandum No.88 (Revision 1), Mathematics and */
/* >    Computer Science Division, Argonne National Laboratory, 9700 */
/* >    South Cass Avenue, Argonne, Illinois 60439, US. */
/* > */
/* > -- Written on 8-February-1989. */
/* >    Jack Dongarra, Argonne National Laboratory. */
/* >    Iain Duff, AERE Harwell. */
/* >    Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/* >    Sven Hammarling, Numerical Algorithms Group Ltd. */
/* > */
/* >    10-9-00:  Change STATUS='NEW' to 'UNKNOWN' so that the testers */
/* >              can be run multiple times without deleting generated */
/* >              output files (susan) */
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
    char* program = "dblat3";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static char snames[6*6] = "DGEMM " "DSYMM " "DTRMM " "DTRSM " "DSYRK "
	    "DSYR2K";

    /* Format strings */
    static char fmt_9997[] = "(\002 NUMBER OF VALUES OF \002,a,\002 IS LESS "
	    "THAN 1 OR GREATER \002,\002THAN \002,i2)";
    static char fmt_9996[] = "(\002 VALUE OF N IS LESS THAN 0 OR GREATER THA"
	    "N \002,i2)";
    static char fmt_9995[] = "(\002 TESTS OF THE DOUBLE PRECISION LEVEL 3 BL"
	    "AS\002,//\002 THE F\002,\002OLLOWING PARAMETER VALUES WILL BE US"
	    "ED:\002)";
    static char fmt_9994[] = "(\002   FOR N              \002,9i6)";
    static char fmt_9993[] = "(\002   FOR ALPHA          \002,7f6.1)";
    static char fmt_9992[] = "(\002   FOR BETA           \002,7f6.1)";
    static char fmt_9984[] = "(\002 ERROR-EXITS WILL NOT BE TESTED\002)";
    static char fmt_9999[] = "(\002 ROUTINES PASS COMPUTATIONAL TESTS IF TES"
	    "T RATIO IS LES\002,\002S THAN\002,f8.2)";
    static char fmt_9988[] = "(a6,l2)";
    static char fmt_9990[] = "(\002 SUBPROGRAM NAME \002,a6,\002 NOT RECOGNI"
	    "ZED\002,/\002 ******* T\002,\002ESTS ABANDONED *******\002)";
    static char fmt_9998[] = "(\002 RELATIVE MACHINE PRECISION IS TAKEN TO"
	    " BE\002,1p,d9.1)";
    static char fmt_9989[] = "(\002 ERROR IN DMMCH -  IN-LINE DOT PRODUCTS A"
	    "RE BEING EVALU\002,\002ATED WRONGLY.\002,/\002 DMMCH WAS CALLED "
	    "WITH TRANSA = \002,a1,\002 AND TRANSB = \002,a1,/\002 AND RETURN"
	    "ED SAME = \002,l1,\002 AND \002,\002ERR = \002,f12.3,\002.\002,"
	    "/\002 THIS MAY BE DUE TO FAULTS IN THE \002,\002ARITHMETIC OR TH"
	    "E COMPILER.\002,/\002 ******* TESTS ABANDONED \002,\002******"
	    "*\002)";
    static char fmt_9987[] = "(1x,a6,\002 WAS NOT TESTED\002)";
    static char fmt_9986[] = "(/\002 END OF TESTS\002)";
    static char fmt_9985[] = "(/\002 ******* FATAL ERROR - TESTS ABANDONED *"
	    "******\002)";
    static char fmt_9991[] = "(\002 AMEND DATA FILE OR INCREASE ARRAY SIZES "
	    "IN PROGRAM\002,/\002 ******* TESTS ABANDONED *******\002)";

    /* System generated locals */
    integer i__1, i__2, i__3;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer s_rsle(cilist *), do_lio(integer *, integer *, char *, ftnlen),
	    e_rsle(void), f_open(olist *), s_wsfe(cilist *), do_fio(integer *,
	     char *, ftnlen), e_wsfe(void), s_wsle(cilist *), e_wsle(void),
	    s_rsfe(cilist *), e_rsfe(void), s_cmp(const char *, const char *, ftnlen,
	    ftnlen);
    /* Subroutine */ int s_stop(char *, ftnlen);
    integer f_clos(cllist *);
    /* Subroutine */ int s_copy(char *, const char *, ftnlen, ftnlen);

    /* Local variables */
    doublereal c__[4225]	/* was [65][65] */, g[65];
    integer i__, j, n;
    doublereal w[130], aa[4225], ab[8450]	/* was [65][130] */, bb[4225],
	     cc[4225], as[4225], bs[4225], cs[4225], ct[65], alf[7];
    extern logical lde_(doublereal *, doublereal *, integer *);
    doublereal bet[7], eps, err;
    integer nalf, idim[9];
    logical same;
    integer nbet, ntra;
    logical rewi;
    integer nout;
    extern /* Subroutine */ int dchk1_(char *, doublereal *, doublereal *,
	    integer *, integer *, logical *, logical *, logical *, integer *,
	    integer *, integer *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, ftnlen), dchk2_(char *,
	    doublereal *, doublereal *, integer *, integer *, logical *,
	    logical *, logical *, integer *, integer *, integer *, doublereal
	    *, integer *, doublereal *, integer *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, ftnlen), dchk3_(char *, doublereal *, doublereal *,
	    integer *, integer *, logical *, logical *, logical *, integer *,
	    integer *, integer *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *, ftnlen),
	    dchk4_(char *, doublereal *, doublereal *, integer *, integer *,
	    logical *, logical *, logical *, integer *, integer *, integer *,
	    doublereal *, integer *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, ftnlen), dchk5_(char *, doublereal *,
	    doublereal *, integer *, integer *, logical *, logical *, logical
	    *, integer *, integer *, integer *, doublereal *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *, ftnlen),
	    dchke_(integer *, char *, integer *, ftnlen);
    logical fatal;
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen);
    logical trace;
    integer nidim;
    char snaps[32];
    integer isnum;
    logical ltest[6], sfatal;
    char snamet[6], transa[1], transb[1];
    doublereal thresh;
    logical ltestt, tsterr;
    char summry[32];
    extern double d_epsilon_(doublereal *);

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 5, 0, 0, 0 };
    static cilist io___4 = { 0, 5, 0, 0, 0 };
    static cilist io___6 = { 0, 5, 0, 0, 0 };
    static cilist io___8 = { 0, 5, 0, 0, 0 };
    static cilist io___11 = { 0, 5, 0, 0, 0 };
    static cilist io___13 = { 0, 5, 0, 0, 0 };
    static cilist io___15 = { 0, 5, 0, 0, 0 };
    static cilist io___17 = { 0, 5, 0, 0, 0 };
    static cilist io___19 = { 0, 5, 0, 0, 0 };
    static cilist io___21 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___22 = { 0, 5, 0, 0, 0 };
    static cilist io___25 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___26 = { 0, 5, 0, 0, 0 };
    static cilist io___28 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___29 = { 0, 5, 0, 0, 0 };
    static cilist io___31 = { 0, 5, 0, 0, 0 };
    static cilist io___33 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___34 = { 0, 5, 0, 0, 0 };
    static cilist io___36 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___37 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___38 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___39 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___40 = { 0, 0, 0, 0, 0 };
    static cilist io___41 = { 0, 0, 0, fmt_9984, 0 };
    static cilist io___42 = { 0, 0, 0, 0, 0 };
    static cilist io___43 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___44 = { 0, 0, 0, 0, 0 };
    static cilist io___46 = { 0, 5, 1, fmt_9988, 0 };
    static cilist io___49 = { 0, 0, 0, fmt_9990, 0 };
    static cilist io___51 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___64 = { 0, 0, 0, fmt_9989, 0 };
    static cilist io___65 = { 0, 0, 0, fmt_9989, 0 };
    static cilist io___66 = { 0, 0, 0, fmt_9989, 0 };
    static cilist io___67 = { 0, 0, 0, fmt_9989, 0 };
    static cilist io___69 = { 0, 0, 0, 0, 0 };
    static cilist io___70 = { 0, 0, 0, fmt_9987, 0 };
    static cilist io___71 = { 0, 0, 0, 0, 0 };
    static cilist io___78 = { 0, 0, 0, fmt_9986, 0 };
    static cilist io___79 = { 0, 0, 0, fmt_9985, 0 };
    static cilist io___80 = { 0, 0, 0, fmt_9991, 0 };



/*  -- Reference BLAS test routine (version 3.4.1) -- */
/*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
/*     .. Executable Statements .. */

/*     Read name and unit number for summary output file and open file. */

    s_rsle(&io___2);
    do_lio(&c__9, &c__1, summry, (ftnlen)32);
    e_rsle();
    s_rsle(&io___4);
    do_lio(&c__3, &c__1, (char *)&nout, (ftnlen)sizeof(integer));
    e_rsle();
    o__1.oerr = 0;
    o__1.ounit = nout;
    o__1.ofnmlen = 32;
    o__1.ofnm = summry;
    o__1.orl = 0;
    o__1.osta = "UNKNOWN";
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    infoc_1.noutc = nout;

/*     Read name and unit number for snapshot output file and open file. */

    s_rsle(&io___6);
    do_lio(&c__9, &c__1, snaps, (ftnlen)32);
    e_rsle();
    s_rsle(&io___8);
    do_lio(&c__3, &c__1, (char *)&ntra, (ftnlen)sizeof(integer));
    e_rsle();
    trace = ntra >= 0;
    if (trace) {
	o__1.oerr = 0;
	o__1.ounit = ntra;
	o__1.ofnmlen = 32;
	o__1.ofnm = snaps;
	o__1.orl = 0;
	o__1.osta = "UNKNOWN";
	o__1.oacc = 0;
	o__1.ofm = 0;
	o__1.oblnk = 0;
	f_open(&o__1);
    }
/*     Read the flag that directs rewinding of the snapshot file. */
    s_rsle(&io___11);
    do_lio(&c__8, &c__1, (char *)&rewi, (ftnlen)sizeof(logical));
    e_rsle();
    rewi = rewi && trace;
/*     Read the flag that directs stopping on any failure. */
    s_rsle(&io___13);
    do_lio(&c__8, &c__1, (char *)&sfatal, (ftnlen)sizeof(logical));
    e_rsle();
/*     Read the flag that indicates whether error exits are to be tested. */
    s_rsle(&io___15);
    do_lio(&c__8, &c__1, (char *)&tsterr, (ftnlen)sizeof(logical));
    e_rsle();
/*     Read the threshold value of the test ratio */
    s_rsle(&io___17);
    do_lio(&c__5, &c__1, (char *)&thresh, (ftnlen)sizeof(doublereal));
    e_rsle();

/*     Read and check the parameter values for the tests. */

/*     Values of N */
    s_rsle(&io___19);
    do_lio(&c__3, &c__1, (char *)&nidim, (ftnlen)sizeof(integer));
    e_rsle();
    if (nidim < 1 || nidim > 9) {
	io___21.ciunit = nout;
	s_wsfe(&io___21);
	do_fio(&c__1, "N", (ftnlen)1);
	do_fio(&c__1, (char *)&c__9, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L220;
    }
    s_rsle(&io___22);
    i__1 = nidim;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__3, &c__1, (char *)&idim[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_rsle();
    i__1 = nidim;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (idim[i__ - 1] < 0 || idim[i__ - 1] > 65) {
	    io___25.ciunit = nout;
	    s_wsfe(&io___25);
	    do_fio(&c__1, (char *)&c__65, (ftnlen)sizeof(integer));
	    e_wsfe();
	    goto L220;
	}
/* L10: */
    }
/*     Values of ALPHA */
    s_rsle(&io___26);
    do_lio(&c__3, &c__1, (char *)&nalf, (ftnlen)sizeof(integer));
    e_rsle();
    if (nalf < 1 || nalf > 7) {
	io___28.ciunit = nout;
	s_wsfe(&io___28);
	do_fio(&c__1, "ALPHA", (ftnlen)5);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L220;
    }
    s_rsle(&io___29);
    i__1 = nalf;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__5, &c__1, (char *)&alf[i__ - 1], (ftnlen)sizeof(doublereal)
		);
    }
    e_rsle();
/*     Values of BETA */
    s_rsle(&io___31);
    do_lio(&c__3, &c__1, (char *)&nbet, (ftnlen)sizeof(integer));
    e_rsle();
    if (nbet < 1 || nbet > 7) {
	io___33.ciunit = nout;
	s_wsfe(&io___33);
	do_fio(&c__1, "BETA", (ftnlen)4);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L220;
    }
    s_rsle(&io___34);
    i__1 = nbet;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__5, &c__1, (char *)&bet[i__ - 1], (ftnlen)sizeof(doublereal)
		);
    }
    e_rsle();

/*     Report values of parameters. */

    io___36.ciunit = nout;
    s_wsfe(&io___36);
    e_wsfe();
    io___37.ciunit = nout;
    s_wsfe(&io___37);
    i__1 = nidim;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&idim[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    io___38.ciunit = nout;
    s_wsfe(&io___38);
    i__1 = nalf;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&alf[i__ - 1], (ftnlen)sizeof(doublereal));
    }
    e_wsfe();
    io___39.ciunit = nout;
    s_wsfe(&io___39);
    i__1 = nbet;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&bet[i__ - 1], (ftnlen)sizeof(doublereal));
    }
    e_wsfe();
    if (! tsterr) {
	io___40.ciunit = nout;
	s_wsle(&io___40);
	e_wsle();
	io___41.ciunit = nout;
	s_wsfe(&io___41);
	e_wsfe();
    }
    io___42.ciunit = nout;
    s_wsle(&io___42);
    e_wsle();
    io___43.ciunit = nout;
    s_wsfe(&io___43);
    do_fio(&c__1, (char *)&thresh, (ftnlen)sizeof(doublereal));
    e_wsfe();
    io___44.ciunit = nout;
    s_wsle(&io___44);
    e_wsle();

/*     Read names of subroutines and flags which indicate */
/*     whether they are to be tested. */

    for (i__ = 1; i__ <= 6; ++i__) {
	ltest[i__ - 1] = FALSE_;
/* L20: */
    }
L30:
    i__1 = s_rsfe(&io___46);
    if (i__1 != 0) {
	goto L60;
    }
    i__1 = do_fio(&c__1, snamet, (ftnlen)6);
    if (i__1 != 0) {
	goto L60;
    }
    i__1 = do_fio(&c__1, (char *)&ltestt, (ftnlen)sizeof(logical));
    if (i__1 != 0) {
	goto L60;
    }
    i__1 = e_rsfe();
    if (i__1 != 0) {
	goto L60;
    }
    for (i__ = 1; i__ <= 6; ++i__) {
	if (s_cmp(snamet, snames + (i__ - 1) * 6, (ftnlen)6, (ftnlen)6) == 0)
		{
	    goto L50;
	}
/* L40: */
    }
    io___49.ciunit = nout;
    s_wsfe(&io___49);
    do_fio(&c__1, snamet, (ftnlen)6);
    e_wsfe();
    s_stop("", (ftnlen)0);
L50:
    ltest[i__ - 1] = ltestt;
    goto L30;

L60:
    cl__1.cerr = 0;
    cl__1.cunit = 5;
    cl__1.csta = 0;
    f_clos(&cl__1);

/*     Compute EPS (the machine precision). */

    eps = d_epsilon_(&c_b86);
    io___51.ciunit = nout;
    s_wsfe(&io___51);
    do_fio(&c__1, (char *)&eps, (ftnlen)sizeof(doublereal));
    e_wsfe();

/*     Check the reliability of DMMCH using exact data. */

    n = 32;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__3 = i__ - j + 1;
	    ab[i__ + j * 65 - 66] = (doublereal) max(i__3,0);
/* L90: */
	}
	ab[j + 4224] = (doublereal) j;
	ab[(j + 65) * 65 - 65] = (doublereal) j;
	c__[j - 1] = 0.;
/* L100: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	cc[j - 1] = (doublereal) (j * ((j + 1) * j) / 2 - (j + 1) * j * (j -
		1) / 3);
/* L110: */
    }
/*     CC holds the exact result. On exit from DMMCH CT holds */
/*     the result computed by DMMCH. */
    *(unsigned char *)transa = 'N';
    *(unsigned char *)transb = 'N';
    dmmch_(transa, transb, &n, &c__1, &n, &c_b96, ab, &c__65, &ab[4225], &
	    c__65, &c_b86, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	     &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lde_(cc, ct, &n);
    if (! same || err != 0.) {
	io___64.ciunit = nout;
	s_wsfe(&io___64);
	do_fio(&c__1, transa, (ftnlen)1);
	do_fio(&c__1, transb, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }
    *(unsigned char *)transb = 'T';
    dmmch_(transa, transb, &n, &c__1, &n, &c_b96, ab, &c__65, &ab[4225], &
	    c__65, &c_b86, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	     &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lde_(cc, ct, &n);
    if (! same || err != 0.) {
	io___65.ciunit = nout;
	s_wsfe(&io___65);
	do_fio(&c__1, transa, (ftnlen)1);
	do_fio(&c__1, transb, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	ab[j + 4224] = (doublereal) (n - j + 1);
	ab[(j + 65) * 65 - 65] = (doublereal) (n - j + 1);
/* L120: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	cc[n - j] = (doublereal) (j * ((j + 1) * j) / 2 - (j + 1) * j * (j -
		1) / 3);
/* L130: */
    }
    *(unsigned char *)transa = 'T';
    *(unsigned char *)transb = 'N';
    dmmch_(transa, transb, &n, &c__1, &n, &c_b96, ab, &c__65, &ab[4225], &
	    c__65, &c_b86, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	     &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lde_(cc, ct, &n);
    if (! same || err != 0.) {
	io___66.ciunit = nout;
	s_wsfe(&io___66);
	do_fio(&c__1, transa, (ftnlen)1);
	do_fio(&c__1, transb, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }
    *(unsigned char *)transb = 'T';
    dmmch_(transa, transb, &n, &c__1, &n, &c_b96, ab, &c__65, &ab[4225], &
	    c__65, &c_b86, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	     &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lde_(cc, ct, &n);
    if (! same || err != 0.) {
	io___67.ciunit = nout;
	s_wsfe(&io___67);
	do_fio(&c__1, transa, (ftnlen)1);
	do_fio(&c__1, transb, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }

/*     Test each subroutine in turn. */

    for (isnum = 1; isnum <= 6; ++isnum) {
	io___69.ciunit = nout;
	s_wsle(&io___69);
	e_wsle();
	if (! ltest[isnum - 1]) {
/*           Subprogram is not to be tested. */
	    io___70.ciunit = nout;
	    s_wsfe(&io___70);
	    do_fio(&c__1, snames + (isnum - 1) * 6, (ftnlen)6);
	    e_wsfe();
	} else {
	    s_copy(srnamc_1.srnamt, snames + (isnum - 1) * 6, (ftnlen)6, (
		    ftnlen)6);
/*           Test error exits. */
	    if (tsterr) {
		dchke_(&isnum, snames + (isnum - 1) * 6, &nout, (ftnlen)6);
		io___71.ciunit = nout;
		s_wsle(&io___71);
		e_wsle();
	    }
/*           Test computations. */
	    infoc_1.infot = 0;
	    infoc_1.ok = TRUE_;
	    fatal = FALSE_;
	    switch (isnum) {
		case 1:  goto L140;
		case 2:  goto L150;
		case 3:  goto L160;
		case 4:  goto L160;
		case 5:  goto L170;
		case 6:  goto L180;
	    }
/*           Test DGEMM, 01. */
L140:
	    dchk1_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test DSYMM, 02. */
L150:
	    dchk2_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test DTRMM, 03, DTRSM, 04. */
L160:
	    dchk3_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &c__65,
		    ab, aa, as, &ab[4225], bb, bs, ct, g, c__, (ftnlen)6);
	    goto L190;
/*           Test DSYRK, 05. */
L170:
	    dchk4_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test DSYR2K, 06. */
L180:
	    dchk5_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, bb, bs, c__, cc, cs, ct, g, w, (
		    ftnlen)6);
	    goto L190;

L190:
	    if (fatal && sfatal) {
		goto L210;
	    }
	}
/* L200: */
    }
    io___78.ciunit = nout;
    s_wsfe(&io___78);
    e_wsfe();
    goto L230;

L210:
    io___79.ciunit = nout;
    s_wsfe(&io___79);
    e_wsfe();
    goto L230;

L220:
    io___80.ciunit = nout;
    s_wsfe(&io___80);
    e_wsfe();

L230:
    if (trace) {
	cl__1.cerr = 0;
	cl__1.cunit = ntra;
	cl__1.csta = 0;
	f_clos(&cl__1);
    }
    cl__1.cerr = 0;
    cl__1.cunit = nout;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_stop("", (ftnlen)0);


/*     End of DBLAT3. */

#ifdef BLIS_ENABLE_HPX
    return bli_thread_finalize_hpx();
#else
	// Return peacefully.
	return 0;
#endif
} /* main */

/* Subroutine */ int dchk1_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublereal *alf,
	integer *nbet, doublereal *bet, integer *nmax, doublereal *a,
	doublereal *aa, doublereal *as, doublereal *b, doublereal *bb,
	doublereal *bs, doublereal *c__, doublereal *cc, doublereal *cs,
	doublereal *ct, doublereal *g, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[3] = "NTC";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002','\002"
	    ",a1,\002',\002,3(i3,\002,\002),f4.1,\002, A,\002,i3,\002, B,\002"
	    ",i3,\002,\002,f4.1,\002, \002,\002C,\002,i3,\002).\002)";
    static char fmt_9994[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
	    "N VALID CALL *\002,\002******\002)";
    static char fmt_9998[] = "(\002 ******* FATAL ERROR - PARAMETER NUMBER"
	    " \002,i2,\002 WAS CH\002,\002ANGED INCORRECTLY *******\002)";
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE COMPUTATIONAL TE"
	    "STS (\002,i6,\002 CALL\002,\002S)\002)";
    static char fmt_9997[] = "(\002 \002,a6,\002 COMPLETED THE COMPUTATIONAL"
	    " TESTS (\002,i6,\002 C\002,\002ALLS)\002,/\002 ******* BUT WITH "
	    "MAXIMUM TEST RATIO\002,f8.2,\002 - SUSPECT *******\002)";
    static char fmt_9996[] = "(\002 ******* \002,a6,\002 FAILED ON CALL NUMB"
	    "ER:\002)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5, i__6;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, k, m, n, ia, ib, ma, mb, na, nb, nc, ik, im, in, ks, ms, ns,
	    ica, icb, laa, lbb, lda, lcc, ldb, ldc;
    extern logical lde_(doublereal *, doublereal *, integer *);
    doublereal als, bls, err, beta;
    integer ldas, ldbs, ldcs;
    logical same, null;
    extern /* Subroutine */ int dmake_(char *, char *, char *, integer *,
	    integer *, doublereal *, integer *, doublereal *, integer *,
	    logical *, doublereal *, ftnlen, ftnlen, ftnlen);
    doublereal alpha;
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen), dgemm_(char *,
	    char *, integer *, integer *, integer *, doublereal *, doublereal
	    *, integer *, doublereal *, integer *, doublereal *, doublereal *,
	     integer *, ftnlen, ftnlen);
    logical isame[13], trana, tranb;
    integer nargs;
    logical reset;
    extern logical lderes_(char *, char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, ftnlen, ftnlen);
    char tranas[1], tranbs[1], transa[1], transb[1];
    doublereal errmax;

    /* Fortran I/O blocks */
    static cilist io___124 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___125 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___128 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___130 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___131 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___132 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___133 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests DGEMM. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --idim;
    --alf;
    --bet;
    --g;
    --ct;
    --cs;
    --cc;
    c_dim1 = *nmax;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --bs;
    --bb;
    b_dim1 = *nmax;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
/*     .. Executable Statements .. */

    nargs = 13;
    nc = 0;
    reset = TRUE_;
    errmax = 0.;

    i__1 = *nidim;
    for (im = 1; im <= i__1; ++im) {
	m = idim[im];

	i__2 = *nidim;
	for (in = 1; in <= i__2; ++in) {
	    n = idim[in];
/*           Set LDC to 1 more than minimum value if room. */
	    ldc = m;
	    if (ldc < *nmax) {
		++ldc;
	    }
/*           Skip tests if not enough room. */
	    if (ldc > *nmax) {
		goto L100;
	    }
	    lcc = ldc * n;
	    null = n <= 0 || m <= 0;

	    i__3 = *nidim;
	    for (ik = 1; ik <= i__3; ++ik) {
		k = idim[ik];

		for (ica = 1; ica <= 3; ++ica) {
		    *(unsigned char *)transa = *(unsigned char *)&ich[ica - 1]
			    ;
		    trana = *(unsigned char *)transa == 'T' || *(unsigned
			    char *)transa == 'C';

		    if (trana) {
			ma = k;
			na = m;
		    } else {
			ma = m;
			na = k;
		    }
/*                 Set LDA to 1 more than minimum value if room. */
		    lda = ma;
		    if (lda < *nmax) {
			++lda;
		    }
/*                 Skip tests if not enough room. */
		    if (lda > *nmax) {
			goto L80;
		    }
		    laa = lda * na;

/*                 Generate the matrix A. */

		    dmake_("GE", " ", " ", &ma, &na, &a[a_offset], nmax, &aa[
			    1], &lda, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);

		    for (icb = 1; icb <= 3; ++icb) {
			*(unsigned char *)transb = *(unsigned char *)&ich[icb
				- 1];
			tranb = *(unsigned char *)transb == 'T' || *(unsigned
				char *)transb == 'C';

			if (tranb) {
			    mb = n;
			    nb = k;
			} else {
			    mb = k;
			    nb = n;
			}
/*                    Set LDB to 1 more than minimum value if room. */
			ldb = mb;
			if (ldb < *nmax) {
			    ++ldb;
			}
/*                    Skip tests if not enough room. */
			if (ldb > *nmax) {
			    goto L70;
			}
			lbb = ldb * nb;

/*                    Generate the matrix B. */

			dmake_("GE", " ", " ", &mb, &nb, &b[b_offset], nmax, &
				bb[1], &ldb, &reset, &c_b86, (ftnlen)2, (
				ftnlen)1, (ftnlen)1);

			i__4 = *nalf;
			for (ia = 1; ia <= i__4; ++ia) {
			    alpha = alf[ia];

			    i__5 = *nbet;
			    for (ib = 1; ib <= i__5; ++ib) {
				beta = bet[ib];

/*                          Generate the matrix C. */

				dmake_("GE", " ", " ", &m, &n, &c__[c_offset],
					 nmax, &cc[1], &ldc, &reset, &c_b86, (
					ftnlen)2, (ftnlen)1, (ftnlen)1);

				++nc;

/*                          Save every datum before calling the */
/*                          subroutine. */

				*(unsigned char *)tranas = *(unsigned char *)
					transa;
				*(unsigned char *)tranbs = *(unsigned char *)
					transb;
				ms = m;
				ns = n;
				ks = k;
				als = alpha;
				i__6 = laa;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    as[i__] = aa[i__];
/* L10: */
				}
				ldas = lda;
				i__6 = lbb;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    bs[i__] = bb[i__];
/* L20: */
				}
				ldbs = ldb;
				bls = beta;
				i__6 = lcc;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    cs[i__] = cc[i__];
/* L30: */
				}
				ldcs = ldc;

/*                          Call the subroutine. */

				if (*trace) {
				    io___124.ciunit = *ntra;
				    s_wsfe(&io___124);
				    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, sname, (ftnlen)6);
				    do_fio(&c__1, transa, (ftnlen)1);
				    do_fio(&c__1, transb, (ftnlen)1);
				    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&alpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&ldb, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&beta, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&ldc, (ftnlen)
					    sizeof(integer));
				    e_wsfe();
				}
				if (*rewi) {
				    al__1.aerr = 0;
				    al__1.aunit = *ntra;
				    f_rew(&al__1);
				}
				dgemm_(transa, transb, &m, &n, &k, &alpha, &
					aa[1], &lda, &bb[1], &ldb, &beta, &cc[
					1], &ldc, (ftnlen)1, (ftnlen)1);

/*                          Check if error-exit was taken incorrectly. */

				if (! infoc_1.ok) {
				    io___125.ciunit = *nout;
				    s_wsfe(&io___125);
				    e_wsfe();
				    *fatal = TRUE_;
				    goto L120;
				}

/*                          See what data changed inside subroutines. */

				isame[0] = *(unsigned char *)transa == *(
					unsigned char *)tranas;
				isame[1] = *(unsigned char *)transb == *(
					unsigned char *)tranbs;
				isame[2] = ms == m;
				isame[3] = ns == n;
				isame[4] = ks == k;
				isame[5] = als == alpha;
				isame[6] = lde_(&as[1], &aa[1], &laa);
				isame[7] = ldas == lda;
				isame[8] = lde_(&bs[1], &bb[1], &lbb);
				isame[9] = ldbs == ldb;
				isame[10] = bls == beta;
				if (null) {
				    isame[11] = lde_(&cs[1], &cc[1], &lcc);
				} else {
				    isame[11] = lderes_("GE", " ", &m, &n, &
					    cs[1], &cc[1], &ldc, (ftnlen)2, (
					    ftnlen)1);
				}
				isame[12] = ldcs == ldc;

/*                          If data was incorrectly changed, report */
/*                          and return. */

				same = TRUE_;
				i__6 = nargs;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    same = same && isame[i__ - 1];
				    if (! isame[i__ - 1]) {
					io___128.ciunit = *nout;
					s_wsfe(&io___128);
					do_fio(&c__1, (char *)&i__, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
/* L40: */
				}
				if (! same) {
				    *fatal = TRUE_;
				    goto L120;
				}

				if (! null) {

/*                             Check the result. */

				    dmmch_(transa, transb, &m, &n, &k, &alpha,
					     &a[a_offset], nmax, &b[b_offset],
					     nmax, &beta, &c__[c_offset],
					    nmax, &ct[1], &g[1], &cc[1], &ldc,
					     eps, &err, fatal, nout, &c_true,
					    (ftnlen)1, (ftnlen)1);
				    errmax = max(errmax,err);
/*                             If got really bad answer, report and */
/*                             return. */
				    if (*fatal) {
					goto L120;
				    }
				}

/* L50: */
			    }

/* L60: */
			}

L70:
			;
		    }

L80:
		    ;
		}

/* L90: */
	    }

L100:
	    ;
	}

/* L110: */
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___130.ciunit = *nout;
	s_wsfe(&io___130);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___131.ciunit = *nout;
	s_wsfe(&io___131);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L130;

L120:
    io___132.ciunit = *nout;
    s_wsfe(&io___132);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___133.ciunit = *nout;
    s_wsfe(&io___133);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, transa, (ftnlen)1);
    do_fio(&c__1, transb, (ftnlen)1);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L130:
    return 0;


/*     End of DCHK1. */

} /* dchk1_ */

/* Subroutine */ int dchk2_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublereal *alf,
	integer *nbet, doublereal *bet, integer *nmax, doublereal *a,
	doublereal *aa, doublereal *as, doublereal *b, doublereal *bb,
	doublereal *bs, doublereal *c__, doublereal *cc, doublereal *cs,
	doublereal *ct, doublereal *g, ftnlen sname_len)
{
    /* Initialized data */

    static char ichs[2] = "LR";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),f4.1,\002, A,\002,i3,\002, B,\002,i"
	    "3,\002,\002,f4.1,\002, C,\002,i3,\002)   \002,\002 .\002)";
    static char fmt_9994[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
	    "N VALID CALL *\002,\002******\002)";
    static char fmt_9998[] = "(\002 ******* FATAL ERROR - PARAMETER NUMBER"
	    " \002,i2,\002 WAS CH\002,\002ANGED INCORRECTLY *******\002)";
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE COMPUTATIONAL TE"
	    "STS (\002,i6,\002 CALL\002,\002S)\002)";
    static char fmt_9997[] = "(\002 \002,a6,\002 COMPLETED THE COMPUTATIONAL"
	    " TESTS (\002,i6,\002 C\002,\002ALLS)\002,/\002 ******* BUT WITH "
	    "MAXIMUM TEST RATIO\002,f8.2,\002 - SUSPECT *******\002)";
    static char fmt_9996[] = "(\002 ******* \002,a6,\002 FAILED ON CALL NUMB"
	    "ER:\002)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, m, n, ia, ib, na, nc, im, in, ms, ns, laa, lbb, lda, lcc,
	    ldb, ldc;
    extern logical lde_(doublereal *, doublereal *, integer *);
    integer ics;
    doublereal als, bls;
    integer icu;
    doublereal err, beta;
    integer ldas, ldbs, ldcs;
    logical same;
    char side[1];
    logical left, null;
    char uplo[1];
    extern /* Subroutine */ int dmake_(char *, char *, char *, integer *,
	    integer *, doublereal *, integer *, doublereal *, integer *,
	    logical *, doublereal *, ftnlen, ftnlen, ftnlen);
    doublereal alpha;
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen);
    logical isame[13];
    char sides[1];
    integer nargs;
    logical reset;
    extern /* Subroutine */ int dsymm_(char *, char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *, ftnlen, ftnlen);
    char uplos[1];
    extern logical lderes_(char *, char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, ftnlen, ftnlen);
    doublereal errmax;

    /* Fortran I/O blocks */
    static cilist io___171 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___172 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___175 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___177 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___178 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___179 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___180 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests DSYMM. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --idim;
    --alf;
    --bet;
    --g;
    --ct;
    --cs;
    --cc;
    c_dim1 = *nmax;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --bs;
    --bb;
    b_dim1 = *nmax;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
/*     .. Executable Statements .. */

    nargs = 12;
    nc = 0;
    reset = TRUE_;
    errmax = 0.;

    i__1 = *nidim;
    for (im = 1; im <= i__1; ++im) {
	m = idim[im];

	i__2 = *nidim;
	for (in = 1; in <= i__2; ++in) {
	    n = idim[in];
/*           Set LDC to 1 more than minimum value if room. */
	    ldc = m;
	    if (ldc < *nmax) {
		++ldc;
	    }
/*           Skip tests if not enough room. */
	    if (ldc > *nmax) {
		goto L90;
	    }
	    lcc = ldc * n;
	    null = n <= 0 || m <= 0;

/*           Set LDB to 1 more than minimum value if room. */
	    ldb = m;
	    if (ldb < *nmax) {
		++ldb;
	    }
/*           Skip tests if not enough room. */
	    if (ldb > *nmax) {
		goto L90;
	    }
	    lbb = ldb * n;

/*           Generate the matrix B. */

	    dmake_("GE", " ", " ", &m, &n, &b[b_offset], nmax, &bb[1], &ldb, &
		    reset, &c_b86, (ftnlen)2, (ftnlen)1, (ftnlen)1);

	    for (ics = 1; ics <= 2; ++ics) {
		*(unsigned char *)side = *(unsigned char *)&ichs[ics - 1];
		left = *(unsigned char *)side == 'L';

		if (left) {
		    na = m;
		} else {
		    na = n;
		}
/*              Set LDA to 1 more than minimum value if room. */
		lda = na;
		if (lda < *nmax) {
		    ++lda;
		}
/*              Skip tests if not enough room. */
		if (lda > *nmax) {
		    goto L80;
		}
		laa = lda * na;

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];

/*                 Generate the symmetric matrix A. */

		    dmake_("SY", uplo, " ", &na, &na, &a[a_offset], nmax, &aa[
			    1], &lda, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			alpha = alf[ia];

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    beta = bet[ib];

/*                       Generate the matrix C. */

			    dmake_("GE", " ", " ", &m, &n, &c__[c_offset],
				    nmax, &cc[1], &ldc, &reset, &c_b86, (
				    ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the */
/*                       subroutine. */

			    *(unsigned char *)sides = *(unsigned char *)side;
			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    ms = m;
			    ns = n;
			    als = alpha;
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				as[i__] = aa[i__];
/* L10: */
			    }
			    ldas = lda;
			    i__5 = lbb;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				bs[i__] = bb[i__];
/* L20: */
			    }
			    ldbs = ldb;
			    bls = beta;
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				cs[i__] = cc[i__];
/* L30: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (*trace) {
				io___171.ciunit = *ntra;
				s_wsfe(&io___171);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, side, (ftnlen)1);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, (char *)&m, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
			    if (*rewi) {
				al__1.aerr = 0;
				al__1.aunit = *ntra;
				f_rew(&al__1);
			    }
			    dsymm_(side, uplo, &m, &n, &alpha, &aa[1], &lda, &
				    bb[1], &ldb, &beta, &cc[1], &ldc, (ftnlen)
				    1, (ftnlen)1);

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___172.ciunit = *nout;
				s_wsfe(&io___172);
				e_wsfe();
				*fatal = TRUE_;
				goto L110;
			    }

/*                       See what data changed inside subroutines. */

			    isame[0] = *(unsigned char *)sides == *(unsigned
				    char *)side;
			    isame[1] = *(unsigned char *)uplos == *(unsigned
				    char *)uplo;
			    isame[2] = ms == m;
			    isame[3] = ns == n;
			    isame[4] = als == alpha;
			    isame[5] = lde_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    isame[7] = lde_(&bs[1], &bb[1], &lbb);
			    isame[8] = ldbs == ldb;
			    isame[9] = bls == beta;
			    if (null) {
				isame[10] = lde_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[10] = lderes_("GE", " ", &m, &n, &cs[1],
					 &cc[1], &ldc, (ftnlen)2, (ftnlen)1);
			    }
			    isame[11] = ldcs == ldc;

/*                       If data was incorrectly changed, report and */
/*                       return. */

			    same = TRUE_;
			    i__5 = nargs;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				same = same && isame[i__ - 1];
				if (! isame[i__ - 1]) {
				    io___175.ciunit = *nout;
				    s_wsfe(&io___175);
				    do_fio(&c__1, (char *)&i__, (ftnlen)
					    sizeof(integer));
				    e_wsfe();
				}
/* L40: */
			    }
			    if (! same) {
				*fatal = TRUE_;
				goto L110;
			    }

			    if (! null) {

/*                          Check the result. */

				if (left) {
				    dmmch_("N", "N", &m, &n, &m, &alpha, &a[
					    a_offset], nmax, &b[b_offset],
					    nmax, &beta, &c__[c_offset], nmax,
					     &ct[1], &g[1], &cc[1], &ldc, eps,
					     &err, fatal, nout, &c_true, (
					    ftnlen)1, (ftnlen)1);
				} else {
				    dmmch_("N", "N", &m, &n, &n, &alpha, &b[
					    b_offset], nmax, &a[a_offset],
					    nmax, &beta, &c__[c_offset], nmax,
					     &ct[1], &g[1], &cc[1], &ldc, eps,
					     &err, fatal, nout, &c_true, (
					    ftnlen)1, (ftnlen)1);
				}
				errmax = max(errmax,err);
/*                          If got really bad answer, report and */
/*                          return. */
				if (*fatal) {
				    goto L110;
				}
			    }

/* L50: */
			}

/* L60: */
		    }

/* L70: */
		}

L80:
		;
	    }

L90:
	    ;
	}

/* L100: */
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___177.ciunit = *nout;
	s_wsfe(&io___177);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___178.ciunit = *nout;
	s_wsfe(&io___178);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L120;

L110:
    io___179.ciunit = *nout;
    s_wsfe(&io___179);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___180.ciunit = *nout;
    s_wsfe(&io___180);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, side, (ftnlen)1);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L120:
    return 0;


/*     End of DCHK2. */

} /* dchk2_ */

/* Subroutine */ int dchk3_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublereal *alf,
	integer *nmax, doublereal *a, doublereal *aa, doublereal *as,
	doublereal *b, doublereal *bb, doublereal *bs, doublereal *ct,
	doublereal *g, doublereal *c__, ftnlen sname_len)
{
    /* Initialized data */

    static char ichu[2] = "UL";
    static char icht[3] = "NTC";
    static char ichd[2] = "UN";
    static char ichs[2] = "LR";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002(\002,4(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),f4.1,\002, A,\002,i3,\002, B,\002,i"
	    "3,\002)        .\002)";
    static char fmt_9994[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
	    "N VALID CALL *\002,\002******\002)";
    static char fmt_9998[] = "(\002 ******* FATAL ERROR - PARAMETER NUMBER"
	    " \002,i2,\002 WAS CH\002,\002ANGED INCORRECTLY *******\002)";
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE COMPUTATIONAL TE"
	    "STS (\002,i6,\002 CALL\002,\002S)\002)";
    static char fmt_9997[] = "(\002 \002,a6,\002 COMPLETED THE COMPUTATIONAL"
	    " TESTS (\002,i6,\002 C\002,\002ALLS)\002,/\002 ******* BUT WITH "
	    "MAXIMUM TEST RATIO\002,f8.2,\002 - SUSPECT *******\002)";
    static char fmt_9996[] = "(\002 ******* \002,a6,\002 FAILED ON CALL NUMB"
	    "ER:\002)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);

    /* Local variables */
    integer i__, j, m, n, ia, na, nc, im, in, ms, ns, laa, icd, lbb, lda, ldb;
    extern logical lde_(doublereal *, doublereal *, integer *);
    integer ics;
    doublereal als;
    integer ict, icu;
    doublereal err;
    char diag[1];
    integer ldas, ldbs;
    logical same;
    char side[1];
    logical left, null;
    char uplo[1];
    extern /* Subroutine */ int dmake_(char *, char *, char *, integer *,
	    integer *, doublereal *, integer *, doublereal *, integer *,
	    logical *, doublereal *, ftnlen, ftnlen, ftnlen);
    doublereal alpha;
    char diags[1];
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen);
    logical isame[13];
    char sides[1];
    integer nargs;
    logical reset;
    extern /* Subroutine */ int dtrmm_(char *, char *, char *, char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *, ftnlen, ftnlen, ftnlen, ftnlen), dtrsm_(
	    char *, char *, char *, char *, integer *, integer *, doublereal *
	    , doublereal *, integer *, doublereal *, integer *, ftnlen,
	    ftnlen, ftnlen, ftnlen);
    char uplos[1];
    extern logical lderes_(char *, char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, ftnlen, ftnlen);
    char tranas[1], transa[1];
    doublereal errmax;

    /* Fortran I/O blocks */
    static cilist io___221 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___222 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___223 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___226 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___228 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___229 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___230 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___231 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests DTRMM and DTRSM. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --idim;
    --alf;
    c_dim1 = *nmax;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --g;
    --ct;
    --bs;
    --bb;
    b_dim1 = *nmax;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
/*     .. Executable Statements .. */

    nargs = 11;
    nc = 0;
    reset = TRUE_;
    errmax = 0.;
/*     Set up zero matrix for DMMCH. */
    i__1 = *nmax;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nmax;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    c__[i__ + j * c_dim1] = 0.;
/* L10: */
	}
/* L20: */
    }

    i__1 = *nidim;
    for (im = 1; im <= i__1; ++im) {
	m = idim[im];

	i__2 = *nidim;
	for (in = 1; in <= i__2; ++in) {
	    n = idim[in];
/*           Set LDB to 1 more than minimum value if room. */
	    ldb = m;
	    if (ldb < *nmax) {
		++ldb;
	    }
/*           Skip tests if not enough room. */
	    if (ldb > *nmax) {
		goto L130;
	    }
	    lbb = ldb * n;
	    null = m <= 0 || n <= 0;

	    for (ics = 1; ics <= 2; ++ics) {
		*(unsigned char *)side = *(unsigned char *)&ichs[ics - 1];
		left = *(unsigned char *)side == 'L';
		if (left) {
		    na = m;
		} else {
		    na = n;
		}
/*              Set LDA to 1 more than minimum value if room. */
		lda = na;
		if (lda < *nmax) {
		    ++lda;
		}
/*              Skip tests if not enough room. */
		if (lda > *nmax) {
		    goto L130;
		}
		laa = lda * na;

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];

		    for (ict = 1; ict <= 3; ++ict) {
			*(unsigned char *)transa = *(unsigned char *)&icht[
				ict - 1];

			for (icd = 1; icd <= 2; ++icd) {
			    *(unsigned char *)diag = *(unsigned char *)&ichd[
				    icd - 1];

			    i__3 = *nalf;
			    for (ia = 1; ia <= i__3; ++ia) {
				alpha = alf[ia];

/*                          Generate the matrix A. */

				dmake_("TR", uplo, diag, &na, &na, &a[
					a_offset], nmax, &aa[1], &lda, &reset,
					 &c_b86, (ftnlen)2, (ftnlen)1, (
					ftnlen)1);

/*                          Generate the matrix B. */

				dmake_("GE", " ", " ", &m, &n, &b[b_offset],
					nmax, &bb[1], &ldb, &reset, &c_b86, (
					ftnlen)2, (ftnlen)1, (ftnlen)1);

				++nc;

/*                          Save every datum before calling the */
/*                          subroutine. */

				*(unsigned char *)sides = *(unsigned char *)
					side;
				*(unsigned char *)uplos = *(unsigned char *)
					uplo;
				*(unsigned char *)tranas = *(unsigned char *)
					transa;
				*(unsigned char *)diags = *(unsigned char *)
					diag;
				ms = m;
				ns = n;
				als = alpha;
				i__4 = laa;
				for (i__ = 1; i__ <= i__4; ++i__) {
				    as[i__] = aa[i__];
/* L30: */
				}
				ldas = lda;
				i__4 = lbb;
				for (i__ = 1; i__ <= i__4; ++i__) {
				    bs[i__] = bb[i__];
/* L40: */
				}
				ldbs = ldb;

/*                          Call the subroutine. */

				if (s_cmp(sname + 3, "MM", (ftnlen)2, (ftnlen)
					2) == 0) {
				    if (*trace) {
					io___221.ciunit = *ntra;
					s_wsfe(&io___221);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, side, (ftnlen)1);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, transa, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&m, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&alpha, (ftnlen)
						sizeof(doublereal));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&ldb, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    dtrmm_(side, uplo, transa, diag, &m, &n, &
					    alpha, &aa[1], &lda, &bb[1], &ldb,
					     (ftnlen)1, (ftnlen)1, (ftnlen)1,
					    (ftnlen)1);
				} else if (s_cmp(sname + 3, "SM", (ftnlen)2, (
					ftnlen)2) == 0) {
				    if (*trace) {
					io___222.ciunit = *ntra;
					s_wsfe(&io___222);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, side, (ftnlen)1);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, transa, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&m, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&alpha, (ftnlen)
						sizeof(doublereal));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&ldb, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    dtrsm_(side, uplo, transa, diag, &m, &n, &
					    alpha, &aa[1], &lda, &bb[1], &ldb,
					     (ftnlen)1, (ftnlen)1, (ftnlen)1,
					    (ftnlen)1);
				}

/*                          Check if error-exit was taken incorrectly. */

				if (! infoc_1.ok) {
				    io___223.ciunit = *nout;
				    s_wsfe(&io___223);
				    e_wsfe();
				    *fatal = TRUE_;
				    goto L150;
				}

/*                          See what data changed inside subroutines. */

				isame[0] = *(unsigned char *)sides == *(
					unsigned char *)side;
				isame[1] = *(unsigned char *)uplos == *(
					unsigned char *)uplo;
				isame[2] = *(unsigned char *)tranas == *(
					unsigned char *)transa;
				isame[3] = *(unsigned char *)diags == *(
					unsigned char *)diag;
				isame[4] = ms == m;
				isame[5] = ns == n;
				isame[6] = als == alpha;
				isame[7] = lde_(&as[1], &aa[1], &laa);
				isame[8] = ldas == lda;
				if (null) {
				    isame[9] = lde_(&bs[1], &bb[1], &lbb);
				} else {
				    isame[9] = lderes_("GE", " ", &m, &n, &bs[
					    1], &bb[1], &ldb, (ftnlen)2, (
					    ftnlen)1);
				}
				isame[10] = ldbs == ldb;

/*                          If data was incorrectly changed, report and */
/*                          return. */

				same = TRUE_;
				i__4 = nargs;
				for (i__ = 1; i__ <= i__4; ++i__) {
				    same = same && isame[i__ - 1];
				    if (! isame[i__ - 1]) {
					io___226.ciunit = *nout;
					s_wsfe(&io___226);
					do_fio(&c__1, (char *)&i__, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
/* L50: */
				}
				if (! same) {
				    *fatal = TRUE_;
				    goto L150;
				}

				if (! null) {
				    if (s_cmp(sname + 3, "MM", (ftnlen)2, (
					    ftnlen)2) == 0) {

/*                                Check the result. */

					if (left) {
					    dmmch_(transa, "N", &m, &n, &m, &
						    alpha, &a[a_offset], nmax,
						     &b[b_offset], nmax, &
						    c_b86, &c__[c_offset],
						    nmax, &ct[1], &g[1], &bb[
						    1], &ldb, eps, &err,
						    fatal, nout, &c_true, (
						    ftnlen)1, (ftnlen)1);
					} else {
					    dmmch_("N", transa, &m, &n, &n, &
						    alpha, &b[b_offset], nmax,
						     &a[a_offset], nmax, &
						    c_b86, &c__[c_offset],
						    nmax, &ct[1], &g[1], &bb[
						    1], &ldb, eps, &err,
						    fatal, nout, &c_true, (
						    ftnlen)1, (ftnlen)1);
					}
				    } else if (s_cmp(sname + 3, "SM", (ftnlen)
					    2, (ftnlen)2) == 0) {

/*                                Compute approximation to original */
/*                                matrix. */

					i__4 = n;
					for (j = 1; j <= i__4; ++j) {
					    i__5 = m;
					    for (i__ = 1; i__ <= i__5; ++i__)
						    {
			  c__[i__ + j * c_dim1] = bb[i__ + (j - 1) * ldb];
			  bb[i__ + (j - 1) * ldb] = alpha * b[i__ + j *
				  b_dim1];
/* L60: */
					    }
/* L70: */
					}

					if (left) {
					    dmmch_(transa, "N", &m, &n, &m, &
						    c_b96, &a[a_offset], nmax,
						     &c__[c_offset], nmax, &
						    c_b86, &b[b_offset], nmax,
						     &ct[1], &g[1], &bb[1], &
						    ldb, eps, &err, fatal,
						    nout, &c_false, (ftnlen)1,
						     (ftnlen)1);
					} else {
					    dmmch_("N", transa, &m, &n, &n, &
						    c_b96, &c__[c_offset],
						    nmax, &a[a_offset], nmax,
						    &c_b86, &b[b_offset],
						    nmax, &ct[1], &g[1], &bb[
						    1], &ldb, eps, &err,
						    fatal, nout, &c_false, (
						    ftnlen)1, (ftnlen)1);
					}
				    }
				    errmax = max(errmax,err);
/*                             If got really bad answer, report and */
/*                             return. */
				    if (*fatal) {
					goto L150;
				    }
				}

/* L80: */
			    }

/* L90: */
			}

/* L100: */
		    }

/* L110: */
		}

/* L120: */
	    }

L130:
	    ;
	}

/* L140: */
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___228.ciunit = *nout;
	s_wsfe(&io___228);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___229.ciunit = *nout;
	s_wsfe(&io___229);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L160;

L150:
    io___230.ciunit = *nout;
    s_wsfe(&io___230);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___231.ciunit = *nout;
    s_wsfe(&io___231);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, side, (ftnlen)1);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, transa, (ftnlen)1);
    do_fio(&c__1, diag, (ftnlen)1);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    e_wsfe();

L160:
    return 0;


/*     End of DCHK3. */

} /* dchk3_ */

/* Subroutine */ int dchk4_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublereal *alf,
	integer *nbet, doublereal *bet, integer *nmax, doublereal *a,
	doublereal *aa, doublereal *as, doublereal *b, doublereal *bb,
	doublereal *bs, doublereal *c__, doublereal *cc, doublereal *cs,
	doublereal *ct, doublereal *g, ftnlen sname_len)
{
    /* Initialized data */

    static char icht[3] = "NTC";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),f4.1,\002, A,\002,i3,\002,\002,f4.1,"
	    "\002, C,\002,i3,\002)           .\002)";
    static char fmt_9993[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
	    "N VALID CALL *\002,\002******\002)";
    static char fmt_9998[] = "(\002 ******* FATAL ERROR - PARAMETER NUMBER"
	    " \002,i2,\002 WAS CH\002,\002ANGED INCORRECTLY *******\002)";
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE COMPUTATIONAL TE"
	    "STS (\002,i6,\002 CALL\002,\002S)\002)";
    static char fmt_9997[] = "(\002 \002,a6,\002 COMPLETED THE COMPUTATIONAL"
	    " TESTS (\002,i6,\002 C\002,\002ALLS)\002,/\002 ******* BUT WITH "
	    "MAXIMUM TEST RATIO\002,f8.2,\002 - SUSPECT *******\002)";
    static char fmt_9995[] = "(\002      THESE ARE THE RESULTS FOR COLUMN"
	    " \002,i3)";
    static char fmt_9996[] = "(\002 ******* \002,a6,\002 FAILED ON CALL NUMB"
	    "ER:\002)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, j, k, n, ia, ib, jc, ma, na, nc, ik, in, jj, lj, ks, ns, laa,
	     lda, lcc, ldc;
    extern logical lde_(doublereal *, doublereal *, integer *);
    doublereal als;
    integer ict, icu;
    doublereal err, beta;
    integer ldas, ldcs;
    logical same;
    doublereal bets;
    logical tran, null;
    char uplo[1];
    extern /* Subroutine */ int dmake_(char *, char *, char *, integer *,
	    integer *, doublereal *, integer *, doublereal *, integer *,
	    logical *, doublereal *, ftnlen, ftnlen, ftnlen);
    doublereal alpha;
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen);
    logical isame[13];
    integer nargs;
    logical reset;
    char trans[1];
    logical upper;
    extern /* Subroutine */ int dsyrk_(char *, char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     integer *, ftnlen, ftnlen);
    char uplos[1];
    extern logical lderes_(char *, char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, ftnlen, ftnlen);
    doublereal errmax;
    char transs[1];

    /* Fortran I/O blocks */
    static cilist io___268 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___269 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___272 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___278 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___279 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___280 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___281 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___282 = { 0, 0, 0, fmt_9994, 0 };



/*  Tests DSYRK. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --idim;
    --alf;
    --bet;
    --g;
    --ct;
    --cs;
    --cc;
    c_dim1 = *nmax;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --bs;
    --bb;
    b_dim1 = *nmax;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
/*     .. Executable Statements .. */

    nargs = 10;
    nc = 0;
    reset = TRUE_;
    errmax = 0.;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
/*        Set LDC to 1 more than minimum value if room. */
	ldc = n;
	if (ldc < *nmax) {
	    ++ldc;
	}
/*        Skip tests if not enough room. */
	if (ldc > *nmax) {
	    goto L100;
	}
	lcc = ldc * n;
	null = n <= 0;

	i__2 = *nidim;
	for (ik = 1; ik <= i__2; ++ik) {
	    k = idim[ik];

	    for (ict = 1; ict <= 3; ++ict) {
		*(unsigned char *)trans = *(unsigned char *)&icht[ict - 1];
		tran = *(unsigned char *)trans == 'T' || *(unsigned char *)
			trans == 'C';
		if (tran) {
		    ma = k;
		    na = n;
		} else {
		    ma = n;
		    na = k;
		}
/*              Set LDA to 1 more than minimum value if room. */
		lda = ma;
		if (lda < *nmax) {
		    ++lda;
		}
/*              Skip tests if not enough room. */
		if (lda > *nmax) {
		    goto L80;
		}
		laa = lda * na;

/*              Generate the matrix A. */

		dmake_("GE", " ", " ", &ma, &na, &a[a_offset], nmax, &aa[1], &
			lda, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (ftnlen)1);

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];
		    upper = *(unsigned char *)uplo == 'U';

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			alpha = alf[ia];

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    beta = bet[ib];

/*                       Generate the matrix C. */

			    dmake_("SY", uplo, " ", &n, &n, &c__[c_offset],
				    nmax, &cc[1], &ldc, &reset, &c_b86, (
				    ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the subroutine. */

			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    *(unsigned char *)transs = *(unsigned char *)
				    trans;
			    ns = n;
			    ks = k;
			    als = alpha;
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				as[i__] = aa[i__];
/* L10: */
			    }
			    ldas = lda;
			    bets = beta;
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				cs[i__] = cc[i__];
/* L20: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (*trace) {
				io___268.ciunit = *ntra;
				s_wsfe(&io___268);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, trans, (ftnlen)1);
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
			    if (*rewi) {
				al__1.aerr = 0;
				al__1.aunit = *ntra;
				f_rew(&al__1);
			    }
			    dsyrk_(uplo, trans, &n, &k, &alpha, &aa[1], &lda,
				    &beta, &cc[1], &ldc, (ftnlen)1, (ftnlen)1)
				    ;

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___269.ciunit = *nout;
				s_wsfe(&io___269);
				e_wsfe();
				*fatal = TRUE_;
				goto L120;
			    }

/*                       See what data changed inside subroutines. */

			    isame[0] = *(unsigned char *)uplos == *(unsigned
				    char *)uplo;
			    isame[1] = *(unsigned char *)transs == *(unsigned
				    char *)trans;
			    isame[2] = ns == n;
			    isame[3] = ks == k;
			    isame[4] = als == alpha;
			    isame[5] = lde_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    isame[7] = bets == beta;
			    if (null) {
				isame[8] = lde_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[8] = lderes_("SY", uplo, &n, &n, &cs[1],
					 &cc[1], &ldc, (ftnlen)2, (ftnlen)1);
			    }
			    isame[9] = ldcs == ldc;

/*                       If data was incorrectly changed, report and */
/*                       return. */

			    same = TRUE_;
			    i__5 = nargs;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				same = same && isame[i__ - 1];
				if (! isame[i__ - 1]) {
				    io___272.ciunit = *nout;
				    s_wsfe(&io___272);
				    do_fio(&c__1, (char *)&i__, (ftnlen)
					    sizeof(integer));
				    e_wsfe();
				}
/* L30: */
			    }
			    if (! same) {
				*fatal = TRUE_;
				goto L120;
			    }

			    if (! null) {

/*                          Check the result column by column. */

				jc = 1;
				i__5 = n;
				for (j = 1; j <= i__5; ++j) {
				    if (upper) {
					jj = 1;
					lj = j;
				    } else {
					jj = j;
					lj = n - j + 1;
				    }
				    if (tran) {
					dmmch_("T", "N", &lj, &c__1, &k, &
						alpha, &a[jj * a_dim1 + 1],
						nmax, &a[j * a_dim1 + 1],
						nmax, &beta, &c__[jj + j *
						c_dim1], nmax, &ct[1], &g[1],
						&cc[jc], &ldc, eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1, (ftnlen)1);
				    } else {
					dmmch_("N", "T", &lj, &c__1, &k, &
						alpha, &a[jj + a_dim1], nmax,
						&a[j + a_dim1], nmax, &beta, &
						c__[jj + j * c_dim1], nmax, &
						ct[1], &g[1], &cc[jc], &ldc,
						eps, &err, fatal, nout, &
						c_true, (ftnlen)1, (ftnlen)1);
				    }
				    if (upper) {
					jc += ldc;
				    } else {
					jc = jc + ldc + 1;
				    }
				    errmax = max(errmax,err);
/*                             If got really bad answer, report and */
/*                             return. */
				    if (*fatal) {
					goto L110;
				    }
/* L40: */
				}
			    }

/* L50: */
			}

/* L60: */
		    }

/* L70: */
		}

L80:
		;
	    }

/* L90: */
	}

L100:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___278.ciunit = *nout;
	s_wsfe(&io___278);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___279.ciunit = *nout;
	s_wsfe(&io___279);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L130;

L110:
    if (n > 1) {
	io___280.ciunit = *nout;
	s_wsfe(&io___280);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L120:
    io___281.ciunit = *nout;
    s_wsfe(&io___281);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___282.ciunit = *nout;
    s_wsfe(&io___282);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, trans, (ftnlen)1);
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L130:
    return 0;


/*     End of DCHK4. */

} /* dchk4_ */

/* Subroutine */ int dchk5_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublereal *alf,
	integer *nbet, doublereal *bet, integer *nmax, doublereal *ab,
	doublereal *aa, doublereal *as, doublereal *bb, doublereal *bs,
	doublereal *c__, doublereal *cc, doublereal *cs, doublereal *ct,
	doublereal *g, doublereal *w, ftnlen sname_len)
{
    /* Initialized data */

    static char icht[3] = "NTC";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),f4.1,\002, A,\002,i3,\002, B,\002,i"
	    "3,\002,\002,f4.1,\002, C,\002,i3,\002)   \002,\002 .\002)";
    static char fmt_9993[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
	    "N VALID CALL *\002,\002******\002)";
    static char fmt_9998[] = "(\002 ******* FATAL ERROR - PARAMETER NUMBER"
	    " \002,i2,\002 WAS CH\002,\002ANGED INCORRECTLY *******\002)";
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE COMPUTATIONAL TE"
	    "STS (\002,i6,\002 CALL\002,\002S)\002)";
    static char fmt_9997[] = "(\002 \002,a6,\002 COMPLETED THE COMPUTATIONAL"
	    " TESTS (\002,i6,\002 C\002,\002ALLS)\002,/\002 ******* BUT WITH "
	    "MAXIMUM TEST RATIO\002,f8.2,\002 - SUSPECT *******\002)";
    static char fmt_9995[] = "(\002      THESE ARE THE RESULTS FOR COLUMN"
	    " \002,i3)";
    static char fmt_9996[] = "(\002 ******* \002,a6,\002 FAILED ON CALL NUMB"
	    "ER:\002)";

    /* System generated locals */
    integer c_dim1, c_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, j, k, n, ia, ib, jc, ma, na, nc, ik, in, jj, lj, ks, ns, laa,
	     lbb, lda, lcc, ldb, ldc;
    extern logical lde_(doublereal *, doublereal *, integer *);
    doublereal als;
    integer ict, icu;
    doublereal err;
    integer jjab;
    doublereal beta;
    integer ldas, ldbs, ldcs;
    logical same;
    doublereal bets;
    logical tran, null;
    char uplo[1];
    extern /* Subroutine */ int dmake_(char *, char *, char *, integer *,
	    integer *, doublereal *, integer *, doublereal *, integer *,
	    logical *, doublereal *, ftnlen, ftnlen, ftnlen);
    doublereal alpha;
    extern /* Subroutine */ int dmmch_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     logical *, integer *, logical *, ftnlen, ftnlen);
    logical isame[13];
    integer nargs;
    logical reset;
    char trans[1];
    logical upper;
    char uplos[1];
    extern /* Subroutine */ int dsyr2k_(char *, char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *, ftnlen, ftnlen);
    extern logical lderes_(char *, char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, ftnlen, ftnlen);
    doublereal errmax;
    char transs[1];

    /* Fortran I/O blocks */
    static cilist io___322 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___323 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___326 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___333 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___334 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___335 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___336 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___337 = { 0, 0, 0, fmt_9994, 0 };



/*  Tests DSYR2K. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --idim;
    --alf;
    --bet;
    --w;
    --g;
    --ct;
    --cs;
    --cc;
    c_dim1 = *nmax;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --bs;
    --bb;
    --as;
    --aa;
    --ab;

    /* Function Body */
/*     .. Executable Statements .. */

    nargs = 12;
    nc = 0;
    reset = TRUE_;
    errmax = 0.;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
/*        Set LDC to 1 more than minimum value if room. */
	ldc = n;
	if (ldc < *nmax) {
	    ++ldc;
	}
/*        Skip tests if not enough room. */
	if (ldc > *nmax) {
	    goto L130;
	}
	lcc = ldc * n;
	null = n <= 0;

	i__2 = *nidim;
	for (ik = 1; ik <= i__2; ++ik) {
	    k = idim[ik];

	    for (ict = 1; ict <= 3; ++ict) {
		*(unsigned char *)trans = *(unsigned char *)&icht[ict - 1];
		tran = *(unsigned char *)trans == 'T' || *(unsigned char *)
			trans == 'C';
		if (tran) {
		    ma = k;
		    na = n;
		} else {
		    ma = n;
		    na = k;
		}
/*              Set LDA to 1 more than minimum value if room. */
		lda = ma;
		if (lda < *nmax) {
		    ++lda;
		}
/*              Skip tests if not enough room. */
		if (lda > *nmax) {
		    goto L110;
		}
		laa = lda * na;

/*              Generate the matrix A. */

		if (tran) {
		    i__3 = *nmax << 1;
		    dmake_("GE", " ", " ", &ma, &na, &ab[1], &i__3, &aa[1], &
			    lda, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);
		} else {
		    dmake_("GE", " ", " ", &ma, &na, &ab[1], nmax, &aa[1], &
			    lda, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);
		}

/*              Generate the matrix B. */

		ldb = lda;
		lbb = laa;
		if (tran) {
		    i__3 = *nmax << 1;
		    dmake_("GE", " ", " ", &ma, &na, &ab[k + 1], &i__3, &bb[1]
			    , &ldb, &reset, &c_b86, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);
		} else {
		    dmake_("GE", " ", " ", &ma, &na, &ab[k * *nmax + 1], nmax,
			     &bb[1], &ldb, &reset, &c_b86, (ftnlen)2, (ftnlen)
			    1, (ftnlen)1);
		}

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];
		    upper = *(unsigned char *)uplo == 'U';

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			alpha = alf[ia];

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    beta = bet[ib];

/*                       Generate the matrix C. */

			    dmake_("SY", uplo, " ", &n, &n, &c__[c_offset],
				    nmax, &cc[1], &ldc, &reset, &c_b86, (
				    ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the subroutine. */

			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    *(unsigned char *)transs = *(unsigned char *)
				    trans;
			    ns = n;
			    ks = k;
			    als = alpha;
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				as[i__] = aa[i__];
/* L10: */
			    }
			    ldas = lda;
			    i__5 = lbb;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				bs[i__] = bb[i__];
/* L20: */
			    }
			    ldbs = ldb;
			    bets = beta;
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				cs[i__] = cc[i__];
/* L30: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (*trace) {
				io___322.ciunit = *ntra;
				s_wsfe(&io___322);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, trans, (ftnlen)1);
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
			    if (*rewi) {
				al__1.aerr = 0;
				al__1.aunit = *ntra;
				f_rew(&al__1);
			    }
			    dsyr2k_(uplo, trans, &n, &k, &alpha, &aa[1], &lda,
				     &bb[1], &ldb, &beta, &cc[1], &ldc, (
				    ftnlen)1, (ftnlen)1);

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___323.ciunit = *nout;
				s_wsfe(&io___323);
				e_wsfe();
				*fatal = TRUE_;
				goto L150;
			    }

/*                       See what data changed inside subroutines. */

			    isame[0] = *(unsigned char *)uplos == *(unsigned
				    char *)uplo;
			    isame[1] = *(unsigned char *)transs == *(unsigned
				    char *)trans;
			    isame[2] = ns == n;
			    isame[3] = ks == k;
			    isame[4] = als == alpha;
			    isame[5] = lde_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    isame[7] = lde_(&bs[1], &bb[1], &lbb);
			    isame[8] = ldbs == ldb;
			    isame[9] = bets == beta;
			    if (null) {
				isame[10] = lde_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[10] = lderes_("SY", uplo, &n, &n, &cs[1]
					, &cc[1], &ldc, (ftnlen)2, (ftnlen)1);
			    }
			    isame[11] = ldcs == ldc;

/*                       If data was incorrectly changed, report and */
/*                       return. */

			    same = TRUE_;
			    i__5 = nargs;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				same = same && isame[i__ - 1];
				if (! isame[i__ - 1]) {
				    io___326.ciunit = *nout;
				    s_wsfe(&io___326);
				    do_fio(&c__1, (char *)&i__, (ftnlen)
					    sizeof(integer));
				    e_wsfe();
				}
/* L40: */
			    }
			    if (! same) {
				*fatal = TRUE_;
				goto L150;
			    }

			    if (! null) {

/*                          Check the result column by column. */

				jjab = 1;
				jc = 1;
				i__5 = n;
				for (j = 1; j <= i__5; ++j) {
				    if (upper) {
					jj = 1;
					lj = j;
				    } else {
					jj = j;
					lj = n - j + 1;
				    }
				    if (tran) {
					i__6 = k;
					for (i__ = 1; i__ <= i__6; ++i__) {
					    w[i__] = ab[(j - 1 << 1) * *nmax
						    + k + i__];
					    w[k + i__] = ab[(j - 1 << 1) * *
						    nmax + i__];
/* L50: */
					}
					i__6 = k << 1;
					i__7 = *nmax << 1;
					i__8 = *nmax << 1;
					dmmch_("T", "N", &lj, &c__1, &i__6, &
						alpha, &ab[jjab], &i__7, &w[1]
						, &i__8, &beta, &c__[jj + j *
						c_dim1], nmax, &ct[1], &g[1],
						&cc[jc], &ldc, eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1, (ftnlen)1);
				    } else {
					i__6 = k;
					for (i__ = 1; i__ <= i__6; ++i__) {
					    w[i__] = ab[(k + i__ - 1) * *nmax
						    + j];
					    w[k + i__] = ab[(i__ - 1) * *nmax
						    + j];
/* L60: */
					}
					i__6 = k << 1;
					i__7 = *nmax << 1;
					dmmch_("N", "N", &lj, &c__1, &i__6, &
						alpha, &ab[jj], nmax, &w[1], &
						i__7, &beta, &c__[jj + j *
						c_dim1], nmax, &ct[1], &g[1],
						&cc[jc], &ldc, eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1, (ftnlen)1);
				    }
				    if (upper) {
					jc += ldc;
				    } else {
					jc = jc + ldc + 1;
					if (tran) {
					    jjab += *nmax << 1;
					}
				    }
				    errmax = max(errmax,err);
/*                             If got really bad answer, report and */
/*                             return. */
				    if (*fatal) {
					goto L140;
				    }
/* L70: */
				}
			    }

/* L80: */
			}

/* L90: */
		    }

/* L100: */
		}

L110:
		;
	    }

/* L120: */
	}

L130:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___333.ciunit = *nout;
	s_wsfe(&io___333);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___334.ciunit = *nout;
	s_wsfe(&io___334);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L160;

L140:
    if (n > 1) {
	io___335.ciunit = *nout;
	s_wsfe(&io___335);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L150:
    io___336.ciunit = *nout;
    s_wsfe(&io___336);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___337.ciunit = *nout;
    s_wsfe(&io___337);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, trans, (ftnlen)1);
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L160:
    return 0;


/*     End of DCHK5. */

} /* dchk5_ */

/* Subroutine */ int dchke_(integer *isnum, char *srnamt, integer *nout,
	ftnlen srnamt_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 \002,a6,\002 PASSED THE TESTS OF ERROR-E"
	    "XITS\002)";
    static char fmt_9998[] = "(\002 ******* \002,a6,\002 FAILED THE TESTS OF"
	    " ERROR-EXITS *****\002,\002**\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Local variables */
    doublereal a[2]	/* was [2][1] */, b[2]	/* was [2][1] */, c__[2]
	    /* was [2][1] */, beta, alpha;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, ftnlen, ftnlen),
	     dtrmm_(char *, char *, char *, char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    ftnlen, ftnlen, ftnlen, ftnlen), dsymm_(char *, char *, integer *,
	     integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, ftnlen, ftnlen),
	     dtrsm_(char *, char *, char *, char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    ftnlen, ftnlen, ftnlen, ftnlen), dsyrk_(char *, char *, integer *,
	     integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, integer *, ftnlen, ftnlen), dsyr2k_(char *, char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, integer *,
	    ftnlen, ftnlen), chkxer_(char *, integer *, integer *, logical *,
	    logical *, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___343 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___344 = { 0, 0, 0, fmt_9998, 0 };



/*  Tests the error exits from the Level 3 Blas. */
/*  Requires a special version of the error-handling routine XERBLA. */
/*  A, B and C should not need to be defined. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*  3-19-92:  Initialize ALPHA and BETA  (eca) */
/*  3-19-92:  Fix argument 12 in calls to SSYMM with INFOT = 9  (eca) */

/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */
/*     OK is set to .FALSE. by the special version of XERBLA or by CHKXER */
/*     if anything is wrong. */
    infoc_1.ok = TRUE_;
/*     LERR is set to .TRUE. by the special version of XERBLA each time */
/*     it is called, and is then tested and re-set by CHKXER. */
    infoc_1.lerr = FALSE_;

/*     Initialize ALPHA and BETA. */

    alpha = 1.;
    beta = 2.;

    switch (*isnum) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L40;
	case 5:  goto L50;
	case 6:  goto L60;
    }
L10:
    infoc_1.infot = 1;
    dgemm_("/", "N", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 1;
    dgemm_("/", "T", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dgemm_("N", "/", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dgemm_("T", "/", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dgemm_("N", "N", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dgemm_("N", "T", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dgemm_("T", "N", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dgemm_("T", "T", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dgemm_("N", "N", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dgemm_("N", "T", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dgemm_("T", "N", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dgemm_("T", "T", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dgemm_("N", "N", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dgemm_("N", "T", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dgemm_("T", "N", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dgemm_("T", "T", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    dgemm_("N", "N", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    dgemm_("N", "T", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    dgemm_("T", "N", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__2, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    dgemm_("T", "T", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dgemm_("N", "N", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dgemm_("T", "N", &c__0, &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dgemm_("N", "T", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dgemm_("T", "T", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    dgemm_("N", "N", &c__2, &c__0, &c__0, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    dgemm_("N", "T", &c__2, &c__0, &c__0, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    dgemm_("T", "N", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    dgemm_("T", "T", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L70;
L20:
    infoc_1.infot = 1;
    dsymm_("/", "U", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dsymm_("L", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsymm_("L", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsymm_("R", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsymm_("L", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsymm_("R", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsymm_("L", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsymm_("R", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsymm_("L", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsymm_("R", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsymm_("R", "U", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsymm_("R", "L", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsymm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsymm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsymm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsymm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L70;
L30:
    infoc_1.infot = 1;
    dtrmm_("/", "U", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dtrmm_("L", "/", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dtrmm_("L", "U", "/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dtrmm_("L", "U", "N", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("L", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("L", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("R", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("R", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("L", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("L", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("R", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrmm_("R", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("L", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("L", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("R", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("R", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("L", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("L", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("R", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrmm_("R", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("R", "U", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("R", "U", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("R", "L", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrmm_("R", "L", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("R", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("R", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("R", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrmm_("R", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L70;
L40:
    infoc_1.infot = 1;
    dtrsm_("/", "U", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dtrsm_("L", "/", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dtrsm_("L", "U", "/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dtrsm_("L", "U", "N", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("L", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("L", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("R", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("R", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("L", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("L", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("R", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    dtrsm_("R", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("L", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("L", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("R", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("R", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("L", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("L", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("R", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    dtrsm_("R", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("R", "U", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("R", "U", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("R", "L", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dtrsm_("R", "L", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("R", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("R", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("R", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    dtrsm_("R", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L70;
L50:
    infoc_1.infot = 1;
    dsyrk_("/", "N", &c__0, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dsyrk_("U", "/", &c__0, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyrk_("U", "N", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyrk_("U", "T", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyrk_("L", "N", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyrk_("L", "T", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyrk_("U", "N", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyrk_("U", "T", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyrk_("L", "N", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyrk_("L", "T", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyrk_("U", "N", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyrk_("U", "T", &c__0, &c__2, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyrk_("L", "N", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyrk_("L", "T", &c__0, &c__2, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dsyrk_("U", "N", &c__2, &c__0, &alpha, a, &c__2, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dsyrk_("U", "T", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dsyrk_("L", "N", &c__2, &c__0, &alpha, a, &c__2, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    dsyrk_("L", "T", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L70;
L60:
    infoc_1.infot = 1;
    dsyr2k_("/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    dsyr2k_("U", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyr2k_("U", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyr2k_("U", "T", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyr2k_("L", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    dsyr2k_("L", "T", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyr2k_("U", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyr2k_("U", "T", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyr2k_("L", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    dsyr2k_("L", "T", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyr2k_("U", "T", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    dsyr2k_("L", "T", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsyr2k_("U", "T", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    dsyr2k_("L", "T", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsyr2k_("U", "T", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    dsyr2k_("L", "T", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);

L70:
    if (infoc_1.ok) {
	io___343.ciunit = *nout;
	s_wsfe(&io___343);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    } else {
	io___344.ciunit = *nout;
	s_wsfe(&io___344);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    }
    return 0;


/*     End of DCHKE. */

} /* dchke_ */

/* Subroutine */ int dmake_(char *type__, char *uplo, char *diag, integer *m,
	integer *n, doublereal *a, integer *nmax, doublereal *aa, integer *
	lda, logical *reset, doublereal *transl, ftnlen type_len, ftnlen
	uplo_len, ftnlen diag_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen);

    /* Local variables */
    integer i__, j;
    logical gen, tri, sym;
    extern doublereal dbeg_(logical *);
    integer ibeg, iend;
    logical unit, lower, upper;


/*  Generates values for an M by N matrix A. */
/*  Stores the values in the array AA in the data structure required */
/*  by the routine, with unwanted elements set to rogue value. */

/*  TYPE is 'GE', 'SY' or 'TR'. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --aa;

    /* Function Body */
    gen = s_cmp(type__, "GE", (ftnlen)2, (ftnlen)2) == 0;
    sym = s_cmp(type__, "SY", (ftnlen)2, (ftnlen)2) == 0;
    tri = s_cmp(type__, "TR", (ftnlen)2, (ftnlen)2) == 0;
    upper = (sym || tri) && *(unsigned char *)uplo == 'U';
    lower = (sym || tri) && *(unsigned char *)uplo == 'L';
    unit = tri && *(unsigned char *)diag == 'U';

/*     Generate data in array A. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (gen || upper && i__ <= j || lower && i__ >= j) {
		a[i__ + j * a_dim1] = dbeg_(reset) + *transl;
		if (i__ != j) {
/*                 Set some elements to zero */
		    if (*n > 3 && j == *n / 2) {
			a[i__ + j * a_dim1] = 0.;
		    }
		    if (sym) {
			a[j + i__ * a_dim1] = a[i__ + j * a_dim1];
		    } else if (tri) {
			a[j + i__ * a_dim1] = 0.;
		    }
		}
	    }
/* L10: */
	}
	if (tri) {
	    a[j + j * a_dim1] += 1.;
	}
	if (unit) {
	    a[j + j * a_dim1] = 1.;
	}
/* L20: */
    }

/*     Store elements in array AS in data structure required by routine. */

    if (s_cmp(type__, "GE", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = a[i__ + j * a_dim1];
/* L30: */
	    }
	    i__2 = *lda;
	    for (i__ = *m + 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10;
/* L40: */
	    }
/* L50: */
	}
    } else if (s_cmp(type__, "SY", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__,
	     "TR", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (upper) {
		ibeg = 1;
		if (unit) {
		    iend = j - 1;
		} else {
		    iend = j;
		}
	    } else {
		if (unit) {
		    ibeg = j + 1;
		} else {
		    ibeg = j;
		}
		iend = *n;
	    }
	    i__2 = ibeg - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10;
/* L60: */
	    }
	    i__2 = iend;
	    for (i__ = ibeg; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = a[i__ + j * a_dim1];
/* L70: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10;
/* L80: */
	    }
/* L90: */
	}
    }
    return 0;

/*     End of DMAKE. */

} /* dmake_ */

/* Subroutine */ int dmmch_(char *transa, char *transb, integer *m, integer *
	n, integer *kk, doublereal *alpha, doublereal *a, integer *lda,
	doublereal *b, integer *ldb, doublereal *beta, doublereal *c__,
	integer *ldc, doublereal *ct, doublereal *g, doublereal *cc, integer *
	ldcc, doublereal *eps, doublereal *err, logical *fatal, integer *nout,
	 logical *mv, ftnlen transa_len, ftnlen transb_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ******* FATAL ERROR - COMPUTED RESULT IS"
	    " LESS THAN HAL\002,\002F ACCURATE *******\002,/\002           EX"
	    "PECTED RESULT   COMPU\002,\002TED RESULT\002)";
    static char fmt_9998[] = "(1x,i7,2g18.6)";
    static char fmt_9997[] = "(\002      THESE ARE THE RESULTS FOR COLUMN"
	    " \002,i3)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, cc_dim1,
	    cc_offset, i__1, i__2, i__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer i__, j, k;
    doublereal erri;
    logical trana, tranb;

    /* Fortran I/O blocks */
    static cilist io___361 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___362 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___363 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___364 = { 0, 0, 0, fmt_9997, 0 };



/*  Checks the results of the computational tests. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --ct;
    --g;
    cc_dim1 = *ldcc;
    cc_offset = 1 + cc_dim1;
    cc -= cc_offset;

    /* Function Body */
    trana = *(unsigned char *)transa == 'T' || *(unsigned char *)transa ==
	    'C';
    tranb = *(unsigned char *)transb == 'T' || *(unsigned char *)transb ==
	    'C';

/*     Compute expected result, one column at a time, in CT using data */
/*     in A, B and C. */
/*     Compute gauges in G. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {

	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ct[i__] = 0.;
	    g[i__] = 0.;
/* L10: */
	}
	if (! trana && ! tranb) {
	    i__2 = *kk;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    ct[i__] += a[i__ + k * a_dim1] * b[k + j * b_dim1];
		    g[i__] += (d__1 = a[i__ + k * a_dim1], abs(d__1)) * (d__2
			    = b[k + j * b_dim1], abs(d__2));
/* L20: */
		}
/* L30: */
	    }
	} else if (trana && ! tranb) {
	    i__2 = *kk;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    ct[i__] += a[k + i__ * a_dim1] * b[k + j * b_dim1];
		    g[i__] += (d__1 = a[k + i__ * a_dim1], abs(d__1)) * (d__2
			    = b[k + j * b_dim1], abs(d__2));
/* L40: */
		}
/* L50: */
	    }
	} else if (! trana && tranb) {
	    i__2 = *kk;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    ct[i__] += a[i__ + k * a_dim1] * b[j + k * b_dim1];
		    g[i__] += (d__1 = a[i__ + k * a_dim1], abs(d__1)) * (d__2
			    = b[j + k * b_dim1], abs(d__2));
/* L60: */
		}
/* L70: */
	    }
	} else if (trana && tranb) {
	    i__2 = *kk;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    ct[i__] += a[k + i__ * a_dim1] * b[j + k * b_dim1];
		    g[i__] += (d__1 = a[k + i__ * a_dim1], abs(d__1)) * (d__2
			    = b[j + k * b_dim1], abs(d__2));
/* L80: */
		}
/* L90: */
	    }
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ct[i__] = *alpha * ct[i__] + *beta * c__[i__ + j * c_dim1];
	    g[i__] = abs(*alpha) * g[i__] + abs(*beta) * (d__1 = c__[i__ + j *
		     c_dim1], abs(d__1));
/* L100: */
	}

/*        Compute the error ratio for this result. */

	*err = 0.;
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    erri = (d__1 = ct[i__] - cc[i__ + j * cc_dim1], abs(d__1)) / *eps;
	    if (g[i__] != 0.) {
		erri /= g[i__];
	    }
	    *err = max(*err,erri);
	    if (*err * sqrt(*eps) >= 1.) {
		goto L130;
	    }
/* L110: */
	}

/* L120: */
    }

/*     If the loop completes, all results are at least half accurate. */
    goto L150;

/*     Report fatal error. */

L130:
    *fatal = TRUE_;
    io___361.ciunit = *nout;
    s_wsfe(&io___361);
    e_wsfe();
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (*mv) {
	    io___362.ciunit = *nout;
	    s_wsfe(&io___362);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&ct[i__], (ftnlen)sizeof(doublereal));
	    do_fio(&c__1, (char *)&cc[i__ + j * cc_dim1], (ftnlen)sizeof(
		    doublereal));
	    e_wsfe();
	} else {
	    io___363.ciunit = *nout;
	    s_wsfe(&io___363);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&cc[i__ + j * cc_dim1], (ftnlen)sizeof(
		    doublereal));
	    do_fio(&c__1, (char *)&ct[i__], (ftnlen)sizeof(doublereal));
	    e_wsfe();
	}
/* L140: */
    }
    if (*n > 1) {
	io___364.ciunit = *nout;
	s_wsfe(&io___364);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L150:
    return 0;


/*     End of DMMCH. */

} /* dmmch_ */

logical lde_(doublereal *ri, doublereal *rj, integer *lr)
{
    /* System generated locals */
    integer i__1;
    logical ret_val;

    /* Local variables */
    integer i__;


/*  Tests if two arrays are identical. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    --rj;
    --ri;

    /* Function Body */
    i__1 = *lr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (ri[i__] != rj[i__]) {
	    goto L20;
	}
/* L10: */
    }
    ret_val = TRUE_;
    goto L30;
L20:
    ret_val = FALSE_;
L30:
    return ret_val;

/*     End of LDE. */

} /* lde_ */

logical lderes_(char *type__, char *uplo, integer *m, integer *n, doublereal *
	aa, doublereal *as, integer *lda, ftnlen type_len, ftnlen uplo_len)
{
    /* System generated locals */
    integer aa_dim1, aa_offset, as_dim1, as_offset, i__1, i__2;
    logical ret_val;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen);

    /* Local variables */
    integer i__, j, ibeg, iend;
    logical upper;


/*  Tests if selected elements in two arrays are equal. */

/*  TYPE is 'GE' or 'SY'. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    as_dim1 = *lda;
    as_offset = 1 + as_dim1;
    as -= as_offset;
    aa_dim1 = *lda;
    aa_offset = 1 + aa_dim1;
    aa -= aa_offset;

    /* Function Body */
    upper = *(unsigned char *)uplo == 'U';
    if (s_cmp(type__, "GE", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *lda;
	    for (i__ = *m + 1; i__ <= i__2; ++i__) {
		if (aa[i__ + j * aa_dim1] != as[i__ + j * as_dim1]) {
		    goto L70;
		}
/* L10: */
	    }
/* L20: */
	}
    } else if (s_cmp(type__, "SY", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (upper) {
		ibeg = 1;
		iend = j;
	    } else {
		ibeg = j;
		iend = *n;
	    }
	    i__2 = ibeg - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		if (aa[i__ + j * aa_dim1] != as[i__ + j * as_dim1]) {
		    goto L70;
		}
/* L30: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		if (aa[i__ + j * aa_dim1] != as[i__ + j * as_dim1]) {
		    goto L70;
		}
/* L40: */
	    }
/* L50: */
	}
    }

    ret_val = TRUE_;
    goto L80;
L70:
    ret_val = FALSE_;
L80:
    return ret_val;

/*     End of LDERES. */

} /* lderes_ */

doublereal dbeg_(logical *reset)
{
    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static integer i__, ic, mi;


/*  Generates random numbers uniformly distributed between -0.5 and 0.5. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Local Scalars .. */
/*     .. Save statement .. */
/*     .. Executable Statements .. */
    if (*reset) {
/*        Initialize local variables. */
	mi = 891;
	i__ = 7;
	ic = 0;
	*reset = FALSE_;
    }

/*     The sequence of values of I is bounded between 1 and 999. */
/*     If initial I = 1,2,3,6,7 or 9, the period will be 50. */
/*     If initial I = 4 or 8, the period will be 25. */
/*     If initial I = 5, the period will be 10. */
/*     IC is used to break up the period by skipping 1 value of I in 6. */

    ++ic;
L10:
    i__ *= mi;
    i__ -= i__ / 1000 * 1000;
    if (ic >= 5) {
	ic = 0;
	goto L10;
    }
    ret_val = (i__ - 500) / 1001.;
    return ret_val;

/*     End of DBEG. */

} /* dbeg_ */

doublereal ddiff_(doublereal *x, doublereal *y)
{
    /* System generated locals */
    doublereal ret_val;


/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Executable Statements .. */
    ret_val = *x - *y;
    return ret_val;

/*     End of DDIFF. */

} /* ddiff_ */

/* Subroutine */ int chkxer_(char *srnamt, integer *infot, integer *nout,
	logical *lerr, logical *ok, ftnlen srnamt_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ***** ILLEGAL VALUE OF PARAMETER NUMBER"
	    " \002,i2,\002 NOT D\002,\002ETECTED BY \002,a6,\002 *****\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Fortran I/O blocks */
    static cilist io___374 = { 0, 0, 0, fmt_9999, 0 };



/*  Tests whether XERBLA has detected an error when it should. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Executable Statements .. */
    if (! (*lerr)) {
	io___374.ciunit = *nout;
	s_wsfe(&io___374);
	do_fio(&c__1, (char *)&(*infot), (ftnlen)sizeof(integer));
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
	*ok = FALSE_;
    }
    *lerr = FALSE_;
    return 0;


/*     End of CHKXER. */

} /* chkxer_ */

/* Subroutine */ int xerbla_(char *srname, integer *info, ftnlen srname_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ******* XERBLA WAS CALLED WITH INFO ="
	    " \002,i6,\002 INSTEAD\002,\002 OF \002,i2,\002 *******\002)";
    static char fmt_9997[] = "(\002 ******* XERBLA WAS CALLED WITH INFO ="
	    " \002,i6,\002 *******\002)";
    static char fmt_9998[] = "(\002 ******* XERBLA WAS CALLED WITH SRNAME ="
	    " \002,a6,\002 INSTE\002,\002AD OF \002,a6,\002 *******\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     s_cmp(const char *, const char *, ftnlen, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___375 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___376 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___377 = { 0, 0, 0, fmt_9998, 0 };



/*  This is a special version of XERBLA to be used only as part of */
/*  the test program for testing error exits from the Level 3 BLAS */
/*  routines. */

/*  XERBLA  is an error handler for the Level 3 BLAS routines. */

/*  It is called by the Level 3 BLAS routines if an input parameter is */
/*  invalid. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */
    infoc_2.lerr = TRUE_;
    if (*info != infoc_2.infot) {
	if (infoc_2.infot != 0) {
	    io___375.ciunit = infoc_2.nout;
	    s_wsfe(&io___375);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&infoc_2.infot, (ftnlen)sizeof(integer));
	    e_wsfe();
	} else {
	    io___376.ciunit = infoc_2.nout;
	    s_wsfe(&io___376);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    e_wsfe();
	}
	infoc_2.ok = FALSE_;
    }
    if (s_cmp(srname, srnamc_1.srnamt, (ftnlen)6, (ftnlen)6) != 0) {
	io___377.ciunit = infoc_2.nout;
	s_wsfe(&io___377);
	do_fio(&c__1, srname, (ftnlen)6);
	do_fio(&c__1, srnamc_1.srnamt, (ftnlen)6);
	e_wsfe();
	infoc_2.ok = FALSE_;
    }
    return 0;


/*     End of XERBLA */

} /* xerbla_ */

/* Main program alias */ int dblat3_ () { main (); return 0; }
