/* zblat3.f -- translated by f2c (version 20100827).
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

static doublecomplex c_b1 = {0.,0.};
static doublecomplex c_b2 = {1.,0.};
static integer c__9 = 9;
static integer c__1 = 1;
static integer c__3 = 3;
static integer c__8 = 8;
static integer c__5 = 5;
static integer c__65 = 65;
static integer c__7 = 7;
static integer c__2 = 2;
static doublereal c_b88 = 0.;
static logical c_true = TRUE_;
static logical c_false = FALSE_;
static integer c__0 = 0;
static integer c_n1 = -1;

/* > \brief \b ZBLAT3 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM ZBLAT3 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > Test program for the COMPLEX*16       Level 3 Blas. */
/* > */
/* > The program must be driven by a short data file. The first 14 records */
/* > of the file are read using list-directed input, the last 9 records */
/* > are read using the format ( A6, L2 ). An annotated example of a data */
/* > file can be obtained by deleting the first 3 characters from the */
/* > following 23 lines: */
/* > 'zblat3.out'      NAME OF SUMMARY OUTPUT FILE */
/* > 6                 UNIT NUMBER OF SUMMARY FILE */
/* > 'ZBLAT3.SNAP'     NAME OF SNAPSHOT OUTPUT FILE */
/* > -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0) */
/* > F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD. */
/* > F        LOGICAL FLAG, T TO STOP ON FAILURES. */
/* > T        LOGICAL FLAG, T TO TEST ERROR EXITS. */
/* > 16.0     THRESHOLD VALUE OF TEST RATIO */
/* > 6                 NUMBER OF VALUES OF N */
/* > 0 1 2 3 5 9       VALUES OF N */
/* > 3                 NUMBER OF VALUES OF ALPHA */
/* > (0.0,0.0) (1.0,0.0) (0.7,-0.9)       VALUES OF ALPHA */
/* > 3                 NUMBER OF VALUES OF BETA */
/* > (0.0,0.0) (1.0,0.0) (1.3,-1.1)       VALUES OF BETA */
/* > ZGEMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZHEMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZSYMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZTRMM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZTRSM  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZHERK  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZSYRK  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZHER2K T PUT F FOR NO TEST. SAME COLUMNS. */
/* > ZSYR2K T PUT F FOR NO TEST. SAME COLUMNS. */
/* > */
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

/* > \ingroup complex16_blas_testing */

/*  ===================================================================== */
/* Main program */ int main(void)
{
#ifdef BLIS_ENABLE_HPX
    char* program = "zblat3";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static char snames[6*9] = "ZGEMM " "ZHEMM " "ZSYMM " "ZTRMM " "ZTRSM "
	    "ZHERK " "ZSYRK " "ZHER2K" "ZSYR2K";

    /* Format strings */
    static char fmt_9997[] = "(\002 NUMBER OF VALUES OF \002,a,\002 IS LESS "
	    "THAN 1 OR GREATER \002,\002THAN \002,i2)";
    static char fmt_9996[] = "(\002 VALUE OF N IS LESS THAN 0 OR GREATER THA"
	    "N \002,i2)";
    static char fmt_9995[] = "(\002 TESTS OF THE COMPLEX*16       LEVEL 3 BL"
	    "AS\002,//\002 THE F\002,\002OLLOWING PARAMETER VALUES WILL BE US"
	    "ED:\002)";
    static char fmt_9994[] = "(\002   FOR N              \002,9i6)";
    static char fmt_9993[] = "(\002   FOR ALPHA          \002,7(\002(\002,f4"
	    ".1,\002,\002,f4.1,\002)  \002,:))";
    static char fmt_9992[] = "(\002   FOR BETA           \002,7(\002(\002,f4"
	    ".1,\002,\002,f4.1,\002)  \002,:))";
    static char fmt_9984[] = "(\002 ERROR-EXITS WILL NOT BE TESTED\002)";
    static char fmt_9999[] = "(\002 ROUTINES PASS COMPUTATIONAL TESTS IF TES"
	    "T RATIO IS LES\002,\002S THAN\002,f8.2)";
    static char fmt_9988[] = "(a6,l2)";
    static char fmt_9990[] = "(\002 SUBPROGRAM NAME \002,a6,\002 NOT RECOGNI"
	    "ZED\002,/\002 ******* T\002,\002ESTS ABANDONED *******\002)";
    static char fmt_9998[] = "(\002 RELATIVE MACHINE PRECISION IS TAKEN TO"
	    " BE\002,1p,d9.1)";
    static char fmt_9989[] = "(\002 ERROR IN ZMMCH -  IN-LINE DOT PRODUCTS A"
	    "RE BEING EVALU\002,\002ATED WRONGLY.\002,/\002 ZMMCH WAS CALLED "
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
    integer i__1, i__2, i__3, i__4, i__5;
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
    doublecomplex c__[4225]	/* was [65][65] */;
    doublereal g[65];
    integer i__, j, n;
    doublecomplex w[130], aa[4225], ab[8450]	/* was [65][130] */, bb[4225],
	     cc[4225], as[4225], bs[4225], cs[4225], ct[65], alf[7], bet[7];
    doublereal eps, err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    integer nalf, idim[9];
    logical same;
    integer nbet, ntra;
    logical rewi;
    integer nout;
    extern /* Subroutine */ int zchk1_(char *, doublereal *, doublereal *,
	    integer *, integer *, logical *, logical *, logical *, integer *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, doublecomplex *,
	    doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *
	    , doublecomplex *, doublecomplex *, doublecomplex *, doublereal *,
	     ftnlen), zchk2_(char *, doublereal *, doublereal *, integer *,
	    integer *, logical *, logical *, logical *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex
	    *, doublecomplex *, doublecomplex *, doublecomplex *,
	    doublecomplex *, doublecomplex *, doublecomplex *, doublereal *,
	    ftnlen), zchk3_(char *, doublereal *, doublereal *, integer *,
	    integer *, logical *, logical *, logical *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *
	    , doublecomplex *, doublecomplex *, doublereal *, doublecomplex *,
	     ftnlen), zchk4_(char *, doublereal *, doublereal *, integer *,
	    integer *, logical *, logical *, logical *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex
	    *, doublecomplex *, doublecomplex *, doublecomplex *,
	    doublecomplex *, doublecomplex *, doublecomplex *, doublereal *,
	    ftnlen), zchk5_(char *, doublereal *, doublereal *, integer *,
	    integer *, logical *, logical *, logical *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex
	    *, doublecomplex *, doublecomplex *, doublecomplex *,
	    doublecomplex *, doublecomplex *, doublereal *, doublecomplex *,
	    ftnlen);
    logical fatal, trace;
    integer nidim;
    extern /* Subroutine */ int zchke_(integer *, char *, integer *, ftnlen),
	    zmmch_(char *, char *, integer *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, doublereal *, doublecomplex *, integer *,
	    doublereal *, doublereal *, logical *, integer *, logical *,
	    ftnlen, ftnlen);
    char snaps[32];
    integer isnum;
    logical ltest[9], sfatal;
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
	do_lio(&c__7, &c__1, (char *)&alf[i__ - 1], (ftnlen)sizeof(
		doublecomplex));
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
	do_lio(&c__7, &c__1, (char *)&bet[i__ - 1], (ftnlen)sizeof(
		doublecomplex));
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
	do_fio(&c__2, (char *)&alf[i__ - 1], (ftnlen)sizeof(doublereal));
    }
    e_wsfe();
    io___39.ciunit = nout;
    s_wsfe(&io___39);
    i__1 = nbet;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__2, (char *)&bet[i__ - 1], (ftnlen)sizeof(doublereal));
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

    for (i__ = 1; i__ <= 9; ++i__) {
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
    for (i__ = 1; i__ <= 9; ++i__) {
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

    eps = d_epsilon_(&c_b88);
    io___51.ciunit = nout;
    s_wsfe(&io___51);
    do_fio(&c__1, (char *)&eps, (ftnlen)sizeof(doublereal));
    e_wsfe();

/*     Check the reliability of ZMMCH using exact data. */

    n = 32;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * 65 - 66;
/* Computing MAX */
	    i__5 = i__ - j + 1;
	    i__4 = max(i__5,0);
	    ab[i__3].r = (doublereal) i__4, ab[i__3].i = 0.;
/* L90: */
	}
	i__2 = j + 4224;
	ab[i__2].r = (doublereal) j, ab[i__2].i = 0.;
	i__2 = (j + 65) * 65 - 65;
	ab[i__2].r = (doublereal) j, ab[i__2].i = 0.;
	i__2 = j - 1;
	c__[i__2].r = 0., c__[i__2].i = 0.;
/* L100: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	i__3 = j * ((j + 1) * j) / 2 - (j + 1) * j * (j - 1) / 3;
	cc[i__2].r = (doublereal) i__3, cc[i__2].i = 0.;
/* L110: */
    }
/*     CC holds the exact result. On exit from ZMMCH CT holds */
/*     the result computed by ZMMCH. */
    *(unsigned char *)transa = 'N';
    *(unsigned char *)transb = 'N';
    zmmch_(transa, transb, &n, &c__1, &n, &c_b2, ab, &c__65, &ab[4225], &
	    c__65, &c_b1, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	    &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lze_(cc, ct, &n);
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
    *(unsigned char *)transb = 'C';
    zmmch_(transa, transb, &n, &c__1, &n, &c_b2, ab, &c__65, &ab[4225], &
	    c__65, &c_b1, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	    &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lze_(cc, ct, &n);
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
	i__2 = j + 4224;
	i__3 = n - j + 1;
	ab[i__2].r = (doublereal) i__3, ab[i__2].i = 0.;
	i__2 = (j + 65) * 65 - 65;
	i__3 = n - j + 1;
	ab[i__2].r = (doublereal) i__3, ab[i__2].i = 0.;
/* L120: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = n - j;
	i__3 = j * ((j + 1) * j) / 2 - (j + 1) * j * (j - 1) / 3;
	cc[i__2].r = (doublereal) i__3, cc[i__2].i = 0.;
/* L130: */
    }
    *(unsigned char *)transa = 'C';
    *(unsigned char *)transb = 'N';
    zmmch_(transa, transb, &n, &c__1, &n, &c_b2, ab, &c__65, &ab[4225], &
	    c__65, &c_b1, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	    &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lze_(cc, ct, &n);
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
    *(unsigned char *)transb = 'C';
    zmmch_(transa, transb, &n, &c__1, &n, &c_b2, ab, &c__65, &ab[4225], &
	    c__65, &c_b1, c__, &c__65, ct, g, cc, &c__65, &eps, &err, &fatal,
	    &nout, &c_true, (ftnlen)1, (ftnlen)1);
    same = lze_(cc, ct, &n);
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

    for (isnum = 1; isnum <= 9; ++isnum) {
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
		zchke_(&isnum, snames + (isnum - 1) * 6, &nout, (ftnlen)6);
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
		case 3:  goto L150;
		case 4:  goto L160;
		case 5:  goto L160;
		case 6:  goto L170;
		case 7:  goto L170;
		case 8:  goto L180;
		case 9:  goto L180;
	    }
/*           Test ZGEMM, 01. */
L140:
	    zchk1_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test ZHEMM, 02, ZSYMM, 03. */
L150:
	    zchk2_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test ZTRMM, 04, ZTRSM, 05. */
L160:
	    zchk3_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &c__65,
		    ab, aa, as, &ab[4225], bb, bs, ct, g, c__, (ftnlen)6);
	    goto L190;
/*           Test ZHERK, 06, ZSYRK, 07. */
L170:
	    zchk4_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &nbet,
		    bet, &c__65, ab, aa, as, &ab[4225], bb, bs, c__, cc, cs,
		    ct, g, (ftnlen)6);
	    goto L190;
/*           Test ZHER2K, 08, ZSYR2K, 09. */
L180:
	    zchk5_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
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


/*     End of ZBLAT3. */

#ifdef BLIS_ENABLE_HPX
    return bli_thread_finalize_hpx();
#else
	// Return peacefully.
	return 0;
#endif
} /* main */

/* Subroutine */ int zchk1_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublecomplex *
	alf, integer *nbet, doublecomplex *bet, integer *nmax, doublecomplex *
	a, doublecomplex *aa, doublecomplex *as, doublecomplex *b,
	doublecomplex *bb, doublecomplex *bs, doublecomplex *c__,
	doublecomplex *cc, doublecomplex *cs, doublecomplex *ct, doublereal *
	g, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[3] = "NTC";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002','\002"
	    ",a1,\002',\002,3(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,"
	    "\002), A,\002,i3,\002, B,\002,i3,\002,(\002,f4.1,\002,\002,f4.1"
	    ",\002), C,\002,i3,\002).\002)";
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
	    i__3, i__4, i__5, i__6, i__7, i__8;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, k, m, n, ia, ib, ma, mb, na, nb, nc, ik, im, in, ks, ms, ns,
	    ica, icb, laa, lbb, lda, lcc, ldb, ldc;
    doublecomplex als, bls;
    doublereal err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    doublecomplex beta;
    integer ldas, ldbs, ldcs;
    logical same, null;
    doublecomplex alpha;
    logical isame[13], trana, tranb;
    extern /* Subroutine */ int zmake_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     logical *, doublecomplex *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int zmmch_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, doublereal *, doublecomplex *,
	    integer *, doublereal *, doublereal *, logical *, integer *,
	    logical *, ftnlen, ftnlen), zgemm_(char *, char *, integer *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen);
    logical reset;
    char tranas[1], tranbs[1], transa[1], transb[1];
    doublereal errmax;
    extern logical lzeres_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, ftnlen, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___124 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___125 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___128 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___130 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___131 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___132 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___133 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests ZGEMM. */

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

		    zmake_("GE", " ", " ", &ma, &na, &a[a_offset], nmax, &aa[
			    1], &lda, &reset, &c_b1, (ftnlen)2, (ftnlen)1, (
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

			zmake_("GE", " ", " ", &mb, &nb, &b[b_offset], nmax, &
				bb[1], &ldb, &reset, &c_b1, (ftnlen)2, (
				ftnlen)1, (ftnlen)1);

			i__4 = *nalf;
			for (ia = 1; ia <= i__4; ++ia) {
			    i__5 = ia;
			    alpha.r = alf[i__5].r, alpha.i = alf[i__5].i;

			    i__5 = *nbet;
			    for (ib = 1; ib <= i__5; ++ib) {
				i__6 = ib;
				beta.r = bet[i__6].r, beta.i = bet[i__6].i;

/*                          Generate the matrix C. */

				zmake_("GE", " ", " ", &m, &n, &c__[c_offset],
					 nmax, &cc[1], &ldc, &reset, &c_b1, (
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
				als.r = alpha.r, als.i = alpha.i;
				i__6 = laa;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    i__7 = i__;
				    i__8 = i__;
				    as[i__7].r = aa[i__8].r, as[i__7].i = aa[
					    i__8].i;
/* L10: */
				}
				ldas = lda;
				i__6 = lbb;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    i__7 = i__;
				    i__8 = i__;
				    bs[i__7].r = bb[i__8].r, bs[i__7].i = bb[
					    i__8].i;
/* L20: */
				}
				ldbs = ldb;
				bls.r = beta.r, bls.i = beta.i;
				i__6 = lcc;
				for (i__ = 1; i__ <= i__6; ++i__) {
				    i__7 = i__;
				    i__8 = i__;
				    cs[i__7].r = cc[i__8].r, cs[i__7].i = cc[
					    i__8].i;
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
				    do_fio(&c__2, (char *)&alpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&ldb, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__2, (char *)&beta, (ftnlen)
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
				zgemm_(transa, transb, &m, &n, &k, &alpha, &
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
				isame[5] = als.r == alpha.r && als.i ==
					alpha.i;
				isame[6] = lze_(&as[1], &aa[1], &laa);
				isame[7] = ldas == lda;
				isame[8] = lze_(&bs[1], &bb[1], &lbb);
				isame[9] = ldbs == ldb;
				isame[10] = bls.r == beta.r && bls.i ==
					beta.i;
				if (null) {
				    isame[11] = lze_(&cs[1], &cc[1], &lcc);
				} else {
				    isame[11] = lzeres_("GE", " ", &m, &n, &
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

				    zmmch_(transa, transb, &m, &n, &k, &alpha,
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
    do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    do_fio(&c__2, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L130:
    return 0;


/*     End of ZCHK1. */

} /* zchk1_ */

/* Subroutine */ int zchk2_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublecomplex *
	alf, integer *nbet, doublecomplex *bet, integer *nmax, doublecomplex *
	a, doublecomplex *aa, doublecomplex *as, doublecomplex *b,
	doublecomplex *bb, doublecomplex *bs, doublecomplex *c__,
	doublecomplex *cc, doublecomplex *cs, doublecomplex *ct, doublereal *
	g, ftnlen sname_len)
{
    /* Initialized data */

    static char ichs[2] = "LR";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,\002)"
	    ", A,\002,i3,\002, B,\002,i3,\002,(\002,f4.1,\002,\002,f4.1,\002)"
	    ", C,\002,i3,\002)    .\002)";
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
	    i__3, i__4, i__5, i__6, i__7;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);

    /* Local variables */
    integer i__, m, n, ia, ib, na, nc, im, in, ms, ns, laa, lbb, lda, lcc,
	    ldb, ldc, ics;
    doublecomplex als, bls;
    integer icu;
    doublereal err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    doublecomplex beta;
    integer ldas, ldbs, ldcs;
    logical same;
    char side[1];
    logical conj, left, null;
    char uplo[1];
    doublecomplex alpha;
    logical isame[13];
    char sides[1];
    extern /* Subroutine */ int zmake_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     logical *, doublecomplex *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int zmmch_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, doublereal *, doublecomplex *,
	    integer *, doublereal *, doublereal *, logical *, integer *,
	    logical *, ftnlen, ftnlen), zhemm_(char *, char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen);
    logical reset;
    char uplos[1];
    extern /* Subroutine */ int zsymm_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *, ftnlen,
	    ftnlen);
    doublereal errmax;
    extern logical lzeres_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, ftnlen, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___172 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___173 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___176 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___178 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___179 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___180 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___181 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests ZHEMM and ZSYMM. */

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
    conj = s_cmp(sname + 1, "HE", (ftnlen)2, (ftnlen)2) == 0;

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

	    zmake_("GE", " ", " ", &m, &n, &b[b_offset], nmax, &bb[1], &ldb, &
		    reset, &c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)1);

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

/*                 Generate the hermitian or symmetric matrix A. */

		    zmake_(sname + 1, uplo, " ", &na, &na, &a[a_offset], nmax,
			     &aa[1], &lda, &reset, &c_b1, (ftnlen)2, (ftnlen)
			    1, (ftnlen)1);

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			i__4 = ia;
			alpha.r = alf[i__4].r, alpha.i = alf[i__4].i;

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    i__5 = ib;
			    beta.r = bet[i__5].r, beta.i = bet[i__5].i;

/*                       Generate the matrix C. */

			    zmake_("GE", " ", " ", &m, &n, &c__[c_offset],
				    nmax, &cc[1], &ldc, &reset, &c_b1, (
				    ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the */
/*                       subroutine. */

			    *(unsigned char *)sides = *(unsigned char *)side;
			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    ms = m;
			    ns = n;
			    als.r = alpha.r, als.i = alpha.i;
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				as[i__6].r = aa[i__7].r, as[i__6].i = aa[i__7]
					.i;
/* L10: */
			    }
			    ldas = lda;
			    i__5 = lbb;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				bs[i__6].r = bb[i__7].r, bs[i__6].i = bb[i__7]
					.i;
/* L20: */
			    }
			    ldbs = ldb;
			    bls.r = beta.r, bls.i = beta.i;
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				cs[i__6].r = cc[i__7].r, cs[i__6].i = cc[i__7]
					.i;
/* L30: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (*trace) {
				io___172.ciunit = *ntra;
				s_wsfe(&io___172);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, side, (ftnlen)1);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, (char *)&m, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(
					doublereal));
				do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(
					integer));
				do_fio(&c__2, (char *)&beta, (ftnlen)sizeof(
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
			    if (conj) {
				zhemm_(side, uplo, &m, &n, &alpha, &aa[1], &
					lda, &bb[1], &ldb, &beta, &cc[1], &
					ldc, (ftnlen)1, (ftnlen)1);
			    } else {
				zsymm_(side, uplo, &m, &n, &alpha, &aa[1], &
					lda, &bb[1], &ldb, &beta, &cc[1], &
					ldc, (ftnlen)1, (ftnlen)1);
			    }

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___173.ciunit = *nout;
				s_wsfe(&io___173);
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
			    isame[4] = als.r == alpha.r && als.i == alpha.i;
			    isame[5] = lze_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    isame[7] = lze_(&bs[1], &bb[1], &lbb);
			    isame[8] = ldbs == ldb;
			    isame[9] = bls.r == beta.r && bls.i == beta.i;
			    if (null) {
				isame[10] = lze_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[10] = lzeres_("GE", " ", &m, &n, &cs[1],
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
				    io___176.ciunit = *nout;
				    s_wsfe(&io___176);
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
				    zmmch_("N", "N", &m, &n, &m, &alpha, &a[
					    a_offset], nmax, &b[b_offset],
					    nmax, &beta, &c__[c_offset], nmax,
					     &ct[1], &g[1], &cc[1], &ldc, eps,
					     &err, fatal, nout, &c_true, (
					    ftnlen)1, (ftnlen)1);
				} else {
				    zmmch_("N", "N", &m, &n, &n, &alpha, &b[
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
	io___178.ciunit = *nout;
	s_wsfe(&io___178);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___179.ciunit = *nout;
	s_wsfe(&io___179);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L120;

L110:
    io___180.ciunit = *nout;
    s_wsfe(&io___180);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___181.ciunit = *nout;
    s_wsfe(&io___181);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, side, (ftnlen)1);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    do_fio(&c__2, (char *)&beta, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
    e_wsfe();

L120:
    return 0;


/*     End of ZCHK2. */

} /* zchk2_ */

/* Subroutine */ int zchk3_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublecomplex *
	alf, integer *nmax, doublecomplex *a, doublecomplex *aa,
	doublecomplex *as, doublecomplex *b, doublecomplex *bb, doublecomplex
	*bs, doublecomplex *ct, doublereal *g, doublecomplex *c__, ftnlen
	sname_len)
{
    /* Initialized data */

    static char ichu[2] = "UL";
    static char icht[3] = "NTC";
    static char ichd[2] = "UN";
    static char ichs[2] = "LR";

    /* Format strings */
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002(\002,4(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,\002)"
	    ", A,\002,i3,\002, B,\002,i3,\002)         \002,\002      .\002)";
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
	    i__3, i__4, i__5, i__6, i__7;
    doublecomplex z__1;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);

    /* Local variables */
    integer i__, j, m, n, ia, na, nc, im, in, ms, ns, laa, icd, lbb, lda, ldb,
	     ics;
    doublecomplex als;
    integer ict, icu;
    doublereal err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    char diag[1];
    integer ldas, ldbs;
    logical same;
    char side[1];
    logical left, null;
    char uplo[1];
    doublecomplex alpha;
    char diags[1];
    logical isame[13];
    char sides[1];
    extern /* Subroutine */ int zmake_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     logical *, doublecomplex *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int zmmch_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, doublereal *, doublecomplex *,
	    integer *, doublereal *, doublereal *, logical *, integer *,
	    logical *, ftnlen, ftnlen);
    logical reset;
    char uplos[1];
    extern /* Subroutine */ int ztrmm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *, ftnlen, ftnlen, ftnlen, ftnlen),
	    ztrsm_(char *, char *, char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, ftnlen, ftnlen, ftnlen, ftnlen);
    char tranas[1], transa[1];
    doublereal errmax;
    extern logical lzeres_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, ftnlen, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___222 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___223 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___224 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___227 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___229 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___230 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___231 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___232 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests ZTRMM and ZTRSM. */

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
/*     Set up zero matrix for ZMMCH. */
    i__1 = *nmax;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *nmax;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    c__[i__3].r = 0., c__[i__3].i = 0.;
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
				i__4 = ia;
				alpha.r = alf[i__4].r, alpha.i = alf[i__4].i;

/*                          Generate the matrix A. */

				zmake_("TR", uplo, diag, &na, &na, &a[
					a_offset], nmax, &aa[1], &lda, &reset,
					 &c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)
					1);

/*                          Generate the matrix B. */

				zmake_("GE", " ", " ", &m, &n, &b[b_offset],
					nmax, &bb[1], &ldb, &reset, &c_b1, (
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
				als.r = alpha.r, als.i = alpha.i;
				i__4 = laa;
				for (i__ = 1; i__ <= i__4; ++i__) {
				    i__5 = i__;
				    i__6 = i__;
				    as[i__5].r = aa[i__6].r, as[i__5].i = aa[
					    i__6].i;
/* L30: */
				}
				ldas = lda;
				i__4 = lbb;
				for (i__ = 1; i__ <= i__4; ++i__) {
				    i__5 = i__;
				    i__6 = i__;
				    bs[i__5].r = bb[i__6].r, bs[i__5].i = bb[
					    i__6].i;
/* L40: */
				}
				ldbs = ldb;

/*                          Call the subroutine. */

				if (s_cmp(sname + 3, "MM", (ftnlen)2, (ftnlen)
					2) == 0) {
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
					do_fio(&c__2, (char *)&alpha, (ftnlen)
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
				    ztrmm_(side, uplo, transa, diag, &m, &n, &
					    alpha, &aa[1], &lda, &bb[1], &ldb,
					     (ftnlen)1, (ftnlen)1, (ftnlen)1,
					    (ftnlen)1);
				} else if (s_cmp(sname + 3, "SM", (ftnlen)2, (
					ftnlen)2) == 0) {
				    if (*trace) {
					io___223.ciunit = *ntra;
					s_wsfe(&io___223);
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
					do_fio(&c__2, (char *)&alpha, (ftnlen)
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
				    ztrsm_(side, uplo, transa, diag, &m, &n, &
					    alpha, &aa[1], &lda, &bb[1], &ldb,
					     (ftnlen)1, (ftnlen)1, (ftnlen)1,
					    (ftnlen)1);
				}

/*                          Check if error-exit was taken incorrectly. */

				if (! infoc_1.ok) {
				    io___224.ciunit = *nout;
				    s_wsfe(&io___224);
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
				isame[6] = als.r == alpha.r && als.i ==
					alpha.i;
				isame[7] = lze_(&as[1], &aa[1], &laa);
				isame[8] = ldas == lda;
				if (null) {
				    isame[9] = lze_(&bs[1], &bb[1], &lbb);
				} else {
				    isame[9] = lzeres_("GE", " ", &m, &n, &bs[
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
					io___227.ciunit = *nout;
					s_wsfe(&io___227);
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
					    zmmch_(transa, "N", &m, &n, &m, &
						    alpha, &a[a_offset], nmax,
						     &b[b_offset], nmax, &
						    c_b1, &c__[c_offset],
						    nmax, &ct[1], &g[1], &bb[
						    1], &ldb, eps, &err,
						    fatal, nout, &c_true, (
						    ftnlen)1, (ftnlen)1);
					} else {
					    zmmch_("N", transa, &m, &n, &n, &
						    alpha, &b[b_offset], nmax,
						     &a[a_offset], nmax, &
						    c_b1, &c__[c_offset],
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
			  i__6 = i__ + j * c_dim1;
			  i__7 = i__ + (j - 1) * ldb;
			  c__[i__6].r = bb[i__7].r, c__[i__6].i = bb[i__7].i;
			  i__6 = i__ + (j - 1) * ldb;
			  i__7 = i__ + j * b_dim1;
			  z__1.r = alpha.r * b[i__7].r - alpha.i * b[i__7].i,
				  z__1.i = alpha.r * b[i__7].i + alpha.i * b[
				  i__7].r;
			  bb[i__6].r = z__1.r, bb[i__6].i = z__1.i;
/* L60: */
					    }
/* L70: */
					}

					if (left) {
					    zmmch_(transa, "N", &m, &n, &m, &
						    c_b2, &a[a_offset], nmax,
						    &c__[c_offset], nmax, &
						    c_b1, &b[b_offset], nmax,
						    &ct[1], &g[1], &bb[1], &
						    ldb, eps, &err, fatal,
						    nout, &c_false, (ftnlen)1,
						     (ftnlen)1);
					} else {
					    zmmch_("N", transa, &m, &n, &n, &
						    c_b2, &c__[c_offset],
						    nmax, &a[a_offset], nmax,
						    &c_b1, &b[b_offset], nmax,
						     &ct[1], &g[1], &bb[1], &
						    ldb, eps, &err, fatal,
						    nout, &c_false, (ftnlen)1,
						     (ftnlen)1);
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
	io___229.ciunit = *nout;
	s_wsfe(&io___229);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___230.ciunit = *nout;
	s_wsfe(&io___230);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L160;

L150:
    io___231.ciunit = *nout;
    s_wsfe(&io___231);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___232.ciunit = *nout;
    s_wsfe(&io___232);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, side, (ftnlen)1);
    do_fio(&c__1, uplo, (ftnlen)1);
    do_fio(&c__1, transa, (ftnlen)1);
    do_fio(&c__1, diag, (ftnlen)1);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
    e_wsfe();

L160:
    return 0;


/*     End of ZCHK3. */

} /* zchk3_ */

/* Subroutine */ int zchk4_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublecomplex *
	alf, integer *nbet, doublecomplex *bet, integer *nmax, doublecomplex *
	a, doublecomplex *aa, doublecomplex *as, doublecomplex *b,
	doublecomplex *bb, doublecomplex *bs, doublecomplex *c__,
	doublecomplex *cc, doublecomplex *cs, doublecomplex *ct, doublereal *
	g, ftnlen sname_len)
{
    /* Initialized data */

    static char icht[2] = "NC";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),f4.1,\002, A,\002,i3,\002,\002,f4.1,"
	    "\002, C,\002,i3,\002)               \002,\002          .\002)";
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,\002)"
	    " , A,\002,i3,\002,(\002,f4.1,\002,\002,f4.1,\002), C,\002,i3,"
	    "\002)          .\002)";
    static char fmt_9992[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
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
	    i__3, i__4, i__5, i__6, i__7;
    doublecomplex z__1;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);

    /* Local variables */
    integer i__, j, k, n, ia, ib, jc, ma, na, nc, ik, in, jj, lj, ks, ns, laa,
	     lda, lcc, ldc;
    doublecomplex als;
    integer ict, icu;
    doublereal err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    doublecomplex beta;
    integer ldas, ldcs;
    logical same, conj;
    doublecomplex bets;
    doublereal rals;
    logical tran, null;
    char uplo[1];
    doublecomplex alpha;
    doublereal rbeta;
    logical isame[13];
    extern /* Subroutine */ int zmake_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     logical *, doublecomplex *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int zmmch_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, doublereal *, doublecomplex *,
	    integer *, doublereal *, doublereal *, logical *, integer *,
	    logical *, ftnlen, ftnlen);
    doublereal rbets;
    logical reset;
    extern /* Subroutine */ int zherk_(char *, char *, integer *, integer *,
	    doublereal *, doublecomplex *, integer *, doublereal *,
	    doublecomplex *, integer *, ftnlen, ftnlen);
    char trans[1];
    logical upper;
    char uplos[1];
    extern /* Subroutine */ int zsyrk_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, ftnlen, ftnlen);
    doublereal ralpha, errmax;
    extern logical lzeres_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, ftnlen, ftnlen);
    char transs[1], transt[1];

    /* Fortran I/O blocks */
    static cilist io___274 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___275 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___276 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___279 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___286 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___287 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___288 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___289 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___290 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___291 = { 0, 0, 0, fmt_9993, 0 };



/*  Tests ZHERK and ZSYRK. */

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
    conj = s_cmp(sname + 1, "HE", (ftnlen)2, (ftnlen)2) == 0;

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

	i__2 = *nidim;
	for (ik = 1; ik <= i__2; ++ik) {
	    k = idim[ik];

	    for (ict = 1; ict <= 2; ++ict) {
		*(unsigned char *)trans = *(unsigned char *)&icht[ict - 1];
		tran = *(unsigned char *)trans == 'C';
		if (tran && ! conj) {
		    *(unsigned char *)trans = 'T';
		}
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

		zmake_("GE", " ", " ", &ma, &na, &a[a_offset], nmax, &aa[1], &
			lda, &reset, &c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)1);

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];
		    upper = *(unsigned char *)uplo == 'U';

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			i__4 = ia;
			alpha.r = alf[i__4].r, alpha.i = alf[i__4].i;
			if (conj) {
			    ralpha = alpha.r;
			    z__1.r = ralpha, z__1.i = 0.;
			    alpha.r = z__1.r, alpha.i = z__1.i;
			}

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    i__5 = ib;
			    beta.r = bet[i__5].r, beta.i = bet[i__5].i;
			    if (conj) {
				rbeta = beta.r;
				z__1.r = rbeta, z__1.i = 0.;
				beta.r = z__1.r, beta.i = z__1.i;
			    }
			    null = n <= 0;
			    if (conj) {
				null = null || (k <= 0 || ralpha == 0.) &&
					rbeta == 1.;
			    }

/*                       Generate the matrix C. */

			    zmake_(sname + 1, uplo, " ", &n, &n, &c__[
				    c_offset], nmax, &cc[1], &ldc, &reset, &
				    c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the subroutine. */

			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    *(unsigned char *)transs = *(unsigned char *)
				    trans;
			    ns = n;
			    ks = k;
			    if (conj) {
				rals = ralpha;
			    } else {
				als.r = alpha.r, als.i = alpha.i;
			    }
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				as[i__6].r = aa[i__7].r, as[i__6].i = aa[i__7]
					.i;
/* L10: */
			    }
			    ldas = lda;
			    if (conj) {
				rbets = rbeta;
			    } else {
				bets.r = beta.r, bets.i = beta.i;
			    }
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				cs[i__6].r = cc[i__7].r, cs[i__6].i = cc[i__7]
					.i;
/* L20: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (conj) {
				if (*trace) {
				    io___274.ciunit = *ntra;
				    s_wsfe(&io___274);
				    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, sname, (ftnlen)6);
				    do_fio(&c__1, uplo, (ftnlen)1);
				    do_fio(&c__1, trans, (ftnlen)1);
				    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&ralpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&rbeta, (ftnlen)
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
				zherk_(uplo, trans, &n, &k, &ralpha, &aa[1], &
					lda, &rbeta, &cc[1], &ldc, (ftnlen)1,
					(ftnlen)1);
			    } else {
				if (*trace) {
				    io___275.ciunit = *ntra;
				    s_wsfe(&io___275);
				    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, sname, (ftnlen)6);
				    do_fio(&c__1, uplo, (ftnlen)1);
				    do_fio(&c__1, trans, (ftnlen)1);
				    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__2, (char *)&alpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__2, (char *)&beta, (ftnlen)
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
				zsyrk_(uplo, trans, &n, &k, &alpha, &aa[1], &
					lda, &beta, &cc[1], &ldc, (ftnlen)1, (
					ftnlen)1);
			    }

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___276.ciunit = *nout;
				s_wsfe(&io___276);
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
			    if (conj) {
				isame[4] = rals == ralpha;
			    } else {
				isame[4] = als.r == alpha.r && als.i ==
					alpha.i;
			    }
			    isame[5] = lze_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    if (conj) {
				isame[7] = rbets == rbeta;
			    } else {
				isame[7] = bets.r == beta.r && bets.i ==
					beta.i;
			    }
			    if (null) {
				isame[8] = lze_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[8] = lzeres_(sname + 1, uplo, &n, &n, &
					cs[1], &cc[1], &ldc, (ftnlen)2, (
					ftnlen)1);
			    }
			    isame[9] = ldcs == ldc;

/*                       If data was incorrectly changed, report and */
/*                       return. */

			    same = TRUE_;
			    i__5 = nargs;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				same = same && isame[i__ - 1];
				if (! isame[i__ - 1]) {
				    io___279.ciunit = *nout;
				    s_wsfe(&io___279);
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

				if (conj) {
				    *(unsigned char *)transt = 'C';
				} else {
				    *(unsigned char *)transt = 'T';
				}
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
					zmmch_(transt, "N", &lj, &c__1, &k, &
						alpha, &a[jj * a_dim1 + 1],
						nmax, &a[j * a_dim1 + 1],
						nmax, &beta, &c__[jj + j *
						c_dim1], nmax, &ct[1], &g[1],
						&cc[jc], &ldc, eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1, (ftnlen)1);
				    } else {
					zmmch_("N", transt, &lj, &c__1, &k, &
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
	io___286.ciunit = *nout;
	s_wsfe(&io___286);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___287.ciunit = *nout;
	s_wsfe(&io___287);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L130;

L110:
    if (n > 1) {
	io___288.ciunit = *nout;
	s_wsfe(&io___288);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L120:
    io___289.ciunit = *nout;
    s_wsfe(&io___289);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (conj) {
	io___290.ciunit = *nout;
	s_wsfe(&io___290);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&ralpha, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&rbeta, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___291.ciunit = *nout;
	s_wsfe(&io___291);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__2, (char *)&beta, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L130:
    return 0;


/*     End of ZCHK4. */

} /* zchk4_ */

/* Subroutine */ int zchk5_(char *sname, doublereal *eps, doublereal *thresh,
	integer *nout, integer *ntra, logical *trace, logical *rewi, logical *
	fatal, integer *nidim, integer *idim, integer *nalf, doublecomplex *
	alf, integer *nbet, doublecomplex *bet, integer *nmax, doublecomplex *
	ab, doublecomplex *aa, doublecomplex *as, doublecomplex *bb,
	doublecomplex *bs, doublecomplex *c__, doublecomplex *cc,
	doublecomplex *cs, doublecomplex *ct, doublereal *g, doublecomplex *w,
	 ftnlen sname_len)
{
    /* Initialized data */

    static char icht[2] = "NC";
    static char ichu[2] = "UL";

    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,\002)"
	    ", A,\002,i3,\002, B,\002,i3,\002,\002,f4.1,\002, C,\002,i3,\002)"
	    "           .\002)";
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002(\002,2(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002(\002,f4.1,\002,\002,f4.1,\002)"
	    ", A,\002,i3,\002, B,\002,i3,\002,(\002,f4.1,\002,\002,f4.1,\002)"
	    ", C,\002,i3,\002)    .\002)";
    static char fmt_9992[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
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
    doublecomplex z__1, z__2;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);
    void d_cnjg(doublecomplex *, const doublecomplex *);

    /* Local variables */
    integer i__, j, k, n, ia, ib, jc, ma, na, nc, ik, in, jj, lj, ks, ns, laa,
	     lbb, lda, lcc, ldb, ldc;
    doublecomplex als;
    integer ict, icu;
    doublereal err;
    extern logical lze_(doublecomplex *, doublecomplex *, integer *);
    integer jjab;
    doublecomplex beta;
    integer ldas, ldbs, ldcs;
    logical same, conj;
    doublecomplex bets;
    logical tran, null;
    char uplo[1];
    doublecomplex alpha;
    doublereal rbeta;
    logical isame[13];
    extern /* Subroutine */ int zmake_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     logical *, doublecomplex *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int zmmch_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, doublereal *, doublecomplex *,
	    integer *, doublereal *, doublereal *, logical *, integer *,
	    logical *, ftnlen, ftnlen);
    doublereal rbets;
    logical reset;
    char trans[1];
    logical upper;
    char uplos[1];
    extern /* Subroutine */ int zher2k_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublereal *, doublecomplex *, integer *, ftnlen,
	    ftnlen), zsyr2k_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *, ftnlen,
	    ftnlen);
    doublereal errmax;
    extern logical lzeres_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, ftnlen, ftnlen);
    char transs[1], transt[1];

    /* Fortran I/O blocks */
    static cilist io___334 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___335 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___336 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___339 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___347 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___348 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___349 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___350 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___351 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___352 = { 0, 0, 0, fmt_9993, 0 };



/*  Tests ZHER2K and ZSYR2K. */

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
    conj = s_cmp(sname + 1, "HE", (ftnlen)2, (ftnlen)2) == 0;

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

	i__2 = *nidim;
	for (ik = 1; ik <= i__2; ++ik) {
	    k = idim[ik];

	    for (ict = 1; ict <= 2; ++ict) {
		*(unsigned char *)trans = *(unsigned char *)&icht[ict - 1];
		tran = *(unsigned char *)trans == 'C';
		if (tran && ! conj) {
		    *(unsigned char *)trans = 'T';
		}
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
		    zmake_("GE", " ", " ", &ma, &na, &ab[1], &i__3, &aa[1], &
			    lda, &reset, &c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)
			    1);
		} else {
		    zmake_("GE", " ", " ", &ma, &na, &ab[1], nmax, &aa[1], &
			    lda, &reset, &c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)
			    1);
		}

/*              Generate the matrix B. */

		ldb = lda;
		lbb = laa;
		if (tran) {
		    i__3 = *nmax << 1;
		    zmake_("GE", " ", " ", &ma, &na, &ab[k + 1], &i__3, &bb[1]
			    , &ldb, &reset, &c_b1, (ftnlen)2, (ftnlen)1, (
			    ftnlen)1);
		} else {
		    zmake_("GE", " ", " ", &ma, &na, &ab[k * *nmax + 1], nmax,
			     &bb[1], &ldb, &reset, &c_b1, (ftnlen)2, (ftnlen)
			    1, (ftnlen)1);
		}

		for (icu = 1; icu <= 2; ++icu) {
		    *(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];
		    upper = *(unsigned char *)uplo == 'U';

		    i__3 = *nalf;
		    for (ia = 1; ia <= i__3; ++ia) {
			i__4 = ia;
			alpha.r = alf[i__4].r, alpha.i = alf[i__4].i;

			i__4 = *nbet;
			for (ib = 1; ib <= i__4; ++ib) {
			    i__5 = ib;
			    beta.r = bet[i__5].r, beta.i = bet[i__5].i;
			    if (conj) {
				rbeta = beta.r;
				z__1.r = rbeta, z__1.i = 0.;
				beta.r = z__1.r, beta.i = z__1.i;
			    }
			    null = n <= 0;
			    if (conj) {
				null = null || (k <= 0 || alpha.r == 0. &&
					alpha.i == 0.) && rbeta == 1.;
			    }

/*                       Generate the matrix C. */

			    zmake_(sname + 1, uplo, " ", &n, &n, &c__[
				    c_offset], nmax, &cc[1], &ldc, &reset, &
				    c_b1, (ftnlen)2, (ftnlen)1, (ftnlen)1);

			    ++nc;

/*                       Save every datum before calling the subroutine. */

			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    *(unsigned char *)transs = *(unsigned char *)
				    trans;
			    ns = n;
			    ks = k;
			    als.r = alpha.r, als.i = alpha.i;
			    i__5 = laa;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				as[i__6].r = aa[i__7].r, as[i__6].i = aa[i__7]
					.i;
/* L10: */
			    }
			    ldas = lda;
			    i__5 = lbb;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				bs[i__6].r = bb[i__7].r, bs[i__6].i = bb[i__7]
					.i;
/* L20: */
			    }
			    ldbs = ldb;
			    if (conj) {
				rbets = rbeta;
			    } else {
				bets.r = beta.r, bets.i = beta.i;
			    }
			    i__5 = lcc;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				i__6 = i__;
				i__7 = i__;
				cs[i__6].r = cc[i__7].r, cs[i__6].i = cc[i__7]
					.i;
/* L30: */
			    }
			    ldcs = ldc;

/*                       Call the subroutine. */

			    if (conj) {
				if (*trace) {
				    io___334.ciunit = *ntra;
				    s_wsfe(&io___334);
				    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, sname, (ftnlen)6);
				    do_fio(&c__1, uplo, (ftnlen)1);
				    do_fio(&c__1, trans, (ftnlen)1);
				    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__2, (char *)&alpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&ldb, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&rbeta, (ftnlen)
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
				zher2k_(uplo, trans, &n, &k, &alpha, &aa[1], &
					lda, &bb[1], &ldb, &rbeta, &cc[1], &
					ldc, (ftnlen)1, (ftnlen)1);
			    } else {
				if (*trace) {
				    io___335.ciunit = *ntra;
				    s_wsfe(&io___335);
				    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, sname, (ftnlen)6);
				    do_fio(&c__1, uplo, (ftnlen)1);
				    do_fio(&c__1, trans, (ftnlen)1);
				    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__1, (char *)&k, (ftnlen)sizeof(
					    integer));
				    do_fio(&c__2, (char *)&alpha, (ftnlen)
					    sizeof(doublereal));
				    do_fio(&c__1, (char *)&lda, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__1, (char *)&ldb, (ftnlen)
					    sizeof(integer));
				    do_fio(&c__2, (char *)&beta, (ftnlen)
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
				zsyr2k_(uplo, trans, &n, &k, &alpha, &aa[1], &
					lda, &bb[1], &ldb, &beta, &cc[1], &
					ldc, (ftnlen)1, (ftnlen)1);
			    }

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___336.ciunit = *nout;
				s_wsfe(&io___336);
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
			    isame[4] = als.r == alpha.r && als.i == alpha.i;
			    isame[5] = lze_(&as[1], &aa[1], &laa);
			    isame[6] = ldas == lda;
			    isame[7] = lze_(&bs[1], &bb[1], &lbb);
			    isame[8] = ldbs == ldb;
			    if (conj) {
				isame[9] = rbets == rbeta;
			    } else {
				isame[9] = bets.r == beta.r && bets.i ==
					beta.i;
			    }
			    if (null) {
				isame[10] = lze_(&cs[1], &cc[1], &lcc);
			    } else {
				isame[10] = lzeres_("HE", uplo, &n, &n, &cs[1]
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
				    io___339.ciunit = *nout;
				    s_wsfe(&io___339);
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

				if (conj) {
				    *(unsigned char *)transt = 'C';
				} else {
				    *(unsigned char *)transt = 'T';
				}
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
					    i__7 = i__;
					    i__8 = (j - 1 << 1) * *nmax + k +
						    i__;
					    z__1.r = alpha.r * ab[i__8].r -
						    alpha.i * ab[i__8].i,
						    z__1.i = alpha.r * ab[
						    i__8].i + alpha.i * ab[
						    i__8].r;
					    w[i__7].r = z__1.r, w[i__7].i =
						    z__1.i;
					    if (conj) {
			  i__7 = k + i__;
			  d_cnjg(&z__2, &alpha);
			  i__8 = (j - 1 << 1) * *nmax + i__;
			  z__1.r = z__2.r * ab[i__8].r - z__2.i * ab[i__8].i,
				  z__1.i = z__2.r * ab[i__8].i + z__2.i * ab[
				  i__8].r;
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
					    } else {
			  i__7 = k + i__;
			  i__8 = (j - 1 << 1) * *nmax + i__;
			  z__1.r = alpha.r * ab[i__8].r - alpha.i * ab[i__8]
				  .i, z__1.i = alpha.r * ab[i__8].i + alpha.i
				  * ab[i__8].r;
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
					    }
/* L50: */
					}
					i__6 = k << 1;
					i__7 = *nmax << 1;
					i__8 = *nmax << 1;
					zmmch_(transt, "N", &lj, &c__1, &i__6,
						 &c_b2, &ab[jjab], &i__7, &w[
						1], &i__8, &beta, &c__[jj + j
						* c_dim1], nmax, &ct[1], &g[1]
						, &cc[jc], &ldc, eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1, (ftnlen)1);
				    } else {
					i__6 = k;
					for (i__ = 1; i__ <= i__6; ++i__) {
					    if (conj) {
			  i__7 = i__;
			  d_cnjg(&z__2, &ab[(k + i__ - 1) * *nmax + j]);
			  z__1.r = alpha.r * z__2.r - alpha.i * z__2.i,
				  z__1.i = alpha.r * z__2.i + alpha.i *
				  z__2.r;
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
			  i__7 = k + i__;
			  i__8 = (i__ - 1) * *nmax + j;
			  z__2.r = alpha.r * ab[i__8].r - alpha.i * ab[i__8]
				  .i, z__2.i = alpha.r * ab[i__8].i + alpha.i
				  * ab[i__8].r;
			  d_cnjg(&z__1, &z__2);
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
					    } else {
			  i__7 = i__;
			  i__8 = (k + i__ - 1) * *nmax + j;
			  z__1.r = alpha.r * ab[i__8].r - alpha.i * ab[i__8]
				  .i, z__1.i = alpha.r * ab[i__8].i + alpha.i
				  * ab[i__8].r;
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
			  i__7 = k + i__;
			  i__8 = (i__ - 1) * *nmax + j;
			  z__1.r = alpha.r * ab[i__8].r - alpha.i * ab[i__8]
				  .i, z__1.i = alpha.r * ab[i__8].i + alpha.i
				  * ab[i__8].r;
			  w[i__7].r = z__1.r, w[i__7].i = z__1.i;
					    }
/* L60: */
					}
					i__6 = k << 1;
					i__7 = *nmax << 1;
					zmmch_("N", "N", &lj, &c__1, &i__6, &
						c_b2, &ab[jj], nmax, &w[1], &
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
	io___347.ciunit = *nout;
	s_wsfe(&io___347);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___348.ciunit = *nout;
	s_wsfe(&io___348);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    goto L160;

L140:
    if (n > 1) {
	io___349.ciunit = *nout;
	s_wsfe(&io___349);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L150:
    io___350.ciunit = *nout;
    s_wsfe(&io___350);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (conj) {
	io___351.ciunit = *nout;
	s_wsfe(&io___351);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&rbeta, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___352.ciunit = *nout;
	s_wsfe(&io___352);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__2, (char *)&alpha, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&ldb, (ftnlen)sizeof(integer));
	do_fio(&c__2, (char *)&beta, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ldc, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L160:
    return 0;


/*     End of ZCHK5. */

} /* zchk5_ */

/* Subroutine */ int zchke_(integer *isnum, char *srnamt, integer *nout,
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
    doublecomplex a[2]	/* was [2][1] */, b[2]	/* was [2][1] */, c__[2]
	    /* was [2][1] */, beta, alpha;
    doublereal rbeta;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen), zhemm_(char *, char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen), zherk_(char *, char *, integer *,
	    integer *, doublereal *, doublecomplex *, integer *, doublereal *,
	     doublecomplex *, integer *, ftnlen, ftnlen), ztrmm_(char *, char
	    *, char *, char *, integer *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, ftnlen,
	    ftnlen, ftnlen, ftnlen), zsymm_(char *, char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen), ztrsm_(char *, char *, char *, char *,
	     integer *, integer *, doublecomplex *, doublecomplex *, integer *
	    , doublecomplex *, integer *, ftnlen, ftnlen, ftnlen, ftnlen),
	    zsyrk_(char *, char *, integer *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen), zher2k_(char *, char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublereal *, doublecomplex *,
	    integer *, ftnlen, ftnlen), zsyr2k_(char *, char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, ftnlen, ftnlen);
    doublereal ralpha;
    extern /* Subroutine */ int chkxer_(char *, integer *, integer *, logical
	    *, logical *, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___360 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___361 = { 0, 0, 0, fmt_9998, 0 };



/*  Tests the error exits from the Level 3 Blas. */
/*  Requires a special version of the error-handling routine XERBLA. */
/*  A, B and C should not need to be defined. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*  3-19-92:  Initialize ALPHA, BETA, RALPHA, and RBETA  (eca) */
/*  3-19-92:  Fix argument 12 in calls to ZSYMM and ZHEMM */
/*            with INFOT = 9  (eca) */
/*  10-9-00:  Declared INTRINSIC DCMPLX (susan) */

/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. Local Arrays .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */
/*     OK is set to .FALSE. by the special version of XERBLA or by CHKXER */
/*     if anything is wrong. */
    infoc_1.ok = TRUE_;
/*     LERR is set to .TRUE. by the special version of XERBLA each time */
/*     it is called, and is then tested and re-set by CHKXER. */
    infoc_1.lerr = FALSE_;

/*     Initialize ALPHA, BETA, RALPHA, and RBETA. */

    alpha.r = 1., alpha.i = -1.;
    beta.r = 2., beta.i = -2.;
    ralpha = 1.f;
    rbeta = 2.f;

    switch (*isnum) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L40;
	case 5:  goto L50;
	case 6:  goto L60;
	case 7:  goto L70;
	case 8:  goto L80;
	case 9:  goto L90;
    }
L10:
    infoc_1.infot = 1;
    zgemm_("/", "N", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 1;
    zgemm_("/", "C", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 1;
    zgemm_("/", "T", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zgemm_("N", "/", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zgemm_("C", "/", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zgemm_("T", "/", &c__0, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("N", "N", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("N", "C", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("N", "T", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("C", "N", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("C", "C", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("C", "T", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("T", "N", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("T", "C", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zgemm_("T", "T", &c_n1, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("N", "N", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("N", "C", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("N", "T", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("C", "N", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("C", "C", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("C", "T", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("T", "N", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("T", "C", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zgemm_("T", "T", &c__0, &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("N", "N", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("N", "C", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("N", "T", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("C", "N", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("C", "C", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("C", "T", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("T", "N", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("T", "C", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    zgemm_("T", "T", &c__0, &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("N", "N", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("N", "C", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("N", "T", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("C", "N", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__2, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("C", "C", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("C", "T", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("T", "N", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__2, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("T", "C", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    zgemm_("T", "T", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("N", "N", &c__0, &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("C", "N", &c__0, &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("T", "N", &c__0, &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("N", "C", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("C", "C", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("T", "C", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("N", "T", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("C", "T", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zgemm_("T", "T", &c__0, &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("N", "N", &c__2, &c__0, &c__0, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("N", "C", &c__2, &c__0, &c__0, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("N", "T", &c__2, &c__0, &c__0, &alpha, a, &c__2, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("C", "N", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("C", "C", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("C", "T", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("T", "N", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("T", "C", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    zgemm_("T", "T", &c__2, &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta,
	    c__, &c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L20:
    infoc_1.infot = 1;
    zhemm_("/", "U", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zhemm_("L", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zhemm_("L", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zhemm_("R", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zhemm_("L", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zhemm_("R", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zhemm_("L", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zhemm_("R", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zhemm_("L", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zhemm_("R", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zhemm_("L", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zhemm_("R", "U", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zhemm_("L", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zhemm_("R", "L", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zhemm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zhemm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zhemm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zhemm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zhemm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zhemm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zhemm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zhemm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L30:
    infoc_1.infot = 1;
    zsymm_("/", "U", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zsymm_("L", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsymm_("L", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsymm_("R", "U", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsymm_("L", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsymm_("R", "L", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsymm_("L", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsymm_("R", "U", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsymm_("L", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsymm_("R", "L", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsymm_("R", "U", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsymm_("R", "L", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsymm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsymm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsymm_("L", "U", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsymm_("R", "U", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsymm_("L", "L", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsymm_("R", "L", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L40:
    infoc_1.infot = 1;
    ztrmm_("/", "U", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ztrmm_("L", "/", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    ztrmm_("L", "U", "/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    ztrmm_("L", "U", "N", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "U", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "U", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "L", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("L", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "L", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrmm_("R", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "U", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "U", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "L", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("L", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "L", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrmm_("R", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "U", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "U", "C", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "U", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "L", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "L", "C", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrmm_("R", "L", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrmm_("R", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L50:
    infoc_1.infot = 1;
    ztrsm_("/", "U", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ztrsm_("L", "/", "N", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    ztrsm_("L", "U", "/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    ztrsm_("L", "U", "N", "/", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "U", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "U", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "U", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "U", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "L", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("L", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "L", "N", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "L", "C", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ztrsm_("R", "L", "T", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "U", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "U", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "U", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "U", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "L", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("L", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "L", "N", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "L", "C", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ztrsm_("R", "L", "T", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "U", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "U", "C", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "U", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__2, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "L", "N", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "L", "C", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ztrsm_("R", "L", "T", "N", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "U", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "U", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "U", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("L", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "L", "N", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "L", "C", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ztrsm_("R", "L", "T", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, (
	    ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L60:
    infoc_1.infot = 1;
    zherk_("/", "N", &c__0, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zherk_("U", "T", &c__0, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zherk_("U", "N", &c_n1, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zherk_("U", "C", &c_n1, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zherk_("L", "N", &c_n1, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zherk_("L", "C", &c_n1, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zherk_("U", "N", &c__0, &c_n1, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zherk_("U", "C", &c__0, &c_n1, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zherk_("L", "N", &c__0, &c_n1, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zherk_("L", "C", &c__0, &c_n1, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zherk_("U", "N", &c__2, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zherk_("U", "C", &c__0, &c__2, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zherk_("L", "N", &c__2, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zherk_("L", "C", &c__0, &c__2, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zherk_("U", "N", &c__2, &c__0, &ralpha, a, &c__2, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zherk_("U", "C", &c__2, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zherk_("L", "N", &c__2, &c__0, &ralpha, a, &c__2, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zherk_("L", "C", &c__2, &c__0, &ralpha, a, &c__1, &rbeta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L70:
    infoc_1.infot = 1;
    zsyrk_("/", "N", &c__0, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zsyrk_("U", "C", &c__0, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyrk_("U", "N", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyrk_("U", "T", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyrk_("L", "N", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyrk_("L", "T", &c_n1, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyrk_("U", "N", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyrk_("U", "T", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyrk_("L", "N", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyrk_("L", "T", &c__0, &c_n1, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyrk_("U", "N", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyrk_("U", "T", &c__0, &c__2, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyrk_("L", "N", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__2, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyrk_("L", "T", &c__0, &c__2, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zsyrk_("U", "N", &c__2, &c__0, &alpha, a, &c__2, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zsyrk_("U", "T", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zsyrk_("L", "N", &c__2, &c__0, &alpha, a, &c__2, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    zsyrk_("L", "T", &c__2, &c__0, &alpha, a, &c__1, &beta, c__, &c__1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L80:
    infoc_1.infot = 1;
    zher2k_("/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zher2k_("U", "T", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zher2k_("U", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zher2k_("U", "C", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zher2k_("L", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zher2k_("L", "C", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zher2k_("U", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zher2k_("U", "C", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zher2k_("L", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zher2k_("L", "C", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zher2k_("U", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zher2k_("U", "C", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zher2k_("L", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zher2k_("L", "C", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zher2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &rbeta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zher2k_("U", "C", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zher2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &rbeta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zher2k_("L", "C", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zher2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zher2k_("U", "C", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zher2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zher2k_("L", "C", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &rbeta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L100;
L90:
    infoc_1.infot = 1;
    zsyr2k_("/", "N", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    zsyr2k_("U", "C", &c__0, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyr2k_("U", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyr2k_("U", "T", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyr2k_("L", "N", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    zsyr2k_("L", "T", &c_n1, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyr2k_("U", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyr2k_("U", "T", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyr2k_("L", "N", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    zsyr2k_("L", "T", &c__0, &c_n1, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyr2k_("U", "T", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    zsyr2k_("L", "T", &c__0, &c__2, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsyr2k_("U", "T", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__2, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    zsyr2k_("L", "T", &c__0, &c__2, &alpha, a, &c__2, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsyr2k_("U", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsyr2k_("U", "T", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsyr2k_("L", "N", &c__2, &c__0, &alpha, a, &c__2, b, &c__2, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 12;
    zsyr2k_("L", "T", &c__2, &c__0, &alpha, a, &c__1, b, &c__1, &beta, c__, &
	    c__1, (ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);

L100:
    if (infoc_1.ok) {
	io___360.ciunit = *nout;
	s_wsfe(&io___360);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    } else {
	io___361.ciunit = *nout;
	s_wsfe(&io___361);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    }
    return 0;


/*     End of ZCHKE. */

} /* zchke_ */

/* Subroutine */ int zmake_(char *type__, char *uplo, char *diag, integer *m,
	integer *n, doublecomplex *a, integer *nmax, doublecomplex *aa,
	integer *lda, logical *reset, doublecomplex *transl, ftnlen type_len,
	ftnlen uplo_len, ftnlen diag_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublereal d__1;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen);
    void d_cnjg(doublecomplex *, const doublecomplex *);

    /* Local variables */
    integer i__, j, jj;
    logical gen, her, tri, sym;
    integer ibeg, iend;
    extern /* Double Complex */ void zbeg_(doublecomplex *, logical *);
    logical unit, lower, upper;


/*  Generates values for an M by N matrix A. */
/*  Stores the values in the array AA in the data structure required */
/*  by the routine, with unwanted elements set to rogue value. */

/*  TYPE is 'GE', 'HE', 'SY' or 'TR'. */

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
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --aa;

    /* Function Body */
    gen = s_cmp(type__, "GE", (ftnlen)2, (ftnlen)2) == 0;
    her = s_cmp(type__, "HE", (ftnlen)2, (ftnlen)2) == 0;
    sym = s_cmp(type__, "SY", (ftnlen)2, (ftnlen)2) == 0;
    tri = s_cmp(type__, "TR", (ftnlen)2, (ftnlen)2) == 0;
    upper = (her || sym || tri) && *(unsigned char *)uplo == 'U';
    lower = (her || sym || tri) && *(unsigned char *)uplo == 'L';
    unit = tri && *(unsigned char *)diag == 'U';

/*     Generate data in array A. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (gen || upper && i__ <= j || lower && i__ >= j) {
		i__3 = i__ + j * a_dim1;
		zbeg_(&z__2, reset);
		z__1.r = z__2.r + transl->r, z__1.i = z__2.i + transl->i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
		if (i__ != j) {
/*                 Set some elements to zero */
		    if (*n > 3 && j == *n / 2) {
			i__3 = i__ + j * a_dim1;
			a[i__3].r = 0., a[i__3].i = 0.;
		    }
		    if (her) {
			i__3 = j + i__ * a_dim1;
			d_cnjg(&z__1, &a[i__ + j * a_dim1]);
			a[i__3].r = z__1.r, a[i__3].i = z__1.i;
		    } else if (sym) {
			i__3 = j + i__ * a_dim1;
			i__4 = i__ + j * a_dim1;
			a[i__3].r = a[i__4].r, a[i__3].i = a[i__4].i;
		    } else if (tri) {
			i__3 = j + i__ * a_dim1;
			a[i__3].r = 0., a[i__3].i = 0.;
		    }
		}
	    }
/* L10: */
	}
	if (her) {
	    i__2 = j + j * a_dim1;
	    i__3 = j + j * a_dim1;
	    d__1 = a[i__3].r;
	    z__1.r = d__1, z__1.i = 0.;
	    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
	}
	if (tri) {
	    i__2 = j + j * a_dim1;
	    i__3 = j + j * a_dim1;
	    z__1.r = a[i__3].r + 1., z__1.i = a[i__3].i + 0.;
	    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
	}
	if (unit) {
	    i__2 = j + j * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;
	}
/* L20: */
    }

/*     Store elements in array AS in data structure required by routine. */

    if (s_cmp(type__, "GE", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + (j - 1) * *lda;
		i__4 = i__ + j * a_dim1;
		aa[i__3].r = a[i__4].r, aa[i__3].i = a[i__4].i;
/* L30: */
	    }
	    i__2 = *lda;
	    for (i__ = *m + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + (j - 1) * *lda;
		aa[i__3].r = -1e10, aa[i__3].i = 1e10;
/* L40: */
	    }
/* L50: */
	}
    } else if (s_cmp(type__, "HE", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__,
	     "SY", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__, "TR", (ftnlen)
	    2, (ftnlen)2) == 0) {
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
		i__3 = i__ + (j - 1) * *lda;
		aa[i__3].r = -1e10, aa[i__3].i = 1e10;
/* L60: */
	    }
	    i__2 = iend;
	    for (i__ = ibeg; i__ <= i__2; ++i__) {
		i__3 = i__ + (j - 1) * *lda;
		i__4 = i__ + j * a_dim1;
		aa[i__3].r = a[i__4].r, aa[i__3].i = a[i__4].i;
/* L70: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + (j - 1) * *lda;
		aa[i__3].r = -1e10, aa[i__3].i = 1e10;
/* L80: */
	    }
	    if (her) {
		jj = j + (j - 1) * *lda;
		i__2 = jj;
		i__3 = jj;
		d__1 = aa[i__3].r;
		z__1.r = d__1, z__1.i = -1e10;
		aa[i__2].r = z__1.r, aa[i__2].i = z__1.i;
	    }
/* L90: */
	}
    }
    return 0;

/*     End of ZMAKE. */

} /* zmake_ */

/* Subroutine */ int zmmch_(char *transa, char *transb, integer *m, integer *
	n, integer *kk, doublecomplex *alpha, doublecomplex *a, integer *lda,
	doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *
	c__, integer *ldc, doublecomplex *ct, doublereal *g, doublecomplex *
	cc, integer *ldcc, doublereal *eps, doublereal *err, logical *fatal,
	integer *nout, logical *mv, ftnlen transa_len, ftnlen transb_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ******* FATAL ERROR - COMPUTED RESULT IS"
	    " LESS THAN HAL\002,\002F ACCURATE *******\002,/\002             "
	    "          EXPECTED RE\002,\002SULT                    COMPUTED R"
	    "ESULT\002)";
    static char fmt_9998[] = "(1x,i7,2(\002  (\002,g15.6,\002,\002,g15.6,"
	    "\002)\002))";
    static char fmt_9997[] = "(\002      THESE ARE THE RESULTS FOR COLUMN"
	    " \002,i3)";

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, cc_dim1,
	    cc_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    double d_imag(const doublecomplex *);
    void d_cnjg(doublecomplex *, const doublecomplex *);
    double sqrt(doublereal);
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer i__, j, k;
    doublereal erri;
    logical trana, tranb, ctrana, ctranb;

    /* Fortran I/O blocks */
    static cilist io___382 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___383 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___384 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___385 = { 0, 0, 0, fmt_9997, 0 };



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
/*     .. Statement Functions .. */
/*     .. Statement Function definitions .. */
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
    ctrana = *(unsigned char *)transa == 'C';
    ctranb = *(unsigned char *)transb == 'C';

/*     Compute expected result, one column at a time, in CT using data */
/*     in A, B and C. */
/*     Compute gauges in G. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {

	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__;
	    ct[i__3].r = 0., ct[i__3].i = 0.;
	    g[i__] = 0.;
/* L10: */
	}
	if (! trana && ! tranb) {
	    i__2 = *kk;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    i__4 = i__;
		    i__5 = i__;
		    i__6 = i__ + k * a_dim1;
		    i__7 = k + j * b_dim1;
		    z__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[i__7].i,
			    z__2.i = a[i__6].r * b[i__7].i + a[i__6].i * b[
			    i__7].r;
		    z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i +
			    z__2.i;
		    ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
		    i__4 = i__ + k * a_dim1;
		    i__5 = k + j * b_dim1;
		    g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 = d_imag(
			    &a[i__ + k * a_dim1]), abs(d__2))) * ((d__3 = b[
			    i__5].r, abs(d__3)) + (d__4 = d_imag(&b[k + j *
			    b_dim1]), abs(d__4)));
/* L20: */
		}
/* L30: */
	    }
	} else if (trana && ! tranb) {
	    if (ctrana) {
		i__2 = *kk;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			i__5 = i__;
			d_cnjg(&z__3, &a[k + i__ * a_dim1]);
			i__6 = k + j * b_dim1;
			z__2.r = z__3.r * b[i__6].r - z__3.i * b[i__6].i,
				z__2.i = z__3.r * b[i__6].i + z__3.i * b[i__6]
				.r;
			z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i +
				z__2.i;
			ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			i__4 = k + i__ * a_dim1;
			i__5 = k + j * b_dim1;
			g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				d_imag(&a[k + i__ * a_dim1]), abs(d__2))) * ((
				d__3 = b[i__5].r, abs(d__3)) + (d__4 = d_imag(
				&b[k + j * b_dim1]), abs(d__4)));
/* L40: */
		    }
/* L50: */
		}
	    } else {
		i__2 = *kk;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			i__5 = i__;
			i__6 = k + i__ * a_dim1;
			i__7 = k + j * b_dim1;
			z__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[i__7]
				.i, z__2.i = a[i__6].r * b[i__7].i + a[i__6]
				.i * b[i__7].r;
			z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i +
				z__2.i;
			ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			i__4 = k + i__ * a_dim1;
			i__5 = k + j * b_dim1;
			g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				d_imag(&a[k + i__ * a_dim1]), abs(d__2))) * ((
				d__3 = b[i__5].r, abs(d__3)) + (d__4 = d_imag(
				&b[k + j * b_dim1]), abs(d__4)));
/* L60: */
		    }
/* L70: */
		}
	    }
	} else if (! trana && tranb) {
	    if (ctranb) {
		i__2 = *kk;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			i__5 = i__;
			i__6 = i__ + k * a_dim1;
			d_cnjg(&z__3, &b[j + k * b_dim1]);
			z__2.r = a[i__6].r * z__3.r - a[i__6].i * z__3.i,
				z__2.i = a[i__6].r * z__3.i + a[i__6].i *
				z__3.r;
			z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i +
				z__2.i;
			ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			i__4 = i__ + k * a_dim1;
			i__5 = j + k * b_dim1;
			g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				d_imag(&a[i__ + k * a_dim1]), abs(d__2))) * ((
				d__3 = b[i__5].r, abs(d__3)) + (d__4 = d_imag(
				&b[j + k * b_dim1]), abs(d__4)));
/* L80: */
		    }
/* L90: */
		}
	    } else {
		i__2 = *kk;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			i__5 = i__;
			i__6 = i__ + k * a_dim1;
			i__7 = j + k * b_dim1;
			z__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[i__7]
				.i, z__2.i = a[i__6].r * b[i__7].i + a[i__6]
				.i * b[i__7].r;
			z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i +
				z__2.i;
			ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			i__4 = i__ + k * a_dim1;
			i__5 = j + k * b_dim1;
			g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				d_imag(&a[i__ + k * a_dim1]), abs(d__2))) * ((
				d__3 = b[i__5].r, abs(d__3)) + (d__4 = d_imag(
				&b[j + k * b_dim1]), abs(d__4)));
/* L100: */
		    }
/* L110: */
		}
	    }
	} else if (trana && tranb) {
	    if (ctrana) {
		if (ctranb) {
		    i__2 = *kk;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    i__5 = i__;
			    d_cnjg(&z__3, &a[k + i__ * a_dim1]);
			    d_cnjg(&z__4, &b[j + k * b_dim1]);
			    z__2.r = z__3.r * z__4.r - z__3.i * z__4.i,
				    z__2.i = z__3.r * z__4.i + z__3.i *
				    z__4.r;
			    z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i
				    + z__2.i;
			    ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			    i__4 = k + i__ * a_dim1;
			    i__5 = j + k * b_dim1;
			    g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				     d_imag(&a[k + i__ * a_dim1]), abs(d__2)))
				     * ((d__3 = b[i__5].r, abs(d__3)) + (d__4
				    = d_imag(&b[j + k * b_dim1]), abs(d__4)));
/* L120: */
			}
/* L130: */
		    }
		} else {
		    i__2 = *kk;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    i__5 = i__;
			    d_cnjg(&z__3, &a[k + i__ * a_dim1]);
			    i__6 = j + k * b_dim1;
			    z__2.r = z__3.r * b[i__6].r - z__3.i * b[i__6].i,
				    z__2.i = z__3.r * b[i__6].i + z__3.i * b[
				    i__6].r;
			    z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i
				    + z__2.i;
			    ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			    i__4 = k + i__ * a_dim1;
			    i__5 = j + k * b_dim1;
			    g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				     d_imag(&a[k + i__ * a_dim1]), abs(d__2)))
				     * ((d__3 = b[i__5].r, abs(d__3)) + (d__4
				    = d_imag(&b[j + k * b_dim1]), abs(d__4)));
/* L140: */
			}
/* L150: */
		    }
		}
	    } else {
		if (ctranb) {
		    i__2 = *kk;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    i__5 = i__;
			    i__6 = k + i__ * a_dim1;
			    d_cnjg(&z__3, &b[j + k * b_dim1]);
			    z__2.r = a[i__6].r * z__3.r - a[i__6].i * z__3.i,
				    z__2.i = a[i__6].r * z__3.i + a[i__6].i *
				    z__3.r;
			    z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i
				    + z__2.i;
			    ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			    i__4 = k + i__ * a_dim1;
			    i__5 = j + k * b_dim1;
			    g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				     d_imag(&a[k + i__ * a_dim1]), abs(d__2)))
				     * ((d__3 = b[i__5].r, abs(d__3)) + (d__4
				    = d_imag(&b[j + k * b_dim1]), abs(d__4)));
/* L160: */
			}
/* L170: */
		    }
		} else {
		    i__2 = *kk;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    i__5 = i__;
			    i__6 = k + i__ * a_dim1;
			    i__7 = j + k * b_dim1;
			    z__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[
				    i__7].i, z__2.i = a[i__6].r * b[i__7].i +
				    a[i__6].i * b[i__7].r;
			    z__1.r = ct[i__5].r + z__2.r, z__1.i = ct[i__5].i
				    + z__2.i;
			    ct[i__4].r = z__1.r, ct[i__4].i = z__1.i;
			    i__4 = k + i__ * a_dim1;
			    i__5 = j + k * b_dim1;
			    g[i__] += ((d__1 = a[i__4].r, abs(d__1)) + (d__2 =
				     d_imag(&a[k + i__ * a_dim1]), abs(d__2)))
				     * ((d__3 = b[i__5].r, abs(d__3)) + (d__4
				    = d_imag(&b[j + k * b_dim1]), abs(d__4)));
/* L180: */
			}
/* L190: */
		    }
		}
	    }
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__;
	    i__4 = i__;
	    z__2.r = alpha->r * ct[i__4].r - alpha->i * ct[i__4].i, z__2.i =
		    alpha->r * ct[i__4].i + alpha->i * ct[i__4].r;
	    i__5 = i__ + j * c_dim1;
	    z__3.r = beta->r * c__[i__5].r - beta->i * c__[i__5].i, z__3.i =
		    beta->r * c__[i__5].i + beta->i * c__[i__5].r;
	    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	    ct[i__3].r = z__1.r, ct[i__3].i = z__1.i;
	    i__3 = i__ + j * c_dim1;
	    g[i__] = ((d__1 = alpha->r, abs(d__1)) + (d__2 = d_imag(alpha),
		    abs(d__2))) * g[i__] + ((d__3 = beta->r, abs(d__3)) + (
		    d__4 = d_imag(beta), abs(d__4))) * ((d__5 = c__[i__3].r,
		    abs(d__5)) + (d__6 = d_imag(&c__[i__ + j * c_dim1]), abs(
		    d__6)));
/* L200: */
	}

/*        Compute the error ratio for this result. */

	*err = 0.;
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__;
	    i__4 = i__ + j * cc_dim1;
	    z__2.r = ct[i__3].r - cc[i__4].r, z__2.i = ct[i__3].i - cc[i__4]
		    .i;
	    z__1.r = z__2.r, z__1.i = z__2.i;
	    erri = ((d__1 = z__1.r, abs(d__1)) + (d__2 = d_imag(&z__1), abs(
		    d__2))) / *eps;
	    if (g[i__] != 0.) {
		erri /= g[i__];
	    }
	    *err = max(*err,erri);
	    if (*err * sqrt(*eps) >= 1.) {
		goto L230;
	    }
/* L210: */
	}

/* L220: */
    }

/*     If the loop completes, all results are at least half accurate. */
    goto L250;

/*     Report fatal error. */

L230:
    *fatal = TRUE_;
    io___382.ciunit = *nout;
    s_wsfe(&io___382);
    e_wsfe();
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (*mv) {
	    io___383.ciunit = *nout;
	    s_wsfe(&io___383);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__2, (char *)&ct[i__], (ftnlen)sizeof(doublereal));
	    do_fio(&c__2, (char *)&cc[i__ + j * cc_dim1], (ftnlen)sizeof(
		    doublereal));
	    e_wsfe();
	} else {
	    io___384.ciunit = *nout;
	    s_wsfe(&io___384);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__2, (char *)&cc[i__ + j * cc_dim1], (ftnlen)sizeof(
		    doublereal));
	    do_fio(&c__2, (char *)&ct[i__], (ftnlen)sizeof(doublereal));
	    e_wsfe();
	}
/* L240: */
    }
    if (*n > 1) {
	io___385.ciunit = *nout;
	s_wsfe(&io___385);
	do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L250:
    return 0;


/*     End of ZMMCH. */

} /* zmmch_ */

logical lze_(doublecomplex *ri, doublecomplex *rj, integer *lr)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
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
	i__2 = i__;
	i__3 = i__;
	if (ri[i__2].r != rj[i__3].r || ri[i__2].i != rj[i__3].i) {
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

/*     End of LZE. */

} /* lze_ */

logical lzeres_(char *type__, char *uplo, integer *m, integer *n,
	doublecomplex *aa, doublecomplex *as, integer *lda, ftnlen type_len,
	ftnlen uplo_len)
{
    /* System generated locals */
    integer aa_dim1, aa_offset, as_dim1, as_offset, i__1, i__2, i__3, i__4;
    logical ret_val;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen);

    /* Local variables */
    integer i__, j, ibeg, iend;
    logical upper;


/*  Tests if selected elements in two arrays are equal. */

/*  TYPE is 'GE' or 'HE' or 'SY'. */

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
		i__3 = i__ + j * aa_dim1;
		i__4 = i__ + j * as_dim1;
		if (aa[i__3].r != as[i__4].r || aa[i__3].i != as[i__4].i) {
		    goto L70;
		}
/* L10: */
	    }
/* L20: */
	}
    } else if (s_cmp(type__, "HE", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__,
	     "SY", (ftnlen)2, (ftnlen)2) == 0) {
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
		i__3 = i__ + j * aa_dim1;
		i__4 = i__ + j * as_dim1;
		if (aa[i__3].r != as[i__4].r || aa[i__3].i != as[i__4].i) {
		    goto L70;
		}
/* L30: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * aa_dim1;
		i__4 = i__ + j * as_dim1;
		if (aa[i__3].r != as[i__4].r || aa[i__3].i != as[i__4].i) {
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

/*     End of LZERES. */

} /* lzeres_ */

/* Double Complex */ void zbeg_(doublecomplex * ret_val, logical *reset)
{
    /* System generated locals */
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, ic, mi, mj;


/*  Generates complex numbers as pairs of random numbers uniformly */
/*  distributed between -0.5 and 0.5. */

/*  Auxiliary routine for test program for Level 3 Blas. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */

/*     .. Scalar Arguments .. */
/*     .. Local Scalars .. */
/*     .. Save statement .. */
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */
    if (*reset) {
/*        Initialize local variables. */
	mi = 891;
	mj = 457;
	i__ = 7;
	j = 7;
	ic = 0;
	*reset = FALSE_;
    }

/*     The sequence of values of I or J is bounded between 1 and 999. */
/*     If initial I or J = 1,2,3,6,7 or 9, the period will be 50. */
/*     If initial I or J = 4 or 8, the period will be 25. */
/*     If initial I or J = 5, the period will be 10. */
/*     IC is used to break up the period by skipping 1 value of I or J */
/*     in 6. */

    ++ic;
L10:
    i__ *= mi;
    j *= mj;
    i__ -= i__ / 1000 * 1000;
    j -= j / 1000 * 1000;
    if (ic >= 5) {
	ic = 0;
	goto L10;
    }
    d__1 = (i__ - 500) / 1001.;
    d__2 = (j - 500) / 1001.;
    z__1.r = d__1, z__1.i = d__2;
     ret_val->r = z__1.r,  ret_val->i = z__1.i;
    return ;

/*     End of ZBEG. */

} /* zbeg_ */

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
    static cilist io___397 = { 0, 0, 0, fmt_9999, 0 };



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
	io___397.ciunit = *nout;
	s_wsfe(&io___397);
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
    static cilist io___398 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___399 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___400 = { 0, 0, 0, fmt_9998, 0 };



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
	    io___398.ciunit = infoc_2.nout;
	    s_wsfe(&io___398);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&infoc_2.infot, (ftnlen)sizeof(integer));
	    e_wsfe();
	} else {
	    io___399.ciunit = infoc_2.nout;
	    s_wsfe(&io___399);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    e_wsfe();
	}
	infoc_2.ok = FALSE_;
    }
    if (s_cmp(srname, srnamc_1.srnamt, (ftnlen)6, (ftnlen)6) != 0) {
	io___400.ciunit = infoc_2.nout;
	s_wsfe(&io___400);
	do_fio(&c__1, srname, (ftnlen)6);
	do_fio(&c__1, srnamc_1.srnamt, (ftnlen)6);
	e_wsfe();
	infoc_2.ok = FALSE_;
    }
    return 0;


/*     End of XERBLA */

} /* xerbla_ */

/* Main program alias */ int zblat3_ () { main (); return 0; }
