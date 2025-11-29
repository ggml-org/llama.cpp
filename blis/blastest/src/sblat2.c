/* sblat2.f -- translated by f2c (version 20100827).
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
static integer c__4 = 4;
static integer c__65 = 65;
static integer c__7 = 7;
static integer c__2 = 2;
static real c_b120 = 0.f;
static real c_b128 = 1.f;
static logical c_true = TRUE_;
static integer c_n1 = -1;
static integer c__0 = 0;
static logical c_false = FALSE_;

/* > \brief \b SBLAT2 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       PROGRAM SBLAT2 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > Test program for the REAL Level 2 Blas. */
/* > */
/* > The program must be driven by a short data file. The first 18 records */
/* > of the file are read using list-directed input, the last 16 records */
/* > are read using the format ( A6, L2 ). An annotated example of a data */
/* > file can be obtained by deleting the first 3 characters from the */
/* > following 34 lines: */
/* > 'sblat2.out'      NAME OF SUMMARY OUTPUT FILE */
/* > 6                 UNIT NUMBER OF SUMMARY FILE */
/* > 'SBLAT2.SNAP'     NAME OF SNAPSHOT OUTPUT FILE */
/* > -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0) */
/* > F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD. */
/* > F        LOGICAL FLAG, T TO STOP ON FAILURES. */
/* > T        LOGICAL FLAG, T TO TEST ERROR EXITS. */
/* > 16.0     THRESHOLD VALUE OF TEST RATIO */
/* > 6                 NUMBER OF VALUES OF N */
/* > 0 1 2 3 5 9       VALUES OF N */
/* > 4                 NUMBER OF VALUES OF K */
/* > 0 1 2 4           VALUES OF K */
/* > 4                 NUMBER OF VALUES OF INCX AND INCY */
/* > 1 2 -1 -2         VALUES OF INCX AND INCY */
/* > 3                 NUMBER OF VALUES OF ALPHA */
/* > 0.0 1.0 0.7       VALUES OF ALPHA */
/* > 3                 NUMBER OF VALUES OF BETA */
/* > 0.0 1.0 0.9       VALUES OF BETA */
/* > SGEMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SGBMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSYMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSBMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSPMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STRMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STBMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STPMV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STRSV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STBSV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > STPSV  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SGER   T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSYR   T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSPR   T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSYR2  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > SSPR2  T PUT F FOR NO TEST. SAME COLUMNS. */
/* > */
/* > Further Details */
/* > =============== */
/* > */
/* >    See: */
/* > */
/* >       Dongarra J. J., Du Croz J. J., Hammarling S.  and Hanson R. J.. */
/* >       An  extended  set of Fortran  Basic Linear Algebra Subprograms. */
/* > */
/* >       Technical  Memoranda  Nos. 41 (revision 3) and 81,  Mathematics */
/* >       and  Computer Science  Division,  Argonne  National Laboratory, */
/* >       9700 South Cass Avenue, Argonne, Illinois 60439, US. */
/* > */
/* >       Or */
/* > */
/* >       NAG  Technical Reports TR3/87 and TR4/87,  Numerical Algorithms */
/* >       Group  Ltd.,  NAG  Central  Office,  256  Banbury  Road, Oxford */
/* >       OX2 7DE, UK,  and  Numerical Algorithms Group Inc.,  1101  31st */
/* >       Street,  Suite 100,  Downers Grove,  Illinois 60515-1263,  USA. */
/* > */
/* > */
/* > -- Written on 10-August-1987. */
/* >    Richard Hanson, Sandia National Labs. */
/* >    Jeremy Du Croz, NAG Central Office. */
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

/* > \ingroup single_blas_testing */

/*  ===================================================================== */
/* Main program */ int main(void)
{
#ifdef BLIS_ENABLE_HPX
    char* program = "sblat2";
    bli_thread_initialize_hpx( 1, &program );
#endif

    /* Initialized data */

    static char snames[6*16] = "SGEMV " "SGBMV " "SSYMV " "SSBMV " "SSPMV "
	    "STRMV " "STBMV " "STPMV " "STRSV " "STBSV " "STPSV " "SGER  "
	    "SSYR  " "SSPR  " "SSYR2 " "SSPR2 ";

    /* Format strings */
    static char fmt_9997[] = "(\002 NUMBER OF VALUES OF \002,a,\002 IS LESS "
	    "THAN 1 OR GREATER \002,\002THAN \002,i2)";
    static char fmt_9996[] = "(\002 VALUE OF N IS LESS THAN 0 OR GREATER THA"
	    "N \002,i2)";
    static char fmt_9995[] = "(\002 VALUE OF K IS LESS THAN 0\002)";
    static char fmt_9994[] = "(\002 ABSOLUTE VALUE OF INCX OR INCY IS 0 OR G"
	    "REATER THAN \002,i2)";
    static char fmt_9993[] = "(\002 TESTS OF THE REAL             LEVEL 2 BL"
	    "AS\002,//\002 THE F\002,\002OLLOWING PARAMETER VALUES WILL BE US"
	    "ED:\002)";
    static char fmt_9992[] = "(\002   FOR N              \002,9i6)";
    static char fmt_9991[] = "(\002   FOR K              \002,7i6)";
    static char fmt_9990[] = "(\002   FOR INCX AND INCY  \002,7i6)";
    static char fmt_9989[] = "(\002   FOR ALPHA          \002,7f6.1)";
    static char fmt_9988[] = "(\002   FOR BETA           \002,7f6.1)";
    static char fmt_9980[] = "(\002 ERROR-EXITS WILL NOT BE TESTED\002)";
    static char fmt_9999[] = "(\002 ROUTINES PASS COMPUTATIONAL TESTS IF TES"
	    "T RATIO IS LES\002,\002S THAN\002,f8.2)";
    static char fmt_9984[] = "(a6,l2)";
    static char fmt_9986[] = "(\002 SUBPROGRAM NAME \002,a6,\002 NOT RECOGNI"
	    "ZED\002,/\002 ******* T\002,\002ESTS ABANDONED *******\002)";
    static char fmt_9998[] = "(\002 RELATIVE MACHINE PRECISION IS TAKEN TO"
	    " BE\002,1p,e9.1)";
    static char fmt_9985[] = "(\002 ERROR IN SMVCH -  IN-LINE DOT PRODUCTS A"
	    "RE BEING EVALU\002,\002ATED WRONGLY.\002,/\002 SMVCH WAS CALLED "
	    "WITH TRANS = \002,a1,\002 AND RETURNED SAME = \002,l1,\002 AND E"
	    "RR = \002,f12.3,\002.\002,/\002 THIS MAY BE DUE TO FAULTS IN THE"
	    " ARITHMETIC OR THE COMPILER.\002,/\002 ******* TESTS ABANDONED *"
	    "******\002)";
    static char fmt_9983[] = "(1x,a6,\002 WAS NOT TESTED\002)";
    static char fmt_9982[] = "(/\002 END OF TESTS\002)";
    static char fmt_9981[] = "(/\002 ******* FATAL ERROR - TESTS ABANDONED *"
	    "******\002)";
    static char fmt_9987[] = "(\002 AMEND DATA FILE OR INCREASE ARRAY SIZES "
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
    real a[4225]	/* was [65][65] */, g[65];
    integer i__, j, n;
    real x[65], y[65], z__[130], aa[4225];
    integer kb[7];
    real as[4225], xs[130], ys[130], yt[65], xx[130], yy[130], alf[7];
    integer inc[7], nkb;
    real bet[7];
    extern logical lse_(real *, real *, integer *);
    real eps, err;
    integer nalf, idim[9];
    logical same;
    integer ninc, nbet, ntra;
    logical rewi;
    integer nout;
    extern /* Subroutine */ int schk1_(char *, real *, real *, integer *,
	    integer *, logical *, logical *, logical *, integer *, integer *,
	    integer *, integer *, integer *, real *, integer *, real *,
	    integer *, integer *, integer *, integer *, real *, real *, real *
	    , real *, real *, real *, real *, real *, real *, real *, real *,
	    ftnlen), schk2_(char *, real *, real *, integer *, integer *,
	    logical *, logical *, logical *, integer *, integer *, integer *,
	    integer *, integer *, real *, integer *, real *, integer *,
	    integer *, integer *, integer *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, ftnlen),
	    schk3_(char *, real *, real *, integer *, integer *, logical *,
	    logical *, logical *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, real *, real *, real *
	    , real *, real *, real *, real *, real *, real *, ftnlen), schk4_(
	    char *, real *, real *, integer *, integer *, logical *, logical *
	    , logical *, integer *, integer *, integer *, real *, integer *,
	    integer *, integer *, integer *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    ftnlen), schk5_(char *, real *, real *, integer *, integer *,
	    logical *, logical *, logical *, integer *, integer *, integer *,
	    real *, integer *, integer *, integer *, integer *, real *, real *
	    , real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, ftnlen), schk6_(char *, real *, real *, integer *,
	     integer *, logical *, logical *, logical *, integer *, integer *,
	     integer *, real *, integer *, integer *, integer *, integer *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, ftnlen);
    logical fatal;
    extern /* Subroutine */ int schke_(integer *, char *, integer *, ftnlen);
    logical trace;
    integer nidim;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    char snaps[32], trans[1];
    integer isnum;
    logical ltest[16], sfatal;
    char snamet[6];
    real thresh;
    logical ltestt, tsterr;
    char summry[32];
    extern real s_epsilon_(real *);

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
    static cilist io___31 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___32 = { 0, 5, 0, 0, 0 };
    static cilist io___34 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___35 = { 0, 5, 0, 0, 0 };
    static cilist io___37 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___38 = { 0, 5, 0, 0, 0 };
    static cilist io___40 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___41 = { 0, 5, 0, 0, 0 };
    static cilist io___43 = { 0, 5, 0, 0, 0 };
    static cilist io___45 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___46 = { 0, 5, 0, 0, 0 };
    static cilist io___48 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___49 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___50 = { 0, 0, 0, fmt_9991, 0 };
    static cilist io___51 = { 0, 0, 0, fmt_9990, 0 };
    static cilist io___52 = { 0, 0, 0, fmt_9989, 0 };
    static cilist io___53 = { 0, 0, 0, fmt_9988, 0 };
    static cilist io___54 = { 0, 0, 0, 0, 0 };
    static cilist io___55 = { 0, 0, 0, fmt_9980, 0 };
    static cilist io___56 = { 0, 0, 0, 0, 0 };
    static cilist io___57 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___58 = { 0, 0, 0, 0, 0 };
    static cilist io___60 = { 0, 5, 1, fmt_9984, 0 };
    static cilist io___63 = { 0, 0, 0, fmt_9986, 0 };
    static cilist io___65 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___78 = { 0, 0, 0, fmt_9985, 0 };
    static cilist io___79 = { 0, 0, 0, fmt_9985, 0 };
    static cilist io___81 = { 0, 0, 0, 0, 0 };
    static cilist io___82 = { 0, 0, 0, fmt_9983, 0 };
    static cilist io___83 = { 0, 0, 0, 0, 0 };
    static cilist io___90 = { 0, 0, 0, fmt_9982, 0 };
    static cilist io___91 = { 0, 0, 0, fmt_9981, 0 };
    static cilist io___92 = { 0, 0, 0, fmt_9987, 0 };



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
    do_lio(&c__4, &c__1, (char *)&thresh, (ftnlen)sizeof(real));
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
	goto L230;
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
	    goto L230;
	}
/* L10: */
    }
/*     Values of K */
    s_rsle(&io___26);
    do_lio(&c__3, &c__1, (char *)&nkb, (ftnlen)sizeof(integer));
    e_rsle();
    if (nkb < 1 || nkb > 7) {
	io___28.ciunit = nout;
	s_wsfe(&io___28);
	do_fio(&c__1, "K", (ftnlen)1);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L230;
    }
    s_rsle(&io___29);
    i__1 = nkb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__3, &c__1, (char *)&kb[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_rsle();
    i__1 = nkb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (kb[i__ - 1] < 0) {
	    io___31.ciunit = nout;
	    s_wsfe(&io___31);
	    e_wsfe();
	    goto L230;
	}
/* L20: */
    }
/*     Values of INCX and INCY */
    s_rsle(&io___32);
    do_lio(&c__3, &c__1, (char *)&ninc, (ftnlen)sizeof(integer));
    e_rsle();
    if (ninc < 1 || ninc > 7) {
	io___34.ciunit = nout;
	s_wsfe(&io___34);
	do_fio(&c__1, "INCX AND INCY", (ftnlen)13);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L230;
    }
    s_rsle(&io___35);
    i__1 = ninc;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__3, &c__1, (char *)&inc[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_rsle();
    i__1 = ninc;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (inc[i__ - 1] == 0 || (i__2 = inc[i__ - 1], abs(i__2)) > 2) {
	    io___37.ciunit = nout;
	    s_wsfe(&io___37);
	    do_fio(&c__1, (char *)&c__2, (ftnlen)sizeof(integer));
	    e_wsfe();
	    goto L230;
	}
/* L30: */
    }
/*     Values of ALPHA */
    s_rsle(&io___38);
    do_lio(&c__3, &c__1, (char *)&nalf, (ftnlen)sizeof(integer));
    e_rsle();
    if (nalf < 1 || nalf > 7) {
	io___40.ciunit = nout;
	s_wsfe(&io___40);
	do_fio(&c__1, "ALPHA", (ftnlen)5);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L230;
    }
    s_rsle(&io___41);
    i__1 = nalf;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__4, &c__1, (char *)&alf[i__ - 1], (ftnlen)sizeof(real));
    }
    e_rsle();
/*     Values of BETA */
    s_rsle(&io___43);
    do_lio(&c__3, &c__1, (char *)&nbet, (ftnlen)sizeof(integer));
    e_rsle();
    if (nbet < 1 || nbet > 7) {
	io___45.ciunit = nout;
	s_wsfe(&io___45);
	do_fio(&c__1, "BETA", (ftnlen)4);
	do_fio(&c__1, (char *)&c__7, (ftnlen)sizeof(integer));
	e_wsfe();
	goto L230;
    }
    s_rsle(&io___46);
    i__1 = nbet;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_lio(&c__4, &c__1, (char *)&bet[i__ - 1], (ftnlen)sizeof(real));
    }
    e_rsle();

/*     Report values of parameters. */

    io___48.ciunit = nout;
    s_wsfe(&io___48);
    e_wsfe();
    io___49.ciunit = nout;
    s_wsfe(&io___49);
    i__1 = nidim;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&idim[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    io___50.ciunit = nout;
    s_wsfe(&io___50);
    i__1 = nkb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&kb[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    io___51.ciunit = nout;
    s_wsfe(&io___51);
    i__1 = ninc;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&inc[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    io___52.ciunit = nout;
    s_wsfe(&io___52);
    i__1 = nalf;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&alf[i__ - 1], (ftnlen)sizeof(real));
    }
    e_wsfe();
    io___53.ciunit = nout;
    s_wsfe(&io___53);
    i__1 = nbet;
    for (i__ = 1; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&bet[i__ - 1], (ftnlen)sizeof(real));
    }
    e_wsfe();
    if (! tsterr) {
	io___54.ciunit = nout;
	s_wsle(&io___54);
	e_wsle();
	io___55.ciunit = nout;
	s_wsfe(&io___55);
	e_wsfe();
    }
    io___56.ciunit = nout;
    s_wsle(&io___56);
    e_wsle();
    io___57.ciunit = nout;
    s_wsfe(&io___57);
    do_fio(&c__1, (char *)&thresh, (ftnlen)sizeof(real));
    e_wsfe();
    io___58.ciunit = nout;
    s_wsle(&io___58);
    e_wsle();

/*     Read names of subroutines and flags which indicate */
/*     whether they are to be tested. */

    for (i__ = 1; i__ <= 16; ++i__) {
	ltest[i__ - 1] = FALSE_;
/* L40: */
    }
L50:
    i__1 = s_rsfe(&io___60);
    if (i__1 != 0) {
	goto L80;
    }
    i__1 = do_fio(&c__1, snamet, (ftnlen)6);
    if (i__1 != 0) {
	goto L80;
    }
    i__1 = do_fio(&c__1, (char *)&ltestt, (ftnlen)sizeof(logical));
    if (i__1 != 0) {
	goto L80;
    }
    i__1 = e_rsfe();
    if (i__1 != 0) {
	goto L80;
    }
    for (i__ = 1; i__ <= 16; ++i__) {
	if (s_cmp(snamet, snames + (i__ - 1) * 6, (ftnlen)6, (ftnlen)6) == 0)
		{
	    goto L70;
	}
/* L60: */
    }
    io___63.ciunit = nout;
    s_wsfe(&io___63);
    do_fio(&c__1, snamet, (ftnlen)6);
    e_wsfe();
    s_stop("", (ftnlen)0);
L70:
    ltest[i__ - 1] = ltestt;
    goto L50;

L80:
    cl__1.cerr = 0;
    cl__1.cunit = 5;
    cl__1.csta = 0;
    f_clos(&cl__1);

/*     Compute EPS (the machine precision). */

    eps = s_epsilon_(&c_b120);
    io___65.ciunit = nout;
    s_wsfe(&io___65);
    do_fio(&c__1, (char *)&eps, (ftnlen)sizeof(real));
    e_wsfe();

/*     Check the reliability of SMVCH using exact data. */

    n = 32;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__3 = i__ - j + 1;
	    a[i__ + j * 65 - 66] = (real) max(i__3,0);
/* L110: */
	}
	x[j - 1] = (real) j;
	y[j - 1] = 0.f;
/* L120: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	yy[j - 1] = (real) (j * ((j + 1) * j) / 2 - (j + 1) * j * (j - 1) / 3)
		;
/* L130: */
    }
/*     YY holds the exact result. On exit from SMVCH YT holds */
/*     the result computed by SMVCH. */
    *(unsigned char *)trans = 'N';
    smvch_(trans, &n, &n, &c_b128, a, &c__65, x, &c__1, &c_b120, y, &c__1, yt,
	     g, yy, &eps, &err, &fatal, &nout, &c_true, (ftnlen)1);
    same = lse_(yy, yt, &n);
    if (! same || err != 0.f) {
	io___78.ciunit = nout;
	s_wsfe(&io___78);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(real));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }
    *(unsigned char *)trans = 'T';
    smvch_(trans, &n, &n, &c_b128, a, &c__65, x, &c_n1, &c_b120, y, &c_n1, yt,
	     g, yy, &eps, &err, &fatal, &nout, &c_true, (ftnlen)1);
    same = lse_(yy, yt, &n);
    if (! same || err != 0.f) {
	io___79.ciunit = nout;
	s_wsfe(&io___79);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&same, (ftnlen)sizeof(logical));
	do_fio(&c__1, (char *)&err, (ftnlen)sizeof(real));
	e_wsfe();
	s_stop("", (ftnlen)0);
    }

/*     Test each subroutine in turn. */

    for (isnum = 1; isnum <= 16; ++isnum) {
	io___81.ciunit = nout;
	s_wsle(&io___81);
	e_wsle();
	if (! ltest[isnum - 1]) {
/*           Subprogram is not to be tested. */
	    io___82.ciunit = nout;
	    s_wsfe(&io___82);
	    do_fio(&c__1, snames + (isnum - 1) * 6, (ftnlen)6);
	    e_wsfe();
	} else {
	    s_copy(srnamc_1.srnamt, snames + (isnum - 1) * 6, (ftnlen)6, (
		    ftnlen)6);
/*           Test error exits. */
	    if (tsterr) {
		schke_(&isnum, snames + (isnum - 1) * 6, &nout, (ftnlen)6);
		io___83.ciunit = nout;
		s_wsle(&io___83);
		e_wsle();
	    }
/*           Test computations. */
	    infoc_1.infot = 0;
	    infoc_1.ok = TRUE_;
	    fatal = FALSE_;
	    switch (isnum) {
		case 1:  goto L140;
		case 2:  goto L140;
		case 3:  goto L150;
		case 4:  goto L150;
		case 5:  goto L150;
		case 6:  goto L160;
		case 7:  goto L160;
		case 8:  goto L160;
		case 9:  goto L160;
		case 10:  goto L160;
		case 11:  goto L160;
		case 12:  goto L170;
		case 13:  goto L180;
		case 14:  goto L180;
		case 15:  goto L190;
		case 16:  goto L190;
	    }
/*           Test SGEMV, 01, and SGBMV, 02. */
L140:
	    schk1_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nkb, kb, &nalf, alf,
		    &nbet, bet, &ninc, inc, &c__65, &c__2, a, aa, as, x, xx,
		    xs, y, yy, ys, yt, g, (ftnlen)6);
	    goto L200;
/*           Test SSYMV, 03, SSBMV, 04, and SSPMV, 05. */
L150:
	    schk2_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nkb, kb, &nalf, alf,
		    &nbet, bet, &ninc, inc, &c__65, &c__2, a, aa, as, x, xx,
		    xs, y, yy, ys, yt, g, (ftnlen)6);
	    goto L200;
/*           Test STRMV, 06, STBMV, 07, STPMV, 08, */
/*           STRSV, 09, STBSV, 10, and STPSV, 11. */
L160:
	    schk3_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nkb, kb, &ninc, inc,
		    &c__65, &c__2, a, aa, as, y, yy, ys, yt, g, z__, (ftnlen)
		    6);
	    goto L200;
/*           Test SGER, 12. */
L170:
	    schk4_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &ninc,
		    inc, &c__65, &c__2, a, aa, as, x, xx, xs, y, yy, ys, yt,
		    g, z__, (ftnlen)6);
	    goto L200;
/*           Test SSYR, 13, and SSPR, 14. */
L180:
	    schk5_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &ninc,
		    inc, &c__65, &c__2, a, aa, as, x, xx, xs, y, yy, ys, yt,
		    g, z__, (ftnlen)6);
	    goto L200;
/*           Test SSYR2, 15, and SSPR2, 16. */
L190:
	    schk6_(snames + (isnum - 1) * 6, &eps, &thresh, &nout, &ntra, &
		    trace, &rewi, &fatal, &nidim, idim, &nalf, alf, &ninc,
		    inc, &c__65, &c__2, a, aa, as, x, xx, xs, y, yy, ys, yt,
		    g, z__, (ftnlen)6);

L200:
	    if (fatal && sfatal) {
		goto L220;
	    }
	}
/* L210: */
    }
    io___90.ciunit = nout;
    s_wsfe(&io___90);
    e_wsfe();
    goto L240;

L220:
    io___91.ciunit = nout;
    s_wsfe(&io___91);
    e_wsfe();
    goto L240;

L230:
    io___92.ciunit = nout;
    s_wsfe(&io___92);
    e_wsfe();

L240:
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


/*     End of SBLAT2. */

#ifdef BLIS_ENABLE_HPX
    return bli_thread_finalize_hpx();
#else
	// Return peacefully.
	return 0;
#endif
} /* main */

/* Subroutine */ int schk1_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nkb, integer *kb, integer *
	nalf, real *alf, integer *nbet, real *bet, integer *ninc, integer *
	inc, integer *nmax, integer *incmax, real *a, real *aa, real *as,
	real *x, real *xx, real *xs, real *y, real *yy, real *ys, real *yt,
	real *g, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[3] = "NTC";

    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "2(i3,\002,\002),f4.1,\002, A,\002,i3,\002, X,\002,i2,\002,\002,f"
	    "4.1,\002, Y,\002,i2,\002)         .\002)";
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "4(i3,\002,\002),f4.1,\002, A,\002,i3,\002, X,\002,i2,\002,\002,f"
	    "4.1,\002, Y,\002,i2,\002) .\002)";
    static char fmt_9993[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
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
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, m, n, ia, ib, ic, nc, nd, im, in, kl, ml, nk, nl, ku, ix, iy,
	     ms, lx, ly, ns, laa, lda;
    real als, bls;
    extern logical lse_(real *, real *, integer *);
    real err;
    integer iku, kls, kus;
    real beta;
    integer ldas;
    logical same;
    integer incx, incy;
    logical full, tran, null;
    real alpha;
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int sgbmv_(char *, integer *, integer *, integer *
	    , integer *, real *, real *, integer *, real *, integer *, real *,
	     real *, integer *, ftnlen), smvch_(char *, integer *, integer *,
	    real *, real *, integer *, real *, integer *, real *, real *,
	    integer *, real *, real *, real *, real *, real *, logical *,
	    integer *, logical *, ftnlen), sgemv_(char *, integer *, integer *
	    , real *, real *, integer *, real *, integer *, real *, real *,
	    integer *, ftnlen);
    logical reset;
    integer incxs, incys;
    char trans[1];
    logical banded;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;
    char transs[1];

    /* Fortran I/O blocks */
    static cilist io___139 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___140 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___141 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___144 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___146 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___147 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___148 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___149 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___150 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests SGEMV and SGBMV. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    --kb;
    --alf;
    --bet;
    --inc;
    --g;
    --yt;
    --y;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ys;
    --yy;
    --xs;
    --xx;

    /* Function Body */
/*     .. Executable Statements .. */
    full = *(unsigned char *)&sname[2] == 'E';
    banded = *(unsigned char *)&sname[2] == 'B';
/*     Define the number of arguments. */
    if (full) {
	nargs = 11;
    } else if (banded) {
	nargs = 13;
    }

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
	nd = n / 2 + 1;

	for (im = 1; im <= 2; ++im) {
	    if (im == 1) {
/* Computing MAX */
		i__2 = n - nd;
		m = max(i__2,0);
	    }
	    if (im == 2) {
/* Computing MIN */
		i__2 = n + nd;
		m = min(i__2,*nmax);
	    }

	    if (banded) {
		nk = *nkb;
	    } else {
		nk = 1;
	    }
	    i__2 = nk;
	    for (iku = 1; iku <= i__2; ++iku) {
		if (banded) {
		    ku = kb[iku];
/* Computing MAX */
		    i__3 = ku - 1;
		    kl = max(i__3,0);
		} else {
		    ku = n - 1;
		    kl = m - 1;
		}
/*              Set LDA to 1 more than minimum value if room. */
		if (banded) {
		    lda = kl + ku + 1;
		} else {
		    lda = m;
		}
		if (lda < *nmax) {
		    ++lda;
		}
/*              Skip tests if not enough room. */
		if (lda > *nmax) {
		    goto L100;
		}
		laa = lda * n;
		null = n <= 0 || m <= 0;

/*              Generate the matrix A. */

		transl = 0.f;
		smake_(sname + 1, " ", " ", &m, &n, &a[a_offset], nmax, &aa[1]
			, &lda, &kl, &ku, &reset, &transl, (ftnlen)2, (ftnlen)
			1, (ftnlen)1);

		for (ic = 1; ic <= 3; ++ic) {
		    *(unsigned char *)trans = *(unsigned char *)&ich[ic - 1];
		    tran = *(unsigned char *)trans == 'T' || *(unsigned char *
			    )trans == 'C';

		    if (tran) {
			ml = n;
			nl = m;
		    } else {
			ml = m;
			nl = n;
		    }

		    i__3 = *ninc;
		    for (ix = 1; ix <= i__3; ++ix) {
			incx = inc[ix];
			lx = abs(incx) * nl;

/*                    Generate the vector X. */

			transl = .5f;
			i__4 = abs(incx);
			i__5 = nl - 1;
			smake_("GE", " ", " ", &c__1, &nl, &x[1], &c__1, &xx[
				1], &i__4, &c__0, &i__5, &reset, &transl, (
				ftnlen)2, (ftnlen)1, (ftnlen)1);
			if (nl > 1) {
			    x[nl / 2] = 0.f;
			    xx[abs(incx) * (nl / 2 - 1) + 1] = 0.f;
			}

			i__4 = *ninc;
			for (iy = 1; iy <= i__4; ++iy) {
			    incy = inc[iy];
			    ly = abs(incy) * ml;

			    i__5 = *nalf;
			    for (ia = 1; ia <= i__5; ++ia) {
				alpha = alf[ia];

				i__6 = *nbet;
				for (ib = 1; ib <= i__6; ++ib) {
				    beta = bet[ib];

/*                             Generate the vector Y. */

				    transl = 0.f;
				    i__7 = abs(incy);
				    i__8 = ml - 1;
				    smake_("GE", " ", " ", &c__1, &ml, &y[1],
					    &c__1, &yy[1], &i__7, &c__0, &
					    i__8, &reset, &transl, (ftnlen)2,
					    (ftnlen)1, (ftnlen)1);

				    ++nc;

/*                             Save every datum before calling the */
/*                             subroutine. */

				    *(unsigned char *)transs = *(unsigned
					    char *)trans;
				    ms = m;
				    ns = n;
				    kls = kl;
				    kus = ku;
				    als = alpha;
				    i__7 = laa;
				    for (i__ = 1; i__ <= i__7; ++i__) {
					as[i__] = aa[i__];
/* L10: */
				    }
				    ldas = lda;
				    i__7 = lx;
				    for (i__ = 1; i__ <= i__7; ++i__) {
					xs[i__] = xx[i__];
/* L20: */
				    }
				    incxs = incx;
				    bls = beta;
				    i__7 = ly;
				    for (i__ = 1; i__ <= i__7; ++i__) {
					ys[i__] = yy[i__];
/* L30: */
				    }
				    incys = incy;

/*                             Call the subroutine. */

				    if (full) {
					if (*trace) {
					    io___139.ciunit = *ntra;
					    s_wsfe(&io___139);
					    do_fio(&c__1, (char *)&nc, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, sname, (ftnlen)6);
					    do_fio(&c__1, trans, (ftnlen)1);
					    do_fio(&c__1, (char *)&m, (ftnlen)
						    sizeof(integer));
					    do_fio(&c__1, (char *)&n, (ftnlen)
						    sizeof(integer));
					    do_fio(&c__1, (char *)&alpha, (
						    ftnlen)sizeof(real));
					    do_fio(&c__1, (char *)&lda, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&incx, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&beta, (
						    ftnlen)sizeof(real));
					    do_fio(&c__1, (char *)&incy, (
						    ftnlen)sizeof(integer));
					    e_wsfe();
					}
					if (*rewi) {
					    al__1.aerr = 0;
					    al__1.aunit = *ntra;
					    f_rew(&al__1);
					}
					sgemv_(trans, &m, &n, &alpha, &aa[1],
						&lda, &xx[1], &incx, &beta, &
						yy[1], &incy, (ftnlen)1);
				    } else if (banded) {
					if (*trace) {
					    io___140.ciunit = *ntra;
					    s_wsfe(&io___140);
					    do_fio(&c__1, (char *)&nc, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, sname, (ftnlen)6);
					    do_fio(&c__1, trans, (ftnlen)1);
					    do_fio(&c__1, (char *)&m, (ftnlen)
						    sizeof(integer));
					    do_fio(&c__1, (char *)&n, (ftnlen)
						    sizeof(integer));
					    do_fio(&c__1, (char *)&kl, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&ku, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&alpha, (
						    ftnlen)sizeof(real));
					    do_fio(&c__1, (char *)&lda, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&incx, (
						    ftnlen)sizeof(integer));
					    do_fio(&c__1, (char *)&beta, (
						    ftnlen)sizeof(real));
					    do_fio(&c__1, (char *)&incy, (
						    ftnlen)sizeof(integer));
					    e_wsfe();
					}
					if (*rewi) {
					    al__1.aerr = 0;
					    al__1.aunit = *ntra;
					    f_rew(&al__1);
					}
					sgbmv_(trans, &m, &n, &kl, &ku, &
						alpha, &aa[1], &lda, &xx[1], &
						incx, &beta, &yy[1], &incy, (
						ftnlen)1);
				    }

/*                             Check if error-exit was taken incorrectly. */

				    if (! infoc_1.ok) {
					io___141.ciunit = *nout;
					s_wsfe(&io___141);
					e_wsfe();
					*fatal = TRUE_;
					goto L130;
				    }

/*                             See what data changed inside subroutines. */

				    isame[0] = *(unsigned char *)trans == *(
					    unsigned char *)transs;
				    isame[1] = ms == m;
				    isame[2] = ns == n;
				    if (full) {
					isame[3] = als == alpha;
					isame[4] = lse_(&as[1], &aa[1], &laa);
					isame[5] = ldas == lda;
					isame[6] = lse_(&xs[1], &xx[1], &lx);
					isame[7] = incxs == incx;
					isame[8] = bls == beta;
					if (null) {
					    isame[9] = lse_(&ys[1], &yy[1], &
						    ly);
					} else {
					    i__7 = abs(incy);
					    isame[9] = lseres_("GE", " ", &
						    c__1, &ml, &ys[1], &yy[1],
						     &i__7, (ftnlen)2, (
						    ftnlen)1);
					}
					isame[10] = incys == incy;
				    } else if (banded) {
					isame[3] = kls == kl;
					isame[4] = kus == ku;
					isame[5] = als == alpha;
					isame[6] = lse_(&as[1], &aa[1], &laa);
					isame[7] = ldas == lda;
					isame[8] = lse_(&xs[1], &xx[1], &lx);
					isame[9] = incxs == incx;
					isame[10] = bls == beta;
					if (null) {
					    isame[11] = lse_(&ys[1], &yy[1], &
						    ly);
					} else {
					    i__7 = abs(incy);
					    isame[11] = lseres_("GE", " ", &
						    c__1, &ml, &ys[1], &yy[1],
						     &i__7, (ftnlen)2, (
						    ftnlen)1);
					}
					isame[12] = incys == incy;
				    }

/*                             If data was incorrectly changed, report */
/*                             and return. */

				    same = TRUE_;
				    i__7 = nargs;
				    for (i__ = 1; i__ <= i__7; ++i__) {
					same = same && isame[i__ - 1];
					if (! isame[i__ - 1]) {
					    io___144.ciunit = *nout;
					    s_wsfe(&io___144);
					    do_fio(&c__1, (char *)&i__, (
						    ftnlen)sizeof(integer));
					    e_wsfe();
					}
/* L40: */
				    }
				    if (! same) {
					*fatal = TRUE_;
					goto L130;
				    }

				    if (! null) {

/*                                Check the result. */

					smvch_(trans, &m, &n, &alpha, &a[
						a_offset], nmax, &x[1], &incx,
						 &beta, &y[1], &incy, &yt[1],
						&g[1], &yy[1], eps, &err,
						fatal, nout, &c_true, (ftnlen)
						1);
					errmax = max(errmax,err);
/*                                If got really bad answer, report and */
/*                                return. */
					if (*fatal) {
					    goto L130;
					}
				    } else {
/*                                Avoid repeating tests with M.le.0 or */
/*                                N.le.0. */
					goto L110;
				    }

/* L50: */
				}

/* L60: */
			    }

/* L70: */
			}

/* L80: */
		    }

/* L90: */
		}

L100:
		;
	    }

L110:
	    ;
	}

/* L120: */
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___146.ciunit = *nout;
	s_wsfe(&io___146);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___147.ciunit = *nout;
	s_wsfe(&io___147);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L140;

L130:
    io___148.ciunit = *nout;
    s_wsfe(&io___148);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (full) {
	io___149.ciunit = *nout;
	s_wsfe(&io___149);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (banded) {
	io___150.ciunit = *nout;
	s_wsfe(&io___150);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&kl, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&ku, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L140:
    return 0;


/*     End of SCHK1. */

} /* schk1_ */

/* Subroutine */ int schk2_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nkb, integer *kb, integer *
	nalf, real *alf, integer *nbet, real *bet, integer *ninc, integer *
	inc, integer *nmax, integer *incmax, real *a, real *aa, real *as,
	real *x, real *xx, real *xs, real *y, real *yy, real *ys, real *yt,
	real *g, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[2] = "UL";

    /* Format strings */
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, A,\002,i3,\002, X,\002,i2,\002,\002,f4.1,"
	    "\002, Y,\002,i2,\002)             .\002)";
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "2(i3,\002,\002),f4.1,\002, A,\002,i3,\002, X,\002,i2,\002,\002,f"
	    "4.1,\002, Y,\002,i2,\002)         .\002)";
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, AP\002,\002, X,\002,i2,\002,\002,f4.1"
	    ",\002, Y,\002,i2,\002)                .\002)";
    static char fmt_9992[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
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
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, k, n, ia, ib, ic, nc, ik, in, nk, ks, ix, iy, ns, lx, ly,
	    laa, lda;
    real als, bls;
    extern logical lse_(real *, real *, integer *);
    real err, beta;
    integer ldas;
    logical same;
    integer incx, incy;
    logical full, null;
    char uplo[1];
    real alpha;
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    logical reset;
    integer incxs, incys;
    extern /* Subroutine */ int ssbmv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    ftnlen);
    char uplos[1];
    extern /* Subroutine */ int sspmv_(char *, integer *, real *, real *,
	    real *, integer *, real *, real *, integer *, ftnlen), ssymv_(
	    char *, integer *, real *, real *, integer *, real *, integer *,
	    real *, real *, integer *, ftnlen);
    logical banded, packed;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;

    /* Fortran I/O blocks */
    static cilist io___189 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___190 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___191 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___192 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___195 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___197 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___198 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___199 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___200 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___201 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___202 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests SSYMV, SSBMV and SSPMV. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    --kb;
    --alf;
    --bet;
    --inc;
    --g;
    --yt;
    --y;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ys;
    --yy;
    --xs;
    --xx;

    /* Function Body */
/*     .. Executable Statements .. */
    full = *(unsigned char *)&sname[2] == 'Y';
    banded = *(unsigned char *)&sname[2] == 'B';
    packed = *(unsigned char *)&sname[2] == 'P';
/*     Define the number of arguments. */
    if (full) {
	nargs = 10;
    } else if (banded) {
	nargs = 11;
    } else if (packed) {
	nargs = 9;
    }

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];

	if (banded) {
	    nk = *nkb;
	} else {
	    nk = 1;
	}
	i__2 = nk;
	for (ik = 1; ik <= i__2; ++ik) {
	    if (banded) {
		k = kb[ik];
	    } else {
		k = n - 1;
	    }
/*           Set LDA to 1 more than minimum value if room. */
	    if (banded) {
		lda = k + 1;
	    } else {
		lda = n;
	    }
	    if (lda < *nmax) {
		++lda;
	    }
/*           Skip tests if not enough room. */
	    if (lda > *nmax) {
		goto L100;
	    }
	    if (packed) {
		laa = n * (n + 1) / 2;
	    } else {
		laa = lda * n;
	    }
	    null = n <= 0;

	    for (ic = 1; ic <= 2; ++ic) {
		*(unsigned char *)uplo = *(unsigned char *)&ich[ic - 1];

/*              Generate the matrix A. */

		transl = 0.f;
		smake_(sname + 1, uplo, " ", &n, &n, &a[a_offset], nmax, &aa[
			1], &lda, &k, &k, &reset, &transl, (ftnlen)2, (ftnlen)
			1, (ftnlen)1);

		i__3 = *ninc;
		for (ix = 1; ix <= i__3; ++ix) {
		    incx = inc[ix];
		    lx = abs(incx) * n;

/*                 Generate the vector X. */

		    transl = .5f;
		    i__4 = abs(incx);
		    i__5 = n - 1;
		    smake_("GE", " ", " ", &c__1, &n, &x[1], &c__1, &xx[1], &
			    i__4, &c__0, &i__5, &reset, &transl, (ftnlen)2, (
			    ftnlen)1, (ftnlen)1);
		    if (n > 1) {
			x[n / 2] = 0.f;
			xx[abs(incx) * (n / 2 - 1) + 1] = 0.f;
		    }

		    i__4 = *ninc;
		    for (iy = 1; iy <= i__4; ++iy) {
			incy = inc[iy];
			ly = abs(incy) * n;

			i__5 = *nalf;
			for (ia = 1; ia <= i__5; ++ia) {
			    alpha = alf[ia];

			    i__6 = *nbet;
			    for (ib = 1; ib <= i__6; ++ib) {
				beta = bet[ib];

/*                          Generate the vector Y. */

				transl = 0.f;
				i__7 = abs(incy);
				i__8 = n - 1;
				smake_("GE", " ", " ", &c__1, &n, &y[1], &
					c__1, &yy[1], &i__7, &c__0, &i__8, &
					reset, &transl, (ftnlen)2, (ftnlen)1,
					(ftnlen)1);

				++nc;

/*                          Save every datum before calling the */
/*                          subroutine. */

				*(unsigned char *)uplos = *(unsigned char *)
					uplo;
				ns = n;
				ks = k;
				als = alpha;
				i__7 = laa;
				for (i__ = 1; i__ <= i__7; ++i__) {
				    as[i__] = aa[i__];
/* L10: */
				}
				ldas = lda;
				i__7 = lx;
				for (i__ = 1; i__ <= i__7; ++i__) {
				    xs[i__] = xx[i__];
/* L20: */
				}
				incxs = incx;
				bls = beta;
				i__7 = ly;
				for (i__ = 1; i__ <= i__7; ++i__) {
				    ys[i__] = yy[i__];
/* L30: */
				}
				incys = incy;

/*                          Call the subroutine. */

				if (full) {
				    if (*trace) {
					io___189.ciunit = *ntra;
					s_wsfe(&io___189);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&alpha, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&beta, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&incy, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    ssymv_(uplo, &n, &alpha, &aa[1], &lda, &
					    xx[1], &incx, &beta, &yy[1], &
					    incy, (ftnlen)1);
				} else if (banded) {
				    if (*trace) {
					io___190.ciunit = *ntra;
					s_wsfe(&io___190);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&k, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&alpha, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&beta, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&incy, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    ssbmv_(uplo, &n, &k, &alpha, &aa[1], &lda,
					     &xx[1], &incx, &beta, &yy[1], &
					    incy, (ftnlen)1);
				} else if (packed) {
				    if (*trace) {
					io___191.ciunit = *ntra;
					s_wsfe(&io___191);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&alpha, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&beta, (ftnlen)
						sizeof(real));
					do_fio(&c__1, (char *)&incy, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    sspmv_(uplo, &n, &alpha, &aa[1], &xx[1], &
					    incx, &beta, &yy[1], &incy, (
					    ftnlen)1);
				}

/*                          Check if error-exit was taken incorrectly. */

				if (! infoc_1.ok) {
				    io___192.ciunit = *nout;
				    s_wsfe(&io___192);
				    e_wsfe();
				    *fatal = TRUE_;
				    goto L120;
				}

/*                          See what data changed inside subroutines. */

				isame[0] = *(unsigned char *)uplo == *(
					unsigned char *)uplos;
				isame[1] = ns == n;
				if (full) {
				    isame[2] = als == alpha;
				    isame[3] = lse_(&as[1], &aa[1], &laa);
				    isame[4] = ldas == lda;
				    isame[5] = lse_(&xs[1], &xx[1], &lx);
				    isame[6] = incxs == incx;
				    isame[7] = bls == beta;
				    if (null) {
					isame[8] = lse_(&ys[1], &yy[1], &ly);
				    } else {
					i__7 = abs(incy);
					isame[8] = lseres_("GE", " ", &c__1, &
						n, &ys[1], &yy[1], &i__7, (
						ftnlen)2, (ftnlen)1);
				    }
				    isame[9] = incys == incy;
				} else if (banded) {
				    isame[2] = ks == k;
				    isame[3] = als == alpha;
				    isame[4] = lse_(&as[1], &aa[1], &laa);
				    isame[5] = ldas == lda;
				    isame[6] = lse_(&xs[1], &xx[1], &lx);
				    isame[7] = incxs == incx;
				    isame[8] = bls == beta;
				    if (null) {
					isame[9] = lse_(&ys[1], &yy[1], &ly);
				    } else {
					i__7 = abs(incy);
					isame[9] = lseres_("GE", " ", &c__1, &
						n, &ys[1], &yy[1], &i__7, (
						ftnlen)2, (ftnlen)1);
				    }
				    isame[10] = incys == incy;
				} else if (packed) {
				    isame[2] = als == alpha;
				    isame[3] = lse_(&as[1], &aa[1], &laa);
				    isame[4] = lse_(&xs[1], &xx[1], &lx);
				    isame[5] = incxs == incx;
				    isame[6] = bls == beta;
				    if (null) {
					isame[7] = lse_(&ys[1], &yy[1], &ly);
				    } else {
					i__7 = abs(incy);
					isame[7] = lseres_("GE", " ", &c__1, &
						n, &ys[1], &yy[1], &i__7, (
						ftnlen)2, (ftnlen)1);
				    }
				    isame[8] = incys == incy;
				}

/*                          If data was incorrectly changed, report and */
/*                          return. */

				same = TRUE_;
				i__7 = nargs;
				for (i__ = 1; i__ <= i__7; ++i__) {
				    same = same && isame[i__ - 1];
				    if (! isame[i__ - 1]) {
					io___195.ciunit = *nout;
					s_wsfe(&io___195);
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

				    smvch_("N", &n, &n, &alpha, &a[a_offset],
					    nmax, &x[1], &incx, &beta, &y[1],
					    &incy, &yt[1], &g[1], &yy[1], eps,
					     &err, fatal, nout, &c_true, (
					    ftnlen)1);
				    errmax = max(errmax,err);
/*                             If got really bad answer, report and */
/*                             return. */
				    if (*fatal) {
					goto L120;
				    }
				} else {
/*                             Avoid repeating tests with N.le.0 */
				    goto L110;
				}

/* L50: */
			    }

/* L60: */
			}

/* L70: */
		    }

/* L80: */
		}

/* L90: */
	    }

L100:
	    ;
	}

L110:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___197.ciunit = *nout;
	s_wsfe(&io___197);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___198.ciunit = *nout;
	s_wsfe(&io___198);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L130;

L120:
    io___199.ciunit = *nout;
    s_wsfe(&io___199);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (full) {
	io___200.ciunit = *nout;
	s_wsfe(&io___200);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (banded) {
	io___201.ciunit = *nout;
	s_wsfe(&io___201);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (packed) {
	io___202.ciunit = *nout;
	s_wsfe(&io___202);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&beta, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L130:
    return 0;


/*     End of SCHK2. */

} /* schk2_ */

/* Subroutine */ int schk3_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nkb, integer *kb, integer *
	ninc, integer *inc, integer *nmax, integer *incmax, real *a, real *aa,
	 real *as, real *x, real *xx, real *xs, real *xt, real *g, real *z__,
	ftnlen sname_len)
{
    /* Initialized data */

    static char ichu[2] = "UL";
    static char icht[3] = "NTC";
    static char ichd[2] = "UN";

    /* Format strings */
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002(\002,3(\002'\002,a1"
	    ",\002',\002),i3,\002, A,\002,i3,\002, X,\002,i2,\002)           "
	    "          .\002)";
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,3(\002'\002,a1"
	    ",\002',\002),2(i3,\002,\002),\002 A,\002,i3,\002, X,\002,i2,\002"
	    ")                 .\002)";
    static char fmt_9995[] = "(1x,i6,\002: \002,a6,\002(\002,3(\002'\002,a1"
	    ",\002',\002),i3,\002, AP, \002,\002X,\002,i2,\002)              "
	    "          .\002)";
    static char fmt_9992[] = "(\002 ******* FATAL ERROR - ERROR-EXIT TAKEN O"
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
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    alist al__1;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen), s_wsfe(cilist *), do_fio(
	    integer *, char *, ftnlen), e_wsfe(void), f_rew(alist *);

    /* Local variables */
    integer i__, k, n, nc, ik, in, nk, ks, ix, ns, lx, laa, icd, lda, ict,
	    icu;
    extern logical lse_(real *, real *, integer *);
    real err;
    char diag[1];
    integer ldas;
    logical same;
    integer incx;
    logical full, null;
    char uplo[1], diags[1];
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    logical reset;
    integer incxs;
    char trans[1];
    extern /* Subroutine */ int stbmv_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, ftnlen, ftnlen,
	    ftnlen), stbsv_(char *, char *, char *, integer *, integer *,
	    real *, integer *, real *, integer *, ftnlen, ftnlen, ftnlen);
    char uplos[1];
    extern /* Subroutine */ int stpmv_(char *, char *, char *, integer *,
	    real *, real *, integer *, ftnlen, ftnlen, ftnlen), strmv_(char *,
	     char *, char *, integer *, real *, integer *, real *, integer *,
	    ftnlen, ftnlen, ftnlen), stpsv_(char *, char *, char *, integer *,
	     real *, real *, integer *, ftnlen, ftnlen, ftnlen), strsv_(char *
	    , char *, char *, integer *, real *, integer *, real *, integer *,
	     ftnlen, ftnlen, ftnlen);
    logical banded, packed;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;
    char transs[1];

    /* Fortran I/O blocks */
    static cilist io___239 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___240 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___241 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___242 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___243 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___244 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___245 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___248 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___250 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___251 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___252 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___253 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___254 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___255 = { 0, 0, 0, fmt_9995, 0 };



/*  Tests STRMV, STBMV, STPMV, STRSV, STBSV and STPSV. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    --kb;
    --inc;
    --z__;
    --g;
    --xt;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --xs;
    --xx;

    /* Function Body */
/*     .. Executable Statements .. */
    full = *(unsigned char *)&sname[2] == 'R';
    banded = *(unsigned char *)&sname[2] == 'B';
    packed = *(unsigned char *)&sname[2] == 'P';
/*     Define the number of arguments. */
    if (full) {
	nargs = 8;
    } else if (banded) {
	nargs = 9;
    } else if (packed) {
	nargs = 7;
    }

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;
/*     Set up zero vector for SMVCH. */
    i__1 = *nmax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__[i__] = 0.f;
/* L10: */
    }

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];

	if (banded) {
	    nk = *nkb;
	} else {
	    nk = 1;
	}
	i__2 = nk;
	for (ik = 1; ik <= i__2; ++ik) {
	    if (banded) {
		k = kb[ik];
	    } else {
		k = n - 1;
	    }
/*           Set LDA to 1 more than minimum value if room. */
	    if (banded) {
		lda = k + 1;
	    } else {
		lda = n;
	    }
	    if (lda < *nmax) {
		++lda;
	    }
/*           Skip tests if not enough room. */
	    if (lda > *nmax) {
		goto L100;
	    }
	    if (packed) {
		laa = n * (n + 1) / 2;
	    } else {
		laa = lda * n;
	    }
	    null = n <= 0;

	    for (icu = 1; icu <= 2; ++icu) {
		*(unsigned char *)uplo = *(unsigned char *)&ichu[icu - 1];

		for (ict = 1; ict <= 3; ++ict) {
		    *(unsigned char *)trans = *(unsigned char *)&icht[ict - 1]
			    ;

		    for (icd = 1; icd <= 2; ++icd) {
			*(unsigned char *)diag = *(unsigned char *)&ichd[icd
				- 1];

/*                    Generate the matrix A. */

			transl = 0.f;
			smake_(sname + 1, uplo, diag, &n, &n, &a[a_offset],
				nmax, &aa[1], &lda, &k, &k, &reset, &transl, (
				ftnlen)2, (ftnlen)1, (ftnlen)1);

			i__3 = *ninc;
			for (ix = 1; ix <= i__3; ++ix) {
			    incx = inc[ix];
			    lx = abs(incx) * n;

/*                       Generate the vector X. */

			    transl = .5f;
			    i__4 = abs(incx);
			    i__5 = n - 1;
			    smake_("GE", " ", " ", &c__1, &n, &x[1], &c__1, &
				    xx[1], &i__4, &c__0, &i__5, &reset, &
				    transl, (ftnlen)2, (ftnlen)1, (ftnlen)1);
			    if (n > 1) {
				x[n / 2] = 0.f;
				xx[abs(incx) * (n / 2 - 1) + 1] = 0.f;
			    }

			    ++nc;

/*                       Save every datum before calling the subroutine. */

			    *(unsigned char *)uplos = *(unsigned char *)uplo;
			    *(unsigned char *)transs = *(unsigned char *)
				    trans;
			    *(unsigned char *)diags = *(unsigned char *)diag;
			    ns = n;
			    ks = k;
			    i__4 = laa;
			    for (i__ = 1; i__ <= i__4; ++i__) {
				as[i__] = aa[i__];
/* L20: */
			    }
			    ldas = lda;
			    i__4 = lx;
			    for (i__ = 1; i__ <= i__4; ++i__) {
				xs[i__] = xx[i__];
/* L30: */
			    }
			    incxs = incx;

/*                       Call the subroutine. */

			    if (s_cmp(sname + 3, "MV", (ftnlen)2, (ftnlen)2)
				    == 0) {
				if (full) {
				    if (*trace) {
					io___239.ciunit = *ntra;
					s_wsfe(&io___239);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    strmv_(uplo, trans, diag, &n, &aa[1], &
					    lda, &xx[1], &incx, (ftnlen)1, (
					    ftnlen)1, (ftnlen)1);
				} else if (banded) {
				    if (*trace) {
					io___240.ciunit = *ntra;
					s_wsfe(&io___240);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&k, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    stbmv_(uplo, trans, diag, &n, &k, &aa[1],
					    &lda, &xx[1], &incx, (ftnlen)1, (
					    ftnlen)1, (ftnlen)1);
				} else if (packed) {
				    if (*trace) {
					io___241.ciunit = *ntra;
					s_wsfe(&io___241);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    stpmv_(uplo, trans, diag, &n, &aa[1], &xx[
					    1], &incx, (ftnlen)1, (ftnlen)1, (
					    ftnlen)1);
				}
			    } else if (s_cmp(sname + 3, "SV", (ftnlen)2, (
				    ftnlen)2) == 0) {
				if (full) {
				    if (*trace) {
					io___242.ciunit = *ntra;
					s_wsfe(&io___242);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    strsv_(uplo, trans, diag, &n, &aa[1], &
					    lda, &xx[1], &incx, (ftnlen)1, (
					    ftnlen)1, (ftnlen)1);
				} else if (banded) {
				    if (*trace) {
					io___243.ciunit = *ntra;
					s_wsfe(&io___243);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&k, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&lda, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    stbsv_(uplo, trans, diag, &n, &k, &aa[1],
					    &lda, &xx[1], &incx, (ftnlen)1, (
					    ftnlen)1, (ftnlen)1);
				} else if (packed) {
				    if (*trace) {
					io___244.ciunit = *ntra;
					s_wsfe(&io___244);
					do_fio(&c__1, (char *)&nc, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, sname, (ftnlen)6);
					do_fio(&c__1, uplo, (ftnlen)1);
					do_fio(&c__1, trans, (ftnlen)1);
					do_fio(&c__1, diag, (ftnlen)1);
					do_fio(&c__1, (char *)&n, (ftnlen)
						sizeof(integer));
					do_fio(&c__1, (char *)&incx, (ftnlen)
						sizeof(integer));
					e_wsfe();
				    }
				    if (*rewi) {
					al__1.aerr = 0;
					al__1.aunit = *ntra;
					f_rew(&al__1);
				    }
				    stpsv_(uplo, trans, diag, &n, &aa[1], &xx[
					    1], &incx, (ftnlen)1, (ftnlen)1, (
					    ftnlen)1);
				}
			    }

/*                       Check if error-exit was taken incorrectly. */

			    if (! infoc_1.ok) {
				io___245.ciunit = *nout;
				s_wsfe(&io___245);
				e_wsfe();
				*fatal = TRUE_;
				goto L120;
			    }

/*                       See what data changed inside subroutines. */

			    isame[0] = *(unsigned char *)uplo == *(unsigned
				    char *)uplos;
			    isame[1] = *(unsigned char *)trans == *(unsigned
				    char *)transs;
			    isame[2] = *(unsigned char *)diag == *(unsigned
				    char *)diags;
			    isame[3] = ns == n;
			    if (full) {
				isame[4] = lse_(&as[1], &aa[1], &laa);
				isame[5] = ldas == lda;
				if (null) {
				    isame[6] = lse_(&xs[1], &xx[1], &lx);
				} else {
				    i__4 = abs(incx);
				    isame[6] = lseres_("GE", " ", &c__1, &n, &
					    xs[1], &xx[1], &i__4, (ftnlen)2, (
					    ftnlen)1);
				}
				isame[7] = incxs == incx;
			    } else if (banded) {
				isame[4] = ks == k;
				isame[5] = lse_(&as[1], &aa[1], &laa);
				isame[6] = ldas == lda;
				if (null) {
				    isame[7] = lse_(&xs[1], &xx[1], &lx);
				} else {
				    i__4 = abs(incx);
				    isame[7] = lseres_("GE", " ", &c__1, &n, &
					    xs[1], &xx[1], &i__4, (ftnlen)2, (
					    ftnlen)1);
				}
				isame[8] = incxs == incx;
			    } else if (packed) {
				isame[4] = lse_(&as[1], &aa[1], &laa);
				if (null) {
				    isame[5] = lse_(&xs[1], &xx[1], &lx);
				} else {
				    i__4 = abs(incx);
				    isame[5] = lseres_("GE", " ", &c__1, &n, &
					    xs[1], &xx[1], &i__4, (ftnlen)2, (
					    ftnlen)1);
				}
				isame[6] = incxs == incx;
			    }

/*                       If data was incorrectly changed, report and */
/*                       return. */

			    same = TRUE_;
			    i__4 = nargs;
			    for (i__ = 1; i__ <= i__4; ++i__) {
				same = same && isame[i__ - 1];
				if (! isame[i__ - 1]) {
				    io___248.ciunit = *nout;
				    s_wsfe(&io___248);
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
				if (s_cmp(sname + 3, "MV", (ftnlen)2, (ftnlen)
					2) == 0) {

/*                             Check the result. */

				    smvch_(trans, &n, &n, &c_b128, &a[
					    a_offset], nmax, &x[1], &incx, &
					    c_b120, &z__[1], &incx, &xt[1], &
					    g[1], &xx[1], eps, &err, fatal,
					    nout, &c_true, (ftnlen)1);
				} else if (s_cmp(sname + 3, "SV", (ftnlen)2, (
					ftnlen)2) == 0) {

/*                             Compute approximation to original vector. */

				    i__4 = n;
				    for (i__ = 1; i__ <= i__4; ++i__) {
					z__[i__] = xx[(i__ - 1) * abs(incx) +
						1];
					xx[(i__ - 1) * abs(incx) + 1] = x[i__]
						;
/* L50: */
				    }
				    smvch_(trans, &n, &n, &c_b128, &a[
					    a_offset], nmax, &z__[1], &incx, &
					    c_b120, &x[1], &incx, &xt[1], &g[
					    1], &xx[1], eps, &err, fatal,
					    nout, &c_false, (ftnlen)1);
				}
				errmax = max(errmax,err);
/*                          If got really bad answer, report and return. */
				if (*fatal) {
				    goto L120;
				}
			    } else {
/*                          Avoid repeating tests with N.le.0. */
				goto L110;
			    }

/* L60: */
			}

/* L70: */
		    }

/* L80: */
		}

/* L90: */
	    }

L100:
	    ;
	}

L110:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___250.ciunit = *nout;
	s_wsfe(&io___250);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___251.ciunit = *nout;
	s_wsfe(&io___251);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L130;

L120:
    io___252.ciunit = *nout;
    s_wsfe(&io___252);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (full) {
	io___253.ciunit = *nout;
	s_wsfe(&io___253);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, diag, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (banded) {
	io___254.ciunit = *nout;
	s_wsfe(&io___254);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, diag, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&k, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (packed) {
	io___255.ciunit = *nout;
	s_wsfe(&io___255);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, trans, (ftnlen)1);
	do_fio(&c__1, diag, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L130:
    return 0;


/*     End of SCHK3. */

} /* schk3_ */

/* Subroutine */ int schk4_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nalf, real *alf, integer *
	ninc, integer *inc, integer *nmax, integer *incmax, real *a, real *aa,
	 real *as, real *x, real *xx, real *xs, real *y, real *yy, real *ys,
	real *yt, real *g, real *z__, ftnlen sname_len)
{
    /* Format strings */
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002(\002,2(i3,\002,\002)"
	    ",f4.1,\002, X,\002,i2,\002, Y,\002,i2,\002, A,\002,i3,\002)     "
	    "             .\002)";
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
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, j, m, n;
    real w[1];
    integer ia, nc, nd, im, in, ms, ix, iy, ns, lx, ly, laa, lda;
    real als;
    extern logical lse_(real *, real *, integer *);
    real err;
    integer ldas;
    logical same;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *);
    integer incx, incy;
    logical null;
    real alpha;
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    logical reset;
    integer incxs, incys;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;

    /* Fortran I/O blocks */
    static cilist io___284 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___285 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___288 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___292 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___293 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___294 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___295 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___296 = { 0, 0, 0, fmt_9994, 0 };



/*  Tests SGER. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
/*     .. Executable Statements .. */
/*     Define the number of arguments. */
    /* Parameter adjustments */
    --idim;
    --alf;
    --inc;
    --z__;
    --g;
    --yt;
    --y;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ys;
    --yy;
    --xs;
    --xx;

    /* Function Body */
    nargs = 9;

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
	nd = n / 2 + 1;

	for (im = 1; im <= 2; ++im) {
	    if (im == 1) {
/* Computing MAX */
		i__2 = n - nd;
		m = max(i__2,0);
	    }
	    if (im == 2) {
/* Computing MIN */
		i__2 = n + nd;
		m = min(i__2,*nmax);
	    }

/*           Set LDA to 1 more than minimum value if room. */
	    lda = m;
	    if (lda < *nmax) {
		++lda;
	    }
/*           Skip tests if not enough room. */
	    if (lda > *nmax) {
		goto L110;
	    }
	    laa = lda * n;
	    null = n <= 0 || m <= 0;

	    i__2 = *ninc;
	    for (ix = 1; ix <= i__2; ++ix) {
		incx = inc[ix];
		lx = abs(incx) * m;

/*              Generate the vector X. */

		transl = .5f;
		i__3 = abs(incx);
		i__4 = m - 1;
		smake_("GE", " ", " ", &c__1, &m, &x[1], &c__1, &xx[1], &i__3,
			 &c__0, &i__4, &reset, &transl, (ftnlen)2, (ftnlen)1,
			(ftnlen)1);
		if (m > 1) {
		    x[m / 2] = 0.f;
		    xx[abs(incx) * (m / 2 - 1) + 1] = 0.f;
		}

		i__3 = *ninc;
		for (iy = 1; iy <= i__3; ++iy) {
		    incy = inc[iy];
		    ly = abs(incy) * n;

/*                 Generate the vector Y. */

		    transl = 0.f;
		    i__4 = abs(incy);
		    i__5 = n - 1;
		    smake_("GE", " ", " ", &c__1, &n, &y[1], &c__1, &yy[1], &
			    i__4, &c__0, &i__5, &reset, &transl, (ftnlen)2, (
			    ftnlen)1, (ftnlen)1);
		    if (n > 1) {
			y[n / 2] = 0.f;
			yy[abs(incy) * (n / 2 - 1) + 1] = 0.f;
		    }

		    i__4 = *nalf;
		    for (ia = 1; ia <= i__4; ++ia) {
			alpha = alf[ia];

/*                    Generate the matrix A. */

			transl = 0.f;
			i__5 = m - 1;
			i__6 = n - 1;
			smake_(sname + 1, " ", " ", &m, &n, &a[a_offset],
				nmax, &aa[1], &lda, &i__5, &i__6, &reset, &
				transl, (ftnlen)2, (ftnlen)1, (ftnlen)1);

			++nc;

/*                    Save every datum before calling the subroutine. */

			ms = m;
			ns = n;
			als = alpha;
			i__5 = laa;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    as[i__] = aa[i__];
/* L10: */
			}
			ldas = lda;
			i__5 = lx;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    xs[i__] = xx[i__];
/* L20: */
			}
			incxs = incx;
			i__5 = ly;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    ys[i__] = yy[i__];
/* L30: */
			}
			incys = incy;

/*                    Call the subroutine. */

			if (*trace) {
			    io___284.ciunit = *ntra;
			    s_wsfe(&io___284);
			    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer)
				    );
			    do_fio(&c__1, sname, (ftnlen)6);
			    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer))
				    ;
			    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer))
				    ;
			    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real)
				    );
			    do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(
				    integer));
			    do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(
				    integer));
			    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
				    integer));
			    e_wsfe();
			}
			if (*rewi) {
			    al__1.aerr = 0;
			    al__1.aunit = *ntra;
			    f_rew(&al__1);
			}
			sger_(&m, &n, &alpha, &xx[1], &incx, &yy[1], &incy, &
				aa[1], &lda);

/*                    Check if error-exit was taken incorrectly. */

			if (! infoc_1.ok) {
			    io___285.ciunit = *nout;
			    s_wsfe(&io___285);
			    e_wsfe();
			    *fatal = TRUE_;
			    goto L140;
			}

/*                    See what data changed inside subroutine. */

			isame[0] = ms == m;
			isame[1] = ns == n;
			isame[2] = als == alpha;
			isame[3] = lse_(&xs[1], &xx[1], &lx);
			isame[4] = incxs == incx;
			isame[5] = lse_(&ys[1], &yy[1], &ly);
			isame[6] = incys == incy;
			if (null) {
			    isame[7] = lse_(&as[1], &aa[1], &laa);
			} else {
			    isame[7] = lseres_("GE", " ", &m, &n, &as[1], &aa[
				    1], &lda, (ftnlen)2, (ftnlen)1);
			}
			isame[8] = ldas == lda;

/*                    If data was incorrectly changed, report and return. */

			same = TRUE_;
			i__5 = nargs;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    same = same && isame[i__ - 1];
			    if (! isame[i__ - 1]) {
				io___288.ciunit = *nout;
				s_wsfe(&io___288);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
/* L40: */
			}
			if (! same) {
			    *fatal = TRUE_;
			    goto L140;
			}

			if (! null) {

/*                       Check the result column by column. */

			    if (incx > 0) {
				i__5 = m;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__] = x[i__];
/* L50: */
				}
			    } else {
				i__5 = m;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__] = x[m - i__ + 1];
/* L60: */
				}
			    }
			    i__5 = n;
			    for (j = 1; j <= i__5; ++j) {
				if (incy > 0) {
				    w[0] = y[j];
				} else {
				    w[0] = y[n - j + 1];
				}
				smvch_("N", &m, &c__1, &alpha, &z__[1], nmax,
					w, &c__1, &c_b128, &a[j * a_dim1 + 1],
					 &c__1, &yt[1], &g[1], &aa[(j - 1) *
					lda + 1], eps, &err, fatal, nout, &
					c_true, (ftnlen)1);
				errmax = max(errmax,err);
/*                          If got really bad answer, report and return. */
				if (*fatal) {
				    goto L130;
				}
/* L70: */
			    }
			} else {
/*                       Avoid repeating tests with M.le.0 or N.le.0. */
			    goto L110;
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

/*     Report result. */

    if (errmax < *thresh) {
	io___292.ciunit = *nout;
	s_wsfe(&io___292);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___293.ciunit = *nout;
	s_wsfe(&io___293);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L150;

L130:
    io___294.ciunit = *nout;
    s_wsfe(&io___294);
    do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
    e_wsfe();

L140:
    io___295.ciunit = *nout;
    s_wsfe(&io___295);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    io___296.ciunit = *nout;
    s_wsfe(&io___296);
    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
    do_fio(&c__1, sname, (ftnlen)6);
    do_fio(&c__1, (char *)&m, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
    do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
    e_wsfe();

L150:
    return 0;


/*     End of SCHK4. */

} /* schk4_ */

/* Subroutine */ int schk5_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nalf, real *alf, integer *
	ninc, integer *inc, integer *nmax, integer *incmax, real *a, real *aa,
	 real *as, real *x, real *xx, real *xs, real *y, real *yy, real *ys,
	real *yt, real *g, real *z__, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[2] = "UL";

    /* Format strings */
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, X,\002,i2,\002, A,\002,i3,\002)         "
	    "               .\002)";
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, X,\002,i2,\002, AP)                     "
	    "      .\002)";
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
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, j, n;
    real w[1];
    integer ia, ja, ic, nc, jj, lj, in, ix, ns, lx, laa, lda;
    real als;
    extern logical lse_(real *, real *, integer *);
    real err;
    integer ldas;
    logical same;
    integer incx;
    logical full, null;
    char uplo[1];
    extern /* Subroutine */ int sspr_(char *, integer *, real *, real *,
	    integer *, real *, ftnlen), ssyr_(char *, integer *, real *, real
	    *, integer *, real *, integer *, ftnlen);
    real alpha;
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    logical reset;
    integer incxs;
    logical upper;
    char uplos[1];
    logical packed;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;

    /* Fortran I/O blocks */
    static cilist io___324 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___325 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___326 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___329 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___336 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___337 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___338 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___339 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___340 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___341 = { 0, 0, 0, fmt_9994, 0 };



/*  Tests SSYR and SSPR. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    --inc;
    --z__;
    --g;
    --yt;
    --y;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ys;
    --yy;
    --xs;
    --xx;

    /* Function Body */
/*     .. Executable Statements .. */
    full = *(unsigned char *)&sname[2] == 'Y';
    packed = *(unsigned char *)&sname[2] == 'P';
/*     Define the number of arguments. */
    if (full) {
	nargs = 7;
    } else if (packed) {
	nargs = 6;
    }

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
/*        Set LDA to 1 more than minimum value if room. */
	lda = n;
	if (lda < *nmax) {
	    ++lda;
	}
/*        Skip tests if not enough room. */
	if (lda > *nmax) {
	    goto L100;
	}
	if (packed) {
	    laa = n * (n + 1) / 2;
	} else {
	    laa = lda * n;
	}

	for (ic = 1; ic <= 2; ++ic) {
	    *(unsigned char *)uplo = *(unsigned char *)&ich[ic - 1];
	    upper = *(unsigned char *)uplo == 'U';

	    i__2 = *ninc;
	    for (ix = 1; ix <= i__2; ++ix) {
		incx = inc[ix];
		lx = abs(incx) * n;

/*              Generate the vector X. */

		transl = .5f;
		i__3 = abs(incx);
		i__4 = n - 1;
		smake_("GE", " ", " ", &c__1, &n, &x[1], &c__1, &xx[1], &i__3,
			 &c__0, &i__4, &reset, &transl, (ftnlen)2, (ftnlen)1,
			(ftnlen)1);
		if (n > 1) {
		    x[n / 2] = 0.f;
		    xx[abs(incx) * (n / 2 - 1) + 1] = 0.f;
		}

		i__3 = *nalf;
		for (ia = 1; ia <= i__3; ++ia) {
		    alpha = alf[ia];
		    null = n <= 0 || alpha == 0.f;

/*                 Generate the matrix A. */

		    transl = 0.f;
		    i__4 = n - 1;
		    i__5 = n - 1;
		    smake_(sname + 1, uplo, " ", &n, &n, &a[a_offset], nmax, &
			    aa[1], &lda, &i__4, &i__5, &reset, &transl, (
			    ftnlen)2, (ftnlen)1, (ftnlen)1);

		    ++nc;

/*                 Save every datum before calling the subroutine. */

		    *(unsigned char *)uplos = *(unsigned char *)uplo;
		    ns = n;
		    als = alpha;
		    i__4 = laa;
		    for (i__ = 1; i__ <= i__4; ++i__) {
			as[i__] = aa[i__];
/* L10: */
		    }
		    ldas = lda;
		    i__4 = lx;
		    for (i__ = 1; i__ <= i__4; ++i__) {
			xs[i__] = xx[i__];
/* L20: */
		    }
		    incxs = incx;

/*                 Call the subroutine. */

		    if (full) {
			if (*trace) {
			    io___324.ciunit = *ntra;
			    s_wsfe(&io___324);
			    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer)
				    );
			    do_fio(&c__1, sname, (ftnlen)6);
			    do_fio(&c__1, uplo, (ftnlen)1);
			    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer))
				    ;
			    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real)
				    );
			    do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(
				    integer));
			    do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
				    integer));
			    e_wsfe();
			}
			if (*rewi) {
			    al__1.aerr = 0;
			    al__1.aunit = *ntra;
			    f_rew(&al__1);
			}
			ssyr_(uplo, &n, &alpha, &xx[1], &incx, &aa[1], &lda, (
				ftnlen)1);
		    } else if (packed) {
			if (*trace) {
			    io___325.ciunit = *ntra;
			    s_wsfe(&io___325);
			    do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer)
				    );
			    do_fio(&c__1, sname, (ftnlen)6);
			    do_fio(&c__1, uplo, (ftnlen)1);
			    do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer))
				    ;
			    do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real)
				    );
			    do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(
				    integer));
			    e_wsfe();
			}
			if (*rewi) {
			    al__1.aerr = 0;
			    al__1.aunit = *ntra;
			    f_rew(&al__1);
			}
			sspr_(uplo, &n, &alpha, &xx[1], &incx, &aa[1], (
				ftnlen)1);
		    }

/*                 Check if error-exit was taken incorrectly. */

		    if (! infoc_1.ok) {
			io___326.ciunit = *nout;
			s_wsfe(&io___326);
			e_wsfe();
			*fatal = TRUE_;
			goto L120;
		    }

/*                 See what data changed inside subroutines. */

		    isame[0] = *(unsigned char *)uplo == *(unsigned char *)
			    uplos;
		    isame[1] = ns == n;
		    isame[2] = als == alpha;
		    isame[3] = lse_(&xs[1], &xx[1], &lx);
		    isame[4] = incxs == incx;
		    if (null) {
			isame[5] = lse_(&as[1], &aa[1], &laa);
		    } else {
			isame[5] = lseres_(sname + 1, uplo, &n, &n, &as[1], &
				aa[1], &lda, (ftnlen)2, (ftnlen)1);
		    }
		    if (! packed) {
			isame[6] = ldas == lda;
		    }

/*                 If data was incorrectly changed, report and return. */

		    same = TRUE_;
		    i__4 = nargs;
		    for (i__ = 1; i__ <= i__4; ++i__) {
			same = same && isame[i__ - 1];
			if (! isame[i__ - 1]) {
			    io___329.ciunit = *nout;
			    s_wsfe(&io___329);
			    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(
				    integer));
			    e_wsfe();
			}
/* L30: */
		    }
		    if (! same) {
			*fatal = TRUE_;
			goto L120;
		    }

		    if (! null) {

/*                    Check the result column by column. */

			if (incx > 0) {
			    i__4 = n;
			    for (i__ = 1; i__ <= i__4; ++i__) {
				z__[i__] = x[i__];
/* L40: */
			    }
			} else {
			    i__4 = n;
			    for (i__ = 1; i__ <= i__4; ++i__) {
				z__[i__] = x[n - i__ + 1];
/* L50: */
			    }
			}
			ja = 1;
			i__4 = n;
			for (j = 1; j <= i__4; ++j) {
			    w[0] = z__[j];
			    if (upper) {
				jj = 1;
				lj = j;
			    } else {
				jj = j;
				lj = n - j + 1;
			    }
			    smvch_("N", &lj, &c__1, &alpha, &z__[jj], &lj, w,
				    &c__1, &c_b128, &a[jj + j * a_dim1], &
				    c__1, &yt[1], &g[1], &aa[ja], eps, &err,
				    fatal, nout, &c_true, (ftnlen)1);
			    if (full) {
				if (upper) {
				    ja += lda;
				} else {
				    ja = ja + lda + 1;
				}
			    } else {
				ja += lj;
			    }
			    errmax = max(errmax,err);
/*                       If got really bad answer, report and return. */
			    if (*fatal) {
				goto L110;
			    }
/* L60: */
			}
		    } else {
/*                    Avoid repeating tests if N.le.0. */
			if (n <= 0) {
			    goto L100;
			}
		    }

/* L70: */
		}

/* L80: */
	    }

/* L90: */
	}

L100:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___336.ciunit = *nout;
	s_wsfe(&io___336);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___337.ciunit = *nout;
	s_wsfe(&io___337);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L130;

L110:
    io___338.ciunit = *nout;
    s_wsfe(&io___338);
    do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
    e_wsfe();

L120:
    io___339.ciunit = *nout;
    s_wsfe(&io___339);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (full) {
	io___340.ciunit = *nout;
	s_wsfe(&io___340);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (packed) {
	io___341.ciunit = *nout;
	s_wsfe(&io___341);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L130:
    return 0;


/*     End of SCHK5. */

} /* schk5_ */

/* Subroutine */ int schk6_(char *sname, real *eps, real *thresh, integer *
	nout, integer *ntra, logical *trace, logical *rewi, logical *fatal,
	integer *nidim, integer *idim, integer *nalf, real *alf, integer *
	ninc, integer *inc, integer *nmax, integer *incmax, real *a, real *aa,
	 real *as, real *x, real *xx, real *xs, real *y, real *yy, real *ys,
	real *yt, real *g, real *z__, ftnlen sname_len)
{
    /* Initialized data */

    static char ich[2] = "UL";

    /* Format strings */
    static char fmt_9993[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, X,\002,i2,\002, Y,\002,i2,\002, A,\002,i"
	    "3,\002)                  .\002)";
    static char fmt_9994[] = "(1x,i6,\002: \002,a6,\002('\002,a1,\002',\002,"
	    "i3,\002,\002,f4.1,\002, X,\002,i2,\002, Y,\002,i2,\002, AP)     "
	    "                .\002)";
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
    integer a_dim1, a_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5,
	    i__6;
    alist al__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
	     f_rew(alist *);

    /* Local variables */
    integer i__, j, n;
    real w[2];
    integer ia, ja, ic, nc, jj, lj, in, ix, iy, ns, lx, ly, laa, lda;
    real als;
    extern logical lse_(real *, real *, integer *);
    real err;
    integer ldas;
    logical same;
    integer incx, incy;
    logical full, null;
    char uplo[1];
    extern /* Subroutine */ int sspr2_(char *, integer *, real *, real *,
	    integer *, real *, integer *, real *, ftnlen), ssyr2_(char *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    integer *, ftnlen);
    real alpha;
    logical isame[13];
    extern /* Subroutine */ int smake_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *,
	    integer *, logical *, real *, ftnlen, ftnlen, ftnlen);
    integer nargs;
    extern /* Subroutine */ int smvch_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    real *, real *, real *, real *, real *, logical *, integer *,
	    logical *, ftnlen);
    logical reset;
    integer incxs, incys;
    logical upper;
    char uplos[1];
    logical packed;
    real errmax;
    extern logical lseres_(char *, char *, integer *, integer *, real *, real
	    *, integer *, ftnlen, ftnlen);
    real transl;

    /* Fortran I/O blocks */
    static cilist io___373 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___374 = { 0, 0, 0, fmt_9994, 0 };
    static cilist io___375 = { 0, 0, 0, fmt_9992, 0 };
    static cilist io___378 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___385 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___386 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___387 = { 0, 0, 0, fmt_9995, 0 };
    static cilist io___388 = { 0, 0, 0, fmt_9996, 0 };
    static cilist io___389 = { 0, 0, 0, fmt_9993, 0 };
    static cilist io___390 = { 0, 0, 0, fmt_9994, 0 };



/*  Tests SSYR2 and SSPR2. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    --inc;
    z_dim1 = *nmax;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --g;
    --yt;
    --y;
    --x;
    --as;
    --aa;
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ys;
    --yy;
    --xs;
    --xx;

    /* Function Body */
/*     .. Executable Statements .. */
    full = *(unsigned char *)&sname[2] == 'Y';
    packed = *(unsigned char *)&sname[2] == 'P';
/*     Define the number of arguments. */
    if (full) {
	nargs = 9;
    } else if (packed) {
	nargs = 8;
    }

    nc = 0;
    reset = TRUE_;
    errmax = 0.f;

    i__1 = *nidim;
    for (in = 1; in <= i__1; ++in) {
	n = idim[in];
/*        Set LDA to 1 more than minimum value if room. */
	lda = n;
	if (lda < *nmax) {
	    ++lda;
	}
/*        Skip tests if not enough room. */
	if (lda > *nmax) {
	    goto L140;
	}
	if (packed) {
	    laa = n * (n + 1) / 2;
	} else {
	    laa = lda * n;
	}

	for (ic = 1; ic <= 2; ++ic) {
	    *(unsigned char *)uplo = *(unsigned char *)&ich[ic - 1];
	    upper = *(unsigned char *)uplo == 'U';

	    i__2 = *ninc;
	    for (ix = 1; ix <= i__2; ++ix) {
		incx = inc[ix];
		lx = abs(incx) * n;

/*              Generate the vector X. */

		transl = .5f;
		i__3 = abs(incx);
		i__4 = n - 1;
		smake_("GE", " ", " ", &c__1, &n, &x[1], &c__1, &xx[1], &i__3,
			 &c__0, &i__4, &reset, &transl, (ftnlen)2, (ftnlen)1,
			(ftnlen)1);
		if (n > 1) {
		    x[n / 2] = 0.f;
		    xx[abs(incx) * (n / 2 - 1) + 1] = 0.f;
		}

		i__3 = *ninc;
		for (iy = 1; iy <= i__3; ++iy) {
		    incy = inc[iy];
		    ly = abs(incy) * n;

/*                 Generate the vector Y. */

		    transl = 0.f;
		    i__4 = abs(incy);
		    i__5 = n - 1;
		    smake_("GE", " ", " ", &c__1, &n, &y[1], &c__1, &yy[1], &
			    i__4, &c__0, &i__5, &reset, &transl, (ftnlen)2, (
			    ftnlen)1, (ftnlen)1);
		    if (n > 1) {
			y[n / 2] = 0.f;
			yy[abs(incy) * (n / 2 - 1) + 1] = 0.f;
		    }

		    i__4 = *nalf;
		    for (ia = 1; ia <= i__4; ++ia) {
			alpha = alf[ia];
			null = n <= 0 || alpha == 0.f;

/*                    Generate the matrix A. */

			transl = 0.f;
			i__5 = n - 1;
			i__6 = n - 1;
			smake_(sname + 1, uplo, " ", &n, &n, &a[a_offset],
				nmax, &aa[1], &lda, &i__5, &i__6, &reset, &
				transl, (ftnlen)2, (ftnlen)1, (ftnlen)1);

			++nc;

/*                    Save every datum before calling the subroutine. */

			*(unsigned char *)uplos = *(unsigned char *)uplo;
			ns = n;
			als = alpha;
			i__5 = laa;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    as[i__] = aa[i__];
/* L10: */
			}
			ldas = lda;
			i__5 = lx;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    xs[i__] = xx[i__];
/* L20: */
			}
			incxs = incx;
			i__5 = ly;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    ys[i__] = yy[i__];
/* L30: */
			}
			incys = incy;

/*                    Call the subroutine. */

			if (full) {
			    if (*trace) {
				io___373.ciunit = *ntra;
				s_wsfe(&io___373);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(
					real));
				do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
			    if (*rewi) {
				al__1.aerr = 0;
				al__1.aunit = *ntra;
				f_rew(&al__1);
			    }
			    ssyr2_(uplo, &n, &alpha, &xx[1], &incx, &yy[1], &
				    incy, &aa[1], &lda, (ftnlen)1);
			} else if (packed) {
			    if (*trace) {
				io___374.ciunit = *ntra;
				s_wsfe(&io___374);
				do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, sname, (ftnlen)6);
				do_fio(&c__1, uplo, (ftnlen)1);
				do_fio(&c__1, (char *)&n, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(
					real));
				do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(
					integer));
				do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
			    if (*rewi) {
				al__1.aerr = 0;
				al__1.aunit = *ntra;
				f_rew(&al__1);
			    }
			    sspr2_(uplo, &n, &alpha, &xx[1], &incx, &yy[1], &
				    incy, &aa[1], (ftnlen)1);
			}

/*                    Check if error-exit was taken incorrectly. */

			if (! infoc_1.ok) {
			    io___375.ciunit = *nout;
			    s_wsfe(&io___375);
			    e_wsfe();
			    *fatal = TRUE_;
			    goto L160;
			}

/*                    See what data changed inside subroutines. */

			isame[0] = *(unsigned char *)uplo == *(unsigned char *
				)uplos;
			isame[1] = ns == n;
			isame[2] = als == alpha;
			isame[3] = lse_(&xs[1], &xx[1], &lx);
			isame[4] = incxs == incx;
			isame[5] = lse_(&ys[1], &yy[1], &ly);
			isame[6] = incys == incy;
			if (null) {
			    isame[7] = lse_(&as[1], &aa[1], &laa);
			} else {
			    isame[7] = lseres_(sname + 1, uplo, &n, &n, &as[1]
				    , &aa[1], &lda, (ftnlen)2, (ftnlen)1);
			}
			if (! packed) {
			    isame[8] = ldas == lda;
			}

/*                    If data was incorrectly changed, report and return. */

			same = TRUE_;
			i__5 = nargs;
			for (i__ = 1; i__ <= i__5; ++i__) {
			    same = same && isame[i__ - 1];
			    if (! isame[i__ - 1]) {
				io___378.ciunit = *nout;
				s_wsfe(&io___378);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(
					integer));
				e_wsfe();
			    }
/* L40: */
			}
			if (! same) {
			    *fatal = TRUE_;
			    goto L160;
			}

			if (! null) {

/*                       Check the result column by column. */

			    if (incx > 0) {
				i__5 = n;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__ + z_dim1] = x[i__];
/* L50: */
				}
			    } else {
				i__5 = n;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__ + z_dim1] = x[n - i__ + 1];
/* L60: */
				}
			    }
			    if (incy > 0) {
				i__5 = n;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__ + (z_dim1 << 1)] = y[i__];
/* L70: */
				}
			    } else {
				i__5 = n;
				for (i__ = 1; i__ <= i__5; ++i__) {
				    z__[i__ + (z_dim1 << 1)] = y[n - i__ + 1];
/* L80: */
				}
			    }
			    ja = 1;
			    i__5 = n;
			    for (j = 1; j <= i__5; ++j) {
				w[0] = z__[j + (z_dim1 << 1)];
				w[1] = z__[j + z_dim1];
				if (upper) {
				    jj = 1;
				    lj = j;
				} else {
				    jj = j;
				    lj = n - j + 1;
				}
				smvch_("N", &lj, &c__2, &alpha, &z__[jj +
					z_dim1], nmax, w, &c__1, &c_b128, &a[
					jj + j * a_dim1], &c__1, &yt[1], &g[1]
					, &aa[ja], eps, &err, fatal, nout, &
					c_true, (ftnlen)1);
				if (full) {
				    if (upper) {
					ja += lda;
				    } else {
					ja = ja + lda + 1;
				    }
				} else {
				    ja += lj;
				}
				errmax = max(errmax,err);
/*                          If got really bad answer, report and return. */
				if (*fatal) {
				    goto L150;
				}
/* L90: */
			    }
			} else {
/*                       Avoid repeating tests with N.le.0. */
			    if (n <= 0) {
				goto L140;
			    }
			}

/* L100: */
		    }

/* L110: */
		}

/* L120: */
	    }

/* L130: */
	}

L140:
	;
    }

/*     Report result. */

    if (errmax < *thresh) {
	io___385.ciunit = *nout;
	s_wsfe(&io___385);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	e_wsfe();
    } else {
	io___386.ciunit = *nout;
	s_wsfe(&io___386);
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&errmax, (ftnlen)sizeof(real));
	e_wsfe();
    }
    goto L170;

L150:
    io___387.ciunit = *nout;
    s_wsfe(&io___387);
    do_fio(&c__1, (char *)&j, (ftnlen)sizeof(integer));
    e_wsfe();

L160:
    io___388.ciunit = *nout;
    s_wsfe(&io___388);
    do_fio(&c__1, sname, (ftnlen)6);
    e_wsfe();
    if (full) {
	io___389.ciunit = *nout;
	s_wsfe(&io___389);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&lda, (ftnlen)sizeof(integer));
	e_wsfe();
    } else if (packed) {
	io___390.ciunit = *nout;
	s_wsfe(&io___390);
	do_fio(&c__1, (char *)&nc, (ftnlen)sizeof(integer));
	do_fio(&c__1, sname, (ftnlen)6);
	do_fio(&c__1, uplo, (ftnlen)1);
	do_fio(&c__1, (char *)&n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&alpha, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&incy, (ftnlen)sizeof(integer));
	e_wsfe();
    }

L170:
    return 0;


/*     End of SCHK6. */

} /* schk6_ */

/* Subroutine */ int schke_(integer *isnum, char *srnamt, integer *nout,
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
    real a[1]	/* was [1][1] */, x[1], y[1], beta;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *), sspr_(char *,
	    integer *, real *, real *, integer *, real *, ftnlen), ssyr_(char
	    *, integer *, real *, real *, integer *, real *, integer *,
	    ftnlen), sspr2_(char *, integer *, real *, real *, integer *,
	    real *, integer *, real *, ftnlen), ssyr2_(char *, integer *,
	    real *, real *, integer *, real *, integer *, real *, integer *,
	    ftnlen);
    real alpha;
    extern /* Subroutine */ int sgbmv_(char *, integer *, integer *, integer *
	    , integer *, real *, real *, integer *, real *, integer *, real *,
	     real *, integer *, ftnlen), sgemv_(char *, integer *, integer *,
	    real *, real *, integer *, real *, integer *, real *, real *,
	    integer *, ftnlen), ssbmv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *,
	    ftnlen), stbmv_(char *, char *, char *, integer *, integer *,
	    real *, integer *, real *, integer *, ftnlen, ftnlen, ftnlen),
	    stbsv_(char *, char *, char *, integer *, integer *, real *,
	    integer *, real *, integer *, ftnlen, ftnlen, ftnlen), sspmv_(
	    char *, integer *, real *, real *, real *, integer *, real *,
	    real *, integer *, ftnlen), stpmv_(char *, char *, char *,
	    integer *, real *, real *, integer *, ftnlen, ftnlen, ftnlen),
	    strmv_(char *, char *, char *, integer *, real *, integer *, real
	    *, integer *, ftnlen, ftnlen, ftnlen), stpsv_(char *, char *,
	    char *, integer *, real *, real *, integer *, ftnlen, ftnlen,
	    ftnlen), ssymv_(char *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *, ftnlen), strsv_(
	    char *, char *, char *, integer *, real *, integer *, real *,
	    integer *, ftnlen, ftnlen, ftnlen), chkxer_(char *, integer *,
	    integer *, logical *, logical *, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___396 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___397 = { 0, 0, 0, fmt_9998, 0 };



/*  Tests the error exits from the Level 2 Blas. */
/*  Requires a special version of the error-handling routine XERBLA. */
/*  ALPHA, BETA, A, X and Y should not need to be defined. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
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
	case 10:  goto L100;
	case 11:  goto L110;
	case 12:  goto L120;
	case 13:  goto L130;
	case 14:  goto L140;
	case 15:  goto L150;
	case 16:  goto L160;
    }
L10:
    infoc_1.infot = 1;
    sgemv_("/", &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sgemv_("N", &c_n1, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    sgemv_("N", &c__0, &c_n1, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    sgemv_("N", &c__2, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    sgemv_("N", &c__0, &c__0, &alpha, a, &c__1, x, &c__0, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    sgemv_("N", &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__0, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L20:
    infoc_1.infot = 1;
    sgbmv_("/", &c__0, &c__0, &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sgbmv_("N", &c_n1, &c__0, &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    sgbmv_("N", &c__0, &c_n1, &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    sgbmv_("N", &c__0, &c__0, &c_n1, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    sgbmv_("N", &c__2, &c__0, &c__0, &c_n1, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    sgbmv_("N", &c__0, &c__0, &c__1, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    sgbmv_("N", &c__0, &c__0, &c__0, &c__0, &alpha, a, &c__1, x, &c__0, &beta,
	     y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 13;
    sgbmv_("N", &c__0, &c__0, &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta,
	     y, &c__0, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L30:
    infoc_1.infot = 1;
    ssymv_("/", &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ssymv_("U", &c_n1, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ssymv_("U", &c__2, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    ssymv_("U", &c__0, &alpha, a, &c__1, x, &c__0, &beta, y, &c__1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 10;
    ssymv_("U", &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__0, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L40:
    infoc_1.infot = 1;
    ssbmv_("/", &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ssbmv_("U", &c_n1, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    ssbmv_("U", &c__0, &c_n1, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    ssbmv_("U", &c__0, &c__1, &alpha, a, &c__1, x, &c__1, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    ssbmv_("U", &c__0, &c__0, &alpha, a, &c__1, x, &c__0, &beta, y, &c__1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 11;
    ssbmv_("U", &c__0, &c__0, &alpha, a, &c__1, x, &c__1, &beta, y, &c__0, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L50:
    infoc_1.infot = 1;
    sspmv_("/", &c__0, &alpha, a, x, &c__1, &beta, y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sspmv_("U", &c_n1, &alpha, a, x, &c__1, &beta, y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    sspmv_("U", &c__0, &alpha, a, x, &c__0, &beta, y, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    sspmv_("U", &c__0, &alpha, a, x, &c__1, &beta, y, &c__0, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L60:
    infoc_1.infot = 1;
    strmv_("/", "N", "N", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    strmv_("U", "/", "N", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    strmv_("U", "N", "/", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    strmv_("U", "N", "N", &c_n1, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    strmv_("U", "N", "N", &c__2, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    strmv_("U", "N", "N", &c__0, a, &c__1, x, &c__0, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L70:
    infoc_1.infot = 1;
    stbmv_("/", "N", "N", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    stbmv_("U", "/", "N", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    stbmv_("U", "N", "/", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    stbmv_("U", "N", "N", &c_n1, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    stbmv_("U", "N", "N", &c__0, &c_n1, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    stbmv_("U", "N", "N", &c__0, &c__1, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    stbmv_("U", "N", "N", &c__0, &c__0, a, &c__1, x, &c__0, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L80:
    infoc_1.infot = 1;
    stpmv_("/", "N", "N", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    stpmv_("U", "/", "N", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    stpmv_("U", "N", "/", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    stpmv_("U", "N", "N", &c_n1, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    stpmv_("U", "N", "N", &c__0, a, x, &c__0, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L90:
    infoc_1.infot = 1;
    strsv_("/", "N", "N", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    strsv_("U", "/", "N", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    strsv_("U", "N", "/", &c__0, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    strsv_("U", "N", "N", &c_n1, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 6;
    strsv_("U", "N", "N", &c__2, a, &c__1, x, &c__1, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 8;
    strsv_("U", "N", "N", &c__0, a, &c__1, x, &c__0, (ftnlen)1, (ftnlen)1, (
	    ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L100:
    infoc_1.infot = 1;
    stbsv_("/", "N", "N", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    stbsv_("U", "/", "N", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    stbsv_("U", "N", "/", &c__0, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    stbsv_("U", "N", "N", &c_n1, &c__0, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    stbsv_("U", "N", "N", &c__0, &c_n1, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    stbsv_("U", "N", "N", &c__0, &c__1, a, &c__1, x, &c__1, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    stbsv_("U", "N", "N", &c__0, &c__0, a, &c__1, x, &c__0, (ftnlen)1, (
	    ftnlen)1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L110:
    infoc_1.infot = 1;
    stpsv_("/", "N", "N", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    stpsv_("U", "/", "N", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 3;
    stpsv_("U", "N", "/", &c__0, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 4;
    stpsv_("U", "N", "N", &c_n1, a, x, &c__1, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    stpsv_("U", "N", "N", &c__0, a, x, &c__0, (ftnlen)1, (ftnlen)1, (ftnlen)1)
	    ;
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L120:
    infoc_1.infot = 1;
    sger_(&c_n1, &c__0, &alpha, x, &c__1, y, &c__1, a, &c__1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sger_(&c__0, &c_n1, &alpha, x, &c__1, y, &c__1, a, &c__1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    sger_(&c__0, &c__0, &alpha, x, &c__0, y, &c__1, a, &c__1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    sger_(&c__0, &c__0, &alpha, x, &c__1, y, &c__0, a, &c__1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    sger_(&c__2, &c__0, &alpha, x, &c__1, y, &c__1, a, &c__1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L130:
    infoc_1.infot = 1;
    ssyr_("/", &c__0, &alpha, x, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ssyr_("U", &c_n1, &alpha, x, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ssyr_("U", &c__0, &alpha, x, &c__0, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    ssyr_("U", &c__2, &alpha, x, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L140:
    infoc_1.infot = 1;
    sspr_("/", &c__0, &alpha, x, &c__1, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sspr_("U", &c_n1, &alpha, x, &c__1, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    sspr_("U", &c__0, &alpha, x, &c__0, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L150:
    infoc_1.infot = 1;
    ssyr2_("/", &c__0, &alpha, x, &c__1, y, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    ssyr2_("U", &c_n1, &alpha, x, &c__1, y, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    ssyr2_("U", &c__0, &alpha, x, &c__0, y, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    ssyr2_("U", &c__0, &alpha, x, &c__1, y, &c__0, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 9;
    ssyr2_("U", &c__2, &alpha, x, &c__1, y, &c__1, a, &c__1, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    goto L170;
L160:
    infoc_1.infot = 1;
    sspr2_("/", &c__0, &alpha, x, &c__1, y, &c__1, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 2;
    sspr2_("U", &c_n1, &alpha, x, &c__1, y, &c__1, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 5;
    sspr2_("U", &c__0, &alpha, x, &c__0, y, &c__1, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);
    infoc_1.infot = 7;
    sspr2_("U", &c__0, &alpha, x, &c__1, y, &c__0, a, (ftnlen)1);
    chkxer_(srnamt, &infoc_1.infot, nout, &infoc_1.lerr, &infoc_1.ok, (ftnlen)
	    6);

L170:
    if (infoc_1.ok) {
	io___396.ciunit = *nout;
	s_wsfe(&io___396);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    } else {
	io___397.ciunit = *nout;
	s_wsfe(&io___397);
	do_fio(&c__1, srnamt, (ftnlen)6);
	e_wsfe();
    }
    return 0;


/*     End of SCHKE. */

} /* schke_ */

/* Subroutine */ int smake_(char *type__, char *uplo, char *diag, integer *m,
	integer *n, real *a, integer *nmax, real *aa, integer *lda, integer *
	kl, integer *ku, logical *reset, real *transl, ftnlen type_len,
	ftnlen uplo_len, ftnlen diag_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Builtin functions */
    integer s_cmp(const char *, const char *, ftnlen, ftnlen);

    /* Local variables */
    integer i__, j, i1, i2, i3, kk;
    logical gen, tri, sym;
    integer ibeg, iend;
    extern real sbeg_(logical *);
    integer ioff;
    logical unit, lower, upper;


/*  Generates values for an M by N matrix A within the bandwidth */
/*  defined by KL and KU. */
/*  Stores the values in the array AA in the data structure required */
/*  by the routine, with unwanted elements set to rogue value. */

/*  TYPE is 'GE', 'GB', 'SY', 'SB', 'SP', 'TR', 'TB' OR 'TP'. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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
    gen = *(unsigned char *)type__ == 'G';
    sym = *(unsigned char *)type__ == 'S';
    tri = *(unsigned char *)type__ == 'T';
    upper = (sym || tri) && *(unsigned char *)uplo == 'U';
    lower = (sym || tri) && *(unsigned char *)uplo == 'L';
    unit = tri && *(unsigned char *)diag == 'U';

/*     Generate data in array A. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (gen || upper && i__ <= j || lower && i__ >= j) {
		if (i__ <= j && j - i__ <= *ku || i__ >= j && i__ - j <= *kl)
			{
		    a[i__ + j * a_dim1] = sbeg_(reset) + *transl;
		} else {
		    a[i__ + j * a_dim1] = 0.f;
		}
		if (i__ != j) {
		    if (sym) {
			a[j + i__ * a_dim1] = a[i__ + j * a_dim1];
		    } else if (tri) {
			a[j + i__ * a_dim1] = 0.f;
		    }
		}
	    }
/* L10: */
	}
	if (tri) {
	    a[j + j * a_dim1] += 1.f;
	}
	if (unit) {
	    a[j + j * a_dim1] = 1.f;
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
		aa[i__ + (j - 1) * *lda] = -1e10f;
/* L40: */
	    }
/* L50: */
	}
    } else if (s_cmp(type__, "GB", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *ku + 1 - j;
	    for (i1 = 1; i1 <= i__2; ++i1) {
		aa[i1 + (j - 1) * *lda] = -1e10f;
/* L60: */
	    }
/* Computing MIN */
	    i__3 = *kl + *ku + 1, i__4 = *ku + 1 + *m - j;
	    i__2 = min(i__3,i__4);
	    for (i2 = i1; i2 <= i__2; ++i2) {
		aa[i2 + (j - 1) * *lda] = a[i2 + j - *ku - 1 + j * a_dim1];
/* L70: */
	    }
	    i__2 = *lda;
	    for (i3 = i2; i3 <= i__2; ++i3) {
		aa[i3 + (j - 1) * *lda] = -1e10f;
/* L80: */
	    }
/* L90: */
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
		aa[i__ + (j - 1) * *lda] = -1e10f;
/* L100: */
	    }
	    i__2 = iend;
	    for (i__ = ibeg; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = a[i__ + j * a_dim1];
/* L110: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10f;
/* L120: */
	    }
/* L130: */
	}
    } else if (s_cmp(type__, "SB", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__,
	     "TB", (ftnlen)2, (ftnlen)2) == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (upper) {
		kk = *kl + 1;
/* Computing MAX */
		i__2 = 1, i__3 = *kl + 2 - j;
		ibeg = max(i__2,i__3);
		if (unit) {
		    iend = *kl;
		} else {
		    iend = *kl + 1;
		}
	    } else {
		kk = 1;
		if (unit) {
		    ibeg = 2;
		} else {
		    ibeg = 1;
		}
/* Computing MIN */
		i__2 = *kl + 1, i__3 = *m + 1 - j;
		iend = min(i__2,i__3);
	    }
	    i__2 = ibeg - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10f;
/* L140: */
	    }
	    i__2 = iend;
	    for (i__ = ibeg; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = a[i__ + j - kk + j * a_dim1];
/* L150: */
	    }
	    i__2 = *lda;
	    for (i__ = iend + 1; i__ <= i__2; ++i__) {
		aa[i__ + (j - 1) * *lda] = -1e10f;
/* L160: */
	    }
/* L170: */
	}
    } else if (s_cmp(type__, "SP", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(type__,
	     "TP", (ftnlen)2, (ftnlen)2) == 0) {
	ioff = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (upper) {
		ibeg = 1;
		iend = j;
	    } else {
		ibeg = j;
		iend = *n;
	    }
	    i__2 = iend;
	    for (i__ = ibeg; i__ <= i__2; ++i__) {
		++ioff;
		aa[ioff] = a[i__ + j * a_dim1];
		if (i__ == j) {
		    if (unit) {
			aa[ioff] = -1e10f;
		    }
		}
/* L180: */
	    }
/* L190: */
	}
    }
    return 0;

/*     End of SMAKE. */

} /* smake_ */

/* Subroutine */ int smvch_(char *trans, integer *m, integer *n, real *alpha,
	real *a, integer *nmax, real *x, integer *incx, real *beta, real *y,
	integer *incy, real *yt, real *g, real *yy, real *eps, real *err,
	logical *fatal, integer *nout, logical *mv, ftnlen trans_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ******* FATAL ERROR - COMPUTED RESULT IS"
	    " LESS THAN HAL\002,\002F ACCURATE *******\002,/\002           EX"
	    "PECTED RESULT   COMPU\002,\002TED RESULT\002)";
    static char fmt_9998[] = "(1x,i7,2g18.6)";

    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    integer i__, j, ml, nl, iy, jx, kx, ky;
    real erri;
    logical tran;
    integer incxl, incyl;

    /* Fortran I/O blocks */
    static cilist io___425 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___426 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___427 = { 0, 0, 0, fmt_9998, 0 };



/*  Checks the results of the computational tests. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

/*     .. Parameters .. */
/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. Local Scalars .. */
/*     .. Intrinsic Functions .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    a_dim1 = *nmax;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;
    --yt;
    --g;
    --yy;

    /* Function Body */
    tran = *(unsigned char *)trans == 'T' || *(unsigned char *)trans == 'C';
    if (tran) {
	ml = *n;
	nl = *m;
    } else {
	ml = *m;
	nl = *n;
    }
    if (*incx < 0) {
	kx = nl;
	incxl = -1;
    } else {
	kx = 1;
	incxl = 1;
    }
    if (*incy < 0) {
	ky = ml;
	incyl = -1;
    } else {
	ky = 1;
	incyl = 1;
    }

/*     Compute expected result in YT using data in A, X and Y. */
/*     Compute gauges in G. */

    iy = ky;
    i__1 = ml;
    for (i__ = 1; i__ <= i__1; ++i__) {
	yt[iy] = 0.f;
	g[iy] = 0.f;
	jx = kx;
	if (tran) {
	    i__2 = nl;
	    for (j = 1; j <= i__2; ++j) {
		yt[iy] += a[j + i__ * a_dim1] * x[jx];
		g[iy] += (r__1 = a[j + i__ * a_dim1] * x[jx], abs(r__1));
		jx += incxl;
/* L10: */
	    }
	} else {
	    i__2 = nl;
	    for (j = 1; j <= i__2; ++j) {
		yt[iy] += a[i__ + j * a_dim1] * x[jx];
		g[iy] += (r__1 = a[i__ + j * a_dim1] * x[jx], abs(r__1));
		jx += incxl;
/* L20: */
	    }
	}
	yt[iy] = *alpha * yt[iy] + *beta * y[iy];
	g[iy] = abs(*alpha) * g[iy] + (r__1 = *beta * y[iy], abs(r__1));
	iy += incyl;
/* L30: */
    }

/*     Compute the error ratio for this result. */

    *err = 0.f;
    i__1 = ml;
    for (i__ = 1; i__ <= i__1; ++i__) {
	erri = (r__1 = yt[i__] - yy[(i__ - 1) * abs(*incy) + 1], abs(r__1)) /
		*eps;
	if (g[i__] != 0.f) {
	    erri /= g[i__];
	}
	*err = max(*err,erri);
	if (*err * sqrt(*eps) >= 1.f) {
	    goto L50;
	}
/* L40: */
    }
/*     If the loop completes, all results are at least half accurate. */
    goto L70;

/*     Report fatal error. */

L50:
    *fatal = TRUE_;
    io___425.ciunit = *nout;
    s_wsfe(&io___425);
    e_wsfe();
    i__1 = ml;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (*mv) {
	    io___426.ciunit = *nout;
	    s_wsfe(&io___426);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&yt[i__], (ftnlen)sizeof(real));
	    do_fio(&c__1, (char *)&yy[(i__ - 1) * abs(*incy) + 1], (ftnlen)
		    sizeof(real));
	    e_wsfe();
	} else {
	    io___427.ciunit = *nout;
	    s_wsfe(&io___427);
	    do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&yy[(i__ - 1) * abs(*incy) + 1], (ftnlen)
		    sizeof(real));
	    do_fio(&c__1, (char *)&yt[i__], (ftnlen)sizeof(real));
	    e_wsfe();
	}
/* L60: */
    }

L70:
    return 0;


/*     End of SMVCH. */

} /* smvch_ */

logical lse_(real *ri, real *rj, integer *lr)
{
    /* System generated locals */
    integer i__1;
    logical ret_val;

    /* Local variables */
    integer i__;


/*  Tests if two arrays are identical. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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

/*     End of LSE. */

} /* lse_ */

logical lseres_(char *type__, char *uplo, integer *m, integer *n, real *aa,
	real *as, integer *lda, ftnlen type_len, ftnlen uplo_len)
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

/*  TYPE is 'GE', 'SY' or 'SP'. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

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

/*     End of LSERES. */

} /* lseres_ */

real sbeg_(logical *reset)
{
    /* System generated locals */
    real ret_val;

    /* Local variables */
    static integer i__, ic, mi;


/*  Generates random numbers uniformly distributed between -0.5 and 0.5. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

/*     .. Scalar Arguments .. */
/*     .. Local Scalars .. */
/*     .. Save statement .. */
/*     .. Intrinsic Functions .. */
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
    ret_val = (real) (i__ - 500) / 1001.f;
    return ret_val;

/*     End of SBEG. */

} /* sbeg_ */

real sdiff_(real *x, real *y)
{
    /* System generated locals */
    real ret_val;


/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */

/*     .. Scalar Arguments .. */
/*     .. Executable Statements .. */
    ret_val = *x - *y;
    return ret_val;

/*     End of SDIFF. */

} /* sdiff_ */

/* Subroutine */ int chkxer_(char *srnamt, integer *infot, integer *nout,
	logical *lerr, logical *ok, ftnlen srnamt_len)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ***** ILLEGAL VALUE OF PARAMETER NUMBER"
	    " \002,i2,\002 NOT D\002,\002ETECTED BY \002,a6,\002 *****\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Fortran I/O blocks */
    static cilist io___437 = { 0, 0, 0, fmt_9999, 0 };



/*  Tests whether XERBLA has detected an error when it should. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

/*     .. Scalar Arguments .. */
/*     .. Executable Statements .. */
    if (! (*lerr)) {
	io___437.ciunit = *nout;
	s_wsfe(&io___437);
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
    static cilist io___438 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___439 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___440 = { 0, 0, 0, fmt_9998, 0 };



/*  This is a special version of XERBLA to be used only as part of */
/*  the test program for testing error exits from the Level 2 BLAS */
/*  routines. */

/*  XERBLA  is an error handler for the Level 2 BLAS routines. */

/*  It is called by the Level 2 BLAS routines if an input parameter is */
/*  invalid. */

/*  Auxiliary routine for test program for Level 2 Blas. */

/*  -- Written on 10-August-1987. */
/*     Richard Hanson, Sandia National Labs. */
/*     Jeremy Du Croz, NAG Central Office. */

/*     .. Scalar Arguments .. */
/*     .. Scalars in Common .. */
/*     .. Common blocks .. */
/*     .. Executable Statements .. */
    infoc_2.lerr = TRUE_;
    if (*info != infoc_2.infot) {
	if (infoc_2.infot != 0) {
	    io___438.ciunit = infoc_2.nout;
	    s_wsfe(&io___438);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    do_fio(&c__1, (char *)&infoc_2.infot, (ftnlen)sizeof(integer));
	    e_wsfe();
	} else {
	    io___439.ciunit = infoc_2.nout;
	    s_wsfe(&io___439);
	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
	    e_wsfe();
	}
	infoc_2.ok = FALSE_;
    }
    if (s_cmp(srname, srnamc_1.srnamt, (ftnlen)6, (ftnlen)6) != 0) {
	io___440.ciunit = infoc_2.nout;
	s_wsfe(&io___440);
	do_fio(&c__1, srname, (ftnlen)6);
	do_fio(&c__1, srnamc_1.srnamt, (ftnlen)6);
	e_wsfe();
	infoc_2.ok = FALSE_;
    }
    return 0;


/*     End of XERBLA */

} /* xerbla_ */

/* Main program alias */ int sblat2_ () { main (); return 0; }
