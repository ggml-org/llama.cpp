#include "blis.h"

#include <float.h>
#include <fenv.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BLIS_ENABLE_LEGACY_LAMCH

double bli_pow_di( bla_double* a, bla_integer* n );

/* Table of constant values */

//static bla_integer c__1 = 1;
static bla_double c_b32 = 0.;

double bli_pow_di(bla_double *ap, bla_integer *bp)
{
	double        pow, x;
	bla_integer       n;
	unsigned long u;

	pow = 1;
	x   = *ap;
	n   = *bp;

	if( n != 0 )
	{
		if( n < 0 )
		{
			n = -n;
			x = 1/x;
		}
		for( u = n; ; )
		{
			if( u & 01 )
				pow *= x;
			if( u >>= 1 )
				x *= x;
			else
				break;
		}
	}
	return pow;
}

bla_double bli_dlamch(bla_character *cmach, ftnlen cmach_len)
{
    /* Initialized data */

    static bla_logical first = TRUE_;

    /* System generated locals */
    bla_integer i__1;
    bla_double ret_val;

    /* Builtin functions */
    double bli_pow_di(bla_double *, bla_integer *);

    /* Local variables */
    static bla_double base;
    static bla_integer beta;
    static bla_double emin, prec, emax;
    static bla_integer imin, imax;
    static bla_logical lrnd;
    static bla_double rmin, rmax, t, rmach;
    extern bla_logical bli_lsame(bla_character *, bla_character *, ftnlen, ftnlen);
    static bla_double smnum, sfmin;
    extern /* Subroutine */ int bli_dlamc2(bla_integer *, bla_integer *, bla_logical *,
	    bla_double *, bla_integer *, bla_double *, bla_integer *, bla_double *);
    static bla_integer it;
    static bla_double rnd, eps;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMCH determines double precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  CMACH   (input) CHARACTER*1 */
/*          Specifies the value to be returned by DLAMCH: */
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	bli_dlamc2(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (bla_double) beta;
	t = (bla_double) it;
	if (lrnd) {
	    rnd = 1.;
	    i__1 = 1 - it;
	    eps = bli_pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.;
	    i__1 = 1 - it;
	    eps = bli_pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (bla_double) imin;
	emax = (bla_double) imax;
	sfmin = rmin;
	smnum = 1. / rmax;
	if (smnum >= sfmin) {

/*           Use SMALL plus a bit, to avoid the possibility of rounding */
/*           causing overflow when computing  1/sfmin. */

	    sfmin = smnum * (eps + 1.);
	}
    }

    if (bli_lsame(cmach, "E", (ftnlen)1, (ftnlen)1)) {
	rmach = eps;
    } else if (bli_lsame(cmach, "S", (ftnlen)1, (ftnlen)1)) {
	rmach = sfmin;
    } else if (bli_lsame(cmach, "B", (ftnlen)1, (ftnlen)1)) {
	rmach = base;
    } else if (bli_lsame(cmach, "P", (ftnlen)1, (ftnlen)1)) {
	rmach = prec;
    } else if (bli_lsame(cmach, "N", (ftnlen)1, (ftnlen)1)) {
	rmach = t;
    } else if (bli_lsame(cmach, "R", (ftnlen)1, (ftnlen)1)) {
	rmach = rnd;
    } else if (bli_lsame(cmach, "M", (ftnlen)1, (ftnlen)1)) {
	rmach = emin;
    } else if (bli_lsame(cmach, "U", (ftnlen)1, (ftnlen)1)) {
	rmach = rmin;
    } else if (bli_lsame(cmach, "L", (ftnlen)1, (ftnlen)1)) {
	rmach = emax;
    } else if (bli_lsame(cmach, "O", (ftnlen)1, (ftnlen)1)) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* bli_dlamch_ */


/* *********************************************************************** */

/* Subroutine */ int bli_dlamc1(bla_integer *beta, bla_integer *t, bla_logical *rnd, bla_logical
	*ieee1)
{
    /* Initialized data */

    static bla_logical first = TRUE_;

    /* System generated locals */
    bla_double d__1, d__2;

    /* Local variables */
    static bla_logical lrnd;
    static bla_double a, b, c__, f;
    static bla_integer lbeta;
    static bla_double savec;
    extern bla_double bli_dlamc3(bla_double *, bla_double *);
    static bla_logical lieee1;
    static bla_double t1, t2;
    static bla_integer lt;
    static bla_double one, qtr;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC1 determines the machine parameters given by BETA, T, RND, and */
/*  IEEE1. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  IEEE1   (output) LOGICAL */
/*          Specifies whether rounding appears to be done in the IEEE */
/*          'round to nearest' style. */

/*  Further Details */
/*  =============== */

/*  The routine is based on the routine  ENVRON  by Malcolm and */
/*  incorporates suggestions by Gentleman and Marovich. See */

/*     Malcolm M. A. (1972) Algorithms to reveal properties of */
/*        floating-point arithmetic. Comms. of the ACM, 15, 949-951. */

/*     Gentleman W. M. and Marovich S. B. (1974) More on algorithms */
/*        that reveal properties of floating point arithmetic units. */
/*        Comms. of the ACM, 17, 276-277. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	one = 1.;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA, */
/*        IEEE1, T and RND. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are  stored and not held in registers,  or */
/*        are not affected by optimizers. */

/*        Compute  a = 2.0**m  with the  smallest positive bla_integer m such */
/*        that */

/*           fl( a + 1.0 ) = a. */

	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = bli_dlamc3(&a, &one);
	    d__1 = -a;
	    c__ = bli_dlamc3(&c__, &d__1);
	    goto L10;
	}
/* +       END WHILE */

/*        Now compute  b = 2.0**m  with the smallest positive bla_integer m */
/*        such that */

/*           fl( a + b ) .gt. a. */

	b = 1.;
	c__ = bli_dlamc3(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = bli_dlamc3(&a, &b);
	    goto L20;
	}
/* +       END WHILE */

/*        Now compute the base.  a and c  are neighbouring floating point */
/*        numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so */
/*        their difference is beta. Adding 0.25 to c is to ensure that it */
/*        is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c__;
	d__1 = -a;
	c__ = bli_dlamc3(&c__, &d__1);
	lbeta = (bla_integer) (c__ + qtr);


/*        Now determine whether rounding or chopping occurs,  by adding a */
/*        bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a. */

	b = (bla_double) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = bli_dlamc3(&d__1, &d__2);
	c__ = bli_dlamc3(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = bli_dlamc3(&d__1, &d__2);
	c__ = bli_dlamc3(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round to */
/*        nearest' style. B/2 is half a unit in the last place of the two */
/*        numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit */
/*        zero, and SAVEC is odd. Thus adding B/2 to A should not  change */
/*        A, but adding B/2 to SAVEC should change SAVEC. */

	d__1 = b / 2;
	t1 = bli_dlamc3(&d__1, &a);
	d__1 = b / 2;
	t2 = bli_dlamc3(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  bla_integer part of */
/*        log to the base beta of a,  however it is safer to determine  t */
/*        by powering.  So we find t as the smallest positive bla_integer for */
/*        which */

/*           fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = bli_dlamc3(&a, &one);
	    d__1 = -a;
	    c__ = bli_dlamc3(&c__, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;

    return 0;

/*     End of DLAMC1 */

} /* bli_dlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int bli_dlamc2(bla_integer *beta, bla_integer *t, bla_logical *rnd,
	bla_double *eps, bla_integer *emin, bla_double *rmin, bla_integer *emax,
	bla_double *rmax)
{
    /* Initialized data */

    static bla_logical first = TRUE_;
    static bla_logical iwarn = FALSE_;

    /* Format strings */
    static bla_character fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre\
ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the value EMIN loo\
ks\002,\002 acceptable please comment out \002,/\002 the IF block as marked \
within the code of routine\002,\002 DLAMC2,\002,/\002 otherwise supply EMIN \
explicitly.\002,/)";

    /* System generated locals */
    bla_integer i__1;
    bla_double d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    double bli_pow_di(bla_double *, bla_integer *);
    //bla_integer s_wsfe(cilist *), do_fio(bla_integer *, bla_character *, ftnlen), e_wsfe();

    /* Local variables */
    static bla_logical ieee;
    static bla_double half;
    static bla_logical lrnd;
    static bla_double leps, zero, a, b, c__;
    static bla_integer i__, lbeta;
    static bla_double rbase;
    static bla_integer lemin, lemax, gnmin;
    static bla_double smnum;
    static bla_integer gpmin;
    static bla_double third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int bli_dlamc1(bla_integer *, bla_integer *, bla_logical *,
	    bla_logical *);
    extern bla_double bli_dlamc3(bla_double *, bla_double *);
    static bla_logical lieee1;
    extern /* Subroutine */ int bli_dlamc4(bla_integer *, bla_double *, bla_integer *),
	    bli_dlamc5(bla_integer *, bla_integer *, bla_integer *, bla_logical *, bla_integer *,
	    bla_double *);
    static bla_integer lt, ngnmin, ngpmin;
    static bla_double one, two;

    /* Fortran I/O blocks */
    //static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC2 determines the machine parameters specified in its argument */
/*  list. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  EPS     (output) DOUBLE PRECISION */
/*          The smallest positive number such that */

/*             fl( 1.0 - EPS ) .LT. 1.0, */

/*          where fl denotes the computed value. */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow occurs. */

/*  RMIN    (output) DOUBLE PRECISION */
/*          The smallest normalized number for the machine, given by */
/*          BASE**( EMIN - 1 ), where  BASE  is the floating point value */
/*          of BETA. */

/*  EMAX    (output) INTEGER */
/*          The maximum exponent before overflow occurs. */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest positive number for the machine, given by */
/*          BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point */
/*          value of BETA. */

/*  Further Details */
/*  =============== */

/*  The computation of  EPS  is based on a routine PARANOIA by */
/*  W. Kahan of the University of California at Berkeley. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	zero = 0.;
	one = 1.;
	two = 2.;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of */
/*        BETA, T, RND, EPS, EMIN and RMIN. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are stored  and not held in registers,  or */
/*        are not affected by optimizers. */

/*        DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. */

	bli_dlamc1(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (bla_double) lbeta;
	i__1 = -lt;
	a = bli_pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = bli_dlamc3(&b, &d__1);
	third = bli_dlamc3(&sixth, &sixth);
	d__1 = -half;
	b = bli_dlamc3(&third, &d__1);
	b = bli_dlamc3(&b, &sixth);
	b = f2c_abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c__ = bli_dlamc3(&d__1, &d__2);
	    d__1 = -c__;
	    c__ = bli_dlamc3(&half, &d__1);
	    b = bli_dlamc3(&half, &c__);
	    d__1 = -b;
	    c__ = bli_dlamc3(&half, &d__1);
	    b = bli_dlamc3(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete. */

/*        Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)). */
/*        Keep dividing  A by BETA until (gradual) underflow occurs. This */
/*        is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	smnum = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    d__1 = smnum * rbase;
	    smnum = bli_dlamc3(&d__1, &zero);
/* L20: */
	}
	a = bli_dlamc3(&one, &smnum);
	bli_dlamc4(&ngpmin, &one, &lbeta);
	d__1 = -one;
	bli_dlamc4(&ngnmin, &d__1, &lbeta);
	bli_dlamc4(&gpmin, &a, &lbeta);
	d__1 = -a;
	bli_dlamc4(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual underflow; */
/*              e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual underflow; */
/*              e.g., IEEE standard followers ) */
	    } else {
		lemin = f2c_min(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, f2c_abs(i__1)) == 1) {
		lemin = f2c_max(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow; */
/*              e.g., CYBER 205 ) */
	    } else {
		lemin = f2c_min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, f2c_abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - f2c_min(ngpmin,ngnmin) == 3) {
		lemin = f2c_max(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflow; */
/*              no known machine ) */
	    } else {
		lemin = f2c_min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = f2c_min(ngpmin,ngnmin), i__1 = f2c_min(i__1,gpmin);
	    lemin = f2c_min(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/* ** */
/* Comment out this if block if EMIN is ok */

	if (iwarn) {
	    first = TRUE_;
/*
	    s_wsfe(&io___58);
	    do_fio(&c__1, (bla_character *)&lemin, (ftnlen)sizeof(bla_integer));
	    e_wsfe();
*/
	    printf( "%s", fmt_9999 );
	}

/* ** */

/*        Assume IEEE arithmetic if we found denormalised  numbers above, */
/*        or if arithmetic seems to round in the  IEEE style,  determined */
/*        in routine DLAMC1. A true IEEE machine should have both  things */
/*        true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could compute */
/*        RMIN as BASE**( EMIN - 1 ),  but some machines underflow during */
/*        this computation. */

	lrmin = 1.;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__1 = lrmin * rbase;
	    lrmin = bli_dlamc3(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	bli_dlamc5(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* bli_dlamc2_ */


/* *********************************************************************** */

bla_double bli_dlamc3(bla_double *a, bla_double *b)
{
    /* System generated locals */
    bla_double ret_val;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing */
/*  the addition of  A  and  B ,  for use in situations where optimizers */
/*  might hold one of these in a register. */

/*  Arguments */
/*  ========= */

/*  A       (input) DOUBLE PRECISION */
/*  B       (input) DOUBLE PRECISION */
/*          The values A and B. */

/* ===================================================================== */

/*     .. Executable Statements .. */

    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* bli_dlamc3_ */


/* *********************************************************************** */

/* Subroutine */ int bli_dlamc4(bla_integer *emin, bla_double *start, bla_integer *base)
{
    /* System generated locals */
    bla_integer i__1;
    bla_double d__1;

    /* Local variables */
    static bla_double zero, a;
    static bla_integer i__;
    static bla_double rbase, b1, b2, c1, c2, d1, d2;
    extern bla_double bli_dlamc3(bla_double *, bla_double *);
    static bla_double one;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC4 is a service routine for DLAMC2. */

/*  Arguments */
/*  ========= */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow, computed by */
/*          setting A = START and dividing by BASE until the previous A */
/*          can not be recovered. */

/*  START   (input) DOUBLE PRECISION */
/*          The starting point for determining EMIN. */

/*  BASE    (input) INTEGER */
/*          The base of the machine. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    a = *start;
    one = 1.;
    rbase = one / *base;
    zero = 0.;
    *emin = 1;
    d__1 = a * rbase;
    b1 = bli_dlamc3(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND. */
/*    $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = bli_dlamc3(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = bli_dlamc3(&d__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = bli_dlamc3(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = bli_dlamc3(&d__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* bli_dlamc4_ */


/* *********************************************************************** */

/* Subroutine */ int bli_dlamc5(bla_integer *beta, bla_integer *p, bla_integer *emin,
	bla_logical *ieee, bla_integer *emax, bla_double *rmax)
{
    /* System generated locals */
    bla_integer i__1;
    bla_double d__1;

    /* Local variables */
    static bla_integer lexp;
    static bla_double oldy;
    static bla_integer uexp, i__;
    static bla_double y, z__;
    static bla_integer nbits;
    extern bla_double bli_dlamc3(bla_double *, bla_double *);
    static bla_double recbas;
    static bla_integer exbits, expsum, try__;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC5 attempts to compute RMAX, the largest machine floating-point */
/*  number, without overflow.  It assumes that EMAX + f2c_abs(EMIN) sum */
/*  approximately to a power of 2.  It will fail on machines where this */
/*  assumption does not hold, for example, the Cyber 205 (EMIN = -28625, */
/*  EMAX = 28718).  It will also fail if the value supplied for EMIN is */
/*  too large (i.e. too close to zero), probably with overflow. */

/*  Arguments */
/*  ========= */

/*  BETA    (input) INTEGER */
/*          The base of floating-point arithmetic. */

/*  P       (input) INTEGER */
/*          The number of base BETA digits in the mantissa of a */
/*          floating-point value. */

/*  EMIN    (input) INTEGER */
/*          The minimum exponent before (gradual) underflow. */

/*  IEEE    (input) LOGICAL */
/*          A bla_logical flag specifying whether or not the arithmetic */
/*          system is thought to comply with the IEEE standard. */

/*  EMAX    (output) INTEGER */
/*          The largest exponent before overflow */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest machine floating-point number. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     First compute LEXP and UEXP, two powers of 2 that bound */
/*     f2c_abs(EMIN). We then assume that EMAX + f2c_abs(EMIN) will sum */
/*     approximately to the bound that is closest to f2c_abs(EMIN). */
/*     (EMAX is the exponent of the required number RMAX). */

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater */
/*     than or equal to EMIN. EXBITS is the number of bits needed to */
/*     store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to */
/*     EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a */
/*     floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a */
/*        floating-point number, which is unlikely, or some bits are */
/*        not used in the representation of numbers, which is possible, */
/*        (e.g. Cray machines) or the mantissa has an implicit bit, */
/*        (e.g. IEEE machines, Dec Vax machines), which is perhaps the */
/*        most likely. We have to assume the last alternative. */
/*        If this is true, then we need to reduce EMAX by one because */
/*        there must be some way of representing zero in an implicit-bit */
/*        system. On machines like Cray, we are reducing EMAX by one */
/*        unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent */
/*        for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should */
/*     be equal to (1.0 - BETA**(-P)) * BETA**EMAX . */

/*     First compute 1.0 - BETA**(-P), being careful that the */
/*     result is less than 1.0 . */

    recbas = 1. / *beta;
    z__ = *beta - 1.;
    y = 0.;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.) {
	    oldy = y;
	}
	y = bli_dlamc3(&y, &z__);
/* L20: */
    }
    if (y >= 1.) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = y * *beta;
	y = bli_dlamc3(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* bli_dlamc5_ */

#else

bla_double bli_dlamch(bla_character *cmach, ftnlen cmach_len)
{
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

	double safe_min = DBL_MIN;
	double small = 1.0f / DBL_MAX;

	if ( small >= safe_min )
		safe_min = small * ( 1.0 + DBL_EPSILON );

	switch ( toupper( *cmach ) )
	{
		case 'E': return DBL_EPSILON;
		case 'S': return safe_min;
		case 'B': return FLT_RADIX;
		case 'P': return FLT_RADIX*DBL_EPSILON;
		case 'N': return DBL_MANT_DIG;
		case 'R': return FLT_ROUNDS == FE_TONEAREST ? 1.0 : 0.0;
		case 'M': return DBL_MIN_EXP;
		case 'U': return DBL_MIN;
		case 'L': return DBL_MAX_EXP;
		case 'O': return DBL_MAX;
	}
	
	return 0.0;
}

#endif

#ifdef __cplusplus
}
#endif
