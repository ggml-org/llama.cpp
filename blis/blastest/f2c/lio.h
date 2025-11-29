/****************************************************************
Copyright 1990 - 1997 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/

/*	copy of ftypes from the compiler */
/* variable types
 * numeric assumptions:
 *	int < reals < complexes
 *	TYDREAL-TYREAL = TYDCOMPLEX-TYCOMPLEX
 */

/* 0-10 retain their old (pre LOGICAL*1, etc.) */
/* values to allow mixing old and new objects. */

#define TYUNKNOWN 0
#define TYADDR 1
#define TYSHORT 2
#define TYLONG 3
#define TYREAL 4
#define TYDREAL 5
#define TYCOMPLEX 6
#define TYDCOMPLEX 7
#define TYLOGICAL 8
#define TYCHAR 9
#define TYSUBR 10
#define TYINT1 11
#define TYLOGICAL1 12
#define TYLOGICAL2 13
#ifdef Allow_TYQUAD
#undef TYQUAD
#define TYQUAD 14
#endif

#define	LINTW	24
#define	LINE	80
#define	LLOGW	2
#define	LGFMT	"%.9G"
/* LEFBL 20 should suffice; 24 overcomes a NeXT bug. */
#define	LEFBL	24

typedef union
{
	char	flchar;
	short	flshort;
	ftnint	flint;
#ifdef Allow_TYQUAD
	longint fllongint;
#endif
	real	flreal;
	doublereal	fldouble;
} flex;
