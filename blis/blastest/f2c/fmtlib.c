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

#include <f2c_config.h>
/*	@(#)fmtlib.c	1.2	*/
#define MAXINTLENGTH 23

#include "f2c.h"
#ifndef Allow_TYQUAD
#undef longint
#define longint long
#undef ulongint
#define ulongint unsigned long
#endif

#ifdef INTEGER_STAR_8
char *f__icvt(longint value, int *ndigit, int *sign, int base)
#else
char *f__icvt(integer value, int *ndigit, int *sign, int base)
#endif
{
	static char buf[MAXINTLENGTH+1];
	register int i;
	ulongint uvalue;

	if(value > 0) {
		uvalue = value;
		*sign = 0;
		}
	else if (value < 0) {
		uvalue = -value;
		*sign = 1;
		}
	else {
		*sign = 0;
		*ndigit = 1;
		buf[MAXINTLENGTH-1] = '0';
		return &buf[MAXINTLENGTH-1];
		}
	i = MAXINTLENGTH;
	do {
		buf[--i] = (uvalue%base) + '0';
		uvalue /= base;
		}
		while(uvalue > 0);
	*ndigit = MAXINTLENGTH - i;
	return &buf[i];
}
