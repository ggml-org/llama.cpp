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
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "f2c.h"
#include "fio.h"
#include "arith.h"

#include "fmt.h"
#include "fp.h"

int wrt_E(ufloat *p, int w, int d, int e, ftnlen len)
{
	char buf[FMAX+EXPMAXDIGS+4], *s, *se;
	int d1, delta, e1, i, sign, signspace;
	double dd;
#ifdef WANT_LEAD_0
	int insert0 = 0;
#endif
	int e0 = e;

	if(e <= 0)
		e = 2;
	if(f__scale) {
		if(f__scale >= d + 2 || f__scale <= -d)
			goto nogood;
		}
	if(f__scale <= 0)
		--d;
	if (len == sizeof(real))
		dd = p->pf;
	else
		dd = p->pd;
	if (dd < 0.) {
		signspace = sign = 1;
		dd = -dd;
		}
	else {
		sign = 0;
		signspace = (int)f__cplus;
		if (!dd) {
#ifdef SIGNED_ZEROS
			if (signbit(dd))
				signspace = sign = 1;
#endif
			dd = 0.;	/* avoid -0 */
		}
	}
	delta = w - (2 /* for the . and the d adjustment above */
			+ 2 /* for the E+ */ + signspace + d + e);
#ifdef WANT_LEAD_0
	if (f__scale <= 0 && delta > 0) {
		delta--;
		insert0 = 1;
		}
	else
#endif
	if (delta < 0) {
nogood:
		while(--w >= 0)
			PUT('*');
		return(0);
		}
	if (f__scale < 0)
		d += f__scale;
	if (d > FMAX) {
		d1 = d - FMAX;
		d = FMAX;
		}
	else
		d1 = 0;
	sprintf(buf,"%#.*E", d, dd);
	/* check for NaN, Infinity */
	if (!isdigit(buf[0])) {
		switch(buf[0]) {
			case 'n':
			case 'N':
				signspace = 0;	/* no sign for NaNs */
		}
		delta = w - strlen(buf) - signspace;
		if (delta < 0)
			goto nogood;
		while(--delta >= 0)
			PUT(' ');
		if (signspace)
			PUT(sign ? '-' : '+');
		for(s = buf; *s; s++)
			PUT(*s);
		return 0;
	}
	se = buf + d + 3;
#ifdef GOOD_SPRINTF_EXPONENT /* When possible, exponent has 2 digits. */
	if (f__scale != 1 && dd)
		sprintf(se, "%+.2d", atoi(se) + 1 - f__scale);
#else
	if (dd)
		sprintf(se, "%+.2d", atoi(se) + 1 - f__scale);
	else
		strcpy(se, "+00");
#endif
	s = ++se;
	if (e < 2) {
		if (*s != '0')
			goto nogood;
		}
	/* accommodate 3 significant digits in exponent */
	if (s[2]) {
#ifdef Pedantic
		if (!e0 && !s[3])
			for(s -= 2, e1 = 2; s[0] = s[1]; s++);

	/* Pedantic gives the behavior that Fortran 77 specifies,	*/
	/* i.e., requires that E be specified for exponent fields	*/
	/* of more than 3 digits.  With Pedantic undefined, we get	*/
	/* the behavior that Cray displays -- you get a bigger		*/
	/* exponent field if it fits.	*/
#else
		if (!e0) {
			for(s -= 2, e1 = 2; s[0] = s[1]; s++)
#ifdef CRAY
				delta--;
			if ((delta += 4) < 0)
				goto nogood
#endif
				;
			}
#endif
		else if (e0 >= 0)
			goto shift;
		else
			e1 = e;
		}
	else
 shift:
		for(s += 2, e1 = 2; *s; ++e1, ++s)
			if (e1 >= e)
				goto nogood;
	while(--delta >= 0)
		PUT(' ');
	if (signspace)
		PUT(sign ? '-' : '+');
	s = buf;
	i = f__scale;
	if (f__scale <= 0) {
#ifdef WANT_LEAD_0
		if (insert0)
			PUT('0');
#endif
		PUT('.');
		for(; i < 0; ++i)
			PUT('0');
		PUT(*s);
		s += 2;
		}
	else if (f__scale > 1) {
		PUT(*s);
		s += 2;
		while(--i > 0)
			PUT(*s++);
		PUT('.');
		}
	if (d1) {
		se -= 2;
		while(s < se) PUT(*s++);
		se += 2;
		do PUT('0'); while(--d1 > 0);
		}
	while(s < se)
		PUT(*s++);
	if (e < 2)
		PUT(s[1]);
	else {
		while(++e1 <= e)
			PUT('0');
		while(*s)
			PUT(*s++);
		}
	return 0;
}

int wrt_F(ufloat *p, int w, int d, ftnlen len)
{
	int d1, sign, n;
	double x;
	char *b, buf[MAXINTDIGS+MAXFRACDIGS+4], *s;

	x= (len==sizeof(real)?p->pf:p->pd);
	if (d < MAXFRACDIGS)
		d1 = 0;
	else {
		d1 = d - MAXFRACDIGS;
		d = MAXFRACDIGS;
		}
	if (x < 0.)
		{ x = -x; sign = 1; }
	else {
		sign = 0;
		if (!x) {
#ifdef SIGNED_ZEROS
			if (signbit(x))
				sign = 2;
#endif
			x = 0.;
			}
	}

	if (n = f__scale)
		if (n > 0)
			do x *= 10.; while(--n > 0);
		else
			do x *= 0.1; while(++n < 0);

	n = sprintf(b = buf, "%#.*f", d, x) + d1;

#ifndef WANT_LEAD_0
	if (buf[0] == '0' && d)
		{ ++b; --n; }
#endif
	if (sign == 1) {
		/* check for all zeros */
		for(s = b;;) {
			while(*s == '0') s++;
			switch(*s) {
				case '.':
					s++; continue;
				case 0:
					sign = 0;
				}
			break;
			}
		}
	if (sign || f__cplus)
		++n;
	if (n > w) {
#ifdef WANT_LEAD_0
		if (buf[0] == '0' && --n == w)
			++b;
		else
#endif
		{
			while(--w >= 0)
				PUT('*');
			return 0;
			}
		}
	for(w -= n; --w >= 0; )
		PUT(' ');
	if (sign)
		PUT('-');
	else if (f__cplus)
		PUT('+');
	while(n = *b++)
		PUT(n);
	while(--d1 >= 0)
		PUT('0');
	return 0;
}
