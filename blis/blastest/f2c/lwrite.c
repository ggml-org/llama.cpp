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
#include "f2c.h"
#include "fio.h"
#include "fmt.h"
#include "lio.h"
#include "arith.h"

ftnint L_len;
int f__Aquote;

static void
donewrec(void)
{
	if (f__recpos)
		(*f__donewrec)();
}

static void lwrt_I(longint n)
{
	char *p;
	int ndigit, sign;

	p = f__icvt(n, &ndigit, &sign, 10);
	if(f__recpos + ndigit >= L_len)
		donewrec();
	PUT(' ');
	if (sign)
		PUT('-');
	while(*p)
		PUT(*p++);
}

static void lwrt_L(ftnint n, ftnlen len)
{
	if(f__recpos+LLOGW>=L_len)
		donewrec();
	wrt_L((Uint *)&n,LLOGW, len);
}

static void lwrt_A(char *p, ftnlen len)
{
	int a;
	char *p1, *pe;

	a = 0;
	pe = p + len;
	if (f__Aquote) {
		a = 3;
		if (len > 1 && p[len-1] == ' ') {
			while(--len > 1 && p[len-1] == ' ');
			pe = p + len;
			}
		p1 = p;
		while(p1 < pe)
			if (*p1++ == '\'')
				a++;
		}
	if(f__recpos+len+a >= L_len)
		donewrec();
	if (a
#ifndef OMIT_BLANK_CC
		|| !f__recpos
#endif
		)
		PUT(' ');
	if (a) {
		PUT('\'');
		while(p < pe) {
			if (*p == '\'')
				PUT('\'');
			PUT(*p++);
			}
		PUT('\'');
		}
	else
		while(p < pe)
			PUT(*p++);
}

static int l_g(char *buf, double n)
{
	register char *b, c, c1;

	b = buf;
	*b++ = ' ';
	if (n < 0) {
		*b++ = '-';
		n = -n;
		}
	else
		*b++ = ' ';
	if (n == 0) {
#ifdef SIGNED_ZEROS
		if (signbit(n))
			*b++ = '-';
#endif
		*b++ = '0';
		*b++ = '.';
		*b = 0;
		goto f__ret;
		}
	sprintf(b, LGFMT, n);
	switch(*b) {
#ifndef WANT_LEAD_0
		case '0':
			while(b[0] = b[1])
				b++;
			break;
#endif
		case 'i':
		case 'I':
			/* Infinity */
		case 'n':
		case 'N':
			/* NaN */
			while(*++b);
			break;

		default:
	/* Fortran 77 insists on having a decimal point... */
		    for(;; b++)
			switch(*b) {
			case 0:
				*b++ = '.';
				*b = 0;
				goto f__ret;
			case '.':
				while(*++b);
				goto f__ret;
			case 'E':
				for(c1 = '.', c = 'E';  *b = c1;
					c1 = c, c = *++b);
				goto f__ret;
			}
		}
 f__ret:
	return b - buf;
}

static void l_put(register char *s)
{
#ifdef KR_headers
	register void (*pn)() = f__putn;
#else
	register void (*pn)(int) = f__putn;
#endif
	register int c;

	while(c = *s++)
		(*pn)(c);
}

static void lwrt_F(double n)
{
	char buf[LEFBL];

	if(f__recpos + l_g(buf,n) >= L_len)
		donewrec();
	l_put(buf);
}

static void lwrt_C(double a, double b)
{
	char *ba, *bb, bufa[LEFBL], bufb[LEFBL];
	int al, bl;

	al = l_g(bufa, a);
	for(ba = bufa; *ba == ' '; ba++)
		--al;
	bl = l_g(bufb, b) + 1;	/* intentionally high by 1 */
	for(bb = bufb; *bb == ' '; bb++)
		--bl;
	if(f__recpos + al + bl + 3 >= L_len)
		donewrec();
#ifdef OMIT_BLANK_CC
	else
#endif
	PUT(' ');
	PUT('(');
	l_put(ba);
	PUT(',');
	if (f__recpos + bl >= L_len) {
		(*f__donewrec)();
#ifndef OMIT_BLANK_CC
		PUT(' ');
#endif
		}
	l_put(bb);
	PUT(')');
}

int l_write(ftnint *number, char *ptr, ftnlen len, ftnint type)
{
#define Ptr ((flex *)ptr)
	int i;
	longint x;
	double y,z;
	real *xx;
	doublereal *yy;
	for(i=0;i< *number; i++)
	{
		switch((int)type)
		{
		default: f__fatal(117,"unknown type in lio");
		case TYINT1:
			x = Ptr->flchar;
			goto xint;
		case TYSHORT:
			x=Ptr->flshort;
			goto xint;
#ifdef Allow_TYQUAD
		case TYQUAD:
			x = Ptr->fllongint;
			goto xint;
#endif
		case TYLONG:
			x=Ptr->flint;
		xint:	lwrt_I(x);
			break;
		case TYREAL:
			y=Ptr->flreal;
			goto xfloat;
		case TYDREAL:
			y=Ptr->fldouble;
		xfloat: lwrt_F(y);
			break;
		case TYCOMPLEX:
			xx= &Ptr->flreal;
			y = *xx++;
			z = *xx;
			goto xcomplex;
		case TYDCOMPLEX:
			yy = &Ptr->fldouble;
			y= *yy++;
			z = *yy;
		xcomplex:
			lwrt_C(y,z);
			break;
		case TYLOGICAL1:
			x = Ptr->flchar;
			goto xlog;
		case TYLOGICAL2:
			x = Ptr->flshort;
			goto xlog;
		case TYLOGICAL:
			x = Ptr->flint;
		xlog:	lwrt_L(Ptr->flint, len);
			break;
		case TYCHAR:
			lwrt_A(ptr,len);
			break;
		}
		ptr += len;
	}
	return(0);
}
