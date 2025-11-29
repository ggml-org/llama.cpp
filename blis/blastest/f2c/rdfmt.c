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
#include <ctype.h>
#include "f2c.h"
#include "fio.h"

#include "fmt.h"
#include "fp.h"

static int rd_Z(Uint *n, int w, ftnlen len)
{
	long x[9];
	char *s, *s0, *s1, *se, *t;
	const char *sc;
	int ch, i, w1, w2;
	static char hex[256];
	static int one = 1;
	int bad = 0;

	if (!hex['0']) {
		sc = "0123456789";
		while(ch = *sc++)
			hex[ch] = ch - '0' + 1;
		sc = "ABCDEF";
		while(ch = *sc++)
			hex[ch] = hex[ch + 'a' - 'A'] = ch - 'A' + 11;
		}
	s = s0 = (char *)x;
	s1 = (char *)&x[4];
	se = (char *)&x[8];
	if (len > 4*sizeof(long))
		return errno = 117;
	while (w) {
		GET(ch);
		if (ch==',' || ch=='\n')
			break;
		w--;
		if (ch > ' ') {
			if (!hex[ch & 0xff])
				bad++;
			*s++ = ch;
			if (s == se) {
				/* discard excess characters */
				for(t = s0, s = s1; t < s1;)
					*t++ = *s++;
				s = s1;
				}
			}
		}
	if (bad)
		return errno = 115;
	w = (int)len;
	w1 = s - s0;
	w2 = w1+1 >> 1;
	t = (char *)n;
	if (*(char *)&one) {
		/* little endian */
		t += w - 1;
		i = -1;
		}
	else
		i = 1;
	for(; w > w2; t += i, --w)
		*t = 0;
	if (!w)
		return 0;
	if (w < w2)
		s0 = s - (w << 1);
	else if (w1 & 1) {
		*t = hex[*s0++ & 0xff] - 1;
		if (!--w)
			return 0;
		t += i;
		}
	do {
		*t = hex[*s0 & 0xff]-1 << 4 | hex[s0[1] & 0xff]-1;
		t += i;
		s0 += 2;
		}
		while(--w);
	return 0;
}

static int rd_I(Uint *n, int w, ftnlen len, register int base)
{
	int ch, sign;
	longint x = 0;

	if (w <= 0)
		goto have_x;
	for(;;) {
		GET(ch);
		if (ch != ' ')
			break;
		if (!--w)
			goto have_x;
		}
	sign = 0;
	switch(ch) {
	  case ',':
	  case '\n':
		w = 0;
		goto have_x;
	  case '-':
		sign = 1;
	  case '+':
		break;
	  default:
		if (ch >= '0' && ch <= '9') {
			x = ch - '0';
			break;
			}
		goto have_x;
		}
	while(--w) {
		GET(ch);
		if (ch >= '0' && ch <= '9') {
			x = x*base + ch - '0';
			continue;
			}
		if (ch != ' ') {
			if (ch == '\n' || ch == ',')
				w = 0;
			break;
			}
		if (f__cblank)
			x *= base;
		}
	if (sign)
		x = -x;
 have_x:
	if(len == sizeof(integer))
		n->il=x;
	else if(len == sizeof(char))
		n->ic = (char)x;
#ifdef Allow_TYQUAD
	else if (len == sizeof(longint))
		n->ili = x;
#endif
	else
		n->is = (short)x;
	if (w) {
		while(--w)
			GET(ch);
		return errno = 115;
		}
	return 0;
}

static int rd_L(ftnint *n, int w, ftnlen len)
{	int ch, dot, lv;

	if (w <= 0)
		goto bad;
	for(;;) {
		GET(ch);
		--w;
		if (ch != ' ')
			break;
		if (!w)
			goto bad;
		}
	dot = 0;
 retry:
	switch(ch) {
	  case '.':
		if (dot++ || !w)
			goto bad;
		GET(ch);
		--w;
		goto retry;
	  case 't':
	  case 'T':
		lv = 1;
		break;
	  case 'f':
	  case 'F':
		lv = 0;
		break;
	  default:
 bad:
		for(; w > 0; --w)
			GET(ch);
		/* no break */
	  case ',':
	  case '\n':
		return errno = 116;
		}
	switch(len) {
		case sizeof(char):	*(char *)n = (char)lv;	 break;
		case sizeof(short):	*(short *)n = (short)lv; break;
		default:		*n = lv;
		}
	while(w-- > 0) {
		GET(ch);
		if (ch == ',' || ch == '\n')
			break;
		}
	return 0;
}

static int rd_F(ufloat *p, int w, int d, ftnlen len)
{
	char s[FMAX+EXPMAXDIGS+4];
	register int ch;
	register char *sp, *spe, *sp1;
	double x;
	int scale1, se;
	long e, exp;

	sp1 = sp = s;
	spe = sp + FMAX;
	exp = -d;
	x = 0.;

	do {
		GET(ch);
		w--;
		} while (ch == ' ' && w);
	switch(ch) {
		case '-': *sp++ = ch; sp1++; spe++;
		case '+':
			if (!w) goto zero;
			--w;
			GET(ch);
		}
	while(ch == ' ') {
blankdrop:
		if (!w--) goto zero; GET(ch); }
	while(ch == '0')
		{ if (!w--) goto zero; GET(ch); }
	if (ch == ' ' && f__cblank)
		goto blankdrop;
	scale1 = f__scale;
	while(isdigit(ch)) {
digloop1:
		if (sp < spe) *sp++ = ch;
		else ++exp;
digloop1e:
		if (!w--) goto done;
		GET(ch);
		}
	if (ch == ' ') {
		if (f__cblank)
			{ ch = '0'; goto digloop1; }
		goto digloop1e;
		}
	if (ch == '.') {
		exp += d;
		if (!w--) goto done;
		GET(ch);
		if (sp == sp1) { /* no digits yet */
			while(ch == '0') {
skip01:
				--exp;
skip0:
				if (!w--) goto done;
				GET(ch);
				}
			if (ch == ' ') {
				if (f__cblank) goto skip01;
				goto skip0;
				}
			}
		while(isdigit(ch)) {
digloop2:
			if (sp < spe)
				{ *sp++ = ch; --exp; }
digloop2e:
			if (!w--) goto done;
			GET(ch);
			}
		if (ch == ' ') {
			if (f__cblank)
				{ ch = '0'; goto digloop2; }
			goto digloop2e;
			}
		}
	switch(ch) {
	  default:
		break;
	  case '-': se = 1; goto signonly;
	  case '+': se = 0; goto signonly;
	  case 'e':
	  case 'E':
	  case 'd':
	  case 'D':
		if (!w--)
			goto bad;
		GET(ch);
		while(ch == ' ') {
			if (!w--)
				goto bad;
			GET(ch);
			}
		se = 0;
	  	switch(ch) {
		  case '-': se = 1;
		  case '+':
signonly:
			if (!w--)
				goto bad;
			GET(ch);
			}
		while(ch == ' ') {
			if (!w--)
				goto bad;
			GET(ch);
			}
		if (!isdigit(ch))
			goto bad;

		e = ch - '0';
		for(;;) {
			if (!w--)
				{ ch = '\n'; break; }
			GET(ch);
			if (!isdigit(ch)) {
				if (ch == ' ') {
					if (f__cblank)
						ch = '0';
					else continue;
					}
				else
					break;
				}
			e = 10*e + ch - '0';
			if (e > EXPMAX && sp > sp1)
				goto bad;
			}
		if (se)
			exp -= e;
		else
			exp += e;
		scale1 = 0;
		}
	switch(ch) {
	  case '\n':
	  case ',':
		break;
	  default:
bad:
		return (errno = 115);
		}
done:
	if (sp > sp1) {
		while(*--sp == '0')
			++exp;
		if (exp -= scale1)
			sprintf(sp+1, "e%ld", exp);
		else
			sp[1] = 0;
		x = atof(s);
		}
zero:
	if (len == sizeof(real))
		p->pf = x;
	else
		p->pd = x;
	return(0);
}

static int rd_A(char *p, ftnlen len)
{	int i,ch;
	for(i=0;i<len;i++)
	{	GET(ch);
		*p++=VAL(ch);
	}
	return(0);
}

static int rd_AW(char *p, int w, ftnlen len)
{	int i,ch;
	if(w>=len)
	{	for(i=0;i<w-len;i++)
			GET(ch);
		for(i=0;i<len;i++)
		{	GET(ch);
			*p++=VAL(ch);
		}
		return(0);
	}
	for(i=0;i<w;i++)
	{	GET(ch);
		*p++=VAL(ch);
	}
	for(i=0;i<len-w;i++) *p++=' ';
	return(0);
}

static int rd_H(int n, char *s)
{	int i,ch;
	for(i=0;i<n;i++)
		if((ch=(*f__getn)())<0) return(ch);
		else *s++ = ch=='\n'?' ':ch;
	return(1);
}

static int rd_POS(char *s)
{	char quote;
	int ch;
	quote= *s++;
	for(;*s;s++)
		if(*s==quote && *(s+1)!=quote) break;
		else if((ch=(*f__getn)())<0) return(ch);
		else *s = ch=='\n'?' ':ch;
	return(1);
}

int rd_ed(struct syl *p, char *ptr, ftnlen len)
{	int ch;
	for(;f__cursor>0;f__cursor--) if((ch=(*f__getn)())<0) return(ch);
	if(f__cursor<0)
	{	if(f__recpos+f__cursor < 0) /*err(elist->cierr,110,"fmt")*/
			f__cursor = -f__recpos;	/* is this in the standard? */
		if(f__external == 0) {
			extern char *f__icptr;
			f__icptr += f__cursor;
		}
		else if(f__curunit && f__curunit->useek)
			(void) FSEEK(f__cf, f__cursor,SEEK_CUR);
		else
			err(f__elist->cierr,106,"fmt");
		f__recpos += f__cursor;
		f__cursor=0;
	}
	switch(p->op)
	{
	default: fprintf(stderr,"rd_ed, unexpected code: %d\n", p->op);
		sig_die(f__fmtbuf, 1);
	case IM:
	case I: ch = rd_I((Uint *)ptr,p->p1,len, 10);
		break;

		/* O and OM don't work right for character, double, complex, */
		/* or doublecomplex, and they differ from Fortran 90 in */
		/* showing a minus sign for negative values. */

	case OM:
	case O: ch = rd_I((Uint *)ptr, p->p1, len, 8);
		break;
	case L: ch = rd_L((ftnint *)ptr,p->p1,len);
		break;
	case A:	ch = rd_A(ptr,len);
		break;
	case AW:
		ch = rd_AW(ptr,p->p1,len);
		break;
	case E: case EE:
	case D:
	case G:
	case GE:
	case F:	ch = rd_F((ufloat *)ptr,p->p1,p->p2.i[0],len);
		break;

		/* Z and ZM assume 8-bit bytes. */

	case ZM:
	case Z:
		ch = rd_Z((Uint *)ptr, p->p1, len);
		break;
	}
	if(ch == 0) return(ch);
	else if(ch == EOF) return(EOF);
	if (f__cf)
		clearerr(f__cf);
	return(errno);
}

int rd_ned(struct syl *p)
{
	switch(p->op)
	{
	default: fprintf(stderr,"rd_ned, unexpected code: %d\n", p->op);
		sig_die(f__fmtbuf, 1);
	case APOS:
		return(rd_POS(p->p2.s));
	case H:	return(rd_H(p->p1,p->p2.s));
	case SLASH: return((*f__donewrec)());
	case TR:
	case X:	f__cursor += p->p1;
		return(1);
	case T: f__cursor=p->p1-f__recpos - 1;
		return(1);
	case TL: f__cursor -= p->p1;
		if(f__cursor < -f__recpos)	/* TL1000, 1X */
			f__cursor = -f__recpos;
		return(1);
	}
}
