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

/* read sequential formatted external */
#include <f2c_config.h>
#include "f2c.h"
#include "fio.h"
#include "fmt.h"

int
xrd_SL(void)
{	int ch;
	if(!f__curunit->uend)
		while((ch=getc(f__cf))!='\n')
			if (ch == EOF) {
				f__curunit->uend = 1;
				break;
				}
	f__cursor=f__recpos=0;
	return 1;
}

int
x_getc(void)
{	int ch;
	if(f__curunit->uend) return EOF;
	ch = getc(f__cf);
	if(ch!=EOF && ch!='\n')
	{	f__recpos++;
		return ch;
	}
	if(ch=='\n')
	{	(void) ungetc(ch,f__cf);
		return ch;
	}
	if(f__curunit->uend || feof(f__cf))
	{	errno=0;
		f__curunit->uend=1;
		return -1;
	}
	return -1;
}

int
x_endp(void)
{
	xrd_SL();
	return f__curunit->uend == 1 ? EOF : 0;
}

int
x_rev(void)
{
	(void) xrd_SL();
	return 0;
}

integer s_rsfe(cilist *a) /* start */
{	int n;
	if(!f__init) f_init();
	f__reading=1;
	f__sequential=1;
	f__formatted=1;
	f__external=1;
	if(n=c_sfe(a)) return n;
	f__elist=a;
	f__cursor=f__recpos=0;
	f__scale=0;
	f__fmtbuf=a->cifmt;
	f__cf=f__curunit->ufd;
	if(pars_f(f__fmtbuf)<0) err(a->cierr,100,"startio");
	f__getn= x_getc;
	f__doed= rd_ed;
	f__doned= rd_ned;
	fmt_bg();
	f__doend=x_endp;
	f__donewrec=xrd_SL;
	f__dorevert=x_rev;
	f__cblank=f__curunit->ublnk;
	f__cplus=0;
	if(f__curunit->uwrt && f__nowreading(f__curunit))
		err(a->cierr,errno,"read start");
	if(f__curunit->uend)
		err(f__elist->ciend,(EOF),"read start");
	return 0;
}
