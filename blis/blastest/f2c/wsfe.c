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

/*write sequential formatted external*/
#include <f2c_config.h>
#include "f2c.h"
#include "fio.h"
#include "fmt.h"

int x_wSL(void)
{
	int n = f__putbuf('\n');
	f__hiwater = f__recpos = f__cursor = 0;
	return(n == 0);
}

static int xw_end(void)
{
	int n;

	if(f__nonl) {
		f__putbuf(n = 0);
		fflush(f__cf);
		}
	else
		n = f__putbuf('\n');
	f__hiwater = f__recpos = f__cursor = 0;
	return n;
}

static int xw_rev(void)
{
	int n = 0;
	if(f__workdone) {
		n = f__putbuf('\n');
		f__workdone = 0;
		}
	f__hiwater = f__recpos = f__cursor = 0;
	return n;
}

integer s_wsfe(cilist *a)	/*start*/
{	int n;
	if(!f__init) f_init();
	f__reading=0;
	f__sequential=1;
	f__formatted=1;
	f__external=1;
	if(n=c_sfe(a)) return(n);
	f__elist=a;
	f__hiwater = f__cursor=f__recpos=0;
	f__nonl = 0;
	f__scale=0;
	f__fmtbuf=a->cifmt;
	f__cf=f__curunit->ufd;
	if(pars_f(f__fmtbuf)<0) err(a->cierr,100,"startio");
	f__putn= x_putc;
	f__doed= w_ed;
	f__doned= w_ned;
	f__doend=xw_end;
	f__dorevert=xw_rev;
	f__donewrec=x_wSL;
	fmt_bg();
	f__cplus=0;
	f__cblank=f__curunit->ublnk;
	if(f__curunit->uwrt != 1 && f__nowwriting(f__curunit))
		err(a->cierr,errno,"write start");
	return(0);
}
