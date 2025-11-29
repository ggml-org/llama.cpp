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

/* sequential formatted external common routines*/
#include <f2c_config.h>
#include "f2c.h"
#include "fio.h"

integer e_rsfe(void)
{	int n;
	n=en_fio();
	f__fmtbuf=NULL;
	return(n);
}

int c_sfe(cilist *a)
{	unit *p;
	f__curunit = p = &f__units[a->ciunit];
	if(a->ciunit >= MXUNIT || a->ciunit<0)
		err(a->cierr,101,"startio");
	if(p->ufd==NULL && fk_open(SEQ,FMT,a->ciunit)) err(a->cierr,114,"sfe")
	if(!p->ufmt) err(a->cierr,102,"sfe")
	return(0);
}

integer e_wsfe(void)
{
	int n = en_fio();
	f__fmtbuf = NULL;
#ifdef ALWAYS_FLUSH
	if (!n && fflush(f__cf))
		err(f__elist->cierr, errno, "write end");
#endif
	return n;
}

