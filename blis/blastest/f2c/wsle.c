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
#include <string.h>
#include "f2c.h"
#include "fio.h"
#include "fmt.h"
#include "lio.h"

integer s_wsle(cilist *a)
{
	int n;
	if(n=c_le(a)) return(n);
	f__reading=0;
	f__external=1;
	f__formatted=1;
	f__putn = x_putc;
	f__lioproc = l_write;
	L_len = LINE;
	f__donewrec = x_wSL;
	if(f__curunit->uwrt != 1 && f__nowwriting(f__curunit))
		err(a->cierr, errno, "list output start");
	return(0);
}

integer e_wsle(void)
{
	int n = f__putbuf('\n');
	f__recpos=0;
#ifdef ALWAYS_FLUSH
	if (!n && fflush(f__cf))
		err(f__elist->cierr, errno, "write end");
#endif
	return(n);
}
