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

//#include <config.h>
#include "f2c.h"
#include "fio.h"

integer f_rew(alist *a)
{
	unit *b;
	if(a->aunit>=MXUNIT || a->aunit<0)
		err(a->aerr,101,"rewind");
	b = &f__units[a->aunit];
	if(b->ufd == NULL || b->uwrt == 3)
		return(0);
	if(!b->useek)
		err(a->aerr,106,"rewind")
	if(b->uwrt) {
		(void) t_runc(a);
		b->uwrt = 3;
		}
	rewind(b->ufd);
	b->uend=0;
	return 0;
}
