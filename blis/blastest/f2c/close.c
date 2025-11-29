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

#undef abs
#undef min
#undef max
#include <stdlib.h>
#if defined(NON_UNIX_STDIO) || defined(_MSC_VER) || defined(__MINGW32__)
# include <stdio.h>
# define unlink remove
#else
# include <unistd.h>
#endif

integer f_clos(cllist *a)
{
	unit *b;

	if(a->cunit >= MXUNIT) return(0);
	b= &f__units[a->cunit];
	if(b->ufd==NULL)
		goto done;
	if (b->uscrtch == 1)
		goto Delete;
	if (!a->csta)
		goto Keep;
	switch(*a->csta) {
		default:
	 	Keep:
		case 'k':
		case 'K':
			if(b->uwrt == 1)
				t_runc((alist *)a);
			if(b->ufnm) {
				fclose(b->ufd);
				free(b->ufnm);
				}
			break;
		case 'd':
		case 'D':
		Delete:
			fclose(b->ufd);
			if(b->ufnm) {
				unlink(b->ufnm); /*SYSDEP*/
				free(b->ufnm);
				}
		}
	b->ufd=NULL;
 done:
	b->uend=0;
	b->ufnm=NULL;
	return(0);
}

void
f_exit(void)
{
	static int run = 0;
	int i;
	static cllist xx;
	/* Do not execute f_exit() twice */
	if (run)
		return;
	run = 1;
	if (!xx.cerr) {
		xx.cerr=1;
		xx.csta=NULL;
		for(i=0;i<MXUNIT;i++)
		{
			xx.cunit=i;
			(void) f_clos(&xx);
		}
	}
}

int flush_(void)
{	int i;
	for(i=0;i<MXUNIT;i++)
		if(f__units[i].ufd != NULL && f__units[i].uwrt)
			fflush(f__units[i].ufd);
return 0;
}
