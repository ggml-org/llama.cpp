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

void g_char(const char *a, ftnlen alen, char *b)
{
	const char *x = a + alen;
	char *y = b + alen;

	for(;; y--) {
		if (x <= a) {
			*b = 0;
			return;
			}
		if (*--x != ' ')
			break;
		}
	*y-- = 0;
	do *y-- = *x;
		while(x-- > a);
}

void b_char(const char *a, char *b, ftnlen blen)
{	int i;
	for(i=0;i<blen && *a!=0;i++) *b++= *a++;
	for(;i<blen;i++) *b++=' ';
}

#ifndef NON_UNIX_STDIO
long f__inode(char *a, int *dev)
{	struct stat x;
	if(stat(a,&x)<0) return(-1);
	*dev = x.st_dev;
	return(x.st_ino);
}
#endif
