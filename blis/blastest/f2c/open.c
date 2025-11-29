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
#ifndef NON_UNIX_STDIO
#include <unistd.h>
#endif
#ifdef _MSC_VER
#include <io.h>
#define access _access
#endif
#include "f2c.h"
#include "fio.h"

const char *f__r_mode[2] = {"rb", "r"};
const char *f__w_mode[4] = {"wb", "w", "r+b", "r+"};

static char f__buf0[400], *f__buf = f__buf0;
static int f__buflen = (int)sizeof(f__buf0);

static void f__bufadj(int n, int c)
{
	unsigned int len;
	char *nbuf, *s, *t, *te;

	if (f__buf == f__buf0)
		f__buflen = 1024;
	while(f__buflen <= n)
		f__buflen <<= 1;
	len = (unsigned int)f__buflen;
	if (len != f__buflen || !(nbuf = (char*)malloc(len))) {
		f__fatal(113, "malloc failure");
	}
	else {
		s = nbuf;
		t = f__buf;
		te = t + c;
		while (t < te)
			*s++ = *t++;
		if (f__buf != f__buf0)
			free(f__buf);
		f__buf = nbuf;
	}
}

int f__putbuf(int c)
{
	char *s, *se;
	int n;

	if (f__hiwater > f__recpos)
		f__recpos = f__hiwater;
	n = f__recpos + 1;
	if (n >= f__buflen)
		f__bufadj(n, f__recpos);
	s = f__buf;
	se = s + f__recpos;
	if (c)
		*se++ = c;
	*se = 0;
	for(;;) {
		fputs(s, f__cf);
		s += strlen(s);
		if (s >= se)
			break;	/* normally happens the first time */
		putc(*s++, f__cf);
		}
	return 0;
}

void x_putc(int c)
{
	if (f__recpos >= f__buflen)
		f__bufadj(f__recpos, f__buflen);
	f__buf[f__recpos++] = c;
}

#define opnerr(f,m,s) {if(f) errno= m; else opn_err(m,s,a); return(m);}

static void opn_err(int m, const char *s, olist *a)
{
	if (a->ofnm) {
		/* supply file name to error message */
		if (a->ofnmlen >= f__buflen)
			f__bufadj((int)a->ofnmlen, 0);
		g_char(a->ofnm, a->ofnmlen, f__curunit->ufnm = f__buf);
		}
	f__fatal(m, s);
}

integer f_open(olist *a)
{	unit *b;
	integer rv;
	char buf[256], *s;
	cllist x;
	int ufmt;
	FILE *tf;
#ifndef NON_UNIX_STDIO
	int n;
#endif
	f__external = 1;
	if(a->ounit>=MXUNIT || a->ounit<0)
		err(a->oerr,101,"open")
	if (!f__init)
		f_init();
	f__curunit = b = &f__units[a->ounit];
	if(b->ufd) {
		if(a->ofnm==0)
		{
		same:	if (a->oblnk)
				b->ublnk = *a->oblnk == 'z' || *a->oblnk == 'Z';
			return(0);
		}
#ifdef NON_UNIX_STDIO
		if (b->ufnm
		 && strlen(b->ufnm) == a->ofnmlen
		 && !strncmp(b->ufnm, a->ofnm, (unsigned)a->ofnmlen))
			goto same;
#else
		g_char(a->ofnm,a->ofnmlen,buf);
		if (f__inode(buf,&n) == b->uinode && n == b->udev)
			goto same;
#endif
		x.cunit=a->ounit;
		x.csta=0;
		x.cerr=a->oerr;
		if ((rv = f_clos(&x)) != 0)
			return rv;
		}
	b->url = (int)a->orl;
	b->ublnk = a->oblnk && (*a->oblnk == 'z' || *a->oblnk == 'Z');
	if(a->ofm==0)
	{	if(b->url>0) b->ufmt=0;
		else b->ufmt=1;
	}
	else if(*a->ofm=='f' || *a->ofm == 'F') b->ufmt=1;
	else b->ufmt=0;
	ufmt = b->ufmt;
#ifdef url_Adjust
	if (b->url && !ufmt)
		url_Adjust(b->url);
#endif
	if (a->ofnm) {
		g_char(a->ofnm,a->ofnmlen,buf);
		if (!buf[0])
			opnerr(a->oerr,107,"open")
		}
	else
		sprintf(buf, "fort.%ld", (long)a->ounit);
	b->uscrtch = 0;
	b->uend=0;
	b->uwrt = 0;
	b->ufd = 0;
	b->urw = 3;
	switch(a->osta ? *a->osta : 'u')
	{
	case 'o':
	case 'O':
		if (access(buf,0))
			opnerr(a->oerr,errno,"open")
		break;
	 case 's':
	 case 'S':
		b->uscrtch=1;
#ifdef HAVE_TMPFILE
		if (!(b->ufd = tmpfile()))
			opnerr(a->oerr,errno,"open")
		b->ufnm = 0;
#ifndef NON_UNIX_STDIO
		b->uinode = b->udev = -1;
#endif
		b->useek = 1;
		return 0;
#else
		(void) strcpy(buf,"tmp.FXXXXXX");
		(void) mktemp(buf);
		goto replace;
#endif

	case 'n':
	case 'N':
		if (!access(buf,0))
			opnerr(a->oerr,128,"open")
		/* no break */
	case 'r':	/* Fortran 90 replace option */
	case 'R':
#ifndef HAVE_TMPFILE
 replace:
#endif
		if (tf = fopen(buf,f__w_mode[0]))
			fclose(tf);
	}

	b->ufnm=(char *) malloc((unsigned int)(strlen(buf)+1));
	if(b->ufnm==NULL) opnerr(a->oerr,113,"no space");
	(void) strcpy(b->ufnm,buf);
	if ((s = a->oacc) && b->url)
		ufmt = 0;
	if(!(tf = fopen(buf, f__w_mode[ufmt|2]))) {
		if (tf = fopen(buf, f__r_mode[ufmt]))
			b->urw = 1;
		else if (tf = fopen(buf, f__w_mode[ufmt])) {
			b->uwrt = 1;
			b->urw = 2;
			}
		else
			err(a->oerr, errno, "open");
		}
	b->useek = f__canseek(b->ufd = tf);
#ifndef NON_UNIX_STDIO
	if((b->uinode = f__inode(buf,&b->udev)) == -1)
		opnerr(a->oerr,108,"open")
#endif
	if(b->useek)
		if (a->orl)
			rewind(b->ufd);
		else if ((s = a->oacc) && (*s == 'a' || *s == 'A')
			&& FSEEK(b->ufd, 0L, SEEK_END))
				opnerr(a->oerr,129,"open");
	return(0);
}

int fk_open(int seq, int fmt, ftnint n)
{
	char nbuf[10];
	olist a;
	// FGVZ: gcc 7.3 outputs a warning that the integer value corresponding
	// to the "%ld" format specifier could (in theory) use up 11 bytes in a
	// string that only allows for five additional bytes. I use the modulo
	// operator to reassure gcc that the integer will be very small.
	//(void) sprintf(nbuf,"fort.%ld",(long)n);
	(void) sprintf(nbuf,"fort.%ld",(long)n % 20);
	a.oerr=1;
	a.ounit=n;
	a.ofnm=nbuf;
	a.ofnmlen=strlen(nbuf);
	a.osta=NULL;
	a.oacc= (char*)(seq==SEQ?"s":"d");
	a.ofm = (char*)(fmt==FMT?"f":"u");
	a.orl = seq==DIR?1:0;
	a.oblnk=NULL;
	return(f_open(&a));
}
