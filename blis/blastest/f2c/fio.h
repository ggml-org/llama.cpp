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
#include <stdio.h>
#include <errno.h>
#include <stddef.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef HAVE_FSEEKO
#define OFF_T off_t
#define FSEEK fseeko
#define FTELL ftello
#else
#define OFF_T long
#define FSEEK fseek
#define FTELL ftell
#endif

#ifdef MSDOS
#ifndef NON_UNIX_STDIO
#define NON_UNIX_STDIO
#endif
#endif

typedef long uiolen;

/*units*/
typedef struct
{	FILE *ufd;	/*0=unconnected*/
	char *ufnm;
#ifndef MSDOS
	long uinode;
	int udev;
#endif
	int url;	/*0=sequential*/
	flag useek;	/*true=can backspace, use dir, ...*/
	flag ufmt;
	flag urw;	/* (1 for can read) | (2 for can write) */
	flag ublnk;
	flag uend;
	flag uwrt;	/*last io was write*/
	flag uscrtch;
} unit;

extern int (*f__getn)(void);	/* for formatted input */
extern void (*f__putn)(int);	/* for formatted output */
extern void x_putc(int);
extern long f__inode(char*,int*);
extern void sig_die(const char*,int);
extern void f__fatal(int, const char*);
extern int t_runc(alist*);
extern int f__nowreading(unit*), f__nowwriting(unit*);
extern int fk_open(int,int,ftnint);
extern int en_fio(void);
extern void f_init(void);
extern int (*f__donewrec)(void), t_putc(int), x_wSL(void);
extern void b_char(const char*,char*,ftnlen), g_char(const char*,ftnlen,char*);
extern int c_sfe(cilist*);
extern int z_rnew(void);
extern int err__fl(int,int,const char*);
extern int xrd_SL(void);
extern int f__putbuf(int);
extern int f__canseek(FILE *f);
extern int z_getc(void);
extern void z_putc(int c);
extern integer f_open(olist *a);
#ifdef INTEGER_STAR_8
extern char *f__icvt(longint value, int *ndigit, int *sign, int base);
#else
extern char *f__icvt(integer value, int *ndigit, int *sign, int base);
#endif
extern int t_getc(void);

extern flag f__init;
extern cilist *f__elist;	/*active external io list*/
extern flag f__reading,f__external,f__sequential,f__formatted;
extern int (*f__doend)(void);
extern FILE *f__cf;	/*current file*/
extern unit *f__curunit;	/*current unit*/
extern unit f__units[];

extern char *f__icptr;
extern char *f__icend;
extern icilist *f__svic;
extern int f__icnum;

#define err(f,m,s) {if(f) errno= m; else f__fatal(m,s); return(m);}
#define errfl(f,m,s) return err__fl((int)f,m,s)

/*Table sizes*/
#define MXUNIT 100

extern int f__recpos;	/*position in current record*/
extern OFF_T f__cursor;	/* offset to move to */
extern OFF_T f__hiwater;	/* so TL doesn't confuse us */

#define WRITE	1
#define READ	2
#define SEQ	3
#define DIR	4
#define FMT	5
#define UNF	6
#define EXT	7
#define INT	8

#define buf_end(x) (x->_flag & _IONBF ? x->_ptr : x->_base + BUFSIZ)

extern const char *f__fmtbuf;
extern const char *f__r_mode[2];
extern const char *f__w_mode[];

extern int l_eof;

extern int c_le(cilist *a);
extern int l_read(ftnint *number, char *ptr, ftnlen len, ftnint type);
extern int l_write(ftnint *number, char *ptr, ftnlen len, ftnint type);

extern flag f__lquit;
extern int f__lcount;
extern char *f__icptr;
extern char *f__icend;
extern icilist *f__svic;
extern int f__icnum, f__recpos;
extern int f__Aquote;

extern int x_rsne(cilist*);
extern void x_wsne(cilist *a);

extern flag f__lquit;
extern int f__lcount, nml_read;
extern int t_getc(void);
extern uiolen f__reclen;
extern ftnint L_len;
extern int f__scale;

extern int (*l_getc)(void);
extern int (*l_ungetc)(int,FILE*);
extern int (*f__lioproc)(ftnint*, char*, ftnlen, ftnint);

int do_us(ftnint *number, char *ptr, ftnlen len);
integer do_ud(ftnint *number, char *ptr, ftnlen len);
integer do_uio(ftnint *number, char *ptr, ftnlen len);
integer do_fio(ftnint *number, char *ptr, ftnlen len);
int en_fio(void);

extern int x_wSL(void);
extern int x_getc(void);
extern int x_endp(void);
