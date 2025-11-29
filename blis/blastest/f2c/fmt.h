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

struct syl
{	int op;
	int p1;
	union { int i[2]; char *s;} p2;
	};
#define RET1 1
#define REVERT 2
#define GOTO 3
#define X 4
#define SLASH 5
#define STACK 6
#define I 7
#define ED 8
#define NED 9
#define IM 10
#define APOS 11
#define H 12
#define TL 13
#define TR 14
#define T 15
#define COLON 16
#define S 17
#define SP 18
#define SS 19
#define P 20
#define BN 21
#define BZ 22
#define F 23
#define E 24
#define EE 25
#define D 26
#define G 27
#define GE 28
#define L 29
#define A 30
#define AW 31
#define O 32
#define NONL 33
#define OM 34
#define Z 35
#define ZM 36
typedef union
{	real pf;
	doublereal pd;
} ufloat;
typedef union
{	short is;
	signed char ic;
	integer il;
#ifdef Allow_TYQUAD
	longint ili;
#endif
} Uint;
#ifdef __cplusplus
extern "C" {
#define Cextern extern "C"
#else
#define Cextern extern
#endif /* __cplusplus */
extern const char *f__fmtbuf;
extern int (*f__doed)(struct syl*, char*, ftnlen),(*f__doned)(struct syl*);
extern int (*f__dorevert)(void);
extern void fmt_bg(void);
extern int pars_f(const char*);
extern int rd_ed(struct syl*, char*, ftnlen),rd_ned(struct syl*);
extern int w_ed(struct syl*, char*, ftnlen),w_ned(struct syl*);
extern int wrt_E(ufloat*, int, int, int, ftnlen);
extern int wrt_F(ufloat*, int, int, ftnlen);
extern int wrt_L(Uint*, int, ftnlen);

extern int f__pc,f__parenlvl,f__revloc;
extern flag f__cblank,f__cplus,f__workdone, f__nonl;
extern int f__scale;
#ifdef __cplusplus
	}
#endif
#define GET(x) if((x=(*f__getn)())<0) return(x)
#define VAL(x) (x!='\n'?x:' ')
#define PUT(x) (*f__putn)(x)

#undef TYQUAD
#ifndef Allow_TYQUAD
#undef longint
#define longint long
#else
#define TYQUAD 14
#endif

