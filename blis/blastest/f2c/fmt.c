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
#include "fmt.h"
#define skip(s) while(*s==' ') s++
#ifdef interdata
#define SYLMX 300
#endif
#ifdef pdp11
#define SYLMX 300
#endif
#ifdef vax
#define SYLMX 300
#endif
#ifndef SYLMX
#define SYLMX 300
#endif
#define GLITCH '\2'
	/* special quote character for stu */
extern flag f__cblank,f__cplus;	/*blanks in I and compulsory plus*/
static struct syl f__syl[SYLMX];
int f__parenlvl,f__pc,f__revloc;

static const char *ap_end(const char *s)
{	char quote;
	quote= *s++;
	for(;*s;s++)
	{	if(*s!=quote) continue;
		if(*++s!=quote) return(s);
	}
	if(f__elist->cierr) {
		errno = 100;
		return(NULL);
	}
	f__fatal(100, "bad string");
	/*NOTREACHED*/ return 0;
}

static int op_gen(int a, int b, int c, int d)
{	struct syl *p= &f__syl[f__pc];
	if(f__pc>=SYLMX)
	{	fprintf(stderr,"format too complicated:\n");
		sig_die(f__fmtbuf, 1);
	}
	p->op=a;
	p->p1=b;
	p->p2.i[0]=c;
	p->p2.i[1]=d;
	return(f__pc++);
}

static const char *f_list(const char*);

static const char *gt_num(const char *s, int *n, int n1)
{	int m=0,f__cnt=0;
	char c;
	for(c= *s;;c = *s)
	{	if(c==' ')
		{	s++;
			continue;
		}
		if(c>'9' || c<'0') break;
		m=10*m+c-'0';
		f__cnt++;
		s++;
	}
	if(f__cnt==0) {
		if (!n1)
			s = 0;
		*n=n1;
		}
	else *n=m;
	return(s);
}

static const char *f_s(const char *s, int curloc)
{
	skip(s);
	if(*s++!='(')
	{
		return(NULL);
	}
	if(f__parenlvl++ ==1) f__revloc=curloc;
	if(op_gen(RET1,curloc,0,0)<0 ||
		(s=f_list(s))==NULL)
	{
		return(NULL);
	}
	skip(s);
	return(s);
}

static int ne_d(const char *s, const char **p)
{	int n,x,sign=0;
	struct syl *sp;
	switch(*s)
	{
	default:
		return(0);
	case ':': (void) op_gen(COLON,0,0,0); break;
	case '$':
		(void) op_gen(NONL, 0, 0, 0); break;
	case 'B':
	case 'b':
		if(*++s=='z' || *s == 'Z') (void) op_gen(BZ,0,0,0);
		else (void) op_gen(BN,0,0,0);
		break;
	case 'S':
	case 's':
		if(*(s+1)=='s' || *(s+1) == 'S')
		{	x=SS;
			s++;
		}
		else if(*(s+1)=='p' || *(s+1) == 'P')
		{	x=SP;
			s++;
		}
		else x=S;
		(void) op_gen(x,0,0,0);
		break;
	case '/': (void) op_gen(SLASH,0,0,0); break;
	case '-': sign=1;
	case '+':	s++;	/*OUTRAGEOUS CODING TRICK*/
	case '0': case '1': case '2': case '3': case '4':
	case '5': case '6': case '7': case '8': case '9':
		if (!(s=gt_num(s,&n,0))) {
 bad:			*p = 0;
			return 1;
			}
		switch(*s)
		{
		default:
			return(0);
		case 'P':
		case 'p': if(sign) n= -n; (void) op_gen(P,n,0,0); break;
		case 'X':
		case 'x': (void) op_gen(X,n,0,0); break;
		case 'H':
		case 'h':
			sp = &f__syl[op_gen(H,n,0,0)];
			sp->p2.s = (char*)s + 1;
			s+=n;
			break;
		}
		break;
	case GLITCH:
	case '"':
	case '\'':
		sp = &f__syl[op_gen(APOS,0,0,0)];
		sp->p2.s = (char*)s;
		if((*p = ap_end(s)) == NULL)
			return(0);
		return(1);
	case 'T':
	case 't':
		if(*(s+1)=='l' || *(s+1) == 'L')
		{	x=TL;
			s++;
		}
		else if(*(s+1)=='r'|| *(s+1) == 'R')
		{	x=TR;
			s++;
		}
		else x=T;
		if (!(s=gt_num(s+1,&n,0)))
			goto bad;
		s--;
		(void) op_gen(x,n,0,0);
		break;
	case 'X':
	case 'x': (void) op_gen(X,1,0,0); break;
	case 'P':
	case 'p': (void) op_gen(P,1,0,0); break;
	}
	s++;
	*p=s;
	return(1);
}

static int e_d(const char *s, const char **p)
{	int i,im,n,w,d,e,found=0,x=0;
	const char *sv=s;
	s=gt_num(s,&n,1);
	(void) op_gen(STACK,n,0,0);
	switch(*s++)
	{
	default: break;
	case 'E':
	case 'e':	x=1;
	case 'G':
	case 'g':
		found=1;
		if (!(s=gt_num(s,&w,0))) {
 bad:
			*p = 0;
			return 1;
			}
		if(w==0) break;
		if(*s=='.') {
			if (!(s=gt_num(s+1,&d,0)))
				goto bad;
			}
		else d=0;
		if(*s!='E' && *s != 'e')
			(void) op_gen(x==1?E:G,w,d,0);	/* default is Ew.dE2 */
		else {
			if (!(s=gt_num(s+1,&e,0)))
				goto bad;
			(void) op_gen(x==1?EE:GE,w,d,e);
			}
		break;
	case 'O':
	case 'o':
		i = O;
		im = OM;
		goto finish_I;
	case 'Z':
	case 'z':
		i = Z;
		im = ZM;
		goto finish_I;
	case 'L':
	case 'l':
		found=1;
		if (!(s=gt_num(s,&w,0)))
			goto bad;
		if(w==0) break;
		(void) op_gen(L,w,0,0);
		break;
	case 'A':
	case 'a':
		found=1;
		skip(s);
		if(*s>='0' && *s<='9')
		{	s=gt_num(s,&w,1);
			if(w==0) break;
			(void) op_gen(AW,w,0,0);
			break;
		}
		(void) op_gen(A,0,0,0);
		break;
	case 'F':
	case 'f':
		if (!(s=gt_num(s,&w,0)))
			goto bad;
		found=1;
		if(w==0) break;
		if(*s=='.') {
			if (!(s=gt_num(s+1,&d,0)))
				goto bad;
			}
		else d=0;
		(void) op_gen(F,w,d,0);
		break;
	case 'D':
	case 'd':
		found=1;
		if (!(s=gt_num(s,&w,0)))
			goto bad;
		if(w==0) break;
		if(*s=='.') {
			if (!(s=gt_num(s+1,&d,0)))
				goto bad;
			}
		else d=0;
		(void) op_gen(D,w,d,0);
		break;
	case 'I':
	case 'i':
		i = I;
		im = IM;
 finish_I:
		if (!(s=gt_num(s,&w,0)))
			goto bad;
		found=1;
		if(w==0) break;
		if(*s!='.')
		{	(void) op_gen(i,w,0,0);
			break;
		}
		if (!(s=gt_num(s+1,&d,0)))
			goto bad;
		(void) op_gen(im,w,d,0);
		break;
	}
	if(found==0)
	{	f__pc--; /*unSTACK*/
		*p=sv;
		return(0);
	}
	*p=s;
	return(1);
}

static const char *i_tem(const char *s)
{	const char *t;
	int n,curloc;
	if(*s==')') return(s);
	if(ne_d(s,&t)) return(t);
	if(e_d(s,&t)) return(t);
	s=gt_num(s,&n,1);
	if((curloc=op_gen(STACK,n,0,0))<0) return(NULL);
	return(f_s(s,curloc));
}

static const char *f_list(const char *s)
{
	for(;*s!=0;)
	{	skip(s);
		if((s=i_tem(s))==NULL) return(NULL);
		skip(s);
		if(*s==',') s++;
		else if(*s==')')
		{	if(--f__parenlvl==0)
			{
				(void) op_gen(REVERT,f__revloc,0,0);
				return(++s);
			}
			(void) op_gen(GOTO,0,0,0);
			return(++s);
		}
	}
	return(NULL);
}

int pars_f(const char *s)
{
	f__parenlvl=f__revloc=f__pc=0;
	if(f_s(s,0) == NULL)
	{
		return(-1);
	}
	return(0);
}

#define STKSZ 10
int f__cnt[STKSZ],f__ret[STKSZ],f__cp,f__rp;
flag f__workdone, f__nonl;

static int type_f(int n)
{
	switch(n)
	{
	default:
		return(n);
	case RET1:
		return(RET1);
	case REVERT: return(REVERT);
	case GOTO: return(GOTO);
	case STACK: return(STACK);
	case X:
	case SLASH:
	case APOS: case H:
	case T: case TL: case TR:
		return(NED);
	case F:
	case I:
	case IM:
	case A: case AW:
	case O: case OM:
	case L:
	case E: case EE: case D:
	case G: case GE:
	case Z: case ZM:
		return(ED);
	}
}

integer do_fio(ftnint *number, char *ptr, ftnlen len)
{	struct syl *p;
	int n,i;
	for(i=0;i<*number;i++,ptr+=len)
	{
loop:	switch(type_f((p= &f__syl[f__pc])->op))
	{
	default:
		fprintf(stderr,"unknown code in do_fio: %d\n%s\n",
			p->op,f__fmtbuf);
		err(f__elist->cierr,100,"do_fio");
	case NED:
		if((*f__doned)(p))
		{	f__pc++;
			goto loop;
		}
		f__pc++;
		continue;
	case ED:
		if(f__cnt[f__cp]<=0)
		{	f__cp--;
			f__pc++;
			goto loop;
		}
		if(ptr==NULL)
			return((*f__doend)());
		f__cnt[f__cp]--;
		f__workdone=1;
		if((n=(*f__doed)(p,ptr,len))>0)
			errfl(f__elist->cierr,errno,"fmt");
		if(n<0)
			err(f__elist->ciend,(EOF),"fmt");
		continue;
	case STACK:
		f__cnt[++f__cp]=p->p1;
		f__pc++;
		goto loop;
	case RET1:
		f__ret[++f__rp]=p->p1;
		f__pc++;
		goto loop;
	case GOTO:
		if(--f__cnt[f__cp]<=0)
		{	f__cp--;
			f__rp--;
			f__pc++;
			goto loop;
		}
		f__pc=1+f__ret[f__rp--];
		goto loop;
	case REVERT:
		f__rp=f__cp=0;
		f__pc = p->p1;
		if(ptr==NULL)
			return((*f__doend)());
		if(!f__workdone) return(0);
		if((n=(*f__dorevert)()) != 0) return(n);
		goto loop;
	case COLON:
		if(ptr==NULL)
			return((*f__doend)());
		f__pc++;
		goto loop;
	case NONL:
		f__nonl = 1;
		f__pc++;
		goto loop;
	case S:
	case SS:
		f__cplus=0;
		f__pc++;
		goto loop;
	case SP:
		f__cplus = 1;
		f__pc++;
		goto loop;
	case P:	f__scale=p->p1;
		f__pc++;
		goto loop;
	case BN:
		f__cblank=0;
		f__pc++;
		goto loop;
	case BZ:
		f__cblank=1;
		f__pc++;
		goto loop;
	}
	}
	return(0);
}

int
en_fio(void)
{	ftnint one=1;
	return(do_fio(&one,(char *)NULL,(ftnint)0));
}

void
fmt_bg(void)
{
	f__workdone=f__cp=f__rp=f__pc=f__cursor=0;
	f__cnt[0]=f__ret[0]=0;
}
