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

#include "f2c.h"

#ifdef __cplusplus
extern "C" {
#endif

    /* Integer */
    shortint pow_hh(shortint *ap, shortint *bp)
    {
        return (shortint)(pow(*ap, *bp));
    }
    integer pow_ii(integer *ap, integer *bp)
    {
        return (integer)(pow(*ap, *bp));
    }
#ifdef INTEGER_STAR_8
    longint pow_qq(longint *ap, longint *bp)
    {
        return (longint)(pow(*ap, *bp));
    }
#endif

    /* Double */
    double pow_ri(real *ap, integer *bp)
    {
        return (pow(*ap, *bp));
    }
    double pow_dd(doublereal *ap, doublereal *bp)
    {
        return (pow(*ap, *bp));
    }
    double pow_di(doublereal *ap, integer *bp)
    {
        return (pow(*ap, *bp));
    }

    /* Complex */
    void pow_ci(complex *p, complex *a, integer *b)
    {
        doublecomplex p1, a1;

        a1.r = a->r;
        a1.i = a->i;

        pow_zi(&p1, &a1, b);

        p->r = p1.r;
        p->i = p1.i;
    }
    void pow_zz(doublecomplex *r, doublecomplex *a, doublecomplex *b)
    {
        double logr, logi, x, y;
        
        logr = log( hypot(a->r, a->i) );
        logi = atan2(a->i, a->r);
        
        x = exp( logr * b->r - logi * b->i );
        y = logr * b->i + logi * b->r;
        
        r->r = x * cos(y);
        r->i = x * sin(y);
    }
    void pow_zi(doublecomplex *p, doublecomplex *a, integer *b)
    {
        integer n;
        unsigned long u;
        double t;
        doublecomplex q, x;
        static doublecomplex one = {1.0, 0.0};
    
        n = *b;
        q.r = 1;
        q.i = 0;
    
        if(n == 0)
            goto done;
        if(n < 0)
        {
            n = -n;
            z_div(&x, &one, a);
        }
        else
        {
            x.r = a->r;
            x.i = a->i;
        }
    
        for(u = n; ; )
        {
            if(u & 01)
            {
                t = q.r * x.r - q.i * x.i;
                q.i = q.r * x.i + q.i * x.r;
                q.r = t;
            }
            if(u >>= 1)
            {
                t = x.r * x.r - x.i * x.i;
                x.i = 2 * x.r * x.i;
                x.r = t;
            }
            else
                break;
        }
        done:
        p->i = q.i;
        p->r = q.r;
    }


#ifdef __cplusplus
}
#endif

