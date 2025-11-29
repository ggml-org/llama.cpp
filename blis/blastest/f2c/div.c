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

    void c_div(complex *c, complex *a, complex *b)
    {
        double ratio, den;
        double abr, abi, cr;
    
        if( (abr = b->r) < 0.)
            abr = - abr;
        if( (abi = b->i) < 0.)
            abi = - abi;
        if( abr <= abi )
        {
            if(abi == 0) {
    #ifdef IEEE_COMPLEX_DIVIDE
                float af, bf;
                af = bf = abr;
                if (a->i != 0 || a->r != 0)
                    af = 1.;
                c->i = c->r = af / bf;
                return;
    #else
                sig_die("complex division by zero", 1);
    #endif
            }
            ratio = (double)b->r / b->i ;
            den = b->i * (1 + ratio*ratio);
            cr = (a->r*ratio + a->i) / den;
            c->i = (a->i*ratio - a->r) / den;
        }
    
        else
        {
            ratio = (double)b->i / b->r ;
            den = b->r * (1 + ratio*ratio);
            cr = (a->r + a->i*ratio) / den;
            c->i = (a->i - a->r*ratio) / den;
        }
        c->r = cr;
    }
    void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
    {
        double ratio, den;
        double abr, abi, cr;
    
        if( (abr = b->r) < 0.)
            abr = - abr;
        if( (abi = b->i) < 0.)
            abi = - abi;
        if( abr <= abi )
        {
            if(abi == 0) {
    #ifdef IEEE_COMPLEX_DIVIDE
                if (a->i != 0 || a->r != 0)
                    abi = 1.;
                c->i = c->r = abi / abr;
                return;
    #else
                sig_die("complex division by zero", 1);
    #endif
            }
            ratio = b->r / b->i ;
            den = b->i * (1 + ratio*ratio);
            cr = (a->r*ratio + a->i) / den;
            c->i = (a->i*ratio - a->r) / den;
        }
    
        else
        {
            ratio = b->i / b->r ;
            den = b->r * (1 + ratio*ratio);
            cr = (a->r + a->i*ratio) / den;
            c->i = (a->i - a->r*ratio) / den;
        }
        c->r = cr;
    }

#ifdef __cplusplus
}
#endif
