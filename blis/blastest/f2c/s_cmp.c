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

    /* compare two strings */
    integer s_cmp(const char *a0, const char *b0, ftnlen la, ftnlen lb)
    {
        register unsigned char *a, *aend, *b, *bend;
        a = (unsigned char *)a0;
        b = (unsigned char *)b0;
        aend = a + la;
        bend = b + lb;

        if(la <= lb)
        {
            while(a < aend)
                if(*a != *b)
                    return( *a - *b );
                else
                {
                    ++a;
                    ++b;
                }

            while(b < bend)
                if(*b != ' ')
                    return( ' ' - *b );
                else	++b;
        }
        else
        {
            while(b < bend)
                if(*a == *b)
                {
                    ++a;
                    ++b;
                }
                else
                    return( *a - *b );
            while(a < aend)
                if(*a != ' ')
                    return(*a - ' ');
                else	++a;
        }
        return(0);
    }

#ifdef __cplusplus
}
#endif
