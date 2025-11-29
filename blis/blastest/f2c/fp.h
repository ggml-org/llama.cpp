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

#define FMAX 40
#define EXPMAXDIGS 8
#define EXPMAX 99999999
/* FMAX = max number of nonzero digits passed to atof() */
/* EXPMAX = 10^EXPMAXDIGS - 1 = largest allowed exponent absolute value */

#ifdef V10 /* Research Tenth-Edition Unix */
#include "local.h"
#endif

/* MAXFRACDIGS and MAXINTDIGS are for wrt_F -- bounds (not necessarily
   tight) on the maximum number of digits to the right and left of
 * the decimal point.
 */

#ifdef VAX
#define MAXFRACDIGS 56
#define MAXINTDIGS 38
#else
#ifdef CRAY
#define MAXFRACDIGS 9880
#define MAXINTDIGS 9864
#else
/* values that suffice for IEEE double */
#define MAXFRACDIGS 344
#define MAXINTDIGS 308
#endif
#endif
