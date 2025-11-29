#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "cblas.h"
#include "cblas_f77.h"

void cblas_xerbla(f77_int info, const char *rout, const char *form, ...)
{
   extern int RowMajorStrg;
   char empty[1] = "";
   va_list argptr;

   va_start(argptr, form);

   if (RowMajorStrg)
   {
      if (strstr(rout,"gemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
         else if (info == 11) info =  9;
         else if (info == 9 ) info = 11;
      }
      else if (strstr(rout,"symm") != 0 || strstr(rout,"hemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
      }
      else if (strstr(rout,"trmm") != 0 || strstr(rout,"trsm") != 0)
      {
         if      (info == 7 ) info =  6;
         else if (info == 6 ) info =  7;
      }
      else if (strstr(rout,"gemv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
      }
      else if (strstr(rout,"gbmv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
         else if (info == 6)  info = 5;
         else if (info == 5)  info = 6;
      }
      else if (strstr(rout,"ger") != 0)
      {
         if      (info == 3) info = 2;
         else if (info == 2) info = 3;
         else if (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
      else if ( (strstr(rout,"her2") != 0 || strstr(rout,"hpr2") != 0)
                 && strstr(rout,"her2k") == 0 )
      {
         if      (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
   }
   if (info)
      fprintf(stderr, "Parameter %jd to routine %s was incorrect\n", ( intmax_t )info, rout);
   vfprintf(stderr, form, argptr);
   va_end(argptr);
   if (info && !info) 
      F77_xerbla(empty, &info, 0); /* Force link of our F77 error handler */
   exit(-1);
}
#endif
