
#ifndef _BLIS_UTEST_H_
#define _BLIS_UTEST_H_

#define BLIS_DEFAULT_MR_S    8
#define BLIS_DEFAULT_NR_S    4

#define BLIS_DEFAULT_MR_D    8
#define BLIS_DEFAULT_NR_D    4

#define BLIS_DEFAULT_MR_C    8
#define BLIS_DEFAULT_NR_C    4

#define BLIS_DEFAULT_MR_Z    8
#define BLIS_DEFAULT_NR_Z    4

typedef unsigned long dim_t;
typedef long inc_t;

// Complex types
typedef struct scomplex_s
{
        float  real;
        float  imag;
} scomplex;

typedef struct dcomplex_s
{
        double real;
        double imag;
} dcomplex;

#define bli_check_error_code(x)

#endif
