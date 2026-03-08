#ifndef GGML_POWER8_COMPAT_H
#define GGML_POWER8_COMPAT_H

#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)

#include <altivec.h>
#include <stddef.h>
#include <string.h>

#define __POWER9_VECTOR__ 1

#ifndef vec_xl_len
#define vec_xl_len(ptr, len) \
    __extension__ ({ \
        __attribute__((aligned(16))) unsigned char __buf[16] = {0}; \
        size_t __len = (len) > 16 ? 16 : (size_t) (len); \
        memcpy(__buf, (ptr), __len); \
        *((__typeof__(vec_xl(0, (ptr))) *) __buf); \
    })
#endif

#ifndef vec_absd
#define vec_absd(a, b) vec_sub(vec_max((a), (b)), vec_min((a), (b)))
#endif

#ifndef vec_vaddudm
#define vec_vaddudm(a, b) vec_add((a), (b))
#endif

#ifndef vec_vsubudm
#define vec_vsubudm(a, b) vec_sub((a), (b))
#endif

#ifndef vaddudm
#define vaddudm(a, b) vec_vaddudm((a), (b))
#endif

#ifndef vsubudm
#define vsubudm(a, b) vec_vsubudm((a), (b))
#endif

#endif

#endif
