#pragma once

#include "spe_kernels.h"
#include "repack.h"

#include <algorithm>
#include <cassert>

#ifdef _MSC_VER
#    define SPINE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__CLANG__)
#    define SPINE_FORCE_INLINE inline __attribute__((always_inline))
#else
#    define SPINE_FORCE_INLINE inline
#endif

#ifndef NDEBUG
#    define SPE_ASSERT(expr) assert(expr)
#else
#    define SPE_ASSERT(expr) ;
#endif

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#ifndef SPE_KERNELS_PRELUDE_SHIM
#define SPE_KERNELS_PRELUDE_SHIM
#endif

#define SPE_ARCH_IME2 ((uint8_t) (1 << 1))

#ifndef PACK_ASM_IME2
#define PACK_ASM_IME2

#define PACK_X8_M32N16_VLD_M8_0               \
    "vle8.v         v0, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v1, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v2, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v3, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v4, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v5, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v6, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v7, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t"

#define PACK_X8_M32N16_VLD_M8_1               \
    "vle8.v         v8, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v9, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v10, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v11, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v12, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v13, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v14, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v15, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t"

#define PACK_X8_M32N16_N128_0                 \
    "vpack.vv       v16, v0, v1, 1      \n\t" \
    "vpack.vv       v18, v2, v3, 1      \n\t" \
    "vpack.vv       v20, v4, v5, 1      \n\t" \
    "vpack.vv       v22, v6, v7, 1      \n\t" \
    "vpack.vv       v0, v16, v18, 2     \n\t" \
    "vpack.vv       v2, v17, v19, 2     \n\t" \
    "vpack.vv       v4, v20, v22, 2     \n\t" \
    "vpack.vv       v6, v21, v23, 2     \n\t" \
    "vpack.vv       v16, v0, v4, 3      \n\t" \
    "vpack.vv       v18, v1, v5, 3      \n\t" \
    "vpack.vv       v20, v2, v6, 3      \n\t" \
    "vpack.vv       v22, v3, v7, 3      \n\t"

#define PACK_X8_M32N16_N128_1                 \
    "vpack.vv       v24, v8, v9, 1      \n\t" \
    "vpack.vv       v26, v10, v11, 1    \n\t" \
    "vpack.vv       v28, v12, v13, 1    \n\t" \
    "vpack.vv       v30, v14, v15, 1    \n\t" \
    "vpack.vv       v8, v24, v26, 2     \n\t" \
    "vpack.vv       v10, v25, v27, 2    \n\t" \
    "vpack.vv       v12, v28, v30, 2    \n\t" \
    "vpack.vv       v14, v29, v31, 2    \n\t" \
    "vpack.vv       v24, v8, v12, 3     \n\t" \
    "vpack.vv       v26, v9, v13, 3     \n\t" \
    "vpack.vv       v28, v10, v14, 3    \n\t" \
    "vpack.vv       v30, v11, v15, 3    \n\t"

#define PACK_X8_M32N16_VST_M16N64_0           \
    "vse8.v         v16, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v24, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v17, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v25, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v18, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v26, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v19, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v27, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t"

#define PACK_X8_M32N16_VST_M16N64_1           \
    "vse8.v         v20, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v28, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v21, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v29, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v22, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v30, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v23, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v31, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t"

#define PACK_X16_M16N8                                          \
    "vpack.vv               v16, v0, v1, 1                \n\t" \
    "vpack.vv               v18, v2, v3, 1                \n\t" \
    "vpack.vv               v20, v4, v5, 1                \n\t" \
    "vpack.vv               v22, v6, v7, 1                \n\t" \
    "vpack.vv               v0, v16, v18, 2               \n\t" \
    "vpack.vv               v2, v17, v19, 2               \n\t" \
    "vpack.vv               v4, v20, v22, 2               \n\t" \
    "vpack.vv               v6, v21, v23, 2               \n\t" \
    "vpack.vv               v16, v0, v4, 3                \n\t" \
    "vpack.vv               v18, v1, v5, 3                \n\t" \
    "vpack.vv               v20, v2, v6, 3                \n\t" \
    "vpack.vv               v22, v3, v7, 3                \n\t" \
    "vpack.vv               v24, v8, v9, 1                \n\t" \
    "vpack.vv               v26, v10, v11, 1              \n\t" \
    "vpack.vv               v28, v12, v13, 1              \n\t" \
    "vpack.vv               v30, v14, v15, 1              \n\t" \
    "vpack.vv               v8, v24, v26, 2               \n\t" \
    "vpack.vv               v10, v25, v27, 2              \n\t" \
    "vpack.vv               v12, v28, v30, 2              \n\t" \
    "vpack.vv               v14, v29, v31, 2              \n\t" \
    "vpack.vv               v24, v8, v12, 3               \n\t" \
    "vpack.vv               v26, v9, v13, 3               \n\t" \
    "vpack.vv               v28, v10, v14, 3              \n\t" \
    "vpack.vv               v30, v11, v15, 3              \n\t"
#endif

#pragma once

// VLEN=1024; strides are in bytes. Callers set vtype.

// clang-format off

#define PACK_CAST_LOAD_F32_M8K64                                               \
    "vle32.v        v16, (t0)                                            \n\t" \
    "vle32.v        v18, (t1)                                            \n\t" \
    "vle32.v        v20, (t5)                                            \n\t" \
    "vle32.v        v22, (t6)                                            \n\t" \
    "sh2add         t0, %[stride], t0                                    \n\t" \
    "sh2add         t1, %[stride], t1                                    \n\t" \
    "sh2add         t5, %[stride], t5                                    \n\t" \
    "sh2add         t6, %[stride], t6                                    \n\t" \
    "vle32.v        v24, (t0)                                            \n\t" \
    "vle32.v        v26, (t1)                                            \n\t" \
    "vle32.v        v28, (t5)                                            \n\t" \
    "vle32.v        v30, (t6)                                            \n\t"

#define PACK_CAST_NARROW_F32_F16_M8_LO                                         \
    "vfncvt.f.f.w   v0, v16                                              \n\t" \
    "vfncvt.f.f.w   v1, v18                                              \n\t" \
    "vfncvt.f.f.w   v2, v20                                              \n\t" \
    "vfncvt.f.f.w   v3, v22                                              \n\t" \
    "vfncvt.f.f.w   v4, v24                                              \n\t" \
    "vfncvt.f.f.w   v5, v26                                              \n\t" \
    "vfncvt.f.f.w   v6, v28                                              \n\t" \
    "vfncvt.f.f.w   v7, v30                                              \n\t"

#define PACK_CAST_NARROW_F32_F16_M8_HI                                         \
    "vfncvt.f.f.w   v8, v16                                              \n\t" \
    "vfncvt.f.f.w   v9, v18                                              \n\t" \
    "vfncvt.f.f.w   v10, v20                                             \n\t" \
    "vfncvt.f.f.w   v11, v22                                             \n\t" \
    "vfncvt.f.f.w   v12, v24                                             \n\t" \
    "vfncvt.f.f.w   v13, v26                                             \n\t" \
    "vfncvt.f.f.w   v14, v28                                             \n\t" \
    "vfncvt.f.f.w   v15, v30                                             \n\t"

#define PACK_CAST_PACK_F16_M8K64                                               \
    "vpack.vv       v8,  v0,  v1,  1                                     \n\t" \
    "vpack.vv       v10, v2,  v3,  1                                     \n\t" \
    "vpack.vv       v12, v4,  v5,  1                                     \n\t" \
    "vpack.vv       v14, v6,  v7,  1                                     \n\t" \
    "vpack.vv       v0,  v8,  v10, 2                                     \n\t" \
    "vpack.vv       v2,  v9,  v11, 2                                     \n\t" \
    "vpack.vv       v4,  v12, v14, 2                                     \n\t" \
    "vpack.vv       v6,  v13, v15, 2                                     \n\t" \
    "vpack.vv       v8,  v0,  v4,  3                                     \n\t" \
    "vpack.vv       v10, v1,  v5,  3                                     \n\t" \
    "vpack.vv       v12, v2,  v6,  3                                     \n\t" \
    "vpack.vv       v14, v3,  v7,  3                                     \n\t"

#define PACK_CAST_STORE_F16_M8K64                                              \
    "vse16.v        v8, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v9, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v10, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v11, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v12, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v13, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v14, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v15, (t2)                                            \n\t"

#define PACK_CAST_STORE_F16_M16K64                                             \
    "vse16.v        v16, (t0)                                            \n\t" \
    "vse16.v        v24, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v17, (t0)                                            \n\t" \
    "vse16.v        v25, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v18, (t0)                                            \n\t" \
    "vse16.v        v26, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v19, (t0)                                            \n\t" \
    "vse16.v        v27, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v20, (t0)                                            \n\t" \
    "vse16.v        v28, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v21, (t0)                                            \n\t" \
    "vse16.v        v29, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v22, (t0)                                            \n\t" \
    "vse16.v        v30, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v23, (t0)                                            \n\t" \
    "vse16.v        v31, (t1)                                            \n\t"

// Preserve v0: later scaling uses its paired-row mask.

#define PACK_CAST_UNPACK_U4_F16_M32K32                                         \
    "vzext.vf2      v8, v4                                               \n\t" \
    "vsrl.vi        v16, v8, 4                                           \n\t" \
    "vand.vi        v8, v8, 15                                           \n\t" \
    "vsll.vi        v16, v16, 8                                          \n\t" \
    "vor.vv         v24, v8, v16                                         \n\t" \
    "vzext.vf2      v8, v24                                              \n\t" \
    "vzext.vf2      v16, v28                                             \n\t" \
    "vsub.vx        v8, v8, %[zero_point]                                \n\t" \
    "vsub.vx        v16, v16, %[zero_point]                              \n\t" \
    "vfcvt.f.x.v    v8, v8                                               \n\t" \
    "vfcvt.f.x.v    v16, v16                                             \n\t"

#define PACK_CAST_LOAD_SCALE_F16_M16_LO                                        \
    "flh            ft0, 0(t1)                                           \n\t" \
    "flh            ft1, 2(t1)                                           \n\t" \
    "flh            ft2, 4(t1)                                           \n\t" \
    "flh            ft3, 6(t1)                                           \n\t" \
    "flh            ft4, 8(t1)                                           \n\t" \
    "flh            ft5, 10(t1)                                          \n\t" \
    "flh            ft6, 12(t1)                                          \n\t" \
    "flh            ft7, 14(t1)                                          \n\t" \
    "flh            fa0, 16(t1)                                          \n\t" \
    "flh            fa1, 18(t1)                                          \n\t" \
    "flh            fa2, 20(t1)                                          \n\t" \
    "flh            fa3, 22(t1)                                          \n\t" \
    "flh            fa4, 24(t1)                                          \n\t" \
    "flh            fa5, 26(t1)                                          \n\t" \
    "flh            fa6, 28(t1)                                          \n\t" \
    "flh            fa7, 30(t1)                                          \n\t"

#define PACK_CAST_LOAD_SCALE_F16_M16_HI                                        \
    "flh            ft0, 32(t1)                                          \n\t" \
    "flh            ft1, 34(t1)                                          \n\t" \
    "flh            ft2, 36(t1)                                          \n\t" \
    "flh            ft3, 38(t1)                                          \n\t" \
    "flh            ft4, 40(t1)                                          \n\t" \
    "flh            ft5, 42(t1)                                          \n\t" \
    "flh            ft6, 44(t1)                                          \n\t" \
    "flh            ft7, 46(t1)                                          \n\t" \
    "flh            fa0, 48(t1)                                          \n\t" \
    "flh            fa1, 50(t1)                                          \n\t" \
    "flh            fa2, 52(t1)                                          \n\t" \
    "flh            fa3, 54(t1)                                          \n\t" \
    "flh            fa4, 56(t1)                                          \n\t" \
    "flh            fa5, 58(t1)                                          \n\t" \
    "flh            fa6, 60(t1)                                          \n\t" \
    "flh            fa7, 62(t1)                                          \n\t"

// v0 selects the upper row of each pair; v1/v2 carry row scales.

#define PACK_CAST_SCALE_F16_M16K32_LO                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v8, v8, v1                                           \n\t" \
    "vfmul.vv       v9, v9, v2                                           \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v10, v10, v1                                         \n\t" \
    "vfmul.vv       v11, v11, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v12, v12, v1                                         \n\t" \
    "vfmul.vv       v13, v13, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v14, v14, v1                                         \n\t" \
    "vfmul.vv       v15, v15, v2                                         \n\t"

#define PACK_CAST_SCALE_F16_M16K32_HI                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v16, v16, v1                                         \n\t" \
    "vfmul.vv       v17, v17, v2                                         \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v18, v18, v1                                         \n\t" \
    "vfmul.vv       v19, v19, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v20, v20, v1                                         \n\t" \
    "vfmul.vv       v21, v21, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v22, v22, v1                                         \n\t" \
    "vfmul.vv       v23, v23, v2                                         \n\t"

// Preserve v0 and the next packed block in v4..v7.

#define PACK_CAST_PACK_F16_M16K32_LO                                           \
    "vpack.vv       v24, v8, v9, 1                                       \n\t" \
    "vpack.vv       v26, v10, v11, 1                                     \n\t" \
    "vpack.vv       v8, v24, v25, 1                                      \n\t" \
    "vpack.vv       v10, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v8, v10, 3                                      \n\t" \
    "vpack.vv       v26, v9, v11, 3                                      \n\t" \
    "vpack.vv       v28, v12, v13, 1                                     \n\t" \
    "vpack.vv       v30, v14, v15, 1                                     \n\t" \
    "vpack.vv       v12, v28, v29, 1                                     \n\t" \
    "vpack.vv       v14, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v12, v14, 3                                     \n\t" \
    "vpack.vv       v30, v13, v15, 3                                     \n\t"

#define PACK_CAST_PACK_F16_M16K32_HI                                           \
    "vpack.vv       v24, v16, v17, 1                                     \n\t" \
    "vpack.vv       v26, v18, v19, 1                                     \n\t" \
    "vpack.vv       v16, v24, v25, 1                                     \n\t" \
    "vpack.vv       v18, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v16, v18, 3                                     \n\t" \
    "vpack.vv       v26, v17, v19, 3                                     \n\t" \
    "vpack.vv       v28, v20, v21, 1                                     \n\t" \
    "vpack.vv       v30, v22, v23, 1                                     \n\t" \
    "vpack.vv       v20, v28, v29, 1                                     \n\t" \
    "vpack.vv       v22, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v20, v22, 3                                     \n\t" \
    "vpack.vv       v30, v21, v23, 3                                     \n\t"

// tile_stride is 512 bytes; t3 starts eight rows after t2.

#define PACK_CAST_STORE_F16_M16K32                                             \
    "vse16.v        v24, (t2)                                            \n\t" \
    "vse16.v        v28, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v25, (t2)                                            \n\t" \
    "vse16.v        v29, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v26, (t2)                                            \n\t" \
    "vse16.v        v30, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v27, (t2)                                            \n\t" \
    "vse16.v        v31, (t3)                                            \n\t"

// clang-format on

#ifndef MMT4D_ASM_IME2
#define MMT4D_ASM_IME2
// clang-format off

// VLEN=1024; A=[kc][16][8], B=[kc][32][8], C uses byte row strides.
// v0..v7 hold A, v8..v15 B, and v16..v31 FP32 accumulators.

#define _A0         "a1"
#define _A1         "a2"

#define _B0         "s1"
#define _B1         "s2"
#define _B2         "s3"
#define _B3         "s4"

#define _TMP_VL     "t0"
#define _TMP_KC     "t2"

#define _A          "%[a]"
#define _B          "%[b]"
#define _C          "%[c]"
#define _M          "%[m]"
#define _N          "%[n]"
#define _KC         "%[kc]"
#define _CS         "%[cs]"
#define _MIN        "%[min]"
#define _MAX        "%[max]"
#define _BIAS       "%[bias]"

#define _FSCALAR    "%[fscalar]"
#define _ALPHA      "%[alpha]"
#define _ZERO       "%[zero]"

#define MMT4D_FP16_M16N32K8_INIT_IMPL(A, A0, A1, B, B0, B1, B2, B3, KC, TMP_VL, TMP_KC)         \
    "vsetvli            " TMP_VL ", zero, e32, m8, tu, mu       \n\t"                           \
    "vxor.vv            v16, v16, v16                           \n\t"                           \
    "vxor.vv            v24, v24, v24                           \n\t"                           \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
    "addi               " A0 ", " A ", 0                        \n\t"                           \
    "addi               " A1 ", " A ", 8*8*2                    \n\t"                           \
    "addi               " B0 ", " B ", 0                        \n\t"                           \
    "addi               " B1 ", " B ", 8*8*2                    \n\t"                           \
    "addi               " B2 ", " B ", 8*8*4                    \n\t"                           \
    "addi               " B3 ", " B ", 8*8*6                    \n\t"                           \
    "srli               " TMP_KC ", " KC ", 2                   \n\t"

#define MMT4D_FP16_M16N32K8_INIT                                                                \
    MMT4D_FP16_M16N32K8_INIT_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _KC, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(KC, A0, A1, B0, B1, B2, B3, TMP_VL, TMP_KC)       \
    "blez               " TMP_KC ", TAIL_KC1%=                  \n\t"                           \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, A0, A1, _B, B0, B1, B2, B3, TMP_VL)                      \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", TAIL_KC4%=                  \n\t"                           \
    "LOOP_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC4%=                  \n\t"                           \
    "TAIL_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                              \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                          \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
    "TAIL_KC1%=:                                                \n\t"                           \
    "andi               " TMP_KC ", " KC ", 3                   \n\t"                           \
    "blez               " TMP_KC ", POST_PROCESS%=              \n\t"                           \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
    "LOOP_KC1%=:                                                \n\t"                           \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC1%=                  \n\t"                           \
    "POST_PROCESS%=:                                            \n\t"

#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP                                                        \
    MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(_KC, _A0, _A1, _B0, _B1, _B2, _B3, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK_IMPL(TMP_VL)                                     \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32

#define MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK                                                  \
    MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK_IMPL(_TMP_VL)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)            \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, CLIP%=                  \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX                                             \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_NEG_SLOPE_ONE(REG, ALPHA, ZERO)                                              \
    "vmflt.vf           v0, " REG ", " ZERO "                   \n\t"                           \
    "vfmul.vf           " REG ", " REG ", " ALPHA ", v0.t       \n\t"

#define MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    MMT4D_FP16_NEG_SLOPE_ONE("v16", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v17", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v18", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v19", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v20", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v21", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v22", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v23", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v24", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v25", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v26", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v27", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v28", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v29", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v30", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v31", ALPHA, ZERO)

#define MMT4D_FP16_PRELU_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                    \
    MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)

#define MMT4D_FP16_PRELU_M16N32                                                                 \
    MMT4D_FP16_PRELU_M16N32_IMPL(_ALPHA, _ZERO, _N, _TMP_VL)

#define MMT4D_FP16_LEAKYRELU_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                \
    MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)

#define MMT4D_FP16_LEAKYRELU_M16N32                                                             \
    MMT4D_FP16_LEAKYRELU_M16N32_IMPL(_ALPHA, _ZERO, _N, _TMP_VL)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)      \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, PRELU%=                 \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "PRELU%=:                                                   \n\t"                           \
    MMT4D_FP16_PRELU_M16N32                                                                     \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX                                       \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)  \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, LEAKYRELU%=             \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "LEAKYRELU%=:                                               \n\t"                           \
    MMT4D_FP16_LEAKYRELU_M16N32                                                                 \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX                                   \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)         \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    "beq                " BIAS ", zero, MINMAX%=                \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "vle16.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "MINMAX%=:                                                  \n\t"                           \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX                                          \
    MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(N, M, TMP_VL, TMP_KC)                         \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32

#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE                                                    \
    MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(_N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_MUL_SCALAR_M16N32_IMPL(FSCALAR, N, TMP_VL)                                   \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "vfmul.vf           v16, v16, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v17, v17, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v18, v18, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v19, v19, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v20, v20, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v21, v21, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v22, v22, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v23, v23, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v24, v24, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v25, v25, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v26, v26, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v27, v27, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v28, v28, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v29, v29, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v30, v30, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v31, v31, " FSCALAR "                   \n\t"
#define MMT4D_FP16_M16N32K8_POSTPROCESS_MULSCALAR                                               \
    MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(_N, _M, _TMP_VL, _TMP_KC) \
MMT4D_FP16_MUL_SCALAR_M16N32_IMPL(_FSCALAR, _N,  _TMP_VL)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE_IMPL(BIAS, C, CS, N, M, TMP_VL, TMP_KC, A0) \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, INPLACE_ADD%=           \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "INPLACE_ADD%=:                                             \n\t"                           \
    MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(C, CS, N, M, TMP_VL, TMP_KC, A0)                      \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE                                     \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE_IMPL(_BIAS, _C, _CS, _N, _M, _TMP_VL, _TMP_KC, _A0)

#define MMT4D_FP16_M16N32K8_STORE_IMPL(C, CS, M, TMP_KC)                                        \
    "bnez               " TMP_KC ", STORE_M16%=                 \n\t"                           \
    MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                       \
    "jal                x0, END%=                               \n\t"                           \
    "STORE_M16%=:                                               \n\t"                           \
    MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                           \
    "END%=:                                                     \n\t"

#define MMT4D_FP16_M16N32K8_STORE                                                               \
    MMT4D_FP16_M16N32K8_STORE_IMPL(_C, _CS, _M, _TMP_KC)

#define MMT4D_FP16_PRELOAD_M16N32_IMPL(A, A0, A1, B, B0, B1, B2, B3, TMP_VL)                    \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"

#define MMT4D_FP16_PRELOAD_M16N32                                                               \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _TMP_VL)

// M1-M4 preload the next tile; E2-E4 drain the final iteration.

#define MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v0, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v1, v8                             \n\t"                           \
    "vfwmadot           v18, v0, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v1, v9                             \n\t"                           \
    "vfwmadot           v20, v0, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v1, v10                            \n\t"                           \
    "vfwmadot           v22, v0, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v1, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M1                                                               \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M2                                                               \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M3                                                               \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M4                                                               \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                          \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E2                                                               \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(_B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                      \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E3                                                               \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(_B3)

#define MMT4D_FP16_COMP_M16N32_E4                                                               \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

#define MMT4D_FP16_UNPACK_M16N32                                                                \
    "vpack.vv           v0, v16, v18, 2                         \n\t"                           \
    "vpack.vv           v2, v20, v22, 2                         \n\t"                           \
    "vpack.vv           v4, v17, v19, 2                         \n\t"                           \
    "vpack.vv           v6, v21, v23, 2                         \n\t"                           \
    "vpack.vv           v8, v24, v26, 2                         \n\t"                           \
    "vpack.vv           v10, v28, v30, 2                        \n\t"                           \
    "vpack.vv           v12, v25, v27, 2                        \n\t"                           \
    "vpack.vv           v14, v29, v31, 2                        \n\t"                           \
    "vpack.vv           v16, v0, v2, 3                          \n\t"                           \
    "vpack.vv           v18, v1, v3, 3                          \n\t"                           \
    "vpack.vv           v20, v4, v6, 3                          \n\t"                           \
    "vpack.vv           v22, v5, v7, 3                          \n\t"                           \
    "vpack.vv           v24, v8, v10, 3                         \n\t"                           \
    "vpack.vv           v26, v9, v11, 3                         \n\t"                           \
    "vpack.vv           v28, v12, v14, 3                        \n\t"                           \
    "vpack.vv           v30, v13, v15, 3                        \n\t"

#define MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                   \
    "vfadd.vv           v16, v16, v0                            \n\t"                           \
    "vfadd.vv           v17, v17, v0                            \n\t"                           \
    "vfadd.vv           v18, v18, v0                            \n\t"                           \
    "vfadd.vv           v19, v19, v0                            \n\t"                           \
    "vfadd.vv           v20, v20, v0                            \n\t"                           \
    "vfadd.vv           v21, v21, v0                            \n\t"                           \
    "vfadd.vv           v22, v22, v0                            \n\t"                           \
    "vfadd.vv           v23, v23, v0                            \n\t"                           \
    "vfadd.vv           v24, v24, v0                            \n\t"                           \
    "vfadd.vv           v25, v25, v0                            \n\t"                           \
    "vfadd.vv           v26, v26, v0                            \n\t"                           \
    "vfadd.vv           v27, v27, v0                            \n\t"                           \
    "vfadd.vv           v28, v28, v0                            \n\t"                           \
    "vfadd.vv           v29, v29, v0                            \n\t"                           \
    "vfadd.vv           v30, v30, v0                            \n\t"                           \
    "vfadd.vv           v31, v31, v0                            \n\t"

#define MMT4D_FP16_ADD_BIAS_M16N32                                                              \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(_BIAS)

#define MMT4D_FP16_CONVERT_M16N32                                                               \
    "vfncvt.f.f.w       v16, v16                                \n\t"                           \
    "vfncvt.f.f.w       v17, v17                                \n\t"                           \
    "vfncvt.f.f.w       v18, v18                                \n\t"                           \
    "vfncvt.f.f.w       v19, v19                                \n\t"                           \
    "vfncvt.f.f.w       v20, v20                                \n\t"                           \
    "vfncvt.f.f.w       v21, v21                                \n\t"                           \
    "vfncvt.f.f.w       v22, v22                                \n\t"                           \
    "vfncvt.f.f.w       v23, v23                                \n\t"                           \
    "vfncvt.f.f.w       v24, v24                                \n\t"                           \
    "vfncvt.f.f.w       v25, v25                                \n\t"                           \
    "vfncvt.f.f.w       v26, v26                                \n\t"                           \
    "vfncvt.f.f.w       v27, v27                                \n\t"                           \
    "vfncvt.f.f.w       v28, v28                                \n\t"                           \
    "vfncvt.f.f.w       v29, v29                                \n\t"                           \
    "vfncvt.f.f.w       v30, v30                                \n\t"                           \
    "vfncvt.f.f.w       v31, v31                                \n\t"

#define MMT4D_FP16_MINMAX_M16N32_IMPL(MIN, MAX)                                                 \
    "vfmin.vf           v16, v16, " MAX "                       \n\t"                           \
    "vfmin.vf           v17, v17, " MAX "                       \n\t"                           \
    "vfmin.vf           v18, v18, " MAX "                       \n\t"                           \
    "vfmin.vf           v19, v19, " MAX "                       \n\t"                           \
    "vfmin.vf           v20, v20, " MAX "                       \n\t"                           \
    "vfmin.vf           v21, v21, " MAX "                       \n\t"                           \
    "vfmin.vf           v22, v22, " MAX "                       \n\t"                           \
    "vfmin.vf           v23, v23, " MAX "                       \n\t"                           \
    "vfmin.vf           v24, v24, " MAX "                       \n\t"                           \
    "vfmin.vf           v25, v25, " MAX "                       \n\t"                           \
    "vfmin.vf           v26, v26, " MAX "                       \n\t"                           \
    "vfmin.vf           v27, v27, " MAX "                       \n\t"                           \
    "vfmin.vf           v28, v28, " MAX "                       \n\t"                           \
    "vfmin.vf           v29, v29, " MAX "                       \n\t"                           \
    "vfmin.vf           v30, v30, " MAX "                       \n\t"                           \
    "vfmin.vf           v31, v31, " MAX "                       \n\t"                           \
    "vfmax.vf           v16, v16, " MIN "                       \n\t"                           \
    "vfmax.vf           v17, v17, " MIN "                       \n\t"                           \
    "vfmax.vf           v18, v18, " MIN "                       \n\t"                           \
    "vfmax.vf           v19, v19, " MIN "                       \n\t"                           \
    "vfmax.vf           v20, v20, " MIN "                       \n\t"                           \
    "vfmax.vf           v21, v21, " MIN "                       \n\t"                           \
    "vfmax.vf           v22, v22, " MIN "                       \n\t"                           \
    "vfmax.vf           v23, v23, " MIN "                       \n\t"                           \
    "vfmax.vf           v24, v24, " MIN "                       \n\t"                           \
    "vfmax.vf           v25, v25, " MIN "                       \n\t"                           \
    "vfmax.vf           v26, v26, " MIN "                       \n\t"                           \
    "vfmax.vf           v27, v27, " MIN "                       \n\t"                           \
    "vfmax.vf           v28, v28, " MIN "                       \n\t"                           \
    "vfmax.vf           v29, v29, " MIN "                       \n\t"                           \
    "vfmax.vf           v30, v30, " MIN "                       \n\t"                           \
    "vfmax.vf           v31, v31, " MIN "                       \n\t"

#define MMT4D_FP16_MINMAX_M16N32                                                                \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                       \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v31, (" C ")                            \n\t"

#define MMT4D_FP16_SAVING_M16                                                                   \
    MMT4D_FP16_SAVING_M16_IMPL(_C, _CS)

#define MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                   \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v31, (" C ")                            \n\t"                           \
    "jal                x0, END%=                               \n\t"

#define MMT4D_FP16_SAVING_LT16                                                                  \
    MMT4D_FP16_SAVING_LT16_IMPL(_C, _CS, _M)

#define MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(C, CS, N, M, TMP_VL, TMP_KC, A0)                  \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "addi               " A0 ", " C ", 0                        \n\t"                           \
    "addi               " TMP_KC ", " M ", 0                    \n\t"                           \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v1, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v3, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v5, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v7, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v8, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v9, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v10, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v11, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v12, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v13, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v14, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
    "vle16.v            v15, (" A0 ")                           \n\t"                           \
    "LOAD_END%=:                                                \n\t"                           \
    "vfwadd.wv          v16, v16, v0                            \n\t"                           \
    "vfwadd.wv          v17, v17, v1                            \n\t"                           \
    "vfwadd.wv          v18, v18, v2                            \n\t"                           \
    "vfwadd.wv          v19, v19, v3                            \n\t"                           \
    "vfwadd.wv          v20, v20, v4                            \n\t"                           \
    "vfwadd.wv          v21, v21, v5                            \n\t"                           \
    "vfwadd.wv          v22, v22, v6                            \n\t"                           \
    "vfwadd.wv          v23, v23, v7                            \n\t"                           \
    "vfwadd.wv          v24, v24, v8                            \n\t"                           \
    "vfwadd.wv          v25, v25, v9                            \n\t"                           \
    "vfwadd.wv          v26, v26, v10                           \n\t"                           \
    "vfwadd.wv          v27, v27, v11                           \n\t"                           \
    "vfwadd.wv          v28, v28, v12                           \n\t"                           \
    "vfwadd.wv          v29, v29, v13                           \n\t"                           \
    "vfwadd.wv          v30, v30, v14                           \n\t"                           \
    "vfwadd.wv          v31, v31, v15                           \n\t"

#define MMT4D_FP16_INPLACE_ADD_LOAD_LT16                                                        \
    MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(_C, _CS, _N, _M, _TMP_VL, _TMP_KC, _A0)

// clang-format on

#endif  // MMT4D_ASM_IME2

namespace mlir::speir::spekernel {

using ggml::cpu::riscv64_spacemit::Q41Block;
using ggml::cpu::riscv64_spacemit::_spe_fp16_;
using ggml::cpu::riscv64_spacemit::uint4x2_t;
}  // namespace mlir::speir::spekernel

namespace mlir::speir::spekernel::ime2 {

SPINE_FORCE_INLINE void _spe_pack_cast_q41_fixed8_f16_m32k32(const void * in, void * out, int32_t zero_point, int64_t blocks) {
    using block_type = Q41Block<32, 32, 2>;
    static_assert(sizeof(uint4x2_t) == 1 && sizeof(_spe_fp16_) == 2);
    if (blocks == 0) {
        return;
    }
    const auto * src = reinterpret_cast<const uint8_t *>(in);
    auto *       dst = reinterpret_cast<uint8_t *>(out);

    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        "vid.v          v1                                                       \n\t"
        "li             t0, 31                                                   \n\t"
        "vmsgtu.vx      v0, v1, t0                                               \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "LOOP_Q41_BLOCK%=:                                                       \n\t"
        "addi           t1, %[in], %[scale_offset]                               \n\t"
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        PACK_CAST_UNPACK_U4_F16_M32K32
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "beqz           %[blocks], Q41_SCALE_BLOCK%=                             \n\t"
        "addi           %[in], %[in], %[block_bytes]                             \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "Q41_SCALE_BLOCK%=:                                                      \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_CAST_LOAD_SCALE_F16_M16_LO
        PACK_CAST_SCALE_F16_M16K32_LO
        PACK_CAST_PACK_F16_M16K32_LO
        "mv             t2, %[out]                                               \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32

        PACK_CAST_LOAD_SCALE_F16_M16_HI
        PACK_CAST_SCALE_F16_M16K32_HI
        PACK_CAST_PACK_F16_M16K32_HI
        "addi           t2, %[out], 256                                          \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32
        "addi           %[out], %[out], 1024                                     \n\t"
        "addi           %[out], %[out], 1024                                     \n\t"
        "bnez           %[blocks], LOOP_Q41_BLOCK%=                              \n\t"
        : [in] "+&r"(src), [out] "+&r"(dst), [blocks] "+&r"(blocks)
        : [zero_point] "r"(zero_point), [tile_stride] "i"(32 * 8 * sizeof(_spe_fp16_)),
          [block_bytes] "i"(sizeof(block_type)), [scale_offset] "i"(offsetof(block_type, scale))
        : "cc", "memory", "t0", "t1", "t2", "t3",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
    // clang-format on
}

// Per-row uint8 zero points: subtract in I16, then convert q-zp in [-255,15] to FP16.

SPINE_FORCE_INLINE void _spe_pack_cast_q41_f16_m32k32(const void * in, void * out, int64_t blocks) {
    using block_type = Q41Block<32, 32, 2>;
    static_assert(sizeof(uint4x2_t) == 1 && sizeof(_spe_fp16_) == 2);
    if (blocks == 0) {
        return;
    }
    const auto * src = reinterpret_cast<const uint8_t *>(in);
    auto *       dst = reinterpret_cast<uint8_t *>(out);

    // FP16 scale * (q - zp) is exact in FP32 before final FP16 rounding.

    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        "vid.v          v1                                                       \n\t"
        "li             t0, 31                                                   \n\t"
        "vmsgtu.vx      v0, v1, t0                                               \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "LOOP_Q41_BLOCK%=:                                                       \n\t"
        "addi           t1, %[in], %[scale_offset]                               \n\t"
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        "vzext.vf2 v8, v4\n\t"
        "vsrl.vi v16, v8, 4\n\t"
        "vand.vi v8, v8, 15\n\t"
        "vsll.vi v16, v16, 8\n\t"
        "vor.vv v24, v8, v16\n\t"
        "vzext.vf2 v8, v24\n\t"
        "vzext.vf2 v16, v28\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "lbu t2, -32(t1)\n\t"
        "lbu t3, -31(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v8, v8, v1\n\t"
        "lbu t2, -30(t1)\n\t"
        "lbu t3, -29(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v9, v9, v1\n\t"
        "lbu t2, -28(t1)\n\t"
        "lbu t3, -27(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v10, v10, v1\n\t"
        "lbu t2, -26(t1)\n\t"
        "lbu t3, -25(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v11, v11, v1\n\t"
        "lbu t2, -24(t1)\n\t"
        "lbu t3, -23(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v12, v12, v1\n\t"
        "lbu t2, -22(t1)\n\t"
        "lbu t3, -21(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v13, v13, v1\n\t"
        "lbu t2, -20(t1)\n\t"
        "lbu t3, -19(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v14, v14, v1\n\t"
        "lbu t2, -18(t1)\n\t"
        "lbu t3, -17(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v15, v15, v1\n\t"
        "lbu t2, -16(t1)\n\t"
        "lbu t3, -15(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v16, v16, v1\n\t"
        "lbu t2, -14(t1)\n\t"
        "lbu t3, -13(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v17, v17, v1\n\t"
        "lbu t2, -12(t1)\n\t"
        "lbu t3, -11(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v18, v18, v1\n\t"
        "lbu t2, -10(t1)\n\t"
        "lbu t3, -9(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v19, v19, v1\n\t"
        "lbu t2, -8(t1)\n\t"
        "lbu t3, -7(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v20, v20, v1\n\t"
        "lbu t2, -6(t1)\n\t"
        "lbu t3, -5(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v21, v21, v1\n\t"
        "lbu t2, -4(t1)\n\t"
        "lbu t3, -3(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v22, v22, v1\n\t"
        "lbu t2, -2(t1)\n\t"
        "lbu t3, -1(t1)\n\t"
        "vmv.v.x v1, t2\n\t"
        "vmerge.vxm v1, v1, t3, v0\n\t"
        "vsub.vv v23, v23, v1\n\t"
        "vsetvli t0, zero, e16, m8, tu, mu\n\t"
        "vfcvt.f.x.v v8, v8\n\t"
        "vfcvt.f.x.v v16, v16\n\t"
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "beqz           %[blocks], Q41_SCALE_BLOCK%=                             \n\t"
        "addi           %[in], %[in], %[block_bytes]                             \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "Q41_SCALE_BLOCK%=:                                                      \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_CAST_LOAD_SCALE_F16_M16_LO
        PACK_CAST_SCALE_F16_M16K32_LO
        PACK_CAST_PACK_F16_M16K32_LO
        "mv             t2, %[out]                                               \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32

        PACK_CAST_LOAD_SCALE_F16_M16_HI
        PACK_CAST_SCALE_F16_M16K32_HI
        PACK_CAST_PACK_F16_M16K32_HI
        "addi           t2, %[out], 256                                          \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32
        "addi           %[out], %[out], 1024                                     \n\t"
        "addi           %[out], %[out], 1024                                     \n\t"
        "bnez           %[blocks], LOOP_Q41_BLOCK%=                              \n\t"
        : [in] "+&r"(src), [out] "+&r"(dst), [blocks] "+&r"(blocks)
        : [tile_stride] "i"(32 * 8 * sizeof(_spe_fp16_)),
          [block_bytes] "i"(sizeof(block_type)), [scale_offset] "i"(offsetof(block_type, scale))
        : "cc", "memory", "t0", "t1", "t2", "t3",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
    // clang-format on
}

// One K8 tile is already row-major.

SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_contiguous(const void * in, void * out, int64_t count) {
    // clang-format off
    __asm__ volatile(
        "LOOP_CONTIGUOUS%=:                                                      \n\t"
        "vsetvli        t0, %[count], e16, m4, tu, mu                            \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "sh2add         %[in], t0, %[in]                                         \n\t"
        "sub            %[count], %[count], t0                                   \n\t"
        "vfncvt.f.f.w   v8, v0                                                   \n\t"
        "vse16.v        v8, (%[out])                                             \n\t"
        "sh1add         %[out], t0, %[out]                                       \n\t"
        "bnez           %[count], LOOP_CONTIGUOUS%=                              \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [count] "+&r"(count)
        :
        : "cc", "memory", "t0");
    // clang-format on
}

// Partial or strided rows must retain the output row spacing.

SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k8(const void * in,
                                                     int64_t      in_stride,
                                                     void *       out,
                                                     int64_t      rows,
                                                     int64_t      cols) {
    if (rows < 16 || cols < 8) {
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, zero, e16, m2, tu, mu                            \n\t"
            "vxor.vv        v4, v4, v4                                           \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [out] "r"(out)
            : "cc", "memory", "t0");
        // clang-format on
    }
    if (cols == 8 && in_stride == 8 * sizeof(float)) {
        const int64_t count = rows * 8;
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, %[count], e16, m2, tu, mu                        \n\t"
            "vle32.v        v0, (%[in])                                          \n\t"
            "vfncvt.f.f.w   v4, v0                                               \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [in] "r"(in), [out] "r"(out), [count] "r"(count)
            : "cc", "memory", "t0");
        // clang-format on
        return;
    }
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "LOOP_ROW%=:                                                             \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "add            %[in], %[in], %[stride]                                  \n\t"
        "addi           %[rows], %[rows], -1                                     \n\t"
        "vfncvt.f.f.w   v2, v0                                                   \n\t"
        "vse16.v        v2, (%[out])                                             \n\t"
        "addi           %[out], %[out], 16                                       \n\t"
        "bnez           %[rows], LOOP_ROW%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [rows] "+&r"(rows)
        : [stride] "r"(in_stride), [cols] "r"(cols)
        : "cc", "memory", "t0");
    // clang-format on
}

// VLEN=1024; write 16 rows with dst_mb-row spacing between K8 tiles.

template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64(const void * in, int64_t in_stride, void * out, int64_t blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols        = 8;
    constexpr int64_t kPanelCols       = 64;
    constexpr int64_t kHalfTileRows    = 8;
    constexpr int64_t kInputPanelBytes = kPanelCols * sizeof(float);
    constexpr int64_t kHalfTileBytes   = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride      = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"
        "LOOP_K%=:                                                               \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_HI
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        PACK_CAST_STORE_F16_M16K64
        "addi           %[in], %[in], %[input_panel_bytes]                       \n\t"
        "addi           %[out], t0, %[tile_stride]                               \n\t"
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "bnez           %[blocks], LOOP_K%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [blocks] "+&r"(blocks)
        : [stride] "r"(in_stride), [tile_stride] "i"(kTileStride), [input_panel_bytes] "i"(kInputPanelBytes),
          [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t5", "t6");
    // clang-format on
}

// VLEN=1024; pack complete Rows x K64 panels into tightly packed [M/dst_mb][K/8][dst_mb][8].

template <size_t dst_mb, size_t Rows>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m8k64_rows(const void * in,
                                                          int64_t      in_stride,
                                                          void *       out,
                                                          int64_t      k_blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    static_assert(Rows == 64 || Rows == 128);
    if (k_blocks == 0) {
        return;
    }
    constexpr int64_t kBatchRows                 = 8;
    constexpr int64_t kTileCols                  = 8;
    constexpr int64_t kPanelCols                 = 64;
    constexpr int64_t kOutputElementBytes        = sizeof(_spe_fp16_);
    constexpr int64_t kInputPanelBytes           = kPanelCols * sizeof(float);
    constexpr int64_t kOutputBatchBytes          = kBatchRows * kTileCols * kOutputElementBytes;
    constexpr int64_t kOutputPanelRemainderBytes = (kPanelCols - kTileCols) * kOutputElementBytes;
    constexpr int64_t kTileStride                = dst_mb * kTileCols * kOutputElementBytes;
    int64_t           remaining                  = k_blocks * (Rows / kBatchRows);
    const int64_t     k                          = k_blocks * kPanelCols;

    const int64_t     in_col_fix                 = kInputPanelBytes - int64_t(Rows) * in_stride;
    const int64_t     out_mb_advance             = int64_t(dst_mb) * (k - kTileCols) * kOutputElementBytes;
    const int64_t     out_col_fix =
        int64_t(dst_mb) * kOutputPanelRemainderBytes - int64_t(Rows - dst_mb) * k * kOutputElementBytes;
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"

        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"

        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_LO
        "LOOP_GROUPED%=:                                                         \n\t"
        "mv             t2, %[out]                                               \n\t"
        "addi           %[remaining], %[remaining], -1                           \n\t"
        "beqz           %[remaining], PACK_GROUPED%=                             \n\t"
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        "addi           %[out], %[out], %[batch_out_bytes]                       \n\t"
        "addi           t3, t3, -1                                               \n\t"
        "addi           t4, t4, -1                                               \n\t"
        "beqz           t3, COLUMN_GROUPED%=                                     \n\t"
        "bnez           t4, LOAD_GROUPED%=                                       \n\t"
        "add            %[out], %[out], %[out_mb_advance]                        \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "j              LOAD_GROUPED%=                                           \n\t"
        "COLUMN_GROUPED%=:                                                       \n\t"
        "add            t0, t0, %[in_col_fix]                                    \n\t"
        "add            t1, t1, %[in_col_fix]                                    \n\t"
        "add            t5, t5, %[in_col_fix]                                    \n\t"
        "add            t6, t6, %[in_col_fix]                                    \n\t"
        "add            %[out], %[out], %[out_col_fix]                           \n\t"
        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"

        "LOAD_GROUPED%=:                                                         \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "PACK_GROUPED%=:                                                         \n\t"
        PACK_CAST_PACK_F16_M8K64
        PACK_CAST_STORE_F16_M8K64

        "beqz           %[remaining], END_GROUPED%=                              \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        "j              LOOP_GROUPED%=                                           \n\t"
        "END_GROUPED%=:                                                          \n\t"
        : [out] "+&r"(out), [remaining] "+&r"(remaining)
        : [in] "r"(in), [stride] "r"(in_stride), [group_segments] "i"(Rows / kBatchRows),
          [mb_segments] "i"(dst_mb / kBatchRows), [in_col_fix] "r"(in_col_fix), [out_mb_advance] "r"(out_mb_advance),
          [out_col_fix] "r"(out_col_fix), [tile_stride] "i"(kTileStride), [batch_out_bytes] "i"(kOutputBatchBytes)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
    // clang-format on
}

// Tail: rows=1..16, cols=1..64; zero-fill inactive lanes before packing.

template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64_tail(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      rows,
                                                           int64_t      cols,
                                                           int64_t      tiles) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols      = 8;
    constexpr int64_t kHalfTileRows  = 8;
    constexpr int64_t kHalfTileBytes = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride    = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        "vxor.vv        v0, v0, v0                                               \n\t"
        "vxor.vv        v8, v8, v8                                               \n\t"
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "mv             t0, %[in]                                                \n\t"
        "mv             t2, %[rows]                                              \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v0, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v1, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v2, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v3, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v4, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v5, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v6, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v7, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v8, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v9, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v10, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v11, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v12, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v13, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v14, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v15, v16                                                 \n\t"
        "PACK%=:                                                                 \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        "mv             t2, %[tiles]                                             \n\t"
        "vse16.v        v16, (t0)                                                \n\t"
        "vse16.v        v24, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v17, (t0)                                                \n\t"
        "vse16.v        v25, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v18, (t0)                                                \n\t"
        "vse16.v        v26, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v19, (t0)                                                \n\t"
        "vse16.v        v27, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v20, (t0)                                                \n\t"
        "vse16.v        v28, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v21, (t0)                                                \n\t"
        "vse16.v        v29, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v22, (t0)                                                \n\t"
        "vse16.v        v30, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v23, (t0)                                                \n\t"
        "vse16.v        v31, (t1)                                                \n\t"
        "END%=:                                                                  \n\t"
        :
        : [in] "r"(in), [stride] "r"(in_stride), [out] "r"(out), [rows] "r"(rows), [cols] "r"(cols), [tiles] "r"(tiles),
          [tile_stride] "i"(kTileStride), [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t2");
    // clang-format on
}

// Pack complete M blocks; crop/pad K to dst_kc K8 tiles.

template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_full_blocks(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      out_stride,
                                                           int64_t      mc,
                                                           int64_t      k,
                                                           int64_t      dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb         = 8;
    constexpr int64_t panel_k    = 64;
    constexpr int64_t inner_rows = 16;
    const int64_t     rows       = mc * mb;
    if (k == kb && dst_kc == 1 && in_stride == kb * sizeof(float) && out_stride == mb * kb * sizeof(_spe_fp16_)) {
        _spe_pack_cast_f32_f16_contiguous(in, out, rows * kb);
        return;
    }

    auto *  src      = reinterpret_cast<const int8_t *>(in);
    auto *  dst      = reinterpret_cast<int8_t *>(out);
    int64_t first_mo = 0;

    if (k % panel_k == 0 && dst_kc == k / kb && in_stride == k * sizeof(float) &&
        out_stride == k * mb * sizeof(_spe_fp16_)) {
        int64_t row = 0;
        for (; row + 128 <= rows; row += 128) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 128>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                       k / panel_k);
        }
        if (row + 64 <= rows) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 64>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                      k / panel_k);
            row += 64;
        }
        first_mo = row / mb;
    }

    const int64_t panels      = k / panel_k;
    const int64_t col         = panels * panel_k;
    const int64_t cols        = k - col;
    const int64_t valid_tiles = (k + kb - 1) / kb;
    for (int64_t mo = first_mo; mo < mc; ++mo) {
        auto * src_block = src + mo * mb * in_stride;
        auto * dst_block = reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride);
        // MB32 consists of two M16 halves sharing the MB32 tile stride.
        for (int64_t mi = 0; mi < mb; mi += inner_rows) {
            auto * src_half = src_block + mi * in_stride;
            auto * dst_half = dst_block + mi * kb;
            if (panels != 0) {
                _spe_pack_cast_f32_f16_m16k64<mb>(src_half, in_stride, dst_half, panels);
            }
            if (cols != 0) {
                if (cols <= kb) {
                    _spe_pack_cast_f32_f16_m16k8(src_half + col * sizeof(float), in_stride, dst_half + col * mb,
                                                 inner_rows, cols);
                } else {
                    _spe_pack_cast_f32_f16_m16k64_tail<mb>(src_half + col * sizeof(float), in_stride,
                                                           dst_half + col * mb, inner_rows, cols, (cols + kb - 1) / kb);
                }
            }
        }
        std::fill_n(dst_block + valid_tiles * mb * kb, (dst_kc - valid_tiles) * mb * kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// Successive K8 tiles are mb*8 elements apart; rows are not contiguous.

template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_row(const void * in, void * out, int64_t k, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            src = reinterpret_cast<const float *>(in);
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        const int64_t cols     = std::max<int64_t>(0, std::min<int64_t>(kb, k - ko * kb));
        auto *        dst_tile = dst + ko * mb * kb;
        if (cols != 0) {
            _spe_pack_cast_f32_f16_contiguous(src + ko * kb, dst_tile, cols);
        }
        std::fill_n(dst_tile + cols, kb - cols, static_cast<_spe_fp16_>(0.0f));
    }
}

template <size_t mb> SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_row(void * out, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        std::fill_n(dst + ko * mb * kb, kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// Zero logical output elements; preserve gaps between M blocks.
template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_blocks(void * out, int64_t out_stride, int64_t mc, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<int8_t *>(out);
    for (int64_t mo = 0; mo < mc; ++mo) {
        std::fill_n(reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride), dst_kc * mb * kb,
                    static_cast<_spe_fp16_>(0.0f));
    }
}

}  // namespace mlir::speir::spekernel::ime2

namespace mlir::speir::spekernel::ime2 {

// GEMM: m=1..16, n=1..32, kc counts K8 groups; c_stride is in bytes.
// Bias may be null; non-HP epilogues take FP32 bias.
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_minmax(size_t       m,
                                                              size_t       n,
                                                              size_t       kc,
                                                              const void * a,
                                                              const void * b,
                                                              const void * bias,
                                                              int8_t *     c,
                                                              size_t       c_stride,
                                                              _spe_fp16_   min_value,
                                                              _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_prelu_minmax(size_t       m,
                                                                    size_t       n,
                                                                    size_t       kc,
                                                                    const void * a,
                                                                    const void * b,
                                                                    const void * bias,
                                                                    int8_t *     c,
                                                                    size_t       c_stride,
                                                                    float        alpha,
                                                                    _spe_fp16_   min_value,
                                                                    _spe_fp16_   max_value) {
    const float zero_value = 0.0f;
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [alpha] "f"(alpha),
          [zero] "f"(zero_value), [min] "f"(min_value), [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_leakyrelu_minmax(size_t       m,
                                                                        size_t       n,
                                                                        size_t       kc,
                                                                        const void * a,
                                                                        const void * b,
                                                                        const void * bias,
                                                                        int8_t *     c,
                                                                        size_t       c_stride,
                                                                        float        alpha,
                                                                        _spe_fp16_   min_value,
                                                                        _spe_fp16_   max_value) {
    const float zero_value = 0.0f;
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [alpha] "f"(alpha),
          [zero] "f"(zero_value), [min] "f"(min_value), [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_none(size_t       m,
                                                       size_t       n,
                                                       size_t       kc,
                                                       const void * a,
                                                       const void * b,
                                                       int8_t *     c,
                                                       size_t       c_stride) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_NONE

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [kc] "r"(kc), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_hp_bias_minmax(size_t       m,
                                                                 size_t       n,
                                                                 size_t       kc,
                                                                 const void * a,
                                                                 const void * b,
                                                                 const void * bias,
                                                                 int8_t *     c,
                                                                 size_t       c_stride,
                                                                 _spe_fp16_   min_value,
                                                                 _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_minmax_inplace_add(size_t       m,
                                                                          size_t       n,
                                                                          size_t       kc,
                                                                          const void * a,
                                                                          const void * b,
                                                                          const void * bias,
                                                                          int8_t *     c,
                                                                          size_t       c_stride,
                                                                          _spe_fp16_   min_value,
                                                                          _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_mulscalar(size_t       m,
                                                            size_t       n,
                                                            size_t       kc,
                                                            const void * a,
                                                            const void * b,
                                                            int8_t *     c,
                                                            size_t       c_stride,
                                                            _spe_fp16_   f_scalar) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_MULSCALAR

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [kc] "r"(kc), [cs] "r"(c_stride), [fscalar] "f"(f_scalar)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

}  // namespace mlir::speir::spekernel::ime2

namespace mlir::speir::spekernel::rvv {

// Split stride stays 32 FP16 elements even when vl < 32.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1(const void * in, void * out, int64_t n_size, int64_t nth) {

    constexpr int64_t src_nb = 32;
    __asm__ volatile(
        "L_N%=:                                               \n\t"
        "vsetvli                t0, %[nsize], e32, m1, tu, mu \n\t"
        "vxor.vv                v8, v8, v8                    \n\t"
        "vsetvli                zero, t0, e16, mf2, tu, mu    \n\t"
        "mv                     t1, %[nth]                    \n\t"
        "mv                     a1, %[in]                     \n\t"

        "L_THREADS%=:                                         \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "vfwadd.wv              v8, v8, v0                    \n\t"
        "sh1add                 a1, %[snb], a1                \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_THREADS%=               \n\t"

        "vsetvli                zero, t0, e32, m1, tu, mu     \n\t"
        "vse32.v                v8, (%[out])                  \n\t"
        "sh1add                 %[in], t0, %[in]              \n\t"
        "sh2add                 %[out], t0, %[out]            \n\t"
        "sub                    %[nsize], %[nsize], t0        \n\t"
        "bnez                   %[nsize], L_N%=               \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [nsize] "+&r"(n_size)
        : [nth] "r"(nth), [snb] "r"(src_nb)
        : "cc", "memory", "t0", "t1", "a1");
}

// VLEN=1024; input=[16,8,32], out_stride is in bytes; preserve t=0..7 summation order.
// Widen all inputs before reuse and restore VL=32 before clearing registers.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1_a100_nth8(const void * in,
                                                                       void *       out,
                                                                       int64_t      out_stride) {
    constexpr int64_t src_nb       = 32;
    constexpr int64_t load_nb      = 2 * src_nb;
    constexpr int64_t row_elements = 8 * src_nb;
    __asm__ volatile(
        "li                     t1, 8                         \n\t"
        "L_PAIR%=:                                            \n\t"
        "mv                     a1, %[in]                     \n\t"
        "sh1add                 a2, %[row_elements], %[in]    \n\t"
        "vsetvli                zero, %[lnb], e16, m1, tu, mu \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v4, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v1, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v5, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v2, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v6, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v3, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v7, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vfwcvt.f.f.v           v16, v0                       \n\t"
        "vfwcvt.f.f.v           v24, v4                       \n\t"
        "vfwcvt.f.f.v           v18, v1                       \n\t"
        "vfwcvt.f.f.v           v26, v5                       \n\t"
        "vfwcvt.f.f.v           v20, v2                       \n\t"
        "vfwcvt.f.f.v           v28, v6                       \n\t"
        "vfwcvt.f.f.v           v22, v3                       \n\t"
        "vfwcvt.f.f.v           v30, v7                       \n\t"
        "vsetvli                zero, %[snb], e32, m1, tu, mu \n\t"
        "vxor.vv                v0, v0, v0                    \n\t"
        "vxor.vv                v1, v1, v1                    \n\t"
        "vfadd.vv               v0, v0, v16                   \n\t"
        "vfadd.vv               v1, v1, v24                   \n\t"
        "vfadd.vv               v0, v0, v17                   \n\t"
        "vfadd.vv               v1, v1, v25                   \n\t"
        "vfadd.vv               v0, v0, v18                   \n\t"
        "vfadd.vv               v1, v1, v26                   \n\t"
        "vfadd.vv               v0, v0, v19                   \n\t"
        "vfadd.vv               v1, v1, v27                   \n\t"
        "vfadd.vv               v0, v0, v20                   \n\t"
        "vfadd.vv               v1, v1, v28                   \n\t"
        "vfadd.vv               v0, v0, v21                   \n\t"
        "vfadd.vv               v1, v1, v29                   \n\t"
        "vfadd.vv               v0, v0, v22                   \n\t"
        "vfadd.vv               v1, v1, v30                   \n\t"
        "vfadd.vv               v0, v0, v23                   \n\t"
        "vfadd.vv               v1, v1, v31                   \n\t"
        "vse32.v                v0, (%[out])                  \n\t"
        "add                    a1, %[out], %[os]             \n\t"
        "vse32.v                v1, (a1)                      \n\t"
        "sh1add                 %[out], %[os], %[out]         \n\t"
        "mv                     %[in], a2                     \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_PAIR%=                  \n\t"
        : [in] "+&r"(in), [out] "+&r"(out)
        : [snb] "r"(src_nb), [lnb] "r"(load_nb), [row_elements] "r"(row_elements), [os] "r"(out_stride)
        : "cc", "memory", "t1", "a1", "a2");
}

}  // namespace mlir::speir::spekernel::rvv
