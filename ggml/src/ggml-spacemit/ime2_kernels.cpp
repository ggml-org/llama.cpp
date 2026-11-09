#include "ggml-impl.h"
#include "ggml.h"
#include "ime_kernels.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "string.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#if defined(RISCV64_SPACEMIT_IME2)
#else
#    error "RISCV64_SPACEMIT_IME2 not defined"
#endif

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace spacemit_kernels {
namespace ime2 {

void gemm_kernel_i8i2k_m1(const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_n,
                          size_t          k_blks) {
    constexpr size_t NB_COLS = 32;
    using blk_type           = nrow_block_q2_k<NB_COLS>;

    int64_t b_ncol_block_stride = sizeof(blk_type) * k_blks;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_ncol_block_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = (float *) c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             s1, %[BK]               \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            // load scale A
            "flw            fa0, (%[A])             \n\t"
            "addi           %[A], %[A], 4           \n\t"

            "li             t1, 4                   \n\t"
            "addi           t2, %[B], 512           \n\t"  // B data addr
            "addi           t3, %[A], 32            \n\t"  // A data addr
            "addi           s3, %[B], 0             \n\t"
            "vxor.vv        v30, v29, v29           \n\t"  // tmp result

            "INNER_K_LOOP%=:                        \n\t"
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vxor.vv        v2, v2, v2              \n\t"
            "vxor.vv        v3, v3, v3              \n\t"
            "vxor.vv        v4, v4, v4              \n\t"
            "vxor.vv        v5, v5, v5              \n\t"
            "vxor.vv        v6, v6, v6              \n\t"
            "vxor.vv        v28, v28, v28           \n\t"
            "vxor.vv        v29, v29, v29           \n\t"

            // load scale  B
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vle8.v         v0, (%[B])              \n\t"
            "addi           %[B], %[B], 128         \n\t"

            // A data, 1x64@i8
            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v2, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v4, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v5, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v6, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetvli        t0, x0, e64, mf2        \n\t"
            "vslideup.vi    v3, v4, 2               \n\t"
            "vslideup.vi    v28, v5, 4              \n\t"
            "vslideup.vi    v29, v6, 6              \n\t"

            // init the accumu to zero
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v20, v18, v18           \n\t"
            "vxor.vv        v22, v18, v18           \n\t"
            "vxor.vv        v24, v18, v18           \n\t"
            "vxor.vv        v26, v18, v18           \n\t"

            // B data, 32x64@i2
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vl4r.v         v4, (t2)                \n\t"
            "addi           t2, t2, 512             \n\t"
            "vand.vi        v8, v4, 0x3             \n\t"  // 0-15
            "vsrl.vi        v9, v4, 2               \n\t"
            "vsrl.vi        v10, v4, 4              \n\t"
            "vsrl.vi        v11, v4, 6              \n\t"  // 48-63
            "vand.vi        v9, v9, 0x3             \n\t"  // 16-31
            "vand.vi        v10, v10, 0x3           \n\t"  // 32-47

            "vand.vi        v12, v5, 0x3            \n\t"  // 0-15
            "vsrl.vi        v13, v5, 2              \n\t"
            "vsrl.vi        v14, v5, 4              \n\t"
            "vsrl.vi        v15, v5, 6              \n\t"  // 48-63
            "vand.vi        v13, v13, 0x3           \n\t"  // 16-31
            "vand.vi        v14, v14, 0x3           \n\t"  // 32-47

            "vand.vi        v16, v6, 0x3            \n\t"  // 0-15
            "vsrl.vi        v17, v6, 2              \n\t"
            "vsrl.vi        v18, v6, 4              \n\t"
            "vsrl.vi        v19, v6, 6              \n\t"  // 48-63
            "vand.vi        v17, v17, 0x3           \n\t"  // 16-31
            "vand.vi        v18, v18, 0x3           \n\t"  // 32-47

            "vand.vi        v4, v7, 0x3             \n\t"  // 0-15
            "vsrl.vi        v5, v7, 2               \n\t"
            "vsrl.vi        v6, v7, 4               \n\t"
            "vsrl.vi        v7, v7, 6               \n\t"  // 48-63
            "vand.vi        v5, v5, 0x3             \n\t"  // 16-31
            "vand.vi        v6, v6, 0x3             \n\t"  // 32-47

            // i2 * i8 vmadot
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vmadotsu       v20, v2, v8, i8         \n\t"
            "vmadotsu       v22, v2, v12, i8        \n\t"
            "vmadotsu       v24, v2, v16, i8        \n\t"
            "vmadotsu       v26, v2, v4, i8         \n\t"

            "vmadotsu       v20, v3, v9, i8         \n\t"
            "vmadotsu       v22, v3, v13, i8        \n\t"
            "vmadotsu       v24, v3, v17, i8        \n\t"
            "vmadotsu       v26, v3, v5, i8         \n\t"

            "vmadotsu       v20, v28, v10, i8       \n\t"
            "vmadotsu       v22, v28, v14, i8       \n\t"
            "vmadotsu       v24, v28, v18, i8       \n\t"
            "vmadotsu       v26, v28, v6, i8        \n\t"

            "vmadotsu       v20, v29, v11, i8       \n\t"
            "vmadotsu       v22, v29, v15, i8       \n\t"
            "vmadotsu       v24, v29, v19, i8       \n\t"
            "vmadotsu       v26, v29, v7, i8        \n\t"

            "vand.vi        v10, v0, 0xf            \n\t"  // scale
            "vwadd.vx       v12, v10, x0            \n\t"
            "vsetvli        t0, x0, e16, m2         \n\t"
            "vwadd.vx       v16, v12, x0            \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vpack.vv       v2, v20, v22, 2         \n\t"
            "vpack.vv       v4, v24, v26, 2         \n\t"
            "vpack.vv       v6, v2, v4, 3           \n\t"  // 0,1
            "vpack.vv       v8, v3, v5, 3           \n\t"  // 2,3

            // mul scale
            "vmacc.vv       v30, v6, v16            \n\t"
            "vmacc.vv       v30, v7, v17            \n\t"
            "vmacc.vv       v30, v8, v18            \n\t"
            "vmacc.vv       v30, v9, v19            \n\t"

            "addi           t1, t1, -1              \n\t"
            "bgtz           t1, INNER_K_LOOP%=      \n\t"

            // load zp B
            "vsetvli        t0, x0, e8, m4          \n\t"
            "vle8.v         v4, (s3)                \n\t"
            "vsrl.vi        v8, v4, 4               \n\t"  // zp

            // asum * zp
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v20, v20, v20           \n\t"
            "vxor.vv        v22, v22, v22           \n\t"
            "vxor.vv        v24, v24, v24           \n\t"
            "vxor.vv        v26, v26, v26           \n\t"

            "vsetvli        t0, x0, e16, mf4        \n\t"
            "vle16.v        v2, (%[A])              \n\t"
            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vnsrl.wi       v12, v2, 0              \n\t"  // low 8
            "vnsra.wi       v13, v2, 8              \n\t"  // high 8

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vmadotsu       v20, v13, v8, i8        \n\t"
            "vmadotsu       v22, v13, v9, i8        \n\t"
            "vmadotsu       v24, v13, v10, i8       \n\t"
            "vmadotsu       v26, v13, v11, i8       \n\t"

            "vsll.vi        v20, v20, 8             \n\t"
            "vsll.vi        v22, v22, 8             \n\t"
            "vsll.vi        v24, v24, 8             \n\t"
            "vsll.vi        v26, v26, 8             \n\t"

            "vmadotu        v20, v12, v8, i8        \n\t"
            "vmadotu        v22, v12, v9, i8        \n\t"
            "vmadotu        v24, v12, v10, i8       \n\t"
            "vmadotu        v26, v12, v11, i8       \n\t"

            "vpack.vv       v2, v20, v22, 2         \n\t"
            "vpack.vv       v4, v24, v26, 2         \n\t"
            "vpack.vv       v28, v2, v4, 3          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v0, (t2)                \n\t"  // scale16
            "addi           t2, t2, 64              \n\t"
            "vle16.v        v1, (t2)                \n\t"  // zero16
            "vfwcvt.f.f.v   v2, v0                  \n\t"
            "vfwcvt.f.f.v   v4, v1                  \n\t"
            "vsetvli        t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v    v30, v30                \n\t"
            "vfcvt.f.x.v    v28, v28                \n\t"
            "addi           %[B], t2, 64            \n\t"
            "mv             %[A], t3                \n\t"

            "vfmul.vv       v30, v30, v2            \n\t"  // mul scale16
            "vfmacc.vv      v30, v28, v4            \n\t"  // + mul zero16
            "vfmacc.vf      v31, fa0, v30           \n\t"
            "addi           s1, s1, -1              \n\t"
            "bgtz           s1, BLK_LOOP%=          \n\t"

            // save
            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "t4", "t5", "t6", "s1", "s2", "s3");
    }
}

void gemm_kernel_i8i2k_m4(const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    constexpr size_t NB_COLS = 32;
    using blk_type           = nrow_block_q2_k<NB_COLS>;

    int64_t  b_ncol_block_stride = sizeof(blk_type) * k_blks;
    _Float16 scale               = 0.0625f;
    _Float16 scale_1             = 16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_ncol_block_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = (float *) c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v28, v31, v31           \n\t"  // init result
            "vxor.vv        v29, v31, v31           \n\t"
            "vxor.vv        v30, v31, v31           \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             s1, %[BK]               \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            // load scale A
            "flw            fa0, (%[A])             \n\t"
            "flw            fa1, 4(%[A])            \n\t"
            "flw            fa2, 8(%[A])            \n\t"
            "flw            fa3, 12(%[A])           \n\t"
            "addi           %[A], %[A], 16          \n\t"

            "li             t1, 4                   \n\t"
            "addi           t2, %[B], 512           \n\t"  // B data addr
            "addi           t3, %[A], 128           \n\t"  // A data addr
            "addi           s4, t2, 1024            \n\t"  // scale16 addr
            "addi           s4, s4, 1024            \n\t"  // TODO
            "addi           s3, %[B], 0             \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v1, (s4)                \n\t"  // load scale16
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vpack.vv       v22, v1, v1, 3          \n\t"

            "addi           s4, t3, 256             \n\t"  // addr 1
            "addi           s5, t3, 512             \n\t"  // addr 2
            "addi           s6, t3, 768             \n\t"  // addr 3

            // init the accu to 0
            "vxor.vv        v24, v24, v24           \n\t"
            "vxor.vv        v25, v25, v25           \n\t"
            "vxor.vv        v26, v26, v26           \n\t"
            "vxor.vv        v27, v27, v27           \n\t"

            "INNER_K_LOOP%=:                        \n\t"
            // load scale  B
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vle8.v         v1, (%[B])              \n\t"
            "addi           %[B], %[B], 128         \n\t"
            "vand.vi        v1, v1, 0xf             \n\t"

            "vfwcvt.f.x.v   v20, v1                 \n\t"  // f16 scale B
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vfmul.vv       v0, v20, v22            \n\t"  // mul scale16
            "vfmul.vv       v1, v21, v22            \n\t"  // mul scale16
            "vfmul.vf       v0, v0, %[SCALE]        \n\t"  // mul magic
            "vfmul.vf       v1, v1, %[SCALE]        \n\t"  // mul magic

            // A data, 4x64@i8
            "vsetvli        t0, x0, e8, mf2         \n\t"
            "vle8.v         v2, (t3)                \n\t"
            "addi           t3, t3, 64              \n\t"
            "vle8.v         v3, (s4)                \n\t"
            "addi           s4, s4, 64              \n\t"
            "vle8.v         v4, (s5)                \n\t"
            "addi           s5, s5, 64              \n\t"
            "vle8.v         v5, (s6)                \n\t"
            "addi           s6, s6, 64              \n\t"

            // 4x64 => 4x16x4
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v6, v2, v3, 1           \n\t"
            "vpack.vv       v8, v4, v5, 1           \n\t"
            "vpack.vv       v2, v6, v8, 2           \n\t"  // 0, 2

            "vpack.vv       v20, v2, v2, 3          \n\t"  // 1
            "vor.vv         v23, v21, v21           \n\t"
            "vpack.vv       v20, v3, v3, 3          \n\t"  // 3

            // B data, 32x64@i2
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vl4r.v         v4, (t2)                \n\t"
            "addi           t2, t2, 512             \n\t"
            "vand.vi        v8, v4, 0x3             \n\t"  // 0-15
            "vsrl.vi        v9, v4, 2               \n\t"
            "vsrl.vi        v10, v4, 4              \n\t"
            "vsrl.vi        v11, v4, 6              \n\t"  // 48-63
            "vand.vi        v9, v9, 0x3             \n\t"  // 16-31
            "vand.vi        v10, v10, 0x3           \n\t"  // 32-47

            "vand.vi        v12, v5, 0x3            \n\t"  // 0-15
            "vsrl.vi        v13, v5, 2              \n\t"
            "vsrl.vi        v14, v5, 4              \n\t"
            "vsrl.vi        v15, v5, 6              \n\t"  // 48-63
            "vand.vi        v13, v13, 0x3           \n\t"  // 16-31
            "vand.vi        v14, v14, 0x3           \n\t"  // 32-47

            "vand.vi        v16, v6, 0x3            \n\t"  // 0-15
            "vsrl.vi        v17, v6, 2              \n\t"
            "vsrl.vi        v18, v6, 4              \n\t"
            "vsrl.vi        v19, v6, 6              \n\t"  // 48-63
            "vand.vi        v17, v17, 0x3           \n\t"  // 16-31
            "vand.vi        v18, v18, 0x3           \n\t"  // 32-47

            "vand.vi        v4, v7, 0x3             \n\t"  // 0-15
            "vsrl.vi        v5, v7, 2               \n\t"
            "vsrl.vi        v6, v7, 4               \n\t"
            "vsrl.vi        v7, v7, 6               \n\t"  // 48-63
            "vand.vi        v5, v5, 0x3             \n\t"  // 16-31
            "vand.vi        v6, v6, 0x3             \n\t"  // 32-47

            // i2 * i8 vmadot
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vmadotsu.hp    v24, v2, v8, v0, 0, i8  \n\t"
            "vmadotsu.hp    v25, v2, v12, v0, 1, i8 \n\t"
            "vmadotsu.hp    v26, v2, v16, v0, 2, i8 \n\t"
            "vmadotsu.hp    v27, v2, v4, v0, 3, i8  \n\t"

            "vmadotsu.hp    v24, v23, v9, v0, 4, i8 \n\t"
            "vmadotsu.hp    v25, v23, v13, v0, 5, i8\n\t"
            "vmadotsu.hp    v26, v23, v17, v0, 6, i8\n\t"
            "vmadotsu.hp    v27, v23, v5, v0, 7, i8 \n\t"

            "vmadotsu.hp    v24, v3, v10, v1, 0, i8 \n\t"
            "vmadotsu.hp    v25, v3, v14, v1, 1, i8 \n\t"
            "vmadotsu.hp    v26, v3, v18, v1, 2, i8 \n\t"
            "vmadotsu.hp    v27, v3, v6, v1, 3, i8  \n\t"

            "vmadotsu.hp    v24, v21, v11, v1, 4, i8\n\t"
            "vmadotsu.hp    v25, v21, v15, v1, 5, i8\n\t"
            "vmadotsu.hp    v26, v21, v19, v1, 6, i8\n\t"
            "vmadotsu.hp    v27, v21, v7, v1, 7, i8 \n\t"

            "addi           t1, t1, -1              \n\t"
            "bgtz           t1, INNER_K_LOOP%=      \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vpack.vv       v2, v24, v25, 1         \n\t"
            "vpack.vv       v4, v26, v27, 1         \n\t"
            "vpack.vv       v6, v2, v4, 2           \n\t"  // 0,1,2,3

            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v20, v20, v20           \n\t"
            "vxor.vv        v22, v22, v22           \n\t"
            "vxor.vv        v24, v24, v24           \n\t"
            // load zp B, 16x8x4@int4
            "vsetvli        t0, x0, e8, m4          \n\t"
            "vle8.v         v0, (s3)                \n\t"
            "vsrl.vi        v0, v0, 4               \n\t"  // zp

            // 4x16@int16
            "vsetvli        t0, x0, e16, m1         \n\t"  // a sum
            "vle16.v         v12, (%[A])             \n\t"
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vnsrl.wi       v10, v12, 0             \n\t"  // low 8
            "vnsra.wi       v11, v12, 8             \n\t"  // high 8

            // asum * zp
            "vsetvli        t0, x0, e32, m1          \n\t"
            "vmadotsu       v18, v11, v0, i8        \n\t"
            "vmadotsu       v20, v11, v1, i8        \n\t"
            "vmadotsu       v22, v11, v2, i8        \n\t"
            "vmadotsu       v24, v11, v3, i8        \n\t"
            "vsll.vi        v18, v18, 8             \n\t"
            "vsll.vi        v20, v20, 8             \n\t"
            "vsll.vi        v22, v22, 8             \n\t"
            "vsll.vi        v24, v24, 8             \n\t"
            "vmadotu        v18, v10, v0, i8        \n\t"
            "vmadotu        v20, v10, v1, i8        \n\t"
            "vmadotu        v22, v10, v2, i8        \n\t"
            "vmadotu        v24, v10, v3, i8        \n\t"

            "vpack.vv       v10, v18, v20, 2        \n\t"
            "vpack.vv       v12, v22, v24, 2        \n\t"
            "vpack.vv       v14, v10, v12, 3        \n\t"
            "vpack.vv       v16, v11, v13, 3        \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t2, t2, 64              \n\t"
            "vle16.v        v20, (t2)               \n\t"  // zero16
            "vfwcvt.f.f.v   v22, v20                \n\t"

            // mul 1/magic
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vfwmul.vf      v0, v6, %[SCALE_1]      \n\t"
            "vfwmul.vf      v2, v7, %[SCALE_1]      \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v    v14, v14                \n\t"
            "vfcvt.f.x.v    v15, v15                \n\t"
            "vfcvt.f.x.v    v16, v16                \n\t"
            "vfcvt.f.x.v    v17, v17                \n\t"

            "addi           %[B], t2, 64            \n\t"
            "mv             %[A], s6                \n\t"

            "vfmacc.vv      v0, v14, v22            \n\t"  // + mul zero16
            "vfmacc.vv      v1, v15, v22            \n\t"
            "vfmacc.vv      v2, v16, v22            \n\t"
            "vfmacc.vv      v3, v17, v22            \n\t"

            "vfmacc.vf      v28, fa0, v0            \n\t"  // mul a scale
            "vfmacc.vf      v29, fa1, v1            \n\t"
            "vfmacc.vf      v30, fa2, v2            \n\t"
            "vfmacc.vf      v31, fa3, v3            \n\t"

            "addi           s1, s1, -1              \n\t"
            "bgtz           s1, BLK_LOOP%=          \n\t"

            // save
            "vsetvli        t0, x0, e32, m1         \n\t"
            "add            t1, %[LDC], %[DST]      \n\t"
            "vse32.v        v28, (%[DST])           \n\t"
            "vse32.v        v29, (t1)               \n\t"
            "add            t1, t1, %[LDC]          \n\t"
            "vse32.v        v30, (t1)               \n\t"
            "add            t1, t1, %[LDC]          \n\t"
            "vse32.v        v31, (t1)               \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks), [LDC] "r"(ldc * 4), [SCALE] "f"(scale), [SCALE_1] "f"(scale_1)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "s5", "s6");
    }
}

void gemm_kernel_i8i3k_m1(const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_n,
                          size_t          k_blks) {
    constexpr size_t NB_COLS = 32;  //only support 32 in ASM
    using blk_type           = nrow_block_q3_k<NB_COLS>;

    const blk_type * b_base = reinterpret_cast<const blk_type *>(quant_b_data);

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    // Constants used by q3_k scaling in HP branch:
    // - k_q3k_scale_step: per-nibble scale factor (1/16).
    // - k_a_scale_post_mul: A_scale needs an extra *16 at the end (pairs with 1/16 above).
    const _Float16 k_q3k_scale_step   = (_Float16) 0.0625f;  // 1 / 16
    const float    k_a_scale_post_mul = 16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t           nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
        const blk_type * quant_b_blk_data = b_base + (ni / NB_COLS) * k_blks;
#if 0
        //------------------------------------------------------------------------------
        // A format
        // Ascale   fp32 * 1    32bit
        // Asum     int16 * 16  256bit
        // A M1K256 int8        2048bit
        //------------------------------------------------------------------------------
        // B format
        // B_scl    uint8*N32*16    4096bit
        // B_Hmask  N32K16*16 1bit  8192bit
        // B_Qs     N32K16*16 2bit  16384bit
        // B scl16  fp16 * N32      512bit;
        //------------------------------------------------------------------------------
        //bias always be nullptr
        __asm__ volatile(
            // t2 = k_blks (each is K256 superblock)
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = 256/64 = 4 (K64 iterations per superblock)
            "li           t3, 4                   \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 4+32         \n\t"  // s3 = pAData, (pA+AScl+ASum)

            // B block layout for nrow_block_q3_k<32>:
            // scales: 512B, hmask: 1024B, qs: 2048B, scales16: 64B
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v31, v0, v0             \n\t"  // clear acc
            "vxor.vv      v30, v0, v0             \n\t"  // clear acc of K256

            // ordinary vmadot: vle*10 vecIns*78 vmadot*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"
            "K64_LPST%=:                          \n\t"

            // K0-15
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v2, (s4)                \n\t"
            "addi         s4, s4, 128             \n\t"

            // load B qs chunk (128B per K16, 16 times => 2048B)
            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v1, (s3)                \n\t"
            "addi         s3, s3, 64              \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"  // N0-N31 in v16

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19           \n\t"

            //K16-31
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 4               \n\t"
            "vsll.vi      v9, v5, 4               \n\t"
            "vsll.vi      v10, v6, 4              \n\t"
            "vsll.vi      v11, v7, 4              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v8, 6              \n\t"
            "vsrl.vi      v13, v9, 6              \n\t"
            "vsrl.vi      v14, v10, 6             \n\t"
            "vsrl.vi      v15, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"  // N0-N31 in v16

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0              \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0              \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19            \n\t"

            //K32-47
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 2               \n\t"
            "vsll.vi      v9, v5, 2               \n\t"
            "vsll.vi      v10, v6, 2              \n\t"
            "vsll.vi      v11, v7, 2              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v8, 6              \n\t"
            "vsrl.vi      v13, v9, 6              \n\t"
            "vsrl.vi      v14, v10, 6             \n\t"
            "vsrl.vi      v15, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0              \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0              \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19            \n\t"

            // K48-63
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v12, v4, 6              \n\t"
            "vsrl.vi      v13, v5, 6              \n\t"
            "vsrl.vi      v14, v6, 6              \n\t"
            "vsrl.vi      v15, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19           \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // load A scale (fp32) and advance A to next superblock
            "flw          f0, (s2)                \n\t"
            "addi         s2, s2, 4+32+256        \n\t"
            "add          t4, s7, %[B_STR]        \n\t"  // t4 = next B blk base
            "addi         s3, s2, 4+32            \n\t"

            // load B scales16[32] (fp16) at end of qs region
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v2, (s6)                \n\t"

            // pointer modify
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s7, t4, 0               \n\t"

            // b_scale fp16 -> fp32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v24, v2                 \n\t"

            // a_scale * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v v26, v30                 \n\t"
            "vfmul.vf     v1, v24, f0             \n\t"
            "vsetvli      t0, x0, e32, m1         \n\t"
            // static_cast<float>(qsum) * a_scale * b_scale;
            "vfmacc.vv    v31, v1, v26            \n\t"

            // next K-superblock
            "addi         t2, t2, -1              \n\t"
            "vxor.vv      v30, v0, v0             \n\t"  // clear acc of K256
            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v31, (%[pC])            \n\t"
            "FUNC_END%=:                          \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "s2", "s3", "s4", "s5", "s6", "s7");
#else

        __asm__ volatile(
            // =========================
            // Kernel overview (M1 x N32)
            // =========================
            // Process one output row (M=1) and 32 columns (N=32) per call.
            //
            // Loop structure:
            //   - Outer loop: K superblocks of size K=256 (k_blks times)
            //   - Each K256 superblock is broken into 4 x K64
            //   - Each K64 is processed as 4 x K16 "sub-blocks" (via unpack+dot)
            //
            // Data layout (high level):
            //   A (q8k K=256, per superblock):
            //     [ fp32 a_scale ][ int16 a_sum[16] ][ int8 a_qs[256] ]
            //   B (nrow_block_q3_k<32>, per superblock):
            //     [ int8  scales[32*16] ][ hmask[1024] ][ qs[2048] ][ fp16 scales16[32] ]
            //
            // Registers/pointers:
            //   s2: pA (points at A superblock header; used to load fp32 a_scale)
            //   s3: pA_qs (points at A int8 data within the current superblock)
            //   s4: pB_scales (points at B int8 per-K16 scales)
            //   s5: pB_hmask (points at B sign mask area)
            //   s6: pB_qs (points at B 2-bit packed qs area)
            //   s8: pB_scales16 (points at B fp16 scales16[32] at the end of block)
            //   s7: pB_base (base pointer to current B block; used for block-to-block stride)

            // t2 = number of K256 superblocks
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = number of K64 chunks per K256 superblock (256 / 64)
            "li           t3, 4                   \n\t"

            // A pointers
            "mv           s2, %[pA]               \n\t"  // s2 = pA_superblock (a_scale at +0)
            "addi         s3, %[pA], 4+32         \n\t"  // s3 = pA_qs (skip a_scale + a_sum[16])

            // B pointers for nrow_block_q3_k<32>
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask  (skip scales[32*16])
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs     (skip hmask)
            // scales16 is at the end of the block: qs(2048) after hmask
            "addi         s8, s6, 1024            \n\t"
            "addi         s8, s8, 1024            \n\t"  // s8 = pB_scales16 (fp16 scales16[32])
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base (for next-block address calc)

            // v31: final FP32 accumulator for N=32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v31, v0, v0             \n\t"

            // ---- Preload B scales16[32] and build FP16 scale vector used by vmadot.hp ----
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v1, (s8)                \n\t"  // load fp16 scales16[32]
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v26, v1, v1, 3          \n\t"  // broadcast/pack to match lanes
            "vmv.v.v      v17, v26                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vf     v30, v17, %[q3_step]    \n\t"  // v30 = scales16 * (1/16)

            // v24-v27: fp16 partial accumulators for a K64 chunk (vmadot.hp outputs)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v16, v16           \n\t"
            "vxor.vv      v25, v16, v16           \n\t"
            "vxor.vv      v26, v16, v16           \n\t"
            "vxor.vv      v27, v16, v16           \n\t"

            // HP vmadot: vle*10 vecIns*38 vmadot.hp*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"  // loop over K256 superblocks
            "K64_LPST%=:                          \n\t"  // loop over 4 x K64 chunks

            // ------------------------------------------------------------
            // K0-15: load B scales + {hmask, qs} + A data; unpack and dot
            // ------------------------------------------------------------
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v2, (s4)                \n\t"  // B int8 scales for this K16
            "addi         s4, s4, 128             \n\t"

            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"  // B hmask for this K16
            "addi         s5, s5, 64              \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v3, (s3)                \n\t"  // A int8 data for this K16
            "addi         s3, s3, 64              \n\t"

            // Convert B int8 scales to FP16 and apply scales16*(1/16)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vfwcvt.f.x.v v28, v2                 \n\t"  // int8 -> fp16
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vv     v1, v28, v30            \n\t"  // v1: FP16 scale vector for vmadot.hp
            "vfmul.vv     v29, v29, v30           \n\t"

            // Unpack B 2-bit qs + hmask -> signed int8 in v12..v15
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"
            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            // (Next K16 unpack path uses a fresh hmask load)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // Prepare another group from packed qs (bit shifts) + apply sign from hmask
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 4               \n\t"
            "vsll.vi      v9, v5, 4               \n\t"
            "vsll.vi      v10, v6, 4              \n\t"
            "vsll.vi      v11, v7, 4              \n\t"
            "vsrl.vi      v16, v8, 6              \n\t"
            "vsrl.vi      v17, v9, 6              \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v18, v10, 6             \n\t"
            "vsrl.vi      v19, v11, 6             \n\t"
            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v16, v16, -4, v0.t      \n\t"

            // A shift for the second dot within this K64
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v2, v3, 2             \n\t"

            // Dot products with FP16 scaling (accumulate into v24..v27)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmadot.hp    v24, v3, v12, v1, 0, i8 \n\t"
            "vmadot.hp    v25, v3, v13, v1, 1, i8 \n\t"
            "vmadot.hp    v26, v3, v14, v1, 2, i8 \n\t"
            "vmadot.hp    v27, v3, v15, v1, 3, i8 \n\t"
            "vmadot.hp    v24, v2, v16, v1, 4, i8 \n\t"
            "vmadot.hp    v25, v2, v17, v1, 5, i8 \n\t"
            "vmadot.hp    v26, v2, v18, v1, 6, i8 \n\t"
            "vmadot.hp    v27, v2, v19, v1, 7, i8 \n\t"

            // (K32-47 / K48-63 blocks continue unchanged...)
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vmv.v.v      v1, v29                 \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v3, v3, 4             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 2               \n\t"
            "vsll.vi      v9, v5, 2               \n\t"
            "vsll.vi      v10, v6, 2              \n\t"
            "vsll.vi      v11, v7, 2              \n\t"

            "vsrl.vi      v20, v8, 6              \n\t"
            "vsrl.vi      v21, v9, 6              \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v22, v10, 6             \n\t"
            "vsrl.vi      v23, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v20, v20, -4, v0.t      \n\t"

            // K48-63
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsrl.vi      v8, v4, 6               \n\t"
            "vsrl.vi      v9, v5, 6               \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v10, v6, 6              \n\t"
            "vsrl.vi      v11, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v8, v8, -4, v0.t        \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v2, v3, 2             \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmadot.hp    v24, v3, v20, v1, 0, i8 \n\t"
            "vmadot.hp    v25, v3, v21, v1, 1, i8 \n\t"
            "vmadot.hp    v26, v3, v22, v1, 2, i8 \n\t"
            "vmadot.hp    v27, v3, v23, v1, 3, i8 \n\t"
            "vmadot.hp    v24, v2, v8, v1, 4, i8  \n\t"
            "vmadot.hp    v25, v2, v9, v1, 5, i8  \n\t"
            "vmadot.hp    v26, v2, v10, v1, 6, i8 \n\t"
            "vmadot.hp    v27, v2, v11, v1, 7, i8 \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // ---- End of K64 chunk: reduce fp16 accumulators -> fp32 and scale by A ----
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v24, v25, 1        \n\t"
            "vpack.vv     v14, v26, v27, 1        \n\t"
            "vpack.vv     v16, v12, v14, 2        \n\t"
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v26, v16                \n\t"  // fp16 -> fp32 vector (qsum * b_scales)

            // Load A scale and advance A pointer to next K256 superblock
            "flw          f0, (s2)                \n\t"
            "addi         s2, s2, 4+32+256        \n\t"
            "add          t4, s7, %[B_STR]        \n\t"  // next B block base
            "addi         s3, s2, 4+32            \n\t"  // reset A data pointer for next block

            // Advance B pointers to next K256 superblock
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s8, s6, 1024            \n\t"
            "addi         s8, s8, 1024            \n\t"
            "addi         s7, t4, 0               \n\t"
            "addi         t2, t2, -1              \n\t"

            // Final per-block scaling: a_scale * 16.0f
            "fmul.s       f0, f0, %[a_post_mul]   \n\t"
            // acc += (qsum * b_scales) * (a_scale*16)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfmacc.vf    v31, f0, v26            \n\t"

            "beqz         t2, BLK_LPND%=          \n\t"

            // Preload next block's scales16 and rebuild v30 for vmadot.hp
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v1, (s8)                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v26, v1, v1, 3          \n\t"
            "vmv.v.v      v17, v26                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vf     v30, v17, %[q3_step]    \n\t"

            // Reset fp16 partial accumulators for next K64 loop(s)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v16, v16           \n\t"
            "vxor.vv      v25, v16, v16           \n\t"
            "vxor.vv      v26, v16, v16           \n\t"
            "vxor.vv      v27, v16, v16           \n\t"

            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v31, (%[pC])            \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride), [q3_step] "f"(k_q3k_scale_step),
              [a_post_mul] "f"(k_a_scale_post_mul)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "f1", "s2", "s3", "s4", "s5", "s6", "s7", "s8");
#endif
    }
}

void gemm_kernel_i8i3k_m4(const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    using blk_type           = nrow_block_q3_k<32>;
    constexpr size_t NB_COLS = 32;  //only support 32 in ASM

    const blk_type * b_base = reinterpret_cast<const blk_type *>(quant_b_data);

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t           nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
        const blk_type * quant_b_blk_data = b_base + (ni / NB_COLS) * k_blks;

        //------------------------------------------------------------------------------
        // A format
        // Ascale   fp32 * 1* 4row    128bit
        // Asum     int16 * 16 4row  1024bit
        // A M1K256 int8 4row        8192bit
        //------------------------------------------------------------------------------
        // B format
        // B_scl    uint8*N32*16    4096bit
        // B_Hmask  N32K16*16 1bit  8192bit
        // B_Qs     N32K16*16 2bit  16384bit
        // B scl16  fp16 * N32      512bit;
        //------------------------------------------------------------------------------
        //bias always be nullptr
        __asm__ volatile(
            // t2 = k_blks (each is K256 superblock)
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = 256/64 = 4 (K64 iterations per superblock)
            "li           t3, 4                   \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 16+128       \n\t"  // s3 = pAData, (pA+AScl+ASum)

            // B block layout for nrow_block_q3_k<32>:
            // scales: 512B, hmask: 1024B, qs: 2048B, scales16: 64B
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask (skip scales)
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs (skip hmask)
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v0, v0             \n\t"  // v24-v27: K256 temp accumulator
            "vxor.vv      v25, v0, v0             \n\t"
            "vxor.vv      v26, v0, v0             \n\t"
            "vxor.vv      v27, v0, v0             \n\t"
            "vxor.vv      v28, v0, v0             \n\t"  // v28-v31: final accumulator
            "vxor.vv      v29, v0, v0             \n\t"
            "vxor.vv      v30, v0, v0             \n\t"
            "vxor.vv      v31, v0, v0             \n\t"

            // ordinary vmadot: vle*13 vecIns*96 vmadot*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"
            "K64_LPST%=:                          \n\t"

            // ========== K0-15: First K16 sub-block ==========
            // Load B INT8 scale factors (32 cols × 16 K16 blocks)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v8, (s4)                \n\t"
            "addi         s4, s4, 128             \n\t"

            // Load B quantized data (32 cols × 16 elements × 2bit, stored in 4 groups)
            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            // Load B hmask (32 cols × 16bit sign mask)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // Load A data (4 rows × 16 elements × INT8)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v12, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"  // Jump to next row
            "vle8.v       v13, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"
            "vle8.v       v14, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"
            "vle8.v       v15, (s3)               \n\t"
            "addi         s3, s3, -768+64         \n\t"  // Back to first row, advance 16 elements

            // Pack A data: merge 4 rows into 2 vectors
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vpack.vv     v16, v12, v13, 1        \n\t"
            "vpack.vv     v18, v14, v15, 1        \n\t"
            "vpack.vv     v2, v16, v18, 2         \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v2, v12, i8        \n\t"  // 4 rows × cols 0-7
            "vmadot       v18, v2, v13, i8        \n\t"  // 4 rows × cols 8-15
            "vmadot       v20, v2, v14, i8        \n\t"  // 4 rows × cols 16-23
            "vmadot       v22, v2, v15, i8        \n\t"  // 4 rows × cols 24-31

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"  // Merge cols 0-15
            "vpack.vv     v14, v20, v22, 2        \n\t"  // Merge cols 16-31
            "vpack.vv     v16, v12, v14, 3        \n\t"  // Inter-row results (INT16)
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // INT8 → INT16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // INT16 → INT32

            // Accumulate to K256 accumulator: qsum * b_scale
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"  // Row 0
            "vmacc.vv     v25, v17, v23           \n\t"  // Row 1
            "vmacc.vv     v26, v18, v23           \n\t"  // Row 2
            "vmacc.vv     v27, v19, v23           \n\t"

            // ========== K16-31, K32-47, K48-63: Similar processing ==========
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 8             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v12, v4, 4              \n\t"
            "vsll.vi      v13, v5, 4              \n\t"
            "vsll.vi      v14, v6, 4              \n\t"
            "vsll.vi      v15, v7, 4              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v12, 6             \n\t"
            "vsrl.vi      v13, v13, 6             \n\t"
            "vsrl.vi      v14, v14, 6             \n\t"
            "vsrl.vi      v15, v15, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v2, v12, i8        \n\t"
            "vmadot       v18, v2, v13, i8        \n\t"
            "vmadot       v20, v2, v14, i8        \n\t"
            "vmadot       v22, v2, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            //K32-47
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v12, v4, 2              \n\t"
            "vsll.vi      v13, v5, 2              \n\t"
            "vsll.vi      v14, v6, 2              \n\t"
            "vsll.vi      v15, v7, 2              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v12, 6             \n\t"
            "vsrl.vi      v13, v13, 6             \n\t"
            "vsrl.vi      v14, v14, 6             \n\t"
            "vsrl.vi      v15, v15, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v12, i8        \n\t"
            "vmadot       v18, v3, v13, i8        \n\t"
            "vmadot       v20, v3, v14, i8        \n\t"
            "vmadot       v22, v3, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            // K48-63
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v3, v3, 8             \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v12, v4, 6              \n\t"
            "vsrl.vi      v13, v5, 6              \n\t"
            "vsrl.vi      v14, v6, 6              \n\t"
            "vsrl.vi      v15, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v12, i8        \n\t"
            "vmadot       v18, v3, v13, i8        \n\t"
            "vmadot       v20, v3, v14, i8        \n\t"
            "vmadot       v22, v3, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // ========== K256 superblock complete, apply scale factors ==========
            // Load A's 4 row scale factors (FP32)
            "flw          f0, (s2)                \n\t"
            "flw          f1, 4(s2)               \n\t"
            "flw          f2, 8(s2)               \n\t"
            "flw          f3, 12(s2)              \n\t"
            "add          s2, s2, %[A_STR]        \n\t"  // Advance to next superblock
            "add          t4, s7, %[B_STR]        \n\t"  // t4 = next B block address
            "addi         s3, s2, (4+32)*4        \n\t"

            // Load B FP16 global scale factors (32 cols)
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v8, (s6)                \n\t"

            // Update B pointers to next block
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s7, t4, 0               \n\t"

            // ========== Type conversion and final scaling ==========
            // FP16 → FP32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v9, v8                 \n\t"

            // INT32 → FP32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v  v24, v24                \n\t"
            "vfcvt.f.x.v  v25, v25                \n\t"
            "vfcvt.f.x.v  v26, v26                \n\t"
            "vfcvt.f.x.v  v27, v27                \n\t"

            // Compute a_scale * b_scale (4 rows)
            "vfmul.vf     v12, v9, f0             \n\t"
            "vfmul.vf     v13, v9, f1             \n\t"
            "vfmul.vf     v14, v9, f2             \n\t"
            "vfmul.vf     v15, v9, f3             \n\t"

            // Final accumulation: result += qsum * a_scale * b_scale
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfmacc.vv    v28, v12, v24           \n\t"
            "vfmacc.vv    v29, v13, v25           \n\t"
            "vfmacc.vv    v30, v14, v26           \n\t"
            "vfmacc.vv    v31, v15, v27           \n\t"

            // Prepare for next K superblock
            "addi         t2, t2, -1              \n\t"
            "vxor.vv      v24, v0, v0             \n\t"  // Clear K256 accumulator
            "vxor.vv      v25, v0, v0             \n\t"
            "vxor.vv      v26, v0, v0             \n\t"
            "vxor.vv      v27, v0, v0             \n\t"
            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"

            // ========== Store results (4 rows × 32 cols) ==========
            "mv           t5, %[pC]               \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v28, (%[pC])            \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v29, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v30, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v31, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "FUNC_END%=:                          \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride), [A_STR] "r"(a_nrow_block_stride), [LDC] "r"(ldc * 4)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "f1", "f2", "f3", "s2", "s3", "s4", "s5", "s6", "s7");
    }
}

void gemm_kernel_i8i4_m1(const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_n,
                         size_t          k_blks) {
    constexpr size_t blk_len = 32;
    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
            // || Element                              || Element                              || ...
#if 0
            //bias always be nullptr
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*7 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"  // Bzp u8 -> u16
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vfcvt.f.x.v  v16, v26                \n\t"  // zp i16 -> fp16
                "vadd.vi      v18, v16, 0             \n\t"
                "vadd.vi      v20, v16, 0             \n\t"
                "vadd.vi      v22, v16, 0             \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                // mac result * b_scale; f16*f16->f32
                "vfwmul.vv     v31, v30, v16          \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");

#endif
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(uint8_t) +  // b zp
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // Bzp uint8_t * N32 256bit;
            // || scl*32..(fp16) | zp*32(uint8) | blk0 blk1 ... blk31 || scl*32..(fp16)  ...
            // || Element                                             || Element         ...

            //bias always be nullptr
#if 0
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4        \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*6 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v31, (s4)               \n\t"  // B zp 32Row*uint8 = 256bit
                "addi         s4, s4, 32+128*4        \n\t"

                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v31, x0            \n\t"  // Bzp u8 -> u16

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vfwcvt.f.f.v v22, v30                \n\t"  // b_scale fp16 -> fp32
                "vfcvt.f.x.v  v18, v26                \n\t"  // zp i16 -> fp16
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vfwadd.vv    v20, v18, v16           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // mac result * b_scale; f32*f32->f32
                "vfmul.vv     v31, v22, v20           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#endif
        }
    }
}

// A stores (dA, -sum(qA), qA); B has a uint8 zero point per row and K32 block.
// Compute dot(qA,qB) - zpB*sum(qA) in INT32 before applying FP32 scales.
void gemm_kernel_i8i4_m1_q41(const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           float *         c_ptr,
                           size_t          count_n,
                           size_t          k_blks) {
    using block_type = ggml::cpu::riscv64_spacemit::Q41Block<32, 32, 2>;
    constexpr size_t b_data_bytes  = 32 * 32 / 2 * sizeof(uint8_t);
    constexpr size_t b_zp_bytes    = 32 * sizeof(uint8_t);
    constexpr size_t b_scale_bytes = 32 * sizeof(_Float16);
    constexpr size_t a_sum_off     = sizeof(float);
    constexpr size_t a_data_off    = a_sum_off + sizeof(int16_t);
    constexpr size_t a_block_bytes = a_data_off + 32 * sizeof(int8_t);
    static_assert(sizeof(block_type) == b_data_bytes + b_zp_bytes + b_scale_bytes &&
                  offsetof(block_type, zp) == b_data_bytes &&
                  offsetof(block_type, scale) == b_data_bytes + b_zp_bytes);
    for (size_t n = 0; n < count_n; n += 32) {
        const size_t nblks = std::min<size_t>(count_n - n, 32);
        const uint8_t * b = quant_b_data + (n / 32) * k_blks * sizeof(block_type);
        float * c = c_ptr + n;
        // clang-format off
        __asm__ volatile(
            "mv t3, %[BCK]\n\t"
            "mv t4, %[NBLKS]\n\t"
            "vsetvli t0, x0, e32, m1\n\t"
            "mv s2, %[pA]\n\t"
            "addi s3, %[pA], %[a_data_off]\n\t"
            "mv s5, %[pB]\n\t"
            "mv s6, %[pC]\n\t"
            "vxor.vv v2, v0, v0\n\t"
            ".align 4\n\t"
            "Q4_AFFINE_GEMV_K%=:\n\t"
            "vsetvli t0, x0, e8, m1\n\t"
            "vl4r.v v4, (s5)\n\t"
            "vsetvli t0, x0, e8, mf4\n\t"
            "vle8.v v3, (s3)\n\t"
            "addi s3, s3, %[a_block_bytes]\n\t"
            "flw f0, (s2)\n\t"
            "lh t2, %[a_sum_off](s2)\n\t"
            "addi s2, s2, %[a_block_bytes]\n\t"
            "vsetvli t0, x0, e8, m1\n\t"
            "vsrl.vi v24, v3, 4\n\t"
            "vnpack4.vv v8, v3, v3, 3\n\t"
            "vnpack4.vv v10, v24, v24, 3\n\t"
            "vsetvli t0, x0, e32, m1\n\t"
            "vxor.vv v16, v16, v16\n\t"
            "vxor.vv v18, v18, v18\n\t"
            "vxor.vv v20, v20, v20\n\t"
            "vxor.vv v22, v22, v22\n\t"
            "vmadotsu v16, v10, v4, i4\n\t"
            "vmadotsu v18, v10, v5, i4\n\t"
            "vmadotsu v20, v10, v6, i4\n\t"
            "vmadotsu v22, v10, v7, i4\n\t"
            "vsll.vi v16, v16, 4\n\t"
            "vsll.vi v18, v18, 4\n\t"
            "vsll.vi v20, v20, 4\n\t"
            "vsll.vi v22, v22, 4\n\t"
            "vmadotu v16, v8, v4, i4\n\t"
            "vmadotu v18, v8, v5, i4\n\t"
            "vmadotu v20, v8, v6, i4\n\t"
            "vmadotu v22, v8, v7, i4\n\t"
            "vsetvli t0, x0, e16, m1\n\t"
            "vpack.vv v24, v16, v18, 2\n\t"
            "vpack.vv v26, v20, v22, 2\n\t"
            "vpack.vv v16, v24, v26, 3\n\t"
            "vsetvli t0, x0, e16, mf2\n\t"
            "addi s4, s5, %[b_zp_off]\n\t"
            "vle8.v v28, (s4)\n\t"
            "vzext.vf2 v26, v28\n\t"
            "vwmacc.vx v16, t2, v26\n\t"
            "addi s4, s5, %[b_scale_off]\n\t"
            "vle16.v v30, (s4)\n\t"
            "vfwcvt.f.f.v v22, v30\n\t"
            "vsetvli t0, x0, e32, m1\n\t"
            "vfcvt.f.x.v v20, v16\n\t"
            "vfmul.vv v20, v20, v22\n\t"
            "vfmacc.vf v2, f0, v20\n\t"
            "addi s5, s5, %[block_bytes]\n\t"
            "addi t3, t3, -1\n\t"
            "bgtz t3, Q4_AFFINE_GEMV_K%=\n\t"
            "vsetvli t0, t4, e32, m1\n\t"
            "vse32.v v2, (s6)\n\t"
            :
            : [BCK] "r"(k_blks), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(b), [pC] "r"(c),
              [block_bytes] "i"(sizeof(block_type)), [b_zp_off] "i"(offsetof(block_type, zp)),
              [b_scale_off] "i"(offsetof(block_type, scale)), [a_sum_off] "i"(a_sum_off),
              [a_data_off] "i"(a_data_off), [a_block_bytes] "i"(a_block_bytes)
            : "cc", "memory", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
        // clang-format on
    }
}

void gemm_kernel_i8i4_m4(const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;
#if 0
            asm volatile(
                "li             t1,  8              \n\t"
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v28, v28, v28       \n\t"
                "vxor.vv        v29, v29, v29       \n\t"
                "vxor.vv        v30, v30, v30       \n\t"
                "vxor.vv        v31, v31, v31       \n\t"
                "mv             t4, %[BK]           \n\t"

                ".align 4                           \n\t"
                "BLK_LOOP%=:                        \n\t"
                // load scale A
                "flw            fa0, (%[A])         \n\t"
                "flw            fa1, 4(%[A])        \n\t"
                "flw            fa2, 8(%[A])        \n\t"
                "flw            fa3, 12(%[A])       \n\t"
                "addi           %[A], %[A], 16      \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2    \n\t"
                "vle16.v        v12, (%[B])         \n\t"
                "addi           %[B], %[B], 64      \n\t"
                "vfwcvt.f.f.v   v14, v12            \n\t"

                "vsetivli       t0, 4, e16, mf2     \n\t"
                "vle16.v        v8, (%[A])          \n\t"  // asum
                "addi           %[A], %[A], 8       \n\t"
                "vwmul.vx       v10, v8, t1         \n\t"  // 8*asum

                "vsetvli        t0, x0, e8, m1      \n\t"
                "vl1r.v         v0, (%[A])          \n\t"
                "addi           %[A], %[A], 128     \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])          \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512     \n\t"
                "vsrl.vi        v1, v0, 4           \n\t"
                "vnpack4.vv     v12, v0, v1, 3      \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2     \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v16, v16, v16       \n\t"
                "vxor.vv        v18, v16, v16       \n\t"
                "vxor.vv        v20, v16, v16       \n\t"
                "vxor.vv        v22, v16, v16       \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vmadotsu       v16, v3, v4, i4     \n\t"   // high 4
                "vmadotsu       v18, v3, v5, i4     \n\t"
                "vmadotsu       v20, v3, v6, i4     \n\t"
                "vmadotsu       v22, v3, v7, i4     \n\t"
                "vsll.vi        v16, v16, 4         \n\t"
                "vsll.vi        v18, v18, 4         \n\t"
                "vsll.vi        v20, v20, 4         \n\t"
                "vsll.vi        v22, v22, 4         \n\t"
                "vmadotu        v16, v2, v4, i4     \n\t"   // low 4
                "vmadotu        v18, v2, v5, i4     \n\t"
                "vmadotu        v20, v2, v6, i4     \n\t"
                "vmadotu        v22, v2, v7, i4     \n\t"

                "vpack.vv       v0, v16, v18, 2     \n\t"
                "vpack.vv       v2, v20, v22, 2     \n\t"
                "vpack.vv       v16, v0, v2, 3      \n\t"
                "vpack.vv       v18, v1, v3, 3      \n\t"

                "vrgather.vi    v0, v10, 0          \n\t"
                "vrgather.vi    v1, v10, 1          \n\t"
                "vrgather.vi    v2, v10, 2          \n\t"
                "vrgather.vi    v3, v10, 3          \n\t"

                "vadd.vv        v16, v16, v0        \n\t"
                "vadd.vv        v17, v17, v1        \n\t"
                "vadd.vv        v18, v18, v2        \n\t"
                "vadd.vv        v19, v19, v3        \n\t"

                "vfcvt.f.x.v    v16, v16            \n\t"
                "vfcvt.f.x.v    v17, v17            \n\t"
                "vfcvt.f.x.v    v18, v18            \n\t"
                "vfcvt.f.x.v    v19, v19            \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14       \n\t"
                "vfmul.vv       v17, v17, v14       \n\t"
                "vfmul.vv       v18, v18, v14       \n\t"
                "vfmul.vv       v19, v19, v14       \n\t"

                "addi           t4, t4, -1          \n\t"
                "vfmacc.vf      v28, fa0, v16       \n\t"
                "vfmacc.vf      v29, fa1, v17       \n\t"
                "vfmacc.vf      v30, fa2, v18       \n\t"
                "vfmacc.vf      v31, fa3, v19       \n\t"

                "bgtz           t4, BLK_LOOP%=      \n\t"

                // save
                "vsetvli        t0, x0, e32, m1     \n\t"
                "add            t2, %[LDC], %[DST]  \n\t"
                "vse32.v        v28, (%[DST])       \n\t"
                "add            t3, %[LDC], t2      \n\t"
                "vse32.v        v29, (t2)           \n\t"
                "add            t2, %[LDC], t3      \n\t"
                "vse32.v        v30, (t3)           \n\t"
                "vse32.v        v31, (t2)           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#else
            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vxor.vv        v31, v31, v31           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t4, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A
                "flw            fa0, (%[A])             \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           %[A], %[A], 16          \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                "vsetivli       t0, 4, e16, mf2         \n\t"
                "vle16.v        v8, (%[A])              \n\t"  // asum
                "addi           %[A], %[A], 8           \n\t"
                "vsll.vi        v8, v8, 3               \n\t"  // asum * 8
                "vfcvt.f.x.v    v9, v8                  \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vrgather.vi    v10, v9, 0              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl1r.v         v16, (%[A])             \n\t"
                "addi           %[A], %[A], 128         \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vsrl.vi        v17, v16, 4             \n\t"
                "vnpack4.vv     v12, v16, v17, 3        \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2         \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v16, v10, v10,0         \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vpack.vv       v20, v16, v16,0         \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vpack.vv       v18, v20, v20, 0        \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"
                "vfwmul.vv      v18, v21, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t4, t4, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"
                "vfmacc.vf      v30, fa2, v18           \n\t"
                "vfmacc.vf      v31, fa3, v19           \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "add            t2, %[LDC], %[DST]      \n\t"
                "vse32.v        v28, (%[DST])           \n\t"
                "add            t3, %[LDC], t2          \n\t"
                "vse32.v        v29, (t2)               \n\t"
                "add            t2, %[LDC], t3          \n\t"
                "vse32.v        v30, (t3)               \n\t"
                "vse32.v        v31, (t2)               \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#endif
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "li             t1,  8          \n\t"
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v28, v28, v28   \n\t"
                "vxor.vv        v29, v29, v29   \n\t"
                "vxor.vv        v30, v30, v30   \n\t"
                "vxor.vv        v31, v31, v31   \n\t"
                "mv             t4, %[BK]       \n\t"

                ".align 4                        \n\t"
                "BLK_LOOP%=:                     \n\t"
                // load scale A
                "flw            fa0, (%[A])     \n\t"
                "flw            fa1, 4(%[A])    \n\t"
                "flw            fa2, 8(%[A])    \n\t"
                "flw            fa3, 12(%[A])   \n\t"
                "addi           %[A], %[A], 16  \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2\n\t"
                "vle16.v        v12, (%[B])     \n\t"
                "addi           %[B], %[B], 64  \n\t"
                "vfwcvt.f.f.v   v14, v12        \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4 \n\t"
                "vle8.v         v8, (%[B])      \n\t"
                "addi           %[B], %[B], 32  \n\t"
                "vwaddu.vx      v10, v8, x0     \n\t"

                // load a sum
                "lh             s1, (%[A])      \n\t"
                "lh             s2, 2(%[A])     \n\t"
                "lh             s3, 4(%[A])     \n\t"
                "lh             s4, 6(%[A])     \n\t"
                "addi           %[A], %[A], 8   \n\t"

                "vsetvli        t0, x0, e8, m1  \n\t"
                "vl1r.v         v0, (%[A])      \n\t"
                "addi           %[A], %[A], 128 \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])      \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512 \n\t"
                "vsrl.vi        v1, v0, 4       \n\t"
                "vnpack4.vv     v12, v0, v1, 3  \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2 \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v16, v16, v16   \n\t"
                "vxor.vv        v18, v16, v16   \n\t"
                "vxor.vv        v20, v16, v16   \n\t"
                "vxor.vv        v22, v16, v16   \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vmadotsu       v16, v3, v4, i4 \n\t"  // high 4
                "vmadotsu       v18, v3, v5, i4 \n\t"
                "vmadotsu       v20, v3, v6, i4 \n\t"
                "vmadotsu       v22, v3, v7, i4 \n\t"
                "vsll.vi        v16, v16, 4     \n\t"
                "vsll.vi        v18, v18, 4     \n\t"
                "vsll.vi        v20, v20, 4     \n\t"
                "vsll.vi        v22, v22, 4     \n\t"
                "vmadotu        v16, v2, v4, i4 \n\t"  // low 4
                "vmadotu        v18, v2, v5, i4 \n\t"
                "vmadotu        v20, v2, v6, i4 \n\t"
                "vmadotu        v22, v2, v7, i4 \n\t"

                "vpack.vv       v0, v16, v18, 2 \n\t"
                "vpack.vv       v2, v20, v22, 2 \n\t"
                "vpack.vv       v16, v0, v2, 3  \n\t"
                "vpack.vv       v18, v1, v3, 3  \n\t"

                "vsetvli        t0, x0, e16, m1 \n\t"
                "vwmul.vx       v0, v10, s1     \n\t"
                "vwmul.vx       v2, v10, s2     \n\t"
                "vwmul.vx       v4, v10, s3     \n\t"
                "vwmul.vx       v6, v10, s4     \n\t"

                "vsetvli        t0, x0, e32, m1 \n\t"
                "vadd.vv        v16, v16, v0    \n\t"
                "vadd.vv        v17, v17, v2    \n\t"
                "vadd.vv        v18, v18, v4    \n\t"
                "vadd.vv        v19, v19, v6    \n\t"

                "vfcvt.f.x.v    v16, v16        \n\t"
                "vfcvt.f.x.v    v17, v17        \n\t"
                "vfcvt.f.x.v    v18, v18        \n\t"
                "vfcvt.f.x.v    v19, v19        \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14   \n\t"
                "vfmul.vv       v17, v17, v14   \n\t"
                "vfmul.vv       v18, v18, v14   \n\t"
                "vfmul.vv       v19, v19, v14   \n\t"

                "addi           t4, t4, -1      \n\t"
                "vfmacc.vf      v28, fa0, v16   \n\t"
                "vfmacc.vf      v29, fa1, v17   \n\t"
                "vfmacc.vf      v30, fa2, v18   \n\t"
                "vfmacc.vf      v31, fa3, v19   \n\t"

                "bgtz           t4, BLK_LOOP%=  \n\t"

                // save
                "vsetvli        t0, x0, e32, m1 \n\t"
                "add            t2, %[LDC], %[DST]\n\t"
                "vse32.v        v28, (%[DST])   \n\t"
                "add            t3, %[LDC], t2  \n\t"
                "vse32.v        v29, (t2)       \n\t"
                "add            t2, %[LDC], t3  \n\t"
                "vse32.v        v30, (t3)       \n\t"
                "vse32.v        v31, (t2)       \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "s1", "s2", "s3", "s4");
        }
    }
}

#define HP_AL_MM8(H0, H1, H2, H3, L0, L1, L2, L3, AH, AL, B0, B1, B2, B3, S) \
    "vmadot.hp      " H0 ", " AH ", " B0 ", " S ", 0, i4 \n\t"             \
    "vmadot.hp      " H1 ", " AH ", " B1 ", " S ", 1, i4 \n\t"             \
    "vmadot.hp      " H2 ", " AH ", " B2 ", " S ", 2, i4 \n\t"             \
    "vmadot.hp      " H3 ", " AH ", " B3 ", " S ", 3, i4 \n\t"             \
    "vmadotus.hp    " L0 ", " AL ", " B0 ", " S ", 4, i4 \n\t"             \
    "vmadotus.hp    " L1 ", " AL ", " B1 ", " S ", 5, i4 \n\t"             \
    "vmadotus.hp    " L2 ", " AL ", " B2 ", " S ", 6, i4 \n\t"             \
    "vmadotus.hp    " L3 ", " AL ", " B3 ", " S ", 7, i4 \n\t"

#define HP_AL_LOAD_Y                                   \
    "flh            fa3, 8(t2)                \n\t"    \
    "addi           t3, s5, %[B_SUBBLK_STRIDE]            \n\t"    \
    "vsetvli        t0, x0, e16, mf2          \n\t"    \
    "vle16.v        v13, (t3)                 \n\t"    \
    "addi           t3, t6, %[A_SUBBLK_STRIDE]            \n\t"   \
    "vl1r.v         v25, (t3)                 \n\t"    \
    "addi           t3, s5, %[B_N1_PAYLOAD]  \n\t"    \
    "vl4r.v         v8, (t3)                  \n\t"

#define HP_AL_LOAD_X_NEXT                              \
    "flh            fa0, 16(t2)               \n\t"    \
    "addi           t3, s5, %[B_2SUBBLK_STRIDE] \n\t"  \
    "vsetvli        t0, x0, e16, mf2          \n\t"    \
    "vle16.v        v12, (t3)                 \n\t"    \
    "addi           t3, t6, %[A_LOOP_STEP] \n\t"  \
    "vl1r.v         v24, (t3)                 \n\t"    \
    "addi           t3, s5, %[B_N3_PAYLOAD]   \n\t"    \
    "vl4r.v         v4, (t3)                  \n\t"

#define HP_AL_COMPUTE(FA, FT, BS, AR, NP, NL, NH, B0, B1, B2, B3)                                            \
    "fmul.h         " FT ", " FA ", %[HP16]   \n\t"                                                          \
    "vfmul.vf       v16, " BS ", " FA "       \n\t"                                                          \
    "vfmul.vf       v17, " BS ", " FT "       \n\t"                                                          \
    "vsetvli        t0, x0, e8, m1            \n\t"                                                          \
    "vsrl.vi        v1, " AR ", 4             \n\t"                                                          \
    "vnpack4.vv     " NP ", " AR ", v1, 3     \n\t"                                                          \
    "vpack.vv       v0, v17, v16, 3           \n\t"                                                          \
    "vupack.vv      " NL ", " NP ", " NP ", 2 \n\t"                                                          \
    "vsetvli        t0, x0, e32, m1           \n\t"                                                          \
    HP_AL_MM8("v18", "v19", "v20", "v21", "v22", "v23", "v26", "v27", NH, NL, B0, B1, B2, B3, "v0")

#define HP_AL_COMPUTE_X HP_AL_COMPUTE("fa0", "fa2", "v12", "v24", "v12", "v2", "v3", "v4", "v5", "v6", "v7")
#define HP_AL_COMPUTE_Y HP_AL_COMPUTE("fa3", "fa4", "v13", "v25", "v13", "v14", "v15", "v8", "v9", "v10", "v11")

void gemm_kernel_i8i4_hp_al_m4(const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               float *         c_ptr,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t NB_ROWS                = 4;
    constexpr size_t k_subblk_len           = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = BLOCK_QNK_LEN / k_subblk_len;
    constexpr size_t b_scale_bytes          = NB_COLS * sizeof(_Float16);
    constexpr size_t b_subblk_payload       = NB_COLS * k_subblk_len / 2 * sizeof(uint8_t);
    constexpr size_t b_subblk_stride        = b_scale_bytes + b_subblk_payload;
    constexpr size_t b_superblk_stride      = b_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    // A group order: K32 data lanes, subblock scales, a_sum[row][subblock], average scales.
    constexpr size_t a_data_bytes           = K_SUBBLKS_PER_SUPERBLK * NB_ROWS * k_subblk_len * sizeof(int8_t);
    constexpr size_t a_scale_bytes          = K_SUBBLKS_PER_SUPERBLK * NB_ROWS * sizeof(_Float16);
    constexpr size_t a_sum_bytes            = NB_ROWS * K_SUBBLKS_PER_SUPERBLK * sizeof(_Float16);
    constexpr size_t a_scale_avg_bytes      = NB_ROWS * sizeof(_Float16);
    constexpr size_t a_scale_off            = a_data_bytes;
    constexpr size_t a_sum_off              = a_scale_off + a_scale_bytes;
    constexpr size_t a_scale_avg_off        = a_sum_off + a_sum_bytes - a_scale_off;
    constexpr size_t a_group_alignment      = 64;
    constexpr size_t a_group_bytes          = a_data_bytes + a_scale_bytes + a_sum_bytes + a_scale_avg_bytes;
    static_assert(q8_hp_m4al_blk_size ==
                  (a_group_bytes + a_group_alignment - 1) / a_group_alignment * a_group_alignment,
                  "A payload, scales/sums and average scales with alignment padding");
    constexpr size_t b_2subblk_stride       = 2 * b_subblk_stride;
    constexpr size_t b_next_payload         = b_subblk_stride + b_scale_bytes;
    constexpr size_t b_3rd_payload          = 2 * b_subblk_stride + b_scale_bytes;
    constexpr size_t a_subblk_stride        = NB_ROWS * k_subblk_len * sizeof(int8_t);
    constexpr size_t a_loop_step            = 2 * a_subblk_stride;
    constexpr size_t a_grp_tail_step        = q8_hp_m4al_blk_size - (K_SUBBLKS_PER_SUPERBLK / 2 - 1) * a_loop_step;

    const size_t   b_tile_stride = k_blks * b_superblk_stride;
    const _Float16 hp_scale_16   = (_Float16) 16.0f;

    for (size_t ni = 0; ni + NB_COLS <= count_n; ni += NB_COLS) {
        const uint8_t * b_tile = quant_b_data + (ni / NB_COLS) * b_tile_stride;
        float *         dst_c  = c_ptr + ni;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmv.v.i        v28, 0                    \n\t"
            "vmv.v.i        v29, 0                    \n\t"
            "vmv.v.i        v30, 0                    \n\t"
            "vmv.v.i        v31, 0                    \n\t"

            ".align 4                                 \n\t"
            "_HPAL_BLK%=:                             \n\t"
            "addi           t2, t6, %[A_SCALE_OFF]       \n\t"
            "flh            fa1, %[A_SCALE_AVG_OFF](t2) \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmv.v.i        v18, 0                    \n\t"
            "vmv.v.i        v19, 0                    \n\t"
            "vmv.v.i        v20, 0                    \n\t"
            "vmv.v.i        v21, 0                    \n\t"
            "vmv.v.i        v22, 0                    \n\t"
            "vmv.v.i        v23, 0                    \n\t"
            "vmv.v.i        v26, 0                    \n\t"
            "vmv.v.i        v27, 0                    \n\t"
            "li             t4, 3                     \n\t"
            "flh            fa0, 0(t2)                \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "vl1r.v         v24, (t6)                 \n\t"
            "addi           t3, s5, %[B_SCALE_BYTES]  \n\t"
            "vl4r.v         v4, (t3)                  \n\t"

            "_HPAL_SUB%=:                             \n\t"
            HP_AL_LOAD_Y
            HP_AL_COMPUTE_X
            HP_AL_LOAD_X_NEXT
            HP_AL_COMPUTE_Y
            "addi           t6, t6, 256               \n\t"
            "addi           t2, t2, 16                \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"
            "addi           t4, t4, -1                \n\t"
            "bgtz           t4, _HPAL_SUB%=           \n\t"

            HP_AL_LOAD_Y
            HP_AL_COMPUTE_X
            "vsetvli        t0, x0, e16, mf2          \n\t"
            HP_AL_COMPUTE_Y
            "addi           t6, t6, %[A_GRP_TAIL_STEP] \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfadd.vv       v18, v18, v22             \n\t"
            "vfadd.vv       v19, v19, v23             \n\t"
            "vfadd.vv       v20, v20, v26             \n\t"
            "vfadd.vv       v21, v21, v27             \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"

            "addi           t5, t5, -1                \n\t"
            "bgtz           t5, _HPAL_BLK%=           \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            :
            : [A] "r"(quant_a_ptr), [B] "r"(b_tile), [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks),
              [HP16] "f"(hp_scale_16), [A_SCALE_OFF] "i"(a_scale_off), [A_SCALE_AVG_OFF] "i"(a_scale_avg_off),
              [B_SCALE_BYTES] "i"(b_scale_bytes), [B_SUBBLK_STRIDE] "i"(b_subblk_stride),
              [B_2SUBBLK_STRIDE] "i"(b_2subblk_stride), [B_N1_PAYLOAD] "i"(b_next_payload),
              [B_N3_PAYLOAD] "i"(b_3rd_payload), [A_SUBBLK_STRIDE] "i"(a_subblk_stride),
              [A_LOOP_STEP] "i"(a_loop_step), [A_GRP_TAIL_STEP] "i"(a_grp_tail_step)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "fa0", "fa1", "fa2", "fa3", "fa4", "v0", "v1", "v2", "v3",
              "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    }
}

#undef HP_AL_COMPUTE_Y
#undef HP_AL_COMPUTE_X
#undef HP_AL_COMPUTE
#undef HP_AL_LOAD_X_NEXT
#undef HP_AL_LOAD_Y
#undef HP_AL_MM8

#define M8_SCALE(OFF_T2, OFF_S5)                              \
    "flh            fa0, " OFF_T2 "(t2)       \n\t"           \
    "fmul.h         fa2, fa0, %[HP16]         \n\t"           \
    "addi           t3, s5, " OFF_S5 "        \n\t"           \
    "vsetvli        t0, x0, e16, mf2          \n\t"           \
    "vle16.v        v1, (t3)                  \n\t"           \
    "vfmul.vf       v2, v1, fa0               \n\t"           \
    "vfmul.vf       v3, v1, fa2               \n\t"           \
    "vsetvli        t0, x0, e8, m1            \n\t"           \
    "vpack.vv       v0, v3, v2, 3             \n\t"           \
    "vsetvli        t0, x0, e32, m1           \n\t"
#define M8_MM(AL, AH, B0, B1, B2, B3)                         \
    "vmadot.hp      v16, " AH ", " B0 ", v0, 0, i4 \n\t"      \
    "vmadot.hp      v17, " AH ", " B1 ", v0, 1, i4 \n\t"      \
    "vmadot.hp      v18, " AH ", " B2 ", v0, 2, i4 \n\t"      \
    "vmadot.hp      v19, " AH ", " B3 ", v0, 3, i4 \n\t"      \
    "vmadotus.hp    v20, " AL ", " B0 ", v0, 4, i4 \n\t"      \
    "vmadotus.hp    v21, " AL ", " B1 ", v0, 5, i4 \n\t"      \
    "vmadotus.hp    v22, " AL ", " B2 ", v0, 6, i4 \n\t"      \
    "vmadotus.hp    v23, " AL ", " B3 ", v0, 7, i4 \n\t"
#define M8_LOAD(OA, OA2, OB, AL, AH, B0)                      \
    "addi           t3, t6, " OA "            \n\t"           \
    "vl1r.v         " AL ", (t3)              \n\t"           \
    "addi           t3, t6, " OA2 "           \n\t"           \
    "vl1r.v         " AH ", (t3)              \n\t"           \
    "addi           t3, s5, " OB "            \n\t"           \
    "vl4r.v         " B0 ", (t3)              \n\t"

void gemm_kernel_i8i4_hp_al_m8(const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               float *         c_ptr,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    constexpr size_t NB_COLS                 = 32;
    constexpr size_t NB_ROWS                 = 8;
    constexpr size_t k_subblk_len            = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK  = BLOCK_QNK_LEN / k_subblk_len;
    constexpr size_t b_scale_bytes           = NB_COLS * sizeof(_Float16);
    constexpr size_t b_subblk_payload        = NB_COLS * k_subblk_len / 2 * sizeof(uint8_t);
    constexpr size_t b_subblk_stride         = b_scale_bytes + b_subblk_payload;
    constexpr size_t b_superblk_stride       = b_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    constexpr size_t a_data_bytes            = K_SUBBLKS_PER_SUPERBLK * NB_ROWS * k_subblk_len * sizeof(int8_t);
    constexpr size_t a_scale_bytes           = K_SUBBLKS_PER_SUPERBLK * sizeof(_Float16);
    constexpr size_t a_scale_avg_bytes       = sizeof(_Float16);
    constexpr size_t a_scale_off             = a_data_bytes;
    constexpr size_t b_2subblk_stride        = 2 * b_subblk_stride;
    constexpr size_t b_next_payload          = b_subblk_stride + b_scale_bytes;
    constexpr size_t b_3rd_payload           = 2 * b_subblk_stride + b_scale_bytes;
    // A stores eight 16-byte low-nibble lanes, then eight high-nibble lanes per K32 block.
    constexpr size_t a_hi_lane_off           = NB_ROWS * (k_subblk_len / 2) * sizeof(uint8_t);
    constexpr size_t a_subblk_stride         = 2 * a_hi_lane_off;
    constexpr size_t a_subblk_stride_plus_hi = a_subblk_stride + a_hi_lane_off;
    constexpr size_t a_loop_step             = 2 * a_subblk_stride;
    constexpr size_t a_grp_tail_step         = q8_hp_m8_blk_size - (K_SUBBLKS_PER_SUPERBLK / 2 - 1) * a_loop_step;
    constexpr size_t a_loop_step_plus_hi     = a_loop_step + a_hi_lane_off;
    constexpr size_t a_group_alignment       = 128;
    constexpr size_t a_group_bytes           = a_data_bytes + a_scale_bytes + a_scale_avg_bytes;
    static_assert(q8_hp_m8_blk_size ==
                  (a_group_bytes + a_group_alignment - 1) / a_group_alignment * a_group_alignment,
                  "A payload, shared subblock scales and average scale with alignment padding");

    const size_t   b_tile_stride = k_blks * b_superblk_stride;
    const _Float16 hp_scale_16   = (_Float16) 16.0f;
    alignas(128) float acc[4][64];

    for (size_t ni = 0; ni + NB_COLS <= count_n; ni += NB_COLS) {
        const uint8_t * b_tile = quant_b_data + (ni / NB_COLS) * b_tile_stride;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m2           \n\t"
            "vmv.v.i        v8, 0                     \n\t"
            "vmv.v.i        v10, 0                    \n\t"
            "vmv.v.i        v12, 0                    \n\t"
            "vmv.v.i        v14, 0                    \n\t"
            ".align 4                                 \n\t"
            "_HPM8_BLK%=:                             \n\t"
            "addi           t2, t6, %[A_SCALE_OFF] - 1   \n\t"  // Split +2048 to fit signed 12-bit immediates.
            "addi           t2, t2, 1                 \n\t"
            "flh            fa1, 16(t2)               \n\t"  // scale[8]: group average.
            "vsetvli        t0, x0, e16, m1           \n\t"
            "vmv.v.i        v16, 0                    \n\t"
            "vmv.v.i        v17, 0                    \n\t"
            "vmv.v.i        v18, 0                    \n\t"
            "vmv.v.i        v19, 0                    \n\t"
            "vmv.v.i        v20, 0                    \n\t"
            "vmv.v.i        v21, 0                    \n\t"
            "vmv.v.i        v22, 0                    \n\t"
            "vmv.v.i        v23, 0                    \n\t"
            "li             t4, 3                     \n\t"
            M8_LOAD("0", "%[A_HI_LANE_OFF]", "%[B_SCALE_BYTES]", "v24", "v25", "v4")
            "_HPM8_SUB%=:                             \n\t"
            M8_LOAD("%[A_SUBBLK_STRIDE]", "%[A_SUBBLK_STRIDE_PLUS_HI]", "%[B_NEXT_PAYLOAD]", "v26", "v27", "v28")
            M8_SCALE("0", "0")
            M8_MM("v24", "v25", "v4", "v5", "v6", "v7")
            M8_LOAD("%[A_LOOP_STEP]", "%[A_LOOP_STEP_PLUS_HI]", "%[B_3RD_PAYLOAD]", "v24", "v25", "v4")
            M8_SCALE("2", "%[B_SUBBLK_STRIDE]")
            M8_MM("v26", "v27", "v28", "v29", "v30", "v31")
            "addi           t6, t6, %[A_LOOP_STEP]   \n\t"
            "addi           t2, t2, 4                 \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"
            "addi           t4, t4, -1                \n\t"
            "bgtz           t4, _HPM8_SUB%=           \n\t"
            M8_LOAD("%[A_SUBBLK_STRIDE]", "%[A_SUBBLK_STRIDE_PLUS_HI]", "%[B_NEXT_PAYLOAD]", "v26", "v27", "v28")
            M8_SCALE("0", "0")
            M8_MM("v24", "v25", "v4", "v5", "v6", "v7")
            M8_SCALE("2", "%[B_SUBBLK_STRIDE]")
            M8_MM("v26", "v27", "v28", "v29", "v30", "v31")
            "addi           t6, t6, %[A_GRP_TAIL_STEP] \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"
            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfadd.vv       v16, v16, v20             \n\t"
            "vfadd.vv       v17, v17, v21             \n\t"
            "vfadd.vv       v18, v18, v22             \n\t"
            "vfadd.vv       v19, v19, v23             \n\t"
            "vfwmacc.vf     v8, fa1, v16              \n\t"
            "vfwmacc.vf     v10, fa1, v17             \n\t"
            "vfwmacc.vf     v12, fa1, v18             \n\t"
            "vfwmacc.vf     v14, fa1, v19             \n\t"
            "addi           t5, t5, -1                \n\t"
            "bgtz           t5, _HPM8_BLK%=           \n\t"
            "vsetvli        t0, x0, e32, m2           \n\t"
            "vse32.v        v8, (%[ACC])              \n\t"
            "addi           t3, %[ACC], 256           \n\t"
            "vse32.v        v10, (t3)                 \n\t"
            "addi           t3, %[ACC], 512           \n\t"
            "vse32.v        v12, (t3)                 \n\t"
            "addi           t3, %[ACC], 768           \n\t"
            "vse32.v        v14, (t3)                 \n\t"
            :
            : [A] "r"(quant_a_ptr), [B] "r"(b_tile), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16), [ACC] "r"(&acc[0][0]),
              [A_SCALE_OFF] "i"(a_scale_off), [A_HI_LANE_OFF] "i"(a_hi_lane_off), [A_GRP_TAIL_STEP] "i"(a_grp_tail_step),
              [A_LOOP_STEP] "i"(a_loop_step), [A_LOOP_STEP_PLUS_HI] "i"(a_loop_step_plus_hi),
              [A_SUBBLK_STRIDE] "i"(a_subblk_stride), [A_SUBBLK_STRIDE_PLUS_HI] "i"(a_subblk_stride_plus_hi),
              [B_SCALE_BYTES] "i"(b_scale_bytes), [B_SUBBLK_STRIDE] "i"(b_subblk_stride),
              [B_NEXT_PAYLOAD] "i"(b_next_payload), [B_3RD_PAYLOAD] "i"(b_3rd_payload),
              [B_2SUBBLK_STRIDE] "i"(b_2subblk_stride)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "fa0", "fa1", "fa2", "v0", "v1", "v2", "v3", "v4", "v5",
              "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");

        for (size_t r = 0; r < 8; ++r) {
            float * c_row = c_ptr + r * ldc + ni;
            for (size_t g = 0; g < 4; ++g)
                for (size_t i = 0; i < 8; ++i)
                    c_row[g * 8 + i] = *(const volatile float *) &acc[g][r * 8 + i];
        }
    }
}

#undef M8_LOAD
#undef M8_MM
#undef M8_SCALE

void gemm_kernel_i8i4_hp_al_m1(const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               float *         c_ptr,
                               size_t          count_n,
                               size_t          k_blks) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t k_subblk_len           = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = BLOCK_QNK_LEN / k_subblk_len;
    constexpr size_t b_scale_bytes          = NB_COLS * sizeof(_Float16);
    constexpr size_t b_subblk_payload       = NB_COLS * k_subblk_len / 2 * sizeof(uint8_t);
    constexpr size_t b_subblk_stride        = b_scale_bytes + b_subblk_payload;
    constexpr size_t b_superblk_stride      = b_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    // A K256 row: 8 x [FP16 scale, 32-byte data], FP16 a_sum[8], FP16 average scale.
    constexpr size_t a_sum_off              = K_SUBBLKS_PER_SUPERBLK * (sizeof(_Float16) + k_subblk_len * sizeof(int8_t));
    constexpr size_t a_scale_avg_off        = a_sum_off + K_SUBBLKS_PER_SUPERBLK * sizeof(_Float16);
    constexpr size_t a_row_blk_stride       = a_scale_avg_off + sizeof(_Float16);

    const size_t b_tile_stride = k_blks * b_superblk_stride;

    for (size_t ni = 0; ni < count_n; ni += 32) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             t4, %[BK]               \n\t"
            "li             t0, 0x4c00              \n\t"
            "fmv.h.x        fa0, t0                 \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            "li             t5, 8                   \n\t"
            "flh            ft1, %[A_SCALE_AVG_OFF](%[A])          \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v16, v18, v18           \n\t"
            "vxor.vv        v17, v18, v18           \n\t"
            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v19, v18, v18           \n\t"

            "INNER_BLK_LOOP%=:                      \n\t"
            "flh            ft0, (%[A])             \n\t"
            "addi           %[A], %[A], 2           \n\t"
            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v3, (%[A])              \n\t"
            "addi           %[A], %[A], 32          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v8, (%[B])              \n\t"
            "addi           %[B], %[B], %[B_SCALE_BYTES] \n\t"
            "vl4r.v         v4, (%[B])              \n\t"
            "addi           %[B], %[B], %[B_PAYLOAD] \n\t"
            "vfmul.vf       v8, v8, ft0             \n\t"
            "vfmul.vf       v9, v8, fa0             \n\t"

            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v0, v8, v9, 3           \n\t"
            "vsrl.vi        v28, v3, 4              \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vnpack4.vv     v2, v3, v3, 3           \n\t"
            "vnpack4.vv     v3, v28, v28, 3         \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vmadot.hp      v16, v3, v4, v0, 4, i4  \n\t"
            "vmadot.hp      v17, v3, v5, v0, 5, i4  \n\t"
            "vmadot.hp      v18, v3, v6, v0, 6, i4  \n\t"
            "vmadot.hp      v19, v3, v7, v0, 7, i4  \n\t"
            "vmadotus.hp    v16, v2, v4, v0, 0, i4  \n\t"
            "vmadotus.hp    v17, v2, v5, v0, 1, i4  \n\t"
            "vmadotus.hp    v18, v2, v6, v0, 2, i4  \n\t"
            "vmadotus.hp    v19, v2, v7, v0, 3, i4  \n\t"

            "addi           t5, t5, -1              \n\t"
            "bgtz           t5, INNER_BLK_LOOP%=    \n\t"

            "vpack.vv       v8, v16, v17, 1         \n\t"
            "vpack.vv       v12, v18, v19, 1        \n\t"
            "vpack.vv       v20, v8, v12, 2         \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t4, t4, -1              \n\t"
            "vfwmacc.vf     v31, ft1, v20           \n\t"

            "addi           %[A], %[A], %[A_ROW_TAIL] \n\t"

            "bgtz           t4, BLK_LOOP%=          \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks), [A_SCALE_AVG_OFF] "i"(a_scale_avg_off),
              [B_SCALE_BYTES] "i"(b_scale_bytes), [B_PAYLOAD] "i"(b_subblk_payload),
              [A_ROW_TAIL] "i"(a_row_blk_stride - a_sum_off)
            : "t0", "t4", "t5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "fa0", "ft0", "ft1", "memory");
    }
}

// B retains unsigned nibbles and a separate zp; A stores a_sum = -8*sum(qA).
// The zp correction therefore uses scale_avg/8 before accumulating in FP32.

#define HPZP_MM8(H0, H1, H2, H3, L0, L1, L2, L3, AH, AL, B0, B1, B2, B3, S) \
    "vmadotsu.hp   " H0 ", " AH ", " B0 ", " S ", 0, i4 \n\t"             \
    "vmadotsu.hp   " H1 ", " AH ", " B1 ", " S ", 1, i4 \n\t"             \
    "vmadotsu.hp   " H2 ", " AH ", " B2 ", " S ", 2, i4 \n\t"             \
    "vmadotsu.hp   " H3 ", " AH ", " B3 ", " S ", 3, i4 \n\t"             \
    "vmadotu.hp    " L0 ", " AL ", " B0 ", " S ", 4, i4 \n\t"             \
    "vmadotu.hp    " L1 ", " AL ", " B1 ", " S ", 5, i4 \n\t"             \
    "vmadotu.hp    " L2 ", " AL ", " B2 ", " S ", 6, i4 \n\t"             \
    "vmadotu.hp    " L3 ", " AL ", " B3 ", " S ", 7, i4 \n\t"

#define HPZP_LOAD_Y                                   \
    "flh            fa3, 8(t2)                \n\t"    \
    "addi           t3, s5, %[B_SUBBLK_STRIDE]            \n\t"    \
    "vsetvli        t0, x0, e16, mf2          \n\t"    \
    "vle16.v        v13, (t3)                 \n\t"    \
    "addi           t3, t6, %[A_SUBBLK_STRIDE]            \n\t"   \
    "vl1r.v         v25, (t3)                 \n\t"    \
    "addi           t3, s5, %[B_N1_PAYLOAD]  \n\t"    \
    "vl4r.v         v8, (t3)                  \n\t"

#define HPZP_LOAD_X_NEXT                              \
    "flh            fa0, 16(t2)               \n\t"    \
    "addi           t3, s5, %[B_2SUBBLK_STRIDE] \n\t"  \
    "vsetvli        t0, x0, e16, mf2          \n\t"    \
    "vle16.v        v12, (t3)                 \n\t"    \
    "addi           t3, t6, %[A_LOOP_STEP] \n\t"  \
    "vl1r.v         v24, (t3)                 \n\t"    \
    "addi           t3, s5, %[B_N3_PAYLOAD]   \n\t"    \
    "vl4r.v         v4, (t3)                  \n\t"

#define HPZP_COMPUTE(FA, FT, BS, AR, NP, NL, NH, B0, B1, B2, B3)                                            \
    "fmul.h         " FT ", " FA ", %[HP16]   \n\t"                                                          \
    "vfmul.vf       v16, " BS ", " FA "       \n\t"                                                          \
    "vfmul.vf       v17, " BS ", " FT "       \n\t"                                                          \
    "vsetvli        t0, x0, e8, m1            \n\t"                                                          \
    "vsrl.vi        v1, " AR ", 4             \n\t"                                                          \
    "vnpack4.vv     " NP ", " AR ", v1, 3     \n\t"                                                          \
    "vpack.vv       v0, v17, v16, 3           \n\t"                                                          \
    "vupack.vv      " NL ", " NP ", " NP ", 2 \n\t"                                                          \
    "vsetvli        t0, x0, e32, m1           \n\t"                                                          \
    HPZP_MM8("v18", "v19", "v20", "v21", "v22", "v23", "v26", "v27", NH, NL, B0, B1, B2, B3, "v0")

#define HPZP_COMPUTE_X HPZP_COMPUTE("fa0", "fa2", "v12", "v24", "v12", "v2", "v3", "v4", "v5", "v6", "v7")
#define HPZP_COMPUTE_Y HPZP_COMPUTE("fa3", "fa4", "v13", "v25", "v13", "v14", "v15", "v8", "v9", "v10", "v11")

void gemm_kernel_i8i4_hpzp_al_m4(const uint8_t * quant_a_ptr,
                                 const uint8_t * quant_b_data,
                                 float *         c_ptr,
                                 size_t          count_n,
                                 size_t          k_blks,
                                 size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t NB_ROWS                = 4;
    constexpr size_t k_subblk_len           = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = BLOCK_QNK_LEN / k_subblk_len;
    constexpr size_t b_scale_bytes          = NB_COLS * sizeof(_Float16);
    constexpr size_t b_subblk_payload       = NB_COLS * k_subblk_len / 2 * sizeof(uint8_t);
    constexpr size_t b_subblk_stride        = b_scale_bytes + b_subblk_payload;
    constexpr size_t b_superblk_data_stride = b_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    // B stores K32 scale/payload subblocks, followed by zp[subblock][column].
    constexpr size_t b_zp_subblk_stride     = NB_COLS * sizeof(uint8_t);
    constexpr size_t b_zp_area_bytes        = b_zp_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    constexpr size_t b_superblk_stride      = b_superblk_data_stride + b_zp_area_bytes;
    // A group order: data lanes, subblock scales, a_sum[4][8], average scales.
    constexpr size_t a_data_bytes           = K_SUBBLKS_PER_SUPERBLK * NB_ROWS * k_subblk_len * sizeof(int8_t);
    constexpr size_t a_scale_bytes          = K_SUBBLKS_PER_SUPERBLK * NB_ROWS * sizeof(_Float16);
    constexpr size_t a_sum_bytes            = NB_ROWS * K_SUBBLKS_PER_SUPERBLK * sizeof(_Float16);
    constexpr size_t a_scale_avg_bytes      = NB_ROWS * sizeof(_Float16);
    constexpr size_t a_scale_off            = a_data_bytes;
    constexpr size_t a_sum_off              = a_scale_off + a_scale_bytes;
    constexpr size_t a_scale_avg_abs_off    = a_sum_off + a_sum_bytes;
    constexpr size_t a_scale_avg_off        = a_scale_avg_abs_off - a_scale_off;  // Relative to t2.
    constexpr size_t b_2subblk_stride       = 2 * b_subblk_stride;
    constexpr size_t b_next_payload         = b_subblk_stride + b_scale_bytes;
    constexpr size_t b_3rd_payload          = 2 * b_subblk_stride + b_scale_bytes;
    constexpr size_t a_subblk_stride        = NB_ROWS * k_subblk_len * sizeof(int8_t);
    constexpr size_t a_loop_step            = 2 * a_subblk_stride;
    constexpr size_t a_grp_tail_step        = q8_hp_m4al_blk_size - (K_SUBBLKS_PER_SUPERBLK / 2 - 1) * a_loop_step;
    constexpr size_t a_group_alignment      = 64;
    constexpr size_t a_group_bytes          = a_data_bytes + a_scale_bytes + a_sum_bytes + a_scale_avg_bytes;
    static_assert(q8_hp_m4al_blk_size ==
                  (a_group_bytes + a_group_alignment - 1) / a_group_alignment * a_group_alignment,
                  "A payload, scales/sums and average scales with alignment padding");

    const size_t   b_tile_stride = k_blks * b_superblk_stride;
    const _Float16 hp_scale_16   = (_Float16) 16.0f;
    const _Float16 hp_scale_0125 = (_Float16) 0.125f;

    for (size_t ni = 0; ni + NB_COLS <= count_n; ni += NB_COLS) {
        const uint8_t * b_tile = quant_b_data + (ni / NB_COLS) * b_tile_stride;
        float *         dst_c  = c_ptr + ni;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s7, %[B]                  \n\t"
            "li             t1, %[B_SUPERBLK_DATA_STRIDE] \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmv.v.i        v28, 0                    \n\t"
            "vmv.v.i        v29, 0                    \n\t"
            "vmv.v.i        v30, 0                    \n\t"
            "vmv.v.i        v31, 0                    \n\t"

            ".align 4                                 \n\t"
            "_HPZP_BLK%=:                             \n\t"
            // Apply the zp correction before the dot loop reuses these registers.
            "mv             s5, s7                    \n\t"
            "add            s6, s7, t1                \n\t"
            "addi           t2, t6, %[A_SUM_OFF]      \n\t"
            "addi           t3, t6, %[A_SCALE_OFF]    \n\t"
            "flh            fa3, %[A_SCALE_AVG_ABS](t6) \n\t"
            "fmul.h         fa3, fa3, %[HP0125]       \n\t"
            "li             t4, 8                     \n\t"
            "_HPZP_ZP%=:                              \n\t"
            "flh            fa0, 0(t3)                \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "vsetvli        t0, x0, e8, mf4           \n\t"
            "vle8.v         v8, (s6)                  \n\t"
            "vfwcvt.f.xu.v  v10, v8                   \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v16, v12, fa0             \n\t"
            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"
            "vfmul.vv       v10, v10, v16             \n\t"
            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfmul.vf       v12, v10, ft1             \n\t"
            "vfmul.vf       v13, v10, ft2             \n\t"
            "vfmul.vf       v24, v10, ft3             \n\t"
            "vfmul.vf       v25, v10, ft4             \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfwmacc.vf     v28, fa3, v12             \n\t"
            "vfwmacc.vf     v29, fa3, v13             \n\t"
            "vfwmacc.vf     v30, fa3, v24             \n\t"
            "vfwmacc.vf     v31, fa3, v25             \n\t"
            "addi           t3, t3, 8                 \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, %[B_SUBBLK_STRIDE] \n\t"
            "addi           s6, s6, %[B_ZP_SUBBLK_STRIDE] \n\t"
            "addi           t4, t4, -1                \n\t"
            "bgtz           t4, _HPZP_ZP%=            \n\t"
            "mv             s5, s7                    \n\t"

            "addi           t2, t6, %[A_SCALE_OFF]    \n\t"
            "flh            fa1, %[A_SCALE_AVG_OFF](t2) \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmv.v.i        v18, 0                    \n\t"
            "vmv.v.i        v19, 0                    \n\t"
            "vmv.v.i        v20, 0                    \n\t"
            "vmv.v.i        v21, 0                    \n\t"
            "vmv.v.i        v22, 0                    \n\t"
            "vmv.v.i        v23, 0                    \n\t"
            "vmv.v.i        v26, 0                    \n\t"
            "vmv.v.i        v27, 0                    \n\t"
            "li             t4, 3                     \n\t"
            "flh            fa0, 0(t2)                \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "vl1r.v         v24, (t6)                 \n\t"
            "addi           t3, s5, %[B_SCALE_BYTES]  \n\t"
            "vl4r.v         v4, (t3)                  \n\t"

            "_HPZP_SUB%=:                             \n\t"
            HPZP_LOAD_Y
            HPZP_COMPUTE_X
            HPZP_LOAD_X_NEXT
            HPZP_COMPUTE_Y
            "addi           t6, t6, 256               \n\t"
            "addi           t2, t2, 16                \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"
            "addi           t4, t4, -1                \n\t"
            "bgtz           t4, _HPZP_SUB%=           \n\t"

            HPZP_LOAD_Y
            HPZP_COMPUTE_X
            "vsetvli        t0, x0, e16, mf2          \n\t"
            HPZP_COMPUTE_Y
            "addi           t6, t6, %[A_GRP_TAIL_STEP] \n\t"
            "addi           s5, s5, %[B_2SUBBLK_STRIDE] \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfadd.vv       v18, v18, v22             \n\t"
            "vfadd.vv       v19, v19, v23             \n\t"
            "vfadd.vv       v20, v20, v26             \n\t"
            "vfadd.vv       v21, v21, v27             \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"

            "addi           s5, s5, %[B_ZP_AREA_BYTES] \n\t"
            "mv             s7, s5                    \n\t"
            "addi           t5, t5, -1                \n\t"
            "bgtz           t5, _HPZP_BLK%=           \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            :
            : [A] "r"(quant_a_ptr), [B] "r"(b_tile), [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks),
              [HP16] "f"(hp_scale_16), [HP0125] "f"(hp_scale_0125), [A_SCALE_OFF] "i"(a_scale_off),
              [A_SUM_OFF] "i"(a_sum_off), [A_SCALE_AVG_ABS] "i"(a_scale_avg_abs_off),
              [A_SCALE_AVG_OFF] "i"(a_scale_avg_off), [B_SUPERBLK_DATA_STRIDE] "i"(b_superblk_data_stride),
              [B_SCALE_BYTES] "i"(b_scale_bytes), [B_SUBBLK_STRIDE] "i"(b_subblk_stride),
              [B_ZP_SUBBLK_STRIDE] "i"(b_zp_subblk_stride), [B_ZP_AREA_BYTES] "i"(b_zp_area_bytes),
              [B_2SUBBLK_STRIDE] "i"(b_2subblk_stride), [B_N1_PAYLOAD] "i"(b_next_payload),
              [B_N3_PAYLOAD] "i"(b_3rd_payload), [A_SUBBLK_STRIDE] "i"(a_subblk_stride),
              [A_LOOP_STEP] "i"(a_loop_step), [A_GRP_TAIL_STEP] "i"(a_grp_tail_step)
            : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s5", "s6", "s7", "fa0", "fa1", "fa2", "fa3", "fa4",
              "ft1", "ft2", "ft3", "ft4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
              "v27", "v28", "v29", "v30", "v31", "memory");
    }
}

#undef HPZP_COMPUTE_Y
#undef HPZP_COMPUTE_X
#undef HPZP_COMPUTE
#undef HPZP_LOAD_X_NEXT
#undef HPZP_LOAD_Y
#undef HPZP_MM8

void gemm_kernel_i8i4_hpzp_m1(const uint8_t * quant_a_ptr,
                              const uint8_t * quant_b_data,
                              float *         c_ptr,
                              size_t          count_n,
                              size_t          k_blks) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t k_subblk_len           = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = BLOCK_QNK_LEN / k_subblk_len;
    constexpr size_t b_scale_bytes          = NB_COLS * sizeof(_Float16);
    constexpr size_t b_subblk_payload       = NB_COLS * k_subblk_len / 2 * sizeof(uint8_t);
    constexpr size_t b_subblk_stride        = b_scale_bytes + b_subblk_payload;
    constexpr size_t b_superblk_data_stride = b_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    // B stores K32 scale/payload subblocks, followed by zp[subblock][column].
    constexpr size_t b_zp_subblk_stride     = NB_COLS * sizeof(uint8_t);
    constexpr size_t b_zp_area_bytes        = b_zp_subblk_stride * K_SUBBLKS_PER_SUPERBLK;
    constexpr size_t b_superblk_stride      = b_superblk_data_stride + b_zp_area_bytes;
    // A K256 row: 8 x [FP16 scale, 32-byte data], FP16 a_sum[8], FP16 average scale.
    constexpr size_t a_sum_off              = K_SUBBLKS_PER_SUPERBLK * (sizeof(_Float16) + k_subblk_len * sizeof(int8_t));
    constexpr size_t a_scale_avg_off        = a_sum_off + K_SUBBLKS_PER_SUPERBLK * sizeof(_Float16);
    constexpr size_t a_row_blk_stride       = a_scale_avg_off + sizeof(_Float16);

    const size_t   b_tile_stride   = k_blks * b_superblk_stride;
    const _Float16 hp_scale_0125   = (_Float16) 0.125f;

    for (size_t ni = 0; ni < count_n; ni += 32) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             t4, %[BK]               \n\t"
            "li             t1, %[B_SUPERBLK_DATA_STRIDE] \n\t"
            "add            s6, %[B], t1            \n\t"
            "li             t0, 0x4c00              \n\t"  // FP16 16.
            "fmv.h.x        fa0, t0                 \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            "li             t5, 8                   \n\t"
            "flh            ft1, %[A_SCALE_AVG_OFF](%[A]) \n\t"
            "fmul.h         ft2, ft1, %[HP0125]     \n\t"
            "addi           t6, %[A], %[A_SUM_OFF]  \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v16, v18, v18           \n\t"
            "vxor.vv        v17, v18, v18           \n\t"
            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v19, v18, v18           \n\t"

            "INNER_BLK_LOOP%=:                      \n\t"
            "flh            fa1, (t6)               \n\t"
            "addi           t6, t6, 2               \n\t"
            "flh            ft0, (%[A])             \n\t"
            "addi           %[A], %[A], 2           \n\t"
            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v3, (%[A])              \n\t"
            "addi           %[A], %[A], 32          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v8, (%[B])              \n\t"
            "addi           %[B], %[B], %[B_SCALE_BYTES] \n\t"
            "vl4r.v         v4, (%[B])              \n\t"
            "addi           %[B], %[B], %[B_PAYLOAD] \n\t"
            "vfmul.vf       v8, v8, ft0             \n\t"
            "vfmul.vf       v9, v8, fa0             \n\t"

            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v10, (s6)               \n\t"
            "addi           s6, s6, %[B_ZP_SUBBLK_STRIDE] \n\t"
            "vfwcvt.f.xu.v  v12, v10                \n\t"
            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vfmul.vv       v12, v12, v8            \n\t"
            "vfmul.vf       v12, v12, fa1           \n\t"
            "vfwmacc.vf     v31, ft2, v12           \n\t"

            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v0, v8, v9, 3           \n\t"
            "vsrl.vi        v28, v3, 4              \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vnpack4.vv     v2, v3, v3, 3           \n\t"
            "vnpack4.vv     v3, v28, v28, 3         \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"
            "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
            "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
            "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
            "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"
            "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
            "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
            "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"

            "addi           t5, t5, -1              \n\t"
            "bgtz           t5, INNER_BLK_LOOP%=    \n\t"

            "vpack.vv       v8, v16, v17, 1         \n\t"
            "vpack.vv       v12, v18, v19, 1        \n\t"
            "vpack.vv       v20, v8, v12, 2         \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t4, t4, -1              \n\t"
            "vfwmacc.vf     v31, ft1, v20           \n\t"

            "addi           %[A], t6, %[A_ROW_TAIL]  \n\t"
            "addi           %[B], %[B], %[B_ZP_AREA_BYTES] \n\t"
            "add            s6, %[B], t1            \n\t"

            "bgtz           t4, BLK_LOOP%=          \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks), [HP0125] "f"(hp_scale_0125),
              [A_SUM_OFF] "i"(a_sum_off), [A_SCALE_AVG_OFF] "i"(a_scale_avg_off), [A_ROW_TAIL] "i"(a_row_blk_stride - a_scale_avg_off),
              [B_SUPERBLK_DATA_STRIDE] "i"(b_superblk_data_stride), [B_SCALE_BYTES] "i"(b_scale_bytes),
              [B_PAYLOAD] "i"(b_subblk_payload), [B_ZP_SUBBLK_STRIDE] "i"(b_zp_subblk_stride),
              [B_ZP_AREA_BYTES] "i"(b_zp_area_bytes)
            : "t0", "t1", "t4", "t5", "t6", "s6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "ft0", "ft1", "ft2", "memory");
    }
}

void gemm_kernel_i8i5_m1(const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_n,
                         size_t          k_blks) {
    constexpr size_t blk_len = 32;
    // =========================================================================
    // i8i5: 8-bit activation × 5-bit weight (4-bit low + 1-bit high mask)
    //
    // B layout per N32K32 k-block (no-zp):
    //   [0  .. 63 ] : scale_fp16 × 32              (64B)
    //   [64 .. 191] : Bh i1-high-bit  × 32N × 32K  (128B = 1 VRF)
    //   [192.. 703] : Bs i4-low-nibble × 32N × 32K (512B = 4 VRF)
    //   Total: 704B per k-block stride
    //
    // B layout per N32K32 k-block (with-zp):
    //   [0  .. 63 ] : scale_fp16 × 32              (64B)
    //   [64 .. 95 ] : zp_uint8 × 32                (32B)
    //   [96 .. 223] : Bh i1-high-bit  × 32N × 32K  (128B = 1 VRF)
    //   [224.. 735] : Bs i4-low-nibble × 32N × 32K (512B = 4 VRF)
    //   Total: 736B per k-block stride
    //
    // Bh format per N8K32 sub-block (32B):
    //   K rows × N cols × 1bit packed as bytes (8 cols per byte, K groups of 4B)
    //   Byte k gives 8 mask bits for columns N7..N0 at k-th K-element.
    //
    // Computation:
    //   B5bit_signed = (Bs | (Bh << 4)) - zp
    //   dot(A, B5) = dot(A, Bs_u4) + 16*dot(A, Bh_u1) - zp*asum
    //   No-zp: implicit zp = 16 (unsigned [0..31] centered at 16)
    //   With-zp: explicit zp from data
    //
    // =========================================================================

    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            // i8i5 no-zp: per column per k-block stride = fp16(2B) + i4(16B) + i1(4B) = 22B
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * (blk_len / 8) +    // Bh i1 mask: n×k_blks×4
                                      n * k_blks * blk_len / 2 +      // Bs i4 data: n×k_blks×16
                                      n * k_blks * sizeof(_Float16);  // scale: n×k_blks×2
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format (same as i8i4):
            //   || scl(fp32,4B) | asum(int16,2B) | data(int8,32B) || × k_blks
            //
            // Register map:
            //   t3 = k_blks loop counter   t4 = nblks (tail)
            //   t2 = A asum (int16) << 4   f0 = A scale (fp32)
            //   s2 = pA (scale/asum)       s3 = pA data
            //   s4 = pB scales (fp16×32)
            //   s5 = pB Bh (i1 mask, 128B)
            //   s6 = pB Bs (i4 packed, 512B)
            //   s7 = pC
            //   v3  = fp32 accumulator (N32)
            //   v2  = B scales fp16 (loaded as bytes; later widened)
            //   v0  = Bh mask bytes (also used as v0.t mask after load)
            //   v1  = A int8 (K32)
            //   v8..v15 / v16..v23 = Bs unpack/pack temporaries (build b5bit bytes)
            //   v24/v26/v28/v30    = int32 dot accumulators & packing temps

            __asm__ volatile(
                "mv           t3, %[BCK]              \n\t"  // t3 = k_blks
                "mv           t4, %[NBLKS]            \n\t"  // t4 = nblks (tail guard)

                // ---- pre-loop: init fp16 constants in e16 m1 context ----
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // v0 = int16(1)
                "vfcvt.f.x.v  v0, v0                  \n\t"  // v0  = 1.0_fp16
                "vxor.vv      v3, v16, v16            \n\t"

                // ---- pointer setup ----
                "mv           s2, %[pA]               \n\t"  // s2 = pA (scale, fp32)
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pA data (skip scale+asum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBh  (pB + 64B scale)
                "addi         s6, %[pB], 32*2+128     \n\t"  // s6 = pBs  (pB + 64 + 128 = pB+192)
                "mv           s7, %[pC]               \n\t"  // s7 = pC

                // =====================================================================
                // K-block loop: each iteration processes one N32×K32 block
                // Stride per k-block = 704B = 64(scl) + 512(Bs) + 128(Bh)
                // =====================================================================
                ".align 4                             \n\t"
                "BLK_LPST%=:                          \n\t"

                // ---- load Bs (512B = 4 VRF) from s6, advance s6 by 704 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v8, (s6)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi         s6, s6, 128*4+128+64    \n\t"  // s6 += 704 (512+128+64)

                // ---- load B scale (64B = 32×fp16) from s4, advance s4 by 704 ----
                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v2, (s4)                \n\t"  // v2 = scale_fp16 × 32
                "addi         s4, s4, 64+128*4+128    \n\t"  // s4 += 704 (64+512+128)

                // ---- load Bh (128B = 1 VRF) from s5, advance s5 by 704 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v0, (s5)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi         s5, s5, 128+64+128*4    \n\t"  // s5 += 704 (128+64+512)

                // ---- load A data (32B = K32 int8) from s3 ----
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v1, (s3)                \n\t"  // v1 = A M1K32 int8
                "addi         s3, s3, 32+6            \n\t"  // s3 += 38 (data + scl + asum)

                // ---- load A scale (fp32) and asum (int16) from s2 ----
                "flw          f0, (s2)                \n\t"  // f0 = A scale (fp32)
                "lh           t2, 4(s2)               \n\t"  // t2 = A asum (int16)
                "addi         s2, s2, 6+32            \n\t"  // s2 += 38

                //// ---- A nibble unpacking ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vand.vi      v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi      v13, v9, 0xF            \n\t"
                "vand.vi      v14, v10, 0xF           \n\t"
                "vand.vi      v15, v11, 0xF           \n\t"
                "vsrl.vi      v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi      v9, v9, 4               \n\t"
                "vsrl.vi      v10, v10, 4             \n\t"
                "vsrl.vi      v11, v11, 4             \n\t"

                "slli         t2, t2, 4               \n\t"  // a_sum * 16;
                // [4*32]*2
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v12, v8, 0         \n\t"
                "vpack.vv     v18, v13, v9, 0         \n\t"
                "vpack.vv     v20, v14, v10, 0        \n\t"
                "vpack.vv     v22, v15, v11, 0        \n\t"

                "li           t1, 16                  \n\t"
                "vsetvli      t0, x0, e8, m8          \n\t"
                "vadd.vx      v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vupack.vv    v8, v16, v17, 1         \n\t"
                "vupack.vv    v10, v18, v19, 1        \n\t"
                "vupack.vv    v12, v20, v21, 1        \n\t"
                "vupack.vv    v14, v22, v23, 1        \n\t"

                "vsetvli      t0, x0, e64, m1         \n\t"
                "vslidedown.vi  v16, v1, 2            \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v24, v16, v16   \n\t"
                "vxor.vv        v26, v16, v16   \n\t"
                "vxor.vv        v28, v16, v16   \n\t"
                "vxor.vv        v30, v16, v16   \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot       v24, v1, v8, i8         \n\t"  // N0..7
                "vmadot       v26, v1, v10, i8        \n\t"  // N8..15
                "vmadot       v28, v1, v12, i8        \n\t"  // N16..23
                "vmadot       v30, v1, v14, i8        \n\t"  // N24..31
                //// vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot       v24, v16, v9, i8        \n\t"  // N0..7
                "vmadot       v26, v16, v11, i8       \n\t"  // N8..15
                "vmadot       v28, v16, v13, i8       \n\t"  // N16..23
                "vmadot       v30, v16, v15, i8       \n\t"  // N24..31

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv     v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv     v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vadd.vx      v24, v24, t2            \n\t"
                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v28, v2                 \n\t"

                // a_scale * b_scale;
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v24                \n\t"
                "vfmul.vf     v30, v28, f0            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v3, v30, v26            \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, BLK_LPST%=          \n\t"
                "BLK_LPND%=:                          \n\t"
                "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
                "vse32.v      v3, (%[pC])             \n\t"
                "FUNC_END%=:                          \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "s7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            // i8i5 with-zp: per column per k-block stride = fp16(2B)+zp(1B)+i4(16B)+i1(4B)=23B
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // Bs i4: n×k_blks×16
                                      n * k_blks * (blk_len / 8) +    // Bh i1: n×k_blks×4
                                      n * k_blks * sizeof(uint8_t) +  // zp: n×k_blks×1
                                      n * k_blks * sizeof(_Float16);  // scale: n×k_blks×2
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format (same as i8i4):
            //   || scl(fp32,4B) | asum(int16,2B) | data(int8,32B) || × k_blks
            //
            // Register map:
            //   t3 = k_blks loop counter   t4 = nblks (tail)
            //   t2 = A asum (int16) << 4   f0 = A scale (fp32)
            //   s2 = pA (scale/asum)       s3 = pA data
            //   s4 = pB scales (fp16×32); 每个 k-block 先 +64 指向 zp，再 +672 到下一个 block
            //   s5 = pB Bh (i1 mask, 128B) (offset +96)
            //   s6 = pB Bs (i4 packed, 512B) (offset +224)
            //   s7 = pC
            //   v3  = fp32 accumulator (N32)
            //   v2  = B scales fp16 (loaded as bytes; later widened)
            //   v0  = Bh mask bytes (also used as v0.t mask after load)
            //   v1  = A int8 (K32) / later reused to hold Bzp bytes
            //   v8..v15 / v16..v23 = Bs unpack/pack temporaries (build b5bit bytes)
            //   v24/v26/v28/v30    = int32 dot accumulators & packing temps

            __asm__ volatile(
                "mv           t3, %[BCK]              \n\t"  // t3 = k_blks
                "mv           t4, %[NBLKS]            \n\t"  // t4 = nblks (tail guard)

                // ---- pre-loop: init fp16 constants in e16 m1 context ----
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // v0 = int16(1)
                "vfcvt.f.x.v  v0, v0                  \n\t"  // v0  = 1.0_fp16
                "vxor.vv      v3, v16, v16            \n\t"

                // ---- pointer setup ----
                "mv           s2, %[pA]               \n\t"  // s2 = pA (scale, fp32)
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pA data (skip scale+asum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBh  (pB + 64B scale + 32B zp = pB+96)
                "addi         s6, %[pB], 32*3+128     \n\t"  // s6 = pBs  (pB + 96 + 128 = pB+224)
                "mv           s7, %[pC]               \n\t"  // s7 = pC

                // =====================================================================
                // K-block loop: each iteration processes one N32×K32 block
                // Stride per k-block = 736B = 64(scale) + 32(zp) + 128(Bh) + 512(Bs)
                // =====================================================================
                ".align 4                             \n\t"
                "BLK_LPST%=:                          \n\t"

                // ---- load Bs (512B = 4 VRF) from s6, advance s6 by 736 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v8, (s6)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi         s6, s6, 128*4+128+96    \n\t"  // s6 += 736 (512+128+96)

                // ---- load B scale (64B = 32×fp16) from s4; then s4 points to zp[32] ----
                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v2, (s4)                \n\t"  // v2 = scale_fp16 × 32
                "addi         s4, s4, 64              \n\t"  // s4 += 64 (now points to zp)

                // ---- load Bh (128B = 1 VRF) from s5, advance s5 by 736 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v0, (s5)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi         s5, s5, 128+96+128*4    \n\t"  // s5 += 736 (128+96+512)

                // ---- load A data (32B = K32 int8) from s3 ----
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v1, (s3)                \n\t"  // v1 = A M1K32 int8
                "addi         s3, s3, 32+6            \n\t"  // s3 += 38 (data + scl + asum)

                // ---- load A scale (fp32) and asum (int16) from s2 ----
                "flw          f0, (s2)                \n\t"  // f0 = A scale (fp32)
                "lh           t2, 4(s2)               \n\t"  // t2 = A asum (int16)
                "addi         s2, s2, 6+32            \n\t"  // s2 += 38

                //// ---- A nibble unpacking ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vand.vi      v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi      v13, v9, 0xF            \n\t"
                "vand.vi      v14, v10, 0xF           \n\t"
                "vand.vi      v15, v11, 0xF           \n\t"
                "vsrl.vi      v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi      v9, v9, 4               \n\t"
                "vsrl.vi      v10, v10, 4             \n\t"
                "vsrl.vi      v11, v11, 4             \n\t"

                // [4*32]*2
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v12, v8, 0         \n\t"
                "vpack.vv     v18, v13, v9, 0         \n\t"
                "vpack.vv     v20, v14, v10, 0        \n\t"
                "vpack.vv     v22, v15, v11, 0        \n\t"

                "li           t1, 16                  \n\t"
                "vsetvli      t0, x0, e8, m8          \n\t"
                "vadd.vx      v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vupack.vv    v8, v16, v17, 1         \n\t"
                "vupack.vv    v10, v18, v19, 1        \n\t"
                "vupack.vv    v12, v20, v21, 1        \n\t"
                "vupack.vv    v14, v22, v23, 1        \n\t"

                "vsetvli      t0, x0, e64, m1         \n\t"
                "vslidedown.vi  v16, v1, 2            \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v24, v16, v16           \n\t"
                "vxor.vv      v26, v16, v16           \n\t"
                "vxor.vv      v28, v16, v16           \n\t"
                "vxor.vv      v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot       v24, v1, v8, i8         \n\t"  // N0..7
                "vmadot       v26, v1, v10, i8        \n\t"  // N8..15
                "vmadot       v28, v1, v12, i8        \n\t"  // N16..23
                "vmadot       v30, v1, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot       v24, v16, v9, i8        \n\t"  // N0..7
                "vmadot       v26, v16, v11, i8       \n\t"  // N8..15
                "vmadot       v28, v16, v13, i8       \n\t"  // N16..23
                "vmadot       v30, v16, v15, i8       \n\t"  // N24..31

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4+128    \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv     v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv     v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v30, v28, t2            \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v28, v2                 \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v24, v24, v30           \n\t"

                // a_scale * b_scale;
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfmul.vf     v30, v28, f0            \n\t"
                "vfcvt.f.x.v  v26, v24                \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v3, v30, v26            \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, BLK_LPST%=          \n\t"
                "BLK_LPND%=:                          \n\t"
                "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
                "vse32.v      v3, (%[pC])             \n\t"
                "FUNC_END%=:                          \n\t"
                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "s7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
    }
}

void gemm_kernel_i8i5_m4(const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    constexpr size_t K_BLK_LEN        = 32;
    constexpr size_t NB_COLS          = 32;

    // This kernel computes a 4x32 output tile. For each K32 block we decode the
    // packed Q5 weights once and reuse the decoded vectors across the 4 A rows.
    constexpr size_t B_SCALE_BYTES    = NB_COLS * sizeof(_Float16);
    constexpr size_t B_ZP_BYTES       = NB_COLS * sizeof(uint8_t);
    constexpr size_t B_QH_BYTES       = NB_COLS * K_BLK_LEN / 8 * sizeof(uint8_t);
    constexpr size_t B_QS_BYTES       = NB_COLS * K_BLK_LEN / 2 * sizeof(uint8_t);
    constexpr size_t B_Q50_BLK_STRIDE = B_SCALE_BYTES + B_QH_BYTES + B_QS_BYTES;
    constexpr size_t B_Q51_BLK_STRIDE = B_SCALE_BYTES + B_ZP_BYTES + B_QH_BYTES + B_QS_BYTES;
    constexpr size_t A_DATA_OFF       = 4 * (sizeof(float) + sizeof(int16_t));
    constexpr size_t A_BLK_STRIDE     = A_DATA_OFF + 4 * K_BLK_LEN * sizeof(int8_t);
    static_assert(B_Q50_BLK_STRIDE == sizeof(nrow_block_q5_0<NB_COLS>) &&
                  B_Q51_BLK_STRIDE == sizeof(nrow_block_q5_1<NB_COLS>), "Q5 packed block layout");

    if (quant_b_zp) {
        // Q5_1 block layout per K32/N32 tile:
        //   [scale_fp16 x 32][zp_u8 x 32][qh high-bit mask x 128B][qs low nibbles x 512B]
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q51_BLK_STRIDE;
            uint8_t * a_data  = (uint8_t *) quant_a_ptr;
            float *   dst_c   = c_ptr + ni;
            size_t    cnt     = k_blks;

            asm volatile(
                // v4-v7 are the fp32 accumulators for rows 0..3 of the current N32 tile.
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v4, v4, v4              \n\t"
                "vxor.vv        v5, v5, v5              \n\t"
                "vxor.vv        v6, v6, v6              \n\t"
                "vxor.vv        v7, v7, v7              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // Load the 4 A-row scales/sums for this K32 block and build row data pointers.
                "flw            fa0, 0(%[A])            \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "lh             s1, 16(%[A])            \n\t"
                "lh             s2, 18(%[A])            \n\t"
                "lh             s3, 20(%[A])            \n\t"
                "lh             s4, 22(%[A])            \n\t"
                "addi           t3, %[A], %[A_DATA_OFF]            \n\t"
                "addi           t4, t3, 32              \n\t"
                "addi           t5, t3, 64              \n\t"
                "addi           t6, t3, 96              \n\t"
                "addi           %[A], %[A], %[A_BLK_STRIDE]         \n\t"

                // B-side pointers:
                //   t1 -> zp stream, t2 -> qh bitmask stream, s5 -> qs low-nibble stream.
                "addi           t1, %[B], %[B_SCALE_BYTES]            \n\t"
                "addi           t2, %[B], %[B_Q51_QH_OFF]            \n\t"
                "addi           s5, %[B], %[B_Q51_QS_OFF]           \n\t"
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v2, (%[B])              \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t2)                \n\t"
                "vl4r.v         v8, (s5)                \n\t"
                "addi           %[B], %[B], %[B_BLK_STRIDE]         \n\t"

                // Decode Q5 payload once for the whole tile:
                //   1) split `qs` low/high nibbles,
                //   2) repack into bytes,
                //   3) use the `qh` mask to inject bit4 (+16) where needed,
                //   4) expand into the vmadot-friendly layout reused by all 4 rows.
                "vand.vi        v12, v8, 0xF            \n\t"
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "li             t2, 16                  \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t2, v0.t      \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                // Convert per-column fp16 scales once; the same scale vector is shared by all 4 rows.
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v18, v2                 \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v3, (t1)                \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"

                // Row 0: dot(A0, decoded_q5) + a_sum0 * zp, then scale by A/B scales.
                // The widen/mul correction sequence intentionally matches the proven m1 Q5_1 path.
                "vle8.v         v1, (t3)                \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vupack.vv      v16, v1, v2, 1          \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v24, v24           \n\t"
                "vxor.vv        v26, v26, v26           \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vmadot         v24, v16, v8, i8        \n\t"
                "vmadot         v26, v16, v10, i8       \n\t"
                "vmadot         v28, v16, v12, i8       \n\t"
                "vmadot         v30, v16, v14, i8       \n\t"
                "vmadot         v24, v17, v9, i8        \n\t"
                "vmadot         v26, v17, v11, i8       \n\t"
                "vmadot         v28, v17, v13, i8       \n\t"
                "vmadot         v30, v17, v15, i8       \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"
                "vpack.vv       v20, v28, v30, 2        \n\t"
                "vpack.vv       v24, v16, v20, 3        \n\t"
                "vpack.vv       v26, v17, v21, 3        \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vwaddu.vx      v28, v3, x0             \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vwmul.vx       v12, v28, s1            \n\t"
                "vwmul.vx       v14, v28, s2            \n\t"
                "vwmul.vx       v20, v28, s3            \n\t"
                "vwmul.vx       v22, v28, s4            \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vadd.vv        v24, v24, v12           \n\t"
                "vadd.vv        v25, v25, v14           \n\t"
                "vadd.vv        v26, v26, v20           \n\t"
                "vadd.vv        v27, v27, v22           \n\t"
                "vfcvt.f.x.v    v12, v24                \n\t"
                "vfcvt.f.x.v    v14, v25                \n\t"
                "vfcvt.f.x.v    v20, v26                \n\t"
                "vfcvt.f.x.v    v22, v27                \n\t"
                "vfmul.vv       v12, v12, v18           \n\t"
                "vfmul.vv       v14, v14, v18           \n\t"
                "vfmul.vv       v20, v20, v18           \n\t"
                "vfmul.vv       v22, v22, v18           \n\t"
                "vfmacc.vf      v4, fa0, v12            \n\t"
                "vfmacc.vf      v5, fa1, v14            \n\t"
                "vfmacc.vf      v6, fa2, v20            \n\t"
                "vfmacc.vf      v7, fa3, v22            \n\t"

                "addi           %[BK], %[BK], -1        \n\t"
                "bgtz           %[BK], BLK_LOOP%=       \n\t"

                // Tail-aware store for the final N tile (`nb_real` may be < 32).
                "vsetvli        t0, %[NBLKS], e32, m1   \n\t"
                "add            t1, %[LDC], %[DST]      \n\t"
                "vse32.v        v4, (%[DST])            \n\t"
                "vse32.v        v5, (t1)                \n\t"
                "add            t2, t1, %[LDC]          \n\t"
                "vse32.v        v6, (t2)                \n\t"
                "add            t3, t2, %[LDC]          \n\t"
                "vse32.v        v7, (t3)                \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(cnt)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [A_DATA_OFF] "i"(A_DATA_OFF), [A_BLK_STRIDE] "i"(A_BLK_STRIDE),
                  [B_SCALE_BYTES] "i"(B_SCALE_BYTES), [B_BLK_STRIDE] "i"(B_Q51_BLK_STRIDE),
                  [B_Q51_QH_OFF] "i"(B_SCALE_BYTES + B_ZP_BYTES),
                  [B_Q51_QS_OFF] "i"(B_SCALE_BYTES + B_ZP_BYTES + B_QH_BYTES), [NBLKS] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "s5", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                  "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
                  "v31", "fa0", "fa1", "fa2", "fa3");
        }
    } else {
        // Q5_0 block layout per K32/N32 tile:
        //   [scale_fp16 x 32][qh high-bit mask x 128B][qs low nibbles x 512B]
        // There is no explicit zp stream; the implicit midpoint correction is +16.
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q50_BLK_STRIDE;
            uint8_t * a_data  = (uint8_t *) quant_a_ptr;
            float *   dst_c   = c_ptr + ni;
            size_t    cnt     = k_blks;

            asm volatile(
                // v4-v7 are the fp32 accumulators for rows 0..3 of the current N32 tile.
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v4, v4, v4              \n\t"
                "vxor.vv        v5, v5, v5              \n\t"
                "vxor.vv        v6, v6, v6              \n\t"
                "vxor.vv        v7, v7, v7              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // Load the 4 A-row scales/sums for this K32 block and build row data pointers.
                "flw            fa0, 0(%[A])            \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "lh             s1, 16(%[A])            \n\t"
                "lh             s2, 18(%[A])            \n\t"
                "lh             s3, 20(%[A])            \n\t"
                "lh             s4, 22(%[A])            \n\t"
                "addi           t3, %[A], %[A_DATA_OFF]            \n\t"
                "addi           t4, t3, 32              \n\t"
                "addi           t5, t3, 64              \n\t"
                "addi           t6, t3, 96              \n\t"
                "addi           %[A], %[A], %[A_BLK_STRIDE]         \n\t"

                // B-side pointers:
                //   t1 -> qh bitmask stream, t2 -> qs low-nibble stream.
                "addi           t1, %[B], %[B_SCALE_BYTES]            \n\t"
                "addi           t2, %[B], %[B_Q50_QS_OFF]           \n\t"
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v2, (%[B])              \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t1)                \n\t"
                "vl4r.v         v8, (t2)                \n\t"
                "addi           %[B], %[B], %[B_BLK_STRIDE]         \n\t"

                // Decode Q5 payload once for the whole tile and expand it into the vmadot layout.
                "vand.vi        v12, v8, 0xF            \n\t"
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "li             t2, 16                  \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t2, v0.t      \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                // Convert per-column fp16 scales once; the same scale vector is shared by all 4 rows.
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v18, v2                 \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"

                // Row 0: dot(A0, decoded_q5) + a_sum0 * 16 (implicit Q5_0 midpoint correction).
                "vle8.v         v1, (t3)                \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vupack.vv      v16, v1, v2, 1          \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v24, v24           \n\t"
                "vxor.vv        v26, v26, v26           \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vmadot         v24, v16, v8, i8        \n\t"
                "vmadot         v26, v16, v10, i8       \n\t"
                "vmadot         v28, v16, v12, i8       \n\t"
                "vmadot         v30, v16, v14, i8       \n\t"
                "vmadot         v24, v17, v9, i8        \n\t"
                "vmadot         v26, v17, v11, i8       \n\t"
                "vmadot         v28, v17, v13, i8       \n\t"
                "vmadot         v30, v17, v15, i8       \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"
                "slli           s1, s1, 4               \n\t"
                "vpack.vv       v20, v28, v30, 2        \n\t"
                "slli           s2, s2, 4               \n\t"
                "vpack.vv       v24, v16, v20, 3        \n\t"
                "slli           s3, s3, 4               \n\t"
                "vpack.vv       v26, v17, v21, 3        \n\t"
                "slli           s4, s4, 4               \n\t"
                "vadd.vx        v24, v24, s1            \n\t"
                "vadd.vx        v25, v25, s2            \n\t"
                "vadd.vx        v26, v26, s3            \n\t"
                "vadd.vx        v27, v27, s4            \n\t"
                "vfcvt.f.x.v    v24, v24                \n\t"
                "vfcvt.f.x.v    v25, v25                \n\t"
                "vfcvt.f.x.v    v26, v26                \n\t"
                "vfcvt.f.x.v    v27, v27                \n\t"
                "vfmul.vv       v24, v24, v18           \n\t"
                "vfmul.vv       v25, v25, v18           \n\t"
                "vfmul.vv       v26, v26, v18           \n\t"
                "vfmul.vv       v27, v27, v18           \n\t"
                "vfmacc.vf      v4, fa0, v24            \n\t"
                "vfmacc.vf      v5, fa1, v25            \n\t"
                "vfmacc.vf      v6, fa2, v26            \n\t"
                "vfmacc.vf      v7, fa3, v27            \n\t"

                "addi           %[BK], %[BK], -1        \n\t"
                "bgtz           %[BK], BLK_LOOP%=       \n\t"

                "vsetvli        t0, %[NBLKS], e32, m1   \n\t"
                "add            t1, %[LDC], %[DST]      \n\t"
                "vse32.v        v4, (%[DST])            \n\t"
                "vse32.v        v5, (t1)                \n\t"
                "add            t2, t1, %[LDC]          \n\t"
                "vse32.v        v6, (t2)                \n\t"
                "add            t3, t2, %[LDC]          \n\t"
                "vse32.v        v7, (t3)                \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(cnt)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [A_DATA_OFF] "i"(A_DATA_OFF), [A_BLK_STRIDE] "i"(A_BLK_STRIDE),
                  [B_SCALE_BYTES] "i"(B_SCALE_BYTES), [B_BLK_STRIDE] "i"(B_Q50_BLK_STRIDE),
                  [B_Q50_QS_OFF] "i"(B_SCALE_BYTES + B_QH_BYTES), [NBLKS] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "v0", "v1", "v2",
                  "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                  "fa0", "fa1", "fa2", "fa3");
        }
    }
}

static void gemm_kernel_i8i8_m4(const uint8_t * quant_a_ptr,
                                const uint8_t * quant_b_data,
                                float *         c_ptr,
                                size_t          count_n,
                                size_t          k_blks,
                                size_t          ldc) {
    constexpr size_t blk_len = 32;
    int64_t b_data_stride = k_blks * sizeof(ggml_fp16_t) + k_blks * blk_len;
    for (size_t ni = 0; ni < count_n; ni += 32) {
        uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = c_ptr + ni;
        size_t    bk     = k_blks;

        asm volatile(
            "vsetvli        t0, x0, e32, m1       \n\t"
            "vxor.vv        v28, v28, v28         \n\t"
            "vxor.vv        v29, v29, v29         \n\t"
            "vxor.vv        v30, v30, v30         \n\t"
            "vxor.vv        v31, v31, v31         \n\t"

            ".align 4                             \n\t"
            "BLK_LOOP%=:                          \n\t"
            "flw            fa0, (%[A])           \n\t"
            "flw            fa1, 4(%[A])          \n\t"
            "flw            fa2, 8(%[A])          \n\t"
            "flw            fa3, 12(%[A])         \n\t"
            "addi           %[A], %[A], %[AHDR]   \n\t"

            "vsetvli        t0, x0, e16, mf2      \n\t"
            "vle16.v        v12, (%[B])           \n\t"
            "addi           %[B], %[B], 64        \n\t"
            "vfwcvt.f.f.v   v14, v12              \n\t"

            "vsetvli        t0, x0, e8, m1        \n\t"
            "vl1r.v         v0, (%[A])            \n\t"
            "addi           %[A], %[A], 128       \n\t"
            "vl4r.v         v24, (%[B])           \n\t"
            "addi           %[B], %[B], 512       \n\t"
            "vl4r.v         v4, (%[B])            \n\t"
            "addi           %[B], %[B], 512       \n\t"

            "vsetvli        t0, zero, e32, m1     \n\t"
            "vupack.vv      v2, v0, v0, 1         \n\t"

            "vxor.vv        v16, v16, v16         \n\t"
            "vxor.vv        v18, v16, v16         \n\t"
            "vxor.vv        v20, v16, v16         \n\t"
            "vxor.vv        v22, v16, v16         \n\t"

            "vmadot         v16, v2, v24, i8      \n\t"
            "vmadot         v18, v2, v26, i8      \n\t"
            "vmadot         v20, v2, v4, i8       \n\t"
            "vmadot         v22, v2, v6, i8       \n\t"
            "vmadot         v16, v3, v25, i8      \n\t"
            "vmadot         v18, v3, v27, i8      \n\t"
            "vmadot         v20, v3, v5, i8       \n\t"
            "vmadot         v22, v3, v7, i8       \n\t"

            "vpack.vv       v0, v16, v18, 2       \n\t"
            "vpack.vv       v2, v20, v22, 2       \n\t"
            "vpack.vv       v16, v0, v2, 3        \n\t"
            "vpack.vv       v18, v1, v3, 3        \n\t"

            "vfcvt.f.x.v    v16, v16              \n\t"
            "vfcvt.f.x.v    v17, v17              \n\t"
            "vfcvt.f.x.v    v18, v18              \n\t"
            "vfcvt.f.x.v    v19, v19              \n\t"

            "vfmul.vv       v16, v16, v14         \n\t"
            "vfmul.vv       v17, v17, v14         \n\t"
            "vfmul.vv       v18, v18, v14         \n\t"
            "vfmul.vv       v19, v19, v14         \n\t"

            "addi           %[BK], %[BK], -1      \n\t"
            "vfmacc.vf      v28, fa0, v16         \n\t"
            "vfmacc.vf      v29, fa1, v17         \n\t"
            "vfmacc.vf      v30, fa2, v18         \n\t"
            "vfmacc.vf      v31, fa3, v19         \n\t"

            "bgtz           %[BK], BLK_LOOP%=     \n\t"

            "vsetvli        t0, x0, e32, m1       \n\t"
            "add            t2, %[LDC], %[DST]    \n\t"
            "vse32.v        v28, (%[DST])         \n\t"
            "add            t3, %[LDC], t2        \n\t"
            "vse32.v        v29, (t2)             \n\t"
            "add            t2, %[LDC], t3        \n\t"
            "vse32.v        v30, (t3)             \n\t"
            "vse32.v        v31, (t2)             \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(bk)
            : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [AHDR] "i"(64)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "memory");
    }
}

static void gemm_kernel_i8i8_m1(const uint8_t * quant_a_ptr,
                                const uint8_t * quant_b_data,
                                float *         c_ptr,
                                size_t          count_n,
                                size_t          k_blks) {
    constexpr size_t blk_len = 32;
    for (size_t n = 0; n < count_n; n += 32) {
        size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
        uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data + n * k_blks * blk_len + n * k_blks * sizeof(_Float16);
        float *   CPtr          = c_ptr + n;
        size_t    cnt           = k_blks;

        __asm__ volatile(
            "mv           t3, %[BCK]              \n\t"
            "mv           t4, %[NBLKS]            \n\t"
            "mv           s2, %[pA]               \n\t"
            "addi         s3, %[pA], 4+2          \n\t"
            "mv           s4, %[pB]               \n\t"
            "addi         s5, %[pB], 32*2         \n\t"
            "mv           s6, %[pC]               \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v2, v0, v0              \n\t"

            ".align 4                             \n\t"
            "_K_LPST%=:                           \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vl4r.v       v24, (s5)               \n\t"
            "addi         s5, s5, 128*4           \n\t"
            "vl4r.v       v28, (s5)               \n\t"
            "addi         s5, s5, 128*4+64        \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s4)                \n\t"
            "addi         s4, s4, 64+128*8        \n\t"

            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vle8.v       v3, (s3)                \n\t"
            "addi         s3, s3, 32+6            \n\t"

            "flw          f0, (s2)                \n\t"
            "addi         s2, s2, 6+32            \n\t"

            "vsetvli      t0, zero, e32, m1       \n\t"
            "vslidedown.vi  v4, v3, 4             \n\t"

            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v24, i8         \n\t"
            "vmadot       v18, v3, v26, i8         \n\t"
            "vmadot       v20, v3, v28, i8         \n\t"
            "vmadot       v22, v3, v30, i8         \n\t"

            "vmadot       v16, v4, v25, i8         \n\t"
            "vmadot       v18, v4, v27, i8         \n\t"
            "vmadot       v20, v4, v29, i8         \n\t"
            "vmadot       v22, v4, v31, i8         \n\t"

            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v24, v0                 \n\t"
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v  v26, v16                \n\t"
            "vfmul.vf     v1, v24, f0             \n\t"
            "vfmacc.vv    v2, v1, v26             \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, _K_LPST%=           \n\t"

            "vsetvli      t0, t4, e32, m1         \n\t"
            "vse32.v      v2, (s6)                \n\t"
            :
            : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
            : "cc", "t0", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "v0", "v1", "v2", "v3", "v4", "v16", "v17",
              "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "memory");
    }
}

size_t gemm_i8i8_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc) {
    GGML_UNUSED(quant_b_zp);
    GGML_ASSERT(blk_len == 32);
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4_al);
        gemm_kernel_i8i8_m4(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i8_m1(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks);
    return 1;
}

void moe_gemm_kernel_i8i4_m2_impl(const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_n,
                                  size_t           k_blks) {
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t3, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A0
                "flw            fa0, (%[A0])            \n\t"  // A0 scale
                "lh             t1, 4(%[A0])            \n\t"  // A0 asum
                "addi           %[A0], %[A0], 6         \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                // load scale A1
                "flw            fa1, (%[A1])            \n\t"  // A1 scale
                "lh             t2, 4(%[A1])            \n\t"  // A1 asum
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmv.v.x        v10, t1                 \n\t"
                "vmv.v.x        v11, t2                 \n\t"

                "vpack.vv       v18, v10, v11, 1        \n\t"
                "vsll.vi        v18, v18, 3             \n\t"  // mul 8
                "vfcvt.f.x.v    v18, v18                \n\t"

                "vsetvli        t0, x0, e8, mf4         \n\t"  // A0 data
                "vle8.v         v16, (%[A0])            \n\t"
                "addi           %[A0], %[A0], 32        \n\t"  // 1*32@i8
                "vle8.v         v20, (%[A1])            \n\t"
                "addi           %[A1], %[A1], 32        \n\t"  // 1*32@i8

                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"

                "vsrl.vi        v17, v16, 4             \n\t"
                "vsrl.vi        v21, v20, 4             \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vnpack4.vv     v2, v16, v20, 2         \n\t"  // low  u4
                "vnpack4.vv     v3, v17, v21, 2         \n\t"  // high s4

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vor.vv         v19, v18, v18           \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t3, t3, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"

                "bgtz           t3, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v28, (%[DST0])          \n\t"
                "vse32.v        v29, (%[DST1])          \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t3, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A0
                "flw            fa0, (%[A0])            \n\t"  // A0 scale
                "lh             t1, 4(%[A0])            \n\t"  // A0 asum
                "addi           %[A0], %[A0], 6         \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                // load scale A1
                "flw            fa1, (%[A1])            \n\t"  // A1 scale
                "lh             t2, 4(%[A1])            \n\t"  // A1 asum
                "addi           %[A1], %[A1], 6         \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v8, (%[B])              \n\t"
                "addi           %[B], %[B], 32          \n\t"
                "vwaddu.vx      v10, v8, x0             \n\t"

                "vsetvli        t0, x0, e8, mf4         \n\t"  // A0 data
                "vle8.v         v16, (%[A0])            \n\t"
                "addi           %[A0], %[A0], 32        \n\t"  // 1*32@i8
                "vle8.v         v20, (%[A1])            \n\t"
                "addi           %[A1], %[A1], 32        \n\t"  // 1*32@i8

                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"

                "vsrl.vi        v17, v16, 4             \n\t"
                "vsrl.vi        v21, v20, 4             \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vnpack4.vv     v2, v16, v20, 2         \n\t"  // low  u4
                "vnpack4.vv     v3, v17, v21, 2         \n\t"  // high s4

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v18, v18, v18           \n\t"
                "vxor.vv        v19, v19, v19           \n\t"
                "vxor.vv        v20, v20, v20           \n\t"
                "vxor.vv        v21, v21, v21           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"
                // asum*zp
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vwmul.vx       v2, v10, t1             \n\t"
                "vwmul.vx       v4, v10, t2             \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "vfcvt.f.x.v    v2, v2                  \n\t"
                "vfcvt.f.x.v    v4, v4                  \n\t"

                "vsetvli        t0, x0, e16, m1         \n\t"
                "vfwcvt.f.f.v   v16, v20                \n\t"

                "vfwcvt.f.f.v   v18, v14                \n\t"

                // +asum*zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vfadd.vv       v16, v16, v2            \n\t"
                "vfadd.vv       v17, v17, v4            \n\t"
                "vfmul.vv       v16, v16, v18           \n\t"
                "vfmul.vv       v17, v17, v18           \n\t"

                "addi           t3, t3, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"

                "bgtz           t3, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v28, (%[DST0])          \n\t"
                "vse32.v        v29, (%[DST1])          \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
        }
    }
}

void moe_gemm_kernel_i8i5_m2_impl(const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_n,
                                  size_t           k_blks) {
    constexpr size_t K_BLK_LEN        = 32;
    constexpr size_t NB_COLS          = 32;
    constexpr size_t B_SCALE_BYTES    = NB_COLS * sizeof(_Float16);
    constexpr size_t B_ZP_BYTES       = NB_COLS * sizeof(uint8_t);
    constexpr size_t B_QH_BYTES       = NB_COLS * K_BLK_LEN / 8 * sizeof(uint8_t);
    constexpr size_t B_QS_BYTES       = NB_COLS * K_BLK_LEN / 2 * sizeof(uint8_t);
    constexpr size_t B_Q50_BLK_STRIDE = B_SCALE_BYTES + B_QH_BYTES + B_QS_BYTES;
    constexpr size_t B_Q51_BLK_STRIDE = B_SCALE_BYTES + B_ZP_BYTES + B_QH_BYTES + B_QS_BYTES;
    static_assert(B_Q50_BLK_STRIDE == sizeof(nrow_block_q5_0<NB_COLS>) &&
                  B_Q51_BLK_STRIDE == sizeof(nrow_block_q5_1<NB_COLS>), "Q5 packed block layout");

    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q50_BLK_STRIDE;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "mv             t4, %[BK]               \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v2, v0, v0              \n\t"
                "vxor.vv        v3, v0, v0              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // ---- load B scale/Bh/Bs and advance to the next q5_0 k-block ----
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v1, (%[B])              \n\t"  // v1 = scale_fp16 × 32
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (%[B])              \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi           %[B], %[B], 128         \n\t"
                "vl4r.v         v8, (%[B])              \n\t"  // v8..v11 = Bs N32K32 i4
                "addi           %[B], %[B], 512         \n\t"

                // ---- load A0/A1 header then payload, each block stride = 38B ----
                "flw            f0, (%[A0])             \n\t"  // f0 = A0 scale (fp32)
                "lh             t2, 4(%[A0])            \n\t"  // t2 = A0 asum (int16)
                "addi           %[A0], %[A0], 6         \n\t"
                "flw            f1, (%[A1])             \n\t"  // f1 = A1 scale (fp32)
                "lh             t3, 4(%[A1])            \n\t"  // t3 = A1 asum (int16)
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (%[A0])             \n\t"  // v4 = A0 M1K32 int8
                "addi           %[A0], %[A0], 32        \n\t"
                "vle8.v         v5, (%[A1])             \n\t"  // v5 = A1 M1K32 int8
                "addi           %[A1], %[A1], 32        \n\t"

                //// ---- A nibble unpacking ----
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vand.vi        v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "slli           t2, t2, 4               \n\t"  // a_sum * 16;
                "slli           t3, t3, 4               \n\t"
                // [4*32]*2
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "li             t1, 16                  \n\t"
                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                "vpack.vv       v6, v4, v5, 2           \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v16, v16           \n\t"
                "vxor.vv        v26, v16, v16           \n\t"
                "vupack.vv      v4, v6, v7, 1           \n\t"
                "vxor.vv        v28, v16, v16           \n\t"
                "vxor.vv        v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot         v24, v4, v8, i8         \n\t"  // N0..7
                "vmadot         v26, v4, v10, i8        \n\t"  // N8..15
                "vmadot         v28, v4, v12, i8        \n\t"  // N16..23
                "vmadot         v30, v4, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot         v24, v5, v9, i8         \n\t"  // N0..7
                "vmadot         v26, v5, v11, i8        \n\t"  // N8..15
                "vmadot         v28, v5, v13, i8        \n\t"  // N16..23
                "vmadot         v30, v5, v15, i8        \n\t"  // N24..31

                "vpack.vv       v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv       v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv       v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vadd.vx        v24, v24, t2            \n\t"
                "vadd.vx        v25, v25, t3            \n\t"
                // b_scale fp16 -> fp32
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v28, v1                 \n\t"

                // a_scale * b_scale;
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v    v26, v24                \n\t"
                "vfcvt.f.x.v    v27, v25                \n\t"
                "vfmul.vf       v30, v28, f0            \n\t"
                "vfmul.vf       v31, v28, f1            \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv      v2, v30, v26            \n\t"
                "vfmacc.vv      v3, v31, v27            \n\t"

                "addi           t4, t4, -1              \n\t"
                "bgtz           t4, BLK_LOOP%=          \n\t"

                "vsetvli        t0, %[NR], e32, m1      \n\t"
                "vse32.v        v2, (%[DST0])           \n\t"
                "vse32.v        v3, (%[DST1])           \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks), [NR] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                  "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                  "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "f0", "f1");
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q51_BLK_STRIDE;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "mv             t4, %[BK]               \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v2, v0, v0              \n\t"
                "vxor.vv        v3, v0, v0              \n\t"
                "addi           t5, %[B], %[B_SCALE_BYTES]            \n\t"  // t5 = zp   (32B)
                "addi           t6, %[B], %[B_QH_OFF]            \n\t"  // t6 = qh   (128B)
                "addi           s1, %[B], %[B_QS_OFF]           \n\t"  // s1 = qs   (512B)

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // ---- load B scale/zp/Bh/Bs and advance to the next q5_1 k-block ----
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v1, (%[B])              \n\t"  // v1 = scale_fp16 × 32
                "addi           %[B], %[B], %[B_BLK_STRIDE]         \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t6)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi           t6, t6, %[B_BLK_STRIDE]             \n\t"
                "vl4r.v         v8, (s1)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi           s1, s1, %[B_BLK_STRIDE]             \n\t"

                // ---- load A0/A1 header then payload, each block stride = 38B ----
                "flw            f0, (%[A0])             \n\t"  // f0 = A0 scale (fp32)
                "lh             t2, 4(%[A0])            \n\t"  // t2 = A0 asum (int16)
                "addi           %[A0], %[A0], 6         \n\t"
                "flw            f1, (%[A1])             \n\t"  // f1 = A1 scale (fp32)
                "lh             t3, 4(%[A1])            \n\t"  // t3 = A1 asum (int16)
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (%[A0])             \n\t"  // v4 = A0 M1K32 int8
                "addi           %[A0], %[A0], 32        \n\t"
                "vle8.v         v5, (%[A1])             \n\t"  // v5 = A1 M1K32 int8
                "addi           %[A1], %[A1], 32        \n\t"

                //// ---- A nibble unpacking ----
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vand.vi        v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                // q5_1 uses explicit zp, so keep a_sum unshifted here.
                // [4*32]*2
                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "li             t1, 16                  \n\t"
                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                "vpack.vv       v6, v4, v5, 2           \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v16, v16           \n\t"
                "vxor.vv        v26, v16, v16           \n\t"
                "vupack.vv      v4, v6, v7, 1           \n\t"
                "vxor.vv        v28, v16, v16           \n\t"
                "vxor.vv        v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot         v24, v4, v8, i8         \n\t"  // N0..7
                "vmadot         v26, v4, v10, i8        \n\t"  // N8..15
                "vmadot         v28, v4, v12, i8        \n\t"  // N16..23
                "vmadot         v30, v4, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot         v24, v5, v9, i8         \n\t"  // N0..7
                "vmadot         v26, v5, v11, i8        \n\t"  // N8..15
                "vmadot         v28, v5, v13, i8        \n\t"  // N16..23
                "vmadot         v30, v5, v15, i8        \n\t"  // N24..31

                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (t5)                \n\t"  // v4 = Bzp N32 uint8
                "addi           t5, t5, %[B_BLK_STRIDE]             \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv       v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv       v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vwaddu.vx      v28, v4, x0             \n\t"

                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vwmul.vx       v30, v28, t2            \n\t"
                "vwmul.vx       v31, v28, t3            \n\t"

                // b_scale fp16 -> fp32
                "vfwcvt.f.f.v   v28, v1                 \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"
                "vadd.vv        v24, v24, v30           \n\t"
                "vadd.vv        v25, v25, v31           \n\t"

                // a_scale * b_scale;
                "vfcvt.f.x.v    v26, v24                \n\t"
                "vfcvt.f.x.v    v27, v25                \n\t"
                "vfmul.vf       v30, v28, f0            \n\t"
                "vfmul.vf       v31, v28, f1            \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv      v2, v30, v26            \n\t"
                "vfmacc.vv      v3, v31, v27            \n\t"

                "addi           t4, t4, -1              \n\t"
                "bgtz           t4, BLK_LOOP%=          \n\t"

                "vsetvli        t0, %[NR], e32, m1      \n\t"
                "vse32.v        v2, (%[DST0])           \n\t"
                "vse32.v        v3, (%[DST1])           \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks), [NR] "r"(nb_real),
                  [B_SCALE_BYTES] "i"(B_SCALE_BYTES), [B_BLK_STRIDE] "i"(B_Q51_BLK_STRIDE),
                  [B_QH_OFF] "i"(B_SCALE_BYTES + B_ZP_BYTES),
                  [B_QS_OFF] "i"(B_SCALE_BYTES + B_ZP_BYTES + B_QH_BYTES)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "v0", "v1", "v2", "v3", "v4", "v5",
                  "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                  "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "f0", "f1");
        }
    }
}

size_t gemm_i8i2k_k256(a_pack          ap,
                              size_t          blk_len,
                              const uint8_t * quant_a_ptr,
                              const uint8_t * quant_b_data,
                              const uint8_t * quant_b_zp,
                              float *         c_ptr,
                              size_t          count_m,
                              size_t          count_n,
                              size_t          k_blks,
                              size_t          ldc) {
    GGML_ASSERT(blk_len == 256);
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4);
        gemm_kernel_i8i2k_m4(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i2k_m1(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks);
    return 1;
}

size_t gemm_i8i3k_k256(a_pack          ap,
                              size_t          blk_len,
                              const uint8_t * quant_a_ptr,
                              const uint8_t * quant_b_data,
                              const uint8_t * quant_b_zp,
                              float *         c_ptr,
                              size_t          count_m,
                              size_t          count_n,
                              size_t          k_blks,
                              size_t          ldc) {
    GGML_ASSERT(blk_len == 256);
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4);
        gemm_kernel_i8i3k_m4(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i3k_m1(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks);
    return 1;
}

size_t gemm_i8i4_k32(a_pack          ap,
                            size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    GGML_ASSERT(blk_len == 32);
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4);
        gemm_kernel_i8i4_m4(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i4_m1(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks);
    return 1;
}

size_t gemm_i8i4_k256(a_pack          ap,
                      size_t          blk_len,
                      const uint8_t * quant_a_ptr,
                      const uint8_t * quant_b_data,
                      const uint8_t * quant_b_zp,
                      float *         c_ptr,
                      size_t          count_m,
                      size_t          count_n,
                      size_t          k_blks,
                      size_t          ldc) {
    GGML_ASSERT(blk_len == 256);
    if (quant_b_zp != nullptr) {
        // Q4_1/Q4_K store unsigned nibbles with a separate zp area.
        if (count_m >= 4) {
            GGML_ASSERT(ap == a_pack::r4_al);
            gemm_kernel_i8i4_hpzp_al_m4(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
            return 4;
        }
        gemm_kernel_i8i4_hpzp_m1(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks);
        return 1;
    }
    // Q4_0 folds its zero point into the nibble encoding.
    if (count_m >= 8) {
        GGML_ASSERT(ap == a_pack::r8);
        gemm_kernel_i8i4_hp_al_m8(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 8;
    }
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4_al);
        gemm_kernel_i8i4_hp_al_m4(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i4_hp_al_m1(quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks);
    return 1;
}

size_t gemm_i8i4_k32_rows(a_pack           ap,
                                 size_t           blk_len,
                                 const uint8_t ** quant_a_ptr,
                                 const uint8_t *  quant_b_data,
                                 const uint8_t *  quant_b_zp,
                                 float **         c_ptr,
                                 size_t           count_m,
                                 size_t           count_n,
                                 size_t           k_blks,
                                 size_t           ldc) {
    GGML_ASSERT(ap == a_pack::r1);
    GGML_ASSERT(blk_len == 32);
    moe_gemm_kernel_i8i4_m2_impl(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks);
    return 2;
}
size_t gemm_i8i5_k32(a_pack          ap,
                            size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    GGML_ASSERT(blk_len == 32);
    if (count_m >= 4) {
        GGML_ASSERT(ap == a_pack::r4);
        gemm_kernel_i8i5_m4(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks, ldc);
        return 4;
    }
    gemm_kernel_i8i5_m1(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks);
    return 1;
}

size_t gemm_i8i5_k32_rows(a_pack           ap,
                                 size_t           blk_len,
                                 const uint8_t ** quant_a_ptr,
                                 const uint8_t *  quant_b_data,
                                 const uint8_t *  quant_b_zp,
                                 float **         c_ptr,
                                 size_t           count_m,
                                 size_t           count_n,
                                 size_t           k_blks,
                                 size_t           ldc) {
    GGML_ASSERT(ap == a_pack::r1);
    GGML_ASSERT(blk_len == 32);
    moe_gemm_kernel_i8i5_m2_impl(quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_n, k_blks);
    return 2;
}

// A/B use [K/8][M16|N32][K8]; the output is row-major FP32.
void gemm_kernel_f16f16f32_m16_vfwmadot(size_t           gemm_k,
                                        const _Float16 * b_f16_k8,
                                        _Float16 *       a_f16_k8,
                                        float *          c_ptr,
                                        size_t           count_m,
                                        size_t           count_n,
                                        size_t           ldc,
                                        bool             accumulate) {
    GGML_ASSERT(count_m > 0 && count_m <= 16);
    GGML_ASSERT(count_n > 0 && count_n <= 32);
    GGML_ASSERT(gemm_k % 8 == 0);
    // Partial-M accumulation requires a full N32 tile.
    GGML_ASSERT(count_m == 16 || count_n == 32 || !accumulate);

    alignas(128) float tile[16 * 32];
    const _Float16 * a0            = a_f16_k8;
    const _Float16 * a1            = a_f16_k8 + 64;
    const _Float16 * b0            = b_f16_k8;
    const _Float16 * b1            = b_f16_k8 + 64;
    const _Float16 * b2            = b_f16_k8 + 128;
    const _Float16 * b3            = b_f16_k8 + 192;
    size_t           kc_left       = gemm_k / 8;
    // Accumulator register pairs are not row-major; add existing C only after unpacking.
    // The local tile also avoids writing beyond a partial-M output.
    const bool       via_tile      = accumulate || count_m != 16;
    float *          out           = via_tile ? tile : c_ptr;
    const size_t     out_row_bytes = (via_tile ? size_t{ 32 } : ldc) * sizeof(float);

    asm volatile(
        "vsetvli t0, zero, e32, m8, tu, mu\n\t"
        "vxor.vv v16, v16, v16\n\t"
        "vxor.vv v24, v24, v24\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "1:\n\t"
        "vle16.v v0,  (%[a0])\n\t"
        "vle16.v v1,  (%[a1])\n\t"
        "vle16.v v8,  (%[b0])\n\t"
        "vle16.v v9,  (%[b1])\n\t"
        "vle16.v v10, (%[b2])\n\t"
        "vle16.v v11, (%[b3])\n\t"
        "vfwmadot v16, v0, v8\n\t"
        "vfwmadot v18, v0, v9\n\t"
        "vfwmadot v20, v0, v10\n\t"
        "vfwmadot v22, v0, v11\n\t"
        "vfwmadot v24, v1, v8\n\t"
        "vfwmadot v26, v1, v9\n\t"
        "vfwmadot v28, v1, v10\n\t"
        "vfwmadot v30, v1, v11\n\t"
        "addi %[a0], %[a0], 256\n\t"
        "addi %[a1], %[a1], 256\n\t"
        "addi %[b0], %[b0], 512\n\t"
        "addi %[b1], %[b1], 512\n\t"
        "addi %[b2], %[b2], 512\n\t"
        "addi %[b3], %[b3], 512\n\t"
        "addi %[kc], %[kc], -1\n\t"
        "bnez %[kc], 1b\n\t"
        // Unpack eight 8x8 accumulator tiles into row-major M16N32.
        "vsetvli t0, zero, e32, m1, tu, mu\n\t"
        "vpack.vv v0,  v16, v18, 2\n\t"
        "vpack.vv v2,  v20, v22, 2\n\t"
        "vpack.vv v4,  v17, v19, 2\n\t"
        "vpack.vv v6,  v21, v23, 2\n\t"
        "vpack.vv v8,  v24, v26, 2\n\t"
        "vpack.vv v10, v28, v30, 2\n\t"
        "vpack.vv v12, v25, v27, 2\n\t"
        "vpack.vv v14, v29, v31, 2\n\t"
        "vpack.vv v16, v0,  v2,  3\n\t"
        "vpack.vv v18, v1,  v3,  3\n\t"
        "vpack.vv v20, v4,  v6,  3\n\t"
        "vpack.vv v22, v5,  v7,  3\n\t"
        "vpack.vv v24, v8,  v10, 3\n\t"
        "vpack.vv v26, v9,  v11, 3\n\t"
        "vpack.vv v28, v12, v14, 3\n\t"
        "vpack.vv v30, v13, v15, 3\n\t"
        "vsetvli t0, %[count_n], e32, m1, tu, mu\n\t"
        "vse32.v v16, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v17, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v18, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v19, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v20, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v21, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v22, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v23, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v24, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v25, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v26, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v27, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v28, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v29, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v30, (%[out])\n\t"
        "add %[out], %[out], %[out_stride]\n\t"
        "vse32.v v31, (%[out])\n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3), [kc] "+r"(kc_left),
          [out] "+r"(out)
        : [count_n] "r"(count_n), [out_stride] "r"(out_row_bytes)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31");

    if (via_tile) {
        for (size_t m = 0; m < count_m; ++m) {
            for (size_t n = 0; n < count_n; ++n) {
                c_ptr[m * ldc + n] = accumulate ? c_ptr[m * ldc + n] + tile[m * 32 + n] : tile[m * 32 + n];
            }
        }
    }
}

}  // namespace ime2
}  // namespace spacemit_kernels
