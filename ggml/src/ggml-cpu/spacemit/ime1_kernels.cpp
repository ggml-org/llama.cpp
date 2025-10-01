#include "ggml.h"
#include "ime_kernels.h"

#include <algorithm>
#include <cmath>

// clang-format off
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
// clang-format on
namespace sqnbitgemm_spacemit_ime {

#define QUANTIZEM4ROW_KERNEL                           \
    "vmv.s.x            v16, zero                \n\t" \
    "vfabs.v            v8, v0                   \n\t" \
    "vfredmax.vs        v16, v8, v16             \n\t" \
    "vfmv.f.s           f10, v16                 \n\t" \
    "fmul.s             f10, f10, %[RMAXREC]     \n\t" \
    "fsw                f10, (a1)                \n\t" \
    "fdiv.s             f11, %[FONE], f10        \n\t" \
    "vfmul.vf           v16, v0, f11             \n\t" \
    "vfcvt.x.f.v        v16, v16                 \n\t" \
    "vsetvli            t0, zero, e16, mf2       \n\t" \
    "vnclip.wx          v16, v16, zero           \n\t" \
    "vnclip.wx          v17, v17, zero           \n\t" \
    "vnclip.wx          v18, v18, zero           \n\t" \
    "vnclip.wx          v19, v19, zero           \n\t" \
    "vnclip.wx          v20, v20, zero           \n\t" \
    "vnclip.wx          v21, v21, zero           \n\t" \
    "vnclip.wx          v22, v22, zero           \n\t" \
    "vnclip.wx          v23, v23, zero           \n\t" \
    "vsetvli            t0, zero, e8, mf4        \n\t" \
    "vnclip.wx          v24, v16, zero           \n\t" \
    "vnclip.wx          v25, v17, zero           \n\t" \
    "vnclip.wx          v26, v18, zero           \n\t" \
    "vnclip.wx          v27, v19, zero           \n\t" \
    "vnclip.wx          v28, v20, zero           \n\t" \
    "vnclip.wx          v29, v21, zero           \n\t" \
    "vnclip.wx          v30, v22, zero           \n\t" \
    "vnclip.wx          v31, v23, zero           \n\t"

#define QUANTIZEM4ROW_STORE                            \
    "addi               t1, %[BlkLen], 0         \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v24, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v25, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v26, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v27, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v28, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v29, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v30, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v31, (s1)                \n\t"

namespace ime1 {
void quantize_a_4row_i8(size_t BlkLen, const float * A, size_t CountK, std::byte * QuantA) {
    constexpr float range_max_reciprocal = 1.0f / ((1 << 7) - 1);
    const float     fone                 = 1.0f;

    if (BlkLen == 16 || BlkLen == 32 || BlkLen == 64) {
        for (size_t row_index = 0; row_index < 4; ++row_index) {
            const float * SRC = A + row_index * CountK;
            std::byte *   DST = QuantA + row_index * sizeof(float);

            const size_t offset = (4 - row_index) * 4 + row_index * 8;
            const size_t stride = 4 * (sizeof(float) + BlkLen);
            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8        \n\t"
                "addi               t2, %[CountK], 0         \n\t"
                "addi               a1, %[DST], 0            \n\t"
                "blt                t2, %[BlkLen], TAIL%=    \n\t"

                "LOOP%=:                                     \n\t"
                "vsetvli            t0, %[BlkLen], e32, m8   \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "sub                t2, t2, t0               \n\t"
                "slli               t1, t0, 2                \n\t"
                "add                %[SRC], %[SRC], t1       \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"

                QUANTIZEM4ROW_KERNEL QUANTIZEM4ROW_STORE

                "add                a1, a1, %[STRIDE]        \n\t"
                "bge                t2, %[BlkLen], LOOP%=    \n\t"

                "TAIL%=:                                     \n\t"
                "blez               t2, QUIT%=               \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vxor.vv            v16, v16, v16            \n\t"
                "vxor.vv            v24, v24, v24            \n\t"
                "vsetvli            t0, t2, e32, m8          \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"

                QUANTIZEM4ROW_KERNEL

                "addi               t3, %[BlkLen], 0         \n\t"
                "addi               s2, s1, 0                \n\t"
                "vsetvli            t0, zero, e8, mf4        \n\t"
                "vxor.vv            v8, v8, v8               \n\t"
                "SET_ZERO%=:                                 \n\t"
                "vse8.v             v8, (s2)                 \n\t"
                "addi               s2, s2, 32               \n\t"
                "addi               t3, t3, -8               \n\t"
                "bnez               t3, SET_ZERO%=           \n\t"

                QUANTIZEM4ROW_STORE

                "QUIT%=:                                     \n\t"
                : [SRC] "+r"(SRC)
                : [DST] "r"(DST), [BlkLen] "r"(BlkLen), [OFFSET] "r"(offset), [STRIDE] "r"(stride),
                  [CountK] "r"(CountK), [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal)
                : "cc", "t0", "t1", "t2", "t3", "a1", "s1", "s2", "f10", "f11");
        }
    } else if (BlkLen == 128) {
        for (size_t row_index = 0; row_index < 4; ++row_index) {
            const float * SRC = A + row_index * CountK;
            std::byte *   DST = QuantA + row_index * sizeof(float);

            const size_t offset = (4 - row_index) * 4 + row_index * 8;
            const size_t stride = 4 * (sizeof(float) + BlkLen);
            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8        \n\t"
                "li                 t6, 32                   \n\t"
                "addi               t2, %[CountK], 0         \n\t"
                "addi               a1, %[DST], 0            \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"
                "blt                t2, %[BlkLen], TAIL%=    \n\t"

                "LOOP%=:                                     \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v8, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "addi               t2, t2, -128             \n\t"

                "QUANTIZE%=:                                 \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"
                "vfabs.v            v16, v0                  \n\t"
                "vfabs.v            v24, v8                  \n\t"
                "vfmax.vv           v16, v24, v16            \n\t"
                "vfredmax.vs        v24, v16, v24            \n\t"
                "vfmv.f.s           f10, v24                 \n\t"
                "fmul.s             f10, f10, %[RMAXREC]     \n\t"
                "fsw                f10, (a1)                \n\t"
                "fdiv.s             f11, %[FONE], f10        \n\t"
                "vfmul.vf           v16, v0, f11             \n\t"
                "vfmul.vf           v24, v8, f11             \n\t"
                "vfcvt.x.f.v        v16, v16                 \n\t"
                "vfcvt.x.f.v        v24, v24                 \n\t"
                "vsetvli            t0, zero, e16, m4        \n\t"
                "vnclip.wx          v16, v16, zero           \n\t"
                "vnclip.wx          v20, v24, zero           \n\t"
                "vsetvli            t0, zero, e8, m4         \n\t"
                "vnclip.wx          v16, v16, zero           \n\t"
                "vsetvli            t0, zero, e64, m4        \n\t"
                "vsse64.v           v16, (s1), t6            \n\t"
                "add                a1, a1, %[STRIDE]        \n\t"
                "bge                t2, %[BlkLen], LOOP%=    \n\t"

                "TAIL%=:                                     \n\t"
                "blez               t2, QUIT%=               \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vxor.vv             v0, v0, v0              \n\t"
                "vxor.vv             v8, v8, v8              \n\t"
                "vxor.vv             v16, v16, v16           \n\t"
                "vxor.vv             v24, v24, v24           \n\t"
                "vsetvli            t0, t2, e32, m8          \n\t"
                "sub                t2, t2, t0               \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vsetvli            t0, t2, e32, m8          \n\t"
                "vle32.v            v8, (%[SRC])             \n\t"
                "sub                t2, t2, t2               \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "jal                x0, QUANTIZE%=           \n\t"

                "QUIT%=:                                     \n\t"
                : [SRC] "+r"(SRC)
                : [DST] "r"(DST), [BlkLen] "r"(BlkLen), [OFFSET] "r"(offset), [STRIDE] "r"(stride),
                  [CountK] "r"(CountK), [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal)
                : "cc", "t0", "t1", "t2", "t6", "a1", "s1", "s2", "f10", "f11");
        }
    } else if (BlkLen == 256) {
        for (size_t row_index = 0; row_index < 4; ++row_index) {
            const float * SRC    = A + row_index * CountK;
            std::byte *   DST    = QuantA + row_index * sizeof(float);
            const size_t  offset = (4 - row_index) * 4 + row_index * 8;
            const size_t  stride = 4 * (sizeof(float) + BlkLen);
            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8        \n\t"
                "li                 t6, 32                   \n\t"
                "addi               t2, %[CountK], 0         \n\t"
                "addi               a1, %[DST], 0            \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"
                "blt                t2, %[BlkLen], TAIL%=    \n\t"

                "LOOP%=:                                     \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v8, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v16, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v24, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], -768     \n\t"
                "addi               t2, t2, -256             \n\t"
                "vfabs.v            v0, v0                   \n\t"
                "vfabs.v            v8, v8                   \n\t"
                "vfabs.v            v16, v16                 \n\t"
                "vfabs.v            v24, v24                 \n\t"
                "vfmax.vv           v8, v0, v8               \n\t"
                "vfmax.vv           v24, v24, v16            \n\t"
                "vfmax.vv           v8, v8, v24              \n\t"
                "vfredmax.vs        v24, v8, v24             \n\t"
                "vfmv.f.s           f10, v24                 \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v8, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v16, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vle32.v            v24, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"

                "QUANTIZE%=:                                 \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"
                "fmul.s             f10, f10, %[RMAXREC]     \n\t"
                "fsw                f10, (a1)                \n\t"
                "fdiv.s             f11, %[FONE], f10        \n\t"
                "vfmul.vf           v0, v0, f11              \n\t"
                "vfmul.vf           v8, v8, f11              \n\t"
                "vfmul.vf           v16, v16, f11            \n\t"
                "vfmul.vf           v24, v24, f11            \n\t"
                "vfcvt.x.f.v        v0, v0                   \n\t"
                "vfcvt.x.f.v        v8, v8                   \n\t"
                "vfcvt.x.f.v        v16, v16                 \n\t"
                "vfcvt.x.f.v        v24, v24                 \n\t"
                "vsetvli            t0, zero, e16, m4        \n\t"
                "vnclip.wx          v0, v0, zero             \n\t"
                "vnclip.wx          v4, v8, zero             \n\t"
                "vnclip.wx          v8, v16, zero            \n\t"
                "vnclip.wx          v12, v24, zero           \n\t"
                "vsetvli            t0, zero, e8, m4         \n\t"
                "vnclip.wx          v0, v0, zero             \n\t"
                "vnclip.wx          v4, v8, zero             \n\t"
                "vsetvli            t0, zero, e64, m8        \n\t"
                "vsse64.v           v0, (s1), t6             \n\t"
                "add                a1, a1, %[STRIDE]        \n\t"
                "bge                t2, %[BlkLen], LOOP%=    \n\t"

                "TAIL%=:                                     \n\t"
                "blez               t2, QUIT%=               \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vxor.vv            v0, v0, v0               \n\t"
                "vxor.vv            v8, v8, v8               \n\t"
                "vxor.vv            v16, v16, v16            \n\t"
                "vxor.vv            v24, v24, v24            \n\t"
                "addi               t1, t2, 0                \n\t"
                "vsetvli            t0, t1, e32, m8          \n\t"
                "sub                t1, t1, t0               \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vsetvli            t0, t1, e32, m8          \n\t"
                "sub                t1, t1, t0               \n\t"
                "vle32.v            v8, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vsetvli            t0, t1, e32, m8          \n\t"
                "sub                t1, t1, t0               \n\t"
                "vle32.v            v16, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], 256      \n\t"
                "vsetvli            t0, t1, e32, m8          \n\t"
                "vle32.v            v24, (%[SRC])            \n\t"
                "addi               %[SRC], %[SRC], -768     \n\t"
                "vsetvli            t0, zero, e32, m8        \n\t"
                "vfabs.v            v0, v0                   \n\t"
                "vfabs.v            v8, v8                   \n\t"
                "vfabs.v            v16, v16                 \n\t"
                "vfabs.v            v24, v24                 \n\t"
                "vfmax.vv           v8, v0, v8               \n\t"
                "vfmax.vv           v24, v16, v24            \n\t"
                "vfmax.vv           v8, v8, v24              \n\t"
                "vfredmax.vs        v24, v8, v24             \n\t"
                "vfmv.f.s           f10, v24                 \n\t"
                "add                s1, a1, %[OFFSET]        \n\t"
                "fmul.s             f10, f10, %[RMAXREC]     \n\t"
                "fsw                f10, (a1)                \n\t"
                "fdiv.s             f11, %[FONE], f10        \n\t"
                "vsetvli            t0, zero, e64, m8        \n\t"
                "vxor.vv            v0, v0, v0               \n\t"
                "vsse64.v           v0, (s1), t6             \n\t"

                "TAIL_LOOP%=:                                \n\t"
                "vsetvli            t0, zero, e32, m4        \n\t"
                "vxor.vv            v0, v0, v0               \n\t"
                "vsetvli            t0, t2, e32, m1          \n\t"
                "sub                t2, t2, t0               \n\t"
                "vle32.v            v0, (%[SRC])             \n\t"
                "addi               %[SRC], %[SRC], 32       \n\t"
                "vfmul.vf           v1, v0, f11              \n\t"
                "vfcvt.x.f.v        v2, v1                   \n\t"
                "vsetvli            t0, zero, e16, mf2       \n\t"
                "vnclip.wx          v3, v2, zero             \n\t"
                "vsetvli            t0, zero, e8, mf4        \n\t"
                "vnclip.wx          v3, v3, zero             \n\t"
                "vse8.v             v3, (s1)                 \n\t"
                "addi               s1, s1, 32               \n\t"
                "bnez               t2, TAIL_LOOP%=          \n\t"

                "QUIT%=:                                     \n\t"
                : [SRC] "+r"(SRC)
                : [DST] "r"(DST), [BlkLen] "r"(BlkLen), [OFFSET] "r"(offset), [STRIDE] "r"(stride),
                  [CountK] "r"(CountK), [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal)
                : "cc", "t0", "t1", "t2", "t6", "a1", "s1", "s2", "f10", "f11");
        }
    }
}

void quantize_a_row_i8(size_t BlkLen, const float * A, size_t CountK, std::byte * QuantA) {
    const float *   SRC                  = A;
    std::byte *     DST                  = QuantA;
    constexpr float range_max_reciprocal = 1.0f / ((1 << 7) - 1);
    const float     fone                 = 1.0f;
    std::byte *     QuantA_offset        = QuantA + CountK + 4 * ((CountK + BlkLen - 1) / BlkLen);
    size_t          offset               = (CountK + BlkLen - 1) / BlkLen * BlkLen - CountK;

    if (CountK <= BlkLen) {
        float max_abs_A = 0.0f;
        for (size_t k = 0; k < CountK; k++) {
            max_abs_A = std::max(max_abs_A, fabsf(A[k]));
        }
        float scale_A = max_abs_A * range_max_reciprocal;

        ((float *) QuantA)[0] = scale_A;

        auto * QuantAData_offset = (int8_t *) (QuantA + sizeof(float));

        for (size_t k = 0; k < CountK; k++) {
            QuantAData_offset[k] =
                (int8_t) std::clamp(roundf(A[k] / scale_A), (float) std::numeric_limits<int8_t>::lowest(),
                                    (float) std::numeric_limits<int8_t>::max());
        }
        for (size_t k = CountK; k < BlkLen; k++) {
            QuantAData_offset[k] = 0;
        }

        return;
    }

    if (BlkLen != 32 || BlkLen != 64 || BlkLen != 128) {
        __asm__ volatile(
            "vsetvli      t0, zero, e8, m8        \n\t"
            "vxor.vv      v24, v24, v24           \n\t"
            "LOOP%=:                              \n\t"
            "vsetvli      t0, %[CNT], e8, m8      \n\t"
            "vse8.v       v24, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 128     \n\t"
            "sub          %[CNT], %[CNT], t0      \n\t"
            "bnez         %[CNT], LOOP%=          \n\t"
            : [DST] "+r"(QuantA_offset), [CNT] "+r"(offset)
            :
            : "cc", "t0");
    }
    if (BlkLen == 16) {
        float buffer[64] = { 0.0f };
        __asm__ volatile(
            "addi         t3, zero, 16*8          \n\t"
            "addi         t2, zero, 16            \n\t"
            "blt          %[K], t3, LOOP_K%=      \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_MAIN%=:                         \n\t"
            "vsetvli      t1, zero, e32, m2       \n\t"
            "addi         %[K], %[K], -128        \n\t"
            "vle32.v      v0, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v2, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v4, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v6, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v8, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v10, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v12, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "vle32.v      v14, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "addi         a1, %[BUFFER], 0        \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vfabs.v      v18, v2                 \n\t"
            "vfabs.v      v20, v4                 \n\t"
            "vfabs.v      v22, v6                 \n\t"
            "vfabs.v      v24, v8                 \n\t"
            "vfabs.v      v26, v10                \n\t"
            "vfabs.v      v28, v12                \n\t"
            "vfabs.v      v30, v14                \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vfmax.vv     v18, v18, v19           \n\t"
            "vfmax.vv     v20, v20, v21           \n\t"
            "vfmax.vv     v22, v22, v23           \n\t"
            "vfmax.vv     v24, v24, v25           \n\t"
            "vfmax.vv     v26, v26, v27           \n\t"
            "vfmax.vv     v28, v28, v29           \n\t"
            "vfmax.vv     v30, v30, v31           \n\t"
            "vse32.v      v16, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v18, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v20, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v22, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v24, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v26, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v28, (a1)               \n\t"
            "addi         a1, a1, 32              \n\t"
            "vse32.v      v30, (a1)               \n\t"
            "addi         a1, %[BUFFER], 0        \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f10, f3, f7             \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f10, %[FONE], f10       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f11, f3, f7             \n\t"
            "fmul.s       f11, f11, %[RMAXREC]    \n\t"
            "fsw          f11, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f11, %[FONE], f11       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f12, f3, f7             \n\t"
            "fmul.s       f12, f12, %[RMAXREC]    \n\t"
            "fsw          f12, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f12, %[FONE], f12       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f13, f3, f7             \n\t"
            "fmul.s       f13, f13, %[RMAXREC]    \n\t"
            "fsw          f13, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f13, %[FONE], f13       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f14, f3, f7             \n\t"
            "fmul.s       f14, f14, %[RMAXREC]    \n\t"
            "fsw          f14, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f14, %[FONE], f14       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f15, f3, f7             \n\t"
            "fmul.s       f15, f15, %[RMAXREC]    \n\t"
            "fsw          f15, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f15, %[FONE], f15       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f16, f3, f7             \n\t"
            "fmul.s       f16, f16, %[RMAXREC]    \n\t"
            "fsw          f16, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "fdiv.s       f16, %[FONE], f16       \n\t"
            "flw          f0, (a1)                \n\t"
            "flw          f1, 4(a1)               \n\t"
            "flw          f2, 8(a1)               \n\t"
            "flw          f3, 12(a1)              \n\t"
            "flw          f4, 16(a1)              \n\t"
            "flw          f5, 20(a1)              \n\t"
            "flw          f6, 24(a1)              \n\t"
            "flw          f7, 28(a1)              \n\t"
            "addi         a1, a1, 32              \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f17, f3, f7             \n\t"
            "fmul.s       f17, f17, %[RMAXREC]    \n\t"
            "fsw          f17, (%[DST])           \n\t"
            "addi         %[DST], %[DST], -136    \n\t"
            "fdiv.s       f17, %[FONE], f17       \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmul.vf     v16, v0, f10            \n\t"
            "vfmul.vf     v18, v2, f11            \n\t"
            "vfmul.vf     v20, v4, f12            \n\t"
            "vfmul.vf     v22, v6, f13            \n\t"
            "vfmul.vf     v24, v8, f14            \n\t"
            "vfmul.vf     v26, v10, f15           \n\t"
            "vfmul.vf     v28, v12, f16           \n\t"
            "vfmul.vf     v30, v14, f17           \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vfcvt.x.f.v  v18, v18                \n\t"
            "vfcvt.x.f.v  v20, v20                \n\t"
            "vfcvt.x.f.v  v22, v22                \n\t"
            "vfcvt.x.f.v  v24, v24                \n\t"
            "vfcvt.x.f.v  v26, v26                \n\t"
            "vfcvt.x.f.v  v28, v28                \n\t"
            "vfcvt.x.f.v  v30, v30                \n\t"
            "vsetvli      t0, zero, e16, m1       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v18, v18, zero          \n\t"
            "vnclip.wx    v20, v20, zero          \n\t"
            "vnclip.wx    v22, v22, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vnclip.wx    v26, v26, zero          \n\t"
            "vnclip.wx    v28, v28, zero          \n\t"
            "vnclip.wx    v30, v30, zero          \n\t"
            "vsetvli      t0, t1, e8, mf2         \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v18, v18, zero          \n\t"
            "vnclip.wx    v20, v20, zero          \n\t"
            "vnclip.wx    v22, v22, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vnclip.wx    v26, v26, zero          \n\t"
            "vnclip.wx    v28, v28, zero          \n\t"
            "vnclip.wx    v30, v30, zero          \n\t"
            "vse8.v       v16, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v18, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v20, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v22, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v24, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v26, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v28, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 20      \n\t"
            "vse8.v       v30, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 16      \n\t"
            "bge          %[K], t3, LOOP_MAIN%=   \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_K%=:                            \n\t"
            "vsetvli      t1, %[K], e32, m2       \n\t"
            "vle32.v      v0, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 64      \n\t"
            "sub          %[K], %[K], t1          \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vse32.v      v16, (%[BUFFER])        \n\t"
            "flw          f0, (%[BUFFER])         \n\t"
            "flw          f1, 4(%[BUFFER])        \n\t"
            "flw          f2, 8(%[BUFFER])        \n\t"
            "flw          f3, 12(%[BUFFER])       \n\t"
            "flw          f4, 16(%[BUFFER])       \n\t"
            "flw          f5, 20(%[BUFFER])       \n\t"
            "flw          f6, 24(%[BUFFER])       \n\t"
            "flw          f7, 28(%[BUFFER])       \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f10, f3, f7             \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 4       \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmul.vf     v16, v0, f11            \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vsetvli      t0, zero, e16, m1       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vsetvli      t0, t1, e8, mf2         \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vse8.v       v16, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 16      \n\t"
            "bge          %[K], t2, LOOP_K%=      \n\t"
            "TAIL%=:                              \n\t"
            "blez         %[K], END%=             \n\t"
            "vsetvli      t0, t3, e32, m2         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "jal          x0, LOOP_K%=            \n\t"
            "END%=:                               \n\t"
            : [SRC] "+r"(SRC), [DST] "+r"(DST), [K] "+r"(CountK)
            : [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal), [BUFFER] "r"(buffer)
            : "cc", "t3", "t2", "t1", "t0", "a1", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f10", "f11", "f12",
              "f13", "f14", "f15", "f16", "f17");
    } else if (BlkLen == 32) {
        __asm__ volatile(
            "addi         t3, zero, 32*4          \n\t"
            "addi         t2, zero, 32            \n\t"

            "addi         a1, %[SRC], 0           \n\t"
            "addi         a2, %[SRC], 128         \n\t"
            "addi         a3, %[SRC], 256         \n\t"
            "addi         a4, %[SRC], 384         \n\t"

            "addi         s1, %[DST], 0           \n\t"
            "addi         s2, %[DST], 36          \n\t"
            "addi         s3, %[DST], 72          \n\t"
            "addi         s4, %[DST], 108         \n\t"
            "blt          %[K], t3, LOOP_K%=      \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"

            "LOOP_MAIN%=:                         \n\t"
            "vsetvli      t1, zero, e32, m4       \n\t"
            "addi         %[K], %[K], -128        \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "addi         a1, a1, 512             \n\t"
            "vle32.v      v4, (a2)                \n\t"
            "addi         a2, a2, 512             \n\t"
            "vle32.v      v8, (a3)                \n\t"
            "addi         a3, a3, 512             \n\t"
            "vle32.v      v12, (a4)               \n\t"
            "addi         a4, a4, 512             \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vfabs.v      v20, v4                 \n\t"
            "vfabs.v      v24, v8                 \n\t"
            "vfabs.v      v28, v12                \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v16, v16, v18           \n\t"
            "vfmax.vv     v20, v20, v22           \n\t"
            "vfmax.vv     v24, v24, v26           \n\t"
            "vfmax.vv     v28, v28, v30           \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vfmax.vv     v20, v20, v21           \n\t"
            "vfmax.vv     v24, v24, v25           \n\t"
            "vfmax.vv     v28, v28, v29           \n\t"

            "vfredmax.vs  v17, v16, v17           \n\t"
            "vfredmax.vs  v21, v20, v21           \n\t"
            "vfredmax.vs  v25, v24, v25           \n\t"
            "vfredmax.vs  v29, v28, v29           \n\t"
            "vfmv.f.s     f10,  v17               \n\t"
            "vfmv.f.s     f11,  v21               \n\t"
            "vfmv.f.s     f12,  v25               \n\t"
            "vfmv.f.s     f13,  v29               \n\t"

            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fmul.s       f11, f11, %[RMAXREC]    \n\t"
            "fmul.s       f12, f12, %[RMAXREC]    \n\t"
            "fmul.s       f13, f13, %[RMAXREC]    \n\t"
            "fsw          f10, (s1)               \n\t"
            "addi         s1, s1, 4               \n\t"

            "fsw          f11, (s2)               \n\t"
            "addi         s2, s2, 4               \n\t"
            "fsw          f12, (s3)               \n\t"
            "addi         s3, s3, 4               \n\t"
            "fsw          f13, (s4)               \n\t"
            "addi         s4, s4, 4               \n\t"
            "fdiv.s       f10, %[FONE], f10       \n\t"
            "fdiv.s       f11, %[FONE], f11       \n\t"
            "fdiv.s       f12, %[FONE], f12       \n\t"
            "fdiv.s       f13, %[FONE], f13       \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmul.vf     v16, v0, f10            \n\t"
            "vfmul.vf     v20, v4, f11            \n\t"
            "vfmul.vf     v24, v8, f12            \n\t"
            "vfmul.vf     v28, v12, f13           \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vfcvt.x.f.v  v20, v20                \n\t"
            "vfcvt.x.f.v  v24, v24                \n\t"
            "vfcvt.x.f.v  v28, v28                \n\t"
            "vsetvli      t0, zero, e16, m2       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v20, v20, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vnclip.wx    v28, v28, zero          \n\t"
            "vsetvli      t0, t1, e8, m1          \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v20, v20, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vnclip.wx    v28, v28, zero          \n\t"
            "vse8.v       v16, (s1)               \n\t"
            "addi         s1, s1, 140             \n\t"
            "vse8.v       v20, (s2)               \n\t"
            "addi         s2, s2, 140             \n\t"
            "vse8.v       v24, (s3)               \n\t"
            "addi         s3, s3, 140             \n\t"
            "vse8.v       v28, (s4)               \n\t"
            "addi         s4, s4, 140             \n\t"
            "bge          %[K], t3, LOOP_MAIN%=   \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_K%=:                            \n\t"
            "vsetvli      t1, %[K], e32, m4       \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "addi         a1, a1, 128             \n\t"
            "sub          %[K], %[K], t1          \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v16, v16, v18           \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vfredmax.vs  v17, v16, v17           \n\t"
            "vfmv.f.s     f10,  v17               \n\t"

            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10, (s1)               \n\t"
            "addi         s1, s1, 4               \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmul.vf     v16, v0, f11            \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vsetvli      t0, zero, e16, m2       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vse8.v       v16, (s1)               \n\t"
            "addi         s1, s1, 32              \n\t"
            "bge          %[K], t2, LOOP_K%=      \n\t"
            "TAIL%=:                              \n\t"
            "blez         %[K], END%=             \n\t"
            "vsetvli      t0, t3, e32, m4         \n\t"
            "vxor.vv      v0, v0, v0              \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "jal          x0, LOOP_K%=            \n\t"
            "END%=:                               \n\t"
            : [K] "+r"(CountK)
            : [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal), [SRC] "r"(SRC), [DST] "r"(DST)
            : "cc", "t3", "t2", "t1", "t0", "a1", "a2", "a3", "a4", "s1", "s2", "s3", "s4", "f10", "f11", "f12", "f13");
    } else if (BlkLen == 64) {
        __asm__ volatile(
            "addi         t3, zero, 64*2          \n\t"
            "addi         t2, zero, 64            \n\t"
            "addi         a1, %[SRC], 0           \n\t"
            "addi         a2, %[SRC], 256         \n\t"
            "addi         s1, %[DST], 0           \n\t"
            "addi         s2, %[DST], 68          \n\t"
            "blt          %[K], t3, LOOP_K%=      \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_MAIN%=:                         \n\t"
            "vsetvli      t1, zero, e32, m8       \n\t"
            "addi         %[K], %[K], -128        \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "addi         a1, a1, 512             \n\t"
            "vle32.v      v8, (a2)                \n\t"
            "addi         a2, a2, 512             \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vfabs.v      v24, v8                 \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmax.vv     v16, v16, v20           \n\t"
            "vfmax.vv     v24, v24, v28           \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v16, v16, v18           \n\t"
            "vfmax.vv     v24, v24, v26           \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vfmax.vv     v24, v24, v25           \n\t"
            "vfredmax.vs  v17, v16, v17           \n\t"
            "vfredmax.vs  v25, v24, v25           \n\t"
            "vfmv.f.s     f10,  v17               \n\t"
            "vfmv.f.s     f11,  v25               \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fmul.s       f11, f11, %[RMAXREC]    \n\t"
            "fsw          f10, (s1)               \n\t"
            "addi         s1, s1, 4               \n\t"
            "fsw          f11, (s2)               \n\t"
            "addi         s2, s2, 4               \n\t"
            "fdiv.s       f10, %[FONE], f10       \n\t"
            "fdiv.s       f11, %[FONE], f11       \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vfmul.vf     v16, v0, f10            \n\t"
            "vfmul.vf     v24, v8, f11            \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vfcvt.x.f.v  v24, v24                \n\t"
            "vsetvli      t0, zero, e16, m4       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vsetvli      t0, t1, e8, m2          \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v24, v24, zero          \n\t"
            "vse8.v       v16, (s1)               \n\t"
            "addi         s1, s1, 132             \n\t"
            "vse8.v       v24, (s2)               \n\t"
            "addi         s2, s2, 132             \n\t"
            "bge          %[K], t3, LOOP_MAIN%=   \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_K%=:                            \n\t"
            "vsetvli      t1, %[K], e32, m8       \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "addi         a1, a1, 256             \n\t"
            "sub          %[K], %[K], t1          \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmax.vv     v16, v16, v20           \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v16, v16, v18           \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v16, v16, v17           \n\t"
            "vfredmax.vs  v17, v16, v17           \n\t"
            "vfmv.f.s     f10,  v17               \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10, (s1)               \n\t"
            "addi         s1, s1, 4               \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vfmul.vf     v16, v0, f11            \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vsetvli      t0, zero, e16, m4       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vsetvli      t0, zero, e8, m2        \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vse8.v       v16, (s1)               \n\t"
            "addi         s1, s1, 64              \n\t"
            "bge          %[K], t2, LOOP_K%=      \n\t"
            "TAIL%=:                              \n\t"
            "blez         %[K], END%=             \n\t"
            "vsetvli      t0, t3, e32, m8         \n\t"
            "vxor.vv      v0, v0, v0              \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "jal          x0, LOOP_K%=            \n\t"
            "END%=:                               \n\t"
            : [K] "+r"(CountK)
            : [SRC] "r"(SRC), [DST] "r"(DST), [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal)
            : "cc", "t3", "t2", "t1", "t0", "a1", "a2", "s1", "s2", "f10", "f11");
    } else if (BlkLen == 128) {
        __asm__ volatile(
            "addi         t2, zero, 128           \n\t"
            "addi         a1, %[SRC], 0           \n\t"
            "addi         a2, %[SRC], 256         \n\t"
            "blt          %[K], t2, TAIL%=        \n\t"
            "LOOP_K%=:                            \n\t"
            "vsetvli      t1, zero, e32, m8       \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "addi         a1, a1, 512             \n\t"
            "vle32.v      v8, (a2)                \n\t"
            "addi         a2, a2, 512             \n\t"
            "sub          %[K], %[K], t2          \n\t"
            "QUANT%=:                             \n\t"
            "vfabs.v      v16, v0                 \n\t"
            "vfabs.v      v24, v8                 \n\t"
            "vfmax.vv     v24, v16, v24           \n\t"
            "vsetvli      t1, zero, e32, m4       \n\t"
            "vfmax.vv     v28, v24, v28           \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v30, v28, v30           \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v30, v30, v31           \n\t"
            "vfredmax.vs  v31, v30, v31           \n\t"
            "vfmv.f.s     f10, v31                \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 4       \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vfmul.vf     v16, v0, f11            \n\t"
            "vfmul.vf     v24, v8, f11            \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vfcvt.x.f.v  v24, v24                \n\t"
            "vsetvli      t0, zero, e16, m4       \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vnclip.wx    v20, v24, zero          \n\t"
            "vsetvli      t0, zero, e8, m4        \n\t"
            "vnclip.wx    v16, v16, zero          \n\t"
            "vse8.v       v16, (%[DST])           \n\t"
            "addi         %[DST], %[DST], 128     \n\t"
            "bge          %[K], t2, LOOP_K%=      \n\t"
            "TAIL%=:                              \n\t"
            "blez         %[K], END%=             \n\t"
            "vsetvli      t1, zero, e32, m8       \n\t"
            "vxor.vv      v0, v0, v0              \n\t"
            "vxor.vv      v8, v8, v8              \n\t"
            "vsetvli      t0, %[K], e32, m8       \n\t"
            "vle32.v      v0, (a1)                \n\t"
            "sub          %[K], %[K], t0          \n\t"
            "vsetvli      t0, %[K], e32, m8       \n\t"
            "vle32.v      v8, (a2)                \n\t"
            "sub          %[K], %[K], t0          \n\t"
            "vsetvli      t1, zero, e32, m8       \n\t"
            "jal          x0, QUANT%=             \n\t"
            "END%=:                               \n\t"

            : [DST] "+r"(DST), [K] "+r"(CountK)
            : [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal), [SRC] "r"(SRC)
            : "cc", "t2", "t1", "t0", "a1", "a2", "f10", "f11");
    } else {
        float  buffer[8] = { 0.0f };
        size_t cnt       = BlkLen / 256;

        __asm__ volatile(
            "slli         t3, %[BLK], 2           \n\t"
            "blt       %[K], %[BLK], LOOP_TAIL%=  \n\t"
            "LOOP_MAIN%=:                         \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vxor.vv      v31, v31, v31           \n\t"
            "vse32.v      v31, (%[BUFFER])        \n\t"
            "addi         t6, %[CNT], 0           \n\t"
            "LOOP_CMP%=:                          \n\t"
            "addi         t6, t6, -1              \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vle32.v      v0, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v8, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v16, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v24, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vfabs.v      v0, v0                  \n\t"
            "vfabs.v      v8, v8                  \n\t"
            "vfabs.v      v16, v16                \n\t"
            "vfabs.v      v24, v24                \n\t"
            "vfmax.vv     v8, v0, v8              \n\t"
            "vfmax.vv     v16, v16, v24           \n\t"
            "vfmax.vv     v0, v0, v16             \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmax.vv     v0, v0, v4              \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v0, v0, v2              \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v0, v0, v1              \n\t"
            "vle32.v      v30, (%[BUFFER])        \n\t"
            "vfmax.vv     v31, v30,  v0           \n\t"
            "vse32.v      v31, (%[BUFFER])        \n\t"
            "bnez         t6, LOOP_CMP%=          \n\t"
            "sub          %[SRC], %[SRC], t3      \n\t"
            "addi         t6, %[CNT], 0           \n\t"
            "flw          f0, (%[BUFFER])         \n\t"
            "flw          f1, 4(%[BUFFER])        \n\t"
            "flw          f2, 8(%[BUFFER])        \n\t"
            "flw          f3, 12(%[BUFFER])       \n\t"
            "flw          f4, 16(%[BUFFER])       \n\t"
            "flw          f5, 20(%[BUFFER])       \n\t"
            "flw          f6, 24(%[BUFFER])       \n\t"
            "flw          f7, 28(%[BUFFER])       \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f10, f3, f7             \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10,  (%[DST])          \n\t"
            "addi         %[DST], %[DST], 4       \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "addi         t6,  %[CNT], 0          \n\t"
            "LOOP_QUANT%=:                        \n\t"
            "addi         t6, t6, -1              \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vle32.v      v0, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v8, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v16, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vle32.v      v24, (%[SRC])           \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vfmul.vf     v0, v0, f11             \n\t"
            "vfmul.vf     v8, v8, f11             \n\t"
            "vfmul.vf     v16, v16, f11           \n\t"
            "vfmul.vf     v24, v24, f11           \n\t"
            "vfcvt.x.f.v  v0, v0                  \n\t"
            "vfcvt.x.f.v  v8, v8                  \n\t"
            "vfcvt.x.f.v  v16, v16                \n\t"
            "vfcvt.x.f.v  v24, v24                \n\t"
            "vsetvli      t0, zero, e16, m4       \n\t"
            "vnclip.wx    v0, v0, zero            \n\t"
            "vnclip.wx    v4, v8, zero            \n\t"
            "vnclip.wx    v8, v16, zero           \n\t"
            "vnclip.wx    v12, v24, zero          \n\t"
            "vsetvli      t0, zero, e8, m4        \n\t"
            "vnclip.wx    v0, v0, zero            \n\t"
            "vnclip.wx    v4, v8, zero            \n\t"
            "vse8.v       v0, (%[DST])            \n\t"
            "addi         %[DST], %[DST], 128     \n\t"
            "vse8.v       v4, (%[DST])            \n\t"
            "addi         %[DST], %[DST], 128     \n\t"
            "bnez         t6, LOOP_QUANT%=        \n\t"
            "sub           %[K], %[K], %[BLK]     \n\t"
            "bge        %[K], %[BLK], LOOP_MAIN%= \n\t"
            "blez         %[K], END%=             \n\t"
            "LOOP_TAIL%=:                         \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vxor.vv      v31, v31, v31           \n\t"
            "vse32.v      v31, (%[BUFFER])        \n\t"
            "addi         t6, %[K], 0             \n\t"
            "addi         s1, %[SRC], 0           \n\t"
            "TAIL_CMP%=:                          \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vxor.vv       v0, v0, v0             \n\t"
            "vsetvli      t0, t6, e32, m8         \n\t"
            "vle32.v      v0, (%[SRC])            \n\t"
            "addi         %[SRC], %[SRC], 256     \n\t"
            "sub          t6, t6, t0              \n\t"
            "vfabs.v      v0, v0                  \n\t"
            "vsetvli      t0, zero, e32, m4       \n\t"
            "vfmax.vv     v0, v0, v4              \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vfmax.vv     v0, v0, v2              \n\t"
            "vsetvli      t0, zero, e32, m1       \n\t"
            "vfmax.vv     v0, v0, v1              \n\t"
            "vle32.v      v30, (%[BUFFER])        \n\t"
            "vfmax.vv     v31, v30,  v0           \n\t"
            "vse32.v      v31, (%[BUFFER])        \n\t"
            "bnez         t6, TAIL_CMP%=          \n\t"
            "addi         t6, %[K], 0             \n\t"
            "flw          f0, (%[BUFFER])         \n\t"
            "flw          f1, 4(%[BUFFER])        \n\t"
            "flw          f2, 8(%[BUFFER])        \n\t"
            "flw          f3, 12(%[BUFFER])       \n\t"
            "flw          f4, 16(%[BUFFER])       \n\t"
            "flw          f5, 20(%[BUFFER])       \n\t"
            "flw          f6, 24(%[BUFFER])       \n\t"
            "flw          f7, 28(%[BUFFER])       \n\t"
            "fmax.s       f1, f0, f1              \n\t"
            "fmax.s       f3, f2, f3              \n\t"
            "fmax.s       f5, f4, f5              \n\t"
            "fmax.s       f7, f6, f7              \n\t"
            "fmax.s       f3, f1, f3              \n\t"
            "fmax.s       f7, f5, f7              \n\t"
            "fmax.s       f10, f3, f7             \n\t"
            "fmul.s       f10, f10, %[RMAXREC]    \n\t"
            "fsw          f10,  (%[DST])          \n\t"
            "addi         %[DST], %[DST], 4       \n\t"
            "fdiv.s       f11, %[FONE], f10       \n\t"
            "addi         t6,  %[K], 0            \n\t"
            "TAIL_QUANT%=:                        \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vxor.vv       v0, v0, v0             \n\t"
            "vsetvli      t1, t6, e32, m8         \n\t"
            "vle32.v      v0, (s1)                \n\t"
            "addi         s1, s1, 256             \n\t"
            "sub          t6, t6, t1              \n\t"
            "vsetvli      t0, zero, e32, m8       \n\t"
            "vfmul.vf     v0, v0, f11             \n\t"
            "vfcvt.x.f.v  v0, v0                  \n\t"
            "vsetvli      t0, zero, e16, m4       \n\t"
            "vnclip.wx    v0, v0, zero            \n\t"
            "vsetvli      t0, t1, e8, m2          \n\t"
            "vnclip.wx    v0, v0, zero            \n\t"
            "vse8.v       v0, (%[DST])            \n\t"
            "addi         %[DST], %[DST], 64      \n\t"
            "bnez         t6, TAIL_QUANT%=        \n\t"
            "END%=:                               \n\t"
            : [SRC] "+r"(SRC), [DST] "+r"(DST), [K] "+r"(CountK)
            : [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal), [BLK] "r"(BlkLen), [BUFFER] "r"(buffer),
              [CNT] "r"(cnt)
            : "cc", "t1", "t0", "t6", "s1", "f0", "f1", "f2", "f3", "f4", "f5", "f6");
    }
}

}  // namespace ime1

namespace {
#define SQ4BIT_KERNEL_COMP_1x8x2_4X8X4          \
    "vmadot       v16, v14, v0            \n\t" \
    "vmadot       v18, v14, v1            \n\t" \
    "vmadot       v20, v14, v2            \n\t" \
    "vmadot       v22, v14, v3            \n\t" \
    "vmadot       v16, v15, v4            \n\t" \
    "vmadot       v18, v15, v5            \n\t" \
    "vmadot       v20, v15, v6            \n\t" \
    "vmadot       v22, v15, v7            \n\t"

#define SQ4BIT_KERNEL_ACC_1X4X4                 \
    "vfcvt.f.x.v  v16,  v16               \n\t" \
    "vfcvt.f.x.v  v18,  v18               \n\t" \
    "vfcvt.f.x.v  v20,  v20               \n\t" \
    "vfcvt.f.x.v  v22,  v22               \n\t" \
    "addi         s2, s1, 16              \n\t" \
    "addi         s3, s1, 32              \n\t" \
    "addi         s4, s1, 48              \n\t" \
    "addi         s6, s5, 12              \n\t" \
    "vfmacc.vv    v28, v16, v24           \n\t" \
    "vfmacc.vv    v29, v18, v25           \n\t" \
    "vfmacc.vv    v30, v20, v26           \n\t" \
    "vfmacc.vv    v31, v22, v27           \n\t"

#define SQ4BIT_KERNEL_ACC_F16_1X4X4             \
    "vfcvt.f.x.v  v16,  v16               \n\t" \
    "vfcvt.f.x.v  v18,  v18               \n\t" \
    "vfcvt.f.x.v  v20,  v20               \n\t" \
    "vfcvt.f.x.v  v22,  v22               \n\t" \
    "addi         s2, s1, 8               \n\t" \
    "addi         s3, s1, 16              \n\t" \
    "addi         s4, s1, 24              \n\t" \
    "addi         s6, s5, 12              \n\t" \
    "vfmacc.vv    v28, v16, v24           \n\t" \
    "vfmacc.vv    v29, v18, v25           \n\t" \
    "vfmacc.vv    v30, v20, v26           \n\t" \
    "vfmacc.vv    v31, v22, v27           \n\t"

#define SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4          \
    "vle8.v       v4, (s1)                \n\t" \
    "addi         s1, s1, 128             \n\t" \
    "vle8.v       v5, (s2)                \n\t" \
    "addi         s2, s2, 128             \n\t" \
    "vle8.v       v6, (s3)                \n\t" \
    "addi         s3, s3, 128             \n\t" \
    "vle8.v       v7, (s4)                \n\t" \
    "addi         s4, s4, 128             \n\t" \
    "vsetvli      t0, zero, e8, mf4       \n\t" \
    "vle8.v       v14, (s5)               \n\t" \
    "addi         s5, s5, 16              \n\t" \
    "vle8.v       v15, (s6)               \n\t" \
    "addi         s6, s6, 16              \n\t" \
    "addi         t5, t5, -1              \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vand.vi      v0, v4, 15              \n\t" \
    "vand.vi      v1, v5, 15              \n\t" \
    "vand.vi      v2, v6, 15              \n\t" \
    "vand.vi      v3, v7, 15              \n\t" \
    "vsrl.vi      v4, v4, 4               \n\t" \
    "vsrl.vi      v5, v5, 4               \n\t" \
    "vsrl.vi      v6, v6, 4               \n\t" \
    "vsrl.vi      v7, v7, 4               \n\t"

#define SQ4BIT_KERNEL_LOAD_ZP_16X1              \
    "vsetvli      t0, zero, e8, mf2       \n\t" \
    "vle8.v       v1, (s7)                \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vrgather.vv  v8, v1, v13             \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v9, v1, v13             \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v10, v1, v13            \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v11, v1, v13            \n\t" \
    "vadd.vi      v13, v13, -12           \n\t"

// using for M4Kernel
#define LOAD_B_16x8x2                           \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vle8.v       v6, (s1)                \n\t" \
    "addi         s1, s1, 32*4            \n\t" \
    "vle8.v       v7, (s2)                \n\t" \
    "addi         s2, s2, 32*4            \n\t" \
    "vle8.v       v8, (s3)                \n\t" \
    "addi         s3, s3, 32*4            \n\t" \
    "vle8.v       v9, (s4)                \n\t" \
    "addi         s4, s4, 32*4            \n\t" \
                                                \
    "vand.vi      v2, v6, 15              \n\t" \
    "vand.vi      v3, v7, 15              \n\t" \
    "vand.vi      v4, v8, 15              \n\t" \
    "vand.vi      v5, v9, 15              \n\t" \
                                                \
    "vsrl.vi      v6, v6, 4               \n\t" \
    "vsrl.vi      v7, v7, 4               \n\t" \
    "vsrl.vi      v8, v8, 4               \n\t" \
    "vsrl.vi      v9, v9, 4               \n\t"

// [s2|s5, s3, s4, s6]
#define LOAD_SCALE_4x16_FP16                    \
    "addi         s2, s5, -8              \n\t" \
    "addi         s3, s5, 8               \n\t" \
    "addi         s4, s5, 16              \n\t" \
    "addi         s6, s5, 24              \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e16, mf4      \n\t" \
    "vle16.v      v9, (s5)                \n\t" \
    "vle16.v      v11, (s3)               \n\t" \
    "vle16.v      v13, (s4)               \n\t" \
    "vle16.v      v15, (s6)               \n\t" \
    "vsetvli      t0, zero, e16, mf2      \n\t" \
    "vle16.v      v9, (s2), v0.t          \n\t" \
    "vle16.v      v11, (s5), v0.t         \n\t" \
    "vle16.v      v13, (s3), v0.t         \n\t" \
    "vle16.v      v15, (s4), v0.t         \n\t" \
    "vfwcvt.f.f.v v8, v9                  \n\t" \
    "vfwcvt.f.f.v v10, v11                \n\t" \
    "vfwcvt.f.f.v v12, v13                \n\t" \
    "vfwcvt.f.f.v v14, v15                \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vmv.v.v      v9, v8                  \n\t" \
    "vmv.v.v      v11, v10                \n\t" \
    "vmv.v.v      v13, v12                \n\t" \
    "vmv.v.v      v15, v14                \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vfmul.vf     v8, v8, f1              \n\t" \
    "vfmul.vf     v10, v10, f1            \n\t" \
    "vfmul.vf     v12, v12, f1            \n\t" \
    "vfmul.vf     v14, v14, f1            \n\t" \
    "vfmul.vf     v9, v9, f3              \n\t" \
    "vfmul.vf     v11, v11, f3            \n\t" \
    "vfmul.vf     v13, v13, f3            \n\t" \
    "vfmul.vf     v15, v15, f3            \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vfmul.vf     v8, v8, f2, v0.t        \n\t" \
    "vfmul.vf     v10, v10, f2, v0.t      \n\t" \
    "vfmul.vf     v12, v12, f2, v0.t      \n\t" \
    "vfmul.vf     v14, v14, f2, v0.t      \n\t" \
    "vfmul.vf     v9, v9, f4, v0.t        \n\t" \
    "vfmul.vf     v11, v11, f4, v0.t      \n\t" \
    "vfmul.vf     v13, v13, f4, v0.t      \n\t" \
    "vfmul.vf     v15, v15, f4, v0.t      \n\t"

// [s2|s5, s3, s4, s6]
#define LOAD_SCALE_4x16                         \
    "addi         s2, s5, -16             \n\t" \
    "addi         s3, s5, 16              \n\t" \
    "addi         s4, s5, 32              \n\t" \
    "addi         s6, s5, 48              \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vle32.v      v8, (s5)                \n\t" \
    "vle32.v      v10, (s3)               \n\t" \
    "vle32.v      v12, (s4)               \n\t" \
    "vle32.v      v14, (s6)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vle32.v      v8, (s2), v0.t          \n\t" \
    "vle32.v      v10, (s5), v0.t         \n\t" \
    "vle32.v      v12, (s3), v0.t         \n\t" \
    "vle32.v      v14, (s4), v0.t         \n\t" \
    "vmv.v.v      v9, v8                  \n\t" \
    "vmv.v.v      v11, v10                \n\t" \
    "vmv.v.v      v13, v12                \n\t" \
    "vmv.v.v      v15, v14                \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vfmul.vf     v8, v8, f1              \n\t" \
    "vfmul.vf     v10, v10, f1            \n\t" \
    "vfmul.vf     v12, v12, f1            \n\t" \
    "vfmul.vf     v14, v14, f1            \n\t" \
    "vfmul.vf     v9, v9, f3              \n\t" \
    "vfmul.vf     v11, v11, f3            \n\t" \
    "vfmul.vf     v13, v13, f3            \n\t" \
    "vfmul.vf     v15, v15, f3            \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vfmul.vf     v8, v8, f2, v0.t        \n\t" \
    "vfmul.vf     v10, v10, f2, v0.t      \n\t" \
    "vfmul.vf     v12, v12, f2, v0.t      \n\t" \
    "vfmul.vf     v14, v14, f2, v0.t      \n\t" \
    "vfmul.vf     v9, v9, f4, v0.t        \n\t" \
    "vfmul.vf     v11, v11, f4, v0.t      \n\t" \
    "vfmul.vf     v13, v13, f4, v0.t      \n\t" \
    "vfmul.vf     v15, v15, f4, v0.t      \n\t"

//[s1| BIAS, s2, s3, s4]
#define LOAD_BIAS                               \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "addi         s1, %[BIAS], -16        \n\t" \
    "addi         s2, %[BIAS], 16         \n\t" \
    "addi         s3, %[BIAS], 32         \n\t" \
    "addi         s4, %[BIAS], 48         \n\t" \
                                                \
    "vle32.v      v24, (%[BIAS])          \n\t" \
    "vle32.v      v26, (s2)               \n\t" \
    "vle32.v      v28, (s3)               \n\t" \
    "vle32.v      v30, (s4)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vle32.v      v24, (s1), v0.t         \n\t" \
    "vle32.v      v26, (%[BIAS]), v0.t    \n\t" \
    "vle32.v      v28, (s2), v0.t         \n\t" \
    "vle32.v      v30, (s3), v0.t         \n\t" \
    "vmv.v.v      v25, v24                \n\t" \
    "vmv.v.v      v27, v26                \n\t" \
    "vmv.v.v      v29, v28                \n\t" \
    "vmv.v.v      v31, v30                \n\t"

#define SQ4BIT_KERNEL_COMP_4x16x16              \
    "vmadot       v16, v10, v2            \n\t" \
    "vmadot       v18, v10, v3            \n\t" \
    "vmadot       v20, v10, v4            \n\t" \
    "vmadot       v22, v10, v5            \n\t" \
    "vmadot       v16, v11, v6            \n\t" \
    "vmadot       v18, v11, v7            \n\t" \
    "vmadot       v20, v11, v8            \n\t" \
    "vmadot       v22, v11, v9            \n\t"

#define SAVE_RESULT_4x16                        \
    "addi         a1, %[C], 0             \n\t" \
    "add          a2, %[C], %[LDC]        \n\t" \
    "add          a3, a2, %[LDC]          \n\t" \
    "add          a4, a3, %[LDC]          \n\t" \
    "addi         a2, a2, -16             \n\t" \
    "addi         a4, a4, -16             \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
                                                \
    "vse32.v      v24, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v25, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v26, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v27, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v28, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v29, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v30, (a1)               \n\t" \
    "vse32.v      v31, (a3)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
                                                \
    "vse32.v      v24, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v25, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v26, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v27, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v28, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v29, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v30, (a2), v0.t         \n\t" \
    "vse32.v      v31, (a4), v0.t         \n\t"

#define SQ4BIT_KERNEL_LOAD_ZP_16X1_v2           \
    "vsetvli      t0, zero, e8, mf2       \n\t" \
    "vle8.v       v11, (s6)               \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vrgather.vv  v12, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v13, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v14, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v15, v11, v1            \n\t" \
    "vadd.vi      v1, v1, -12             \n\t"

template <bool HasZeroPoint>
void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl(size_t            BlkLen,
                                                const std::byte * QuantA,
                                                const std::byte * QuantBData,
                                                const float *     QuantBScale,
                                                const std::byte * QuantBZeroPoint,
                                                float *           C,
                                                size_t            CountN,
                                                size_t            BlockCountK,
                                                const float *     Bias,
                                                const size_t      ldc) {
    GGML_UNUSED(QuantBScale);
    GGML_UNUSED(QuantBZeroPoint);
    size_t       LDC   = ldc * sizeof(float);
    const size_t INNER = BlkLen / 16;
    float        tmp[4 * 16];

    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +           //
                                        n * BlockCountK * BlkLen / 2 +       // b data
                                        n * BlockCountK * sizeof(uint8_t) +  // zp
                                        n * BlockCountK * sizeof(_Float16);    // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                if (NBLKS < 16) {
                    __asm__ volatile(
                        "vsetvli        t0, %[N], e32, m2     \n\t"
                        "vle32.v        v0, (%[SRC])          \n\t"
                        "vse32.v        v0, (%[DST])          \n\t"
                        :
                        : [SRC] "r"(bias), [DST] "r"(tmp), [N] "r"(NBLKS)
                        : "cc", "t0");
                    bias = tmp;
                }
                __asm__ volatile(LOAD_BIAS

                                 "addi               t3, %[BlockCountK], 0       \n\t"

                                 "vsetvli            t0, zero, e8, m1            \n\t"
                                 "li                 s1, 24                      \n\t"
                                 "vmv.v.i            v1, 3                       \n\t"
                                 "vsetvli            t0, s1, e8, m1              \n\t"
                                 "vmv.v.i            v1, 2                       \n\t"
                                 "vsetvli            t0, zero, e8, mf2           \n\t"
                                 "vmv.v.i            v1, 1                       \n\t"
                                 "vsetvli            t0, zero, e8, mf4           \n\t"
                                 "vmv.v.i            v1, 0                       \n\t"

                                 "addi               a1, %[A], 0                 \n\t"
                                 "addi               s1, %[B], 0                 \n\t"

                                 "BLOCK_COUNTK_LOOP%=:                           \n\t"
                                 // scale offset
                                 "addi               s5, s1, 0                   \n\t"
                                 // zp offset
                                 "addi               s6, s1, 32                  \n\t"
                                 "addi               s1, s6, 16                  \n\t"
                                 "addi               s2, s1, 32                  \n\t"
                                 "addi               s3, s1, 32*2                \n\t"
                                 "addi               s4, s1, 32*3                \n\t"

                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vxor.vv            v16, v16, v16               \n\t"
                                 // load a scale
                                 "flw                f1, (a1)                    \n\t"
                                 "flw                f2, 4(a1)                   \n\t"
                                 "flw                f3, 8(a1)                   \n\t"
                                 "flw                f4, 12(a1)                  \n\t"
                                 "addi               a1, a1, 16                  \n\t"
                                 "addi               t2, %[INNER], 0             \n\t"

                                 SQ4BIT_KERNEL_LOAD_ZP_16X1_v2

                                 "BLOCK_INNER_LOOP%=:                            \n\t"

                                 LOAD_B_16x8x2

                                 "vle8.v             v10, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vle8.v             v11, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vsub.vv            v2, v2, v12                 \n\t"
                                 "vsub.vv            v6, v6, v12                 \n\t"
                                 "vsub.vv            v3, v3, v13                 \n\t"
                                 "vsub.vv            v7, v7, v13                 \n\t"
                                 "vsub.vv            v4, v4, v14                 \n\t"
                                 "vsub.vv            v8, v8, v14                 \n\t"
                                 "vsub.vv            v5, v5, v15                 \n\t"
                                 "vsub.vv            v9, v9, v15                 \n\t"

                                 SQ4BIT_KERNEL_COMP_4x16x16

                                 "addi               t2, t2, -1                  \n\t"
                                 "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                                 LOAD_SCALE_4x16_FP16

                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vfcvt.f.x.v        v16, v16                    \n\t"
                                 "vfmacc.vv          v24, v16, v8                \n\t"
                                 "addi               t3, t3, -1                  \n\t"
                                 "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                                 "RESULT_SAVE%=:                                 \n\t"

                                 SAVE_RESULT_4x16

                                 :
                                 : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                                   [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr), [BIAS] "r"(bias)
                                 : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1",
                                   "s2", "s3", "s4", "s5", "s6");

            } else {
                __asm__ volatile(
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v24, v24, v24               \n\t"
                    "addi               t3, %[BlockCountK], 0       \n\t"
                    "vsetvli            t0, zero, e8, m1            \n\t"
                    "li                 s1, 24                      \n\t"
                    "vmv.v.i            v1, 3                       \n\t"
                    "vsetvli            t0, s1, e8, m1              \n\t"
                    "vmv.v.i            v1, 2                       \n\t"
                    "vsetvli            t0, zero, e8, mf2           \n\t"
                    "vmv.v.i            v1, 1                       \n\t"
                    "vsetvli            t0, zero, e8, mf4           \n\t"
                    "vmv.v.i            v1, 0                       \n\t"
                    "addi               a1, %[A], 0                 \n\t"
                    "addi               s1, %[B], 0                 \n\t"
                    "BLOCK_COUNTK_LOOP%=:                           \n\t"
                    // scale offset
                    "addi               s5, s1, 0                   \n\t"
                    // zp offset
                    "addi               s6, s1, 32                  \n\t"
                    "addi               s1, s6, 16                  \n\t"
                    "addi               s2, s1, 32                  \n\t"
                    "addi               s3, s1, 32*2                \n\t"
                    "addi               s4, s1, 32*3                \n\t"

                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v16, v16, v16               \n\t"
                    // load a scale
                    "flw                f1, (a1)                    \n\t"
                    "flw                f2, 4(a1)                   \n\t"
                    "flw                f3, 8(a1)                   \n\t"
                    "flw                f4, 12(a1)                  \n\t"
                    "addi               a1, a1, 16                  \n\t"
                    "addi               t2, %[INNER], 0             \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1_v2

                    "BLOCK_INNER_LOOP%=:                            \n\t"

                    LOAD_B_16x8x2

                    "vle8.v             v10, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vle8.v             v11, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vsub.vv            v2, v2, v12                 \n\t"
                    "vsub.vv            v6, v6, v12                 \n\t"
                    "vsub.vv            v3, v3, v13                 \n\t"
                    "vsub.vv            v7, v7, v13                 \n\t"
                    "vsub.vv            v4, v4, v14                 \n\t"
                    "vsub.vv            v8, v8, v14                 \n\t"
                    "vsub.vv            v5, v5, v15                 \n\t"
                    "vsub.vv            v9, v9, v15                 \n\t"

                    SQ4BIT_KERNEL_COMP_4x16x16

                    "addi               t2, t2, -1                  \n\t"
                    "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                    LOAD_SCALE_4x16_FP16

                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vfcvt.f.x.v        v16, v16                    \n\t"
                    "vfmacc.vv          v24, v16, v8                \n\t"
                    "addi               t3, t3, -1                  \n\t"
                    "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                    "RESULT_SAVE%=:                                 \n\t"

                    SAVE_RESULT_4x16

                    :
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                      [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                    : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3",
                      "s4", "s5", "s6");
            }
        }
    } else {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +         //
                                        n * BlockCountK * BlkLen / 2 +     // b data
                                        n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                if (NBLKS < 16) {
                    __asm__ volatile(
                        "vsetvli        t0, %[N], e32, m2     \n\t"
                        "vle32.v        v0, (%[SRC])          \n\t"
                        "vse32.v        v0, (%[DST])          \n\t"
                        :
                        : [SRC] "r"(bias), [DST] "r"(tmp), [N] "r"(NBLKS)
                        : "cc", "t0");
                    bias = tmp;
                }
                __asm__ volatile(LOAD_BIAS

                                 "addi               t3, %[BlockCountK], 0       \n\t"
                                 "addi               a1, %[A], 0                 \n\t"
                                 "addi               s1, %[B], 0                 \n\t"
                                 "BLOCK_COUNTK_LOOP%=:                           \n\t"
                                 "addi               s5, s1, 0                   \n\t"
                                 "addi               s1, s5, 32                  \n\t"
                                 "addi               s2, s1, 32                  \n\t"
                                 "addi               s3, s1, 32*2                \n\t"
                                 "addi               s4, s1, 32*3                \n\t"
                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vxor.vv            v16, v16, v16               \n\t"
                                 // load a scale
                                 "flw                f1, (a1)                    \n\t"
                                 "flw                f2, 4(a1)                   \n\t"
                                 "flw                f3, 8(a1)                   \n\t"
                                 "flw                f4, 12(a1)                  \n\t"
                                 "addi               a1, a1, 16                  \n\t"
                                 "addi               t2, %[INNER], 0             \n\t"
                                 "BLOCK_INNER_LOOP%=:                            \n\t"

                                 LOAD_B_16x8x2

                                 "vsetvli            t0, zero, e8, m1            \n\t"
                                 "vle8.v             v10, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vle8.v             v11, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vadd.vi            v2, v2, -8                  \n\t"
                                 "vadd.vi            v3, v3, -8                  \n\t"
                                 "vadd.vi            v4, v4, -8                  \n\t"
                                 "vadd.vi            v5, v5, -8                  \n\t"
                                 "vadd.vi            v6, v6, -8                  \n\t"
                                 "vadd.vi            v7, v7, -8                  \n\t"
                                 "vadd.vi            v8, v8, -8                  \n\t"
                                 "vadd.vi            v9, v9, -8                  \n\t"

                                 SQ4BIT_KERNEL_COMP_4x16x16

                                 "addi               t2, t2, -1                  \n\t"
                                 "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                                 LOAD_SCALE_4x16_FP16

                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vfcvt.f.x.v        v16, v16                    \n\t"
                                 "vfmacc.vv          v24, v16, v8                \n\t"
                                 "addi               t3, t3, -1                  \n\t"
                                 "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"
                                 "RESULT_SAVE%=:                                 \n\t"

                                 SAVE_RESULT_4x16

                                 :
                                 : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                                   [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr), [BIAS] "r"(bias)
                                 : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1",
                                   "s2", "s3", "s4", "s5", "s6");

            } else {
                __asm__ volatile(
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v24, v24, v24               \n\t"
                    "addi               t3, %[BlockCountK], 0       \n\t"
                    "addi               a1, %[A], 0                 \n\t"
                    "addi               s1, %[B], 0                 \n\t"
                    "BLOCK_COUNTK_LOOP%=:                           \n\t"
                    "addi               s5, s1, 0                   \n\t"
                    "addi               s1, s5, 32                  \n\t"
                    "addi               s2, s1, 32                  \n\t"
                    "addi               s3, s1, 32*2                \n\t"
                    "addi               s4, s1, 32*3                \n\t"
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v16, v16, v16               \n\t"
                    // load a scale
                    "flw                f1, (a1)                    \n\t"
                    "flw                f2, 4(a1)                   \n\t"
                    "flw                f3, 8(a1)                   \n\t"
                    "flw                f4, 12(a1)                  \n\t"
                    "addi               a1, a1, 16                  \n\t"
                    "addi               t2, %[INNER], 0             \n\t"
                    "BLOCK_INNER_LOOP%=:                            \n\t"

                    LOAD_B_16x8x2

                    "vsetvli            t0, zero, e8, m1            \n\t"
                    "vle8.v             v10, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vle8.v             v11, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vadd.vi            v2, v2, -8                  \n\t"
                    "vadd.vi            v3, v3, -8                  \n\t"
                    "vadd.vi            v4, v4, -8                  \n\t"
                    "vadd.vi            v5, v5, -8                  \n\t"
                    "vadd.vi            v6, v6, -8                  \n\t"
                    "vadd.vi            v7, v7, -8                  \n\t"
                    "vadd.vi            v8, v8, -8                  \n\t"
                    "vadd.vi            v9, v9, -8                  \n\t"

                    SQ4BIT_KERNEL_COMP_4x16x16

                    "addi               t2, t2, -1                  \n\t"
                    "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                    LOAD_SCALE_4x16_FP16

                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vfcvt.f.x.v        v16, v16                    \n\t"
                    "vfmacc.vv          v24, v16, v8                \n\t"
                    "addi               t3, t3, -1                  \n\t"
                    "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"
                    "RESULT_SAVE%=:                                 \n\t"

                    SAVE_RESULT_4x16

                    :
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                      [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                    : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3",
                      "s4", "s5", "s6");
            }
        }
    }
    if (CountN % 16 != 0) {
        // stroe output from tmp to C when NBLKS less than 16.
        float *      CPtr = C + CountN / 16 * 16;
        const size_t N    = CountN % 16;
        LDC               = ldc * sizeof(float);
        __asm__ volatile(
            "vsetvli            t0, %[N], e32, m2       \n\t"
            "vle32.v            v0, (%[SRC])            \n\t"
            "addi               s2, %[SRC], 64          \n\t"
            "addi               s3, %[SRC], 64*2        \n\t"
            "addi               s4, %[SRC], 64*3        \n\t"
            "vle32.v            v2, (s2)                \n\t"
            "vle32.v            v4, (s3)                \n\t"
            "vle32.v            v6, (s4)                \n\t"
            "add                t2, %[DST], %[LDC]      \n\t"
            "add                t3, t2, %[LDC]          \n\t"
            "add                t4, t3, %[LDC]          \n\t"
            "vse32.v            v0, (%[DST])            \n\t"
            "vse32.v            v2, (t2)                \n\t"
            "vse32.v            v4, (t3)                \n\t"
            "vse32.v            v6, (t4)                \n\t"
            :
            : [N] "r"(N), [SRC] "r"(tmp), [DST] "r"(CPtr), [LDC] "r"(LDC)
            : "cc", "t0", "t2", "t3", "t4", "s2", "s3", "s4");
    }
}

template <bool HasZeroPoint>
void SQ4BitGemmM4Kernel_CompInt8_Impl(size_t            BlkLen,
                                      const std::byte * QuantA,
                                      const std::byte * QuantBData,
                                      const float *     QuantBScale,
                                      const std::byte * QuantBZeroPoint,
                                      float *           C,
                                      size_t            CountN,
                                      size_t            BlockCountK,
                                      const float *     Bias,
                                      const size_t      ldc) {
    GGML_UNUSED(QuantBScale);
    GGML_UNUSED(QuantBZeroPoint);
    size_t       LDC   = ldc * sizeof(float);
    const size_t INNER = BlkLen / 16;
    float        tmp[4 * 16];

    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +           //
                                        n * BlockCountK * BlkLen / 2 +       // b data
                                        n * BlockCountK * sizeof(uint8_t) +  // zp
                                        n * BlockCountK * sizeof(float);     // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                if (NBLKS < 16) {
                    __asm__ volatile(
                        "vsetvli        t0, %[N], e32, m2     \n\t"
                        "vle32.v        v0, (%[SRC])          \n\t"
                        "vse32.v        v0, (%[DST])          \n\t"
                        :
                        : [SRC] "r"(bias), [DST] "r"(tmp), [N] "r"(NBLKS)
                        : "cc", "t0");
                    bias = tmp;
                }

                __asm__ volatile(LOAD_BIAS
                                 "addi               t3, %[BlockCountK], 0       \n\t"
                                 "vsetvli            t0, zero, e8, m1            \n\t"
                                 "li                 s1, 24                      \n\t"
                                 "vmv.v.i            v1, 3                       \n\t"
                                 "vsetvli            t0, s1, e8, m1              \n\t"
                                 "vmv.v.i            v1, 2                       \n\t"
                                 "vsetvli            t0, zero, e8, mf2           \n\t"
                                 "vmv.v.i            v1, 1                       \n\t"
                                 "vsetvli            t0, zero, e8, mf4           \n\t"
                                 "vmv.v.i            v1, 0                       \n\t"
                                 "addi               a1, %[A], 0                 \n\t"
                                 "addi               s1, %[B], 0                 \n\t"
                                 "BLOCK_COUNTK_LOOP%=:                           \n\t"
                                 // scale offset
                                 "addi               s5, s1, 0                   \n\t"
                                 // zp offset
                                 "addi               s6, s1, 64                  \n\t"
                                 "addi               s1, s6, 16                  \n\t"
                                 "addi               s2, s1, 32                  \n\t"
                                 "addi               s3, s1, 32*2                \n\t"
                                 "addi               s4, s1, 32*3                \n\t"
                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vxor.vv            v16, v16, v16               \n\t"
                                 // load a scale
                                 "flw                f1, (a1)                    \n\t"
                                 "flw                f2, 4(a1)                   \n\t"
                                 "flw                f3, 8(a1)                   \n\t"
                                 "flw                f4, 12(a1)                  \n\t"
                                 "addi               a1, a1, 16                  \n\t"
                                 "addi               t2, %[INNER], 0             \n\t"

                                 SQ4BIT_KERNEL_LOAD_ZP_16X1_v2

                                 "BLOCK_INNER_LOOP%=:                            \n\t"

                                 LOAD_B_16x8x2

                                 "vle8.v             v10, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vle8.v             v11, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vsub.vv            v2, v2, v12                 \n\t"
                                 "vsub.vv            v6, v6, v12                 \n\t"
                                 "vsub.vv            v3, v3, v13                 \n\t"
                                 "vsub.vv            v7, v7, v13                 \n\t"
                                 "vsub.vv            v4, v4, v14                 \n\t"
                                 "vsub.vv            v8, v8, v14                 \n\t"
                                 "vsub.vv            v5, v5, v15                 \n\t"
                                 "vsub.vv            v9, v9, v15                 \n\t"

                                 SQ4BIT_KERNEL_COMP_4x16x16

                                 "addi               t2, t2, -1                  \n\t"
                                 "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                                 LOAD_SCALE_4x16

                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vfcvt.f.x.v        v16, v16                    \n\t"
                                 "vfmacc.vv          v24, v16, v8                \n\t"
                                 "addi               t3, t3, -1                  \n\t"
                                 "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                                 "RESULT_SAVE%=:                                 \n\t"

                                 SAVE_RESULT_4x16

                                 :
                                 : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                                   [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr), [BIAS] "r"(bias)
                                 : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1",
                                   "s2", "s3", "s4", "s5", "s6");

            } else {
                __asm__ volatile(
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v24, v24, v24               \n\t"
                    "addi               t3, %[BlockCountK], 0       \n\t"
                    "vsetvli            t0, zero, e8, m1            \n\t"
                    "li                 s1, 24                      \n\t"
                    "vmv.v.i            v1, 3                       \n\t"
                    "vsetvli            t0, s1, e8, m1              \n\t"
                    "vmv.v.i            v1, 2                       \n\t"
                    "vsetvli            t0, zero, e8, mf2           \n\t"
                    "vmv.v.i            v1, 1                       \n\t"
                    "vsetvli            t0, zero, e8, mf4           \n\t"
                    "vmv.v.i            v1, 0                       \n\t"
                    "addi               a1, %[A], 0                 \n\t"
                    "addi               s1, %[B], 0                 \n\t"
                    "BLOCK_COUNTK_LOOP%=:                           \n\t"
                    // scale offset
                    "addi               s5, s1, 0                   \n\t"
                    // zp offset
                    "addi               s6, s1, 64                  \n\t"
                    "addi               s1, s6, 16                  \n\t"
                    "addi               s2, s1, 32                  \n\t"
                    "addi               s3, s1, 32*2                \n\t"
                    "addi               s4, s1, 32*3                \n\t"
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v16, v16, v16               \n\t"
                    // load a scale
                    // load a scale
                    "flw                f1, (a1)                    \n\t"
                    "flw                f2, 4(a1)                   \n\t"
                    "flw                f3, 8(a1)                   \n\t"
                    "flw                f4, 12(a1)                  \n\t"
                    "addi               a1, a1, 16                  \n\t"
                    "addi               t2, %[INNER], 0             \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1_v2

                    "BLOCK_INNER_LOOP%=:                            \n\t"

                    LOAD_B_16x8x2

                    "vle8.v             v10, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vle8.v             v11, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vsub.vv            v2, v2, v12                 \n\t"
                    "vsub.vv            v6, v6, v12                 \n\t"
                    "vsub.vv            v3, v3, v13                 \n\t"
                    "vsub.vv            v7, v7, v13                 \n\t"
                    "vsub.vv            v4, v4, v14                 \n\t"
                    "vsub.vv            v8, v8, v14                 \n\t"
                    "vsub.vv            v5, v5, v15                 \n\t"
                    "vsub.vv            v9, v9, v15                 \n\t"

                    SQ4BIT_KERNEL_COMP_4x16x16

                    "addi               t2, t2, -1                  \n\t"
                    "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                    LOAD_SCALE_4x16

                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vfcvt.f.x.v        v16, v16                    \n\t"
                    "vfmacc.vv          v24, v16, v8                \n\t"
                    "addi               t3, t3, -1                  \n\t"
                    "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                    "RESULT_SAVE%=:                                 \n\t"

                    SAVE_RESULT_4x16

                    :
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                      [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                    : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3",
                      "s4", "s5", "s6");
            }
        }
    } else {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +        //
                                        n * BlockCountK * BlkLen / 2 +    // b data
                                        n * BlockCountK * sizeof(float);  // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                if (NBLKS < 16) {
                    __asm__ volatile(
                        "vsetvli        t0, %[N], e32, m2     \n\t"
                        "vle32.v        v0, (%[SRC])          \n\t"
                        "vse32.v        v0, (%[DST])          \n\t"
                        :
                        : [SRC] "r"(bias), [DST] "r"(tmp), [N] "r"(NBLKS)
                        : "cc", "t0");
                    bias = tmp;
                }
                __asm__ volatile(LOAD_BIAS
                                 "addi               t3, %[BlockCountK], 0       \n\t"
                                 "addi               a1, %[A], 0                 \n\t"
                                 "addi               s1, %[B], 0                 \n\t"
                                 "BLOCK_COUNTK_LOOP%=:                           \n\t"
                                 "addi               s5, s1, 0                   \n\t"
                                 "addi               s1, s5, 64                  \n\t"
                                 "addi               s2, s1, 32                  \n\t"
                                 "addi               s3, s1, 32*2                \n\t"
                                 "addi               s4, s1, 32*3                \n\t"
                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vxor.vv            v16, v16, v16               \n\t"
                                 // load a scale
                                 "flw                f1, (a1)                    \n\t"
                                 "flw                f2, 4(a1)                   \n\t"
                                 "flw                f3, 8(a1)                   \n\t"
                                 "flw                f4, 12(a1)                  \n\t"
                                 "addi               a1, a1, 16                  \n\t"
                                 "addi               t2, %[INNER], 0             \n\t"
                                 "BLOCK_INNER_LOOP%=:                            \n\t"

                                 LOAD_B_16x8x2

                                 "vsetvli            t0, zero, e8, m1            \n\t"
                                 "vle8.v             v10, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vle8.v             v11, (a1)                   \n\t"
                                 "addi               a1, a1, 32                  \n\t"
                                 "vadd.vi            v2, v2, -8                  \n\t"
                                 "vadd.vi            v3, v3, -8                  \n\t"
                                 "vadd.vi            v4, v4, -8                  \n\t"
                                 "vadd.vi            v5, v5, -8                  \n\t"
                                 "vadd.vi            v6, v6, -8                  \n\t"
                                 "vadd.vi            v7, v7, -8                  \n\t"
                                 "vadd.vi            v8, v8, -8                  \n\t"
                                 "vadd.vi            v9, v9, -8                  \n\t"

                                 SQ4BIT_KERNEL_COMP_4x16x16

                                 "addi               t2, t2, -1                  \n\t"
                                 "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                                 LOAD_SCALE_4x16

                                 "vsetvli            t0, zero, e32, m8           \n\t"
                                 "vfcvt.f.x.v        v16, v16                    \n\t"
                                 "vfmacc.vv          v24, v16, v8                \n\t"
                                 "addi               t3, t3, -1                  \n\t"
                                 "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                                 "RESULT_SAVE%=:                                 \n\t"

                                 SAVE_RESULT_4x16

                                 :
                                 : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                                   [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr), [BIAS] "r"(bias)
                                 : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1",
                                   "s2", "s3", "s4", "s5", "s6");

            } else {
                __asm__ volatile(
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v24, v24, v24               \n\t"
                    "addi               t3, %[BlockCountK], 0       \n\t"
                    "addi               a1, %[A], 0                 \n\t"
                    "addi               s1, %[B], 0                 \n\t"
                    "BLOCK_COUNTK_LOOP%=:                           \n\t"
                    "addi               s5, s1, 0                   \n\t"
                    "addi               s1, s5, 64                  \n\t"
                    "addi               s2, s1, 32                  \n\t"
                    "addi               s3, s1, 32*2                \n\t"
                    "addi               s4, s1, 32*3                \n\t"
                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vxor.vv            v16, v16, v16               \n\t"
                    // load a scale
                    "flw                f1, (a1)                    \n\t"
                    "flw                f2, 4(a1)                   \n\t"
                    "flw                f3, 8(a1)                   \n\t"
                    "flw                f4, 12(a1)                  \n\t"
                    "addi               a1, a1, 16                  \n\t"
                    "addi               t2, %[INNER], 0             \n\t"
                    "BLOCK_INNER_LOOP%=:                            \n\t"

                    LOAD_B_16x8x2

                    "vsetvli            t0, zero, e8, m1            \n\t"
                    "vle8.v             v10, (a1)                   \n\t"

                    "addi               a1, a1, 32                  \n\t"
                    "vle8.v             v11, (a1)                   \n\t"
                    "addi               a1, a1, 32                  \n\t"
                    "vadd.vi            v2, v2, -8                  \n\t"
                    "vadd.vi            v3, v3, -8                  \n\t"
                    "vadd.vi            v4, v4, -8                  \n\t"
                    "vadd.vi            v5, v5, -8                  \n\t"
                    "vadd.vi            v6, v6, -8                  \n\t"
                    "vadd.vi            v7, v7, -8                  \n\t"
                    "vadd.vi            v8, v8, -8                  \n\t"
                    "vadd.vi            v9, v9, -8                  \n\t"

                    SQ4BIT_KERNEL_COMP_4x16x16

                    "addi               t2, t2, -1                  \n\t"
                    "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                    LOAD_SCALE_4x16

                    "vsetvli            t0, zero, e32, m8           \n\t"
                    "vfcvt.f.x.v        v16, v16                    \n\t"
                    "vfmacc.vv          v24, v16, v8                \n\t"
                    "addi               t3, t3, -1                  \n\t"
                    "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                    "RESULT_SAVE%=:                                 \n\t"

                    SAVE_RESULT_4x16

                    :
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                      [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                    : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3",
                      "s4", "s5", "s6");
            }
        }
    }
    if (CountN % 16 != 0) {
        // stroe output from tmp to C when NBLKS less than 16.
        float *      CPtr = C + CountN / 16 * 16;
        const size_t N    = CountN % 16;
        LDC               = ldc * sizeof(float);
        __asm__ volatile(
            "vsetvli            t0, %[N], e32, m2       \n\t"
            "vle32.v            v0, (%[SRC])            \n\t"
            "addi               s2, %[SRC], 64          \n\t"
            "addi               s3, %[SRC], 64*2        \n\t"
            "addi               s4, %[SRC], 64*3        \n\t"
            "vle32.v            v2, (s2)                \n\t"
            "vle32.v            v4, (s3)                \n\t"
            "vle32.v            v6, (s4)                \n\t"
            "add                t2, %[DST], %[LDC]      \n\t"
            "add                t3, t2, %[LDC]          \n\t"
            "add                t4, t3, %[LDC]          \n\t"
            "vse32.v            v0, (%[DST])            \n\t"
            "vse32.v            v2, (t2)                \n\t"
            "vse32.v            v4, (t3)                \n\t"
            "vse32.v            v6, (t4)                \n\t"
            :
            : [N] "r"(N), [SRC] "r"(tmp), [DST] "r"(CPtr), [LDC] "r"(LDC)
            : "cc", "t0", "t2", "t3", "t4", "s2", "s3", "s4");
    }
}

template <bool HasZeroPoint>
void SQ4BitGemmM1Kernel_CompInt8_ScaleFp16_Impl(size_t            BlkLen,
                                                const std::byte * QuantA,
                                                const std::byte * QuantBData,
                                                const float *     QuantBScale,
                                                const std::byte * QuantBZeroPoint,
                                                float *           C,
                                                size_t            CountN,
                                                size_t            BlockCountK,
                                                const float *     Bias) {
    GGML_UNUSED(QuantBScale);
    GGML_UNUSED(QuantBZeroPoint);
    size_t INNER = BlkLen / 16;

    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +           //
                                        n * BlockCountK * BlkLen / 2 +       // b data
                                        n * BlockCountK * sizeof(uint8_t) +  // zp
                                        n * BlockCountK * sizeof(_Float16);    // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                __asm__ volatile(
                    "addi         t3, %[NBLKS], 0         \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"

                    "vmv.v.i      v13, 3                  \n\t"
                    "li           s1, 24                  \n\t"
                    "vsetvli      t0, s1, e8, m1          \n\t"
                    "vmv.v.i      v13, 2                  \n\t"
                    "vsetvli      t0, zero, e8, mf2       \n\t"
                    "vmv.v.i      v13, 1                  \n\t"
                    "vsetvli      t0, zero, e8, mf4       \n\t"
                    "vmv.v.i      v13, 0                  \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 8             \n\t"
                    "addi         s3, %[B], 16            \n\t"
                    "addi         s4, %[B], 24            \n\t"
                    // zp offset
                    "addi         s7, %[B], 32            \n\t"
                    // a offset
                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"

                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v28, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v29, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v30, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v31, (%[BIAS])          \n\t"

                    "LOOP_K%=:                            \n\t"
                    "vsetvli      t0, zero, e16, mf4      \n\t"

                    "vle16.v      v4, (s1)                \n\t"
                    "addi         s1, s1, 48              \n\t"
                    "vle16.v      v5, (s2)                \n\t"
                    "addi         s2, s2, 72              \n\t"
                    "vle16.v      v6, (s3)                \n\t"
                    "addi         s3, s3, 96              \n\t"
                    "vle16.v      v7, (s4)                \n\t"
                    "addi         s4, s4, 120             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"
                    "vfwcvt.f.f.v v8, v4                  \n\t"
                    "vfwcvt.f.f.v v9, v5                  \n\t"
                    "vfwcvt.f.f.v v10, v6                 \n\t"
                    "vfwcvt.f.f.v v11, v7                 \n\t"

                    "vsetvli      t0, zero, e32, mf2      \n\t"
                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1

                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vsub.vv      v0, v0, v8              \n\t"
                    "vsub.vv      v4, v4, v8              \n\t"
                    "vsub.vv      v1, v1, v9              \n\t"
                    "vsub.vv      v5, v5, v9              \n\t"
                    "vsub.vv      v2, v2, v10             \n\t"
                    "vsub.vv      v6, v6, v10             \n\t"
                    "vsub.vv      v3, v3, v11             \n\t"
                    "vsub.vv      v7, v7, v11             \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_F16_1X4X4
                    "addi         s7, s1, 32              \n\t"

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks), [BIAS] "+r"(bias)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
            } else {
                __asm__ volatile(
                    "vsetvli      t0, zero, e32, m4       \n\t"
                    "vxor.vv      v28, v28, v28           \n\t"

                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "vmv.v.i      v13, 3                  \n\t"
                    "li           s1, 24                  \n\t"
                    "vsetvli      t0, s1, e8, m1          \n\t"
                    "vmv.v.i      v13, 2                  \n\t"
                    "vsetvli      t0, zero, e8, mf2       \n\t"
                    "vmv.v.i      v13, 1                  \n\t"
                    "vsetvli      t0, zero, e8, mf4       \n\t"
                    "vmv.v.i      v13, 0                  \n\t"

                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 8             \n\t"
                    "addi         s3, %[B], 16            \n\t"
                    "addi         s4, %[B], 24            \n\t"

                    "addi         s7, %[B], 32            \n\t"

                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "LOOP_K%=:                            \n\t"
                    "vsetvli      t0, zero, e16, mf4      \n\t"
                    "vle16.v      v4, (s1)                \n\t"
                    "addi         s1, s1, 48              \n\t"
                    "vle16.v      v5, (s2)                \n\t"
                    "addi         s2, s2, 72              \n\t"
                    "vle16.v      v6, (s3)                \n\t"
                    "addi         s3, s3, 96              \n\t"
                    "vle16.v      v7, (s4)                \n\t"
                    "addi         s4, s4, 120             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "vfwcvt.f.f.v v8, v4                  \n\t"
                    "vfwcvt.f.f.v v9, v5                  \n\t"
                    "vfwcvt.f.f.v v10, v6                 \n\t"
                    "vfwcvt.f.f.v v11, v7                 \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1

                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vsub.vv      v0, v0, v8              \n\t"
                    "vsub.vv      v4, v4, v8              \n\t"
                    "vsub.vv      v1, v1, v9              \n\t"
                    "vsub.vv      v5, v5, v9              \n\t"
                    "vsub.vv      v2, v2, v10             \n\t"
                    "vsub.vv      v6, v6, v10             \n\t"
                    "vsub.vv      v3, v3, v11             \n\t"
                    "vsub.vv      v7, v7, v11             \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_F16_1X4X4
                    "addi         s7, s1, 32              \n\t"

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
            }
        }
    } else {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +         //
                                        n * BlockCountK * BlkLen / 2 +     // b data
                                        n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                __asm__ volatile(
                    "addi         t3, %[NBLKS], 0         \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 8             \n\t"
                    "addi         s3, %[B], 16            \n\t"
                    "addi         s4, %[B], 24            \n\t"
                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v28, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v29, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v30, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v31, (%[BIAS])          \n\t"

                    "LOOP_K%=:                            \n\t"
                    "vsetvli      t0, zero, e16, mf4      \n\t"

                    "vle16.v      v4, (s1)                \n\t"
                    "addi         s1, s1, 32              \n\t"
                    "vle16.v      v5, (s2)                \n\t"
                    "addi         s2, s2, 56              \n\t"
                    "vle16.v      v6, (s3)                \n\t"
                    "addi         s3, s3, 80              \n\t"
                    "vle16.v      v7, (s4)                \n\t"
                    "addi         s4, s4, 104             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"
                    "vfwcvt.f.f.v v8, v4                  \n\t"
                    "vfwcvt.f.f.v v9, v5                  \n\t"
                    "vfwcvt.f.f.v v10, v6                 \n\t"
                    "vfwcvt.f.f.v v11, v7                 \n\t"

                    "vsetvli      t0, zero, e32, mf2      \n\t"
                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vadd.vi      v0, v0, -8              \n\t"
                    "vadd.vi      v1, v1, -8              \n\t"
                    "vadd.vi      v2, v2, -8              \n\t"
                    "vadd.vi      v3, v3, -8              \n\t"
                    "vadd.vi      v4, v4, -8              \n\t"
                    "vadd.vi      v5, v5, -8              \n\t"
                    "vadd.vi      v6, v6, -8              \n\t"
                    "vadd.vi      v7, v7, -8              \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_F16_1X4X4

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks), [BIAS] "+r"(bias)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6");
            } else {
                __asm__ volatile(
                    "vsetvli      t0, zero, e32, m4       \n\t"
                    "vxor.vv      v28, v28, v28           \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 8             \n\t"
                    "addi         s3, %[B], 16            \n\t"
                    "addi         s4, %[B], 24            \n\t"

                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "LOOP_K%=:                            \n\t"
                    "vsetvli      t0, zero, e16, mf4      \n\t"
                    "vle16.v      v4, (s1)                \n\t"
                    "addi         s1, s1, 32              \n\t"
                    "vle16.v      v5, (s2)                \n\t"
                    "addi         s2, s2, 56              \n\t"
                    "vle16.v      v6, (s3)                \n\t"
                    "addi         s3, s3, 80              \n\t"
                    "vle16.v      v7, (s4)                \n\t"
                    "addi         s4, s4, 104             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "vfwcvt.f.f.v v8, v4                  \n\t"
                    "vfwcvt.f.f.v v9, v5                  \n\t"
                    "vfwcvt.f.f.v v10, v6                 \n\t"
                    "vfwcvt.f.f.v v11, v7                 \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vadd.vi      v0, v0, -8              \n\t"
                    "vadd.vi      v1, v1, -8              \n\t"
                    "vadd.vi      v2, v2, -8              \n\t"
                    "vadd.vi      v3, v3, -8              \n\t"
                    "vadd.vi      v4, v4, -8              \n\t"
                    "vadd.vi      v5, v5, -8              \n\t"
                    "vadd.vi      v6, v6, -8              \n\t"
                    "vadd.vi      v7, v7, -8              \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_F16_1X4X4

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6");
            }
        }
    }
}

template <bool HasZeroPoint>
void SQ4BitGemmM1Kernel_CompInt8_Impl(size_t            BlkLen,
                                      const std::byte * QuantA,
                                      const std::byte * QuantBData,
                                      const float *     QuantBScale,
                                      const std::byte * QuantBZeroPoint,
                                      float *           C,
                                      size_t            CountN,
                                      size_t            BlockCountK,
                                      const float *     Bias) {
    GGML_UNUSED(QuantBScale);
    GGML_UNUSED(QuantBZeroPoint);
    const size_t INNER = BlkLen / 16;
    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +           //
                                        n * BlockCountK * BlkLen / 2 +       // b data
                                        n * BlockCountK * sizeof(uint8_t) +  // zp
                                        n * BlockCountK * sizeof(float);     // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                __asm__ volatile(
                    "addi         t3, %[NBLKS], 0         \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "vmv.v.i      v13, 3                  \n\t"
                    "li           s1, 24                  \n\t"
                    "vsetvli      t0, s1, e8, m1          \n\t"
                    "vmv.v.i      v13, 2                  \n\t"
                    "vsetvli      t0, zero, e8, mf2       \n\t"
                    "vmv.v.i      v13, 1                  \n\t"
                    "vsetvli      t0, zero, e8, mf4       \n\t"
                    "vmv.v.i      v13, 0                  \n\t"
                    "vsetvli      t0, zero, e32, m4       \n\t"
                    "vxor.vv      v28, v28, v28           \n\t"

                    // scale offset, scale0.0, scale1.0, scale2.0, scale3.0....scale15.0
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 16            \n\t"
                    "addi         s3, %[B], 32            \n\t"
                    "addi         s4, %[B], 48            \n\t"
                    // zp offset
                    "addi         s7, %[B], 64            \n\t"
                    // a offset
                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"

                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v28, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v29, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v30, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v31, (%[BIAS])          \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"
                    "LOOP_K%=:                            \n\t"

                    // load scale
                    "vle32.v      v8, (s1)                \n\t"
                    "addi         s1, s1, 80              \n\t"
                    "vle32.v      v9, (s2)                \n\t"
                    "addi         s2, s2, 96              \n\t"
                    "vle32.v      v10, (s3)               \n\t"
                    "addi         s3, s3, 112             \n\t"
                    "vle32.v      v11, (s4)               \n\t"
                    "addi         s4, s4, 128             \n\t"

                    // load a scale
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"

                    // a scale * b scale
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1

                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vsub.vv      v0, v0, v8              \n\t"
                    "vsub.vv      v4, v4, v8              \n\t"
                    "vsub.vv      v1, v1, v9              \n\t"
                    "vsub.vv      v5, v5, v9              \n\t"
                    "vsub.vv      v2, v2, v10             \n\t"
                    "vsub.vv      v6, v6, v10             \n\t"
                    "vsub.vv      v3, v3, v11             \n\t"
                    "vsub.vv      v7, v7, v11             \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_1X4X4
                    "addi         s7, s1, 64              \n\t"

                    "bnez         %[CNT], LOOP_K%=        \n\t"

                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks), [BIAS] "+r"(bias)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
            } else {
                __asm__ volatile(
                    "vsetvli      t0, zero, e32, m4       \n\t"
                    "vxor.vv      v28, v28, v28           \n\t"

                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "vmv.v.i      v13, 3                  \n\t"
                    "li           s1, 24                  \n\t"
                    "vsetvli      t0, s1, e8, m1          \n\t"
                    "vmv.v.i      v13, 2                  \n\t"
                    "vsetvli      t0, zero, e8, mf2       \n\t"
                    "vmv.v.i      v13, 1                  \n\t"
                    "vsetvli      t0, zero, e8, mf4       \n\t"
                    "vmv.v.i      v13, 0                  \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 16            \n\t"
                    "addi         s3, %[B], 32            \n\t"
                    "addi         s4, %[B], 48            \n\t"

                    "addi         s7, %[B], 64            \n\t"

                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    "LOOP_K%=:                            \n\t"
                    "vle32.v      v8, (s1)                \n\t"
                    "addi         s1, s1, 80              \n\t"
                    "vle32.v      v9, (s2)                \n\t"
                    "addi         s2, s2, 96              \n\t"
                    "vle32.v      v10, (s3)               \n\t"
                    "addi         s3, s3, 112             \n\t"
                    "vle32.v      v11, (s4)               \n\t"
                    "addi         s4, s4, 128             \n\t"

                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"

                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"

                    SQ4BIT_KERNEL_LOAD_ZP_16X1

                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vsub.vv      v0, v0, v8              \n\t"
                    "vsub.vv      v4, v4, v8              \n\t"
                    "vsub.vv      v1, v1, v9              \n\t"
                    "vsub.vv      v5, v5, v9              \n\t"
                    "vsub.vv      v2, v2, v10             \n\t"
                    "vsub.vv      v6, v6, v10             \n\t"
                    "vsub.vv      v3, v3, v11             \n\t"
                    "vsub.vv      v7, v7, v11             \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_1X4X4
                    "addi         s7, s1, 64              \n\t"

                    "bnez         %[CNT], LOOP_K%=        \n\t"

                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
            }
        }
    } else {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t      nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            std::byte * QuantBDataPtr = (std::byte *) QuantBData +        //
                                        n * BlockCountK * BlkLen / 2 +    // b data
                                        n * BlockCountK * sizeof(float);  // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;
            if (Bias != nullptr) {
                const float * bias = Bias + n;
                __asm__ volatile(
                    "addi         t3, %[NBLKS], 0         \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 16            \n\t"
                    "addi         s3, %[B], 32            \n\t"
                    "addi         s4, %[B], 48            \n\t"
                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v28, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v29, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v30, (%[BIAS])          \n\t"
                    "sub          t3, t3, t0              \n\t"
                    "addi         %[BIAS], %[BIAS], 16    \n\t"
                    "vsetvli      t0, t3, e32, mf2        \n\t"
                    "vle32.v      v31, (%[BIAS])          \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"
                    "LOOP_K%=:                            \n\t"
                    "vle32.v      v8, (s1)                \n\t"
                    "addi         s1, s1, 64              \n\t"
                    "vle32.v      v9, (s2)                \n\t"
                    "addi         s2, s2, 80              \n\t"
                    "vle32.v      v10, (s3)               \n\t"
                    "addi         s3, s3, 96              \n\t"
                    "vle32.v      v11, (s4)               \n\t"
                    "addi         s4, s4, 112             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vadd.vi      v0, v0, -8              \n\t"
                    "vadd.vi      v1, v1, -8              \n\t"
                    "vadd.vi      v2, v2, -8              \n\t"
                    "vadd.vi      v3, v3, -8              \n\t"
                    "vadd.vi      v4, v4, -8              \n\t"
                    "vadd.vi      v5, v5, -8              \n\t"
                    "vadd.vi      v6, v6, -8              \n\t"
                    "vadd.vi      v7, v7, -8              \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_1X4X4

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks), [BIAS] "+r"(bias)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6");
            } else {
                __asm__ volatile(
                    "vsetvli      t0, zero, e32, m4       \n\t"
                    "vxor.vv      v28, v28, v28           \n\t"
                    "addi         s1, %[B], 0             \n\t"
                    "addi         s2, %[B], 16            \n\t"
                    "addi         s3, %[B], 32            \n\t"
                    "addi         s4, %[B], 48            \n\t"

                    "addi         s5, %[A], 0             \n\t"
                    "addi         s6, %[A], 12            \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"
                    "LOOP_K%=:                            \n\t"
                    "vle32.v      v8, (s1)                \n\t"
                    "addi         s1, s1, 64              \n\t"
                    "vle32.v      v9, (s2)                \n\t"
                    "addi         s2, s2, 80              \n\t"
                    "vle32.v      v10, (s3)               \n\t"
                    "addi         s3, s3, 96              \n\t"
                    "vle32.v      v11, (s4)               \n\t"
                    "addi         s4, s4, 112             \n\t"
                    "flw          f1, (s5)                \n\t"
                    "addi         s5, s5, 4               \n\t"

                    "addi         t5, %[INNER], 0         \n\t"
                    "vxor.vv      v16, v16, v16           \n\t"
                    "vxor.vv      v18, v18, v18           \n\t"
                    "vxor.vv      v20, v20, v20           \n\t"
                    "vxor.vv      v22, v22, v22           \n\t"
                    "vfmul.vf     v24, v8, f1             \n\t"
                    "vfmul.vf     v25, v9, f1             \n\t"
                    "vfmul.vf     v26, v10, f1            \n\t"
                    "vfmul.vf     v27, v11, f1            \n\t"
                    "addi         %[CNT], %[CNT], -1      \n\t"
                    "vsetvli      t0, zero, e8, m1        \n\t"
                    "LOOP_INNER%=:                        \n\t"

                    SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                    "vadd.vi      v0, v0, -8              \n\t"
                    "vadd.vi      v1, v1, -8              \n\t"
                    "vadd.vi      v2, v2, -8              \n\t"
                    "vadd.vi      v3, v3, -8              \n\t"
                    "vadd.vi      v4, v4, -8              \n\t"
                    "vadd.vi      v5, v5, -8              \n\t"
                    "vadd.vi      v6, v6, -8              \n\t"
                    "vadd.vi      v7, v7, -8              \n\t"

                    SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                    "bnez         t5, LOOP_INNER%=        \n\t"
                    "vsetvli      t0, zero, e32, mf2      \n\t"

                    SQ4BIT_KERNEL_ACC_1X4X4

                    "bnez         %[CNT], LOOP_K%=        \n\t"
                    "addi         t3, zero, 16            \n\t"
                    "addi         s1, %[C], 16            \n\t"
                    "addi         s2, %[C], 32            \n\t"
                    "addi         s3, %[C], 48            \n\t"
                    "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "jal          x0, END%=               \n\t"

                    "ST_TAIL%=:                           \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v28, (%[C])             \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v29, (s1)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v30, (s2)               \n\t"
                    "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                    "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                    "vse32.v      v31, (s3)               \n\t"
                    "END%=:                               \n\t"

                    : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                    : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
                    : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6");
            }
        }
    }
}

template <bool HasZeroPoint>
inline void SQ4BitGemmM4Kernel_CompInt8_DispatchOnBlkLen(size_t            BlkLen,
                                                         const std::byte * QuantA,
                                                         const std::byte * QuantBData,
                                                         const float *     QuantBScale,
                                                         const std::byte * QuantBZeroPoint,
                                                         float *           C,
                                                         size_t            CountM,
                                                         size_t            CountN,
                                                         size_t            BlockStrideQuantB,
                                                         const float *     Bias,
                                                         const size_t      ldc,
                                                         const size_t      scalestride) {
    if (scalestride == 4) {
        SQ4BitGemmM4Kernel_CompInt8_Impl<HasZeroPoint>(BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
                                                       CountN, BlockStrideQuantB, Bias, ldc);

    } else if (scalestride == 2) {
        SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl<HasZeroPoint>(
            BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, BlockStrideQuantB, Bias, ldc);
    }
}

template <bool HasZeroPoint>
inline void SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen(size_t            BlkLen,
                                                         const std::byte * QuantA,
                                                         const std::byte * QuantBData,
                                                         const float *     QuantBScale,
                                                         const std::byte * QuantBZeroPoint,
                                                         float *           C,
                                                         size_t            CountM,
                                                         size_t            CountN,
                                                         size_t            BlockStrideQuantB,
                                                         const float *     Bias,
                                                         const size_t      ldc,
                                                         const size_t      scalestride) {
    if (scalestride == 4) {
        SQ4BitGemmM1Kernel_CompInt8_Impl<HasZeroPoint>(BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
                                                       CountN, BlockStrideQuantB, Bias);
    } else if (scalestride == 2) {
        SQ4BitGemmM1Kernel_CompInt8_ScaleFp16_Impl<HasZeroPoint>(BlkLen, QuantA, QuantBData, QuantBScale,
                                                                 QuantBZeroPoint, C, CountN, BlockStrideQuantB, Bias);
    }
}

}  // namespace

namespace ime1 {
size_t gemm_kernel_i8i4(size_t            BlkLen,
                        const std::byte * QuantA,
                        const std::byte * QuantBData,
                        const float *     QuantBScale,
                        const std::byte * QuantBZeroPoint,
                        float *           C,
                        size_t            CountM,
                        size_t            CountN,
                        size_t            CountK,
                        size_t            BlockCountK,
                        size_t            ldc,
                        const float *     Bias,
                        const size_t      ScaleStride) {
    GGML_UNUSED(CountM);
    GGML_UNUSED(CountK);
    GGML_UNUSED(ldc);
    if (CountM >= 4) {
        if (QuantBZeroPoint != nullptr) {
            SQ4BitGemmM4Kernel_CompInt8_DispatchOnBlkLen<true>(BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint,
                                                               C, CountM, CountN, BlockCountK, Bias, ldc, ScaleStride);
        } else {
            SQ4BitGemmM4Kernel_CompInt8_DispatchOnBlkLen<false>(BlkLen, QuantA, QuantBData, QuantBScale,
                                                                QuantBZeroPoint, C, CountM, CountN, BlockCountK, Bias,
                                                                ldc, ScaleStride);
        }
        return 4;
    } else {
        if (QuantBZeroPoint != nullptr) {
            SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<true>(BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint,
                                                               C, CountM, CountN, BlockCountK, Bias, ldc, ScaleStride);
        } else {
            SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<false>(BlkLen, QuantA, QuantBData, QuantBScale,
                                                                QuantBZeroPoint, C, CountM, CountN, BlockCountK, Bias,
                                                                ldc, ScaleStride);
        }
        return 1;
    }
}
}  // namespace ime1
}  // namespace sqnbitgemm_spacemit_ime
