#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

static inline size_t ggml_ifairy_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_ifairy_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst) {
    const bool   dbg         = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    const char * enabled_env = getenv("GGML_IFAIRY_LUT");
    if (enabled_env && strcmp(enabled_env, "0") == 0) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: disabled by env GGML_IFAIRY_LUT=0\n");
        }
        return false;
    }

#if !defined(__ARM_NEON) || !defined(__aarch64__)
    if (dbg) {
        GGML_LOG_WARN("ifairy_lut: disabled (requires __aarch64__ + __ARM_NEON)\n");
    }
    return false;
#endif

    if (src0->type != GGML_TYPE_IFAIRY || (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_IFAIRY_Q16)) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: type mismatch src0=%s src1=%s dst=%s\n", ggml_type_name(src0->type),
                          ggml_type_name(src1->type), ggml_type_name(dst->type));
        }
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: dst type not F32 (%s)\n", ggml_type_name(dst->type));
        }
        return false;
    }
    // require logical K aligned to block
    if (src0->ne[0] % QK_K != 0 || src1->ne[0] != src0->ne[0]) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: K misaligned K0=%lld K1=%lld QK_K=%d\n", (long long) src0->ne[0],
                          (long long) src1->ne[0], QK_K);
        }
        return false;
    }
    if (dbg) {
        GGML_LOG_INFO("ifairy_lut: can_mul_mat=true\n");
    }
    return true;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst,
                                 int                        n_threads) {
    if (!ggml_ifairy_lut_can_mul_mat(src0, src1, dst)) {
        return 0;
    }
    (void) n_threads;

    const int64_t K              = src0->ne[0];
    const int64_t N              = src1->ne[1];
    const int64_t blocks_per_col = K / QK_K;
    const int64_t groups         = blocks_per_col * ((QK_K + 2) / 3);

    size_t quant_bytes = 0;
    if (src1->type == GGML_TYPE_F32) {
        const size_t q_elems = ggml_ifairy_checked_mul_size((size_t) N, (size_t) blocks_per_col);
        quant_bytes          = GGML_PAD(ggml_ifairy_checked_mul_size(q_elems, sizeof(block_ifairy_q16)), 64);
    }

    const size_t lut_groups  = ggml_ifairy_checked_mul_size((size_t) N, (size_t) groups);
    const size_t lut_bytes   = ggml_ifairy_checked_mul_size(lut_groups, (size_t) k_ifairy_lut_merged64_group_bytes);
    const size_t scale_bytes = ggml_ifairy_checked_mul_size(
        ggml_ifairy_checked_mul_size(ggml_ifairy_checked_mul_size((size_t) N, (size_t) blocks_per_col), 2u),
        sizeof(float));
    const size_t shared_bytes = GGML_PAD(ggml_ifairy_checked_add_size(lut_bytes, scale_bytes), 64);

    return ggml_ifairy_checked_add_size(quant_bytes, shared_bytes);
}
