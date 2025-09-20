// ggml/src/ggml-sycl/set.cpp
//
// SYCL backend for GGML SET operator.
//
// Semantics:
//   1) dst <- src0
//   2) copy a sub-block from src1 into dst at byte `offset`,
//      using destination byte-strides (nb1, nb2, nb3) for dims 1..3.
//
// Notes:
//   - (nb1, nb2, nb3, offset) are BYTES (CPU-compatible).
//   - Uses two fast paths (bulk memcpy; row-wise memcpy) and a generic 4D kernel.
//   - Work-group size is configured in presets (SYCL_SET_BLOCK_SIZE).
//
//   Implementation style aligned with other SYCL operators:
//   - No host std::memcpy fallback; no USM detection.
//   - Copies use queue->memcpy; generic case uses a parallel_for kernel.

#include "presets.hpp"   // SYCL_* tuning (incl. SYCL_SET_BLOCK_SIZE)
#include "common.hpp"
#include "ggml.h"
#include "set.hpp"

#include <cstdint>
#include <cstring>

// ---------------- helpers (file-local) ----------------

// Byte-accurate 4D copy with independent src/dst byte strides.
// One work-item copies exactly one element (ts bytes).
static inline void launch_copy_4d_bytes(
    dpct::queue_ptr q,
    const void *p_src, void *p_dst,
    const int64_t ne[4],
    const size_t  sb[4],
    const size_t  db[4],
    const size_t  ts
) {
    const size_t N  = (size_t)(ne[0] * ne[1] * ne[2] * ne[3]);
    if (N == 0) return;

    const size_t WG = (size_t)SYCL_SET_BLOCK_SIZE;
    const size_t NG = ((N + WG - 1) / WG) * WG;

    const size_t ge0 = (size_t) ne[0];
    const size_t ge1 = ge0 * (size_t) ne[1];
    const size_t ge2 = ge1 * (size_t) ne[2];

    q->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(NG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= N) return;

            // 4D indexing
            size_t i3 = idx / ge2;  size_t r2 = idx % ge2;
            size_t i2 = r2 / ge1;   size_t r1 = r2 % ge1;
            size_t i1 = r1 / ge0;   size_t i0 = r1 % ge0;

            const char *s = (const char *)p_src + (i0*sb[0] + i1*sb[1] + i2*sb[2] + i3*sb[3]);
            char       *d = (char       *)p_dst + (i0*db[0] + i1*db[1] + i2*db[2] + i3*db[3]);

            #pragma unroll
            for (size_t b = 0; b < ts; ++b) {
                d[b] = s[b];
            }
        }
    );
}

// --------------------------- operator ---------------------------

void ggml_sycl_op_set(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst != nullptr);
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[1] != nullptr);
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src0 && src1);

    // Type constraints (CPU-compatible)
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src1->type == dst->type);
#if defined(GGML_SYCL_F16)
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_I32);
#else
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32);
#endif

    dpct::queue_ptr q  = ctx.stream();
    const size_t    ts = ggml_type_size(dst->type);

    // Stage 1: dst <- src0
    {
        const bool same_type = (src0->type == dst->type);
        const bool src_cont  = ggml_is_contiguous(src0);
        const bool dst_cont  = ggml_is_contiguous(dst);

        const void *p_src0 = src0->data;
        void       *p_dst  = dst->data;

        if (same_type && src_cont && dst_cont &&
            ggml_nelements(src0) == ggml_nelements(dst)) {
            SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(p_dst, p_src0, ggml_nbytes(dst))));
        } else {
            // generic 4D copy
            const int64_t ne[4] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
            const size_t  sb[4] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
            const size_t  db[4] = { dst ->nb[0], dst ->nb[1], dst ->nb[2], dst ->nb[3] };
            launch_copy_4d_bytes(q, p_src0, p_dst, ne, sb, db, ts);
        }
    }

    // Stage 2: paste src1 sub-block into dst
    {
        // op_params: [ nb1, nb2, nb3, offset ] (BYTES)
        const int32_t *p = (const int32_t *) dst->op_params;
        const size_t nb1    = (size_t) p[0];
        const size_t nb2    = (size_t) p[1];
        const size_t nb3    = (size_t) p[2];
        const size_t offset = (size_t) p[3];

        const void *p_src1 = src1->data;
        void       *p_base = (char *) dst->data + offset;

        const bool src1_cont = ggml_is_contiguous(src1);
        const bool dst_tight = (dst->nb[0] == ts); // tightly-packed rows

        if (src1_cont && dst_tight) {
            // Row-wise device memcpy of src1 into dst at the given offset
            const char *s_base     = (const char *) p_src1;
            char       *d_base     = (char       *) p_base;
            const size_t row_bytes = (size_t) src1->ne[0] * ts;

            const size_t sb1 = src1->nb[1];
            const size_t sb2 = src1->nb[2];
            const size_t sb3 = src1->nb[3];

            for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < src1->ne[1]; ++i1) {
                        const char *s_row = s_base + i1*sb1 + i2*sb2 + i3*sb3;
                        char       *d_row = d_base + i1*nb1 + i2*nb2 + i3*nb3;
                        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(d_row, s_row, row_bytes)));
                    }
                }
            }
        } else {
            // Generic 4D copy from src1 into (offsetted) dst base
            const int64_t ne[4] = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] };
            const size_t  sb[4] = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] };
            const size_t  db[4] = { dst->nb[0],  nb1,         nb2,         nb3         };
            launch_copy_4d_bytes(q, p_src1, p_base, ne, sb, db, ts);
        }
    }
}
