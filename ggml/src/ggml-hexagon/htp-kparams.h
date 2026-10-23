#ifndef HTP_KPARAMS_H
#define HTP_KPARAMS_H

#include "htp-common.h"

#include "htp/allreduce-ops.h"
#include "htp/argsort-ops.h"
#include "htp/binary-ops.h"
#include "htp/concat-ops.h"
#include "htp/cpy-ops.h"
#include "htp/cumsum-ops.h"
#include "htp/diag-ops.h"
#include "htp/fill-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/glu-ops.h"
#include "htp/im2col-ops.h"
#include "htp/matmul-ops.h"
#include "htp/pad-ops.h"
#include "htp/pool-ops.h"
#include "htp/repeat-ops.h"
#include "htp/roll-ops.h"
#include "htp/rope-ops.h"
#include "htp/set-rows-ops.h"
#include "htp/softmax-ops.h"
#include "htp/solve-tri-ops.h"
#include "htp/ssm-conv.h"
#include "htp/sum-rows-ops.h"
#include "htp/unary-ops.h"

#include <type_traits>
#include <utility>

bool        htp_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride);
bool        htp_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
ggml_tensor htp_tensor_collapse_rows(const struct ggml_tensor * t);

template <typename... Args>
auto htp_kparams_type(bool (*)(Args...)) -> std::remove_pointer_t<std::remove_reference_t<decltype((std::declval<Args>(), ...))>>;

template <auto SolveFn, typename... Args>
static inline bool htp_kparams_check(Args &&... args) {
    decltype(htp_kparams_type(SolveFn)) kparams;
    return SolveFn(std::forward<Args>(args)..., &kparams);
}

bool htp_kparams_solve(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    void * kparams = nullptr
);

static inline bool htp_kparams_check(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op
) {
    return htp_kparams_solve(sess, op, nullptr);
}

bool htp_kparams_solve_allreduce(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    struct htp_allreduce_kernel_params * kparams
);

bool htp_kparams_solve_rms_norm_mul(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
);

bool htp_kparams_solve_matmul_add(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

bool htp_kparams_solve_mmnx(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

bool htp_kparams_solve_mmidnx(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

bool htp_kparams_solve_repeat_get_rows(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
);

bool htp_kparams_solve_unary_mul_glu(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_glu_kernel_params * kparams
);

#endif // HTP_KPARAMS_H
