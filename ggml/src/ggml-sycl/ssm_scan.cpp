#include "ssm_scan.hpp"
#include "common.hpp"

#include <sycl/sycl.hpp>

using namespace sycl;

// ---------------------------------------------------------------------------
// Mamba-2 kernel
//
// Ported from ssm_scan_f32_group in ggml-cuda/ssm-scan.cu.
//
// Template params:
//   D_STATE : d_state (128 or 256)
//
// Each work-group covers D_STATE threads and processes one
// (seq, warp_idx) tile, where warp_idx maps to (head_idx, head_dim_offset).
// Within the work-group, threads are split into sub-groups of WARP_SIZE.
// Each sub-group (warp) owns one warp_idx.
// Each thread within the sub-group holds c_factor = D_STATE/WARP_SIZE state
// elements, indexed as WARP_SIZE*j + lane.
// The dot product C·state is reduced across the sub-group with
// warp_reduce_sum.
// ---------------------------------------------------------------------------
template <int D_STATE>
static void kernel_ssm_scan_f32_group(
        nd_item<2>   item,
        const float * __restrict__ src0,   // s  [d_state, d_head, n_head, n_seq]
        const float * __restrict__ src1,   // x  [d_head,  n_head, n_tok,  n_seq]
        const float * __restrict__ src2,   // dt [n_head,  n_tok,  n_seq]
        const float * __restrict__ src3,   // A  [1,       n_head]  (Mamba-2: scalar per head)
        const float * __restrict__ src4,   // B  [d_state, n_group, n_tok,  n_seq]
        const float * __restrict__ src5,   // C  [d_state, n_group, n_tok,  n_seq]
        const int32_t * __restrict__ src6, // ids[n_seq]
        float       * __restrict__ dst,
        const int src0_nb2, const int src0_nb3,
        const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2,
        const int src3_nb1,
        const int src4_nb2, const int src4_nb3,
        const int src5_nb2, const int src5_nb3,
        const int64_t s_off,
        const int64_t n_head, const int64_t d_head,
        const int64_t n_group, const int64_t n_tok) {

    constexpr int c_factor = D_STATE / WARP_SIZE;

    const int tid      = item.get_local_id(0);
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    // Number of sub-groups (warps) per work-group = D_STATE / WARP_SIZE = c_factor
    const int warp_idx = item.get_group(0) * c_factor + warp;
    const int seq_idx  = item.get_group(1);

    const int head_idx = warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * (int)sizeof(float);
    const int group_off = (head_idx / (n_head / n_group)) * D_STATE * (int)sizeof(float);

    const float * s0_warp = (const float *) ((const char *) src0
        + src6[seq_idx] * src0_nb3
        + head_idx      * src0_nb2
        + head_off      * D_STATE);

    const float * x_warp  = (const float *) ((const char *) src1
        + seq_idx  * src1_nb3
        + warp_idx * (int)sizeof(float));

    const float * dt_warp = (const float *) ((const char *) src2
        + seq_idx  * src2_nb2
        + head_idx * (int)sizeof(float));

    const float * A_warp  = (const float *) ((const char *) src3
        + head_idx * src3_nb1);

    const float * B_warp  = (const float *) ((const char *) src4
        + seq_idx  * src4_nb3
        + group_off);

    const float * C_warp  = (const float *) ((const char *) src5
        + seq_idx  * src5_nb3
        + group_off);

    float * y_warp = dst
        + seq_idx * n_tok * n_head * d_head
        + warp_idx;

    float * s_warp = (float *) ((char *) dst
        + s_off
        + seq_idx  * src0_nb3
        + head_idx * src0_nb2
        + head_off * D_STATE);

    const int stride_x  = src1_nb2 / (int)sizeof(float);
    const int stride_dt = src2_nb1 / (int)sizeof(float);
    const int stride_B  = src4_nb2 / (int)sizeof(float);
    const int stride_C  = src5_nb2 / (int)sizeof(float);
    const int stride_y  = n_head * d_head;

    // Load initial state into registers
    float state[c_factor];
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        const float dt_raw      = dt_warp[i * stride_dt];
        const float dt_soft_plus = (dt_raw <= 20.0f
                                    ? sycl::log1p(sycl::exp(dt_raw))
                                    : dt_raw);
        const float dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;

        float state_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j]   = state[j] * dA + B_val * x_dt;
            state_sum += state[j] * C_val;
        }

        // Reduce state_sum across the sub-group (warp)
        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }
    }

    // Write updated state back
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

template <int D_STATE>
static void launch_ssm_scan_f32_group(
        queue & q,
        const float * src0, const float * src1, const float * src2,
        const float * src3, const float * src4, const float * src5,
        const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3,
        const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2,
        const int src3_nb1,
        const int src4_nb2, const int src4_nb3,
        const int src5_nb2, const int src5_nb3,
        const int64_t s_off,
        const int64_t n_head, const int64_t d_head,
        const int64_t n_group, const int64_t n_tok, const int64_t n_seq) {

    constexpr int c_factor  = D_STATE / WARP_SIZE;
    constexpr int local_sz  = D_STATE;  // one work-group = D_STATE threads = c_factor sub-groups

    // Each work-group covers c_factor warp_idxs.
    // Total warp_idxs = n_head * d_head.
    const int64_t num_wg_x = (n_head * d_head + c_factor - 1) / c_factor;
    const int64_t num_wg_y = n_seq;

    const range<2> global_range(num_wg_x * local_sz, num_wg_y);
    const range<2> local_range(local_sz, 1);

    q.parallel_for(
        nd_range<2>(global_range, local_range),
        [=](nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            kernel_ssm_scan_f32_group<D_STATE>(
                item,
                src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1,
                src4_nb2, src4_nb3, src5_nb2, src5_nb3,
                s_off, n_head, d_head, n_group, n_tok);
        });
}

// ---------------------------------------------------------------------------
// Mamba-1 kernel
//
// Ported from ssm_scan_f32<splitD=128, N=16, L_template=0> in
// ggml-cuda/ssm-scan.cu (the non-CUB fallback path).
//
// Each work-item handles N=16 state elements for one channel (thread).
// B and C (size N) are loaded cooperatively into local (shared) memory.
// No warp-level reduction is needed.
// ---------------------------------------------------------------------------
static void kernel_ssm_scan_f32_mamba1(
        nd_item<2>   item,
        const float * __restrict__ src0,   // s
        const float * __restrict__ src1,   // x
        const float * __restrict__ src2,   // dt
        const float * __restrict__ src3,   // A
        const float * __restrict__ src4,   // B
        const float * __restrict__ src5,   // C
        const int32_t * __restrict__ src6, // ids
        float       * __restrict__ dst,
        const int src0_nb2, const int src0_nb3,
        const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2,
        const int src3_nb1,
        const int src4_nb2, const int src4_nb3,
        const int src5_nb2, const int src5_nb3,
        const int64_t s_off,
        const int64_t d_inner,    // = n_head (when d_head == 1)
        const int64_t n_tok,
        local_accessor<float, 1> smemB,
        local_accessor<float, 1> smemC) {

    constexpr int N       = 16;
    constexpr int splitD  = 128;

    const int tid     = item.get_local_id(0);
    const int seq_idx = item.get_group(0);
    const int blk_y   = item.get_group(1);  // which block of splitD channels

    const int stride_s0 = src0_nb2 / (int)sizeof(float);
    const int stride_A  = src3_nb1 / (int)sizeof(float);
    const int stride_x  = src1_nb2 / (int)sizeof(float);
    const int stride_dt = src2_nb1 / (int)sizeof(float);
    const int stride_B  = src4_nb2 / (int)sizeof(float);
    const int stride_C  = src5_nb2 / (int)sizeof(float);
    const int stride_y  = d_inner;

    const float * s0_block  = (const float *) ((const char *) src0
        + src6[seq_idx] * src0_nb3
        + blk_y * splitD * src0_nb2);

    const float * x_block   = (const float *) ((const char *) src1
        + seq_idx * src1_nb3
        + blk_y * splitD * (int)sizeof(float));

    const float * dt_block  = (const float *) ((const char *) src2
        + seq_idx * src2_nb2
        + blk_y * splitD * (int)sizeof(float));

    const float * A_block   = (const float *) ((const char *) src3
        + blk_y * splitD * src3_nb1);

    const float * B_block   = (const float *) ((const char *) src4
        + seq_idx * src4_nb3);

    const float * C_block   = (const float *) ((const char *) src5
        + seq_idx * src5_nb3);

    float * y_block = (float *) ((char *) dst
        + seq_idx * d_inner * n_tok * (int)sizeof(float)
        + blk_y * splitD * (int)sizeof(float));

    float * s_block = (float *) ((char *) dst
        + s_off
        + src6[seq_idx] * src0_nb3
        + blk_y * splitD * src0_nb2);

    // Load A and initial state into registers
    float regA[N];
    float regs0[N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        regA[n]  = A_block[tid * stride_A + n];
        regs0[n] = s0_block[tid * stride_s0 + n];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // Cooperative load of B and C into local memory
        if (tid < N) {
            smemB[tid] = B_block[i * stride_B + tid];
            smemC[tid] = C_block[i * stride_C + tid];
        }
        item.barrier(access::fence_space::local_space);

        float dt_soft_plus = dt_block[i * stride_dt + tid];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = sycl::log1p(sycl::exp(dt_soft_plus));
        }
        const float x_dt = x_block[i * stride_x + tid] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++) {
            float st = regs0[n] * sycl::exp(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf    += st * smemC[n];
            regs0[n] = st;
        }
        y_block[i * stride_y + tid] = sumf;
    }

    // Write updated state back
#pragma unroll
    for (int n = 0; n < N; ++n) {
        s_block[tid * stride_s0 + n] = regs0[n];
    }
}

static void launch_ssm_scan_f32_mamba1(
        queue & q,
        const float * src0, const float * src1, const float * src2,
        const float * src3, const float * src4, const float * src5,
        const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3,
        const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2,
        const int src3_nb1,
        const int src4_nb2, const int src4_nb3,
        const int src5_nb2, const int src5_nb3,
        const int64_t s_off,
        const int64_t n_head,   // d_inner when d_head==1
        const int64_t n_tok, const int64_t n_seq) {

    constexpr int N       = 16;
    constexpr int splitD  = 128;

    // grid: (n_seq, n_head/splitD)  local: (splitD, 1)
    const int64_t num_wg_x = n_seq;
    const int64_t num_wg_y = n_head / splitD;

    const range<2> global_range(num_wg_x * splitD, num_wg_y);
    const range<2> local_range(splitD, 1);

    q.submit([&](handler & h) {
        auto smemB = local_accessor<float, 1>(range<1>(N), h);
        auto smemC = local_accessor<float, 1>(range<1>(N), h);

        h.parallel_for(
            nd_range<2>(global_range, local_range),
            [=](nd_item<2> item) {
                kernel_ssm_scan_f32_mamba1(
                    item,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3,
                    src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3,
                    s_off, n_head, n_tok,
                    smemB, smemC);
            });
    });
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // s
    const ggml_tensor * src1 = dst->src[1];  // x
    const ggml_tensor * src2 = dst->src[2];  // dt
    const ggml_tensor * src3 = dst->src[3];  // A
    const ggml_tensor * src4 = dst->src[4];  // B
    const ggml_tensor * src5 = dst->src[5];  // C
    const ggml_tensor * src6 = dst->src[6];  // ids

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_F32);
    GGML_ASSERT(src4->type == GGML_TYPE_F32);
    GGML_ASSERT(src5->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

    const int64_t d_state  = src0->ne[0];  // nc
    const int64_t d_head   = src0->ne[1];  // nr
    const int64_t n_head   = src1->ne[1];  // nh
    const int64_t n_group  = src4->ne[1];  // ng
    const int64_t n_tok    = src1->ne[2];
    const int64_t n_seq    = src1->ne[3];

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + d_state * d_head * n_head * n_seq == ggml_nelements(dst));

    const float   * src0_d = (const float   *) src0->data;
    const float   * src1_d = (const float   *) src1->data;
    const float   * src2_d = (const float   *) src2->data;
    const float   * src3_d = (const float   *) src3->data;
    const float   * src4_d = (const float   *) src4->data;
    const float   * src5_d = (const float   *) src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float         * dst_d  = (float         *) dst->data;
    queue         * q      = ctx.stream();

    if (src3->nb[1] == sizeof(float)) {
        // Mamba-2
        if (d_state == 128) {
            launch_ssm_scan_f32_group<128>(
                *q,
                src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                src2->nb[1], src2->nb[2], src3->nb[1],
                src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                s_off, n_head, d_head, n_group, n_tok, n_seq);
        } else if (d_state == 256) {
            launch_ssm_scan_f32_group<256>(
                *q,
                src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                src2->nb[1], src2->nb[2], src3->nb[1],
                src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                s_off, n_head, d_head, n_group, n_tok, n_seq);
        } else {
            GGML_ABORT("SSM_SCAN Mamba-2: unsupported d_state (must be 128 or 256)");
        }
    } else {
        // Mamba-1
        GGML_ASSERT(d_state == 16);
        GGML_ASSERT(d_head  == 1);
        GGML_ASSERT(n_group == 1);
        GGML_ASSERT(n_head % 128 == 0);
        launch_ssm_scan_f32_mamba1(
            *q,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
            src2->nb[1], src2->nb[2], src3->nb[1],
            src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
            s_off, n_head, n_tok, n_seq);
    }
}
