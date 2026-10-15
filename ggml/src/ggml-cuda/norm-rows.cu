#include "norm-rows.cuh"
#include "unary.cuh"

#include <cstdlib>

enum norm_rows_mode {
    NORM_ROWS_MUL,      // dst = scale*x*w
    NORM_ROWS_SCALE,    // dst = s*(scale*x)
    NORM_ROWS_MUL_SILU, // dst = (scale*x*w) * silu(z)
};

// the xor tree of warp_reduce_sum
static __device__ __forceinline__ float norm_rows_xor_tree(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset, 32);
    }
    return v;
}

// One wave (32 lanes) per row of ncols <= 256. rms_norm_f32<256> gives thread t column t (or 0 past the row), reduces
// each warp's 32 columns with the xor tree, then reduces the 8 warp sums padded with zeros with the same tree; this
// does the same sums in the same order.
template <norm_rows_mode mode>
static __global__ void __launch_bounds__(256) rms_rows_f32(
        const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ z, float * __restrict__ dst,
        const int ncols, const int64_t nrows, const int64_t nchannels, const int64_t total_rows,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const float eps, const float scale_out) {
    const int     lane = threadIdx.x % 32;
    const int64_t g    = (int64_t) blockIdx.x * 8 + threadIdx.x / 32;
    if (g >= total_rows) {
        return;
    }
    const int64_t row     = g % nrows;
    const int64_t channel = (g / nrows) % nchannels;
    const int64_t sample  = g / (nrows * nchannels);
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += g*ncols;
    if constexpr (mode == NORM_ROWS_MUL_SILU) {
        z += g*ncols;
    }

    float part = 0.0f; // lane wv holds the sum of columns [32*wv, 32*wv + 32)
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32*wv + lane;
        // accumulate exactly as rms_norm_f32 does (tmp = 0; tmp += xi*xi): written as a bare product, the square would
        // be contracted with the first add of the tree into one fma and round differently
        float tmp = 0.0f;
        if (col < ncols) {
            const float xi = x[col];
            tmp += xi*xi;
        }
        const float s = norm_rows_xor_tree(tmp);
        if (lane == wv) {
            part = s;
        }
    }
    const float tmp   = norm_rows_xor_tree(part);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32*wv + lane;
        if (col < ncols) {
            if constexpr (mode == NORM_ROWS_SCALE) {
                dst[col] = scale_out * (scale * x[col]);
            } else {
                const float t = scale * x[col] * w[col];
                if constexpr (mode == NORM_ROWS_MUL_SILU) {
                    dst[col] = t * ggml_cuda_op_silu_single(z[col]);
                } else {
                    dst[col] = t;
                }
            }
        }
    }
}

static bool norm_rows_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_DISABLE_NORM_ROWS");
        return env == nullptr || atoi(env) == 0;
    }();
    return enabled;
}

static bool norm_rows_is_view(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
        t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static bool norm_rows_computed(const ggml_tensor * t) {
    return (t->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
}

// x: the RMS norm's input; out: the tensor the fusion writes (same shape, contiguous)
static bool norm_rows_shape_ok(const ggml_tensor * x, const ggml_tensor * out) {
    if (x->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[0] < 1 || x->ne[0] > 256 || x->nb[0] != sizeof(float)) {
        return false;
    }
    // with fewer rows one wave per row is slower than the stock norm (176 per-head norms of a 4-token verify: 617 vs 339 us)
    if (x->ne[1]*x->ne[2]*x->ne[3] < 4096) {
        return false;
    }
    return ggml_is_contiguous(out) && ggml_are_same_shape(x, out);
}

// a weight broadcast over all rows
static bool norm_rows_weight_ok(const ggml_tensor * w, int64_t ncols) {
    return w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == ncols &&
        w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1;
}

static const ggml_tensor * norm_rows_other_src(const ggml_tensor * node, const ggml_tensor * src) {
    if (node->src[0] == src) {
        return node->src[1];
    }
    return node->src[1] == src ? node->src[0] : nullptr;
}

template <norm_rows_mode mode>
static void norm_rows_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
                             const ggml_tensor * z, ggml_tensor * dst, float eps, float scale_out) {
    const int64_t total = x->ne[1]*x->ne[2]*x->ne[3];
    const dim3 grid((unsigned) ((total + 7) / 8));
    rms_rows_f32<mode><<<grid, 256, 0, ctx.stream()>>>(
        (const float *) x->data, w ? (const float *) w->data : nullptr, z ? (const float *) z->data : nullptr,
        (float *) dst->data, (int) x->ne[0], x->ne[1], x->ne[2], total,
        (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)), (int64_t) (x->nb[3] / sizeof(float)),
        eps, scale_out);
    CUDA_CHECK(cudaGetLastError());
}

int ggml_cuda_try_fuse_norm_rows(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!norm_rows_enabled() || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * rms = cgraph->nodes[i];
    ggml_tensor *       nxt = cgraph->nodes[i + 1];
    if (rms->op != GGML_OP_RMS_NORM || !norm_rows_computed(nxt) || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    const ggml_tensor * x = rms->src[0];
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    if (eps < 0.0f) {
        return 0;
    }

    if (nxt->op == GGML_OP_SCALE && nxt->src[0] == rms) {
        float s, b;
        memcpy(&s, (const float *) nxt->op_params + 0, sizeof(float));
        memcpy(&b, (const float *) nxt->op_params + 1, sizeof(float));
        if (b != 0.0f || !norm_rows_shape_ok(x, nxt)) {
            return 0;
        }
        norm_rows_launch<NORM_ROWS_SCALE>(ctx, x, nullptr, nullptr, nxt, eps, s);
        return 1;
    }

    if (nxt->op != GGML_OP_MUL) {
        return 0;
    }
    const ggml_tensor * w = norm_rows_other_src(nxt, rms);
    if (!norm_rows_weight_ok(w, x->ne[0]) || !ggml_are_same_shape(rms, nxt)) {
        return 0;
    }

    // gated form: [views] -> UNARY(silu, z) -> MUL(nxt, silu)
    int k = i + 2;
    while (k < cgraph->n_nodes && k < i + 6 && norm_rows_is_view(cgraph->nodes[k])) {
        ++k;
    }
    if (k + 1 < cgraph->n_nodes && ggml_node_has_n_uses(cgraph, i + 1, 1)) {
        const ggml_tensor * sil = cgraph->nodes[k];
        ggml_tensor *       out = cgraph->nodes[k + 1];
        if (sil->op == GGML_OP_UNARY && ggml_get_unary_op(sil) == GGML_UNARY_OP_SILU && out->op == GGML_OP_MUL &&
                norm_rows_computed(sil) && norm_rows_computed(out) && ggml_node_has_n_uses(cgraph, k, 1) &&
                norm_rows_other_src(out, nxt) == sil) {
            const ggml_tensor * z = sil->src[0];
            if (z->type == GGML_TYPE_F32 && ggml_is_contiguous(z) && ggml_are_same_shape(z, out) &&
                    ggml_are_same_shape(sil, out) && norm_rows_shape_ok(x, out)) {
                norm_rows_launch<NORM_ROWS_MUL_SILU>(ctx, x, w, z, out, eps, 1.0f);
                return (k + 1) - i;
            }
        }
    }

    if (!norm_rows_shape_ok(x, nxt)) {
        return 0;
    }
    norm_rows_launch<NORM_ROWS_MUL>(ctx, x, w, nullptr, nxt, eps, 1.0f);
    return 1;
}
