#include "common.cuh"
#include "wkv.cuh"

template <int block_size>
static __global__ void rwkv_wkv_f32(const int B, const int T, const int C, const int H, const float * k, const float * v, const float * r, const float * tf, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    __syncthreads();
    _tf[tid] = tf[head_i * head_size + tid];
    __syncthreads();

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            y += r.x * (tf.x * kv.x + s.x);
            y += r.y * (tf.y * kv.y + s.y);
            y += r.z * (tf.z * kv.z + s.z);
            y += r.w * (tf.w * kv.w + s.w);

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <int block_size, bool fused_ab>
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * state_dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

#ifndef GGML_USE_MUSA
    #pragma unroll
#endif
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        if constexpr (fused_ab) {
            _a[tid] = -a[t];
            _b[tid] =  a[t] * b[t];
        } else {
            _a[tid] = a[t];
            _b[tid] = b[t];
        }
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& a = (float4&)(_a[j]);
            const float4& s = (float4&)(state[j]);
            sa += a.x * s.x;
            sa += a.y * s.y;
            sa += a.z * s.z;
            sa += a.w * s.w;
        }

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& b = (float4&)(_b[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * w.x + kv.x + sa * b.x;
            s.y = s.y * w.y + kv.y + sa * b.y;
            s.z = s.z * w.z + kv.z + sa * b.z;
            s.w = s.w * w.w + kv.w + sa * b.w;

            y += s.x * r.x;
            y += s.y * r.y;
            y += s.z * r.z;
            y += s.w * r.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state_dst[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

template <int rows_per_block, bool fused_ab, bool fused_l2>
static __global__ void __launch_bounds__(WARP_SIZE * rows_per_block, 2)
rwkv_wkv7_f32_t1_warp_row(const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * state_dst, const float * kk_weight, const float kk_eps) {
    constexpr int head_size = CUDA_WKV_BLOCK_SIZE;
    constexpr int half_head = head_size / 2;

    const int lane = threadIdx.x;
    const int row  = blockIdx.y * rows_per_block + threadIdx.y;
    const int bid  = blockIdx.x;

    const int batch_i = bid / H;
    const int head_i  = bid % H;
    const int state_size = C * head_size;
    const int head_off = head_i * head_size;
    const int t = batch_i * C + head_off + row;

    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

    if (threadIdx.y == 0) {
        const int i0 = batch_i * C + head_off + lane;
        const int i1 = i0 + half_head;
        _r[lane] = r[i0];
        _w[lane] = w[i0];
        _k[lane] = k[i0];
        _r[lane + half_head] = r[i1];
        _w[lane + half_head] = w[i1];
        _k[lane + half_head] = k[i1];

        if constexpr (fused_l2) {
            const float kk0 = a[i0] * (kk_weight != nullptr ? kk_weight[head_off + lane] : 1.0f);
            const float kk1 = a[i1] * (kk_weight != nullptr ? kk_weight[head_off + lane + half_head] : 1.0f);
            const float norm2 = warp_reduce_sum(kk0 * kk0 + kk1 * kk1);
            const float scale = rsqrtf(fmaxf(norm2, kk_eps * kk_eps));
            _a[lane]             = -kk0 * scale;
            _b[lane]             =  kk0 * scale * b[i0];
            _a[lane + half_head] = -kk1 * scale;
            _b[lane + half_head] =  kk1 * scale * b[i1];
        } else if constexpr (fused_ab) {
            _a[lane] = -a[i0];
            _b[lane] =  a[i0] * b[i0];
            _a[lane + half_head] = -a[i1];
            _b[lane + half_head] =  a[i1] * b[i1];
        } else {
            _a[lane] = a[i0];
            _b[lane] = b[i0];
            _a[lane + half_head] = a[i1];
            _b[lane + half_head] = b[i1];
        }
    }
    __syncthreads();

    const int64_t state_base = batch_i * state_size + head_i * head_size * head_size + row * head_size;
    const float s0 = s[state_base + lane];
    const float s1 = s[state_base + lane + half_head];
    const float sa = warp_reduce_sum(_a[lane] * s0 + _a[lane + half_head] * s1);

    const float vt  = v[t];
    const float st0 = s0 * _w[lane]             + _k[lane]             * vt + sa * _b[lane];
    const float st1 = s1 * _w[lane + half_head] + _k[lane + half_head] * vt + sa * _b[lane + half_head];
    const float y   = warp_reduce_sum(st0 * _r[lane] + st1 * _r[lane + half_head]);

    state_dst[state_base + lane]             = st0;
    state_dst[state_base + lane + half_head] = st1;

    if (lane == 0) {
        dst[t] = y;
    }
}

template <int head_size, int warps_per_block, int rows_per_warp, bool fused_ab>
static __global__ void __launch_bounds__(WARP_SIZE * warps_per_block, 2)
rwkv_wkv7_f32_warp_row_tile(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * state_dst) {
    static_assert(head_size % WARP_SIZE == 0, "head size must be a multiple of the warp size");

    constexpr int cols_per_lane = head_size / WARP_SIZE;
    constexpr int rows_per_block = warps_per_block * rows_per_warp;
    constexpr int block_threads  = WARP_SIZE * warps_per_block;

    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int row_base = blockIdx.y * rows_per_block + warp * rows_per_warp;
    const int bid      = blockIdx.x;

    const int batch_i      = bid / H;
    const int head_i       = bid % H;
    const int state_size   = C * head_size;
    const int n_seq_tokens = T / B;
    const int head_off     = head_i * head_size;

    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];
    __shared__ float _v[rows_per_block];

    const int64_t head_state_base = batch_i * state_size + head_i * head_size * head_size;
    float state[rows_per_warp][cols_per_lane];

#pragma unroll
    for (int row_i = 0; row_i < rows_per_warp; ++row_i) {
#pragma unroll
        for (int col_i = 0; col_i < cols_per_lane; ++col_i) {
            state[row_i][col_i] = s[head_state_base + (row_base + row_i) * head_size + lane + col_i * WARP_SIZE];
        }
    }

    const int thread_i = warp * WARP_SIZE + lane;
    for (int token_i = 0; token_i < n_seq_tokens; ++token_i) {
        const int token_base = (batch_i * n_seq_tokens + token_i) * C + head_off;

        for (int col = thread_i; col < head_size; col += block_threads) {
            _r[col] = r[token_base + col];
            _w[col] = w[token_base + col];
            _k[col] = k[token_base + col];
            if constexpr (fused_ab) {
                _a[col] = -a[token_base + col];
                _b[col] =  a[token_base + col] * b[token_base + col];
            } else {
                _a[col] = a[token_base + col];
                _b[col] = b[token_base + col];
            }
        }
        for (int row_i = thread_i; row_i < rows_per_block; row_i += block_threads) {
            _v[row_i] = v[token_base + blockIdx.y * rows_per_block + row_i];
        }
        __syncthreads();

#pragma unroll
        for (int row_i = 0; row_i < rows_per_warp; ++row_i) {
            float sa = 0.0f;
#pragma unroll
            for (int col_i = 0; col_i < cols_per_lane; ++col_i) {
                const int col = lane + col_i * WARP_SIZE;
                sa += _a[col] * state[row_i][col_i];
            }
            sa = warp_reduce_sum(sa);

            const float vt = _v[warp * rows_per_warp + row_i];
            float y = 0.0f;
#pragma unroll
            for (int col_i = 0; col_i < cols_per_lane; ++col_i) {
                const int col = lane + col_i * WARP_SIZE;
                state[row_i][col_i] = state[row_i][col_i] * _w[col] + _k[col] * vt + sa * _b[col];
                y += state[row_i][col_i] * _r[col];
            }
            y = warp_reduce_sum(y);

            if (lane == 0) {
                dst[token_base + row_base + row_i] = y;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row_i = 0; row_i < rows_per_warp; ++row_i) {
#pragma unroll
        for (int col_i = 0; col_i < cols_per_lane; ++col_i) {
            state_dst[head_state_base + (row_base + row_i) * head_size + lane + col_i * WARP_SIZE] = state[row_i][col_i];
        }
    }
}

template <int head_size, int warps_per_block, int rows_per_warp, bool fused_ab>
static void launch_rwkv_wkv7_f32_warp_row_tile(
        const int B, const int T, const int C, const int H,
        const float * r, const float * w, const float * k, const float * v,
        const float * a, const float * b, const float * s, float * dst, float * state_dst,
        cudaStream_t stream) {
    constexpr int rows_per_block = warps_per_block * rows_per_warp;
    static_assert(head_size % rows_per_block == 0, "rows per block must divide the head size");
    rwkv_wkv7_f32_warp_row_tile<head_size, warps_per_block, rows_per_warp, fused_ab>
        <<<dim3(B * H, head_size / rows_per_block), dim3(WARP_SIZE, warps_per_block), 0, stream>>>
        (B, T, C, H, r, w, k, v, a, b, s, dst, state_dst);
}

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * tf_d = (const float *)dst->src[3]->data;
    const float * td_d = (const float *)dst->src[4]->data;
    const float * s_d  = (const float *)dst->src[5]->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    } else {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    }
}

template <bool fused_ab, bool fused_l2>
static void ggml_cuda_op_rwkv_wkv7_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d,
        const float * kk_d, const float * gate_d, const float * kk_weight_d, float kk_eps) {
    const float * r_d = (const float *)dst->src[0]->data;
    const float * w_d = (const float *)dst->src[1]->data;
    const float * k_d = (const float *)dst->src[2]->data;
    const float * v_d = (const float *)dst->src[3]->data;
    const float * a_d = fused_ab ? kk_d   : (const float *) dst->src[4]->data;
    const float * b_d = fused_ab ? gate_d : (const float *) dst->src[5]->data;
    const float * s_d = (const float *)dst->src[6]->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(!fused_ab || (a_d != nullptr && b_d != nullptr));
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    const int cc           = ggml_cuda_info().devices[ctx.device].cc;
    const int head_size    = C / H;
    const int n_seq_tokens = T / B;

    if constexpr (fused_l2) {
        GGML_ASSERT(GGML_CUDA_CC_IS_NVIDIA(cc));
        GGML_ASSERT(head_size == CUDA_WKV_BLOCK_SIZE && n_seq_tokens == 1);
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block, true, true><<<
            dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>
            (T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, kk_weight_d, kk_eps);
        return;
    }

    if (GGML_CUDA_CC_IS_NVIDIA(cc) && n_seq_tokens == 1 && head_size == CUDA_WKV_BLOCK_SIZE) {
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block, fused_ab, false><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, nullptr, 0.0f);
    } else if (GGML_CUDA_CC_IS_NVIDIA(cc) && head_size == CUDA_WKV_BLOCK_SIZE) {
        if (n_seq_tokens <= 4) {
            launch_rwkv_wkv7_f32_warp_row_tile<CUDA_WKV_BLOCK_SIZE, 4, 1, fused_ab>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, stream);
        } else {
            rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE, fused_ab><<<B * H, head_size, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d);
        }
    } else if (GGML_CUDA_CC_IS_NVIDIA(cc) && head_size == CUDA_WKV_BLOCK_SIZE * 2) {
        if (n_seq_tokens <= 4) {
            launch_rwkv_wkv7_f32_warp_row_tile<CUDA_WKV_BLOCK_SIZE * 2, 4, 1, fused_ab>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, stream);
        } else {
            launch_rwkv_wkv7_f32_warp_row_tile<CUDA_WKV_BLOCK_SIZE * 2, 4, 8, fused_ab>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, stream);
        }
    } else if (T / B == 1 && C / H == CUDA_WKV_BLOCK_SIZE) {
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block, fused_ab, false><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d, nullptr, 0.0f);
    } else if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE, fused_ab><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2, fused_ab><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, state_d);
    }
}

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float * state_d = (float *) dst->data + ggml_nelements(dst->src[0]);
    ggml_cuda_op_rwkv_wkv7_impl<false, false>(ctx, dst, state_d, nullptr, nullptr, nullptr, 0.0f);
}

void ggml_cuda_op_rwkv_wkv7_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_rwkv_wkv7_fused_cache cache) {
    if (cache.kk != nullptr) {
        GGML_ASSERT(cache.gate != nullptr);
        if (cache.kk_eps >= 0.0f) {
            ggml_cuda_op_rwkv_wkv7_impl<true, true>(
                ctx, dst, cache.data, cache.kk, cache.gate, cache.kk_weight, cache.kk_eps);
        } else {
            ggml_cuda_op_rwkv_wkv7_impl<true, false>(
                ctx, dst, cache.data, cache.kk, cache.gate, nullptr, 0.0f);
        }
    } else {
        ggml_cuda_op_rwkv_wkv7_impl<false, false>(ctx, dst, cache.data, nullptr, nullptr, nullptr, 0.0f);
    }
}
