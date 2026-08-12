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

template <int block_size>
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
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
        _a[tid] = a[t];
        _b[tid] = b[t];
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
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

template <int rows_per_block>
static __global__ void __launch_bounds__(WARP_SIZE * rows_per_block, 2)
rwkv_wkv7_f32_t1_warp_row(const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
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
        _r[lane] = r[batch_i * C + head_off + lane];
        _w[lane] = w[batch_i * C + head_off + lane];
        _k[lane] = k[batch_i * C + head_off + lane];
        _a[lane] = a[batch_i * C + head_off + lane];
        _b[lane] = b[batch_i * C + head_off + lane];

        _r[lane + half_head] = r[batch_i * C + head_off + lane + half_head];
        _w[lane + half_head] = w[batch_i * C + head_off + lane + half_head];
        _k[lane + half_head] = k[batch_i * C + head_off + lane + half_head];
        _a[lane + half_head] = a[batch_i * C + head_off + lane + half_head];
        _b[lane + half_head] = b[batch_i * C + head_off + lane + half_head];
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

    dst[T * C + state_base + lane]             = st0;
    dst[T * C + state_base + lane + half_head] = st1;

    if (lane == 0) {
        dst[t] = y;
    }
}

template <int head_size, bool has_gate>
static __global__ void __launch_bounds__(head_size)
rwkv7_output_f32(
        const int H,
        const float eps,
        const float * x,
        const float * norm_weight,
        const float * norm_bias,
        const float * k,
        const float * r,
        const float * r_k,
        const float * v,
        const float * gate,
        float * dst) {
    const int tid      = threadIdx.x;
    const int token    = blockIdx.x / H;
    const int head     = blockIdx.x % H;
    const int C        = H * head_size;
    const int head_off = head * head_size;
    const int base     = token * C + head_off;

    __shared__ float stats[3];
    __shared__ float corr_reduce[32];

    // Match the standalone norm kernel's 32-thread reduction order. One warp
    // handles all values in the head while the remaining warps prepare the
    // correction reduction below.
    if (tid < WARP_SIZE) {
        float2 mean_var = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int col = tid; col < head_size; col += WARP_SIZE) {
            const float xi = x[base + col];
            mean_var.x += xi;
            mean_var.y += xi * xi;
        }
        mean_var = warp_reduce_sum(mean_var);
        if (tid == 0) {
            const float mean = mean_var.x / head_size;
            const float var  = mean_var.y / head_size - mean * mean;
            stats[0] = mean;
            stats[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float corr = k[base + tid] * r[base + tid];
    corr *= r_k[head_off + tid];
    corr = block_reduce<block_reduce_method::SUM, head_size>(corr, corr_reduce);
    if (tid == 0) {
        stats[2] = corr;
    }
    __syncthreads();

    float out = (x[base + tid] - stats[0]) * stats[1];
    out *= norm_weight[head_off + tid];
    out += norm_bias[head_off + tid];
    out += v[base + tid] * stats[2];
    if constexpr (has_gate) {
        out *= gate[base + tid];
    }
    dst[base + tid] = out;
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

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * r_d = (const float *)dst->src[0]->data;
    const float * w_d = (const float *)dst->src[1]->data;
    const float * k_d = (const float *)dst->src[2]->data;
    const float * v_d = (const float *)dst->src[3]->data;
    const float * a_d = (const float *)dst->src[4]->data;
    const float * b_d = (const float *)dst->src[5]->data;
    const float * s_d = (const float *)dst->src[6]->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (T / B == 1 && C / H == CUDA_WKV_BLOCK_SIZE) {
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    } else if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    }
}

void ggml_cuda_op_rwkv7_output_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm,
        const ggml_tensor * norm_weight,
        const ggml_tensor * norm_bias,
        const ggml_tensor * k,
        const ggml_tensor * r,
        const ggml_tensor * r_k,
        const ggml_tensor * v,
        const ggml_tensor * gate,
        ggml_tensor * dst) {
    const ggml_tensor * x = norm->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(norm->type == GGML_TYPE_F32);
    GGML_ASSERT(norm_weight->type == GGML_TYPE_F32);
    GGML_ASSERT(norm_bias->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(r->type == GGML_TYPE_F32);
    GGML_ASSERT(r_k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(gate == nullptr || gate->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int head_size = x->ne[0];
    const int H         = x->ne[1];
    const int n_tokens  = ggml_nelements(x) / (head_size * H);

    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const dim3 blocks(H * n_tokens, 1, 1);
    cudaStream_t stream = ctx.stream();

    if (head_size == 64) {
        if (gate != nullptr) {
            rwkv7_output_f32<64, true><<<blocks, 64, 0, stream>>>(
                H, eps,
                (const float *) x->data,
                (const float *) norm_weight->data,
                (const float *) norm_bias->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) r_k->data,
                (const float *) v->data,
                (const float *) gate->data,
                (float *) dst->data);
        } else {
            rwkv7_output_f32<64, false><<<blocks, 64, 0, stream>>>(
                H, eps,
                (const float *) x->data,
                (const float *) norm_weight->data,
                (const float *) norm_bias->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) r_k->data,
                (const float *) v->data,
                nullptr,
                (float *) dst->data);
        }
    } else {
        GGML_ASSERT(head_size == 128);
        if (gate != nullptr) {
            rwkv7_output_f32<128, true><<<blocks, 128, 0, stream>>>(
                H, eps,
                (const float *) x->data,
                (const float *) norm_weight->data,
                (const float *) norm_bias->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) r_k->data,
                (const float *) v->data,
                (const float *) gate->data,
                (float *) dst->data);
        } else {
            rwkv7_output_f32<128, false><<<blocks, 128, 0, stream>>>(
                H, eps,
                (const float *) x->data,
                (const float *) norm_weight->data,
                (const float *) norm_bias->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) r_k->data,
                (const float *) v->data,
                nullptr,
                (float *) dst->data);
        }
    }
}
