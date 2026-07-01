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
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * kk, const float * a, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _kk[head_size], _a[head_size];

#ifndef GGML_USE_MUSA
    #pragma unroll
#endif
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        constexpr float w_scale = -0.6065306597126334f;
        _r[tid] = r[t];
        _w[tid] = __expf(w_scale / (1.0f + __expf(-w[t])));
        _k[tid] = k[t];
        _kk[tid] = kk[t];
        _a[tid] = a[t];
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& kk = (float4&)(_kk[j]);
            const float4& s = (float4&)(state[j]);
            sa += kk.x * s.x;
            sa += kk.y * s.y;
            sa += kk.z * s.z;
            sa += kk.w * s.w;
        }
        sa = -sa;

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& kk = (float4&)(_kk[j]);
            const float4& a = (float4&)(_a[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * w.x + kv.x + sa * kk.x * a.x;
            s.y = s.y * w.y + kv.y + sa * kk.y * a.y;
            s.z = s.z * w.z + kv.z + sa * kk.z * a.z;
            s.w = s.w * w.w + kv.w + sa * kk.w * a.w;

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
rwkv_wkv7_f32_t1_warp_row(const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * kk, const float * a, const float * s, float * dst) {
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

    __shared__ float _r[head_size], _w[head_size], _k[head_size], _kk[head_size], _a[head_size];

    if (threadIdx.y == 0) {
        constexpr float w_scale = -0.6065306597126334f;

        _r[lane]          = r[batch_i * C + head_off + lane];
        _w[lane]          = __expf(w_scale / (1.0f + __expf(-w[batch_i * C + head_off + lane])));
        _k[lane]          = k[batch_i * C + head_off + lane];
        _kk[lane]         = kk[batch_i * C + head_off + lane];
        _a[lane]          = a[batch_i * C + head_off + lane];

        _r[lane + half_head]   = r[batch_i * C + head_off + lane + half_head];
        _w[lane + half_head]   = __expf(w_scale / (1.0f + __expf(-w[batch_i * C + head_off + lane + half_head])));
        _k[lane + half_head]   = k[batch_i * C + head_off + lane + half_head];
        _kk[lane + half_head]  = kk[batch_i * C + head_off + lane + half_head];
        _a[lane + half_head]   = a[batch_i * C + head_off + lane + half_head];
    }
    __syncthreads();

    const int64_t state_base = batch_i * state_size + head_i * head_size * head_size + row * head_size;
    const float s0 = s[state_base + lane];
    const float s1 = s[state_base + lane + half_head];

    const float sa = -warp_reduce_sum(_kk[lane] * s0 + _kk[lane + half_head] * s1);

    const float vt  = v[t];
    const float st0 = s0 * _w[lane]             + _k[lane]             * vt + sa * _kk[lane]             * _a[lane];
    const float st1 = s1 * _w[lane + half_head] + _k[lane + half_head] * vt + sa * _kk[lane + half_head] * _a[lane + half_head];
    const float y   = warp_reduce_sum(st0 * _r[lane] + st1 * _r[lane + half_head]);

    dst[T * C + state_base + lane]             = st0;
    dst[T * C + state_base + lane + half_head] = st1;

    if (lane == 0) {
        dst[t] = y;
    }
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
    const float * kk_d = (const float *)dst->src[4]->data;
    const float * a_d  = (const float *)dst->src[5]->data;
    const float * s_d   = (const float *)dst->src[6]->data;

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
        rwkv_wkv7_f32_t1_warp_row<rows_per_block><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(T, C, H, r_d, w_d, k_d, v_d, kk_d, a_d, s_d, dst_d);
    } else if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, kk_d, a_d, s_d, dst_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, kk_d, a_d, s_d, dst_d);
    }
}
