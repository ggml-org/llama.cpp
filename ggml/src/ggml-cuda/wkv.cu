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
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * kk, const float * a, const float * r_k, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _kk[head_size], _a[head_size], _r_k[head_size];

#ifndef GGML_USE_MUSA
    #pragma unroll
#endif
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }
    _r_k[tid] = r_k[head_i * head_size + tid];

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
        float rk = 0;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& r = (float4&)(_r[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& kk = (float4&)(_kk[j]);
            const float4& r_k = (float4&)(_r_k[j]);
            const float4& s = (float4&)(state[j]);
            sa += kk.x * s.x;
            sa += kk.y * s.y;
            sa += kk.z * s.z;
            sa += kk.w * s.w;
            rk += r.x * k.x * r_k.x;
            rk += r.y * k.y * r_k.y;
            rk += r.z * k.z * r_k.z;
            rk += r.w * k.w * r_k.w;
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
        dst[T * C + t] = _v * rk;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[2 * T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
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
    const float * r_k_d = (const float *)dst->src[6]->data;
    const float * s_d   = (const float *)dst->src[7]->data;

    const int64_t B = dst->src[7]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[7]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, kk_d, a_d, r_k_d, s_d, dst_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, kk_d, a_d, r_k_d, s_d, dst_d);
    }
}
