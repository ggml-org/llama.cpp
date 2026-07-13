#include "lightning-indexer.cuh"

// GGML_OP_LIGHTNING_INDEXER (naive, correctness-first, F16/F32 K)
template <typename T>
static __global__ void lightning_indexer_kernel(
        const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ w, const char * __restrict__ m,
        float * __restrict__ dst,
        const int n_embd, const int n_head, const int n_kv,
        const size_t nbq1, const size_t nbq2, const size_t nbq3,
        const size_t nbk2, const size_t nbk3,
        const size_t nbw1, const size_t nbw3,
        const size_t nbm1, const size_t nbm3, const int nem3,
        const size_t nb1,  const size_t nb3) {
    const int ik = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    const int s  = blockIdx.z;
    if (ik >= n_kv) return;

    const float       * w_row = (const float       *)(w + t*nbw1 + s*nbw3);
    const ggml_fp16_t * m_row = (const ggml_fp16_t *)(m + t*nbm1 + (s % nem3)*nbm3);
    float             * d_row = (float             *)((char *)dst + t*nb1 + s*nb3);
    const T           * k_row = (const T           *)(k + (size_t)ik*nbk2 + s*nbk3);

    float score = 0.0f;
    for (int h = 0; h < n_head; ++h) {
        const float * q_row = (const float *)(q + (size_t)h*nbq1 + t*nbq2 + s*nbq3);
        float qk = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            float kv;
            if constexpr (sizeof(T) == 2) { kv = __half2float(((const __half *)k_row)[i]); }
            else                          { kv = ((const float *)k_row)[i]; }
            qk += q_row[i] * kv;
        }
        score += fmaxf(qk, 0.0f) * w_row[h];
    }
    d_row[ik] = score + __half2float(*(const __half *)&m_row[ik]);
}

void ggml_cuda_op_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F16);

    const int n_embd = q->ne[0], n_head = q->ne[1];
    const int n_tokens = q->ne[2], n_stream = q->ne[3], n_kv = k->ne[2];
    const int block = 128;
    const dim3 grid((n_kv + block - 1)/block, n_tokens, n_stream);
    cudaStream_t stream = ctx.stream();

    if (k->type == GGML_TYPE_F16) {
        lightning_indexer_kernel<__half><<<grid, block, 0, stream>>>(
            (const char*)q->data,(const char*)k->data,(const char*)w->data,(const char*)m->data,(float*)dst->data,
            n_embd,n_head,n_kv, q->nb[1],q->nb[2],q->nb[3], k->nb[2],k->nb[3],
            w->nb[1],w->nb[3], m->nb[1],m->nb[3],(int)m->ne[3], dst->nb[1],dst->nb[3]);
    } else if (k->type == GGML_TYPE_F32) {
        lightning_indexer_kernel<float><<<grid, block, 0, stream>>>(
            (const char*)q->data,(const char*)k->data,(const char*)w->data,(const char*)m->data,(float*)dst->data,
            n_embd,n_head,n_kv, q->nb[1],q->nb[2],q->nb[3], k->nb[2],k->nb[3],
            w->nb[1],w->nb[3], m->nb[1],m->nb[3],(int)m->ne[3], dst->nb[1],dst->nb[3]);
    } else {
        GGML_ABORT("lightning_indexer: unsupported K type %d", (int)k->type);
    }
}
