#include "ggml-cuda/common.cuh"
#include "gated_delta_net.cuh"

template<int S_v>
__global__ void gated_delta_net_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * curr_state,
    float * dst,
    int64_t H,
    int64_t n_tokens,
    int64_t n_seqs
) {
    const int64_t h_idx    = blockIdx.x;
    const int64_t sequence = blockIdx.y;
    const int col = threadIdx.x; // each thread owns one column

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float * attn_data = dst;
    float * state     = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_offset;
    curr_state += state_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    // Copy input state to output state (working area)
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        state[i * S_v + col] = curr_state[i * S_v + col];
    }

    for (int t = 0; t < n_tokens; t++) {
        const int64_t qkv_offset = sequence * n_tokens * H * S_v + t * H * S_v + h_idx * S_v;
        const float * q_t = q + qkv_offset;
        const float * k_t = k + qkv_offset;
        const float * v_t = v + qkv_offset;

        const int64_t gb_offset = sequence * n_tokens * H + t * H + h_idx;
        const float beta_val = 1.0f / (1.0f + expf(-beta[gb_offset]));
        const float g_val = expf(g[gb_offset]);

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            kv_col += state[i * S_v + col] * k_t[i];
        }

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[i * S_v + col] = g_val * state[i * S_v + col] + k_t[i] * delta_col;
        }

        // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
        float attn_col = 0.0f;
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            attn_col += state[i * S_v + col] * q_t[i];
        }
        attn_data[col] = attn_col;
        attn_data += S_v * H;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_q->ne[0];
    const int64_t H        = src_q->ne[1];
    const int64_t n_tokens = src_q->ne[2];
    const int64_t n_seqs   = src_q->ne[3];

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;

    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;

    float * dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(src_q));
    GGML_ASSERT(ggml_is_contiguous(src_k));
    GGML_ASSERT(ggml_is_contiguous(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    dim3 grid_dims(H, n_seqs, 1);
    dim3 block_dims(S_v, 1, 1);

    cudaStream_t stream = ctx.stream();

    switch(S_v) {
        case 32:
            gated_delta_net_cuda<32><<<grid_dims, block_dims, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs);
            break;
        case 64:
            gated_delta_net_cuda<64><<<grid_dims, block_dims, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs);
            break;
        case 128:
            gated_delta_net_cuda<128><<<grid_dims, block_dims, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
