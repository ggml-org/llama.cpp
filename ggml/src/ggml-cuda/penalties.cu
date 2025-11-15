#include "penalties.cuh"

static __global__ void penalties_f32(
        const float   * logits,
        const int32_t * history,
        const int32_t * n_history,
        const int32_t   n_max_history,
        float         * dst,
        const int       n_vocab,
        const float     repeat,
        const float     freq,
        const float     present) {

    const int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token >= n_vocab) {
        return;
    }

    __shared__ int32_t n_history_shared;

    if (threadIdx.x == 0) {
        // Read history size from global memory
        int32_t hs = *n_history;

        // Clamp to valid range
        if (hs < 0) {
            hs = 0;
        }
        if (hs > n_max_history) {
            hs = n_max_history;
        }
        // Write history size to shared memory.
        n_history_shared = hs;
    }

    __syncthreads();

    // Read history size from shared memory to a private register.
    const int32_t n_history_local = n_history_shared;

    // Read the logit for this threads token from global memory to a private register.
    float logit = logits[token];

    // Find out how many times the token appears in the token history.
    int count = 0;
    for (int i = 0; i < n_history_local; i++) {
        if (history[i] == token) {
            count++;
        }
    }

    // If this threads token appears in the token history we apply penalties.
    if (count > 0) {
        // We only apply repeat penalty if it's not 1.0 (unit for multiplication/division).
        if (repeat != 1.0f) {
            if (logit > 0.0f) {
                printf("applying positive (/) repeat penalty to token: %d, count: %d\n", token, count);
                logit /= repeat; // make the logit smaller
            } else {
                printf("applying negative (*) repeat penalty to token: %d, count: %d\n", token, count);
                logit *= repeat; // make the logit more negative
            }
        }

        // Apply frequency (based on the number of times, the count, the token
        // was present in the token history) and presence (a flat penalty which
        // is applied to the token regardless of how many times it appeared).
        logit -= (freq * count) + (present > 0 ? present : 0.0f);
    }

    // Update the global memory with the new logit value.
    dst[token] = logit;
}

static void penalties_f32_cuda(
        const float   * logits,
        const int32_t * history,
        const int32_t * n_history,
        const int32_t   n_max_history,
        float         * dst,
        const int       n_vocab,
        const float     repeat,
        const float     freq,
        const float     present,
        cudaStream_t    stream) {

    const int block_size = 256;
    const int num_blocks = (n_vocab + block_size - 1) / block_size;

    penalties_f32<<<num_blocks, block_size, 0, stream>>>(
            logits,
            history,
            n_history,
            n_max_history,
            dst,
            n_vocab,
            repeat,
            freq, present);
}

void ggml_cuda_op_penalties(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits        = dst->src[0];
    const ggml_tensor * history       = dst->src[1];
    const ggml_tensor * n_history     = dst->src[2];

    const float   * logits_d    = (const float *)   logits->data;
    const int32_t * history_d   = (const int32_t *) history->data;
    const int32_t * n_history_d = (const int32_t *) n_history->data;
    float         * dst_d       = (float *)         dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(logits->type    == GGML_TYPE_F32);
    GGML_ASSERT(history->type   == GGML_TYPE_I32);
    GGML_ASSERT(n_history->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int     n_vocab       = logits->ne[0];
    const int32_t n_max_history = history->ne[0];

    const float repeat  = ggml_get_op_params_f32(dst, 0);
    const float freq    = ggml_get_op_params_f32(dst, 1);
    const float present = ggml_get_op_params_f32(dst, 2);

    penalties_f32_cuda(
        logits_d,
        history_d,
        n_history_d,
        n_max_history,
        dst_d,
        n_vocab,
        repeat,
        freq,
        present,
        stream);
}
