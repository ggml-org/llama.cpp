#include "common.cuh"
#include "delta-net.cuh"
#include <cstdlib>

// Delta Net Linear Attention Kernel for Qwen3-Next (HEAD_DIM=128)
// State layout: [S_v, S_v*H_v, 1, n_seqs] (column-major)

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Token-by-token recurrent kernel
// One block per (batch, head) pair, processes all tokens sequentially
// State is kept in global memory (too large for shared memory at HEAD_DIM=128)
template <int HEAD_DIM>
__global__ void delta_net_recurrent_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state concatenated
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,          // offset where state starts in output
    const float eps)
{
    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;  // 0-7 for 256 threads
    const int lane_id = tid % WARP_SIZE;  // 0-31
    constexpr int NUM_WARPS = 8;          // 256 / 32

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;

    // G/Beta: [n_tokens, 1, n_heads, n_seqs] / [1, n_tokens, n_heads, n_seqs]
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for scalars (moved outside loop for clarity)
    __shared__ float shared_g_val, shared_beta_val, shared_decay, shared_attn_score;

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM
    float * sV = sK + HEAD_DIM;                 // HEAD_DIM
    float * sKBeta = sV + HEAD_DIM;             // HEAD_DIM (plain k for state update)
    float * sVBeta = sKBeta + HEAD_DIM;         // HEAD_DIM (v * sigmoid(beta))
    float * sOut = sVBeta + HEAD_DIM;           // HEAD_DIM
    float * sKCumdecay = sOut + HEAD_DIM;       // HEAD_DIM (k * sigmoid(beta) * exp(g))
    float * sVPrime = sKCumdecay + HEAD_DIM;    // HEAD_DIM (state @ k_cumdecay)
    float * sVNew = sVPrime + HEAD_DIM;         // HEAD_DIM (v_beta - v_prime)
    float * sNorm = sVNew + HEAD_DIM;           // 2 (for Q and K norms)

    const float scale = rsqrtf((float)HEAD_DIM);

    // Copy initial state to output buffer (will be updated in place)
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        int col = i / HEAD_DIM;
        int row = i % HEAD_DIM;
        // Column-major: state[row, col] at index row + col*HEAD_DIM
        state_dst[row + col * HEAD_DIM] = state_src[row + col * HEAD_DIM];
    }
    __syncthreads();

    // Process each token sequentially
    for (int64_t t = 0; t < n_tokens; t++) {
        // Reset norm accumulators
        if (tid < 2) {
            sNorm[tid] = 0.0f;
        }
        __syncthreads();

        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sQ[i] = q_ptr[t * qkv_stride_token + i];
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sV[i] = v_ptr[t * qkv_stride_token + i];
        }
        __syncthreads();

        float q_sq_local = 0.0f;
        float k_sq_local = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            q_sq_local += sQ[i] * sQ[i];
            k_sq_local += sK[i] * sK[i];
        }

        // Warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq_local += __shfl_xor_sync(0xffffffff, q_sq_local, offset);
            k_sq_local += __shfl_xor_sync(0xffffffff, k_sq_local, offset);
        }

        // Cross-warp reduction using shared memory atomics
        if (tid % WARP_SIZE == 0) {
            atomicAdd(&sNorm[0], q_sq_local);
            atomicAdd(&sNorm[1], k_sq_local);
        }
        __syncthreads();

        float q_norm = rsqrtf(sNorm[0] + eps);
        float k_norm = rsqrtf(sNorm[1] + eps);

        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sQ[i] = sQ[i] * q_norm * scale;
            sK[i] = sK[i] * k_norm;
        }
        __syncthreads();

        if (tid == 0) {
            shared_g_val = g_ptr[t];
            shared_beta_val = sigmoid_f(beta_ptr[t]);
            shared_decay = expf(fminf(shared_g_val, 50.0f));
        }
        __syncthreads();

        float beta_val = shared_beta_val;
        float decay = shared_decay;

        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sKBeta[i] = sK[i];
            sVBeta[i] = sV[i] * beta_val;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }
        __syncthreads();

        for (int row_out = warp_id; row_out < HEAD_DIM; row_out += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
                sum += state_dst[row_out + col * HEAD_DIM] * sKCumdecay[col];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                sVPrime[row_out] = sum;
            }
        }
        __syncthreads();

        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum += sK[i] * sQ[i];
            }
            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                shared_attn_score = sum;
            }
        }
        __syncthreads();

        for (int row_out = warp_id; row_out < HEAD_DIM; row_out += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
                float state_val = state_dst[row_out + col * HEAD_DIM];
                sum += sQ[col] * decay * state_val;
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                float v_attn = sVNew[row_out] * shared_attn_score;
                sOut[row_out] = sum + v_attn;
            }
        }
        __syncthreads();

        for (int out_dim = tid; out_dim < HEAD_DIM; out_dim += blockDim.x) {
            for (int row = 0; row < HEAD_DIM; row++) {
                float state_val = state_dst[row + out_dim * HEAD_DIM];
                float safe_decay = decay;
                if (isnan(safe_decay) || isinf(safe_decay)) {
                    safe_decay = 1.0f;
                }
                float new_state_val = safe_decay * state_val + sVNew[row] * sKBeta[out_dim];
                new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
                state_dst[row + out_dim * HEAD_DIM] = new_state_val;
            }
        }
        __syncthreads();

        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            out_base[t * out_token_stride + i] = sOut[i];
        }
        __syncthreads();
    }
}

// Generic kernel that handles any HEAD_DIM at runtime (slower but flexible)
__global__ void delta_net_recurrent_generic_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta_in,
    const float * __restrict__ state_in,
    float * __restrict__ dst,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps)
{
    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;

    // Strides (column-major)
    const int64_t qkv_stride_token = head_dim;
    const int64_t qkv_stride_head = head_dim * n_tokens;
    const int64_t qkv_stride_batch = head_dim * n_tokens * n_heads;

    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    const int64_t state_head_offset = head_idx * head_dim * head_dim;
    const int64_t state_batch_stride = head_dim * head_dim * n_heads;

    // Pointers
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    float * out_base = dst + batch_idx * (head_dim * n_heads * n_tokens) + head_idx * head_dim;
    const int64_t out_token_stride = head_dim * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for scalars (outside loop)
    __shared__ float shared_g_val, shared_beta_val, shared_decay, shared_attn_score;

    // Dynamic shared memory
    extern __shared__ float smem[];
    float * sQ = smem;
    float * sK = sQ + head_dim;
    float * sV = sK + head_dim;
    float * sKBeta = sV + head_dim;             // plain k for state update
    float * sVBeta = sKBeta + head_dim;         // v * sigmoid(beta)
    float * sOut = sVBeta + head_dim;
    float * sKCumdecay = sOut + head_dim;       // k * sigmoid(beta) * exp(g)
    float * sVPrime = sKCumdecay + head_dim;    // state @ k_cumdecay
    float * sVNew = sVPrime + head_dim;         // v_beta - v_prime
    float * sNorm = sVNew + head_dim;

    const float scale = rsqrtf((float)head_dim);

    // Copy initial state to output buffer
    for (int i = tid; i < head_dim * head_dim; i += blockDim.x) {
        int col = i / head_dim;
        int row = i % head_dim;
        state_dst[row + col * head_dim] = state_src[row + col * head_dim];
    }
    __syncthreads();

    // Process each token
    for (int64_t t = 0; t < n_tokens; t++) {
        if (tid < 2) sNorm[tid] = 0.0f;
        __syncthreads();

        // Load Q, K, V
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sQ[i] = q_ptr[t * qkv_stride_token + i];
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sV[i] = v_ptr[t * qkv_stride_token + i];
        }
        __syncthreads();

        // L2 normalize Q and K
        float q_sq = 0.0f, k_sq = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            q_sq += sQ[i] * sQ[i];
            k_sq += sK[i] * sK[i];
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq += __shfl_xor_sync(0xffffffff, q_sq, offset);
            k_sq += __shfl_xor_sync(0xffffffff, k_sq, offset);
        }

        if (tid % WARP_SIZE == 0) {
            atomicAdd(&sNorm[0], q_sq);
            atomicAdd(&sNorm[1], k_sq);
        }
        __syncthreads();

        float q_norm = rsqrtf(sNorm[0] + eps);
        float k_norm = rsqrtf(sNorm[1] + eps);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            sQ[i] *= q_norm * scale;
            sK[i] *= k_norm;
        }
        __syncthreads();

        // Load g and beta, compute decay
        if (tid == 0) {
            shared_g_val = g_ptr[t];
            shared_beta_val = sigmoid_f(beta_ptr[t]);
            shared_decay = expf(fminf(shared_g_val, 50.0f));
        }
        __syncthreads();

        float beta_val = shared_beta_val;
        float decay = shared_decay;

        // Compute k_beta, v_beta, k_cumdecay
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sKBeta[i] = sK[i];
            sVBeta[i] = sV[i] * beta_val;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }
        __syncthreads();

        // Compute v_prime = state @ k_cumdecay
        for (int row_out = tid; row_out < head_dim; row_out += blockDim.x) {
            float v_prime_val = 0.0f;
            for (int col = 0; col < head_dim; col++) {
                // Access state[row_out, col] = state_dst[row_out + col * head_dim] for state @ k
                v_prime_val += state_dst[row_out + col * head_dim] * sKCumdecay[col];
            }
            sVPrime[row_out] = v_prime_val;
        }
        __syncthreads();

        // Compute v_new = v_beta - v_prime (the value residual)
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // Compute attn_score = dot(k, q) (L2 normalized vectors)
        if (tid == 0) {
            float dot_sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                dot_sum += sK[i] * sQ[i];
            }
            shared_attn_score = dot_sum;
        }
        __syncthreads();

        // Compute output: o[t] = attn_inter + v_attn
        // attn_inter = state @ (q * exp(g)) = sum_col(state[row_out, col] * q[col] * exp(g))
        // The decomposed path uses: attn_inter = ggml_mul_mat(state_t, q_g_exp)
        // Since ggml_mul_mat(A,B) = A^T @ B, attn_inter = state_t^T @ q_g_exp = state @ (q * exp(g))
        for (int row_out = tid; row_out < head_dim; row_out += blockDim.x) {
            float attn_inter = 0.0f;

            for (int col = 0; col < head_dim; col++) {
                // Access state[row_out, col] = state_dst[row_out + col * head_dim] for state @ q
                float state_val = state_dst[row_out + col * head_dim];
                attn_inter += sQ[col] * decay * state_val;
            }

            // v_attn = v_new * attn_score
            float v_attn = sVNew[row_out] * shared_attn_score;

            // Output = attn_inter + v_attn (correct DeltaNet formula)
            sOut[row_out] = attn_inter + v_attn;
        }
        __syncthreads();

        // Update state: state_new = decay * state + outer(v_new, k)
        // Fixed: outer product orientation matches decomposed: state[v_idx, k_idx] += v_new[v_idx] * k[k_idx]
        // Uses transposed indexing: state_dst[row + out_dim * head_dim] = state[row][out_dim]
        // Only protect against NaN/Inf - do NOT clamp decay value
        float safe_decay = decay;
        if (isnan(safe_decay) || isinf(safe_decay)) {
            safe_decay = 1.0f;
        }

        for (int out_dim = tid; out_dim < head_dim; out_dim += blockDim.x) {
            for (int row = 0; row < head_dim; row++) {
                float state_val = state_dst[row + out_dim * head_dim];

                // state_new[row][out_dim] = decay * state[row][out_dim] + v_new[row] * k[out_dim]
                // Fix: outer product matches decomposed path: state[v_idx, k_idx] += v_new[v_idx] * k[k_idx]
                float new_state_val = safe_decay * state_val + sVNew[row] * sKBeta[out_dim];

                // Clamp state to prevent overflow
                new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
                state_dst[row + out_dim * head_dim] = new_state_val;
            }
        }
        __syncthreads();

        // Write output
        for (int i = tid; i < head_dim; i += blockDim.x) {
            out_base[t * out_token_stride + i] = sOut[i];
        }
        __syncthreads();
    }
}

// FP16 DeltaNet kernel using __hfma2 for 2x throughput
#if !defined(GGML_USE_HIP)
template <int HEAD_DIM>
__global__ void delta_net_fp16_optimized(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta_in,
    const float * __restrict__ state_in,
    float * __restrict__ dst,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps)
{
    static_assert(HEAD_DIM == 128, "FP16 kernel requires HEAD_DIM=128");
    static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even for half2");

    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = 8;  // 256 threads / 32

    // Strides (column-major)
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // Pointers
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory layout:
    // - FP16 state COLUMN-MAJOR: 128×128 = 16384 half = 32KB
    // - FP16 vectors: K, KCumdecay, Q_scaled = 3 × 128 = 384 half = 768 bytes
    // - FP32 vectors: V, KBeta, VBeta, Out, VPrime, VNew = 6 × 128 = 768 floats = 3KB
    // Total: ~36KB

    extern __shared__ char smem_raw[];

    // FP16 state COLUMN-MAJOR: state[row, col] = state_smem[row + col * HEAD_DIM]
    half * state_smem = (half *)smem_raw;

    // FP16 vectors
    half * sK_fp16 = (half *)(smem_raw + HEAD_DIM * HEAD_DIM * sizeof(half));
    half * sKCumdecay_fp16 = sK_fp16 + HEAD_DIM;
    half * sQ_fp16 = sKCumdecay_fp16 + HEAD_DIM;

    // FP32 vectors
    float * sV = (float *)(sQ_fp16 + HEAD_DIM);
    float * sKBeta = sV + HEAD_DIM;
    float * sVBeta = sKBeta + HEAD_DIM;
    float * sOut = sVBeta + HEAD_DIM;
    float * sVPrime = sOut + HEAD_DIM;
    float * sVNew = sVPrime + HEAD_DIM;
    float * sNorm = sVNew + HEAD_DIM;

    __shared__ float shared_decay, shared_attn_score;

    const float scale = rsqrtf((float)HEAD_DIM);

    // Load initial state DIRECTLY (no transpose - same layout as global)
    // state[row, col] = state_smem[row + col * HEAD_DIM]
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        state_smem[i] = __float2half(state_src[i]);
    }
    __syncthreads();

    // Process each token
    for (int64_t t = 0; t < n_tokens; t++) {
        // Reset norms
        if (tid < 2) {
            sNorm[tid] = 0.0f;
        }
        __syncthreads();

        // 1. Load Q, K, V and compute norms
        float q_sq_local = 0.0f, k_sq_local = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            float q_val = q_ptr[t * qkv_stride_token + i];
            float k_val = k_ptr[t * qkv_stride_token + i];
            sV[i] = v_ptr[t * qkv_stride_token + i];
            q_sq_local += q_val * q_val;
            k_sq_local += k_val * k_val;
            sVPrime[i] = q_val;  // Temp storage for Q
            sVNew[i] = k_val;    // Temp storage for K
        }

        // Warp reduction for norms
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq_local += __shfl_xor_sync(0xffffffff, q_sq_local, offset);
            k_sq_local += __shfl_xor_sync(0xffffffff, k_sq_local, offset);
        }
        if (lane_id == 0) {
            atomicAdd(&sNorm[0], q_sq_local);
            atomicAdd(&sNorm[1], k_sq_local);
        }
        __syncthreads();

        float q_norm = rsqrtf(sNorm[0] + eps);
        float k_norm = rsqrtf(sNorm[1] + eps);

        // 2. Load g and beta, compute decay
        if (tid == 0) {
            shared_decay = expf(fminf(g_ptr[t], 50.0f));  // Clamp g to prevent overflow
        }
        __syncthreads();
        float decay = shared_decay;
        float beta_val = sigmoid_f(beta_ptr[t]);

        // 3. Compute normalized vectors and convert to FP16
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            float q_normalized = sVPrime[i] * q_norm * scale;
            float k_normalized = sVNew[i] * k_norm;

            sQ_fp16[i] = __float2half(q_normalized * decay);
            sK_fp16[i] = __float2half(k_normalized);
            sKCumdecay_fp16[i] = __float2half(k_normalized * beta_val * decay);

            sKBeta[i] = k_normalized;
            sVBeta[i] = sV[i] * beta_val;
        }
        __syncthreads();

        // 4. v_prime = state @ k_cumdecay using half2
        // Column-major: state[row, col] = state_smem[row + col * HEAD_DIM]
        // v_prime[col] = sum_row(state[row, col] * k_cumdecay[row])
        // For fixed col, state[0,col], state[1,col], ... = state_smem[col*128], state_smem[col*128+1], ...
        // These ARE contiguous! Can use half2.
        for (int col = warp_id; col < HEAD_DIM; col += NUM_WARPS) {
            half2 sum_h2 = __float2half2_rn(0.0f);
            half2 * state_col = (half2 *)(&state_smem[col * HEAD_DIM]);
            half2 * vec_h2 = (half2 *)sKCumdecay_fp16;

            #pragma unroll 2
            for (int row = lane_id; row < HEAD_DIM / 2; row += WARP_SIZE) {
                sum_h2 = __hfma2(state_col[row], vec_h2[row], sum_h2);
            }

            float sum = __half2float(sum_h2.x) + __half2float(sum_h2.y);
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                sVPrime[col] = sum;
            }
        }
        __syncthreads();

        // 5. v_new = v_beta - v_prime
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // 6. attn_score = dot(k, q) in FP32
        if (warp_id == 0) {
            float sum = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum += sKBeta[i] * __half2float(sQ_fp16[i]) / decay;
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                shared_attn_score = sum;
            }
        }
        __syncthreads();

        // 7. output = attn_inter + v_attn
        // attn_inter[col] = sum_row(state[row, col] * q_scaled[row])
        // Same pattern as v_prime - columns are contiguous!
        for (int col = warp_id; col < HEAD_DIM; col += NUM_WARPS) {
            half2 sum_h2 = __float2half2_rn(0.0f);
            half2 * state_col = (half2 *)(&state_smem[col * HEAD_DIM]);
            half2 * vec_h2 = (half2 *)sQ_fp16;

            #pragma unroll 2
            for (int row = lane_id; row < HEAD_DIM / 2; row += WARP_SIZE) {
                sum_h2 = __hfma2(state_col[row], vec_h2[row], sum_h2);
            }

            float sum = __half2float(sum_h2.x) + __half2float(sum_h2.y);
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                float v_attn = sVNew[col] * shared_attn_score;
                sOut[col] = sum + v_attn;
            }
        }
        __syncthreads();

        // 8. Update state: state_new = decay * state + outer(k, v_new)
        // state[row, col] = decay * state[row, col] + k[row] * v_new[col]
        half decay_h = __float2half(fminf(fmaxf(decay, 0.0f), 10.0f));

        for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
            int col = i / HEAD_DIM;
            int row = i % HEAD_DIM;

            half state_val = state_smem[row + col * HEAD_DIM];
            half k_val = sK_fp16[row];
            half v_new_h = __float2half(sVNew[col]);

            half new_val = __hfma(decay_h, state_val, __hmul(k_val, v_new_h));

            float new_val_f = __half2float(new_val);
            new_val_f = fminf(fmaxf(new_val_f, -1e4f), 1e4f);
            state_smem[row + col * HEAD_DIM] = __float2half(new_val_f);
        }
        __syncthreads();

        // 9. Write output
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            out_base[t * out_token_stride + i] = sOut[i];
        }
        __syncthreads();
    }

    // Write final state DIRECTLY (no transpose needed - same layout)
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        state_dst[i] = __half2float(state_smem[i]);
    }
}

#endif // !defined(GGML_USE_HIP)

// Blackwell kernel (SM 12.0+): Full 64KB state in shared memory
#if !defined(GGML_USE_HIP)

template <int HEAD_DIM>
__global__ __launch_bounds__(256, 1)  // 256 threads, 1 block per SM for max shared mem
void delta_net_blackwell_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta_in,
    const float * __restrict__ state_in,
    float * __restrict__ dst,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps)
{
    static_assert(HEAD_DIM == 128, "Blackwell kernel optimized for HEAD_DIM=128");

    // One block per (batch, head) - NO column splitting!
    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = 8;  // 256 / 32

    // Strides (column-major)
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // Pointers
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory: 64KB state + 4.5KB vectors + scratch
    extern __shared__ char smem_raw[];
    float * state_smem = (float *)smem_raw;
    float * sQ = (float *)(smem_raw + HEAD_DIM * HEAD_DIM * sizeof(float));
    float * sK = sQ + HEAD_DIM;
    float * sV = sK + HEAD_DIM;
    float * sKBeta = sV + HEAD_DIM;
    float * sVBeta = sKBeta + HEAD_DIM;
    float * sKCumdecay = sVBeta + HEAD_DIM;
    float * sVPrime = sKCumdecay + HEAD_DIM;
    float * sVNew = sVPrime + HEAD_DIM;
    float * sOut = sVNew + HEAD_DIM;

    float * warp_scratch = sOut + HEAD_DIM;
    __shared__ float shared_decay, shared_beta, shared_attn_score, shared_q_norm, shared_k_norm;
    const float scale = rsqrtf((float)HEAD_DIM);

    // Load state (transposed for coalesced access)
    #pragma unroll 8
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        int col = i / HEAD_DIM, row = i % HEAD_DIM;
        state_smem[row + col * HEAD_DIM] = state_src[col + row * HEAD_DIM];
    }
    __syncthreads();

    for (int64_t t = 0; t < n_tokens; t++) {
        // Load Q, K, V and compute norms
        float q_sq_local = 0.0f, k_sq_local = 0.0f;
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            float q_val = q_ptr[t * qkv_stride_token + i];
            float k_val = k_ptr[t * qkv_stride_token + i];
            sQ[i] = q_val;
            sK[i] = k_val;
            sV[i] = v_ptr[t * qkv_stride_token + i];
            q_sq_local += q_val * q_val;
            k_sq_local += k_val * k_val;
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq_local += __shfl_xor_sync(0xffffffff, q_sq_local, offset);
            k_sq_local += __shfl_xor_sync(0xffffffff, k_sq_local, offset);
        }

        if (lane_id == 0) {
            warp_scratch[warp_id * 2] = q_sq_local;
            warp_scratch[warp_id * 2 + 1] = k_sq_local;
        }
        __syncthreads();

        if (tid == 0) {
            float total_q = 0.0f, total_k = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                total_q += warp_scratch[w * 2];
                total_k += warp_scratch[w * 2 + 1];
            }
            shared_q_norm = rsqrtf(total_q + eps);
            shared_k_norm = rsqrtf(total_k + eps);
            shared_decay = expf(fminf(g_ptr[t], 50.0f));
            shared_beta = sigmoid_f(beta_ptr[t]);
        }
        __syncthreads();

        float q_norm = shared_q_norm;
        float k_norm = shared_k_norm;
        float decay = shared_decay;
        float beta_val = shared_beta;

        // Normalize and prepare vectors
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sQ[i] = sQ[i] * q_norm * scale;
            sK[i] = sK[i] * k_norm;
            sKBeta[i] = sK[i];
            sVBeta[i] = sV[i] * beta_val;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }
        __syncthreads();

        // v_prime = state @ k_cumdecay
        for (int col = warp_id; col < HEAD_DIM; col += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int row = lane_id; row < HEAD_DIM; row += WARP_SIZE) {
                sum += state_smem[row + col * HEAD_DIM] * sKCumdecay[row];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) sVPrime[col] = sum;
        }
        __syncthreads();

        // v_new = v_beta - v_prime
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // attn_score = dot(K, Q)
        if (warp_id == 0) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum += sK[i] * sQ[i];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) shared_attn_score = sum;
        }
        __syncthreads();

        float attn_score = shared_attn_score;

        // output = (state @ q*decay) + v_new * attn_score
        for (int col = warp_id; col < HEAD_DIM; col += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int row = lane_id; row < HEAD_DIM; row += WARP_SIZE) {
                sum += state_smem[row + col * HEAD_DIM] * sQ[row] * decay;
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) sOut[col] = sum + sVNew[col] * attn_score;
        }
        __syncthreads();

        // Update state: state_new = decay * state + outer(v_new, k)
        float safe_decay = (isnan(decay) || isinf(decay)) ? 1.0f : decay;
        for (int col = tid; col < HEAD_DIM; col += blockDim.x) {
            float v_col = sVNew[col];
            for (int row = 0; row < HEAD_DIM; row++) {
                float old_state = state_smem[row + col * HEAD_DIM];
                float new_state = safe_decay * old_state + v_col * sKBeta[row];
                state_smem[row + col * HEAD_DIM] = fminf(fmaxf(new_state, -1e6f), 1e6f);
            }
        }
        __syncthreads();

        // Write output
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            out_base[t * out_token_stride + i] = sOut[i];
        }
        __syncthreads();
    }

    // Write final state (transpose back)
    #pragma unroll 8
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        int col = i / HEAD_DIM, row = i % HEAD_DIM;
        state_dst[col + row * HEAD_DIM] = state_smem[row + col * HEAD_DIM];
    }
}

// Blackwell V2: Bank-conflict-free with padded layout (128→132)
template <int HEAD_DIM>
__global__ __launch_bounds__(256, 1)
void delta_net_blackwell_optimized_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta_in,
    const float * __restrict__ state_in,
    float * __restrict__ dst,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps)
{
    static_assert(HEAD_DIM == 128, "Optimized kernel for HEAD_DIM=128");
    constexpr int PADDED_DIM = HEAD_DIM + 4;  // Bank conflict elimination

    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = 8;

    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory: 67.5KB padded state + 4.5KB vectors
    extern __shared__ char smem_raw[];
    float * state_smem = (float *)smem_raw;
    float * sQ = (float *)(smem_raw + HEAD_DIM * PADDED_DIM * sizeof(float));
    float * sK = sQ + HEAD_DIM;
    float * sV = sK + HEAD_DIM;
    float * sKBeta = sV + HEAD_DIM;
    float * sVBeta = sKBeta + HEAD_DIM;
    float * sKCumdecay = sVBeta + HEAD_DIM;
    float * sVPrime = sKCumdecay + HEAD_DIM;
    float * sVNew = sVPrime + HEAD_DIM;
    float * sOut = sVNew + HEAD_DIM;
    float * warp_scratch = sOut + HEAD_DIM;
    __shared__ float shared_decay, shared_beta, shared_attn_score, shared_q_norm, shared_k_norm;
    const float scale = rsqrtf((float)HEAD_DIM);

    // Load state with padding
    #pragma unroll 8
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        int col = i / HEAD_DIM, row = i % HEAD_DIM;
        state_smem[row + col * PADDED_DIM] = state_src[row + col * HEAD_DIM];
    }
    __syncthreads();

    for (int64_t t = 0; t < n_tokens; t++) {
        // Load Q, K, V (vectorized)
        float q_sq_local = 0.0f, k_sq_local = 0.0f;
        const float4 * q_ptr_v = (const float4 *)(q_ptr + t * qkv_stride_token);
        const float4 * k_ptr_v = (const float4 *)(k_ptr + t * qkv_stride_token);
        const float4 * v_ptr_v = (const float4 *)(v_ptr + t * qkv_stride_token);

        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM / 4; i += blockDim.x) {
            float4 q_val = q_ptr_v[i];
            float4 k_val = k_ptr_v[i];
            float4 v_val = v_ptr_v[i];
            int base = i * 4;
            sQ[base] = q_val.x; sQ[base+1] = q_val.y; sQ[base+2] = q_val.z; sQ[base+3] = q_val.w;
            sK[base] = k_val.x; sK[base+1] = k_val.y; sK[base+2] = k_val.z; sK[base+3] = k_val.w;
            sV[base] = v_val.x; sV[base+1] = v_val.y; sV[base+2] = v_val.z; sV[base+3] = v_val.w;
            q_sq_local += q_val.x*q_val.x + q_val.y*q_val.y + q_val.z*q_val.z + q_val.w*q_val.w;
            k_sq_local += k_val.x*k_val.x + k_val.y*k_val.y + k_val.z*k_val.z + k_val.w*k_val.w;
        }

        // Warp reduction for norms
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq_local += __shfl_xor_sync(0xffffffff, q_sq_local, offset);
            k_sq_local += __shfl_xor_sync(0xffffffff, k_sq_local, offset);
        }

        // Cross-warp reduction using shared memory
        if (lane_id == 0) {
            warp_scratch[warp_id * 2] = q_sq_local;
            warp_scratch[warp_id * 2 + 1] = k_sq_local;
        }
        __syncthreads();

        if (tid == 0) {
            float total_q = 0.0f, total_k = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                total_q += warp_scratch[w * 2];
                total_k += warp_scratch[w * 2 + 1];
            }
            shared_q_norm = rsqrtf(total_q + eps);
            shared_k_norm = rsqrtf(total_k + eps);
            shared_decay = expf(fminf(g_ptr[t], 50.0f));
            shared_beta = sigmoid_f(beta_ptr[t]);
        }
        __syncthreads();

        float q_norm = shared_q_norm, k_norm = shared_k_norm;
        float decay = shared_decay, beta_val = shared_beta;

        // Normalize vectors
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sQ[i] = sQ[i] * q_norm * scale;
            sK[i] = sK[i] * k_norm;
            sKBeta[i] = sK[i];
            sVBeta[i] = sV[i] * beta_val;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }
        __syncthreads();

        // v_prime = state @ k_cumdecay
        for (int row_out = warp_id; row_out < HEAD_DIM; row_out += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
                sum += state_smem[row_out + col * PADDED_DIM] * sKCumdecay[col];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) sVPrime[row_out] = sum;
        }
        __syncthreads();

        // v_new = v_beta - v_prime
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // attn_score = dot(K, Q)
        if (warp_id == 0) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum += sK[i] * sQ[i];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) shared_attn_score = sum;
        }
        __syncthreads();

        float attn_score = shared_attn_score;

        // output = (state @ q*decay) + v_new * attn_score
        for (int row_out = warp_id; row_out < HEAD_DIM; row_out += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
                sum += state_smem[row_out + col * PADDED_DIM] * sQ[col] * decay;
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) sOut[row_out] = sum + sVNew[row_out] * attn_score;
        }
        __syncthreads();

        // State update
        #pragma unroll 4
        for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
            int col = i / HEAD_DIM, row = i % HEAD_DIM;
            float old_state = state_smem[row + col * PADDED_DIM];
            float new_state = decay * old_state + sKBeta[row] * sVNew[col];
            state_smem[row + col * PADDED_DIM] = fminf(fmaxf(new_state, -1e6f), 1e6f);
        }
        __syncthreads();

        // Write output (vectorized)
        float4 * out_ptr_v = (float4 *)(out_base + t * out_token_stride);
        #pragma unroll 2
        for (int i = tid; i < HEAD_DIM / 4; i += blockDim.x) {
            int base = i * 4;
            float4 out_val = {sOut[base], sOut[base+1], sOut[base+2], sOut[base+3]};
            out_ptr_v[i] = out_val;
        }
        __syncthreads();
    }

    // Write final state (remove padding)
    #pragma unroll 8
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += blockDim.x) {
        int col = i / HEAD_DIM, row = i % HEAD_DIM;
        state_dst[row + col * HEAD_DIM] = state_smem[row + col * PADDED_DIM];
    }
}

#endif // !defined(GGML_USE_HIP)

// Multi-block column-parallel kernel (pre-Blackwell fallback)
// Each block handles COLS_PER_BLOCK columns of the 128x128 state
// With COLS_PER_BLOCK=16: 128/16 = 8 blocks per head, 16 heads = 128 blocks total
// State tile per block: 128 rows × 16 cols = 2048 floats = 8KB (fits in shared memory!)
template <int HEAD_DIM, int COLS_PER_BLOCK>
__global__ void delta_net_multiblock_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state concatenated
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps)
{
    static_assert(HEAD_DIM % COLS_PER_BLOCK == 0, "HEAD_DIM must be divisible by COLS_PER_BLOCK");
    constexpr int NUM_COL_GROUPS = HEAD_DIM / COLS_PER_BLOCK;

    // Decode block index: (batch_idx, head_idx, col_group)
    const int blocks_per_seq = n_heads * NUM_COL_GROUPS;
    const int batch_idx = blockIdx.x / blocks_per_seq;
    const int remaining = blockIdx.x % blocks_per_seq;
    const int head_idx = remaining / NUM_COL_GROUPS;
    const int col_group = remaining % NUM_COL_GROUPS;
    const int col_start = col_group * COLS_PER_BLOCK;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = 8;

    // Strides (column-major)
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // Pointers
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory layout:
    // - State tile: HEAD_DIM × COLS_PER_BLOCK = 128 × 16 = 2048 floats = 8KB
    // - Full vectors: K, KCumdecay, Q (need all HEAD_DIM elements) = 3 × 128 = 1.5KB
    // - Local vectors: V, VBeta, VPrime, VNew, Out (only COLS_PER_BLOCK) = 5 × 16 = 320 bytes
    // - Norms: 2 floats
    // Total: ~10KB (excellent for occupancy!)
    extern __shared__ float smem[];

    // State tile in shared memory: state_tile[row + local_col * HEAD_DIM]
    // local_col ∈ [0, COLS_PER_BLOCK), global_col = col_start + local_col
    float * state_tile = smem;                                  // HEAD_DIM * COLS_PER_BLOCK

    // Full vectors (need all HEAD_DIM for matrix-vector and dot products)
    float * sK = state_tile + HEAD_DIM * COLS_PER_BLOCK;        // HEAD_DIM
    float * sKCumdecay = sK + HEAD_DIM;                         // HEAD_DIM
    float * sQ = sKCumdecay + HEAD_DIM;                         // HEAD_DIM

    // Local vectors (only need COLS_PER_BLOCK elements)
    float * sV = sQ + HEAD_DIM;                                 // COLS_PER_BLOCK
    float * sVBeta = sV + COLS_PER_BLOCK;                       // COLS_PER_BLOCK
    float * sVPrime = sVBeta + COLS_PER_BLOCK;                  // COLS_PER_BLOCK
    float * sVNew = sVPrime + COLS_PER_BLOCK;                   // COLS_PER_BLOCK
    float * sOut = sVNew + COLS_PER_BLOCK;                      // COLS_PER_BLOCK
    float * sNorm = sOut + COLS_PER_BLOCK;                      // 2

    __shared__ float shared_decay, shared_beta, shared_attn_score;

    const float scale = rsqrtf((float)HEAD_DIM);

    // Load initial state tile from global to shared
    // state_tile[row + local_col * HEAD_DIM] = state[row, col_start + local_col]
    for (int i = tid; i < HEAD_DIM * COLS_PER_BLOCK; i += blockDim.x) {
        int row = i % HEAD_DIM;
        int local_col = i / HEAD_DIM;
        int global_col = col_start + local_col;
        state_tile[row + local_col * HEAD_DIM] = state_src[row + global_col * HEAD_DIM];
    }
    __syncthreads();

    // Process each token
    for (int64_t t = 0; t < n_tokens; t++) {
        // Reset norms
        if (tid < 2) {
            sNorm[tid] = 0.0f;
        }
        __syncthreads();

        // 1. Load full K, Q (all HEAD_DIM elements - needed for matrix-vector and attn_score)
        float q_sq_local = 0.0f, k_sq_local = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            float q_val = q_ptr[t * qkv_stride_token + i];
            float k_val = k_ptr[t * qkv_stride_token + i];
            sQ[i] = q_val;
            sK[i] = k_val;
            q_sq_local += q_val * q_val;
            k_sq_local += k_val * k_val;
        }

        // Load V for our columns only
        for (int i = tid; i < COLS_PER_BLOCK; i += blockDim.x) {
            sV[i] = v_ptr[t * qkv_stride_token + col_start + i];
        }

        // Warp reduction for norms
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq_local += __shfl_xor_sync(0xffffffff, q_sq_local, offset);
            k_sq_local += __shfl_xor_sync(0xffffffff, k_sq_local, offset);
        }
        if (lane_id == 0) {
            atomicAdd(&sNorm[0], q_sq_local);
            atomicAdd(&sNorm[1], k_sq_local);
        }
        __syncthreads();

        float q_norm = rsqrtf(sNorm[0] + eps);
        float k_norm = rsqrtf(sNorm[1] + eps);

        // 2. Load g, beta and normalize vectors
        if (tid == 0) {
            shared_decay = expf(fminf(g_ptr[t], 50.0f));  // Clamp g to prevent overflow
            shared_beta = sigmoid_f(beta_ptr[t]);
        }
        __syncthreads();

        float decay = shared_decay;
        float beta_val = shared_beta;

        // Normalize and compute KCumdecay
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            sQ[i] = sQ[i] * q_norm * scale;
            sK[i] = sK[i] * k_norm;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }

        // Compute VBeta for our columns
        for (int i = tid; i < COLS_PER_BLOCK; i += blockDim.x) {
            sVBeta[i] = sV[i] * beta_val;
        }
        __syncthreads();

        // 3. Compute v_prime for our columns: v_prime[local_col] = sum_row(state_tile[row, local_col] * k_cumdecay[row])
        // Each warp handles one local column
        for (int local_col = warp_id; local_col < COLS_PER_BLOCK; local_col += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int row = lane_id; row < HEAD_DIM; row += WARP_SIZE) {
                sum += state_tile[row + local_col * HEAD_DIM] * sKCumdecay[row];
            }
            // Warp reduction
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                sVPrime[local_col] = sum;
            }
        }
        __syncthreads();

        // 4. Compute v_new for our columns
        for (int i = tid; i < COLS_PER_BLOCK; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // 5. Compute attn_score = dot(k, q) - all blocks compute this redundantly
        if (warp_id == 0) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                sum += sK[i] * sQ[i];
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                shared_attn_score = sum;
            }
        }
        __syncthreads();

        // 6. Compute output for our columns: out[local_col] = attn_inter + v_attn
        // attn_inter[local_col] = sum_row(state_tile[row, local_col] * q_scaled[row])
        for (int local_col = warp_id; local_col < COLS_PER_BLOCK; local_col += NUM_WARPS) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int row = lane_id; row < HEAD_DIM; row += WARP_SIZE) {
                sum += state_tile[row + local_col * HEAD_DIM] * sQ[row] * decay;
            }
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                float v_attn = sVNew[local_col] * shared_attn_score;
                sOut[local_col] = sum + v_attn;
            }
        }
        __syncthreads();

        // 7. Update state tile: state_new[row, local_col] = decay * state[row, local_col] + v_new[row] * k[local_col]
        // Fixed: outer product orientation matches decomposed: state[v_idx, k_idx] += v_new[v_idx] * k[k_idx]
        float safe_decay = fminf(fmaxf(decay, 0.0f), 10.0f);
        for (int i = tid; i < HEAD_DIM * COLS_PER_BLOCK; i += blockDim.x) {
            int row = i % HEAD_DIM;
            int local_col = i / HEAD_DIM;

            float state_val = state_tile[row + local_col * HEAD_DIM];
            // Fix: v_new[row=v_idx] * k[local_col=k_idx] to match decomposed
            float new_val = safe_decay * state_val + sVNew[row] * sK[local_col];
            new_val = fminf(fmaxf(new_val, -1e6f), 1e6f);
            state_tile[row + local_col * HEAD_DIM] = new_val;
        }
        __syncthreads();

        // 8. Write output for our columns
        for (int i = tid; i < COLS_PER_BLOCK; i += blockDim.x) {
            int global_col = col_start + i;
            out_base[t * out_token_stride + global_col] = sOut[i];
        }
        __syncthreads();
    }

    // Write final state tile back to global
    for (int i = tid; i < HEAD_DIM * COLS_PER_BLOCK; i += blockDim.x) {
        int row = i % HEAD_DIM;
        int local_col = i / HEAD_DIM;
        int global_col = col_start + local_col;
        state_dst[row + global_col * HEAD_DIM] = state_tile[row + local_col * HEAD_DIM];
    }
}

// Dispatch function
// device_id and cc (compute capability) are passed from caller to avoid CUDA runtime API calls
static void delta_net_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const float eps,
    const int device_id,
    const int cc,  // compute capability (e.g., 890 for SM 8.9, 1200 for SM 12.0)
    cudaStream_t stream)
{
    const int64_t output_offset = head_dim * n_tokens * n_heads * n_seqs;

    // One block per (batch, head) pair
    const int num_blocks = n_seqs * n_heads;
    const int threads_per_block = 256;

    // Shared memory: 9 * head_dim (for Q, K, V, KBeta, VBeta, Out, KCumdecay, VPrime, VNew)
    // Plus 6 floats for Norm[2], g_val, beta_val, decay, attn_score
    const size_t smem_size = (9 * head_dim + 6) * sizeof(float);

    // Use templated kernel for common head dimensions, generic for others
    if (head_dim == 64) {
        delta_net_recurrent_f32<64><<<num_blocks, threads_per_block, smem_size, stream>>>(
            q, k, v, g, beta, state_in, dst, n_tokens, n_heads, n_seqs, output_offset, eps);
    } else if (head_dim == 128) {
#if !defined(GGML_USE_HIP)
        // Check for Blackwell (SM 12.0+) which has 228KB shared memory
        // cc is in format MAJOR*100 + MINOR*10 (e.g., 890 for 8.9, 1200 for 12.0)
        const int sm_major = cc / 100;

        if (sm_major >= 12) {
            // Blackwell path: single block per head with FULL state in shared memory
            const int blackwell_num_blocks = n_seqs * n_heads;
            const int blackwell_threads = 256;

            // Shared memory calculation with explicit breakdown:
            // - State matrix: HEAD_DIM × HEAD_DIM × sizeof(float) = 128×128×4 = 65536 bytes (64KB)
            // - Vectors (Q,K,V,KBeta,VBeta,KCumdecay,VPrime,VNew,Out): 9 × HEAD_DIM × sizeof(float) = 4608 bytes
            // - Warp scratch: 16 × sizeof(float) = 64 bytes
            // Total: 65536 + 4608 + 64 = 70208 bytes (~68.6KB)
            // __shared__ scalars (decay, beta, etc.) are static, not dynamic
            constexpr size_t state_bytes = 128 * 128 * sizeof(float);       // 64KB
            constexpr size_t vector_bytes = 9 * 128 * sizeof(float);         // 4.5KB
            constexpr size_t warp_scratch_bytes = 16 * sizeof(float);        // 64B
            constexpr size_t blackwell_smem_size = state_bytes + vector_bytes + warp_scratch_bytes;

            static_assert(blackwell_smem_size == 70208, "Shared memory size mismatch");

            // A/B comparison mode (set GGML_CUDA_DELTA_NET_AB=1)
            static const bool ab_mode = []() {
                const char* env = std::getenv("GGML_CUDA_DELTA_NET_AB");
                if (env != nullptr) {
                    fprintf(stderr, "[DELTA_NET] A/B comparison mode ENABLED\n");
                    return true;
                }
                return false;
            }();

            if (ab_mode) {
                // A/B mode: run both kernels and compare outputs
                const int64_t total_output_size = output_offset + head_dim * head_dim * n_heads * n_seqs;

                // Allocate temp buffer for recurrent kernel output
                float * temp_dst = nullptr;
                CUDA_CHECK(cudaMallocAsync(&temp_dst, total_output_size * sizeof(float), stream));

                // Run recurrent kernel (reference) to temp buffer
                delta_net_recurrent_f32<128><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, temp_dst, n_tokens, n_heads, n_seqs, output_offset, eps);

                // Request extended shared memory for Blackwell
                CUDA_CHECK(cudaFuncSetAttribute(
                    delta_net_blackwell_f32<128>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    blackwell_smem_size));

                // Run Blackwell kernel to dst
                delta_net_blackwell_f32<128><<<blackwell_num_blocks, blackwell_threads, blackwell_smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_tokens, n_heads, n_seqs, output_offset, eps);

                // Sync to ensure both kernels complete
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Copy results back to host for comparison
                const int64_t output_elements = head_dim * n_tokens * n_heads * n_seqs;
                const int64_t state_elements = head_dim * head_dim * n_heads * n_seqs;

                std::vector<float> ref_output(output_elements);
                std::vector<float> ref_state(state_elements);
                std::vector<float> bw_output(output_elements);
                std::vector<float> bw_state(state_elements);

                CUDA_CHECK(cudaMemcpy(ref_output.data(), temp_dst, output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ref_state.data(), temp_dst + output_offset, state_elements * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(bw_output.data(), dst, output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(bw_state.data(), dst + output_offset, state_elements * sizeof(float), cudaMemcpyDeviceToHost));

                // Compare outputs
                float max_out_diff = 0.0f;
                int64_t max_out_idx = 0;
                for (int64_t i = 0; i < output_elements; i++) {
                    float diff = fabsf(ref_output[i] - bw_output[i]);
                    if (diff > max_out_diff) {
                        max_out_diff = diff;
                        max_out_idx = i;
                    }
                }

                // Compare states
                float max_state_diff = 0.0f;
                int64_t max_state_idx = 0;
                for (int64_t i = 0; i < state_elements; i++) {
                    float diff = fabsf(ref_state[i] - bw_state[i]);
                    if (diff > max_state_diff) {
                        max_state_diff = diff;
                        max_state_idx = i;
                    }
                }

                // Report results
                static int ab_call_count = 0;
                ab_call_count++;
                fprintf(stderr, "[DELTA_NET A/B #%d] n_tokens=%lld output_diff=%e (idx=%lld ref=%e bw=%e) state_diff=%e (idx=%lld ref=%e bw=%e)\n",
                    ab_call_count,
                    (long long)n_tokens,
                    max_out_diff, (long long)max_out_idx, ref_output[max_out_idx], bw_output[max_out_idx],
                    max_state_diff, (long long)max_state_idx, ref_state[max_state_idx], bw_state[max_state_idx]);

                // Report first 4 output values for head 0
                if (ab_call_count <= 10) {
                    fprintf(stderr, "  ref_out[0:3]=[%e,%e,%e,%e] bw_out[0:3]=[%e,%e,%e,%e]\n",
                        ref_output[0], ref_output[1], ref_output[2], ref_output[3],
                        bw_output[0], bw_output[1], bw_output[2], bw_output[3]);
                    fprintf(stderr, "  ref_state[0,1,128,129]=[%e,%e,%e,%e] bw_state=[%e,%e,%e,%e]\n",
                        ref_state[0], ref_state[1], ref_state[128], ref_state[129],
                        bw_state[0], bw_state[1], bw_state[128], bw_state[129]);
                }

                CUDA_CHECK(cudaFreeAsync(temp_dst, stream));
            } else {
                // Normal mode: just run Blackwell kernel
                // Request extended shared memory for Blackwell
                CUDA_CHECK(cudaFuncSetAttribute(
                    delta_net_blackwell_f32<128>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    blackwell_smem_size));

                delta_net_blackwell_f32<128><<<blackwell_num_blocks, blackwell_threads, blackwell_smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_tokens, n_heads, n_seqs, output_offset, eps);
            }
        } else
#endif // !defined(GGML_USE_HIP)
        {
            // Pre-Blackwell path: Use recurrent kernel
            delta_net_recurrent_f32<128><<<num_blocks, threads_per_block, smem_size, stream>>>(
                q, k, v, g, beta, state_in, dst, n_tokens, n_heads, n_seqs, output_offset, eps);
        }
    } else {
        delta_net_recurrent_generic_f32<<<num_blocks, threads_per_block, smem_size, stream>>>(
            q, k, v, g, beta, state_in, dst, head_dim, n_tokens, n_heads, n_seqs, output_offset, eps);
    }

    // Check for errors (but don't sync during graph capture)
    CUDA_CHECK(cudaGetLastError());

#ifdef GGML_CUDA_DEBUG_SYNC
    // Only sync when not capturing CUDA graphs
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g
    const ggml_tensor * src4 = dst->src[4];  // beta
    const ggml_tensor * src5 = dst->src[5];  // state

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int64_t n_heads = src0->ne[2];
    const int64_t n_seqs = src0->ne[3];

    // Dimension validation
    // Q/K: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src1->ne[0] == head_dim && src1->ne[1] == n_tokens && src1->ne[2] == n_heads && src1->ne[3] == n_seqs);
    // V: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src2->ne[0] == head_dim && src2->ne[1] == n_tokens && src2->ne[2] == n_heads && src2->ne[3] == n_seqs);
    // G: [n_tokens, 1, n_heads, n_seqs]
    GGML_ASSERT(src3->ne[0] == n_tokens && src3->ne[1] == 1 && src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
    // Beta: [1, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src4->ne[0] == 1 && src4->ne[1] == n_tokens && src4->ne[2] == n_heads && src4->ne[3] == n_seqs);
    // State: [head_dim, head_dim*n_heads, 1, n_seqs]
    GGML_ASSERT(src5->ne[0] == head_dim && src5->ne[1] == head_dim * n_heads && src5->ne[2] == 1 && src5->ne[3] == n_seqs);

    // Verify output tensor size
    const int64_t output_size = head_dim * n_tokens * n_heads * n_seqs;
    const int64_t state_size = head_dim * head_dim * n_heads * n_seqs;
    GGML_ASSERT(ggml_nelements(dst) == output_size + state_size);

    const float eps = 1e-6f;

    GGML_ASSERT(head_dim <= 256);  // Reasonable limit for shared memory

    // Get device info from ctx (avoids calling CUDA runtime APIs inside dispatch)
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;

    delta_net_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        head_dim, n_tokens, n_heads, n_seqs, eps,
        device_id, cc,
        ctx.stream());

}
