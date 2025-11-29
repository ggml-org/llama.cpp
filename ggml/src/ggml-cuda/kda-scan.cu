#include "kda-scan.cuh"

// KDA (Kimi Delta Attention) scan CUDA kernel
// Recurrence:
//   h[t] = exp(g[t]) * h[t-1] + k[t]^T * (beta[t] * (v[t] - h[t-1] @ k[t]))
//   o[t] = q[t]^T @ h[t]
// 
// This kernel uses global memory for the hidden state to avoid shared memory limits.
// Each block processes one head for one sequence.

__global__ void kda_scan_f32_kernel(
    const float * __restrict__ src0,   // h:    [head_dim, head_dim, n_head, n_seqs+]
    const float * __restrict__ src1,   // q:    [head_dim, n_head, n_seq_tokens, n_seqs]
    const float * __restrict__ src2,   // k:    [head_dim, n_head, n_seq_tokens, n_seqs]
    const float * __restrict__ src3,   // v:    [head_dim, n_head, n_seq_tokens, n_seqs]
    const float * __restrict__ src4,   // g:    [head_dim, n_head, n_seq_tokens, n_seqs]
    const float * __restrict__ src5,   // beta: [n_head, n_seq_tokens, n_seqs]
    const int32_t * __restrict__ src6, // ids:  [n_seqs]
    float * __restrict__ dst,
    const int64_t head_dim,
    const int64_t n_head,
    const int64_t n_seq_tokens,
    const int64_t n_seqs,
    const int64_t y_off)  // offset to state output in dst (in floats)
{
    // Each block handles one head for one sequence
    const int seq_idx = blockIdx.x / n_head;
    const int head_idx = blockIdx.x % n_head;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    
    if (seq_idx >= n_seqs || head_idx >= n_head) return;
    
    // Get sequence ID for initial state
    const int src_seq = src6[seq_idx];
    
    // Shared memory for temporary buffers
    extern __shared__ float smem[];
    float * hk_buf = smem;                    // [head_dim] - h @ k buffer
    float * q_norm = smem + head_dim;         // [head_dim] - normalized q
    float * k_norm = q_norm + head_dim;       // [head_dim] - normalized k
    float * warp_sums = k_norm + head_dim;    // [64] - for reductions
    
    // Pointers to input/output data for this head
    const int64_t h_stride_head = head_dim * head_dim;
    const int64_t h_stride_seq = h_stride_head * n_head;
    const int64_t qkv_stride_head = head_dim;
    const int64_t qkv_stride_token = head_dim * n_head;
    const int64_t qkv_stride_seq = qkv_stride_token * n_seq_tokens;
    const int64_t beta_stride_token = n_head;
    const int64_t beta_stride_seq = beta_stride_token * n_seq_tokens;
    
    const float * h_in = src0 + src_seq * h_stride_seq + head_idx * h_stride_head;
    float * h_out = dst + y_off + seq_idx * h_stride_seq + head_idx * h_stride_head;
    float * y_out = dst + seq_idx * qkv_stride_seq + head_idx * qkv_stride_head;
    
    // Copy initial state to output (we'll update in place)
    for (int i = tid; i < head_dim * head_dim; i += n_threads) {
        float val = h_in[i];
        if (!isfinite(val) || fabsf(val) > 1e6f) {
            val = 0.0f;
        }
        h_out[i] = val;
    }
    __syncthreads();
    
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    // Process each token sequentially
    for (int t = 0; t < n_seq_tokens; ++t) {
        const float * q_raw = src1 + t * qkv_stride_token + seq_idx * qkv_stride_seq + head_idx * qkv_stride_head;
        const float * k_raw = src2 + t * qkv_stride_token + seq_idx * qkv_stride_seq + head_idx * qkv_stride_head;
        const float * v = src3 + t * qkv_stride_token + seq_idx * qkv_stride_seq + head_idx * qkv_stride_head;
        const float * g = src4 + t * qkv_stride_token + seq_idx * qkv_stride_seq + head_idx * qkv_stride_head;
        const float beta = src5[t * beta_stride_token + seq_idx * beta_stride_seq + head_idx];
        float * y = y_out + t * qkv_stride_token;
        
        // Step 1: L2 normalize q and k
        float q_sq_sum = 0.0f, k_sq_sum = 0.0f;
        for (int i = tid; i < head_dim; i += n_threads) {
            q_sq_sum += q_raw[i] * q_raw[i];
            k_sq_sum += k_raw[i] * k_raw[i];
        }
        
        // Warp reduction
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            q_sq_sum += __shfl_down_sync(0xffffffff, q_sq_sum, offset);
            k_sq_sum += __shfl_down_sync(0xffffffff, k_sq_sum, offset);
        }
        
        // Cross-warp reduction
        int warp_id = tid / warpSize;
        int lane_id = tid % warpSize;
        if (lane_id == 0 && warp_id < 32) {
            warp_sums[warp_id] = q_sq_sum;
            warp_sums[32 + warp_id] = k_sq_sum;
        }
        __syncthreads();
        
        if (tid == 0) {
            float total_q = 0.0f, total_k = 0.0f;
            for (int i = 0; i < (n_threads + warpSize - 1) / warpSize; ++i) {
                total_q += warp_sums[i];
                total_k += warp_sums[32 + i];
            }
            warp_sums[0] = rsqrtf(total_q + 1e-6f) * scale;
            warp_sums[1] = rsqrtf(total_k + 1e-6f);
        }
        __syncthreads();
        
        float q_norm_factor = warp_sums[0];
        float k_norm_factor = warp_sums[1];
        
        // Store normalized q and k
        for (int i = tid; i < head_dim; i += n_threads) {
            q_norm[i] = q_raw[i] * q_norm_factor;
            k_norm[i] = k_raw[i] * k_norm_factor;
        }
        __syncthreads();
        
        // KDA recurrence: h[t] = exp(g[t]) * h[t-1] + k[t]^T * (beta[t] * (v[t] - h[t-1] @ k[t]))
        // Apply decay first, then compute retrieval and update
        
        // Step 2: Apply decay to h: h = h * exp(g)
        for (int idx = tid; idx < head_dim * head_dim; idx += n_threads) {
            int i = idx / head_dim;
            float exp_gi = expf(g[i]);
            h_out[idx] *= exp_gi;
        }
        __syncthreads();
        
        // Step 3: Compute h^T @ k -> hk_buf
        for (int j = tid; j < head_dim; j += n_threads) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                sum += h_out[i * head_dim + j] * k_norm[i];
            }
            hk_buf[j] = sum;
        }
        __syncthreads();
        
        // Step 4: Update h: h = h + outer(k, beta * (v - hk))
        for (int idx = tid; idx < head_dim * head_dim; idx += n_threads) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            float delta_j = beta * (v[j] - hk_buf[j]);
            h_out[idx] += k_norm[i] * delta_j;
        }
        __syncthreads();
        
        // Step 5: Compute output y = h^T @ q
        for (int j = tid; j < head_dim; j += n_threads) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                sum += h_out[i * head_dim + j] * q_norm[i];
            }
            y[j] = sum;
        }
        __syncthreads();
    }
}

void ggml_cuda_op_kda_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // h
    const ggml_tensor * src1 = dst->src[1]; // q
    const ggml_tensor * src2 = dst->src[2]; // k
    const ggml_tensor * src3 = dst->src[3]; // v
    const ggml_tensor * src4 = dst->src[4]; // g
    const ggml_tensor * src5 = dst->src[5]; // beta
    const ggml_tensor * src6 = dst->src[6]; // ids
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_F32);
    GGML_ASSERT(src4->type == GGML_TYPE_F32);
    GGML_ASSERT(src5->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    
    const int64_t head_dim = src0->ne[0];
    const int64_t n_head = src1->ne[1];
    const int64_t n_seq_tokens = src1->ne[2];
    const int64_t n_seqs = src1->ne[3];
    
    // Output offset for hidden state (after attention output) - in floats
    const int64_t y_off = ggml_nelements(src1);
    
    const float * h_d = (const float *)src0->data;
    const float * q_d = (const float *)src1->data;
    const float * k_d = (const float *)src2->data;
    const float * v_d = (const float *)src3->data;
    const float * g_d = (const float *)src4->data;
    const float * beta_d = (const float *)src5->data;
    const int32_t * ids_d = (const int32_t *)src6->data;
    float * dst_d = (float *)dst->data;
    
    cudaStream_t stream = ctx.stream();
    
    // Launch kernel: one block per (sequence, head) pair
    const int n_blocks = n_seqs * n_head;
    const int n_threads = 128;
    
    // Shared memory: hk_buf[head_dim] + q_norm[head_dim] + k_norm[head_dim] + warp_sums[64]
    size_t smem_size = (3 * head_dim + 64) * sizeof(float);
    
    kda_scan_f32_kernel<<<n_blocks, n_threads, smem_size, stream>>>(
        h_d, q_d, k_d, v_d, g_d, beta_d, ids_d, dst_d,
        head_dim, n_head, n_seq_tokens, n_seqs, y_off);
}
