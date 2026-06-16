#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#include "ssm-scan.cuh"


// Minimum number of tokens to use SSD (State Space Duality) matmul path instead of scan path.
// For n_tok <= this threshold, the scan kernel is used (lower overhead for short sequences).
#define SSM_SSD_MIN_TOKENS 128

// prepare_dt kernel dimensions: one block per (head, seq), each block handles DT_MAX_ITEMS items.
#define SSM_SSD_DT_BLOCK     256
#define SSM_SSD_DT_MAX_ITEMS  32

// Maximum tokens the SSD path supports — derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)

// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.
#define SSM_SSD_CHUNK_SIZE 256

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
                 const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
                 const int32_t * src6_ptr, float * dst_ptr,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L_param)
{
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const size_t L = L_template == 0 ? L_param : L_template;
    ggml_cuda_pdl_sync();
    const float *s0_block = (const float *)((const char *)src0 + src6[blockIdx.x] * src0_nb3 + blockIdx.y * splitD * src0_nb2);
    const float *x_block = (const float *)((const char *)src1 + (blockIdx.x * src1_nb3) + blockIdx.y * splitD * sizeof(float));
    const float *dt_block = (const float *)((const char *)src2 + (blockIdx.x * src2_nb2) + blockIdx.y * splitD * sizeof(float));
    const float *A_block = (const float *)((const char *)src3 + blockIdx.y * splitD * src3_nb1);
    const float *B_block = (const float *)((const char *)src4 + (blockIdx.x * src4_nb3));
    const float *C_block = (const float *)((const char *)src5 + (blockIdx.x * src5_nb3));
    float *y_block = (float *)((char *)dst + (blockIdx.x * d_inner * L * sizeof(float)) + blockIdx.y * splitD * sizeof(float));
    float *s_block = (float *)((char *)dst + s_off + blockIdx.x * src0_nb3 + blockIdx.y * splitD * src0_nb2);

    const int stride_x = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B = src4_nb2 / sizeof(float);
    const int stride_C = src5_nb2 / sizeof(float);
    const int stride_y = d_inner;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(A_block, regA);
    BlockLoad(cub_temp_storage.load_temp).Load(s0_block, regs0);
#else
    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_A = src3_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        regA[n] = A_block[threadIdx.x * stride_A + n];
        regs0[n] = s0_block[threadIdx.x * stride_s0 + n];
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++)
    {
        if (threadIdx.x < N)
        {
            smemB[threadIdx.x] = B_block[i * stride_B + threadIdx.x];
            smemC[threadIdx.x] = C_block[i * stride_C + threadIdx.x];
        }
        __syncthreads();

        float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
        if (dt_soft_plus <= 20.0f)
        {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + threadIdx.x] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++)
        {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + threadIdx.x] = sumf;
    }

#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(s_block, regs0);
#else
    const int stride_s = stride_s0;
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        s_block[threadIdx.x * stride_s + n] = regs0[n];
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// assumes as many threads as d_state
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan_f32_group(
        const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
        const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
        const int32_t * src6_ptr, float * dst_ptr,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;

    const int head_idx =  warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    ggml_cuda_pdl_sync();
    // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the codebase
    const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];
    float state_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
        // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? log1pf(expf(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);

        state_sum = 0.0f;
        const float dA   = expf(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        // parallel accumulation for output
        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int32_t * src6, float * dst,
                              const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
                              const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
                              const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
                              const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
                              cudaStream_t stream) {
    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (src3_nb1 == sizeof(float)) {
        // Mamba-2
        if (d_state == 128) {
            constexpr int threads   = 128;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<128/WARP_SIZE, 128>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else if (d_state == 256) { // Falcon-H1
            constexpr int threads   = 256;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<256/WARP_SIZE, 256>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        constexpr int threads = 128;
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);
        const dim3 blocks(n_seq, (n_head + threads - 1) / threads, 1);
        const int  smem_size = (threads * (d_state + 1) * 2) * sizeof(float);
        if (d_state == 16) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, smem_size, stream);
            switch (n_tok)
            {
            case 1:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 1>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 2:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 2>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 3:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 3>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 4:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 4>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 5:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 5>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 6:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 6>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 7:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 7>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 8:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 8>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            default:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 0>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            }
        } else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}

// ============================================================================
// SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS)
//
// Instead of a sequential scan, SSD reformulates the output as:
//   Y = (L (.) (C @ B^T)) @ (X * dt)  +  decay * C @ s_init
// where L is a causal decay mask derived from A and dt.
//
// This converts the O(T*N) sequential scan into parallel matmuls.
// ============================================================================
// Softplus(dt) and inclusive prefix sum per head using CUB BlockScan.
// Grid: (n_head, n_seqs)
template <int BLOCK_SIZE, int MAX_ITEMS>
__global__ void ssm_ssd_prepare_dt_kernel(
        const float * __restrict__ dt_raw,
        float * __restrict__ dt_sp_out,
        float * __restrict__ cs_out,
        const int n_head, const int n_tok,
        const int dt_stride_tok,   // elements between tokens in dt
        const int dt_stride_seq) { // elements between sequences in dt

    const int h = blockIdx.x;
    const int s = blockIdx.y;

    const float * dt_seq = dt_raw + s * dt_stride_seq;

    float * dt_sp_seq = dt_sp_out + s * n_tok * n_head;
    float * cs_seq    = cs_out    + s * n_tok * n_head;

    const int items_per_thread = (n_tok + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: parallel softplus, each thread accumulates a local sum
    float local_vals[MAX_ITEMS];
    float local_sum = 0.0f;

    for (int i = 0; i < items_per_thread; i++) {
        const int t = threadIdx.x * items_per_thread + i;  // blocked distribution
        if (t < n_tok) {
            float val = dt_seq[h + t * dt_stride_tok];
            float sp = (val <= 20.0f) ? log1pf(expf(val)) : val;
            local_vals[i] = sp;
            local_sum += sp;
            dt_sp_seq[t * n_head + h] = sp;
        } else {
            local_vals[i] = 0.0f;
        }
    }

    // Phase 2: parallel prefix sum of per-thread totals
#ifdef USE_CUB
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    float thread_inclusive;
    BlockScan(scan_temp).InclusiveSum(local_sum, thread_inclusive);
    float exclusive_prefix = thread_inclusive - local_sum;
#else
    // Fallback: sequential prefix sum in shared memory
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < BLOCK_SIZE && i * items_per_thread < n_tok; i++) {
            sdata[i] += sdata[i - 1];
        }
    }
    __syncthreads();
    float exclusive_prefix = (threadIdx.x > 0) ? sdata[threadIdx.x - 1] : 0.0f;
#endif

    // Phase 3: write cumsum = exclusive_prefix + local running sum
    float running = exclusive_prefix;
    for (int i = 0; i < items_per_thread; i++) {
        const int t = threadIdx.x * items_per_thread + i;
        if (t < n_tok) {
            running += local_vals[i];
            cs_seq[t * n_head + h] = running;
        }
    }
}

// Prepare SSD matmul inputs for one chunk: X_dt, B_weighted, C_scaled.
// T_matmul controls precision for X_dt, B_weighted (float or half).
// C_scaled is always float (pairs with float s_cur in step 3c).
// Computation is always FP32; only the final store converts to T_matmul.
// Also materializes the causal M matrix = exp(A*(cs_out - cs_in)) * CB (fused with prep to save a launch).
// Grid: (ceil(max(C*head_dim, d_state*C, chunk_len^2) / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE, typename T_matmul>
__global__ void ssm_ssd_pre_matmul_kernel(
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ dt_sp,      // {n_tok, n_head} softplus(dt)
        const float * __restrict__ A,          // {1, n_head}
        const float * __restrict__ x,          // {head_dim, n_head, n_tok, n_seqs}
        const float * __restrict__ B,          // {d_state, n_group, n_tok, n_seqs}
        const float * __restrict__ C_src,      // {d_state, n_group, n_tok, n_seqs}
        T_matmul * __restrict__ X_dt,          // {head_dim, C, n_head} x * dt, d-fastest
        T_matmul * __restrict__ B_weighted,    // {d_state, C, n_head} B * decay_from_end
        float * __restrict__ C_scaled,         // {d_state, C, n_head} C * decay_to_pos (always float)
        const float * __restrict__ CB,         // {chunk_len, chunk_len, n_group, n_seqs}
        half * __restrict__ M_out,             // {chunk_len, chunk_len, n_head, n_seqs}
        const int chunk_len, const int head_dim, const int n_head, const int n_group,
        const int d_state, const int A_stride,
        const int x_stride_tok, const int x_stride_seq,
        const int B_stride_tok, const int B_stride_seq,
        const int C_stride_tok, const int C_stride_seq,
        const int chunk_offset,
        const int n_tok_total) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int g = h / (n_head / n_group);

    const float A_h = A[h * A_stride];
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;

    // Prepare X_dt = x * dt, stored d-fastest for coalesced reads and writes.
    const int n_xdt = chunk_len * head_dim;
    if (idx < n_xdt) {
        const int d = idx % head_dim;
        const int t = idx / head_dim;

        const float x_val = x[s * x_stride_seq + (chunk_offset + t) * x_stride_tok + d + h * head_dim];
        const float dt_val = dt_sp[cs_seq_off + (chunk_offset + t) * n_head + h];

        X_dt[d + t * head_dim + h * n_xdt + s * n_xdt * n_head] = (T_matmul)(x_val * dt_val);
    }

    // Prepare B_weighted = B * decay_from_end for state update matmul.
    const int n_bw = d_state * chunk_len;
    if (idx < n_bw) {
        const int n = idx % d_state;
        const int t = idx / d_state;

        const float cs_t = cs[cs_seq_off + (chunk_offset + t) * n_head + h] - cs_base;
        const float decay_from_end = __expf(A_h * (cs_last - cs_t));

        const float B_val = B[s * B_stride_seq + (chunk_offset + t) * B_stride_tok + g * d_state + n];

        B_weighted[n + t * d_state + h * n_bw + s * n_bw * n_head] = (T_matmul)(B_val * decay_from_end);
    }

    // Prepare C_scaled = C * decay_to_pos for state contribution matmul.
    const int n_cs = d_state * chunk_len;
    if (idx < n_cs) {
        const int n = idx % d_state;
        const int t = idx / d_state;

        const float cs_t = cs[cs_seq_off + (chunk_offset + t) * n_head + h] - cs_base;
        const float decay_to_pos = __expf(A_h * cs_t);

        const float C_val = C_src[s * C_stride_seq + (chunk_offset + t) * C_stride_tok + g * d_state + n];

        C_scaled[n + t * d_state + h * n_cs + s * n_cs * n_head] = C_val * decay_to_pos;
    }

    // Materialize M = exp(A*(cs_out - cs_in)) * CB with causal mask.
    const int n_M = chunk_len * chunk_len;
    if (idx < n_M) {
        const int t_out = idx % chunk_len;
        const int t_in  = idx / chunk_len;

        half val;
        if (t_in <= t_out) {
            const float cs_out = cs[cs_seq_off + (chunk_offset + t_out) * n_head + h] - cs_base;
            const float cs_in  = cs[cs_seq_off + (chunk_offset + t_in)  * n_head + h] - cs_base;
            const float decay  = __expf(A_h * (cs_out - cs_in));
            const float * CB_g = CB + (int64_t)s * chunk_len * chunk_len * n_group
                                   + (int64_t)g * chunk_len * chunk_len;
            const float cb_val = CB_g[t_out + t_in * chunk_len];
            val = __float2half(decay * cb_val);
        } else {
            val = __float2half(0.0f);
        }

        M_out[(int64_t)s * n_M * n_head + (int64_t)h * n_M + t_in * chunk_len + t_out] = val;
    }
}

// Scale running state in-place: s_cur *= decay_total(chunk).
// Called BEFORE cuBLAS state update (beta=1) to fuse inter-chunk decay.
// Eliminates the s_old buffer and D2D memcpy vs the old approach of:
//   memcpy(s_old, s_cur) -> cuBLAS(beta=0) -> s_cur += decay * s_old
// Grid: (ceil(d_state * head_dim / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_scale_state_kernel(
        float * __restrict__ s_cur,            // {d_state, head_dim, n_head, n_seqs}
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ A,          // {1, n_head}
        const int d_state, const int head_dim, const int n_head,
        const int chunk_offset, const int chunk_len,
        const int n_tok_total, const int A_stride) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int state_per_head = d_state * head_dim;
    if (idx >= state_per_head) return;

    const float A_h = A[h * A_stride];
    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;
    const float decay_total = __expf(A_h * cs_last);

    const int off = s * state_per_head * n_head + h * state_per_head + idx;
    s_cur[off] *= decay_total;
}

// Copy initial state from src0[ids[s]] into s_cur for each sequence.
// Grid: (ceil(d_state * head_dim * n_head / BLOCK), n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_init_state_kernel(
        const float * __restrict__ src0,       // {d_state, head_dim, n_head, n_rs}
        const int32_t * __restrict__ ids,      // {n_seqs}
        float * __restrict__ s_cur,            // {d_state, head_dim, n_head, n_seqs}
        const int state_size,                  // d_state * head_dim * n_head
        const int s0_stride_seq) {             // elements between state rows
    const int s = blockIdx.y;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= state_size) return;

    const float * s_src = src0 + ids[s] * s0_stride_seq;
    s_cur[s * state_size + idx] = s_src[idx];
}

// SSD (State Space Duality) dispatch for Mamba-2 prefill.
// Chunked matmuls: CB, materialize M + cuBLAS Y, S@C, B@X_dt.
// All strides are in elements (floats), not bytes.
static void ssm_scan_ssd_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, const float * src1_d, const float * src2_d, const float * src3_d,
        const float * src4_d, const float * src5_d, const int32_t * src6_d, float * dst_d,
        const int s0_stride_seq,                                       // state (src0) stride between seqs
        const int x_stride_tok,  const int x_stride_seq,               // x (src1) strides
        const int dt_stride_tok, const int dt_stride_seq,              // dt (src2) strides
        const int A_stride,                                            // A (src3) stride between heads
        const int B_stride_tok,  const int B_stride_seq,               // B (src4) strides
        const int C_stride_tok,  const int C_stride_seq,               // C (src5) strides
        const int64_t s_off, const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq) {

    cudaStream_t stream = ctx.stream();
    const int64_t d_inner = head_dim * n_head;

    const int64_t chunk_size = SSM_SSD_CHUNK_SIZE;
    const int64_t n_chunks = (n_tok + chunk_size - 1) / chunk_size;

    const int64_t state_per_head = d_state * head_dim;

    using matmul_t = half;
    static constexpr cudaDataType_t matmul_dtype = CUDA_R_16F;

    ggml_cuda_pool_alloc<float>    dt_sp_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    cs_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    CB_buf(ctx.pool(), chunk_size * chunk_size * n_group * n_seq);
    ggml_cuda_pool_alloc<matmul_t> X_dt_buf(ctx.pool(), chunk_size * head_dim * n_head * n_seq);
    ggml_cuda_pool_alloc<matmul_t> B_w_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    C_s_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    float    * dt_sp      = dt_sp_buf.get();
    float    * cs         = cs_buf.get();
    float    * CB         = CB_buf.get();
    matmul_t * X_dt       = X_dt_buf.get();
    matmul_t * B_weighted = B_w_buf.get();
    float    * C_scaled   = C_s_buf.get();
    float    * s_cur      = (float *)((char *)dst_d + s_off); // write state directly to dst

    // Step 1: softplus(dt) and parallel prefix sum over full sequence
    {
        dim3 grid(n_head, n_seq);
        ssm_ssd_prepare_dt_kernel<SSM_SSD_DT_BLOCK, SSM_SSD_DT_MAX_ITEMS><<<grid, SSM_SSD_DT_BLOCK, 0, stream>>>(
            src2_d, dt_sp, cs, n_head, n_tok, dt_stride_tok, dt_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: initialize running state from src0[ids[s]]
    {
        constexpr int BLOCK = 256;
        const int64_t state_size = d_state * head_dim * n_head;
        dim3 grid((state_size + BLOCK - 1) / BLOCK, n_seq);
        ssm_ssd_init_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
            src0_d, src6_d, s_cur, state_size, s0_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: chunked SSD loop
    // Per chunk: pre_matmul (incl. M) + 4 cuBLAS (CB, Y, S@C, state update) + scale_state
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha_one  = 1.0f;
    const float beta_zero  = 0.0f;
    const float beta_one   = 1.0f;
    const int lda_C_src = C_stride_tok;  // leading dim for C in CB = C^T @ B
    const int ldb_B_src = B_stride_tok;  // leading dim for B in CB = C^T @ B

    // Scratch buffer for causal M matrix, reused across chunks (max size at chunk_size)
    const int64_t n_M_max = chunk_size * chunk_size;
    ggml_cuda_pool_alloc<half> M_buf(ctx.pool(), n_M_max * n_head * n_seq);
    half * M_mat = M_buf.get();

    for (int64_t k = 0; k < n_chunks; k++) {
        const int64_t chunk_offset = k * chunk_size;
        const int64_t chunk_len = (chunk_offset + chunk_size <= n_tok) ? chunk_size : (n_tok - chunk_offset);

        // 3a: CB = C^T @ B per group
        for (int64_t s = 0; s < n_seq; s++) {
            const float * C_s = src5_d + s * C_stride_seq + chunk_offset * C_stride_tok;
            const float * B_s = src4_d + s * B_stride_seq + chunk_offset * B_stride_tok;
            float *      CB_s = CB + s * chunk_len * chunk_len * n_group;

            if (n_group == 1) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one, C_s, lda_C_src, B_s, ldb_B_src,
                    &beta_zero, CB_s, (int)chunk_len));
            } else {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one,
                    C_s, CUDA_R_32F, lda_C_src, d_state,
                    B_s, CUDA_R_32F, ldb_B_src, d_state,
                    &beta_zero,
                    CB_s, CUDA_R_32F, (int)chunk_len, (long long)(chunk_len * chunk_len),
                    n_group,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3b: prepare X_dt, B_weighted, C_scaled + materialize causal M matrix
        const int64_t n_M = chunk_len * chunk_len;
        {
            constexpr int BLOCK = 256;
            const int64_t n_xdt   = chunk_len * head_dim;
            const int64_t n_bw    = d_state * chunk_len;
            int64_t max_work = n_xdt;
            if (n_bw  > max_work) max_work = n_bw;
            if (n_M   > max_work) max_work = n_M;
            dim3 grid((max_work + BLOCK - 1) / BLOCK, n_head, n_seq);
            ssm_ssd_pre_matmul_kernel<BLOCK, matmul_t><<<grid, BLOCK, 0, stream>>>(
                cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                X_dt, B_weighted, C_scaled,
                CB, M_mat,
                chunk_len, head_dim, n_head, n_group, d_state, A_stride,
                x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
                chunk_offset, n_tok);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3c: dst = S_cur^T @ C_scaled (state contribution)
        {
            const int64_t stride_S  = state_per_head;
            const int64_t stride_Cs = d_state * chunk_len;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;

                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    head_dim, chunk_len, d_state,
                    &alpha_one,
                    s_cur    + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    C_scaled + s * stride_Cs * n_head, CUDA_R_32F, d_state, stride_Cs,
                    &beta_zero,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3d: dst += X_dt @ M^T (intra-chunk contribution, adds to 3c result)
        // M is stored as M[t_out, t_in] (lower-triangular), transpose needed for Y = X @ M^T.
        {
            const int64_t stride_M = n_M;
            const int64_t stride_X_h = (int64_t)chunk_len * head_dim;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    head_dim, chunk_len, chunk_len,
                    &alpha_one,
                    X_dt       + s * stride_X_h * n_head, matmul_dtype, head_dim, stride_X_h,
                    M_mat      + s * stride_M   * n_head, matmul_dtype, chunk_len, stride_M,
                    &beta_one,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)
        {
            // Scale s_cur in-place by per-head decay_total BEFORE cuBLAS overwrites it
            constexpr int BLOCK = 256;
            dim3 grid((state_per_head + BLOCK - 1) / BLOCK, n_head, n_seq);
            ssm_ssd_scale_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
                s_cur, cs, src3_d,
                d_state, head_dim, n_head,
                chunk_offset, chunk_len, n_tok, A_stride);
            CUDA_CHECK(cudaGetLastError());

            // cuBLAS with beta=1: s_cur = B_weighted @ X_dt^T + 1.0 * s_cur (already scaled)
            const int64_t stride_Bw = d_state * chunk_len;
            const int64_t stride_X  = chunk_len * head_dim;
            const int64_t stride_S  = state_per_head;

            for (int64_t s = 0; s < n_seq; s++) {
                // X_dt is d-fastest {hd, C}, read as OP_T to get {C, hd}
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    d_state, head_dim, chunk_len,
                    &alpha_one,
                    B_weighted + s * stride_Bw * n_head, matmul_dtype, d_state, stride_Bw,
                    X_dt       + s * stride_X  * n_head, matmul_dtype, head_dim, stride_X,
                    &beta_one,
                    s_cur      + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
    }
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C
    const struct ggml_tensor * src6 = dst->src[6];  // ids

    const int64_t nc  = src0->ne[0];  // d_state
    const int64_t nr  = src0->ne[1];  // head_dim or 1
    const int64_t nh  = src1->ne[1];  // n_head
    const int64_t ng  = src4->ne[1];  // n_group
    const int64_t n_t = src1->ne[2];  // number of tokens per sequence
    const int64_t n_s = src1->ne[3];  // number of sequences in the batch

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*n_s == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
    // Requires NVIDIA Turing+ otherwise fallback to scan.
    const bool is_mamba2 = (src3->nb[1] == sizeof(float));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS
                      && n_t <= SSM_SSD_MAX_TOKENS
                      && GGML_CUDA_CC_IS_NVIDIA(cc)
                      && cc >= GGML_CUDA_CC_TURING
                      && nr % 8 == 0;  // head_dim must be 8-aligned for cp.async 16-byte copies

    if (use_ssd) {
        // Convert byte strides to element strides (all tensors are f32-contiguous on dim 0)
        ssm_scan_ssd_f32_cuda(ctx,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            src0->nb[3] / sizeof(float),
            src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            src2->nb[1] / sizeof(float), src2->nb[2] / sizeof(float),
            src3->nb[1] / sizeof(float),
            src4->nb[2] / sizeof(float), src4->nb[3] / sizeof(float),
            src5->nb[2] / sizeof(float), src5->nb[3] / sizeof(float),
            s_off, nc, nr, nh, ng, n_t, n_s);
    } else {
        ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                          src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
                          src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                          s_off, nc, nr, nh, ng, n_t, n_s, stream);
    }
}
