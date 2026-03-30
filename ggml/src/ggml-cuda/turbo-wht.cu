#include "common.cuh"

// Dynamic Walsh-Hadamard Transform — supports any power-of-2 group size
// Group size = head_dim, passed via op_params[1]
// Uses shared memory for groups > 32 (warp size)

// Sign pattern from golden ratio hash
static __device__ float turbo_sign(int i) {
    return ((((unsigned)i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static __global__ void turbo_wht_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                         const int64_t n_total, const int group_size, const int direction) {
    extern __shared__ float smem[];

    const int64_t group_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = group_id * group_size;

    if (base + tid >= n_total) return;

    // Load + apply first signs (forward) or just load (inverse)
    float val;
    if (direction == 0) {
        val = src[base + tid] * turbo_sign(tid);
    } else {
        val = src[base + tid];
    }
    smem[tid] = val;
    __syncthreads();

    // WHT butterfly in shared memory
    for (int step = 1; step < group_size; step <<= 1) {
        int partner = tid ^ step;
        float other = smem[partner];
        __syncthreads();
        if (tid & step) {
            smem[tid] = other - val;
        } else {
            smem[tid] = other + val;
        }
        val = smem[tid];
        __syncthreads();
    }

    // Normalize + apply second signs (inverse) or just normalize (forward)
    float norm = 1.0f / sqrtf((float)group_size);
    if (direction == 0) {
        dst[base + tid] = val * norm;
    } else {
        dst[base + tid] = val * norm * turbo_sign(tid);
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;

    int32_t params[2];
    memcpy(params, dst->op_params, sizeof(params));
    const int direction = params[0];
    // Group size from ne[0] (head_dim) — must be power of 2
    const int group_size = (int)src->ne[0];
    GGML_ASSERT((group_size & (group_size - 1)) == 0); // power of 2

    const int64_t n_total = ggml_nelements(src);
    const int64_t n_groups = n_total / group_size;

    cudaStream_t stream = ctx.stream();
    turbo_wht_kernel<<<n_groups, group_size, group_size * sizeof(float), stream>>>(
        src_d, dst_d, n_total, group_size, direction);
}
