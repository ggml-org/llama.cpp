/**
 * TQ4_0 Pre-Rotation Kernel
 *
 * Applies the TurboQuant rotation matrix to KV vectors BEFORE they enter
 * the SET_ROWS path. This is a separate kernel that runs before SET_ROWS,
 * avoiding any changes to the existing Q4_0 quantization kernels.
 *
 * The approach:
 * 1. This kernel rotates src data in-place (or to a temp buffer)
 * 2. Standard Q4_0 SET_ROWS then quantizes the pre-rotated data
 * 3. On dequant (to_float), the inverse rotation is applied
 *
 * This avoids modifying any existing llama.cpp kernel infrastructure.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Device-side rotation matrix (uploaded once by tq_cuda_init)
static __device__ float * d_rotation_t = nullptr;
static __device__ float * d_rotation   = nullptr;
static int g_head_dim_cuda = 0;

// Host-side pointers for cudaMemcpyToSymbol
static float * h_d_rotation_t = nullptr;
static float * h_d_rotation   = nullptr;

/**
 * Upload rotation matrices to GPU memory.
 * Called from tq_init() when CUDA is available.
 */
extern "C" void tq_cuda_init(const float * rotation, const float * rotation_t, int head_dim) {
    g_head_dim_cuda = head_dim;
    size_t size = head_dim * head_dim * sizeof(float);

    if (h_d_rotation_t) { cudaFree(h_d_rotation_t); cudaFree(h_d_rotation); }

    cudaMalloc(&h_d_rotation_t, size);
    cudaMalloc(&h_d_rotation, size);
    cudaMemcpy(h_d_rotation_t, rotation_t, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_d_rotation, rotation, size, cudaMemcpyHostToDevice);
}

extern "C" void tq_cuda_free(void) {
    if (h_d_rotation_t) { cudaFree(h_d_rotation_t); h_d_rotation_t = nullptr; }
    if (h_d_rotation)   { cudaFree(h_d_rotation);   h_d_rotation   = nullptr; }
    g_head_dim_cuda = 0;
}

/**
 * Kernel: rotate head vectors in a KV row.
 *
 * Each thread block handles one head_dim-sized vector.
 * Shared memory holds the rotated result.
 *
 * @param data     KV data to rotate IN-PLACE (n_heads * head_dim floats per row)
 * @param rot_t    Rotation matrix Pi^T (head_dim x head_dim)
 * @param head_dim Head dimension
 * @param n_total  Total number of head vectors to rotate
 */
__global__ void k_tq4_rotate_heads(
    float * __restrict__ data,
    const float * __restrict__ rot_t,
    const int head_dim,
    const int n_total
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_total) return;

    float * vec = data + vec_idx * head_dim;

    // Shared memory for input copy (we rotate in-place)
    extern __shared__ float smem[];

    // Copy input to shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        smem[i] = vec[i];
    }
    __syncthreads();

    // Rotate: output[i] = sum_j rot_t[i][j] * input[j]
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sum += rot_t[i * head_dim + j] * smem[j];
        }
        vec[i] = sum;
    }
}

/**
 * Kernel: inverse-rotate head vectors (for dequantization).
 *
 * Same as above but uses Pi (not Pi^T) for the inverse rotation.
 */
__global__ void k_tq4_unrotate_heads(
    float * __restrict__ data,
    const float * __restrict__ rot,
    const int head_dim,
    const int n_total
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_total) return;

    float * vec = data + vec_idx * head_dim;
    extern __shared__ float smem[];

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        smem[i] = vec[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sum += rot[i * head_dim + j] * smem[j];
        }
        vec[i] = sum;
    }
}

/**
 * Host function: rotate KV data before SET_ROWS.
 *
 * @param data      Float data on GPU (src for SET_ROWS)
 * @param n_elements Total elements (n_heads * head_dim per row, times n_rows)
 * @param stream    CUDA stream
 */
extern "C" void tq_cuda_rotate_before_set_rows(float * data, int n_elements, void * stream) {
    if (!h_d_rotation_t || g_head_dim_cuda == 0) return;

    int n_vectors = n_elements / g_head_dim_cuda;
    if (n_vectors == 0) return;

    int threads = min(g_head_dim_cuda, 256);
    int smem_size = g_head_dim_cuda * sizeof(float);

    k_tq4_rotate_heads<<<n_vectors, threads, smem_size, (cudaStream_t)stream>>>(
        data, h_d_rotation_t, g_head_dim_cuda, n_vectors
    );
}

/**
 * Host function: unrotate KV data after dequantization.
 */
extern "C" void tq_cuda_unrotate_after_dequant(float * data, int n_elements, void * stream) {
    if (!h_d_rotation || g_head_dim_cuda == 0) return;

    int n_vectors = n_elements / g_head_dim_cuda;
    if (n_vectors == 0) return;

    int threads = min(g_head_dim_cuda, 256);
    int smem_size = g_head_dim_cuda * sizeof(float);

    k_tq4_unrotate_heads<<<n_vectors, threads, smem_size, (cudaStream_t)stream>>>(
        data, h_d_rotation, g_head_dim_cuda, n_vectors
    );
}
