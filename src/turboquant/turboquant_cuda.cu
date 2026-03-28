#include "turboquant_cuda.cuh"
#include <cstdio>
#include <cmath>

namespace turboquant {
namespace cuda {

// Helper device function to pack bits
__device__ void pack_bits_device(
    uint8_t* packed,
    const uint8_t* indices,
    float qjl_bit,
    int n_indices,
    int bit_width,
    size_t bytes_per_vector) {

    // Initialize packed data to zero
    for (size_t i = 0; i < bytes_per_vector; i++) {
        packed[i] = 0;
    }

    int bit_pos = 0;

    // Pack indices
    for (int i = 0; i < n_indices; i++) {
        uint8_t idx = indices[i];
        // Ensure index fits in bit_width bits
        idx &= (1 << bit_width) - 1;

        // Pack bits
        for (int b = 0; b < bit_width; b++) {
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;

            if (idx & (1 << b)) {
                packed[byte_idx] |= (1 << bit_idx);
            }
            bit_pos++;
        }
    }

    // Pack QJL bit (1.0 -> 1, -1.0 -> 0)
    int byte_idx = bit_pos / 8;
    int bit_idx = bit_pos % 8;
    if (qjl_bit > 0.0f) {
        packed[byte_idx] |= (1 << bit_idx);
    }
}

// Helper device function to unpack bits
__device__ void unpack_bits_device(
    const uint8_t* packed,
    uint8_t* indices,
    float* qjl_bit,
    int n_indices,
    int bit_width,
    size_t bytes_per_vector) {

    int bit_pos = 0;

    // Unpack indices
    for (int i = 0; i < n_indices; i++) {
        uint8_t idx = 0;
        for (int b = 0; b < bit_width; b++) {
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;

            if (packed[byte_idx] & (1 << bit_idx)) {
                idx |= (1 << b);
            }
            bit_pos++;
        }
        indices[i] = idx;
    }

    // Unpack QJL bit (1 -> 1.0f, 0 -> -1.0f)
    int byte_idx = bit_pos / 8;
    int bit_idx = bit_pos % 8;
    *qjl_bit = (packed[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
}

// Helper device function to find nearest centroid
__device__ int find_nearest_centroid_device(
    float value,
    const float* centroids,
    int num_centroids) {

    float min_dist = fabsf(value - centroids[0]);
    int best_idx = 0;

    for (int i = 1; i < num_centroids; i++) {
        float dist = fabsf(value - centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

// CRITICAL FIX: CUDA kernel for quantizing vectors with norm extraction
// Implements PolarQuant-style norm extraction and rescaling
__global__ void quantize_kernel(
    const float* input,
    uint8_t* packed_output,
    float* vector_norms,        // NEW: output norms for rescaling
    const float* rotation,
    const float* centroids,
    int embedding_dim,
    int bit_width,
    int num_centroids,
    int n_vectors,
    size_t bytes_per_vector) {

    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vector_idx >= n_vectors) return;

    // Get input vector pointer
    const float* input_vec = input + vector_idx * embedding_dim;

    // Get output packed data pointer
    uint8_t* packed = packed_output + vector_idx * bytes_per_vector;

    // CRITICAL STEP 1: Compute L2 norm of input vector
    float norm = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        norm += input_vec[i] * input_vec[i];
    }
    norm = sqrtf(norm);

    // Store norm for later rescaling
    vector_norms[vector_idx] = norm;

    // CRITICAL STEP 2: Normalize vector (avoid division by zero)
    float safe_norm = (norm > 0.0f) ? norm : 1.0f;

    // Temporary buffer for normalized then rotated vector
    float normalized[1024];  // Maximum embedding dimension
    float rotated[1024];

    // Normalize input vector
    for (int i = 0; i < embedding_dim; i++) {
        normalized[i] = input_vec[i] / safe_norm;
    }

    // CRITICAL STEP 3: Apply rotation to normalized vector
    for (int i = 0; i < embedding_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < embedding_dim; j++) {
            sum += rotation[i * embedding_dim + j] * normalized[j];
        }
        rotated[i] = sum;
    }

    // Stage 2: (b-1)-bit MSE quantization on rotated normalized vector
    uint8_t indices[1024];
    for (int i = 0; i < embedding_dim; i++) {
        indices[i] = find_nearest_centroid_device(rotated[i], centroids, num_centroids);
    }

    // Compute MSE quantized vector for residual
    float mse_quantized[1024];
    for (int i = 0; i < embedding_dim; i++) {
        mse_quantized[i] = centroids[indices[i]];
    }

    // Compute residual
    float residual[1024];
    for (int i = 0; i < embedding_dim; i++) {
        residual[i] = rotated[i] - mse_quantized[i];
    }

    // Stage 3: 1-bit QJL on residual (sign of random projection)
    float qjl_bit = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        qjl_bit += residual[i] * ((i % 2 == 0) ? 1.0f : -1.0f);
    }
    qjl_bit = (qjl_bit >= 0.0f) ? 1.0f : -1.0f;

    // Pack indices and QJL bit
    pack_bits_device(packed, indices, qjl_bit, embedding_dim, bit_width, bytes_per_vector);
}

// CRITICAL FIX: CUDA kernel for dequantizing vectors with norm rescaling
// Implements PolarQuant-style norm rescaling after inverse rotation
__global__ void dequantize_kernel(
    const uint8_t* packed_input,
    const float* vector_norms,    // NEW: input norms for rescaling
    float* output,
    const float* rotation_transposed,
    const float* centroids,
    int embedding_dim,
    int bit_width,
    int num_centroids,
    int n_vectors,
    size_t bytes_per_vector) {

    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vector_idx >= n_vectors) return;

    // Get input packed data pointer
    const uint8_t* packed = packed_input + vector_idx * bytes_per_vector;

    // Get output vector pointer
    float* output_vec = output + vector_idx * embedding_dim;

    // Get norm for this vector
    float norm = vector_norms[vector_idx];

    // Unpack indices and QJL bit
    uint8_t indices[1024];
    float qjl_bit;
    unpack_bits_device(packed, indices, &qjl_bit, embedding_dim, bit_width, bytes_per_vector);

    // Dequantize indices to centroids (unit-norm reconstruction)
    float dequantized[1024];
    for (int i = 0; i < embedding_dim; i++) {
        dequantized[i] = centroids[indices[i]];
    }

    // Apply inverse rotation (transpose of rotation matrix)
    // This gives us unit-norm reconstruction
    float unit_norm_recon[1024];
    for (int i = 0; i < embedding_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < embedding_dim; j++) {
            sum += rotation_transposed[i * embedding_dim + j] * dequantized[j];
        }
        unit_norm_recon[i] = sum;
    }

    // CRITICAL STEP: Rescale by original vector norm
    for (int i = 0; i < embedding_dim; i++) {
        output_vec[i] = unit_norm_recon[i] * norm;
    }
}

// Host function to launch quantization kernel
void quantize_vectors_cuda(
    const float* d_input,
    uint8_t* d_packed_output,
    float* d_vector_norms,        // NEW: output norms
    const float* d_rotation,
    const float* d_centroids,
    int embedding_dim,
    int bit_width,
    int num_centroids,
    int n_vectors,
    size_t bytes_per_vector,
    cudaStream_t stream) {

    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (n_vectors + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    quantize_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_input,
        d_packed_output,
        d_vector_norms,
        d_rotation,
        d_centroids,
        embedding_dim,
        bit_width,
        num_centroids,
        n_vectors,
        bytes_per_vector
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in quantize_vectors_cuda: %s\n", cudaGetErrorString(err));
    }
}

// Host function to launch dequantization kernel
void dequantize_vectors_cuda(
    const uint8_t* d_packed_input,
    const float* d_vector_norms,  // NEW: input norms
    float* d_output,
    const float* d_rotation_transposed,
    const float* d_centroids,
    int embedding_dim,
    int bit_width,
    int num_centroids,
    int n_vectors,
    size_t bytes_per_vector,
    cudaStream_t stream) {

    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (n_vectors + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    dequantize_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_packed_input,
        d_vector_norms,
        d_output,
        d_rotation_transposed,
        d_centroids,
        embedding_dim,
        bit_width,
        num_centroids,
        n_vectors,
        bytes_per_vector
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in dequantize_vectors_cuda: %s\n", cudaGetErrorString(err));
    }
}

// Copy rotation matrix and centroids to device memory
void copy_quantizer_to_device(
    const float* rotation,
    const float* centroids,
    int embedding_dim,
    int num_centroids,
    float** d_rotation,
    float** d_centroids) {

    // Allocate device memory for rotation matrix
    size_t rotation_size = embedding_dim * embedding_dim * sizeof(float);
    cudaMalloc(d_rotation, rotation_size);
    cudaMemcpy(*d_rotation, rotation, rotation_size, cudaMemcpyHostToDevice);

    // Allocate device memory for centroids
    size_t centroids_size = num_centroids * sizeof(float);
    cudaMalloc(d_centroids, centroids_size);
    cudaMemcpy(*d_centroids, centroids, centroids_size, cudaMemcpyHostToDevice);
}

// Free device memory for quantizer
void free_quantizer_device(
    float* d_rotation,
    float* d_centroids) {

    if (d_rotation) cudaFree(d_rotation);
    if (d_centroids) cudaFree(d_centroids);
}

} // namespace cuda
} // namespace turboquant
