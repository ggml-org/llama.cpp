#ifndef TURBOQUANT_CUDA_CUH
#define TURBOQUANT_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace turboquant {
namespace cuda {

// CUDA kernel for quantizing vectors with norm extraction
// Each thread processes one vector
// CRITICAL: Extracts norm, normalizes, then quantizes (PolarQuant style)
__global__ void quantize_kernel(
    const float* input,         // Input vectors [n_vectors, embedding_dim]
    uint8_t* packed_output,     // Packed indices output [n_vectors, bytes_per_vector]
    float* vector_norms,        // Output norms [n_vectors] - NEW: for rescaling
    const float* rotation,      // Rotation matrix [embedding_dim, embedding_dim]
    const float* centroids,     // Centroids [num_centroids]
    int embedding_dim,          // Dimension of each vector
    int bit_width,              // Bits per index (e.g., 4)
    int num_centroids,          // Number of centroids (2^bit_width)
    int n_vectors,              // Number of vectors to process
    size_t bytes_per_vector     // Bytes per packed vector
);

// CUDA kernel for dequantizing vectors with norm rescaling
// Each thread processes one vector
// CRITICAL: Applies norm rescaling after inverse rotation (PolarQuant style)
__global__ void dequantize_kernel(
    const uint8_t* packed_input,  // Packed indices input [n_vectors, bytes_per_vector]
    const float* vector_norms,    // Input norms [n_vectors] - NEW: for rescaling
    float* output,                // Output vectors [n_vectors, embedding_dim]
    const float* rotation,        // Rotation matrix [embedding_dim, embedding_dim] (transposed for inverse)
    const float* centroids,       // Centroids [num_centroids]
    int embedding_dim,            // Dimension of each vector
    int bit_width,                // Bits per index (e.g., 4)
    int num_centroids,            // Number of centroids (2^bit_width)
    int n_vectors,                // Number of vectors to process
    size_t bytes_per_vector       // Bytes per packed vector
);

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
    cudaStream_t stream = 0
);

// Host function to launch dequantization kernel
void dequantize_vectors_cuda(
    const uint8_t* d_packed_input,
    const float* d_vector_norms,  // NEW: input norms
    float* d_output,
    const float* d_rotation,
    const float* d_centroids,
    int embedding_dim,
    int bit_width,
    int num_centroids,
    int n_vectors,
    size_t bytes_per_vector,
    cudaStream_t stream = 0
);

// Copy rotation matrix and centroids to device memory
void copy_quantizer_to_device(
    const float* rotation,
    const float* centroids,
    int embedding_dim,
    int num_centroids,
    float** d_rotation,
    float** d_centroids
);

// Free device memory for quantizer
void free_quantizer_device(
    float* d_rotation,
    float* d_centroids
);

} // namespace cuda
} // namespace turboquant

#endif // TURBOQUANT_CUDA_CUH