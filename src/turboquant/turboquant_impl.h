#ifndef TURBOQUANT_IMPL_H
#define TURBOQUANT_IMPL_H

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include "../../ggml/include/ggml.h"

// TurboQuant implementation for KV cache quantization
// Based on: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
// CRITICAL FIX: Implements norm extraction and rescaling (PolarQuant style)

namespace turboquant {

// Generate random rotation matrix (Haar distributed)
std::vector<float> generate_random_rotation(int dim, int seed);

// Apply rotation to vector
void apply_rotation(const std::vector<float>& Q, const float* input, float* output, int dim);

// Apply inverse rotation (transpose for orthogonal matrix)
void apply_inverse_rotation(const std::vector<float>& Q, const float* input, float* output, int dim);

// Lloyd-Max quantization for 1D distribution
std::vector<float> lloyd_max_quantize(const std::vector<float>& data, int num_centroids, int max_iter);

// Quantize value to nearest centroid
int quantize_value(float value, const std::vector<float>& centroids);

// Dequantize index to centroid value
float dequantize_index(int index, const std::vector<float>& centroids);

// 1-bit QJL transform (sign of random projection)
float qjl_1bit(const float* vector, const std::vector<float>& random_proj, int dim);

// TurboQuant KV cache quantization
class TurboQuantKVCache {
public:
    TurboQuantKVCache(int embedding_dim, int bit_width, int seed = 42);
    
    void init();
    
    // CRITICAL FIX: Quantize with norm extraction
    // Returns: {mse_indices, qjl_bit, vector_norm}
    // The vector_norm MUST be stored and passed to dequantize_to_buffer for correct reconstruction
    std::tuple<std::vector<uint8_t>, float, float> quantize(const float* vector, int dim);
    
    // Dequantize vector (MSE reconstruction, no norm rescaling)
    std::vector<float> dequantize(const std::vector<uint8_t>& mse_indices);
    
    // CRITICAL FIX: Dequantize with norm rescaling
    // This is the version used in get_k/get_v
    void dequantize_to_buffer(const uint8_t* mse_indices, float* output_buffer, int dim, float vector_norm);
    
    // Estimate inner product using TurboQuant (unbiased)
    float estimate_inner_product(
        const std::vector<uint8_t>& q_indices, float q_bit,
        const std::vector<uint8_t>& k_indices, float k_bit);
    
    // Pack indices and QJL bit into bytes (for 4-bit quantization)
    std::pair<std::vector<uint8_t>, size_t> pack_indices(
        const std::vector<uint8_t>& indices, float qjl_bit) const;
    
    // Unpack indices and QJL bit from packed bytes
    std::pair<std::vector<uint8_t>, float> unpack_indices(
        const uint8_t* packed_data, size_t n_bytes) const;
    
private:
    int embedding_dim_;
    int bit_width_;
    int seed_;
    std::vector<float> rotation_;        // Random rotation matrix (embedding_dim x embedding_dim)
    std::vector<float> mse_centroids_;   // Centroids for MSE quantization
    std::vector<float> qjl_projection_;  // Random projection for 1-bit QJL
};

} // namespace turboquant

#endif // TURBOQUANT_IMPL_H
