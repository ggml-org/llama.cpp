#include "turboquant_impl.h"
#include "turboquant_simd.h"
#include <cmath>
#include <cstring>

namespace turboquant {

// Generate random rotation matrix (Haar distributed)  
std::vector<float> generate_random_rotation(int dim, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random Gaussian matrix
    std::vector<float> Q(dim * dim);
    for (int i = 0; i < dim * dim; ++i) {
        Q[i] = dist(gen);
    }
    
    // QR decomposition (simplified Gram-Schmidt)
    for (int i = 0; i < dim; ++i) {
        // Normalize column i
        float norm = 0.0f;
        for (int j = 0; j < dim; ++j) {
            norm += Q[j * dim + i] * Q[j * dim + i];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < dim; ++j) {
            Q[j * dim + i] /= norm;
        }
        
        // Make subsequent columns orthogonal
        for (int k = i + 1; k < dim; ++k) {
            float dot = 0.0f;
            for (int j = 0; j < dim; ++j) {
                dot += Q[j * dim + i] * Q[j * dim + k];
            }
            for (int j = 0; j < dim; ++j) {
                Q[j * dim + k] -= dot * Q[j * dim + i];
            }
        }
    }
    
    return Q;
}

// Apply rotation to vector (SIMD optimized)
void apply_rotation(const std::vector<float>& Q, const float* input, float* output, int dim) {
    // Use SIMD-optimized version if available
    simd::apply_rotation_auto(Q.data(), input, output, dim);
}

// Apply inverse rotation (transpose for orthogonal matrix)
void apply_inverse_rotation(const std::vector<float>& Q, const float* input, float* output, int dim) {
    for (int i = 0; i < dim; ++i) {
        float val = 0.0f;
        for (int j = 0; j < dim; ++j) {
            val += Q[j * dim + i] * input[j]; // Transpose: Q[j][i] instead of Q[i][j]
        }
        output[i] = val;
    }
}

// Lloyd-Max quantization
std::vector<float> lloyd_max_quantize(const std::vector<float>& data, int num_centroids, int max_iter) {
    if (data.empty() || num_centroids < 2) return {};
    
    std::vector<float> centroids(num_centroids);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, data.size() - 1);
    
    // Initialize centroids
    for (int c = 0; c < num_centroids; ++c) {
        centroids[c] = data[dist(gen)];
    }
    
    // Lloyd-Max iterations
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<int> counts(num_centroids, 0);
        std::vector<float> sums(num_centroids, 0.0f);
        
        for (size_t i = 0; i < data.size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best = 0;
            for (int c = 0; c < num_centroids; ++c) {
                float d = std::abs(data[i] - centroids[c]);
                if (d < min_dist) {
                    min_dist = d;
                    best = c;
                }
            }
            counts[best]++;
            sums[best] += data[i];
        }
        
        bool converged = true;
        for (int c = 0; c < num_centroids; ++c) {
            if (counts[c] > 0) {
                float new_c = sums[c] / counts[c];
                if (std::abs(new_c - centroids[c]) > 1e-6f) converged = false;
                centroids[c] = new_c;
            }
        }
        
        if (converged) break;
    }
    
    return centroids;
}

// Quantize value to nearest centroid
int quantize_value(float value, const std::vector<float>& centroids) {
    int best = 0;
    float min_dist = std::abs(value - centroids[0]);
    for (size_t i = 1; i < centroids.size(); ++i) {
        float dist = std::abs(value - centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best = i;
        }
    }
    return best;
}

// Dequantize index
float dequantize_index(int index, const std::vector<float>& centroids) {
    return centroids[index];
}

// QJL 1-bit transform
float qjl_1bit(const float* vector, const std::vector<float>& proj, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += vector[i] * proj[i];
    }
    return (dot >= 0.0f) ? 1.0f : -1.0f;
}

// Constructor
TurboQuantKVCache::TurboQuantKVCache(int embedding_dim, int bit_width, int seed)
: embedding_dim_(embedding_dim), bit_width_(bit_width), seed_(seed) {
init();
}

// Initialize quantizer
void TurboQuantKVCache::init() {
    // Generate rotation matrix
    rotation_ = generate_random_rotation(embedding_dim_, seed_);
    
    // Generate sample data for centroid training
    std::vector<float> training_data(embedding_dim_ * 1000);
    std::mt19937 gen(seed_ + 1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : training_data) v = dist(gen);
    
    // Rotate training data
    std::vector<float> rotated(training_data.size());
    for (int i = 0; i < 1000; ++i) {
        apply_rotation(rotation_, &training_data[i * embedding_dim_], &rotated[i * embedding_dim_], embedding_dim_);
    }
    
    // Train centroids on rotated data
    int num_centroids = 1 << (bit_width_ - 1);  // b-1 bits
    mse_centroids_ = lloyd_max_quantize(rotated, num_centroids, 20);
    
    // Generate QJL projection
    qjl_projection_.resize(embedding_dim_);
    std::uniform_real_distribution<float> real_dist(-1.0f, 1.0f);
    std::mt19937 gen2(seed_ + 2);
    for (auto& v : qjl_projection_) v = real_dist(gen2);
}

// CRITICAL FIX: Quantize with norm extraction (PolarQuant style)
std::tuple<std::vector<uint8_t>, float, float> TurboQuantKVCache::quantize(const float* vector, int dim) {
    if (dim != embedding_dim_) {
        throw std::invalid_argument("Dimension mismatch");
    }

    // CRITICAL STEP 1: Extract vector norm (L2 norm)
    float vector_norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        vector_norm += vector[i] * vector[i];
    }
    vector_norm = std::sqrt(vector_norm);

    // CRITICAL STEP 2: Normalize vector to unit sphere
    std::vector<float> normalized(dim);
    float safe_norm = (vector_norm > 0.0f) ? vector_norm : 1.0f;
    for (int i = 0; i < dim; ++i) {
        normalized[i] = vector[i] / safe_norm;
    }

    // Stage 1: Random rotation of normalized vector
    std::vector<float> rotated(dim);
    apply_rotation(rotation_, normalized.data(), rotated.data(), dim);

    // Stage 2: (b-1)-bit MSE quantization on rotated normalized vector
    std::vector<uint8_t> mse_indices(dim);
    std::vector<float> mse_residuals(dim);

    for (int i = 0; i < dim; ++i) {
        int idx = quantize_value(rotated[i], mse_centroids_);
        float quantized = dequantize_index(idx, mse_centroids_);
        mse_indices[i] = static_cast<uint8_t>(idx);
        mse_residuals[i] = rotated[i] - quantized;
    }

    // Stage 3: 1-bit QJL on residual
    float qjl_bit = qjl_1bit(mse_residuals.data(), qjl_projection_, dim);

    // Return indices, QJL bit, AND the norm for storage
    return {mse_indices, qjl_bit, vector_norm};
}

// Dequantize (MSE only) - OLD VERSION, kept for compatibility
std::vector<float> TurboQuantKVCache::dequantize(const std::vector<uint8_t>& mse_indices) {
    std::vector<float> result(embedding_dim_);
    for (int i = 0; i < embedding_dim_; ++i) {
        result[i] = dequantize_index(mse_indices[i], mse_centroids_);
    }
    
    // Apply inverse rotation
    std::vector<float> output(embedding_dim_);
    apply_inverse_rotation(rotation_, result.data(), output.data(), embedding_dim_);
    
    return output;
}

// CRITICAL FIX: Dequantize to buffer with norm rescaling
void TurboQuantKVCache::dequantize_to_buffer(const uint8_t* mse_indices, float* output_buffer, int dim, float vector_norm) {
    // Step 1: Dequantize indices to centroids
    std::vector<float> result(dim);
    for (int i = 0; i < dim; ++i) {
        result[i] = dequantize_index(mse_indices[i], mse_centroids_);
    }
    
    // Step 2: Apply inverse rotation
    apply_inverse_rotation(rotation_, result.data(), output_buffer, dim);
    
    // CRITICAL STEP 3: Rescale by original vector norm
    for (int i = 0; i < dim; ++i) {
        output_buffer[i] *= vector_norm;
    }
}

// Estimate inner product
float TurboQuantKVCache::estimate_inner_product(
    const std::vector<uint8_t>& q_indices, float q_bit,
    const std::vector<uint8_t>& k_indices, float k_bit) {
    
    float mse_part = 0.0f;
    for (size_t i = 0; i < q_indices.size(); ++i) {
        float q_val = dequantize_index(q_indices[i], mse_centroids_);
        float k_val = dequantize_index(k_indices[i], mse_centroids_);
        mse_part += q_val * k_val;
    }
    
    return mse_part + q_bit * k_bit;
}

// Pack indices
std::pair<std::vector<uint8_t>, size_t> TurboQuantKVCache::pack_indices(
    const std::vector<uint8_t>& indices, float qjl_bit) const {
    
    int bits_per_index = bit_width_ - 1;  // b-1 bits for MSE
    int total_bits = indices.size() * bits_per_index + 1;  // +1 for QJL bit
    size_t n_bytes = (total_bits + 7) / 8;
    
    std::vector<uint8_t> packed(n_bytes, 0);
    int bit_pos = 0;
    
    for (uint8_t idx : indices) {
        for (int b = 0; b < bits_per_index; ++b) {
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            if (idx & (1 << b)) {
                packed[byte_idx] |= (1 << bit_idx);
            }
            bit_pos++;
        }
    }
    
    // Pack QJL bit
    int byte_idx = bit_pos / 8;
    int bit_idx = bit_pos % 8;
    if (qjl_bit > 0.0f) {
        packed[byte_idx] |= (1 << bit_idx);
    }
    
    return {packed, n_bytes};
}

// Unpack indices
std::pair<std::vector<uint8_t>, float> TurboQuantKVCache::unpack_indices(
    const uint8_t* packed_data, size_t n_bytes) const {
    
    int bits_per_index = bit_width_ - 1;
    int total_bits = n_bytes * 8;
    int n_indices = (total_bits - 1) / bits_per_index;
    
    std::vector<uint8_t> indices(n_indices);
    int bit_pos = 0;
    
    for (int i = 0; i < n_indices; ++i) {
        uint8_t idx = 0;
        for (int b = 0; b < bits_per_index; ++b) {
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            if (packed_data[byte_idx] & (1 << bit_idx)) {
                idx |= (1 << b);
            }
            bit_pos++;
        }
        indices[i] = idx;
    }
    
    // Unpack QJL bit
    int byte_idx = bit_pos / 8;
    int bit_idx = bit_pos % 8;
    float qjl_bit = (packed_data[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
    
    return {indices, qjl_bit};
}

} // namespace turboquant
