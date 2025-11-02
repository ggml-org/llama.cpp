#include "rknpu2-calibration.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>

namespace rknpu2_calibration {

// --- Calibration Implementations ---

float calculate_percentile_amax(const float * data, size_t n_elements, float percentile) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Creating a vector of absolute values. We need a mutable copy to use nth_element.
    std::vector<float> abs_values(n_elements);
    #pragma omp parallel for
    for (size_t i = 0; i < n_elements; ++i) {
        abs_values[i] = std::abs(data[i]);
    }

    // Finding the index corresponding to the percentile
    size_t index = static_cast<size_t>((float)n_elements * percentile / 100.0f);

    // Clamping the index to be within the valid range
    index = std::min(index, n_elements - 1);

    // Useing std::nth_element to find the k-th smallest element without a full sort.
    std::nth_element(abs_values.begin(), abs_values.begin() + index, abs_values.end());

    return abs_values[index];
}

float calculate_min_mse_amax(const float * data, size_t n_elements, int num_steps) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Determining search range for amax
    float abs_max_val = 0.0f;
    for (size_t i = 0; i < n_elements; ++i) {
        abs_max_val = std::max(abs_max_val, std::abs(data[i]));
    }
    if (abs_max_val == 0.0f) {
        return 0.0f;
    }
    // Using a high percentile as the lower bound to narrow the search range
    float percentile_max_val = calculate_percentile_amax(data, n_elements, 99.9f);

    float search_min = percentile_max_val;
    float search_max = abs_max_val;

    if (search_min >= search_max) {
        return search_max;
    }

    // Iteratively searching for the best amax
    float best_amax = search_max;
    double min_mse = std::numeric_limits<double>::max();
    const float step_size = (search_max - search_min) / num_steps;

    for (int i = 0; i < num_steps; ++i) {
        const float current_amax = search_min + (float)i * step_size;
        const float current_scale = current_amax / 7.0f;
        if (current_scale < 1e-9f) continue;
        const float iscale = 1.0f / current_scale;

        // Calculating MSE for the current amax candidate
        double current_mse = 0.0;
        #pragma omp parallel for reduction(+:current_mse)
        for (size_t j = 0; j < n_elements; ++j) {
            const float original_val = data[j];
            
            // Quantizing
            const float quantized_f = original_val * iscale;
            const int8_t quantized_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(quantized_f)));
            
            // De-quantizing
            const float dequantized_val = (float)quantized_i * current_scale;
            
            // Accumulating error
            const double diff = (double)original_val - (double)dequantized_val;
            current_mse += diff * diff;
        }

        if (current_mse < min_mse) {
            min_mse = current_mse;
            best_amax = current_amax;
        }
    }
    return best_amax;
}

float calculate_entropy_amax(const float* data, size_t n_elements, int num_bins, int num_steps) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Defining search range and creating reference distribution P
    float abs_max_val = 0.0f;
    for (size_t i = 0; i < n_elements; ++i) {
        abs_max_val = std::max(abs_max_val, std::abs(data[i]));
    }
    if (abs_max_val == 0.0f) {
        return 0.0f;
    }

    std::vector<double> p_dist(num_bins, 0.0);
    const double bin_width = (double)abs_max_val * 2.0 / num_bins;
    for (size_t i = 0; i < n_elements; ++i) {
        int bin_index = static_cast<int>(((data[i] + abs_max_val) / bin_width));
        bin_index = std::min(num_bins - 1, bin_index);
        p_dist[bin_index]++;
    }

    // Normalizing histogram to a probability distribution, adding epsilon for stability
    const double epsilon = 1e-9;
    for (int i = 0; i < num_bins; ++i) {
        p_dist[i] = (p_dist[i] / n_elements) + epsilon;
    }

    // Iteratively searching for the best amax by minimizing KL-divergence
    float best_amax = abs_max_val;
    double min_kl_div = std::numeric_limits<double>::max();
    
    // Narrowing the search range to avoid wasting time on obviously bad values
    const float search_min = calculate_percentile_amax(data, n_elements, 99.5f);
    const float search_max = abs_max_val;
    const float step_size = (search_max - search_min) / num_steps;

    for (int i = 0; i < num_steps; ++i) {
        const float current_amax = search_min + (float)i * step_size;
        const float current_scale = current_amax / 7.0f;
        if (current_scale < 1e-9f) continue;
        const float iscale = 1.0f / current_scale;

        // Creating quantized distribution Q based on P
        std::vector<double> q_dist(num_bins, 0.0);
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            const float original_val = -abs_max_val + (bin_idx + 0.5f) * bin_width;

            // Quantizing-dequantizing the value from the center of the bin
            const float quantized_f = original_val * iscale;
            const int8_t quantized_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(quantized_f)));
            const float dequantized_val = (float)quantized_i * current_scale;

            // Finding which bin the de-quantized value falls into
            int new_bin_idx = static_cast<int>(((dequantized_val + abs_max_val) / bin_width));
            new_bin_idx = std::max(0, std::min(num_bins - 1, new_bin_idx));

            // Transfering the "probability mass" from the old bin to the new one
            q_dist[new_bin_idx] += p_dist[bin_idx];
        }

        // Normalizing Q
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            q_dist[bin_idx] += epsilon;
        }

        // Calculating KL-divergence
        double current_kl_div = 0.0;
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            if (p_dist[bin_idx] > epsilon * 1.1) {
                current_kl_div += p_dist[bin_idx] * log(p_dist[bin_idx] / q_dist[bin_idx]);
            }
        }

        if (current_kl_div < min_kl_div) {
            min_kl_div = current_kl_div;
            best_amax = current_amax;
        }
    }
    return best_amax;
}


// --- Hadamard Transform Implementations ---

// Helper to check if a number is a power of two
static bool is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Iterative Fast Walsh-Hadamard Transform (in-place)
static void fwht_iterative(float* data, int size) {
    for (int h = 1; h < size; h <<= 1) {
        for (int i = 0; i < size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

int next_power_of_two(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void hadamard_transform(float* dst, const float* src, int K, int padded_size) {
    // If no padding is needed, copy and perform in-place.
    if (K == padded_size) {
        memcpy(dst, src, K * sizeof(float));
        fwht_iterative(dst, K);
        return;
    }

    // Using a thread-local buffer to avoid repeated heap allocations.
    thread_local static std::vector<float> padded_data;
    
    // Resizing the buffer only if the current one is too small.
    if (padded_data.size() < (size_t)padded_size) {
        padded_data.resize(padded_size);
    }
    
    // Copying source data and zero-fill the rest (padding).
    memcpy(padded_data.data(), src, K * sizeof(float));
    if (padded_size > K) {
        memset(padded_data.data() + K, 0, (padded_size - K) * sizeof(float));
    }

    // Applying the transform to our temporary buffer
    fwht_iterative(padded_data.data(), padded_size);

    // Copying the result to the destination buffer
    memcpy(dst, padded_data.data(), padded_size * sizeof(float));
}

} // namespace rknpu2_calibration