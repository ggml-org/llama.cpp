#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the 31M router predictor from flat FP16 binary.
// Allocates ~63 MB on GPU and uploads weights.
bool moe_predictor_init(const char * bin_path, int n_layer, int in_dim, int h_dim, int out_dim);

// Release GPU buffers and cleanup.
void moe_predictor_free();

// Check if predictor is initialized and ready.
bool moe_predictor_is_ready();

// Execute prediction.
// history_k4: host pointer to [4, n_layer, n_used] uint16 array (e.g. 4 * 48 * 10 = 1920 elements)
// out_top16:  host pointer to [n_layer, 16] uint16 array where top-16 candidate experts will be written
// Returns inference latency in milliseconds.
float moe_predictor_predict(const uint16_t * history_k4, uint16_t * out_top16);

#ifdef __cplusplus
}
#endif
