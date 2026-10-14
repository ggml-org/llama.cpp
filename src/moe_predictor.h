#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the 60M router predictor from flat FP16 binary.
bool moe_predictor_init(const char * bin_path, int n_layer, int in_dim, int h_dim, int out_dim);

// Release GPU buffers and cleanup.
void moe_predictor_free();

// Check if predictor is initialized and ready.
bool moe_predictor_is_ready();

// Execute prediction.
// history_k4:     host pointer to [4, n_layer, n_used] uint16 array
// out_top16:      host pointer to [n_layer, 16] uint16 array where top-16 candidate experts will be written
// resident_slots: optional host pointer to [n_layer, out_dim] int32 array (moe.slot_of). If provided,
//                 experts with resident_slots[l * out_dim + e] >= 0 are masked out, guaranteeing
//                 that out_top16 contains exclusively COLD (non-resident) candidate experts.
// Returns inference latency in milliseconds.
float moe_predictor_predict(const uint16_t * history_k4, uint16_t * out_top16, const int32_t * resident_slots);

#ifdef __cplusplus
}
#endif
