#pragma once

//
// tessera-flrq.h
//
// FLRQ (Factored Low-Rank Quantization) regime expert. Selects a rank
// via spectral compactness (LieQ rho gate), then computes low-rank
// factors from a sketched truncated SVD.
//

#include <cstdint>
#include <vector>

struct ts_flrq_result {
    std::vector<float> U;       // (out_dim x rank)
    std::vector<float> V;       // (rank x in_dim)
    int64_t            rank;    // selected rank
    float              mse;
    float              spectral_compactness;  // rho
};

struct ts_flrq_params {
    int64_t max_rank;       // upper bound on rank search
    float     rho_thresh;   // spectral compactness gate (default 0.8)
    int64_t   sketch_oversample;  // default 10
    uint32_t  seed;
};

// Select rank via spectral compactness, then compute low-rank factors.
int ts_train_flrq(const float * weights, int64_t out_dim, int64_t in_dim,
                  const ts_flrq_params * params, ts_flrq_result * result);
