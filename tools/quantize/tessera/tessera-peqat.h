#pragma once

//
// tessera-peqat.h
//
// PE-QAT: parameter-efficient quantization-aware training. Ports
// tools/tessera/pe_qat.py. Trains LoRA adapters (A, B) on top of a
// frozen ternary fake-quantized weight, with SmoothQuant per-channel
// smoothing, via STE backprop and AdamW. Deterministic given seed.
//

#include <cstdint>
#include <vector>
#include <string>

struct ts_peqat_params {
    int64_t lora_rank;          // LoRA rank (default 8)
    float   smooth_alpha;       // SmoothQuant alpha (default 0.5)
    float   lr;                 // AdamW learning rate (default 1e-4)
    float   weight_decay;       // AdamW weight decay (default 0.01)
    float   beta1;              // Adam beta1 (default 0.9)
    float   beta2;              // Adam beta2 (default 0.999)
    float   eps;                // Adam epsilon (default 1e-8)
    int64_t max_epochs;         // training epochs (default 10)
    float   tol;                // convergence tolerance on loss (default 1e-6)
    uint32_t seed;
    bool    verbose;
};

struct ts_peqat_result {
    std::vector<float> lora_A;      // (in_dim x rank)
    std::vector<float> lora_B;      // (rank x out_dim)
    std::vector<float> smooth_s;    // (in_dim,) SmoothQuant scales
    float              final_loss;
    int64_t            epochs_run;
};

// Run PE-QAT training for one tensor.
// weights: (out_dim x in_dim) FP32 pretrained weights.
// calib_X: (n_tokens x in_dim) calibration activations.
// ref_output: (n_tokens x out_dim) = calib_X @ weights^T (FP16 reference).
int ts_peqat_train(const float * weights,
                   const float * calib_X,
                   const float * ref_output,
                   int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                   const ts_peqat_params * params,
                   ts_peqat_result * result);
