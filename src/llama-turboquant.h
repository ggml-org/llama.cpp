#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_turboquant_runtime_config {
    bool enabled = false;
    bool so8_enabled = true;
    bool so8_learned = false;
    bool triality_enabled = true;
    float triality_mix = 0.5f;
    uint32_t rotation_seed = 0;
};

struct llama_turboquant_training_config {
    uint32_t head_dim = 128;
    uint32_t steps = 500;
    float lr = 1e-3f;
    uint32_t seed = 0;
    bool so8_learned = true;
    bool triality_enabled = true;
};

struct llama_turboquant_triality_metrics {
    float exact_mse = 0.0f;
    float triality_mse = 0.0f;
    float relative_mse_reduction = 0.0f;
};

struct llama_turboquant_artifact {
    uint32_t head_dim = 0;
    bool so8_learned = false;
    bool triality_enabled = false;
    std::vector<float> so8_rotation;
    std::vector<float> triality_codebook;
};

llama_turboquant_runtime_config llama_turboquant_runtime_from_env();

// Applies an in-place block-SO(8) rotation for vectors laid out as [n_vec, head_dim].
// If head_dim is not a multiple of 8, this function leaves the tail unchanged.
void llama_turboquant_apply_so8_rotation(
    std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & rotation_matrix);

// Produces a compact proxy codebook from training vectors using simple centroid slicing.
std::vector<float> llama_turboquant_train_triality_codebook(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim);

llama_turboquant_triality_metrics llama_turboquant_evaluate_triality(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & codebook);

bool llama_turboquant_save_artifact(
    const std::string & path,
    const llama_turboquant_artifact & artifact,
    std::string * error);

bool llama_turboquant_load_artifact(
    const std::string & path,
    llama_turboquant_artifact & artifact,
    std::string * error);
