#include "llama-turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace {
bool env_flag(const char * name, bool fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    return std::atoi(v) != 0;
}

float env_float(const char * name, float fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    return std::strtof(v, nullptr);
}

uint32_t env_u32(const char * name, uint32_t fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    const int val = std::atoi(v);
    return val < 0 ? fallback : static_cast<uint32_t>(val);
}

float sqr(float x) {
    return x * x;
}
} // namespace

llama_turboquant_runtime_config llama_turboquant_runtime_from_env() {
    llama_turboquant_runtime_config cfg;
    cfg.enabled = env_flag("LLAMA_TURBOQUANT", false);
    cfg.so8_enabled = env_flag("LLAMA_TURBOQUANT_SO8", true);
    cfg.so8_learned = env_flag("LLAMA_TURBOQUANT_SO8_LEARNED", false);
    cfg.triality_enabled = env_flag("LLAMA_TURBOQUANT_TRIALITY", true);
    cfg.triality_mix = std::clamp(env_float("LLAMA_TURBOQUANT_TRIALITY_MIX", 0.5f), 0.0f, 1.0f);
    cfg.rotation_seed = env_u32("LLAMA_TURBOQUANT_ROTATION_SEED", 0);
    return cfg;
}

void llama_turboquant_apply_so8_rotation(
    std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & rotation_matrix) {
    if (head_dim == 0 || n_vec == 0 || values.empty()) {
        return;
    }
    const uint32_t n_blocks = head_dim / 8;
    if (n_blocks == 0 || rotation_matrix.size() < 64) {
        return;
    }

    for (uint32_t i = 0; i < n_vec; ++i) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            float in[8];
            float out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const uint32_t off = i * head_dim + b * 8;
            for (uint32_t r = 0; r < 8; ++r) {
                in[r] = values[off + r];
            }
            for (uint32_t r = 0; r < 8; ++r) {
                for (uint32_t c = 0; c < 8; ++c) {
                    out[r] += rotation_matrix[r * 8 + c] * in[c];
                }
            }
            for (uint32_t r = 0; r < 8; ++r) {
                values[off + r] = out[r];
            }
        }
    }
}

std::vector<float> llama_turboquant_train_triality_codebook(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim) {
    if (n_vec == 0 || head_dim == 0 || values.size() < static_cast<size_t>(n_vec) * head_dim) {
        return {};
    }

    const uint32_t n_centroids = 3;
    std::vector<float> codebook(static_cast<size_t>(n_centroids) * head_dim, 0.0f);
    const uint32_t stride = std::max(1u, n_vec / n_centroids);
    for (uint32_t c = 0; c < n_centroids; ++c) {
        const uint32_t src = std::min(n_vec - 1, c * stride);
        const size_t src_off = static_cast<size_t>(src) * head_dim;
        const size_t dst_off = static_cast<size_t>(c) * head_dim;
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(src_off), head_dim, codebook.begin() + static_cast<std::ptrdiff_t>(dst_off));
    }
    return codebook;
}

llama_turboquant_triality_metrics llama_turboquant_evaluate_triality(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & codebook) {
    llama_turboquant_triality_metrics m;
    if (n_vec == 0 || head_dim == 0 || codebook.size() < static_cast<size_t>(3) * head_dim) {
        return m;
    }

    double mse_exact = 0.0;
    double mse_triality = 0.0;
    for (uint32_t i = 0; i < n_vec; ++i) {
        const size_t off = static_cast<size_t>(i) * head_dim;
        for (uint32_t j = 0; j < head_dim; ++j) {
            const float v = values[off + j];
            mse_exact += sqr(v);
            float best = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < 3; ++c) {
                const float d = std::fabs(v - codebook[static_cast<size_t>(c) * head_dim + j]);
                best = std::min(best, d);
            }
            mse_triality += sqr(best);
        }
    }
    const double denom = static_cast<double>(n_vec) * head_dim;
    m.exact_mse = static_cast<float>(mse_exact / denom);
    m.triality_mse = static_cast<float>(mse_triality / denom);
    m.relative_mse_reduction = m.exact_mse > 0.0f ? (m.exact_mse - m.triality_mse) / m.exact_mse : 0.0f;
    return m;
}

bool llama_turboquant_save_artifact(
    const std::string & path,
    const llama_turboquant_artifact & artifact,
    std::string * error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) {
            *error = "failed to open artifact for write";
        }
        return false;
    }

    out << "TQCUDA1\n";
    out << artifact.head_dim << "\n";
    out << (artifact.so8_learned ? 1 : 0) << "\n";
    out << (artifact.triality_enabled ? 1 : 0) << "\n";
    out << artifact.so8_rotation.size() << "\n";
    for (float v : artifact.so8_rotation) {
        out << v << "\n";
    }
    out << artifact.triality_codebook.size() << "\n";
    for (float v : artifact.triality_codebook) {
        out << v << "\n";
    }
    return true;
}

bool llama_turboquant_load_artifact(
    const std::string & path,
    llama_turboquant_artifact & artifact,
    std::string * error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) {
            *error = "failed to open artifact for read";
        }
        return false;
    }

    std::string magic;
    std::getline(in, magic);
    if (magic != "TQCUDA1") {
        if (error) {
            *error = "invalid artifact magic";
        }
        return false;
    }

    uint32_t so8 = 0;
    uint32_t tri = 0;
    size_t rot_size = 0;
    size_t cb_size = 0;
    in >> artifact.head_dim;
    in >> so8;
    in >> tri;
    artifact.so8_learned = so8 != 0;
    artifact.triality_enabled = tri != 0;
    in >> rot_size;
    artifact.so8_rotation.resize(rot_size);
    for (size_t i = 0; i < rot_size; ++i) {
        in >> artifact.so8_rotation[i];
    }
    in >> cb_size;
    artifact.triality_codebook.resize(cb_size);
    for (size_t i = 0; i < cb_size; ++i) {
        in >> artifact.triality_codebook[i];
    }
    if (!in.good() && !in.eof()) {
        if (error) {
            *error = "artifact parse failed";
        }
        return false;
    }
    return true;
}
