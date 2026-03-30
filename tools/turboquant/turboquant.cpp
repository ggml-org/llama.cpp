#include "llama-turboquant.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {
void print_usage() {
    std::printf(
        "Usage:\n"
        "  llama-turboquant train --out <artifact.tq> [--head-dim N] [--vecs N] [--seed N] [--so8-learned 0|1] [--triality 0|1]\n"
        "  llama-turboquant eval  --artifact <artifact.tq> [--vecs N] [--seed N]\n");
}

bool arg_value(int argc, char ** argv, const std::string & key, std::string & value) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) {
            value = argv[i + 1];
            return true;
        }
    }
    return false;
}

uint32_t arg_u32(int argc, char ** argv, const std::string & key, uint32_t fallback) {
    std::string value;
    if (!arg_value(argc, argv, key, value)) {
        return fallback;
    }
    const int parsed = std::atoi(value.c_str());
    return parsed < 0 ? fallback : static_cast<uint32_t>(parsed);
}

bool arg_bool(int argc, char ** argv, const std::string & key, bool fallback) {
    std::string value;
    if (!arg_value(argc, argv, key, value)) {
        return fallback;
    }
    return std::atoi(value.c_str()) != 0;
}

std::vector<float> make_synthetic(uint32_t n_vec, uint32_t head_dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(static_cast<size_t>(n_vec) * head_dim);
    for (float & v : out) {
        v = dist(rng);
    }
    return out;
}

std::vector<float> identity_so8() {
    std::vector<float> m(64, 0.0f);
    for (int i = 0; i < 8; ++i) {
        m[i * 8 + i] = 1.0f;
    }
    return m;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    const std::string mode = argv[1];
    if (mode == "train") {
        std::string out_path;
        if (!arg_value(argc, argv, "--out", out_path)) {
            std::fprintf(stderr, "missing --out\n");
            return 2;
        }
        const uint32_t head_dim = arg_u32(argc, argv, "--head-dim", 128);
        const uint32_t n_vec = arg_u32(argc, argv, "--vecs", 4096);
        const uint32_t seed = arg_u32(argc, argv, "--seed", 0);
        const bool so8_learned = arg_bool(argc, argv, "--so8-learned", true);
        const bool triality = arg_bool(argc, argv, "--triality", true);

        auto values = make_synthetic(n_vec, head_dim, seed);
        llama_turboquant_artifact artifact;
        artifact.head_dim = head_dim;
        artifact.so8_learned = so8_learned;
        artifact.triality_enabled = triality;
        artifact.so8_rotation = identity_so8();
        if (triality) {
            artifact.triality_codebook = llama_turboquant_train_triality_codebook(values, n_vec, head_dim);
        }

        std::string err;
        if (!llama_turboquant_save_artifact(out_path, artifact, &err)) {
            std::fprintf(stderr, "save failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("saved artifact: %s\n", out_path.c_str());
        return 0;
    }

    if (mode == "eval") {
        std::string artifact_path;
        if (!arg_value(argc, argv, "--artifact", artifact_path)) {
            std::fprintf(stderr, "missing --artifact\n");
            return 2;
        }
        const uint32_t n_vec = arg_u32(argc, argv, "--vecs", 1024);
        const uint32_t seed = arg_u32(argc, argv, "--seed", 1);

        llama_turboquant_artifact artifact;
        std::string err;
        if (!llama_turboquant_load_artifact(artifact_path, artifact, &err)) {
            std::fprintf(stderr, "load failed: %s\n", err.c_str());
            return 1;
        }
        if (!artifact.triality_enabled || artifact.triality_codebook.empty()) {
            std::fprintf(stderr, "artifact has no triality codebook\n");
            return 1;
        }

        auto values = make_synthetic(n_vec, artifact.head_dim, seed);
        const auto metrics = llama_turboquant_evaluate_triality(values, n_vec, artifact.head_dim, artifact.triality_codebook);
        std::printf("triality_exact_mse=%.8f\n", metrics.exact_mse);
        std::printf("triality_proxy_mse=%.8f\n", metrics.triality_mse);
        std::printf("relative_mse_reduction=%.8f\n", metrics.relative_mse_reduction);
        return 0;
    }

    print_usage();
    return 2;
}
