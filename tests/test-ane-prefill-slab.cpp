#include "ane-mtp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s PREFILL_GGUF [OUTPUT_F32]\n", argv[0]);
        return 2;
    }
    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(argv[1], &manifest) ||
            manifest.sequence_buckets.size() != 1 ||
            manifest.sequence_buckets[0] != 128 ||
            manifest.architecture != "gemma4" ||
            manifest.layer_first != 0 || manifest.layer_last != 0 ||
            manifest.cache_requirement != "empty_contiguous_prompt") {
        std::fprintf(stderr, "unexpected Gemma ANE slab manifest\n");
        return 1;
    }
    auto program = common_ane_prefill_program_load(argv[1], 128);
    if (!program || !common_ane_mtp_program_is_warm(program)) {
        std::fprintf(stderr, "failed to warm Gemma ANE slab\n");
        return 1;
    }
    std::vector<int32_t> tokens(128, 0);
    std::vector<int32_t> positions(128);
    for (int32_t i = 0; i < 128; ++i) {
        positions[i] = i;
    }
    const size_t hidden_count = 128ull * manifest.hidden_size;
    const size_t kv_count = 128ull * manifest.kv_heads * manifest.head_dim;
    std::vector<float> hidden(hidden_count);
    std::vector<float> keys(kv_count);
    std::vector<float> values(kv_count);
    if (!common_ane_compute_prefill_slab(
            program, 128, tokens.data(), positions.data(), 1,
            manifest.hidden_size, manifest.kv_heads, manifest.head_dim,
            hidden.data(), keys.data(), values.data())) {
        std::fprintf(stderr, "Gemma ANE slab execution failed\n");
        return 1;
    }
    for (const float value : {hidden[0], hidden.back(), keys[0], keys.back(), values[0], values.back()}) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "Gemma ANE slab returned non-finite data\n");
            return 1;
        }
    }
    if (argc == 3) {
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) {
            std::fprintf(stderr, "failed to open ANE slab output file\n");
            return 1;
        }
        for (const auto * part : {&hidden, &keys, &values}) {
            output.write(reinterpret_cast<const char *>(part->data()),
                    (std::streamsize) (part->size() * sizeof(float)));
        }
        if (!output) {
            std::fprintf(stderr, "failed to write ANE slab output file\n");
            return 1;
        }
    }
    std::printf("Gemma ANE prefill slab passed: hidden=%zu kv=%zu cache=%s\n",
            hidden_count, kv_count, manifest.cache_requirement.c_str());
    return 0;
}
