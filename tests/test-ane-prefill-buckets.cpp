// test-ane-prefill-buckets
//
// Iterate every sequence bucket declared by a Tessera ANE prefill manifest
// and confirm each one loads, warms, and returns finite outputs for a zero
// token prompt. The bucket sizes are read from the manifest itself so a new
// bucket can be qualified without editing the test.
//
// Usage: test-ane-prefill-buckets PREFILL_GGUF [OUTPUT_DIR]

#include "ane-mtp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s PREFILL_GGUF [OUTPUT_DIR]\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(gguf_path, &manifest)) {
        std::fprintf(stderr, "failed to load Tessera ANE prefill manifest from %s\n", gguf_path.c_str());
        return 1;
    }
    if (manifest.abi_version != 1 || manifest.architecture != "gemma4" ||
            manifest.execution_stage != "layer_slab" ||
            manifest.hidden_layout != "token_major.f32.v1" ||
            manifest.cache_requirement != "empty_contiguous_prompt" ||
            manifest.batch_size != 1 || manifest.sequence_buckets.empty()) {
        std::fprintf(stderr, "unexpected Tessera ANE prefill manifest\n");
        return 1;
    }

    const fs::path output_root = argc == 3 ? fs::path(argv[2]) : fs::path();
    if (argc == 3 && !output_root.empty() && !fs::exists(output_root)) {
        fs::create_directories(output_root);
    }

    int failures = 0;
    for (const uint32_t bucket : manifest.sequence_buckets) {
        auto program = common_ane_prefill_program_load(gguf_path, bucket);
        if (!program) {
            std::fprintf(stderr, "bucket %u: program load returned null\n", bucket);
            ++failures;
            continue;
        }
        if (!common_ane_mtp_program_is_warm(program)) {
            std::fprintf(stderr, "bucket %u: program not warm (cache_path=%s)\n",
                    bucket, common_ane_mtp_program_cache_path(program));
            ++failures;
            continue;
        }
        std::vector<int32_t> tokens(bucket, 0);
        std::vector<int32_t> positions(bucket);
        for (uint32_t i = 0; i < bucket; ++i) {
            positions[i] = (int32_t) i;
        }
        const size_t hidden_count = (size_t) bucket * manifest.hidden_size;
        const size_t kv_count = (size_t) bucket * manifest.kv_heads * manifest.head_dim;
        std::vector<float> hidden(hidden_count);
        std::vector<float> keys(kv_count);
        std::vector<float> values(kv_count);
        if (!common_ane_compute_prefill_slab(
                program, bucket, tokens.data(), positions.data(), 1,
                manifest.hidden_size, manifest.kv_heads, manifest.head_dim,
                hidden.data(), keys.data(), values.data())) {
            std::fprintf(stderr, "bucket %u: slab execution failed\n", bucket);
            ++failures;
            continue;
        }
        bool all_finite = true;
        for (const float * part : {hidden.data(), keys.data(), values.data()}) {
            const size_t count = part == hidden.data() ? hidden_count : kv_count;
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(part[i])) {
                    all_finite = false;
                    break;
                }
            }
            if (!all_finite) {
                break;
            }
        }
        if (!all_finite) {
            std::fprintf(stderr, "bucket %u: returned non-finite data\n", bucket);
            ++failures;
            continue;
        }
        if (argc == 3) {
            const fs::path output_path = output_root / ("prefill-s" + std::to_string(bucket) + ".f32");
            std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
            if (!output) {
                std::fprintf(stderr, "bucket %u: failed to open %s\n", bucket, output_path.c_str());
                ++failures;
                continue;
            }
            for (const auto * part : {&hidden, &keys, &values}) {
                output.write(reinterpret_cast<const char *>(part->data()),
                        (std::streamsize) (part->size() * sizeof(float)));
            }
            if (!output) {
                std::fprintf(stderr, "bucket %u: failed to write %s\n", bucket, output_path.c_str());
                ++failures;
                continue;
            }
        }
        std::printf("bucket %u: hidden=%zu kv=%zu finite=ok\n",
                bucket, hidden_count, kv_count);
    }
    if (failures > 0) {
        std::fprintf(stderr, "%d bucket(s) failed qualification\n", failures);
        return 1;
    }
    std::printf("all %zu bucket(s) qualified\n", manifest.sequence_buckets.size());
    return 0;
}
