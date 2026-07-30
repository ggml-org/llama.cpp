#include "ane-mtp.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s PREFILL_GGUF\n", argv[0]);
        return 2;
    }
    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(argv[1], &manifest)) {
        std::fprintf(stderr, "failed to load Tessera ANE prefill manifest\n");
        return 1;
    }
    if (manifest.abi_version != 1 || manifest.architecture.empty() ||
            manifest.hidden_size == 0 || manifest.kv_layout.empty() ||
            manifest.sequence_buckets.empty() ||
            manifest.execution_stage != "layer_slab" ||
            manifest.hidden_layout != "token_major.f32.v1" ||
            manifest.cache_requirement.empty() || manifest.kv_heads == 0 ||
            manifest.head_dim == 0 || manifest.batch_size == 0) {
        std::fprintf(stderr, "invalid Tessera ANE prefill manifest\n");
        return 1;
    }
    common_ane_prefill_manifest dynamic_manifest;
    dynamic_manifest.sequence_buckets = { 64, 128, 256 };
    if (common_ane_prefill_select_bucket(dynamic_manifest, 64) != 64 ||
            common_ane_prefill_select_bucket(dynamic_manifest, 65) != 0 ||
            common_ane_prefill_select_bucket(dynamic_manifest, 300) != 0) {
        std::fprintf(stderr, "exact ANE prefill bucket selection failed\n");
        return 1;
    }
    dynamic_manifest.causal_right_padding = true;
    if (common_ane_prefill_select_bucket(dynamic_manifest, 65) != 128 ||
            common_ane_prefill_select_bucket(dynamic_manifest, 128) != 128 ||
            common_ane_prefill_select_bucket(dynamic_manifest, 257) != 0) {
        std::fprintf(stderr, "causal right-padded ANE prefill bucket selection failed\n");
        return 1;
    }
    std::printf("Tessera ANE prefill manifest: arch=%s layers=%u-%u hidden=%u kv=%s buckets=%zu\n",
            manifest.architecture.c_str(), manifest.layer_first, manifest.layer_last,
            manifest.hidden_size, manifest.kv_layout.c_str(), manifest.sequence_buckets.size());
    return 0;
}
