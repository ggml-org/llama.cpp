// Test: dispatch tuning cache parsing and lookup
//
// Verifies that benchmark summary JSON is parsed into kernel choices and
// that shape bucketing maps to expected kernels.

#include "dispatch-tuning.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>

static void write_file(const std::string & path, const std::string & content) {
    std::ofstream out(path, std::ios::trunc);
    out << content;
    out.close();
}

int main() {
    const std::string path = "/tmp/dispatch_tuning_test.json";
    const std::string json = R"JSON(
{
  "results": [
    {
      "tensor": "blk.0.attn_q.weight",
      "quant": "Q4_0",
      "dim_m": 1,
      "dim_n": 4096,
      "dim_k": 4096,
      "tensor_instances": 3,
      "winner": "onednn_woq_gemm",
      "runs": []
    },
    {
      "tensor": "blk.0.attn_k.weight",
      "quant": "Q4_0",
      "dim_m": 16,
      "dim_n": 4096,
      "dim_k": 4096,
      "tensor_instances": 2,
      "winner": "mmvq_coalesced",
      "runs": []
    },
    {
      "tensor": "blk.0.ffn_up.weight",
      "quant": "Q6_K",
      "dim_m": 128,
      "dim_n": 4096,
      "dim_k": 4096,
      "tensor_instances": 1,
      "winner": "mmq_soa",
      "runs": []
    }
  ]
}
)JSON";

    write_file(path, json);

    ggml_sycl::dispatch_tuning::DispatchTuningCache cache;
    std::string error;
    bool ok = ggml_sycl::dispatch_tuning::load_dispatch_tuning_from_file(path, cache, &error);
    if (!ok) {
        std::fprintf(stderr, "FAILED: load_dispatch_tuning_from_file: %s\n", error.c_str());
        return 1;
    }

    auto key_onednn = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(
        GGML_TYPE_Q4_0, 1, 4096, 4096);
    auto entry_onednn = cache.lookup(key_onednn);
    if (!entry_onednn.has_value()) {
        std::fprintf(stderr, "FAILED: missing onednn entry\n");
        return 1;
    }
    if (entry_onednn->kernel != ggml_sycl_mul_mat_kernel::ONEDNN_AOS) {
        std::fprintf(stderr, "FAILED: onednn kernel mismatch\n");
        return 1;
    }

    auto key_mmvq = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(
        GGML_TYPE_Q4_0, 16, 4096, 4096);
    auto entry_mmvq = cache.lookup(key_mmvq);
    if (!entry_mmvq.has_value()) {
        std::fprintf(stderr, "FAILED: missing mmvq entry\n");
        return 1;
    }
    if (entry_mmvq->kernel != ggml_sycl_mul_mat_kernel::MMVQ_COALESCED) {
        std::fprintf(stderr, "FAILED: mmvq kernel mismatch\n");
        return 1;
    }

    auto key_mmq = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(
        GGML_TYPE_Q6_K, 128, 4096, 4096);
    auto entry_mmq = cache.lookup(key_mmq);
    if (!entry_mmq.has_value()) {
        std::fprintf(stderr, "FAILED: missing mmq entry\n");
        return 1;
    }
    if (entry_mmq->kernel != ggml_sycl_mul_mat_kernel::MMQ_SOA) {
        std::fprintf(stderr, "FAILED: mmq kernel mismatch\n");
        return 1;
    }

    std::remove(path.c_str());
    std::fprintf(stderr, "PASS\n");
    return 0;
}
