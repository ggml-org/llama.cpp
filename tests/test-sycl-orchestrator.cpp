// SYCL UnifiedMatmulOrchestrator selection tests.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-sycl.h"
#include "ggml.h"
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/orchestrator.hpp"

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                                 ggml_backend_buffer_usage usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static ggml_tensor_extra_gpu * attach_extra(ggml_tensor * tensor, uint64_t model_id) {
    if (!tensor) {
        return nullptr;
    }
    auto * extra = new ggml_tensor_extra_gpu();
    extra->model_id = model_id;
    tensor->extra = extra;
    return extra;
}

static void detach_extra(ggml_tensor * tensor) {
    if (!tensor || !tensor->extra) {
        return;
    }
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
    tensor->extra = nullptr;
    release_extra_gpu(extra);
}

static bool write_tuning_json(const std::string & path,
                              const std::string & winner,
                              const std::string & quant,
                              int64_t M, int64_t N, int64_t K) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << "{\n";
    out << "  \"results\": [\n";
    out << "    {\n";
    out << "      \"quant\": \"" << quant << "\",\n";
    out << "      \"dim_m\": " << M << ",\n";
    out << "      \"dim_n\": " << N << ",\n";
    out << "      \"dim_k\": " << K << ",\n";
    out << "      \"winner\": \"" << winner << "\",\n";
    out << "      \"tensor_instances\": 1\n";
    out << "    }\n";
    out << "  ]\n";
    out << "}\n";
    return true;
}

static bool test_prompt_vs_decode_selection(ggml_backend_sycl_context & ctx) {
    printf("  test_prompt_vs_decode_selection: ");

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    constexpr int64_t K = 1024;
    constexpr int64_t N = 256;
    ggml_tensor * src0 = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_0, K, N);

    ggml_tensor * src1_decode = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * dst_decode  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, N, 1);

    ggml_tensor * src1_prompt = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 16);
    ggml_tensor * dst_prompt  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, N, 16);

    ggml_sycl::UnifiedMatmulOrchestrator orch(ctx);

    auto decision_decode = orch.select(src0, src1_decode, dst_decode);
    auto decision_prompt = orch.select(src0, src1_prompt, dst_prompt);

    ggml_free(gctx);

    if (!decision_decode.valid || !decision_prompt.valid) {
        printf("FAIL (invalid decision)\n");
        return false;
    }

    auto is_small_batch_kernel = [](ggml_sycl_mul_mat_kernel k) {
        return k == ggml_sycl_mul_mat_kernel::DMMV_SOA ||
               k == ggml_sycl_mul_mat_kernel::DMMV_COALESCED ||
               k == ggml_sycl_mul_mat_kernel::MMVQ_AOS ||
               k == ggml_sycl_mul_mat_kernel::MMVQ_SOA ||
               k == ggml_sycl_mul_mat_kernel::MMVQ_COALESCED ||
               k == ggml_sycl_mul_mat_kernel::ONEDNN_AOS;
    };

    if (!is_small_batch_kernel(decision_decode.kernel)) {
        printf("FAIL (decode path selected unexpected kernel=%d)\n",
               static_cast<int>(decision_decode.kernel));
        return false;
    }

    if (decision_prompt.kernel == ggml_sycl_mul_mat_kernel::DMMV_SOA ||
        decision_prompt.kernel == ggml_sycl_mul_mat_kernel::DMMV_COALESCED ||
        decision_prompt.kernel == ggml_sycl_mul_mat_kernel::MMVQ_AOS ||
        decision_prompt.kernel == ggml_sycl_mul_mat_kernel::MMVQ_SOA ||
        decision_prompt.kernel == ggml_sycl_mul_mat_kernel::MMVQ_COALESCED) {
        printf("FAIL (prompt path should avoid DMMV/MMVQ for batch>8, got kernel=%d)\n",
               static_cast<int>(decision_prompt.kernel));
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_tuning_prefers_onednn(ggml_backend_sycl_context & ctx) {
    printf("  test_tuning_prefers_onednn: ");

    const std::string path = "/tmp/dispatch_orchestrator_onednn.json";
    if (!write_tuning_json(path, "onednn_woq", "Q4_0", 1, 256, 1024)) {
        printf("SKIP (failed to write tuning file)\n");
        return true;
    }
    setenv("GGML_SYCL_DISPATCH_TUNING_JSON", path.c_str(), 1);

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    constexpr int64_t K = 1024;
    constexpr int64_t N = 256;
    ggml_tensor * src0 = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * dst  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, N, 1);
    attach_extra(src0, 901);

    ggml_sycl::UnifiedMatmulOrchestrator orch(ctx);
    auto decision = orch.select(src0, src1, dst);

    detach_extra(src0);
    ggml_free(gctx);

    if (!decision.valid) {
        printf("FAIL (invalid decision)\n");
        return false;
    }

    if (decision.kernel != ggml_sycl_mul_mat_kernel::ONEDNN_AOS) {
        printf("FAIL (expected ONEDNN_AOS, got kernel=%d)\n",
               static_cast<int>(decision.kernel));
        return false;
    }

    if (decision.onednn_path == ggml_sycl::OneDnnPath::None) {
        printf("FAIL (expected onednn path selection)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_layout_unavailable_fallback(ggml_backend_sycl_context & ctx) {
    printf("  test_layout_unavailable_fallback: ");

    const std::string path = "/tmp/dispatch_orchestrator_mmq_soa.json";
    if (!write_tuning_json(path, "mmq_soa", "Q4_0", 16, 256, 1024)) {
        printf("SKIP (failed to write tuning file)\n");
        return true;
    }
    setenv("GGML_SYCL_DISPATCH_TUNING_JSON", path.c_str(), 1);

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    constexpr int64_t K = 1024;
    constexpr int64_t N = 256;
    ggml_tensor * src0 = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 16);
    ggml_tensor * dst  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, N, 16);
    attach_extra(src0, 902);

    ggml_sycl::UnifiedMatmulOrchestrator orch(ctx);
    auto decision = orch.select(src0, src1, dst);

    detach_extra(src0);
    ggml_free(gctx);

    if (!decision.valid) {
        printf("FAIL (invalid decision)\n");
        return false;
    }

    if (decision.kernel == ggml_sycl_mul_mat_kernel::MMQ_SOA ||
        decision.kernel == ggml_sycl_mul_mat_kernel::MMQ_COALESCED) {
        printf("FAIL (expected fallback from SOA/COALESCED, got kernel=%d)\n",
               static_cast<int>(decision.kernel));
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_mul_mat_uses_orchestrator(ggml_backend_t backend) {
    printf("  test_mul_mat_uses_orchestrator: ");

    ggml_sycl::test_reset_orchestrator_call_count();

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    constexpr int64_t K = 32;
    constexpr int64_t N = 16;
    constexpr int64_t M = 4;
    ggml_tensor * src0 = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = ggml_mul_mat(gctx, src0, src1);
    attach_extra(src0, 903);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t wbuf = alloc_tensor_buffer(buft, src0, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ibuf = alloc_tensor_buffer(buft, src1, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t obuf = alloc_tensor_buffer(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!wbuf || !ibuf || !obuf) {
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        ggml_free(gctx);
        printf("FAIL (buffer allocation)\n");
        return false;
    }

    std::vector<uint8_t> weights(ggml_nbytes(src0), 0);
    std::vector<float> input(static_cast<size_t>(K * M), 0.01f);
    ggml_backend_tensor_set(src0, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(src1, input.data(), 0, input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, dst);
    ggml_status status = ggml_backend_graph_compute(backend, graph);

    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(ibuf);
    ggml_backend_buffer_free(obuf);
    detach_extra(src0);
    ggml_free(gctx);

    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL (graph compute)\n");
        return false;
    }

    if (ggml_sycl::test_get_orchestrator_call_count() <= 0) {
        printf("FAIL (no orchestrator call)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_moe_uses_orchestrator(ggml_backend_t backend) {
    printf("  test_moe_uses_orchestrator: ");

    ggml_sycl::test_reset_orchestrator_call_count();
    setenv("GGML_SYCL_DISABLE_FUSED_MOE", "1", 1);

    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    constexpr int64_t K = 32;
    constexpr int64_t N = 16;
    constexpr int64_t N_EXPERTS = 2;
    constexpr int64_t N_USED = 1;
    constexpr int64_t N_TOKENS = 33;

    ggml_tensor * weights = ggml_new_tensor_3d(gctx, GGML_TYPE_Q4_0, K, N, N_EXPERTS);
    ggml_tensor * input   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, K, N_USED, N_TOKENS);
    ggml_tensor * ids     = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, N_USED, N_TOKENS);
    ggml_tensor * out     = ggml_mul_mat_id(gctx, weights, input, ids);
    attach_extra(weights, 904);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t wbuf = alloc_tensor_buffer(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ibuf = alloc_tensor_buffer(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t idbuf = alloc_tensor_buffer(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t obuf = alloc_tensor_buffer(buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!wbuf || !ibuf || !idbuf || !obuf) {
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (idbuf) ggml_backend_buffer_free(idbuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        ggml_free(gctx);
        printf("FAIL (buffer allocation)\n");
        return false;
    }

    std::vector<uint8_t> weights_data(ggml_nbytes(weights), 0);
    std::vector<float> input_data(static_cast<size_t>(K * N_USED * N_TOKENS), 0.02f);
    std::vector<int32_t> ids_data(static_cast<size_t>(N_USED * N_TOKENS), 0);
    ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, out);
    ggml_status status = ggml_backend_graph_compute(backend, graph);

    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(ibuf);
    ggml_backend_buffer_free(idbuf);
    ggml_backend_buffer_free(obuf);
    detach_extra(weights);
    ggml_free(gctx);

    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL (graph compute)\n");
        return false;
    }

    if (ggml_sycl::test_get_orchestrator_call_count() <= 0) {
        printf("FAIL (no orchestrator call)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

int main() {
    // Ensure stable env before any static dispatch state is initialized.
    setenv("GGML_SYCL_UNIFIED_DISPATCH", "0", 1);
    setenv("GGML_SYCL_DISABLE_REORDER", "1", 1);
    setenv("GGML_SYCL_DISPATCH_TUNING", "1", 1);

    printf("Running SYCL orchestrator selection tests...\n\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP (SYCL backend unavailable)\n");
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    if (!ctx) {
        printf("SKIP (SYCL backend context unavailable)\n");
        ggml_backend_free(backend);
        return 0;
    }

    int passed = 0;
    int failed = 0;
    int total = 5;

    if (test_prompt_vs_decode_selection(*ctx)) passed++; else failed++;
    if (test_tuning_prefers_onednn(*ctx)) passed++; else failed++;
    if (test_layout_unavailable_fallback(*ctx)) passed++; else failed++;
    if (test_mul_mat_uses_orchestrator(backend)) passed++; else failed++;
    if (test_moe_uses_orchestrator(backend)) passed++; else failed++;

    ggml_backend_free(backend);

    printf("\n%d/%d tests passed\n", passed, total);
    return (failed == 0) ? 0 : 1;
}
