#include "llama-adapter.h"
#include "llama-graph.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

static void test_clamp(llm_arch arch, bool clamp_before_silu, float limit, bool fused, bool dense = false) {
    std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 1;
    hp.n_embd = 1;
    hp.n_head_arr[0] = 1;
    hp.n_head_kv_arr[0] = 1;
    hp.n_expert = 1;
    hp.n_expert_used_arr[0] = 1;
    hp.swiglu_clamp_exp[0] = limit;
    hp.swiglu_clamp_shexp[0] = limit;
    hp.dsv4_hc_mult = arch == LLM_ARCH_DFLASH ? 4 : 0;

    llm_graph_result result(1024);
    llama_adapter_loras loras;
    llm_graph_params params = {};
    params.model = model.get();
    params.loras = &loras;
    params.res = &result;
    params.ubatch.n_tokens = 3;
    llm_graph_context graph(params);
    auto * ctx = result.get_ctx();

    const std::vector<float> gates = { 4, -4, 1, -1, 2, -2, 0, 0.5f };
    const std::vector<float> ups   = { 3, -3, 2, -2, -4, 4, 1, -0.5f };
    const std::vector<float> inputs = { 1, 2, -1 };
    const int64_t n_ff = gates.size();
    auto * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, inputs.size());
    auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    auto * down   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, 1, 1);
    auto * gate   = fused ? nullptr : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_ff, 1);
    auto * up     = fused ? nullptr : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_ff, 1);
    auto * gate_up = fused ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2 * n_ff, 1) : nullptr;
    auto * output = dense ? graph.build_ffn(input,
            up, nullptr, gate, nullptr, down, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, 0) :
        graph.build_moe_ffn(input, router, up, gate, down, nullptr, 1, 1,
            LLM_FFN_SILU, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0, nullptr, gate_up);
    ggml_build_forward_expand(result.get_gf(), output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    const float zero = 0;
    ggml_backend_tensor_set(router, &zero, 0, sizeof(zero));
    const std::vector<float> ones(n_ff, 1.0f);
    ggml_backend_tensor_set(down, ones.data(), 0, ones.size() * sizeof(float));
    if (fused) {
        ggml_backend_tensor_set(gate_up, gates.data(), 0, gates.size() * sizeof(float));
        ggml_backend_tensor_set(gate_up, ups.data(), gates.size() * sizeof(float), ups.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(gate, gates.data(), 0, gates.size() * sizeof(float));
        ggml_backend_tensor_set(up, ups.data(), 0, ups.size() * sizeof(float));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, result.get_gf()) == GGML_STATUS_SUCCESS);
    std::vector<float> values(inputs.size());
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    for (size_t i = 0; i < inputs.size(); ++i) {
        float expected = 0;
        for (size_t j = 0; j < gates.size(); ++j) {
            float g = gates[j] * inputs[i];
            float u = ups[j] * inputs[i];
            if (limit > 0) {
                u = std::clamp(u, -limit, limit);
                if (clamp_before_silu) {
                    g = std::min(g, limit);
                }
            }
            g = g / (1.0f + expf(-g));
            if (limit > 0 && !clamp_before_silu) {
                g = std::min(g, limit);
            }
            expected += g * u;
        }
        GGML_ASSERT(std::isfinite(values[i]));
        GGML_ASSERT(fabsf(values[i] - expected) < 1e-4f);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}

int main() {
    for (auto arch : { LLM_ARCH_GIGACHAT35, LLM_ARCH_DEEPSEEK4, LLM_ARCH_DEEPSEEK41,
                      LLM_ARCH_MAPLE, LLM_ARCH_HY_V4, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DFLASH, LLM_ARCH_LLAMA }) {
        for (float limit : { 0.0f, 1.0f, 3.0f }) {
            if (arch == LLM_ARCH_GIGACHAT35) {
                test_clamp(arch, true, limit, false, true);
            }
            for (bool fused : { false, true }) {
                test_clamp(arch, arch != LLM_ARCH_LLAMA, limit, fused);
            }
        }
        printf("%s: OK\n", llm_arch_name(arch));
    }
    return 0;
}
