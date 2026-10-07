// test_slot_remap_parity.cpp -- gating correctness check for the sparse MoE slot
// remap, before any 48-layer generation is attempted.
//
// Verifies the core claim: running MUL_MAT_ID against the STOCK full 512-expert
// tensor and against a COMPACT per-expert pool (selected experts placed in
// consecutive slots, ids remapped to 0..n_used-1) produces BIT-EXACT output.
//
// If this passes, the slot remap cannot corrupt activations. If it fails, every
// downstream number is meaningless.
//
// The model is loaded mmap'd with n_gpu_layers = 0, so no VRAM is spent on
// weights and only the tensor under test is uploaded to the GPU.
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "llama-model.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] :
        "/mnt/windows/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/IQ2_XS/"
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf";
    const int layer = 0;
    const int n_used = 10;
    const int n_slots = 85;              // the 4096/48 per-layer pool

    llama_backend_init();

    // ---- load the model mmap'd, CPU only: weights are never touched
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    llama_model * lmodel = llama_model_load_from_file(model, mp);
    if (!lmodel) { fprintf(stderr, "FAIL: model load\n"); return 1; }

    char tname[128];
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", layer);

    const ggml_tensor * full = nullptr;
    for (const auto & kv : llama_internal_get_tensor_map(lmodel)) {
        if (kv.first == tname) { full = kv.second; break; }
    }
    if (!full) { fprintf(stderr, "FAIL: tensor %s not found\n", tname); return 1; }

    const int64_t ne0 = full->ne[0], ne1 = full->ne[1], ne2 = full->ne[2];
    const size_t  per_expert = (size_t)ggml_nbytes(full) / (size_t)ne2;
    printf("tensor %s\n  ne = [%lld, %lld, %lld]  type = %s\n",
           tname, (long long)ne0, (long long)ne1, (long long)ne2, ggml_type_name(full->type));
    printf("  bytes/expert = %zu, total = %zu (%.1f MiB)\n",
           per_expert, ggml_nbytes(full), ggml_nbytes(full)/1048576.0);
    printf("  nb = [%zu, %zu, %zu, %zu]\n",
           full->nb[0], full->nb[1], full->nb[2], full->nb[3]);
    if ((int64_t)ne2 < n_used) { fprintf(stderr, "FAIL: ne2 < n_used\n"); return 1; }
    if (full->buffer) { fprintf(stderr, "note: tensor is already buffered\n"); }

    // ---- pick a deterministic, non-contiguous expert set (adversarial for any
    // off-by-one in the slot mapping)
    std::vector<int32_t> sel(n_used);
    for (int i = 0; i < n_used; i++) sel[i] = (int32_t)((i * 37 + 11) % (int)ne2);
    printf("  selected experts:");
    for (int i = 0; i < n_used; i++) printf(" %d", sel[i]);
    printf("\n");

    // ---- activations; mul_mat_id contracts over ne0
    const int64_t n_embd = ne0;
    std::vector<float> act((size_t)n_embd);
    for (int64_t i = 0; i < n_embd; i++) act[i] = 0.001f * (float)((i * 2654435761u) % 1000);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "FAIL: cuda init\n"); return 1; }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // mul_mat_id over `w` with `ids`, returning ne1 values
    auto run_case = [&](ggml_tensor* w, const std::vector<int32_t>& ids,
                        std::vector<float>& out) -> bool {
        const size_t gsize = ggml_tensor_overhead()*16 + ggml_graph_overhead();
        ggml_init_params ip = { gsize, nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_tensor* i_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)ids.size());

        ggml_tensor* r = ggml_mul_mat_id(ctx, w, a, i_t);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "  alloc fail\n"); ggml_free(ctx); return false;
        }
        // inputs can only be written once they have device storage
        ggml_backend_tensor_set(a, act.data(), 0, (size_t)n_embd*sizeof(float));
        ggml_backend_tensor_set(i_t, ids.data(), 0, ids.size()*sizeof(int32_t));
        for (int i = 0; i < 5; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        out.resize((size_t)ne1);
        ggml_backend_tensor_get(r, out.data(), 0, (size_t)ne1*sizeof(float));
        ggml_free(ctx);
        return true;
    };

    const size_t pool_ctx = ggml_tensor_overhead()*8 + 512ull*1024*1024;

    // ---- CASE A: stock full 512-expert tensor with real expert ids
    {
        ggml_init_params ipw = { pool_ctx, nullptr, true };
        ggml_context* wctx = ggml_init(ipw);
        ggml_tensor* w_full = ggml_new_tensor_3d(wctx, full->type, ne0, ne1, ne2);
        // a no_alloc context gives no device storage; allocate it explicitly
        if (!ggml_backend_alloc_ctx_tensors(wctx, backend)) {
            fprintf(stderr, "FAIL: could not allocate full expert tensor\n"); return 1;
        }
        std::vector<uint8_t> raw((size_t)per_expert * (size_t)ne2);
        const uint8_t* src = (const uint8_t*) full->data;
        if (!src) { fprintf(stderr, "FAIL: tensor has no host data pointer\n"); return 1; }
        memcpy(raw.data(), src, raw.size());
        ggml_backend_tensor_set(w_full, raw.data(), 0, ggml_nbytes(w_full));

        std::vector<float> out_full;
        if (!run_case(w_full, sel, out_full)) return 1;
        printf("  CASE A full  : [%.6f %.6f %.6f %.6f %.6f]\n",
               out_full[0], out_full[1], out_full[2], out_full[3], out_full[4]);

        // ---- CASE B: compact n_slots pool, selected experts in slots 0..n_used-1
        std::vector<uint8_t> compact((size_t)per_expert * (size_t)n_slots, 0);
        for (int i = 0; i < n_used; i++)
            memcpy(compact.data() + (size_t)i*per_expert,
                   raw.data() + (size_t)sel[i]*per_expert, per_expert);
        ggml_context* cctx = ggml_init(pool_ctx ? (ggml_init_params){ pool_ctx, nullptr, true }
                                               : (ggml_init_params){ pool_ctx, nullptr, true });
        ggml_tensor* w_comp = ggml_new_tensor_3d(cctx, full->type, ne0, ne1, n_slots);
        if (!ggml_backend_alloc_ctx_tensors(cctx, backend)) {
            fprintf(stderr, "FAIL: could not allocate compact pool tensor\n"); return 1;
        }
        ggml_backend_tensor_set(w_comp, compact.data(), 0, ggml_nbytes(w_comp));

        std::vector<int32_t> remap(n_used);
        for (int i = 0; i < n_used; i++) remap[i] = i;
        std::vector<float> out_comp;
        if (!run_case(w_comp, remap, out_comp)) return 1;
        printf("  CASE B compact: [%.6f %.6f %.6f %.6f %.6f]\n",
               out_comp[0], out_comp[1], out_comp[2], out_comp[3], out_comp[4]);

        double maxabs = 0.0, maxrel = 0.0; size_t nbit = 0;
        for (size_t i = 0; i < out_full.size(); i++) {
            double x = out_full[i], y = out_comp[i];
            if (x != y) nbit++;
            double d = fabs(x - y);
            if (d > maxabs) maxabs = d;
            double den = fabs(x) > 1e-9 ? fabs(x) : 1.0;
            if (d/den > maxrel) maxrel = d/den;
        }
        printf("\n  elements     = %zu\n  differing    = %zu\n"
               "  max abs diff = %.3e\n  max rel diff = %.3e\n",
               out_full.size(), nbit, maxabs, maxrel);
        bool ok = (maxabs <= 1e-4) || (maxrel <= 1e-4);
        printf("\n  RESULT: %s   (tolerance 1e-4)\n",
               ok ? "PASS - slot remap is numerically exact"
                  : "FAIL - slot remap changes the result");

        ggml_free(cctx); ggml_free(wctx);
        ggml_gallocr_free(galloc);
        llama_model_free(lmodel);
        llama_backend_free();
        return ok ? 0 : 1;
    }
}