// Test: a Gemma4 dflash .gguf carries the Gemma4-specific extras
// (attn_post_norm, ffn_post_norm, rope_freqs, layer_output_scale).  These
// are what the loader's factory function uses to dispatch to
// `llama_model_dflash_gemma4` instead of the arch-agnostic base.
//
// Pass a path to a Gemma4-trained dflash .gguf as argv[1] (or via the
// `LLAMACPP_TEST_MODELFILE` env var).  When no model is provided the test
// prints a warning and exits 0 -- this keeps the test optional in CI.
//
// What we check:
//   * the file opens as a valid GGUF
//   * the architecture is "dflash"
//   * the Gemma4-specific tensors exist in the GGUF tensor list
//   * the canonical dflash shared tensors (token_embd, output_norm, etc.)
//     also exist (sanity)

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

static bool gguf_has_tensor(struct gguf_context * ctx, const char * name) {
    return gguf_find_tensor(ctx, name) >= 0;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = std::getenv("LLAMACPP_TEST_MODELFILE");
    }

    if (!model_path || std::strlen(model_path) == 0) {
        std::fprintf(stderr,
            "\033[33mWARNING: No model file provided. Skipping test-dflash-gemma4-gemma4. "
            "Pass a Gemma4 DFlash .gguf as argv[1] or set LLAMACPP_TEST_MODELFILE.\n\033[0m");
        return 0;
    }

    std::fprintf(stderr, "=== test-dflash-gemma4-gemma4 ===\n");
    std::fprintf(stderr, "model: %s\n", model_path);

    struct gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;

    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    TEST_ASSERT(ctx != nullptr, "gguf_init_from_file returned null");

    // Architecture must be "dflash".
    char arch[64] = {};
    const int arch_key = gguf_find_key(ctx, "general.architecture");
    TEST_ASSERT(arch_key >= 0, "missing 'general.architecture' in GGUF");
    const char * arch_str = gguf_get_val_str(ctx, arch_key);
    TEST_ASSERT(arch_str != nullptr, "null architecture value");
    std::snprintf(arch, sizeof(arch), "%s", arch_str);
    TEST_ASSERT(std::strcmp(arch, "dflash") == 0, "expected architecture 'dflash'");
    std::fprintf(stderr, "architecture: %s\n", arch);

    // Gemma4-specific tensors must be present in a Gemma4-trained dflash.
    // The factory uses `blk.0.post_attention_norm.weight` as the trigger
    // (the LLM_TENSOR_ATTN_POST_NORM / "weight" suffix in the canonical
    // GGUF naming); we check the full set of gemma4 drafter extras.
    TEST_ASSERT(gguf_has_tensor(ctx, "blk.0.post_attention_norm.weight"),
        "expected Gemma4-specific 'blk.0.post_attention_norm.weight' tensor");
    TEST_ASSERT(gguf_has_tensor(ctx, "blk.0.post_ffw_norm.weight"),
        "expected Gemma4-specific 'blk.0.post_ffw_norm.weight' tensor");
    TEST_ASSERT(gguf_has_tensor(ctx, "rope_freqs.weight"),
        "expected Gemma4-specific 'rope_freqs.weight' tensor");

    // layer_output_scale is optional in the converter; the factory still
    // routes to the gemma4 subclass without it, so we just warn if missing.
    if (!gguf_has_tensor(ctx, "blk.0.layer_output_scale.weight")) {
        std::fprintf(stderr,
            "info: 'blk.0.layer_output_scale.weight' is absent (optional).\n");
    }

    // Sanity check: canonical dflash shared tensors also present.
    TEST_ASSERT(gguf_has_tensor(ctx, "token_embd.weight"),
        "expected shared 'token_embd.weight' tensor");
    TEST_ASSERT(gguf_has_tensor(ctx, "output_norm.weight"),
        "expected shared 'output_norm.weight' tensor");

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::fprintf(stderr, "n_tensors: %lld\n", (long long) n_tensors);

    gguf_free(ctx);
    std::fprintf(stderr, "PASS: Gemma4 DFlash GGUF has the expected extras.\n");
    return 0;
}
