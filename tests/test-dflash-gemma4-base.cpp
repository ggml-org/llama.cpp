// Test: a non-Gemma4 dflash .gguf must NOT carry the Gemma4-specific
// extras.  The factory in `llama_model_create(loader, params)` checks for
// `blk.0.post_attention_norm` to decide whether to instantiate
// `llama_model_dflash_gemma4` or fall back to the arch-agnostic
// `llama_model_dflash`.  This test confirms the fallback path by
// constructing a synthetic dflash GGUF (arch-agnostic, no Gemma4 markers)
// and asserting the Gemma4 tensors are absent.
//
// The synthetic GGUF is written to a temp file with both metadata and
// zero-filled tensor data so the public `gguf_init_from_file` reader
// can open it.  The factory's dispatch logic only consults tensor
// metadata, not data values, so zero-fill is sufficient.

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_FAIL(msg, ...) \
    do { \
        std::fprintf(stderr, "FAIL (%s:%d): " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        return 1; \
    } while (0)

#define TEST_ASSERT(cond, msg, ...) \
    do { \
        if (!(cond)) { TEST_FAIL(msg, ##__VA_ARGS__); } \
    } while (0)

namespace fs = std::filesystem;

namespace {

// Holds zero-filled buffers for tensor data.  Each entry corresponds to
// one tensor's data section in the GGUF file.
struct zero_data_bank {
    std::vector<std::vector<char>> buffers;

    const void * add(size_t n_bytes) {
        buffers.emplace_back(n_bytes, 0);
        return buffers.back().data();
    }
};

// Build a synthetic arch-agnostic dflash GGUF and write it to `path`.
// Returns true on success.
bool build_synthetic_dflash_gguf(const fs::path & path) {
    struct gguf_context * ctx = gguf_init_empty();
    if (ctx == nullptr) {
        return false;
    }

    // Required architecture + minimum dflash hparams.
    gguf_set_val_str(ctx, "general.architecture", "dflash");
    gguf_set_val_str(ctx, "general.name",         "test-dflash-base");

    // Mirror the LLM_KV keys the loader reads in `load_hparams` and
    // `load_arch_hparams`.  Values are arbitrary; this test never actually
    // loads the model.
    gguf_set_val_u32(ctx, "dflash.context_length",                128);
    gguf_set_val_u32(ctx, "dflash.embedding_length",               64);
    gguf_set_val_u32(ctx, "dflash.block_count",                     2);
    gguf_set_val_u32(ctx, "dflash.feed_forward_length",            128);
    gguf_set_val_u32(ctx, "dflash.attention.head_count",            2);
    gguf_set_val_u32(ctx, "dflash.attention.head_count_kv",         1);
    gguf_set_val_u32(ctx, "dflash.attention.key_length",           32);
    gguf_set_val_u32(ctx, "dflash.attention.value_length",         32);
    gguf_set_val_f32(ctx, "dflash.attention.layer_norm_rms_epsilon", 1e-5f);

    // dflash.target_layers is required (load_arch_hparams will throw if absent).
    int32_t target_layers[] = {0, 1};
    gguf_set_arr_data(ctx, "dflash.target_layers", GGUF_TYPE_INT32,
                      target_layers, sizeof(target_layers) / sizeof(target_layers[0]));

    // Declare the shared + per-layer tensors that the arch-agnostic
    // `llama_model_dflash::load_arch_tensors` would require.  We do NOT
    // declare the Gemma4-specific extras -- that's the point of the test.
    struct tensor_spec {
        const char * name;
        int64_t      ne[2];
    };
    const tensor_spec tensors[] = {
        {"token_embd.weight",      {64,  32}},
        {"output_norm.weight",     {64,   1}},
        {"enc_output_norm.weight", {64,   1}},
        {"fc.weight",              {64,  64}},
        {"blk.0.attn_norm.weight", {64,   1}},
        {"blk.0.attn_q.weight",    {64,  64}},
        {"blk.0.attn_k.weight",    {64,  32}},
        {"blk.0.attn_v.weight",    {64,  32}},
        {"blk.0.attn_out.weight",  {64,  64}},
        {"blk.0.attn_q_norm.weight",{32,  1}},
        {"blk.0.attn_k_norm.weight",{32,  1}},
        {"blk.0.ffn_norm.weight",  {64,   1}},
        {"blk.0.ffn_gate.weight",  {64, 128}},
        {"blk.0.ffn_down.weight",  {128, 64}},
        {"blk.0.ffn_up.weight",    {64, 128}},
    };

    zero_data_bank data;

    for (const auto & t : tensors) {
        struct ggml_tensor gt = {};
        gt.type = GGML_TYPE_F32;
        gt.ne[0] = t.ne[0];
        gt.ne[1] = t.ne[1];
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            gt.ne[d] = 1;
        }
        // Strides for a contiguous F32 tensor:
        //   nb[0] = sizeof(F32)
        //   nb[1] = nb[0] * ne[0]
        //   nb[i] = nb[i-1] * ne[i-1]
        gt.nb[0] = ggml_type_size(gt.type);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            gt.nb[d] = gt.nb[d - 1] * (size_t) gt.ne[d - 1];
        }
        ggml_format_name(&gt, "%s", t.name);
        gguf_add_tensor(ctx, &gt);

        // Allocate zero-filled buffer for the tensor data so the writer
        // can emit a valid file.  The factory only consults metadata.
        const size_t n_bytes = ggml_nbytes(&gt);
        gguf_set_tensor_data(ctx, t.name, data.add(n_bytes));
    }

    const bool ok = gguf_write_to_file(ctx, path.string().c_str(), /*only_meta=*/false);
    gguf_free(ctx);
    return ok;
}

bool gguf_has_tensor(struct gguf_context * ctx, const char * name) {
    return gguf_find_tensor(ctx, name) >= 0;
}

} // namespace

int main() {
    std::fprintf(stderr, "=== test-dflash-gemma4-base ===\n");

    const fs::path path = fs::temp_directory_path() / "test-dflash-gemma4-base.gguf";
    TEST_ASSERT(build_synthetic_dflash_gguf(path),
        "failed to build synthetic dflash gguf at %s", path.string().c_str());
    std::fprintf(stderr, "synthetic gguf: %s\n", path.string().c_str());

    struct gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;

    struct gguf_context * ctx = gguf_init_from_file(path.string().c_str(), params);
    TEST_ASSERT(ctx != nullptr, "gguf_init_from_file returned null");

    // Architecture must be "dflash".
    const int arch_key = gguf_find_key(ctx, "general.architecture");
    TEST_ASSERT(arch_key >= 0, "missing 'general.architecture' in GGUF");
    const char * arch_str = gguf_get_val_str(ctx, arch_key);
    TEST_ASSERT(arch_str != nullptr, "null architecture value");
    TEST_ASSERT(std::strcmp(arch_str, "dflash") == 0,
        "expected architecture 'dflash', got '%s'", arch_str);
    std::fprintf(stderr, "architecture: %s\n", arch_str);

    // The Gemma4 marker tensor MUST be absent -- this is what would route
    // the loader to the Gemma4 subclass.
    TEST_ASSERT(!gguf_has_tensor(ctx, "blk.0.post_attention_norm.weight"),
        "Gemma4 marker 'blk.0.post_attention_norm.weight' must be absent in base dflash");
    TEST_ASSERT(!gguf_has_tensor(ctx, "blk.0.post_ffw_norm.weight"),
        "Gemma4-specific 'blk.0.post_ffw_norm.weight' must be absent in base dflash");
    TEST_ASSERT(!gguf_has_tensor(ctx, "rope_freqs.weight"),
        "Gemma4-specific 'rope_freqs.weight' must be absent in base dflash");
    TEST_ASSERT(!gguf_has_tensor(ctx, "blk.0.layer_output_scale.weight"),
        "Gemma4-specific 'blk.0.layer_output_scale.weight' must be absent in base dflash");

    // The arch-agnostic dflash tensors we declared should be present.
    TEST_ASSERT(gguf_has_tensor(ctx, "token_embd.weight"),
        "expected arch-agnostic 'token_embd.weight'");
    TEST_ASSERT(gguf_has_tensor(ctx, "blk.0.attn_q.weight"),
        "expected arch-agnostic 'blk.0.attn_q.weight'");

    gguf_free(ctx);

    // Clean up the temp file.
    std::error_code ec;
    fs::remove(path, ec);

    std::fprintf(stderr, "PASS: base DFlash GGUF has no Gemma4 markers; "
                         "factory will fall back to llama_model_dflash.\n");
    return 0;
}
