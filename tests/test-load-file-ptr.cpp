// Loads a model and a LoRA adapter through llama_model_load_from_file_ptr /
// llama_adapter_lora_init_from_file_ptr from a GGUF embedded inside a bigger file,
// and compares the logits with a normal load.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_npass = 0;
static int g_ntest = 0;

static bool g_mmap_disabled = false;

static void log_callback(ggml_log_level /*level*/, const char * text, void * /*user_data*/) {
    if (strstr(text, "mmap is disabled")) {
        g_mmap_disabled = true;
    }
    fputs(text, stderr);
}

#define TEST_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return false; \
    } \
} while (0)

static void test_begin(const char * name) {
    g_ntest++;
    printf("  %-64s ", name);
}

static void test_ok() {
    g_npass++;
    printf("OK\n");
}

// copy of src, with prefix junk bytes in front and some after, so reads past the GGUF see wrong data
static FILE * embed(FILE * src, long prefix) {
    FILE * dst = tmpfile();
    if (!dst) { return nullptr; }

    for (long i = 0; i < prefix; ++i) { fputc(0xAB, dst); }

    rewind(src);
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), src)) > 0) { fwrite(buf.data(), 1, n, dst); }

    for (int i = 0; i < 4096; ++i) { fputc(0xCD, dst); }

    fseek(dst, prefix, SEEK_SET);
    return dst;
}

static bool logits_of(llama_model * model, llama_adapter_lora * adapter, std::vector<float> & out) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;

    llama_context * ctx = llama_init_from_model(model, cparams);
    TEST_ASSERT(ctx != nullptr);

    if (adapter) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &adapter, 1, &scale)) {
            llama_free(ctx);
            return false;
        }
    }

    llama_token tokens[2] = { 1, 2 };
    const bool ok = llama_decode(ctx, llama_batch_get_one(tokens, 2)) == 0;
    if (ok) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, -1);
        out.assign(logits, logits + n_vocab);
    }

    llama_free(ctx);
    return ok;
}

static bool test_model(FILE * model_file, const std::vector<float> & ref, long prefix, llama_load_mode mode, bool expect_mmap_disabled, const char * name) {
    test_begin(name);

    FILE * file = embed(model_file, prefix);
    TEST_ASSERT(file != nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = mode;

    g_mmap_disabled = false;
    llama_model * model = llama_model_load_from_file_ptr(file, mparams);
    TEST_ASSERT(model != nullptr);
    TEST_ASSERT(g_mmap_disabled == expect_mmap_disabled);

    std::vector<float> logits;
    const bool ok = logits_of(model, nullptr, logits);
    llama_model_free(model);
    fclose(file);

    TEST_ASSERT(ok);
    TEST_ASSERT(logits == ref);

    test_ok();
    return true;
}

// a rank 4 LoRA on blk.0.attn_q.weight, large enough that a misread changes the logits
static FILE * make_lora(int64_t n_embd) {
    const int rank = 4;

    struct gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.type",         "adapter");
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "adapter.type",         "lora");
    gguf_set_val_f32(ctx, "adapter.lora.alpha",   1.0f);

    struct ggml_init_params ip = { 16ull*1024*1024, NULL, false };
    struct ggml_context * ctx_data = ggml_init(ip);
    if (!ctx_data) { gguf_free(ctx); return nullptr; }

    int64_t ne_a[2] = { n_embd, rank };
    int64_t ne_b[2] = { rank, n_embd };
    struct ggml_tensor * a = ggml_new_tensor(ctx_data, GGML_TYPE_F32, 2, ne_a);
    struct ggml_tensor * b = ggml_new_tensor(ctx_data, GGML_TYPE_F32, 2, ne_b);
    ggml_set_name(a, "blk.0.attn_q.weight.lora_a");
    ggml_set_name(b, "blk.0.attn_q.weight.lora_b");

    float * pa = (float *) a->data;
    for (int64_t i = 0; i < ggml_nelements(a); ++i) { pa[i] = 0.5f*(float)(i % 13) - 3.0f; }
    float * pb = (float *) b->data;
    for (int64_t i = 0; i < ggml_nelements(b); ++i) { pb[i] = 0.5f*(float)(i % 7) - 1.5f; }

    gguf_add_tensor(ctx, a);
    gguf_add_tensor(ctx, b);

    FILE * file = tmpfile();
    if (file) {
        gguf_write_to_file_ptr(ctx, file, false);
        rewind(file);
    }

    ggml_free(ctx_data);
    gguf_free(ctx);
    return file;
}

static bool test_lora(llama_model * model, const std::vector<float> & base) {
    test_begin("lora: file_ptr at offset 7 matches offset 0, and changes the logits");

    FILE * lora = make_lora(llama_model_n_embd(model));
    TEST_ASSERT(lora != nullptr);

    llama_adapter_lora * ad_0 = llama_adapter_lora_init_from_file_ptr(model, lora);
    TEST_ASSERT(ad_0 != nullptr);
    std::vector<float> logits_0;
    TEST_ASSERT(logits_of(model, ad_0, logits_0));

    FILE * lora_7 = embed(lora, 7);
    TEST_ASSERT(lora_7 != nullptr);
    llama_adapter_lora * ad_7 = llama_adapter_lora_init_from_file_ptr(model, lora_7);
    TEST_ASSERT(ad_7 != nullptr);
    std::vector<float> logits_7;
    TEST_ASSERT(logits_of(model, ad_7, logits_7));

    llama_adapter_lora_free(ad_7);
    llama_adapter_lora_free(ad_0);
    fclose(lora_7);
    fclose(lora);

    TEST_ASSERT(logits_0 != base);
    TEST_ASSERT(logits_7 == logits_0);

    test_ok();
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: %s -m <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set(log_callback, nullptr);

    printf("test-load-file-ptr\n\n");

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }

    std::vector<float> ref;
    if (!logits_of(model, nullptr, ref)) { fprintf(stderr, "reference decode failed\n"); return 1; }

    FILE * model_file = fopen(model_path, "rb");
    if (!model_file) { perror("fopen"); return 1; }

    test_model(model_file, ref, 7,    LLAMA_LOAD_MODE_NONE, false, "model: offset 7, no mmap");
    test_model(model_file, ref, 7,    LLAMA_LOAD_MODE_AUTO, true,  "model: offset 7, mmap requested, falls back");
    test_model(model_file, ref, 4096, LLAMA_LOAD_MODE_AUTO, false, "model: offset 4096, mmap stays enabled");
    test_lora(model, ref);

    fclose(model_file);
    llama_model_free(model);
    llama_backend_free();

    printf("\n  %d/%d tests passed\n", g_npass, g_ntest);
    if (g_npass != g_ntest) {
        printf("  FAIL\n");
        return 1;
    }
    printf("  OK\n");
    return 0;
}
