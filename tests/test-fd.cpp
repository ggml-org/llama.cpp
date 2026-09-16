// Tests for gguf_init_from_fd: loading GGUF metadata and tensor data from a file descriptor
// at an arbitrary byte offset, as needed for a GGUF embedded in a container file
// (for example an Android APK asset stored with android:noCompress).

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int g_npass = 0;
static int g_ntest = 0;

// $TMPDIR is needed to run this on Android, which has no /tmp
static std::string tmp_template(const char * name) {
    const char * dir = getenv("TMPDIR");
    return std::string(dir && dir[0] ? dir : "/tmp") + "/" + name + "-XXXXXX";
}

#define TEST_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return false; \
    } \
} while (0)

static void test_begin(const char * name) {
    g_ntest++;
    printf("  %-60s ", name);
}

static void test_ok() {
    g_npass++;
    printf("OK\n");
}

// the test GGUF holds 3 KV pairs and 4 F32 tensors, element j of tensor i is 100*i + j
static constexpr int      N_KV      = 3;
static constexpr int      N_TENSORS = 4;
static constexpr uint32_t KV_UINT32 = 42;
static constexpr float    KV_FLOAT  = 3.14f;
static const char *       KV_STRING = "hello";

static bool create_test_gguf(const char * path) {
    struct gguf_context * ctx = gguf_init_empty();

    gguf_set_val_u32(ctx, "test.uint32",  KV_UINT32);
    gguf_set_val_str(ctx, "test.string",  KV_STRING);
    gguf_set_val_f32(ctx, "test.float32", KV_FLOAT);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);
    if (!ctx_data) {
        gguf_free(ctx);
        return false;
    }

    for (int i = 0; i < N_TENSORS; ++i) {
        const std::string name = "tensor_" + std::to_string(i);
        int64_t ne[2] = { 4 + i, 3 };
        struct ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, 2, ne);
        ggml_set_name(cur, name.c_str());

        float * data = (float *) cur->data;
        for (int j = 0; j < ggml_nelements(cur); ++j) {
            data[j] = (float)(100*i + j);
        }

        gguf_add_tensor(ctx, cur);
    }

    gguf_write_to_file(ctx, path, false);

    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

#if !defined(_WIN32)

// container layout: prefix_size bytes of 0xAB, a verbatim copy of the GGUF, then 256 bytes of 0xCD
// the suffix makes a read past the declared length land on wrong data instead of EOF
static bool create_container_file(const char * gguf_path,
                                  const char * container_path,
                                  size_t       prefix_size,
                                  size_t     * out_offset,
                                  size_t     * out_length) {
    FILE * src = fopen(gguf_path, "rb");
    if (!src) { return false; }

    fseek(src, 0, SEEK_END);
    const size_t gguf_size = (size_t) ftell(src);
    fseek(src, 0, SEEK_SET);

    std::vector<uint8_t> gguf_data(gguf_size);
    bool ok = fread(gguf_data.data(), 1, gguf_size, src) == gguf_size;
    fclose(src);
    if (!ok) { return false; }

    FILE * dst = fopen(container_path, "wb");
    if (!dst) { return false; }

    const std::vector<uint8_t> prefix(prefix_size, 0xAB);
    const std::vector<uint8_t> suffix(256, 0xCD);

    ok = fwrite(prefix.data(), 1, prefix_size, dst) == prefix_size
      && fwrite(gguf_data.data(), 1, gguf_size, dst) == gguf_size
      && fwrite(suffix.data(), 1, suffix.size(), dst) == suffix.size();
    fclose(dst);

    if (!ok) { return false; }

    *out_offset = prefix_size;
    *out_length = gguf_size;
    return true;
}

static bool verify_kv(const struct gguf_context * gguf_ctx) {
    TEST_ASSERT(gguf_get_n_kv(gguf_ctx) == N_KV);
    {
        const int id = gguf_find_key(gguf_ctx, "test.uint32");
        TEST_ASSERT(id >= 0);
        TEST_ASSERT(gguf_get_val_u32(gguf_ctx, id) == KV_UINT32);
    }
    {
        const int id = gguf_find_key(gguf_ctx, "test.string");
        TEST_ASSERT(id >= 0);
        TEST_ASSERT(strcmp(gguf_get_val_str(gguf_ctx, id), KV_STRING) == 0);
    }
    {
        const int id = gguf_find_key(gguf_ctx, "test.float32");
        TEST_ASSERT(id >= 0);
        TEST_ASSERT(gguf_get_val_f32(gguf_ctx, id) == KV_FLOAT);
    }
    return true;
}

static bool verify_tensor_metadata(const struct gguf_context * gguf_ctx) {
    TEST_ASSERT(gguf_get_n_tensors(gguf_ctx) == N_TENSORS);
    for (int i = 0; i < N_TENSORS; ++i) {
        const std::string name = "tensor_" + std::to_string(i);
        const int id = gguf_find_tensor(gguf_ctx, name.c_str());
        TEST_ASSERT(id >= 0);
        TEST_ASSERT(gguf_get_tensor_type(gguf_ctx, id) == GGML_TYPE_F32);
    }
    return true;
}

static bool verify_tensor_data(struct ggml_context * ctx) {
    for (int i = 0; i < N_TENSORS; ++i) {
        const std::string name = "tensor_" + std::to_string(i);
        const struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        TEST_ASSERT(t != NULL);
        TEST_ASSERT(t->type == GGML_TYPE_F32);
        TEST_ASSERT(t->ne[0] == 4 + i);
        TEST_ASSERT(t->ne[1] == 3);

        const float * data = (const float *) t->data;
        for (int j = 0; j < ggml_nelements(t); ++j) {
            TEST_ASSERT(data[j] == (float)(100*i + j));
        }
    }
    return true;
}

// an fd at offset 0 must give the same metadata as a plain file load
static bool test_baseline_offset_zero(const char * gguf_path) {
    test_begin("baseline: fd at offset 0 matches file load");

    struct stat st;
    TEST_ASSERT(stat(gguf_path, &st) == 0);

    const int fd = open(gguf_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ &ctx };

    struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, 0, (size_t) st.st_size, params);

    struct gguf_context * gguf_ref = gguf_init_from_file(gguf_path, { /*.no_alloc=*/ true, /*.ctx=*/ NULL });

    close(fd);

    TEST_ASSERT(gguf_ctx != NULL);
    TEST_ASSERT(gguf_ref != NULL);
    TEST_ASSERT(ctx != NULL);
    TEST_ASSERT(verify_kv(gguf_ctx));
    TEST_ASSERT(verify_tensor_metadata(gguf_ctx));
    TEST_ASSERT(gguf_get_data_offset(gguf_ctx) == gguf_get_data_offset(gguf_ref));

    ggml_free(ctx);
    gguf_free(gguf_ctx);
    gguf_free(gguf_ref);

    test_ok();
    return true;
}

// the GGUF sits at a non-zero offset that is a multiple of the default alignment (32)
static bool test_embedded_aligned(const char * container_path, size_t offset, size_t length) {
    test_begin("embedded (aligned): metadata + tensor data");

    const int fd = open(container_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = { /*.no_alloc=*/ false, /*.ctx=*/ &ctx };

    struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, offset, length, params);
    close(fd);

    TEST_ASSERT(gguf_ctx != NULL);
    TEST_ASSERT(ctx != NULL);
    TEST_ASSERT(verify_kv(gguf_ctx));
    TEST_ASSERT(verify_tensor_metadata(gguf_ctx));
    TEST_ASSERT(verify_tensor_data(ctx));

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    test_ok();
    return true;
}

// the data section must be aligned relative to the start of the GGUF, not to the start of the file,
// so embedding at an offset that is not a multiple of the alignment must still work
static bool test_alignment_fixup(const char * gguf_path) {
    test_begin("non-aligned offsets: metadata + tensor data (1,7,31,4096)");

    struct stat st;
    TEST_ASSERT(stat(gguf_path, &st) == 0);
    const size_t gguf_size = (size_t) st.st_size;

    size_t reference_data_offset;
    {
        const int fd = open(gguf_path, O_RDONLY);
        TEST_ASSERT(fd >= 0);
        struct gguf_init_params p = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
        struct gguf_context * g = gguf_init_from_fd(fd, 0, gguf_size, p);
        close(fd);
        TEST_ASSERT(g != NULL);
        reference_data_offset = gguf_get_data_offset(g);
        gguf_free(g);
    }

    // remainders modulo the default alignment 32: 1, 7, 31 and 0
    const size_t prefixes[] = { 1, 7, 31, 4096 };

    std::string container_path = tmp_template("test-fd-align");
    const int tmp = mkstemp(&container_path[0]);
    TEST_ASSERT(tmp >= 0);
    close(tmp);

    for (const size_t prefix : prefixes) {
        size_t offset = 0, length = 0;
        TEST_ASSERT(create_container_file(gguf_path, container_path.c_str(), prefix, &offset, &length));
        TEST_ASSERT(offset == prefix);
        TEST_ASSERT(length == gguf_size);

        const int fd = open(container_path.c_str(), O_RDONLY);
        TEST_ASSERT(fd >= 0);

        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = { /*.no_alloc=*/ false, /*.ctx=*/ &ctx };

        struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, offset, length, params);
        close(fd);

        TEST_ASSERT(gguf_ctx != NULL);
        TEST_ASSERT(ctx != NULL);
        TEST_ASSERT(verify_kv(gguf_ctx));
        TEST_ASSERT(verify_tensor_metadata(gguf_ctx));
        TEST_ASSERT(verify_tensor_data(ctx));

        // offsets are relative to the GGUF start, so they must not depend on the embedding offset
        TEST_ASSERT(gguf_get_data_offset(gguf_ctx) == reference_data_offset);

        ggml_free(ctx);
        gguf_free(gguf_ctx);
    }

    unlink(container_path.c_str());

    test_ok();
    return true;
}

// a length that covers the header but cuts into the tensor data must be rejected
static bool test_reject_truncated_length(const char * gguf_path) {
    test_begin("reject: length too small for tensor data");

    struct stat st;
    TEST_ASSERT(stat(gguf_path, &st) == 0);
    const size_t file_size = (size_t) st.st_size;

    size_t data_offset;
    {
        const int fd = open(gguf_path, O_RDONLY);
        TEST_ASSERT(fd >= 0);
        struct gguf_init_params p = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
        struct gguf_context * g = gguf_init_from_fd(fd, 0, file_size, p);
        close(fd);
        TEST_ASSERT(g != NULL);
        data_offset = gguf_get_data_offset(g);
        gguf_free(g);
    }

    const size_t truncated_length = data_offset + 1;
    TEST_ASSERT(truncated_length < file_size);

    const int fd = open(gguf_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
    struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, 0, truncated_length, params);
    close(fd);

    TEST_ASSERT(gguf_ctx == NULL);

    test_ok();
    return true;
}

// a length that cuts into the header must be rejected by the reader itself
static bool test_reject_truncated_header(const char * gguf_path) {
    test_begin("reject: length too small for the header");

    const int fd = open(gguf_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
    struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, 0, 8, params);
    close(fd);

    TEST_ASSERT(gguf_ctx == NULL);

    test_ok();
    return true;
}

static bool test_reject_bad_fd() {
    test_begin("reject: invalid fd (-1)");

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
    struct gguf_context * gguf_ctx = gguf_init_from_fd(-1, 0, 1024, params);
    TEST_ASSERT(gguf_ctx == NULL);

    test_ok();
    return true;
}

// the file position of the caller's fd must not change
static bool test_fd_position_preserved(const char * container_path, size_t offset, size_t length) {
    test_begin("fd position is not changed");

    const int fd = open(container_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);
    TEST_ASSERT(lseek(fd, 123, SEEK_SET) == 123);

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ NULL };
    struct gguf_context * gguf_ctx = gguf_init_from_fd(fd, offset, length, params);

    const off_t pos = lseek(fd, 0, SEEK_CUR);
    close(fd);

    TEST_ASSERT(gguf_ctx != NULL);
    TEST_ASSERT(pos == 123);

    gguf_free(gguf_ctx);

    test_ok();
    return true;
}

// ---------------------------------------------------------------------------
// llama level tests - need a model, so they only run when -m is given
// ---------------------------------------------------------------------------

// These guard the convention that gguf_get_data_offset() and llama_file positions are both
// relative to the start of the GGUF. A mismatch reads plausible garbage rather than failing,
// so the check has to compare real inference output.

static bool logits_of(llama_model * model, llama_adapter_lora * adapter, std::vector<float> & out) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;

    llama_context * ctx = llama_init_from_model(model, cparams);
    TEST_ASSERT(ctx != NULL);

    if (adapter) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &adapter, 1, &scale)) {
            llama_free(ctx);
            return false;
        }
    }

    llama_token tokens[2] = { 1, 2 };
    llama_batch batch = llama_batch_get_one(tokens, 2);
    const bool ok = llama_decode(ctx, batch) == 0;

    if (ok) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, -1);
        out.assign(logits, logits + n_vocab);
    }

    llama_free(ctx);
    return ok;
}

static bool test_model_from_fd(const char * model_path, const char * container_path) {
    test_begin("model: llama_model_load_from_fd matches a file load");

    size_t offset = 0, length = 0;
    TEST_ASSERT(create_container_file(model_path, container_path, 7, &offset, &length));

    llama_model_params mparams = llama_model_default_params();

    llama_model * m_file = llama_model_load_from_file(model_path, mparams);
    TEST_ASSERT(m_file != NULL);
    std::vector<float> logits_file;
    TEST_ASSERT(logits_of(m_file, NULL, logits_file));

    const int fd = open(container_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);
    TEST_ASSERT(lseek(fd, 123, SEEK_SET) == 123);

    llama_model * m_fd = llama_model_load_from_fd(fd, offset, length, mparams);
    TEST_ASSERT(m_fd != NULL);

    // pread must leave the caller's file offset alone
    TEST_ASSERT(lseek(fd, 0, SEEK_CUR) == 123);

    std::vector<float> logits_fd;
    TEST_ASSERT(logits_of(m_fd, NULL, logits_fd));

    TEST_ASSERT(llama_model_n_params(m_fd) == llama_model_n_params(m_file));
    TEST_ASSERT(!logits_file.empty());
    TEST_ASSERT(logits_fd == logits_file);

    close(fd);
    llama_model_free(m_fd);
    llama_model_free(m_file);

    test_ok();
    return true;
}

static bool test_model_from_fd_rejects_short_length(const char * container_path) {
    test_begin("model: reject a length that truncates the tensor data");

    const int fd = open(container_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_fd(fd, 7, 4096, mparams);
    close(fd);

    TEST_ASSERT(model == NULL);

    test_ok();
    return true;
}

// a rank 4 LoRA on blk.0.attn_q.weight, large enough that a misread changes the logits
static bool create_test_lora(const char * path, int64_t n_embd) {
    const int rank = 4;

    struct gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.type",         "adapter");
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "adapter.type",         "lora");
    gguf_set_val_f32(ctx, "adapter.lora.alpha",   1.0f);

    struct ggml_init_params ip = { 16ull*1024*1024, NULL, false };
    struct ggml_context * ctx_data = ggml_init(ip);
    if (!ctx_data) { gguf_free(ctx); return false; }

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
    gguf_write_to_file(ctx, path, false);

    ggml_free(ctx_data);
    gguf_free(ctx);
    return true;
}

static bool test_lora_from_fd(const char * model_path, const char * lora_path, const char * container_path) {
    test_begin("lora: llama_adapter_lora_init_from_fd matches a file load");

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    TEST_ASSERT(model != NULL);

    std::vector<float> logits_base;
    TEST_ASSERT(logits_of(model, NULL, logits_base));

    TEST_ASSERT(create_test_lora(lora_path, llama_model_n_embd(model)));

    llama_adapter_lora * ad_file = llama_adapter_lora_init(model, lora_path);
    TEST_ASSERT(ad_file != NULL);
    std::vector<float> logits_file;
    TEST_ASSERT(logits_of(model, ad_file, logits_file));

    size_t offset = 0, length = 0;
    TEST_ASSERT(create_container_file(lora_path, container_path, 7, &offset, &length));

    const int fd = open(container_path, O_RDONLY);
    TEST_ASSERT(fd >= 0);

    llama_adapter_lora * ad_fd = llama_adapter_lora_init_from_fd(model, fd, offset, length);
    TEST_ASSERT(ad_fd != NULL);
    std::vector<float> logits_fd;
    TEST_ASSERT(logits_of(model, ad_fd, logits_fd));

    close(fd);

    // the adapter must do something, otherwise the comparison below proves nothing
    TEST_ASSERT(logits_file != logits_base);
    TEST_ASSERT(logits_fd == logits_file);

    llama_adapter_lora_free(ad_fd);
    llama_adapter_lora_free(ad_file);
    llama_model_free(model);

    test_ok();
    return true;
}

static bool run_model_tests(const char * model_path) {
    llama_backend_init();

    std::string container = tmp_template("test-fd-model");
    int tmp = mkstemp(&container[0]);
    if (tmp < 0) { perror("mkstemp"); return false; }
    close(tmp);

    std::string lora = tmp_template("test-fd-lora");
    tmp = mkstemp(&lora[0]);
    if (tmp < 0) { perror("mkstemp"); unlink(container.c_str()); return false; }
    close(tmp);

    std::string lora_container = tmp_template("test-fd-lora-container");
    tmp = mkstemp(&lora_container[0]);
    if (tmp < 0) { perror("mkstemp"); unlink(container.c_str()); unlink(lora.c_str()); return false; }
    close(tmp);

    test_model_from_fd(model_path, container.c_str());
    test_model_from_fd_rejects_short_length(container.c_str());
    test_lora_from_fd(model_path, lora.c_str(), lora_container.c_str());

    unlink(container.c_str());
    unlink(lora.c_str());
    unlink(lora_container.c_str());

    llama_backend_free();
    return true;
}

#endif // !defined(_WIN32)

int main(int argc, char ** argv) {
#if defined(_WIN32)
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
    printf("fd based loading is not supported on Windows, skipping.\n");
    return 0;
#else
    const char * model_path = NULL;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }

    printf("test-fd: gguf_init_from_fd tests\n\n");

    std::string gguf_path = tmp_template("test-fd");
    int tmp = mkstemp(&gguf_path[0]);
    if (tmp < 0) { perror("mkstemp"); return 1; }
    close(tmp);

    if (!create_test_gguf(gguf_path.c_str())) {
        fprintf(stderr, "failed to create test GGUF file\n");
        unlink(gguf_path.c_str());
        return 1;
    }

    std::string container_path = tmp_template("test-fd-container");
    tmp = mkstemp(&container_path[0]);
    if (tmp < 0) { perror("mkstemp"); unlink(gguf_path.c_str()); return 1; }
    close(tmp);

    size_t container_offset = 0, container_length = 0;
    if (!create_container_file(gguf_path.c_str(), container_path.c_str(), 512, &container_offset, &container_length)) {
        fprintf(stderr, "failed to create container file\n");
        unlink(gguf_path.c_str());
        unlink(container_path.c_str());
        return 1;
    }

    test_baseline_offset_zero(gguf_path.c_str());
    test_embedded_aligned(container_path.c_str(), container_offset, container_length);
    test_alignment_fixup(gguf_path.c_str());
    test_reject_truncated_length(gguf_path.c_str());
    test_reject_truncated_header(gguf_path.c_str());
    test_reject_bad_fd();
    test_fd_position_preserved(container_path.c_str(), container_offset, container_length);

    unlink(gguf_path.c_str());
    unlink(container_path.c_str());

    if (model_path) {
        printf("\n");
        run_model_tests(model_path);
    }

    printf("\n  %d/%d tests passed\n", g_npass, g_ntest);
    if (g_npass != g_ntest) {
        printf("  FAIL\n");
        return 1;
    }
    printf("  OK\n");
    return 0;
#endif
}
