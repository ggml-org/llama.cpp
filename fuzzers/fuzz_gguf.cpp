#include "gguf.h"

#include <cstddef>
#include <cstdint>

static void gguf_fuzz_log(enum ggml_log_level, const char *, void *) {
}

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    ggml_log_set(gguf_fuzz_log, nullptr);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const gguf_init_params params = { true, nullptr };
    gguf_context * ctx = gguf_init_from_buffer(data, size, params);
    gguf_free(ctx);
    return 0;
}
