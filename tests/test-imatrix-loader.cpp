#include "imatrix-loader.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

static void test_activation_type(ggml_type type) {
    const auto path = std::filesystem::temp_directory_path() /
        ("test-imatrix-loader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    const auto filename = path.string();
    constexpr int64_t n_values = 32;
    ggml_context * ctx = ggml_init({4096 + 3 * ggml_tensor_overhead(), nullptr, false});
    GGML_ASSERT(ctx);
    gguf_context * gguf = gguf_init_empty();
    GGML_ASSERT(gguf);

    ggml_tensor * sums = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(sums, "blk.0.attn_q.weight.in_sum2");
    for (int64_t i = 0; i < n_values; ++i) {
        static_cast<float *>(sums->data)[i] = 4.0f;
    }
    gguf_add_tensor(gguf, sums);

    ggml_tensor * counts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(counts, "blk.0.attn_q.weight.counts");
    static_cast<float *>(counts->data)[0] = 2.0f;
    gguf_add_tensor(gguf, counts);

    // GGML_TYPE_COUNT covers older imatrices without optional activation sums.
    if (type != GGML_TYPE_COUNT) {
        ggml_tensor * activations = ggml_new_tensor_1d(ctx, type, n_values);
        ggml_set_name(activations, "blk.0.attn_q.weight.in_sum");
        std::memset(activations->data, 0, ggml_nbytes(activations));
        if (type == GGML_TYPE_F32) {
            for (int64_t i = 0; i < n_values; ++i) {
                static_cast<float *>(activations->data)[i] = 1.0f;
            }
        }
        gguf_add_tensor(gguf, activations);
    }
    GGML_ASSERT(gguf_write_to_file(gguf, filename.c_str(), false));
    gguf_free(gguf);
    ggml_free(ctx);

    common_imatrix imatrix;
    const bool loaded = common_imatrix_load(filename, imatrix);
    GGML_ASSERT(std::remove(filename.c_str()) == 0);
    const bool valid = type == GGML_TYPE_COUNT || type == GGML_TYPE_F32;
    // Equal element counts must not permit interpreting compact I8/F16 storage as floats.
    GGML_ASSERT(loaded == valid);
    if (valid) {
        const auto & entry = imatrix.entries.at("blk.0.attn_q.weight");
        GGML_ASSERT(entry.sums == std::vector<float>(n_values, 4.0f));
        GGML_ASSERT(entry.counts == std::vector<int64_t>({2}));
        GGML_ASSERT(entry.activations == (type == GGML_TYPE_F32 ? std::vector<float>(n_values, 1.0f) : std::vector<float>()));
    } else {
        GGML_ASSERT(imatrix.entries.empty());
    }
}

int main() {
    for (const auto type : {GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8}) {
        test_activation_type(type);
    }
    std::puts("PASS: missing/F32 activations accepted; F16/I8 activations rejected");
}
