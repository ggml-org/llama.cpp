#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    GGML_UNUSED(userdata);
}

int main(int argc, char ** argv) {
    const uint32_t n_ctx   = 32;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_vocab = 256;

    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture",                   "llama");
    gguf_set_val_u32(metadata.get(), "llama.context_length",                   n_ctx);
    gguf_set_val_u32(metadata.get(), "llama.embedding_length",                 n_embd);
    gguf_set_val_u32(metadata.get(), "llama.attention.head_count",             n_head);
    gguf_set_val_u32(metadata.get(), "llama.feed_forward_length",              n_ff);
    gguf_set_val_u32(metadata.get(), "llama.block_count",                      2);
    gguf_set_val_f32(metadata.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(metadata.get(), "llama.vocab_size",                       n_vocab);

    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model",  "no_vocab");

    auto add_tensor = [&](const std::string & name, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F32;
        t.ne[0] = ne0;
        t.ne[1] = ne1;
        t.ne[2] = ne2;
        t.ne[3] = ne3;
        t.nb[0] = 4;
        for (int dim = 1; dim < GGML_MAX_DIMS; dim++) {
            t.nb[dim] = t.nb[dim - 1] * t.ne[dim - 1];
        }
        ggml_set_name(&t, name.c_str());
        gguf_add_tensor(metadata.get(), &t);
    };

    add_tensor("token_embd.weight", n_embd, n_vocab);

    llama_model_params params = llama_model_default_params();
    llama_model_ptr(llama_model_init(metadata.get(), set_tensor_data, nullptr, params));
}
