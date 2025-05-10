#pragma once

#include <initializer_list>

#include "common.h"

struct common_catalog_entry {
    const char * name;
    const char * description;
    std::initializer_list<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    void (*handler)(common_params & params);
};

// This is a list of models that are available in the catalog
// The rule for naming is: [<capability>]-<model_name>
// The <capability> is optional, for example: "fim" or "embd"
// The <model_name> is the name of the model, for example: "qwen-7b"

// For contributors:
// - Model MUST be hosted on hf.co/ggml-org
// - If you want to add your model to the catalog, please open an issue, we will consider copying it to ggml-org
// - For better user experience, we don't add models that are:
//     - NSFW or not having NSFW safeguard
//     - Not having an open-source license
//     - Too old (more than 1 year old)
//     - Having too many issues or poor quality (for ex. no chat templates, sensitive to system prompts)
//     - Or, having too little usage (less than 1000 downloads monthly)

const std::initializer_list<common_catalog_entry> model_catalog = {
    {
        "tts-oute",
        "OuteTTS model",
        {LLAMA_EXAMPLE_TTS},
        [](common_params & params) {
            params.model.hf_repo = "OuteAI/OuteTTS-0.2-500M-GGUF";
            params.model.hf_file = "OuteTTS-0.2-500M-Q8_0.gguf";
            params.vocoder.model.hf_repo = "ggml-org/WavTokenizer";
            params.vocoder.model.hf_file = "WavTokenizer-Large-75-F16.gguf";
        }
    },
    {
        "embd-bge-small-en",
        "bge-small-en-v1.5 text embedding model",
        {LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/bge-small-en-v1.5-Q8_0-GGUF";
            params.model.hf_file = "bge-small-en-v1.5-q8_0.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    },
    {
        "embd-e5-small-en",
        "e5-small-v2 text embedding model",
        {LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/e5-small-v2-Q8_0-GGUF";
            params.model.hf_file = "e5-small-v2-q8_0.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    },
    {
        "embd-gte-small",
        "gte-small text embedding model",
        {LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gte-small-Q8_0-GGUF";
            params.model.hf_file = "gte-small-q8_0.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    },
    {
        "fim-qwen-1.5b",
        "Qwen 2.5 Coder 1.5B",
        {LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_gpu_layers = 99;
            params.flash_attn = true;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    },
    {
        "fim-qwen-3b",
        "Qwen 2.5 Coder 3B (support fill-in-the-middle)",
        {LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_gpu_layers = 99;
            params.flash_attn = true;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    },
    {
        "fim-qwen-7b",
        "Qwen 2.5 Coder 7B (support fill-in-the-middle)",
        {LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_gpu_layers = 99;
            params.flash_attn = true;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    },
    {
        "fim-qwen-7b-spec",
        "use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (support fill-in-the-middle)",
        {LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.model.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.model.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.speculative.n_gpu_layers = 99;
            params.port = 8012;
            params.n_gpu_layers = 99;
            params.flash_attn = true;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    },
    {
        "fim-qwen-14b-spec",
        "use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (support fill-in-the-middle)",
        {LLAMA_EXAMPLE_SERVER},
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.model.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.model.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.speculative.n_gpu_layers = 99;
            params.port = 8012;
            params.n_gpu_layers = 99;
            params.flash_attn = true;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    },
};
