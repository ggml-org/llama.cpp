
#include <common.h>
#include <llama.h>
#include <string.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
// NOTE: the llm_arch enum is in the private API
#include "../src/llama-model.h"
#include "ggml.h"
#include "gguf.h"
// For gguf_type_size
#include "../ggml/src/ggml-impl.h"

struct random_tensor {
    const std::string name;
    const std::vector<int64_t> shape;
    const ggml_type type = GGML_TYPE_F32; // TODO: maybe make this configurable?
    const std::function<float(std::mt19937 &, int64_t i)> distribution;

    random_tensor(std::string name, std::vector<int64_t> shape,
                  const std::function<float(std::mt19937 &)> & distribution = std::normal_distribution<float>()) :
        name(std::move(name)),
        shape(std::move(shape)),
        distribution([distribution](std::mt19937 & rng, int64_t i) {
            GGML_UNUSED(i);
            return distribution(rng);
        }) {}

    random_tensor(std::string name, std::vector<int64_t> shape, const std::function<float(int64_t)> & distribution) :
        name(std::move(name)),
        shape(std::move(shape)),
        distribution([distribution](std::mt19937 & rng, int64_t i) {
            GGML_UNUSED(rng);
            return distribution(i);
        }) {}

    random_tensor(std::string name, std::vector<int64_t> shape, const std::function<float()> & distribution) :
        name(std::move(name)),
        shape(std::move(shape)),
        distribution([distribution](std::mt19937 & rng, int64_t i) {
            GGML_UNUSED(rng);
            GGML_UNUSED(i);
            return distribution();
        }) {}

    random_tensor(std::string name, std::vector<int64_t> shape,
                  std::function<float(std::mt19937 &, int64_t i)> distribution) :
        name(std::move(name)),
        shape(std::move(shape)),
        distribution(std::move(distribution)) {}

    random_tensor(const random_tensor & other) :
        name(other.name),
        shape(other.shape),
        type(other.type),
        distribution(other.distribution) {}

    size_t n_bytes() const {
        int64_t prod = 1;
        for (int64_t d : shape) {
            prod *= d;
        }
        return ggml_row_size(type, prod);
    }

    ggml_tensor * to_ggml_tensor(ggml_context * ctx, std::mt19937 & rng) const {
        ggml_tensor * tensor = ggml_new_tensor(ctx, type, shape.size(), shape.data());
        GGML_ASSERT(tensor->data != nullptr);

        int64_t n_elems = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            n_elems *= shape[i];
        }

        for (int64_t i = 0; i < n_elems; ++i) {
            ((float *) tensor->data)[i] = distribution(rng, i);
        }

        return tensor;
    }
};

// TODO: move this to the gguf library?

struct gguf_value {
    const gguf_type type;
    union {
        uint8_t                   uint8;
        int8_t                    int8;
        uint16_t                  uint16;
        int16_t                   int16;
        uint32_t                  uint32;
        int32_t                   int32;
        float                     float32;
        bool                      boolean;
        std::string *             string;
        std::vector<gguf_value> * array;
        uint64_t                  uint64;
        int64_t                   int64;
        double                    float64;
    } value;

    ~gguf_value() {
        switch (type) {
            case GGUF_TYPE_STRING:
                delete value.string;
                break;
            case GGUF_TYPE_ARRAY:
                delete value.array;
                break;
            default:
                break;
        }
    }

    gguf_value(const gguf_value & other) : type(other.type) {
        switch (type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_BOOL:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_COUNT:
                value = other.value;
                break;
            case GGUF_TYPE_STRING:
                value.string = new std::string(*other.value.string);
                break;
            case GGUF_TYPE_ARRAY:
                value.array = new std::vector(*other.value.array);
                break;
        }
    }

    gguf_value(int8_t val) : type(GGUF_TYPE_INT8) { value.int8 = val; }

    gguf_value(uint8_t val) : type(GGUF_TYPE_UINT8) { value.uint8 = val; }

    gguf_value(int16_t val) : type(GGUF_TYPE_INT16) { value.int16 = val; }

    gguf_value(uint16_t val) : type(GGUF_TYPE_UINT16) { value.uint16 = val; }

    gguf_value(int32_t val) : type(GGUF_TYPE_INT32) { value.int32 = val; }

    gguf_value(uint32_t val) : type(GGUF_TYPE_UINT32) { value.uint32 = val; }

    gguf_value(float val) : type(GGUF_TYPE_FLOAT32) { value.float32 = val; }

    gguf_value(bool val) : type(GGUF_TYPE_BOOL) { value.boolean = val; }

    gguf_value(uint64_t val) : type(GGUF_TYPE_UINT64) { value.uint64 = val; }

    gguf_value(int64_t val) : type(GGUF_TYPE_INT64) { value.int64 = val; }

    gguf_value(double val) : type(GGUF_TYPE_FLOAT64) { value.float64 = val; }

    gguf_value(std::string val) : type(GGUF_TYPE_STRING) { value.string = new std::string(std::move(val)); }

    gguf_value(const char * val) : type(GGUF_TYPE_STRING) { value.string = new std::string(val); }

    gguf_value(std::vector<gguf_value> val) : type(GGUF_TYPE_ARRAY) { value.array = new std::vector(std::move(val)); }

    gguf_value(const std::vector<gguf_value> & val) : type(GGUF_TYPE_ARRAY) { value.array = new std::vector(val); }

    template <typename T> gguf_value(const std::vector<T> & val) : type(GGUF_TYPE_ARRAY) {
        value.array = new std::vector<gguf_value>();
        for (const auto & v : val) {
            value.array->push_back(gguf_value(v));
        }
    }

    void set(gguf_context * ctx, const std::string & key) const {
        const char * k = key.c_str();
        switch (type) {
            case GGUF_TYPE_UINT8:
                gguf_set_val_u8(ctx, k, value.uint8);
                break;
            case GGUF_TYPE_INT8:
                gguf_set_val_i8(ctx, k, value.int8);
                break;
            case GGUF_TYPE_UINT16:
                gguf_set_val_u16(ctx, k, value.uint16);
                break;
            case GGUF_TYPE_INT16:
                gguf_set_val_i16(ctx, k, value.int16);
                break;
            case GGUF_TYPE_UINT32:
                gguf_set_val_u32(ctx, k, value.uint32);
                break;
            case GGUF_TYPE_INT32:
                gguf_set_val_i32(ctx, k, value.int32);
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_set_val_f32(ctx, k, value.float32);
                break;
            case GGUF_TYPE_BOOL:
                gguf_set_val_bool(ctx, k, value.boolean);
                break;
            case GGUF_TYPE_STRING:
                gguf_set_val_str(ctx, k, value.string->c_str());
                break;
            case GGUF_TYPE_ARRAY:
                {
                    const size_t arr_size = value.array->size();
                    if (arr_size > 0) {
                        const gguf_type arr_type = (*value.array)[0].type;
                        if (arr_type == GGUF_TYPE_STRING) {
                            std::vector<const char *> strings(arr_size);
                            for (size_t i = 0; i < arr_size; ++i) {
                                strings[i] = (*value.array)[i].value.string->c_str();
                            }
                            gguf_set_arr_str(ctx, k, strings.data(), strings.size());
                        } else {
                            const size_t type_size = gguf_type_size(arr_type);
                            std::vector<uint8_t> data(type_size * arr_size);
                            for (size_t i = 0; i < arr_size; ++i) {
                                memcpy(data.data() + type_size * i, &(*value.array)[i].value, type_size);
                            }
                            gguf_set_arr_data(ctx, k, arr_type, data.data(), data.size() / type_size);
                        }
                        // TODO: handle nested arrays
                    }
                    break;
                }
            case GGUF_TYPE_UINT64:
                gguf_set_val_u64(ctx, k, value.uint64);
                break;
            case GGUF_TYPE_INT64:
                gguf_set_val_i64(ctx, k, value.int64);
                break;
            case GGUF_TYPE_FLOAT64:
                gguf_set_val_f64(ctx, k, value.float64);
                break;
            case GGUF_TYPE_COUNT:
                break;
        }
    }
};

struct model_variant {
    llm_arch arch;
    std::string name;
    std::vector<random_tensor> tensors;
    std::vector<std::pair<llm_kv, gguf_value>> metadata;

    model_variant(llm_arch arch, const std::string & name) : arch(arch), name(name) {
        add_kv(LLM_KV_GENERAL_TYPE, "model");
        add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(arch));
        add_kv(LLM_KV_GENERAL_NAME, name);
    }

    model_variant(const model_variant & other) :
        arch(other.arch),
        name(other.name),
        tensors(other.tensors),
        metadata(other.metadata) {}

    void add_tensor(const std::string & name, const std::vector<int64_t> & shape,
                    const std::function<float(std::mt19937 &)> & distribution = std::normal_distribution<float>()) {
        tensors.push_back(random_tensor(name, shape, distribution));
    }

    void add_tensor(const std::string & name, const std::vector<int64_t> & shape,
                    const std::function<float(std::mt19937 &, int64_t)> & distribution) {
        tensors.push_back(random_tensor(name, shape, distribution));
    }

    void add_tensor(const std::string & name, const std::vector<int64_t> & shape,
                    const std::function<float(int64_t)> & distribution) {
        tensors.push_back(random_tensor(name, shape, distribution));
    }

    void add_tensor(const std::string & name, const std::vector<int64_t> & shape,
                    const std::function<float()> & distribution) {
        tensors.push_back(random_tensor(name, shape, distribution));
    }

    void add_kv(llm_kv kv, const gguf_value & value) { metadata.push_back(std::pair(kv, value)); }

    bool write_to_file(const char * fname, std::mt19937 & rng) const {
        gguf_context * ctx_gguf = gguf_init_empty();

        auto kv = LLM_KV(arch);

        for (const auto & m : metadata) {
            m.second.set(ctx_gguf, kv(m.first));
        }

        size_t total_size = 0;
        for (const auto & t : tensors) {
            total_size += t.n_bytes() + ggml_tensor_overhead();
        }

        ggml_init_params init_params = {
            total_size,
            nullptr, // allocate internally
            false,   // do allocate memory when creating tensors
        };

        ggml_context * ctx = ggml_init(init_params);

        // initialize the tensors and add to GGUF
        for (const auto & t : tensors) {
            ggml_tensor * tensor = t.to_ggml_tensor(ctx, rng);
            ggml_set_name(tensor, t.name.c_str());

            gguf_add_tensor(ctx_gguf, tensor);
        }

        bool status = gguf_write_to_file(ctx_gguf, fname, false);

        ggml_free(ctx);
        gguf_free(ctx_gguf);

        return status;
    }

    static void insert_from_arch(std::vector<model_variant> & variants, llm_arch arch) {
        uint32_t n_vocab = 256;
        uint32_t n_embd = 32;
        uint32_t n_ff = 3 * n_embd;

        uint32_t n_layer = 2;

        auto tn = LLM_TN(arch);

        // random vocab
        const auto add_tokenizer = [](model_variant & m, uint32_t n_vocab) {
            std::vector<std::string> vocab_tokens(n_vocab);
            std::vector<float>       vocab_scores(n_vocab);
            std::vector<int32_t>     vocab_types(n_vocab);

            char buf[32];
            for (size_t i = 0; i < n_vocab; ++i) {
                snprintf(buf, sizeof(buf), "<%zu>", i);
                vocab_tokens[i] = std::string(buf);
                vocab_scores[i] = -1000.0f;
                vocab_types[i]  = 4;  // USER_DEFINED type
            }

            m.add_kv(LLM_KV_TOKENIZER_MODEL, "llama");
            m.add_kv(LLM_KV_TOKENIZER_PRE, "default");
            m.add_kv(LLM_KV_TOKENIZER_LIST, vocab_tokens);
            m.add_kv(LLM_KV_TOKENIZER_SCORES, vocab_scores);
            m.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE, vocab_types);
        };

        // TODO: fill the variants
        // TODO: how to make the variants more modular?
        switch (arch) {
            case LLM_ARCH_LLAMA:
                {
                    variants.push_back(model_variant(arch, "Llama2"));
                    model_variant & cur = variants.back();

                    n_embd = 16;
                    const uint32_t n_head = 4;
                    const uint32_t n_head_kv = n_head / 2;
                    const uint32_t n_embd_head_k = n_embd / n_head;
                    const uint32_t n_embd_k_gqa = n_embd_head_k * n_head_kv;
                    const uint32_t n_embd_v_gqa = n_embd_k_gqa;

                    cur.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
                    cur.add_kv(LLM_KV_CONTEXT_LENGTH, (uint32_t) 4096);
                    cur.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
                    cur.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
                    cur.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
                    cur.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, n_embd / n_head);

                    add_tokenizer(cur, n_vocab);

                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                    // omitting the actual output tensor to leave it use tok_embd

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_LLAMA4: // has chunked interleaved sliding-window
                {
                    variants.push_back(model_variant(arch, "Llama4"));
                    model_variant & cur = variants.back();

                    n_layer = 4; // for the swa pattern
                    n_embd = 16;
                    const uint32_t n_head = 4;
                    const uint32_t n_head_kv = n_head / 2;
                    const uint32_t n_embd_head_k = n_embd / n_head;
                    const uint32_t n_embd_k_gqa = n_embd_head_k * n_head_kv;
                    const uint32_t n_embd_v_gqa = n_embd_k_gqa;
                    const uint32_t n_moe_layer_step = 2;
                    const uint32_t n_ff_exp = n_embd * 2;
                    const uint32_t n_expert = 4;

                    cur.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
                    cur.add_kv(LLM_KV_CONTEXT_LENGTH, (uint32_t) 4096);
                    cur.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
                    cur.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
                    cur.add_kv(LLM_KV_EXPERT_COUNT, n_expert);
                    cur.add_kv(LLM_KV_EXPERT_USED_COUNT, (uint32_t) 2);
                    cur.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
                    cur.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, n_embd / n_head);
                    cur.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP, n_moe_layer_step);
                    cur.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp);
                    // FIXME: this isn't used because the default is 8192
                    cur.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW, (uint32_t) 389); // prime number

                    add_tokenizer(cur, n_vocab);

                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                    // omitting the actual output tensor to leave it use tok_embd

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        bool is_moe_layer = (i + 1) % n_moe_layer_step == 0;

                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        if (is_moe_layer) {
                            cur.add_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff_exp, n_embd, n_expert});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert});

                            // Shared expert
                            const int64_t n_ff_shexp = n_ff_exp;
                            cur.add_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd    });
                            cur.add_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp});
                        } else {
                            cur.add_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                            cur.add_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                        }
                    }
                } break;
            case LLM_ARCH_DECI:
            case LLM_ARCH_FALCON:
            case LLM_ARCH_BAICHUAN:
            case LLM_ARCH_GROK:
            case LLM_ARCH_GPT2:
            case LLM_ARCH_GPTJ:
            case LLM_ARCH_GPTNEOX:
            case LLM_ARCH_MPT:
            case LLM_ARCH_STARCODER:
            case LLM_ARCH_REFACT:
            case LLM_ARCH_BERT:
            case LLM_ARCH_NOMIC_BERT:
            case LLM_ARCH_NOMIC_BERT_MOE:
            case LLM_ARCH_JINA_BERT_V2:
            case LLM_ARCH_BLOOM:
            case LLM_ARCH_STABLELM:
            case LLM_ARCH_QWEN:
            case LLM_ARCH_QWEN2:
            case LLM_ARCH_QWEN2MOE:
            case LLM_ARCH_QWEN2VL:
            case LLM_ARCH_QWEN3:
            case LLM_ARCH_QWEN3MOE:
            case LLM_ARCH_PHI2:
            case LLM_ARCH_PHI3:
            case LLM_ARCH_PHIMOE:
            case LLM_ARCH_PLAMO:
            case LLM_ARCH_CODESHELL:
            case LLM_ARCH_ORION:
            case LLM_ARCH_INTERNLM2:
            case LLM_ARCH_MINICPM:
            case LLM_ARCH_MINICPM3:
            case LLM_ARCH_GEMMA:
                break;
            case LLM_ARCH_GEMMA2: // has standard interleaved sliding-window
                {
                    variants.push_back(model_variant(arch, "Gemma2"));
                    model_variant & cur = variants.back();

                    n_layer = 2; // minimum for the swa pattern
                    n_embd = 16;
                    const uint32_t n_head = 4;
                    const uint32_t n_head_kv = n_head / 2;
                    const uint32_t n_embd_head_k = n_embd / n_head;
                    const uint32_t n_embd_k_gqa = n_embd_head_k * n_head_kv;
                    const uint32_t n_embd_v_gqa = n_embd_k_gqa;

                    cur.add_kv(LLM_KV_CONTEXT_LENGTH, (uint32_t) 4096);
                    cur.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
                    cur.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
                    cur.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);
                    cur.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
                    cur.add_kv(LLM_KV_ATTN_LOGIT_SOFTCAPPING, 50.0f);
                    cur.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
                    cur.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW, (uint32_t) 389); // prime number

                    add_tokenizer(cur, n_vocab);

                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd});
                    }
                } break;
            case LLM_ARCH_GEMMA3:
            case LLM_ARCH_STARCODER2:
                break;
            case LLM_ARCH_MAMBA:
                {
                    variants.push_back(model_variant(arch, "Mamba"));
                    model_variant & cur = variants.back();

                    const uint32_t d_inner = 2 * n_embd;
                    const uint32_t d_conv = 4;
                    const uint32_t d_state = 16;
                    const uint32_t dt_rank = (n_embd + 15) / 16;

                    const auto init_A_S4D = [](int64_t i) {
                        return -((i % d_state) + 1);
                    };

                    cur.add_kv(LLM_KV_CONTEXT_LENGTH, (uint32_t) 1024 * 1024);
                    cur.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
                    cur.add_kv(LLM_KV_FEED_FORWARD_LENGTH, (uint32_t) 0);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, (uint32_t) 0);
                    cur.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
                    cur.add_kv(LLM_KV_SSM_CONV_KERNEL, d_conv);
                    cur.add_kv(LLM_KV_SSM_INNER_SIZE, d_inner);
                    cur.add_kv(LLM_KV_SSM_STATE_SIZE, d_state);
                    cur.add_kv(LLM_KV_SSM_TIME_STEP_RANK, dt_rank);
                    cur.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);

                    add_tokenizer(cur, n_vocab);

                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab });
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd });
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab });

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd });
                        cur.add_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), { n_embd, 2 * d_inner });

                        cur.add_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), { d_conv, d_inner });
                        cur.add_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), { d_inner });

                        cur.add_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), { d_inner, dt_rank + 2 * d_state });

                        cur.add_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), { dt_rank, d_inner });
                        cur.add_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), { d_inner });

                        // no "weight" suffix for these
                        cur.add_tensor(tn(LLM_TENSOR_SSM_A, i), { d_state, d_inner }, init_A_S4D);
                        cur.add_tensor(tn(LLM_TENSOR_SSM_D, i), { d_inner }, []() { return 1.0f; });

                        // out_proj
                        cur.add_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), { d_inner, n_embd });
                    }
                }
                break;
            case LLM_ARCH_XVERSE:
            case LLM_ARCH_COMMAND_R:
            case LLM_ARCH_COHERE2:
            case LLM_ARCH_DBRX:
            case LLM_ARCH_OLMO:
            case LLM_ARCH_OLMO2:
            case LLM_ARCH_OLMOE:
            case LLM_ARCH_OPENELM:
            case LLM_ARCH_ARCTIC:
            case LLM_ARCH_DEEPSEEK:
            case LLM_ARCH_DEEPSEEK2:
            case LLM_ARCH_CHATGLM:
            case LLM_ARCH_GLM4:
            case LLM_ARCH_BITNET:
            case LLM_ARCH_T5:
            case LLM_ARCH_T5ENCODER:
            case LLM_ARCH_JAIS:
            case LLM_ARCH_NEMOTRON:
            case LLM_ARCH_EXAONE:
            case LLM_ARCH_RWKV6:
            case LLM_ARCH_RWKV6QWEN2:
                break;
            case LLM_ARCH_RWKV7:
                break;
                // TODO: proper initialization of the tensors
                // ref: https://github.com/BlinkDL/RWKV-LM/blob/247ce631e372b743ff496908d3e29df710506661/RWKV-v7/train_temp/src/model.py
                {
                    variants.push_back(model_variant(arch, "RWKV7"));
                    model_variant & cur = variants.back();

                    // TODO: use more realistic hyperparams

                    n_embd = 128; // TODO: why does this need to be bigger than head_size?

                    const uint32_t head_size = 64; // TODO: is this assumed by the ops?

                    const auto calc_lora_rank = [](uint32_t hidden_size, float exponent, float multiplier) {
                        return std::max((uint32_t) 1,
                                        (uint32_t) std::round(std::pow(hidden_size, exponent) * multiplier / 32.0f)) *
                               (uint32_t) 32;
                    };

                    const uint32_t n_lora_decay = calc_lora_rank(n_embd, 0.5, 1.8);
                    const uint32_t n_lora_iclr = calc_lora_rank(n_embd, 0.5, 1.8);
                    const uint32_t n_lora_value_res_mix = calc_lora_rank(n_embd, 0.5, 1.3);
                    const uint32_t n_lora_gate = calc_lora_rank(n_embd, 0.8, 0.6);
                    const uint32_t attn_hidden_size = n_embd;
                    const uint32_t ffn_size = n_ff;

                    cur.add_kv(LLM_KV_CONTEXT_LENGTH, (uint32_t) 1024 * 1024);
                    cur.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
                    cur.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
                    cur.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS, 1e-5f);
                    cur.add_kv(LLM_KV_WKV_HEAD_SIZE, head_size);
                    cur.add_kv(LLM_KV_ATTENTION_DECAY_LORA_RANK, n_lora_decay);
                    cur.add_kv(LLM_KV_ATTENTION_ICLR_LORA_RANK, n_lora_iclr);
                    cur.add_kv(LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, n_lora_value_res_mix);
                    cur.add_kv(LLM_KV_ATTENTION_GATE_LORA_RANK, n_lora_gate);
                    cur.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
                    cur.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, (uint32_t) 0);

                    add_tokenizer(cur, n_vocab);

                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // Block 0, LN0
                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
                    cur.add_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), {n_embd});

                    // output
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd});
                    cur.add_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab});

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_W0, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, n_lora_decay});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {n_lora_decay, n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_A0, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_A1, "weight", i), {n_embd, n_lora_iclr});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_A2, "weight", i), {n_lora_iclr, n_embd});

                        if (i == 0) {
                            // actually not used
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd});
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_iclr});
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_iclr, n_embd});
                        } else {
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd});
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_value_res_mix});
                            cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_value_res_mix, n_embd});
                        }

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_G1, "weight", i), {n_embd, n_lora_gate});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_G2, "weight", i), {n_lora_gate, n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 6});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_K_K, "weight", i), {attn_hidden_size});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_K_A, "weight", i), {attn_hidden_size});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_R_K, "weight", i), {attn_hidden_size});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd});

                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd});
                        cur.add_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size});

                        cur.add_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1});

                        cur.add_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), {n_embd, ffn_size});
                        cur.add_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), {ffn_size, n_embd});
                    }
                } break;
            case LLM_ARCH_ARWKV7:
            case LLM_ARCH_GRANITE:
            case LLM_ARCH_GRANITE_MOE:
            case LLM_ARCH_CHAMELEON:
            case LLM_ARCH_WAVTOKENIZER_DEC:
            case LLM_ARCH_PLM:
            case LLM_ARCH_BAILINGMOE:
            case LLM_ARCH_UNKNOWN:
                break;
        }
    }
};

struct reference_logits {
    int32_t n_vocab;
    int32_t prompt_len;
    std::vector<llama_token> inputs;
    std::vector<float> outputs;

    reference_logits(llama_context * ctx, int32_t seq_len, std::mt19937 & rng) {
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        std::uniform_int_distribution<llama_token> rand_token(0, n_vocab - 1);
        std::uniform_int_distribution<int32_t> rand_prompt_len(seq_len / 4, 3 * seq_len / 4);

        llama_batch batch = llama_batch_init(seq_len, 0, 1);

        outputs.reserve(n_vocab * seq_len);

        prompt_len = rand_prompt_len(rng);

        for (int32_t i = 0; i < prompt_len; ++i) {
            const llama_token token = rand_token(rng);
            inputs.push_back(token);

            common_batch_add(batch, token, i, { 0 }, true);
        }

        const int status_prompt = llama_decode(ctx, batch);
        GGML_ASSERT(status_prompt == 0);

        const float * output_prompt = llama_get_logits(ctx);
        GGML_ASSERT(output_prompt);
        outputs.insert(outputs.end(), output_prompt, output_prompt + n_vocab * prompt_len);

        for (int32_t i = prompt_len; i < seq_len; ++i) {
            common_batch_clear(batch);

            const llama_token token = rand_token(rng); // no real need to sample
            inputs.push_back(token);
            common_batch_add(batch, token, i, { 0 }, true);

            const int status = llama_decode(ctx, batch);
            GGML_ASSERT(status == 0);

            const float * output = llama_get_logits_ith(ctx, -1);
            GGML_ASSERT(output);
            outputs.insert(outputs.end(), output, output + n_vocab);
        }

        llama_batch_free(batch);
    }

    // TODO: unlink pos from indice into inputs
    // TODO: randomize which ouputs are enabled
    void add_to_batch(llama_batch & batch, llama_pos initial_pos, int32_t seq_len, llama_seq_id seq_id) {
        for (int32_t i = 0; i < seq_len; ++i) {
            llama_pos pos = initial_pos + i;
            common_batch_add(batch, inputs[pos], pos, { seq_id }, i == seq_len - 1);
        }
    }

    // returns normalized squared error of the output for the seq_id
    float validate_batch(llama_context * ctx, const llama_batch & batch, llama_seq_id seq_id) {
        float sumr2 = 0.0f;
        float sumo2 = 0.0f;

        float sumerr2 = 0.0f;

        llama_pos first_pos_error = -1;

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (batch.seq_id[i][0] == seq_id && batch.logits[i]) {
                const llama_pos pos = batch.pos[i];

                const float * out           = llama_get_logits_ith(ctx, i);
                const float * reference_out = &outputs[pos * n_vocab];

                GGML_ASSERT(out);

                for (int j = 0; j < n_vocab; ++j) {
                    sumo2 += out[j] * out[j];
                    sumr2 += reference_out[j] * reference_out[j];

                    const float err = (out[j] - reference_out[j]);
                    if (err > 0.0f) {
                        if (first_pos_error < 0 || pos < first_pos_error) {
                            first_pos_error = pos;
                        }
                    }

                    sumerr2 += err * err;
                }
            }
        }
        if (first_pos_error >= 0) {
            // fprintf(stderr, "Potential error in seq_id %i starting from pos %i\n", seq_id, first_pos_error);
        }

        const float denom = std::sqrt(sumr2) * std::sqrt(sumo2);

        return sumerr2 / (denom > 0.0f ? denom : 1.0f);
    }
};

struct reference_embd {
    size_t n_embd;
    std::vector<float> inputs;
    std::vector<float> outputs;

    // TODO: constructor
};

static void batch_add_embd(struct llama_batch & batch, const std::vector<float> & embd, llama_pos pos, llama_seq_id seq_id, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    memcpy(batch.embd + batch.n_tokens * embd.size(), embd.data(), sizeof(float) * embd.size());
    batch.pos[batch.n_tokens]       = pos;
    batch.n_seq_id[batch.n_tokens]  = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens]    = logits;

    batch.n_tokens++;
}

static void permute_from_ids(uint8_t * array, size_t elem_size, const std::vector<int32_t> & ids) {
    std::vector<uint8_t> tmp(elem_size * ids.size(), 0);

    for (size_t i = 0; i < ids.size(); ++i) {
        memcpy(tmp.data() + i * elem_size, array + ids[i] * elem_size, elem_size);
    }

    memcpy(array, tmp.data(), ids.size() * elem_size);
}

static void shuffle_batch(struct llama_batch & batch, std::mt19937 & rng) {
    std::vector<int32_t> ids(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        ids[i] = i;
    }

    std::shuffle(ids.begin(), ids.end(), rng);

    if (batch.token) {
        permute_from_ids((uint8_t *) batch.token, sizeof(*batch.token), ids);
    }
    if (batch.embd) {
        permute_from_ids((uint8_t *) batch.embd, sizeof(*batch.embd), ids);
    }
    permute_from_ids((uint8_t *) batch.pos, sizeof(*batch.pos), ids);
    permute_from_ids((uint8_t *) batch.n_seq_id, sizeof(*batch.n_seq_id), ids);
    permute_from_ids((uint8_t *) batch.seq_id, sizeof(*batch.seq_id), ids);
    permute_from_ids((uint8_t *) batch.logits, sizeof(*batch.logits), ids);
}

// TODO: use more args
int main(int argc, char ** argv) {

    std::string tmp_fname = "test-model-random-tmp.gguf";

    if (argc > 1) {
        // TODO: ensure it ends with .gguf
        std::string arg(argv[1]);
        const std::string suffix = ".gguf";

        if (suffix == arg.substr(arg.size() - suffix.size())) {
            tmp_fname = std::move(arg);
        } else {
            // TODO: print usage
            exit(1);
        }
    }

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // TODO: use specific backends
    llama_backend_init();

    // TODO: maybe use a faster rng algorithm
    std::mt19937 rng(42);

    // TODO: multiple sequences per token
    const int32_t n_batch = 3 * 512;
    const int32_t n_seq_len = 643; // prime number

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    // TODO: batch with embeddings

    std::vector<model_variant> model_variants;

    for (int i = 0; i < LLM_ARCH_UNKNOWN; ++i) {
        llm_arch arch = (llm_arch) i;
        model_variant::insert_from_arch(model_variants, arch);
    }

    // TODO: concurrent tests?

    for (const model_variant & variant : model_variants) {
        std::vector<std::string> splits = {};

        variant.write_to_file(tmp_fname.c_str(), rng);

        llama_model_params model_params = llama_model_default_params();

        model_params.check_tensors = true;

        llama_model * model = llama_model_load_from_file(tmp_fname.c_str(), model_params);

        GGML_ASSERT(model);

        // const auto n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        // const auto n_embd = llama_model_n_embd(model);

        for (int32_t n_seq_max : { 1, 2, 5 } ) {

            // TODO(later): context shift testing
            for (int32_t n_ctx : { n_seq_len * n_seq_max }) {

                std::vector<reference_logits> ref_outputs;

                {
                    llama_context_params ref_params = llama_context_default_params();
                    ref_params.n_batch = n_seq_len;
                    ref_params.n_ubatch = 1;
                    ref_params.n_ctx = n_seq_len;
                    ref_params.n_seq_max = 1;

                    llama_context * ref_ctx = llama_init_from_model(model, ref_params);

                    llama_memory_t mem = llama_get_memory(ref_ctx);

                    for (llama_seq_id seq_id = 0; seq_id < n_seq_max; ++seq_id) {
                        llama_memory_clear(mem, true);
                        ref_outputs.push_back(reference_logits(ref_ctx, n_seq_len, rng));
                    }

                    llama_free(ref_ctx);
                }

                for (bool shuffle : { false, true }) {

                    // skip shuffling the batch for non-recurrent models
                    // (simple splits don't handle shuffled batches correctly)
                    // FIXME: remove this
                    if (shuffle && !llama_model_is_recurrent(model)) {
                        continue;
                    }

                    for (int32_t n_ubatch : { 1, 2, 512 } ) {

                        std::vector<bool> valid(n_seq_max, true);

                        llama_context_params ctx_params = llama_context_default_params();
                        ctx_params.n_ctx = n_ctx;
                        ctx_params.n_seq_max = n_seq_max;
                        ctx_params.n_ubatch = n_ubatch;
                        ctx_params.n_batch = n_batch;

                        llama_context * ctx = llama_init_from_model(model, ctx_params);

                        common_batch_clear(batch);

                        std::set<llama_seq_id> seq_ids_in_batch;
                        std::vector<llama_pos> seq_id_n_past(n_seq_max, 0);

                        // start filling the batch with prompts
                        while (std::any_of(seq_id_n_past.begin(), seq_id_n_past.end(),
                                           [](llama_pos p) { return p < n_seq_len; })) {
                            for (llama_seq_id seq_id = 0; seq_id < n_seq_max; ++seq_id) {
                                if (seq_id_n_past[seq_id] >= ref_outputs[seq_id].prompt_len) {
                                    continue;
                                }

                                if (batch.n_tokens < n_batch) {
                                    const int64_t seq_len =
                                        std::min(n_batch - batch.n_tokens,
                                                 ref_outputs[seq_id].prompt_len - seq_id_n_past[seq_id]);

                                    ref_outputs[seq_id].add_to_batch(batch, seq_id_n_past[seq_id], seq_len, seq_id);
                                    seq_ids_in_batch.insert(seq_id);
                                    seq_id_n_past[seq_id] += seq_len;
                                }
                            }
                            if (shuffle) {
                                shuffle_batch(batch, rng);
                            }

                            llama_decode(ctx, batch);

                            for (llama_seq_id seq_id = 0; seq_id < n_seq_max; ++seq_id) {
                                float err = ref_outputs[seq_id].validate_batch(ctx, batch, seq_id);
                                if (!isfinite(err) || err > 1.0f / 1024.0f) {
                                    fprintf(stderr, "Error for seq_id %i is %f\n", seq_id, err);
                                    valid[seq_id] = false;
                                }
                            }

                            common_batch_clear(batch);

                            GGML_ASSERT(n_seq_max <= n_batch); // not handling splitting this across batches here

                            // cont batching
                            for (llama_seq_id s : seq_ids_in_batch) {
                                llama_pos & pos = seq_id_n_past[s];
                                if (pos >= n_seq_len) {
                                    continue;
                                }
                                ref_outputs[s].add_to_batch(batch, pos, 1, s);
                                pos += 1;
                            }
                        }

                        fprintf(stdout,
                                "Comparing output for '%s', with shuffle=%i, n_seq_max=%i, n_ctx=%i, n_ubatch=%i: ",
                                variant.name.c_str(), shuffle, n_seq_max, n_ctx, n_ubatch);
                        if (std::all_of(valid.begin(), valid.end(), [](bool v) { return v; })) {
                            fprintf(stdout, "\033[1;32mOK\033[0m\n");
                        } else {
                            fprintf(stdout, "(%zu%%) \033[1;31mFAILED\033[0m\n",
                                    std::count_if(valid.begin(), valid.end(), [](bool v) { return v == false; }) * 100 / valid.size());
                            // cleanup and exit on first failure
                            llama_free(ctx);
                            llama_model_free(model);
                            llama_batch_free(batch);
                            exit(1);
                        }

                        // TODO: use seq_rm, seq_cp, etc. to test if they work properly

                        // TODO: test pooled embeddings

                        llama_free(ctx);
                    }
                }
            }
        }

        llama_model_free(model);
    }

    llama_batch_free(batch);

    return 0;
}
