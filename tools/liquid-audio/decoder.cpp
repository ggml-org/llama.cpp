#include "decoder.h"

#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "log.h"

#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <vector>

namespace liquid {
namespace audio {

namespace {

ggml_tensor * build_rms_norm(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * mw, const float eps) {
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, mw);

    return cur;
}

ggml_tensor * build_ffn(ggml_context * ctx0,
                        ggml_tensor *  cur,
                        ggml_tensor *  ffn_up,
                        ggml_tensor *  ffn_gate,
                        ggml_tensor *  ffn_down) {
    auto * up      = ggml_mul_mat(ctx0, ffn_up, cur);
    auto * gate    = ggml_mul_mat(ctx0, ffn_gate, cur);
    auto * swiglu  = ggml_swiglu_split(ctx0, gate, up);
    auto * ffn_out = ggml_mul_mat(ctx0, ffn_down, swiglu);

    return ffn_out;
}

struct audio_decoder_ggml_ctx {
    gguf_context * ctx_gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_context * ctx_gf   = nullptr;

    std::vector<ggml_backend_t>             backends;
    std::vector<ggml_backend_buffer_type_t> bufts;

    ggml_backend_buffer_t  buf = nullptr;
    ggml_backend_sched_ptr sched;

    ggml_cgraph *        gf = nullptr;
    std::vector<uint8_t> buf_compute_meta;
    int                  max_nodes = 16 * 1024;

    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, uint32_t>      hyperparameters;

    explicit audio_decoder_ggml_ctx(const common_params & params) {
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        GGML_ASSERT(backend_cpu);

        if (params.mmproj_use_gpu) {
            ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            if (!backend_gpu) {
                backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
            if (backend_gpu) {
                LOG_INF("%s: using %s backend\n", __func__, ggml_backend_name(backend_gpu));
                backends.push_back(backend_gpu);
                bufts.push_back(ggml_backend_get_default_buffer_type(backend_gpu));
            }
        }

        // CPU must be last (scheduler requirement)
        backends.push_back(backend_cpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        if (backends.size() == 1) {
            LOG_INF("%s: using CPU backend\n", __func__);
        } else {
            LOG_INF("%s: using GPU+CPU backend\n", __func__);
        }

        sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), max_nodes, false, true));
        buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    void load_gguf(const char * fname) {
        ggml_context * meta = nullptr;

        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_init_from_file(fname, params);

        // load tensors
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);

        std::vector<uint8_t> read_buf;
        ggml_init_params     ggml_params = {
            /*.mem_size   =*/(n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
        };

        ctx_data = ggml_init(ggml_params);
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            ggml_free(meta);
            throw std::runtime_error("cannot open model file for loading tensors");
        }

        // hyperparameters
        for (const auto & key : { "depthformer_n_layer", "depthformer_n_embd" }) {
            auto key_id = gguf_find_key(ctx_gguf, key);
            if (key_id < 0) {
                throw std::runtime_error(string_format("key not found in gguf: %s", key));
            }
            hyperparameters[key] = gguf_get_val_u32(ctx_gguf, key_id);
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char *  name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * t    = ggml_get_tensor(meta, name);
            ggml_tensor * cur  = ggml_dup_tensor(ctx_data, t);
            ggml_set_name(cur, name);
            tensors.insert({ name, cur });
        }

        // alloc memory on primary backend (GPU if available)
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, bufts.front());
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < n_tensors; ++i) {
            const char *  name   = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * cur    = ggml_get_tensor(ctx_data, name);
            const size_t  offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            // printf("%s: Loading tensor \"%s\"\n", __func__, name);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                ggml_free(meta);
                throw std::runtime_error(string_format("failed to seek for tensor: %s", name));
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(bufts.front())) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        LOG_INF("%s: Loaded %d tensors from %s\n", __func__, n_tensors, fname);
        fin.close();

        ggml_free(meta);
    }

    /**
     * Build a cgraph using the given builder function.
     *
     * The built cgraph will be stored in `ctx.gf`
     */
    void build_graph(const std::function<void(ggml_context *, ggml_cgraph *)> & builder_fn) {
        ggml_free(ctx_gf);
        struct ggml_init_params params = {
            /*.mem_size   =*/buf_compute_meta.size(),
            /*.mem_buffer =*/buf_compute_meta.data(),
            /*.no_alloc   =*/true,
        };

        ctx_gf = ggml_init(params);
        ggml_backend_sched_reset(sched.get());
        gf = ggml_new_graph_custom(ctx_gf, max_nodes, false);

        builder_fn(ctx_gf, gf);
        ggml_backend_sched_alloc_graph(sched.get(), gf);
    }

    ggml_status compute() const { return ggml_backend_sched_graph_compute(sched.get(), gf); }

    void set_tensor_data(const std::string & name, const void * data) const {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }

    std::pair<ggml_tensor *, std::vector<uint8_t>> get_tensor_data(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return std::make_pair(t, data);
    }

    ggml_tensor * get_weight(const char * fmt, ...) {
        std::vector<char> str(128);
        va_list           va;
        va_start(va, fmt);
        vsnprintf(str.data(), 128, fmt, va);
        va_end(va);
        auto it = tensors.find(str.data());
        if (it == tensors.end()) {
            throw std::runtime_error(string_format("weight tensor not found: %s", str.data()));
        }
        return it->second;
    }

    ~audio_decoder_ggml_ctx() {
        ggml_free(ctx_data);
        gguf_free(ctx_gguf);
        ggml_backend_buffer_free(buf);
        for (auto * backend : backends) {
            ggml_backend_free(backend);
        }
    }
};

template <typename Container>
std::vector<float> run_graph(
    audio_decoder_ggml_ctx &                                                           ctx,
    const Container &                                                                  data,
    const std::function<ggml_tensor *(ggml_context *, ggml_cgraph *, ggml_tensor *)> & builder_fn) {
    ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf) {
        using T = typename Container::value_type;

        ggml_tensor * input = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, data.size());
        } else if constexpr (std::is_same_v<T, int32_t>) {
            input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, data.size());
        } else {
            static_assert(!sizeof(T), "Unsupported type");
        }
        GGML_ASSERT(input);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        auto * output = builder_fn(ctx0, gf, input);

        ggml_build_forward_expand(gf, output);
        ggml_set_name(output, "output");
        ggml_set_output(output);
    });

    ctx.set_tensor_data("input", data.data());

    ctx.compute();

    ggml_tensor * t = ggml_get_tensor(ctx.ctx_gf, "output");
    GGML_ASSERT(t);
    std::vector<float> output(ggml_nelements(t));
    ggml_backend_tensor_get(t, output.data(), 0, ggml_nbytes(t));
    return output;
}

// used for KV and conv cache
class Cache {
  public:
    void init(int n_tensors) {
        GGML_ASSERT(!ctx);
        ggml_init_params params = {
            n_tensors * ggml_tensor_overhead(), nullptr,
            true  // no_alloc
        };
        ctx = ggml_init(params);
    }

    ~Cache() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    ggml_tensor * new_tensor(enum ggml_type type, const std::vector<int64_t> & shape) {
        GGML_ASSERT(ctx);
        return ggml_new_tensor(ctx, type, shape.size(), shape.data());
    }

    void alloc(ggml_backend_buffer_type_t buft) {
        GGML_ASSERT(!buf);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    }

  private:
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

class DepthformerModel {
  public:
    void init(audio_decoder_ggml_ctx & ctx) {
        // cache
        int n_cache_tensors = config.n_layer * 2;  // kv per layer
        cache.init(n_cache_tensors);

        weights.layers.resize(config.n_layer);
        for (int il = 0; il < config.n_layer; il++) {
            auto & l = weights.layers[il];

            l.operator_norm = ctx.get_weight("depthformer.layers.%d.operator_norm.weight", il);
            l.wqkv          = ctx.get_weight("depthformer.layers.%d.operator.qkv_proj.weight", il);
            l.attn_q_norm   = ctx.get_weight("depthformer.layers.%d.operator.attention.q_layernorm.weight", il);
            l.attn_k_norm   = ctx.get_weight("depthformer.layers.%d.operator.attention.k_layernorm.weight", il);
            l.wo            = ctx.get_weight("depthformer.layers.%d.operator.out_proj.weight", il);
            l.ffn_norm      = ctx.get_weight("depthformer.layers.%d.ffn_norm.weight", il);
            l.w1            = ctx.get_weight("depthformer.layers.%d.feed_forward.w1.weight", il);
            l.w2            = ctx.get_weight("depthformer.layers.%d.feed_forward.w2.weight", il);
            l.w3            = ctx.get_weight("depthformer.layers.%d.feed_forward.w3.weight", il);
            l.k_cache = cache.new_tensor(GGML_TYPE_F32, { config.n_embd_head, config.n_head_kv, config.max_seq_len });
            l.v_cache = cache.new_tensor(GGML_TYPE_F32, { config.n_embd_head, config.n_head_kv, config.max_seq_len });
        }

        cache.alloc(ctx.bufts.front());
    }

    void reset() { n_past = 0; }

    void advance(int n_tokens) { n_past += n_tokens; }

    ggml_tensor * graph(ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * cur) {
        auto &    c        = config;
        const int n_tokens = cur->ne[1];

        for (int i = 0; i < c.n_layer; ++i) {
            const auto & l = weights.layers[i];
            auto *       x = cur;

            // operator_norm
            cur = build_rms_norm(ctx0, x, l.operator_norm, c.f_norm_rms_eps);

            // attention
            {
                ggml_tensor * qkv = ggml_mul_mat(ctx0, l.wqkv, cur);

                ggml_tensor * q = ggml_view_3d(ctx0, qkv, c.n_embd_head, c.n_head, n_tokens,
                                               c.n_embd_head * ggml_element_size(qkv), qkv->nb[1], 0);
                ggml_tensor * k = ggml_view_3d(ctx0, qkv, c.n_embd_head, c.n_head_kv, n_tokens,
                                               c.n_embd_head * ggml_element_size(qkv), qkv->nb[1],
                                               c.n_embd_head * c.n_head * ggml_element_size(qkv));
                ggml_tensor * v = ggml_view_3d(ctx0, qkv, c.n_embd_head, c.n_head_kv, n_tokens,
                                               c.n_embd_head * ggml_element_size(qkv), qkv->nb[1],
                                               c.n_embd_head * (c.n_head + c.n_head_kv) * ggml_element_size(qkv));

                q = build_rms_norm(ctx0, q, l.attn_q_norm, c.f_norm_rms_eps);
                k = build_rms_norm(ctx0, k, l.attn_k_norm, c.f_norm_rms_eps);

                auto   n_rot   = c.n_embd_head;
                auto * inp_pos = ggml_cast(ctx0, ggml_arange(ctx0, n_past, n_past + n_tokens, 1), GGML_TYPE_I32);
                q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, c.rope_type, c.n_ctx_orig, c.rope_freq_base,
                                  c.rope_freq_scale, 0, 1, 0, 0);
                k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, c.rope_type, c.n_ctx_orig, c.rope_freq_base,
                                  c.rope_freq_scale, 0, 1, 0, 0);

                auto * k_cache = l.k_cache;
                auto * v_cache = l.v_cache;

                // write current k/v to cache at position n_past
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, k,
                                                       ggml_view_2d(ctx0, k_cache, k_cache->ne[0], k_cache->ne[1],
                                                                    k_cache->nb[1], n_past * k_cache->nb[2])));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, v,
                                                       ggml_view_2d(ctx0, v_cache, v_cache->ne[0], v_cache->ne[1],
                                                                    v_cache->nb[1], n_past * v_cache->nb[2])));

                // read k/v from cache [0..n_tokens]
                k = ggml_view_3d(ctx0, k_cache, k_cache->ne[0], k_cache->ne[1], n_past + n_tokens, k_cache->nb[1],
                                 k_cache->nb[2], 0);
                v = ggml_view_3d(ctx0, v_cache, v_cache->ne[0], v_cache->ne[1], n_past + n_tokens, v_cache->nb[1],
                                 v_cache->nb[2], 0);

                float kq_scale = 1.0f / sqrtf((float) c.n_embd_head);

                // manual attention, faster for small size
                {
                    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                    k = ggml_permute(ctx0, k, 0, 2, 1, 3);

                    auto * kq = ggml_mul_mat(ctx0, k, q);
                    kq        = ggml_scale(ctx0, kq, kq_scale);
                    kq        = ggml_soft_max(ctx0, kq);

                    v = ggml_permute(ctx0, v, 1, 2, 0, 3);
                    v = ggml_cont(ctx0, v);

                    auto * kqv = ggml_mul_mat(ctx0, v, kq);
                    kqv        = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                    cur        = ggml_cont_2d(ctx0, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2]);
                }

                cur = ggml_mul_mat(ctx0, l.wo, cur);
            }

            cur = ggml_add(ctx0, cur, x);

            auto * ffn_norm = build_rms_norm(ctx0, cur, l.ffn_norm, c.f_norm_rms_eps);
            auto * ffn_out  = build_ffn(ctx0, ffn_norm, l.w3, l.w1, l.w2);

            cur = ggml_add(ctx0, cur, ffn_out);
        }

        return cur;
    }

  private:
    struct {
        // TODO(tarek): read from gguf
        int n_layer = 6;
        int n_embd  = 1024;

        int   n_embd_head    = 32;
        int   n_head         = 32;
        int   n_head_kv      = 8;
        float f_norm_rms_eps = 1e-5f;

        int   n_ctx_orig      = 128000;
        int   rope_type       = LLAMA_ROPE_TYPE_NORM;
        float rope_freq_base  = 1000000.0f;
        float rope_freq_scale = 1.f;

        int max_seq_len = 8;
    } config;

    struct {
        struct Layer {
            ggml_tensor * operator_norm = nullptr;
            ggml_tensor * wqkv          = nullptr;
            ggml_tensor * attn_q_norm   = nullptr;
            ggml_tensor * attn_k_norm   = nullptr;
            ggml_tensor * wo            = nullptr;
            ggml_tensor * ffn_norm      = nullptr;
            ggml_tensor * w1            = nullptr;
            ggml_tensor * w2            = nullptr;
            ggml_tensor * w3            = nullptr;

            ggml_tensor * k_cache = nullptr;
            ggml_tensor * v_cache = nullptr;
        };

        std::vector<Layer> layers;
    } weights;

    // state
    Cache   cache;
    int32_t n_past = 0;
};

class DecoderModel {
  public:
    void init(audio_decoder_ggml_ctx & ctx) {
        depthformer_model.init(ctx);

        weights.depth_linear_w = ctx.get_weight("depth_linear.weight");
        weights.depth_linear_b = ctx.get_weight("depth_linear.bias");

        weights.depth_embd_layers.resize(config.n_codebook);
        for (int ic = 0; ic < config.n_codebook; ic++) {
            auto & cl = weights.depth_embd_layers[ic];

            cl.norm      = ctx.get_weight("depth_embeddings.%d.embedding_norm.weight", ic);
            cl.embd      = ctx.get_weight("depth_embeddings.%d.embedding.weight", ic);
            cl.to_logits = ctx.get_weight("depth_embeddings.%d.to_logits.weight", ic);
        }

        weights.audio_embedding.norm      = ctx.get_weight("audio_embedding.embedding_norm.weight");
        weights.audio_embedding.embd      = ctx.get_weight("audio_embedding.embedding.weight");
        weights.audio_embedding.to_logits = ctx.get_weight("audio_embedding.to_logits.weight");

        weights.audio_tokenizer.embd = ctx.get_weight("emb.emb.weight");
    }

    ggml_tensor * graph(ggml_context * ctx0, ggml_tensor * cur, ggml_cgraph * gf, int j, llama_token prev_token) {
        auto & depth_embd_layer = weights.depth_embd_layers[j];

        // calculate depthformer_in chunk for codebook j
        {
            auto * w        = weights.depth_linear_w;
            auto * b        = weights.depth_linear_b;
            auto   n_embd_d = depth_embd_layer.embd->ne[0];
            cur = ggml_mul_mat(ctx0, ggml_view_2d(ctx0, w, w->ne[0], n_embd_d, w->nb[1], j * n_embd_d * w->nb[1]), cur);
            cur = ggml_add(ctx0, cur, ggml_view_1d(ctx0, b, n_embd_d, j * n_embd_d * b->nb[0]));
        }

        if (j > 0) {
            auto * prev_token_tensor = ggml_cast(ctx0, ggml_arange(ctx0, prev_token, prev_token + 1, 1), GGML_TYPE_I32);
            auto * depthformer_token = ggml_get_rows(ctx0, weights.depth_embd_layers[j - 1].embd, prev_token_tensor);
            cur                      = ggml_add(ctx0, cur, depthformer_token);
        }

        cur = depthformer_model.graph(ctx0, gf, cur);

        cur = build_rms_norm(ctx0, cur, depth_embd_layer.norm, config.f_norm_rms_eps);

        cur = ggml_mul_mat(ctx0, depth_embd_layer.to_logits, cur);

        return cur;
    }

    ggml_tensor * embed(ggml_context * ctx0, ggml_tensor * input) const {
        ggml_tensor * codebook_offsets = ggml_arange(ctx0, 0, config.n_vocab * config.n_codebook, config.n_vocab);
        // add codebook_offsets
        auto *        out_tokens_offsets =
            ggml_cast(ctx0, ggml_add(ctx0, ggml_cast(ctx0, input, GGML_TYPE_F32), codebook_offsets), GGML_TYPE_I32);

        // sum
        auto * out_embd = ggml_get_rows(ctx0, weights.audio_embedding.embd, out_tokens_offsets);
        out_embd        = ggml_cont(ctx0, ggml_permute(ctx0, out_embd, 1, 0, 2, 3));
        out_embd        = ggml_sum_rows(ctx0, out_embd);
        out_embd        = ggml_reshape_1d(ctx0, out_embd, config.n_embd);

        return out_embd;
    }

    ggml_tensor * embed_for_detokenizer(ggml_context * ctx0, ggml_tensor * input) const {
        const int n_codes         = 8;
        const int n_output_tokens = 6;

        ggml_tensor * cur = input;

        GGML_ASSERT(!(cur->ne[0] % n_codes));
        ggml_tensor * codes = ggml_reshape_2d(ctx0, cur, n_codes, cur->ne[0] / n_codes);

        // TODO(tarek): remove transpose
        codes = ggml_transpose(ctx0, codes);

        cur = codes;

        // embedding
        {
            int           n_embd_code = weights.audio_tokenizer.embd->ne[1] / n_codes;
            ggml_tensor * offsets =
                ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0, n_embd_code * n_codes, n_embd_code), 1, n_codes);
            auto * x        = ggml_cast(ctx0, cur, GGML_TYPE_F32);
            auto * offset_x = ggml_cast(ctx0, ggml_add(ctx0, x, offsets), GGML_TYPE_I32);

            offset_x = ggml_reshape_1d(ctx0, offset_x, x->ne[0] * x->ne[1]);

            auto * embedding = ggml_get_rows(ctx0, weights.audio_tokenizer.embd, offset_x);
            embedding        = ggml_reshape_3d(ctx0, embedding, embedding->ne[0], x->ne[0], n_codes);
            embedding        = ggml_cont(ctx0, ggml_permute(ctx0, embedding, 2, 1, 0, 3));
            embedding        = ggml_mean(ctx0, embedding);
            embedding        = ggml_cont(ctx0, ggml_permute(ctx0, embedding, 2, 1, 0, 3));
            cur              = embedding;
        }

        // upsample
        {
            auto upsample_size = n_output_tokens * cur->ne[1];
            cur                = ggml_interpolate(ctx0, cur, cur->ne[0], upsample_size, cur->ne[2], cur->ne[3],
                                                  0);  // linear interp
        }

        return cur;
    }

    audio_token_t sample(audio_decoder_ggml_ctx & ctx, const std::vector<float> & embd, llama_sampler * smpl) {
        audio_token_t token;
        llama_token   prev_token = -1;

        GGML_ASSERT((int) token.size() == config.n_codebook);
        depthformer_model.reset();
        for (int i = 0; i < config.n_codebook; ++i) {
            {
                auto depthformer_logits =
                    run_graph(ctx, embd, [&](ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * cur) {
                        return graph(ctx0, cur, gf, i, prev_token);
                    });

                std::vector<llama_token_data> cur;
                cur.reserve(config.n_vocab);
                for (llama_token token_id = 0; token_id < config.n_vocab; token_id++) {
                    cur.emplace_back(llama_token_data{ token_id, depthformer_logits[token_id], 0.0f });
                }

                llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

                llama_sampler_apply(smpl, &cur_p);

                GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

                prev_token = cur_p.data[cur_p.selected].id;

                token[i] = prev_token;
            }
            depthformer_model.advance(1);
        }

        return token;
    }

  private:
    struct {
        int n_embd     = 2048;
        int n_codebook = 8;
        int n_vocab    = 2049;

        float f_norm_rms_eps = 1e-5f;
    } config;

    struct {
        ggml_tensor * depth_linear_w = nullptr;
        ggml_tensor * depth_linear_b = nullptr;

        struct EmbdLayer {
            ggml_tensor * norm      = nullptr;
            ggml_tensor * embd      = nullptr;
            ggml_tensor * to_logits = nullptr;
        };

        std::vector<EmbdLayer> depth_embd_layers;
        EmbdLayer              audio_embedding;

        struct {
            ggml_tensor * embd;
        } audio_tokenizer;
    } weights;

    DepthformerModel depthformer_model;
};

}  // namespace

class Decoder::DecoderImpl {
  public:
    DecoderModel decoder_model;

    audio_decoder_ggml_ctx ctx;

    DecoderImpl(const common_params & params) : ctx(params) {
        const std::string & gguf_path = params.vocoder.model.path;

        ctx.load_gguf(gguf_path.c_str());

        decoder_model.init(ctx);
    }

    template <typename T> ggml_type get_ggml_type() {
        if constexpr (std::is_same_v<T, float>) {
            return GGML_TYPE_F32;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return GGML_TYPE_I32;
        } else {
            static_assert(!sizeof(T *), "Unsupported type");
        }
    }

    audio_token_t sample_audio_frame(const std::vector<float> & embd, float temperature, int top_k) {
        // sampling
        const bool                        greedy = temperature <= 0;
        struct llama_sampler_chain_params sparams;
        llama_sampler *                   smpl = llama_sampler_chain_init(sparams);
        if (greedy) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
        }

        audio_token_t token = decoder_model.sample(ctx, embd, smpl);

        llama_sampler_free(smpl);

        return token;
    }

    std::vector<float> embed(const audio_token_t & token) {
        return run_graph(ctx, token, [&](ggml_context * ctx0, ggml_cgraph *, ggml_tensor * input) {
            return decoder_model.embed(ctx0, input);
        });
    }

    std::vector<float> embed_for_detokenizer(const audio_token_t & token) {
        return run_graph(ctx, token, [&](ggml_context * ctx0, ggml_cgraph *, ggml_tensor * input) {
            return decoder_model.embed_for_detokenizer(ctx0, input);
        });
    }

    void set_threadpool(ggml_threadpool * tp, int n_threads) {
        auto * backend_cpu = ctx.backends.back();
        GGML_ASSERT(backend_cpu);
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        if (auto * set_threadpool_fn = (void (*)(ggml_backend_t, ggml_threadpool *)) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cpu_set_threadpool");
            set_threadpool_fn && tp) {
            set_threadpool_fn(backend_cpu, tp);
        }
        if (auto set_n_threads_fn =
                (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            set_n_threads_fn) {
            set_n_threads_fn(backend_cpu, n_threads);
        }
    }
};

Decoder::Decoder(const common_params & params) : pimpl(std::make_unique<DecoderImpl>(params)) {}

Decoder::~Decoder() = default;

audio_token_t Decoder::sample_audio_frame(const std::vector<float> & embedding, float temperature, int top_k) {
    return pimpl->sample_audio_frame(embedding, temperature, top_k);
}

std::vector<float> Decoder::embed(const audio_token_t & token) {
    return pimpl->embed(token);
}

std::vector<float> Decoder::embed_for_detokenizer(const audio_token_t & token) {
    return pimpl->embed_for_detokenizer(token);
}

void Decoder::set_threadpool(ggml_threadpool * threadpool, int n_threads) {
    pimpl->set_threadpool(threadpool, n_threads);
}
}  // namespace audio
}  // namespace liquid
