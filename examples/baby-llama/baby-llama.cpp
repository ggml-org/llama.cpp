#include "ggml.h"
#include "train.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef LLAMA_DEFAULT_RMS_EPS
constexpr float rms_norm_eps = LLAMA_DEFAULT_RMS_EPS;
#else
constexpr float rms_norm_eps = 5e-6f;
#endif

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static struct ggml_tensor * randomize_tensor(
    struct ggml_tensor * tensor, int ndims, const int64_t ne[], float fmin, float fmax
) {
    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return tensor;
}

struct llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;

    bool operator!=(const llama_hparams & other) const {
        return memcmp(this, &other, sizeof(llama_hparams));
    }
};

static uint32_t get_n_ff(const struct llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}

struct llama_hparams_lora {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    uint32_t n_lora  = 64;

    bool operator!=(const llama_hparams_lora & other) const {
        return memcmp(this, &other, sizeof(llama_hparams_lora)) != 0;
    }
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_layer_lora {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wqa;
    struct ggml_tensor * wqb;
    struct ggml_tensor * wka;
    struct ggml_tensor * wkb;
    struct ggml_tensor * wva;
    struct ggml_tensor * wvb;
    struct ggml_tensor * woa;
    struct ggml_tensor * wob;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};


struct llama_kv_cache {
    struct ggml_context * ctx = NULL;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    // llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache
};

struct llama_model {
    struct ggml_context * ctx = NULL;

    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;
};

struct llama_model_lora {
    struct ggml_context * ctx = NULL;

    llama_hparams_lora hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * outputa;
    struct ggml_tensor * outputb;

    std::vector<llama_layer_lora> layers;
};

static void init_model(struct llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);

    struct ggml_context * ctx = model->ctx;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); // ("tok_embeddings.weight", {n_embd, n_vocab});
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);          // ("norm.weight",           {n_embd});
    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); // ("output.weight",         {n_embd, n_vocab});

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        // std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); // (layers_i + ".attention_norm.weight", {n_embd});

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wq.weight", {n_embd, n_embd});
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wk.weight", {n_embd, n_embd});
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wv.weight", {n_embd, n_embd});
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wo.weight", {n_embd, n_embd});

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);       // (layers_i + ".ffn_norm.weight", {n_embd});

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff});
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);     // (layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd});
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff});
    }
}


static void init_model_lora(struct llama_model_lora * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_mult  = hparams.n_mult;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;
    const uint32_t n_lora  = hparams.n_lora;

    const uint32_t n_ff = ((2*(4*n_embd)/3 + n_mult - 1)/n_mult)*n_mult;

    struct ggml_context * ctx = model->ctx;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); // ("tok_embeddings.weight", {n_embd, n_vocab});
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);          // ("norm.weight",           {n_embd});
    model->outputa        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_lora, n_vocab); // ("output.weight",         {n_embd, n_vocab});
    model->outputb        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,  n_lora); // ("output.weight",         {n_embd, n_vocab});

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        // std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); // (layers_i + ".attention_norm.weight", {n_embd});

        layer.wqa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_lora, n_embd);    // (layers_i + ".attention.wq.weight", {n_embd, n_embd});
        layer.wqb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_lora);    // (layers_i + ".attention.wq.weight", {n_embd, n_embd});
        layer.wka = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_lora, n_embd);    // (layers_i + ".attention.wk.weight", {n_embd, n_embd});
        layer.wkb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_lora);    // (layers_i + ".attention.wk.weight", {n_embd, n_embd});
        layer.wva = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_lora, n_embd);    // (layers_i + ".attention.wv.weight", {n_embd, n_embd});
        layer.wvb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_lora);    // (layers_i + ".attention.wv.weight", {n_embd, n_embd});
        layer.woa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_lora, n_embd);    // (layers_i + ".attention.wo.weight", {n_embd, n_embd});
        layer.wob = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_lora);    // (layers_i + ".attention.wo.weight", {n_embd, n_embd});

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);       // (layers_i + ".ffn_norm.weight", {n_embd});

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff});
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);     // (layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd});
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff});
    }
}

static void set_param_model(struct llama_model * model) {
    const auto& hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->tok_embeddings);
    ggml_set_param(ctx, model->norm);
    ggml_set_param(ctx, model->output);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        ggml_set_param(ctx, layer.attention_norm);
        ggml_set_param(ctx, layer.wq);
        ggml_set_param(ctx, layer.wk);
        ggml_set_param(ctx, layer.wv);
        ggml_set_param(ctx, layer.wo);
        ggml_set_param(ctx, layer.ffn_norm);
        ggml_set_param(ctx, layer.w1);
        ggml_set_param(ctx, layer.w2);
        ggml_set_param(ctx, layer.w3);
    }
}

static void set_param_model_lora(struct llama_model_lora * model) {
    const auto& hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->tok_embeddings);
    ggml_set_param(ctx, model->norm);
    ggml_set_param(ctx, model->outputa);
    ggml_set_param(ctx, model->outputb);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        ggml_set_param(ctx, layer.attention_norm);
        ggml_set_param(ctx, layer.wqa);
        ggml_set_param(ctx, layer.wqb);
        ggml_set_param(ctx, layer.wka);
        ggml_set_param(ctx, layer.wkb);
        ggml_set_param(ctx, layer.wva);
        ggml_set_param(ctx, layer.wvb);
        ggml_set_param(ctx, layer.woa);
        ggml_set_param(ctx, layer.wob);
        ggml_set_param(ctx, layer.ffn_norm);
        ggml_set_param(ctx, layer.w1);
        ggml_set_param(ctx, layer.w2);
        ggml_set_param(ctx, layer.w3);
    }
}

static void randomize_model(struct llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings , rnd);
    randomize_tensor_normal(model->norm           , rnd);
    randomize_tensor_normal(model->output         , rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, rnd);

        randomize_tensor_normal(layer.wq, rnd);
        randomize_tensor_normal(layer.wk, rnd);
        randomize_tensor_normal(layer.wv, rnd);
        randomize_tensor_normal(layer.wo, rnd);

        randomize_tensor_normal(layer.ffn_norm, rnd);

        randomize_tensor_normal(layer.w1, rnd);
        randomize_tensor_normal(layer.w2, rnd);
        randomize_tensor_normal(layer.w3, rnd);
    }

    free_random_normal_distribution(rnd);
}


static void randomize_model_lora(
    struct llama_model_lora * model, int seed, float mean, float std, float min, float max
) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings, rnd);
    randomize_tensor_normal(model->norm          , rnd);
    randomize_tensor_normal(model->outputa       , rnd);
    randomize_tensor_normal(model->outputb       , rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, rnd);

        randomize_tensor_normal(layer.wqa, rnd);
        randomize_tensor_normal(layer.wqb, rnd);
        randomize_tensor_normal(layer.wka, rnd);
        randomize_tensor_normal(layer.wkb, rnd);
        randomize_tensor_normal(layer.wva, rnd);
        randomize_tensor_normal(layer.wvb, rnd);
        randomize_tensor_normal(layer.woa, rnd);
        randomize_tensor_normal(layer.wob, rnd);

        randomize_tensor_normal(layer.ffn_norm, rnd);

        randomize_tensor_normal(layer.w1, rnd);
        randomize_tensor_normal(layer.w2, rnd);
        randomize_tensor_normal(layer.w3, rnd);
    }

    free_random_normal_distribution(rnd);
}

static void init_kv_cache(struct llama_kv_cache* cache, struct llama_model * model, int n_batch) {
    const auto & hparams = model->hparams;

    const uint32_t n_ctx   = hparams.n_ctx;
    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx*n_batch;
    const int64_t n_elements = n_embd*n_mem;

    // cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    // struct ggml_init_params params;
    // params.mem_size   = cache.buf.size;
    // params.mem_buffer = cache.buf.addr;
    // params.no_alloc   = false;
    if (!cache->ctx) {
        struct ggml_init_params params;
        params.mem_size   = 2u*n_elements*ggml_type_size(GGML_TYPE_F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params);

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            exit(1);
        }
    }

    cache->k = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);
    cache->v = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);
}

static bool init_kv_cache_lora(struct llama_kv_cache* cache, struct llama_model_lora * model, int n_batch) {
    const auto & hparams = model->hparams;

    const uint32_t n_ctx   = hparams.n_ctx;
    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx*n_batch;
    const int64_t n_elements = n_embd*n_mem;

    // cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    // struct ggml_init_params params;
    // params.mem_size   = cache.buf.size;
    // params.mem_buffer = cache.buf.addr;
    // params.no_alloc   = false;
    if (!cache->ctx) {
        struct ggml_init_params params;
        params.mem_size   = 2u*n_elements*ggml_type_size(GGML_TYPE_F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params);

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            return false;
        }
    }

    cache->k = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);
    cache->v = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);

    return true;
}

static struct ggml_tensor * forward(
    struct llama_model    * model,
    struct llama_kv_cache * cache,
    struct ggml_context   * ctx0,
    struct ggml_cgraph    * gf,
    struct ggml_tensor    * tokens_input,
    const  int              n_tokens,
    const  int              n_past
) {
    const int N = n_tokens;

    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < N; ++i) {
            data[i] = n_past + i;
        }
    }

    // inpL shape [n_embd,N,1,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Kcur shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * Qcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N), KQ_pos, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N), KQ_pos, n_rot, 0);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [n_embd, N, 1, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wv, cur), n_embd, N)));

                // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
                // kv_self.v shape [n_embd * n_ctx * n_layer, 1]
                // k         shape [n_embd * N, 1]   == kv_self.k[:,n_past:n_past+N,il,0]
                // v         shape [N, n_embd, 1, 1] == kv_self.v[:,n_past:n_past+N,il,0]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_1d(ctx0, kc, ggml_reshape_1d(ctx0, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                vc = ggml_set_2d(ctx0, vc, Vcur, (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));
            }

            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Q shape    [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
            // K shape [n_embd/n_head, n_past + N, n_head, 1]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kc, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kc)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // KQ shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, 1.0f/sqrtf(float(n_embd)/n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // split cached V into n_head heads
            //// V shape [n_past + N, n_embd/n_head, n_head, 1]
            // V shape [n_past + N, n_embd/n_head, n_head, 1] == kv_self.v[:,:(n_past+N),il,1]
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(vc),
                        n_ctx*ggml_element_size(vc)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(vc)*n_embd);

            // KQV shape [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N,1,1]
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
            }

            // tmp shape [n_ff,N,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);

            // SILU activation
            // cur shape [n_ff,N,1,1]
            cur = ggml_silu(ctx0, cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul(ctx0, cur, tmp);

            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
        }

        // cur shape [n_embd,N,1,1]
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        // inpL shape [n_embd,N,1,1]
        inpL = cur;
    }

    // norm
    {

        // inpL shape [n_embd,N,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

        // inpL = norm*inpL
        // inpL shape [n_embd,N,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

static struct ggml_tensor * forward_batch(
    struct llama_model    * model,
    struct llama_kv_cache * cache,
    struct ggml_context   * ctx0,
    struct ggml_cgraph    * gf,
    struct ggml_tensor    * tokens_input,
    const  int              n_tokens,
    const  int              n_past,
    const  int              n_batch
) {
    const int N = n_tokens;

    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch);
    memcpy(tokens->data, tokens_input->data, ggml_element_size(tokens)*N*n_batch);

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < N; ++i) {
            data[i] = n_past + i;
        }
    }

    // inpL shape [n_embd,N*n_batch,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    assert_shape_2d(inpL, n_embd, N*n_batch);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Kcur shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * Qcur = ggml_rope(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), KQ_pos, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), KQ_pos, n_rot, 0);
            assert_shape_4d(Qcur, n_embd/n_head, n_head, N, n_batch);
            assert_shape_4d(Kcur, n_embd/n_head, n_head, N, n_batch);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [N, n_embd, n_batch, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0,
                    ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_mul_mat(ctx0,
                                model->layers[il].wv,
                                cur),
                        n_embd, N, n_batch),
                        1, 0, 2, 3));

                assert_shape_3d(Vcur, N, n_embd, n_batch);

                // kv_self.k shape [n_embd * n_ctx * n_batch * n_layer]
                // kv_self.v shape [n_ctx * n_embd * n_batch * n_layer]
                // k         shape [n_embd * N, n_batch]   == kv_self.k[:,n_past:n_past+N,:,il]
                // v         shape [N, n_embd, n_batch, 1] == kv_self.v[:,n_past:n_past+N,:,il]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_2d(ctx0, kc,
                        ggml_reshape_2d(ctx0, Kcur, n_embd*N, n_batch),
                        ggml_element_size(kc)*n_embd*n_ctx,
                        (ggml_element_size(kc)*n_embd)*(il*n_batch*n_ctx + n_past));
                vc = ggml_set_2d(ctx0, vc,
                        ggml_reshape_2d(ctx0, Vcur, N*n_embd, n_batch),
                        ggml_element_size(vc)*n_ctx*n_embd,
                        ggml_element_size(vc)*(n_past + il*n_embd*n_batch*n_ctx));

                assert_shape_1d(kc, n_embd * n_ctx * n_batch * n_layer);
                assert_shape_1d(vc, n_embd * n_ctx * n_batch * n_layer);
            }

            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Q shape    [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch);

            // kv_self.k shape [n_embd * n_ctx * n_batch * n_layer]
            // K shape [n_embd/n_head, n_past + N, n_head, n_batch]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0,
                            ggml_view_3d(ctx0,
                                kc,
                                n_embd,
                                (n_past + N),
                                n_batch,
                                n_embd*ggml_element_size(kc),
                                n_ctx*n_embd*ggml_element_size(kc),
                                il*n_batch*n_ctx*n_embd*ggml_element_size(kc)),
                            n_embd/n_head, n_head, n_past + N, n_batch),
                        0, 2, 1, 3);
            assert_shape_4d(K, n_embd/n_head, n_past + N, n_head, n_batch);

            // K * Q
            // KQ shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            assert_shape_4d(KQ, n_past + N, N, n_head, n_batch);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, 1.0f/sqrtf(float(n_embd)/n_head));
            assert_shape_4d(KQ_scaled, n_past + N, N, n_head, n_batch);

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
            assert_shape_4d(KQ_masked, n_past + N, N, n_head, n_batch);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            assert_shape_4d(KQ_soft_max, n_past + N, N, n_head, n_batch);

            // split cached V into n_head heads
            // kv_self.v shape [n_ctx * n_embd * n_batch * n_layer]
            // V shape [n_past + N, n_embd/n_head, n_head, n_batch] == kv_self.v[:(n_past+N),:,:,il]
            struct ggml_tensor * V =
                ggml_view_4d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head, n_batch,
                        ggml_element_size(vc)*n_ctx,
                        ggml_element_size(vc)*n_ctx*n_embd/n_head,
                        ggml_element_size(vc)*n_ctx*n_embd,
                        il*n_batch*n_ctx*n_embd*ggml_element_size(vc));
            assert_shape_4d(V, n_past + N, n_embd/n_head, n_head, n_batch);

            // KQV shape [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N*n_batch);
            assert_shape_2d(cur, n_embd, N*n_batch);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N*n_batch,1,1]
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
                assert_shape_2d(cur, n_embd, N*n_batch);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
                assert_shape_2d(cur, n_embd, N*n_batch);
            }

            // tmp shape [n_ff,N*n_batch,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);
            assert_shape_2d(tmp, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // SILU activation
            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_silu(ctx0, cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul(ctx0, cur, tmp);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // cur shape [n_embd,N*n_batch,1,1]
        cur = ggml_add(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // input for next layer
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // norm
    {

        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
        assert_shape_2d(inpL, n_embd, N*n_batch);

        // inpL = norm*inpL
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        assert_shape_2d(inpL, n_embd, N*n_batch);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N*n_batch,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);
    assert_shape_2d(inpL, n_vocab, N*n_batch);

    {
        // inpL shape [n_vocab,N,n_batch,1]
        inpL = ggml_reshape_3d(ctx0,
                        inpL,
                        n_vocab, N, n_batch);
        assert_shape_3d(inpL, n_vocab, N, n_batch);
    }

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

static struct ggml_tensor * forward_lora(
    struct llama_model_lora * model,
    struct llama_kv_cache   * cache,
    struct ggml_context     * ctx0,
    struct ggml_cgraph      * gf,
    struct ggml_tensor      * tokens_input,
    const  int                n_tokens,
    const  int                n_past
) {
    const int N = n_tokens;

    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;

    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < N; ++i) {
            data[i] = n_past + i;
        }
    }

    // inpL shape [n_embd,N,1,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            // cur shape [n_embd,N,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Kcur shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * Qcur = ggml_rope(ctx0,
                                            ggml_reshape_3d(ctx0,
                                                ggml_mul_mat(ctx0,
                                                    model->layers[il].wqa,
                                                    ggml_mul_mat(ctx0,
                                                        model->layers[il].wqb,
                                                        cur)),
                                                n_embd/n_head, n_head, N),
                                            KQ_pos, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx0,
                                            ggml_reshape_3d(ctx0,
                                                ggml_mul_mat(ctx0,
                                                    model->layers[il].wka,
                                                    ggml_mul_mat(ctx0,
                                                        model->layers[il].wkb,
                                                        cur)),
                                                n_embd/n_head, n_head, N),
                                            KQ_pos, n_rot, 0);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [n_embd, N, 1, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0,
                                                ggml_transpose(ctx0,
                                                    ggml_reshape_2d(ctx0,
                                                        ggml_mul_mat(ctx0,
                                                            model->layers[il].wva,
                                                            ggml_mul_mat(ctx0,
                                                                model->layers[il].wvb,
                                                                cur)),
                                                        n_embd, N)));

                // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
                // kv_self.v shape [n_embd * n_ctx * n_layer, 1]
                // k         shape [n_embd * N, 1]   == kv_self.k[:,n_past:n_past+N,il,0]
                // v         shape [N, n_embd, 1, 1] == kv_self.v[:,n_past:n_past+N,il,0]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_1d(ctx0, kc, ggml_reshape_1d(ctx0, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                vc = ggml_set_2d(ctx0, vc, Vcur, (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));
            }

            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Q shape    [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
            // K shape [n_embd/n_head, n_past + N, n_head, 1]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kc, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kc)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // KQ shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, 1.0f/sqrtf(float(n_embd)/n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // split cached V into n_head heads
            //// V shape [n_past + N, n_embd/n_head, n_head, 1]
            // V shape [n_past + N, n_embd/n_head, n_head, 1] == kv_self.v[:,:(n_past+N),il,1]
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(vc),
                        n_ctx*ggml_element_size(vc)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(vc)*n_embd);

            // KQV shape [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].woa,
                    ggml_mul_mat(ctx0,
                        model->layers[il].wob,
                        cur));
        }

        // inpFF shape [n_embd,N,1,1]
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
            }

            // tmp shape [n_ff,N,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);

            // SILU activation
            // cur shape [n_ff,N,1,1]
            cur = ggml_silu(ctx0, cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul(ctx0, cur, tmp);

            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
        }

        // cur shape [n_embd,N,1,1]
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        // inpL shape [n_embd,N,1,1]
        inpL = cur;
    }

    // norm
    {

        // inpL shape [n_embd,N,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

        // inpL = norm*inpL
        // inpL shape [n_embd,N,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        //embeddings = inpL;
    }


    // lm_head
    // inpL shape [n_vocab,N,1,1]
    inpL = ggml_mul_mat(ctx0,
                model->outputa,
                    ggml_mul_mat(ctx0,
                        model->outputb,
                        inpL));

    // ggml_set_scratch(ctx0, { 0, 0, nullptr, });
    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

static void sample_softmax(struct ggml_tensor * logits, struct ggml_tensor * probs, struct ggml_tensor * best_samples) {
    assert(ggml_is_matrix(logits));
    assert(ggml_is_matrix(probs));
    assert(ggml_is_vector(best_samples));
    assert(logits->ne[1] == best_samples->ne[0]);
    assert(logits->ne[0] == probs->ne[0]);
    assert(logits->ne[1] == probs->ne[1]);
    for (int i = 0; i < logits->ne[1]; ++i) {
        float max_logit = ggml_get_f32_1d(logits, i * logits->ne[0]);
        ggml_set_i32_1d(best_samples, i, 0);
        for (int k = 0; k < logits->ne[0]; ++k) {
            float logit = ggml_get_f32_1d(logits, i * logits->ne[0] + k);
            if (logit > max_logit) {
                max_logit = logit;
                ggml_set_i32_1d(best_samples, i, k);
            }
        }
        float psum = 0;
        for (int k = 0; k < logits->ne[0]; ++k) {
            float logit = ggml_get_f32_1d(logits, i * logits->ne[0] + k);
            float p = (logit == -INFINITY) ? 0 : expf(logit - max_logit);
            psum += p;
            ggml_set_f32_1d(probs, i * probs->ne[0] + k, p);
        }
        for (int k = 0; k < logits->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            ggml_set_f32_1d(probs, i * probs->ne[0] + k, p / psum);
        }
    }
}

static void sample_softmax_batch(
    struct ggml_context * ctx, struct ggml_tensor * logits, struct ggml_tensor * probs,
    struct ggml_tensor * best_samples
) {
    GGML_ASSERT(ggml_is_matrix(best_samples));
    GGML_ASSERT(ggml_is_3d(logits));
    GGML_ASSERT(ggml_is_3d(probs));
    int n_tokens = best_samples->ne[0];
    int n_batch  = best_samples->ne[1];
    int n_vocab  = logits->ne[0];
    GGML_ASSERT(n_tokens == logits->ne[1]);
    GGML_ASSERT(n_batch  == logits->ne[2]);
    GGML_ASSERT(n_vocab  == probs->ne[0]);
    GGML_ASSERT(n_tokens == probs->ne[1]);
    GGML_ASSERT(n_batch  == probs->ne[2]);

    for (int k = 0; k < n_batch; ++k) {
        struct ggml_tensor * best_samples_k = ggml_view_1d(ctx,
                                                best_samples,
                                                best_samples->ne[0],
                                                k*best_samples->nb[1]);
        struct ggml_tensor * logits_k       = ggml_view_2d(ctx,
                                                logits,
                                                logits->ne[0],
                                                logits->ne[1],
                                                logits->nb[1],
                                                k*logits->nb[2]);
        struct ggml_tensor * probs_k        = ggml_view_2d(ctx,
                                                probs,
                                                probs->ne[0],
                                                probs->ne[1],
                                                probs->nb[1],
                                                k*probs->nb[2]);
        sample_softmax(logits_k, probs_k, best_samples_k);
    }
}

static void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
        printf(" %.2f", p);
    }
    printf("\n");
}

static void print_matrix(struct ggml_tensor * probs) {
    assert(ggml_is_matrix(probs));
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}

static void print_token(int token, int n_vocab) {
    for (int k = 0; k < token; ++k) {
        printf(" ");
    }
    printf("X");
    for (int k = token+1; k < n_vocab; ++k) {
        printf(" ");
    }
    printf("\n");
}

static void print_tokens(struct ggml_tensor * tokens, int n_vocab) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(token, n_vocab);
    }
}

static void get_example_targets(int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    float randomness = 0.0f;
    // ggml_set_zero(targets);
    ggml_set_f32(targets, -1.0f);
    ggml_set_i32_1d(tokens_input, 0, 0);
    for (int i=1; i<n_tokens+1; ++i) {
        float x = example_id + i * 3.14159f * 2.0f * 1.0f * 0.5f / n_tokens;
        float y = sinf(x);//*cosf(x*1.1f+1.0f);
        float z = (y+1.0f)*0.5f; // scale to [0..1]
        z += (frand()-0.5f)*(randomness/n_vocab);
        z = (z < 0.0f) ? 0.0f : (z > 1.0f) ? 1.0f : z; // clamp to [0..1]
        int token = std::max(1,std::min(1+(int)(z*(float)(n_vocab-1)), n_vocab-1));
        ggml_set_f32_1d(targets, (i-1)*n_vocab + token, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}

static void get_example_targets_batch(
    struct ggml_context * ctx, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets
) {
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(targets));
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_tokens == targets->ne[1]);
    GGML_ASSERT(n_batch  == targets->ne[2]);

    for (int k=0; k<n_batch; ++k) {
        struct ggml_tensor * tokens_input_k = ggml_view_1d(ctx,
                                                tokens_input,
                                                tokens_input->ne[0],
                                                k*tokens_input->nb[1]);
        struct ggml_tensor * targets_k    = ggml_view_2d(ctx,
                                                targets,
                                                targets->ne[0],
                                                targets->ne[1],
                                                targets->nb[1],
                                                k*targets->nb[2]);
        get_example_targets(example_id*n_batch + k, tokens_input_k, targets_k);
    }
}

static void lshift_examples(struct ggml_tensor * tokens_input, struct ggml_tensor * targets, int n_shift) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    for (int i=0; i<n_tokens-n_shift; ++i) {
        ggml_set_i32_1d(tokens_input, i, ggml_get_i32_1d(tokens_input, i + n_shift));
        for (int k=0; k<n_vocab; ++k) {
            ggml_set_f32_1d(targets, i*n_vocab + k, ggml_get_f32_1d(targets, (i + n_shift)*n_vocab + k));
        }
    }
}

static struct ggml_tensor * square_error_loss(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b
) {
    // todo: instead of a-b: a[1:]-b[:-1]
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, a, b)));
}

static struct ggml_tensor * cross_entropy_loss(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b
) {
    const float eps = 1e-3f;
    return
        ggml_sum(ctx,
            ggml_neg(ctx,
                ggml_sum_rows(ctx,
                    ggml_mul(ctx,
                        ggml_soft_max(ctx, a),
                        ggml_log(ctx,
                            ggml_add1(ctx,
                                ggml_soft_max(ctx, b),
                                ggml_new_f32(ctx, eps)))))));
}

int main(int argc, char ** argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: %s\n", argv[0]);

        return 1;
    }

    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll;
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    struct llama_model model;
    model.hparams.n_vocab = 8;
    model.hparams.n_ctx   = 8;
    model.hparams.n_embd  = 32;
    model.hparams.n_mult  = 2;
    model.hparams.n_head  = 8;
    model.hparams.n_layer = 1;
    model.hparams.n_rot   = std::min(16u, model.hparams.n_embd / model.hparams.n_head);

    // model.hparams.n_embd  = 32;
    // model.hparams.n_mult  = 2;
    // model.hparams.n_head  = 4;
    // model.hparams.n_layer = 8;
    // model.hparams.n_rot   = 8;

    model.ctx = ggml_init(lcparams);
    printf("init model\n");
    init_model(&model);
    set_param_model(&model);

    randomize_model(&model, 1337, 0.0f, 1.0f, -1.0f, +1.0f);

/*
    struct llama_model_lora model_lora;
    // model.hparams.n_vocab = 6;
    // model.hparams.n_ctx   = 64;
    // model.hparams.n_embd  = 128;
    // model.hparams.n_mult  = 2;
    // model.hparams.n_head  = 8;
    // model.hparams.n_layer = 6;
    // model.hparams.n_rot   = model.hparams.n_embd / model.hparams.n_head;

    model_lora.hparams.n_vocab = 16;
    model_lora.hparams.n_ctx   = 32;
    model_lora.hparams.n_embd  = 256;
    model_lora.hparams.n_mult  = 2;
    model_lora.hparams.n_head  = 16;
    model_lora.hparams.n_layer = 1;
    model_lora.hparams.n_lora  = 64;
    model_lora.hparams.n_rot   = MIN(16, model_lora.hparams.n_embd / model_lora.hparams.n_head);
    // model.hparams.n_rot   = (model.hparams.n_embd / model.hparams.n_head) / 2;

    // model.hparams.n_embd  = 32;
    // model.hparams.n_mult  = 2;
    // model.hparams.n_head  = 4;
    // model.hparams.n_layer = 8;
    // model.hparams.n_rot   = 8;

    model_lora.ctx = ggml_init(lcparams);
    printf("init model_lora\n");
    init_model_lora(&model_lora);
    set_param_model_lora(&model_lora);

    randomize_model_lora(&model_lora, 1337, 0.0f, 1.0f, -1.0f, +1.0f);
*/
    int n_batch = 8;
    // key + value cache for the self attention
    struct llama_kv_cache kv_self;
    printf("init_kv_cache\n");
    kv_self.ctx = model.ctx;
    init_kv_cache(&kv_self, &model, n_batch);
    //init_kv_cache_lora(&kv_self, &model_lora);

    size_t    compute_size = 1024ll*1024ll*1024ll;
    uint8_t * compute_addr = new uint8_t[compute_size];

    int n_examples = 256;
    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;

    std::vector<uint8_t> work_buffer;

    for (int ex=0; ex<n_examples; ++ex) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * after_opt_best_samples  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * after_opt_probs         = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens, n_batch);
        struct ggml_tensor * tokens_input            = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * targets                 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens, n_batch);

        int n_past = 0;

        struct ggml_cgraph * gf = NULL;
        gf = ggml_new_graph_custom(ctx0, LLAMA_TRAIN_MAX_NODES, true);

        get_example_targets_batch(ctx0, 64*ex+0,  tokens_input, targets);

        struct ggml_tensor * logits = forward_batch(&model, &kv_self, ctx0, gf, tokens_input, n_tokens, n_past, n_batch);
        // struct ggml_tensor * e = cross_entropy_loss(ctx0, targets, logits);
        struct ggml_tensor * e = square_error_loss(ctx0, targets, logits);

        ggml_build_forward_expand(gf, e);
        ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);

        float error_before_opt = ggml_get_f32_1d(e, 0);

        struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_TYPE_LBFGS);
        opt_params_lbfgs.print_forward_graph = false;
        opt_params_lbfgs.print_backward_graph = false;
        opt_params_lbfgs.lbfgs.n_iter = 16;
        ggml_opt(ctx0, opt_params_lbfgs, e);
        //
        ggml_build_forward_expand(gf, e);
        ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);

        float error_after_opt = ggml_get_f32_1d(e, 0);

        if (ex % 8 == 0) {
            printf("Example %d\n", (ex+1));
            printf("error_before_opt: %.2f\n", error_before_opt);
            printf("error_after_opt:  %.2f\n", error_after_opt);
        }

        if (ex % 64 == 0) {
            sample_softmax_batch(ctx0, logits, after_opt_probs, after_opt_best_samples);
            // printf("probabilities after optimization:\n");
            // print_matrix(after_opt_probs);
            printf("best samples after optimization:\n");
            print_tokens(after_opt_best_samples, n_vocab);
        }

        ggml_free(ctx0);
    }

    {
        int n_gen = 128;
        int sample_ctx = n_tokens-n_tokens/8;

        printf("Generating %d tokens.\n", n_gen);

        struct ggml_tensor * tokens_input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * targets      = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab, n_tokens);

        get_example_targets(137, tokens_input, targets);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(ggml_get_i32_1d(tokens_input, i), n_vocab);
        }
        printf("---\n");
        for (int i=0; i<n_gen; ++i) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ compute_addr,
                /*.no_alloc   =*/ false,
            };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_cgraph * gf = NULL;
            gf = ggml_new_graph_custom(ctx0, LLAMA_TRAIN_MAX_NODES, true);

            int n_past = 0;
            struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, gf, tokens_input, sample_ctx, n_past);

            ggml_build_forward_expand(gf, logits);
            ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);

            struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sample_ctx);
            struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, sample_ctx);

            sample_softmax(logits, probs, best_samples);

            // int sample_at = n_tokens-1;
            int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

            // print_row(probs, sample_at);
            print_token(token, n_vocab);

            lshift_examples(tokens_input, targets, 1);
            ggml_set_i32_1d(tokens_input, 0, 0);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);

            ggml_free(ctx0);
        }
    }

    print_matrix(model.tok_embeddings);
    printf("done\n");

    // ggml_free(kv_self.ctx);
    // ggml_free(model_lora.ctx);
    ggml_free(model.ctx);

    return 0;
}
