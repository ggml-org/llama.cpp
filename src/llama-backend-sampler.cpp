#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <chrono>
#include <random>
#include <unordered_map>
#include <vector>

static void llama_sampler_backend_greedy_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    GGML_UNUSED(smpl);
    struct ggml_tensor * argmax_result = ggml_argmax(ctx, ggml_data->logits);
    ggml_set_name(argmax_result, "argmax_result");
    ggml_data->sampled = argmax_result;
}

static const char * llama_sampler_backend_greedy_sampler_name(const struct llama_sampler *) {
    return "test-ggml";
}

static struct llama_sampler * llama_sampler_backend_greedy_clone(const struct llama_sampler * smpl) {
    (void) smpl;
    return llama_sampler_backend_init_greedy();
}

struct llama_sampler * llama_sampler_backend_init_greedy() {
    static const llama_sampler_i iface = {
        /*.name                =*/ llama_sampler_backend_greedy_sampler_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_backend_greedy_clone,
        /*.free                =*/ nullptr,
        /*.apply_ggml          =*/ llama_sampler_backend_greedy_apply_ggml,
        /*.accept_ggml         =*/ nullptr,
        /*.set_input_ggml      =*/ nullptr,
        /*.init_ggml           =*/ nullptr,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ nullptr,
    };

    return sampler;
}

struct llama_sampler_backend_temp_ctx {
    float temp;
};


static void llama_sampler_backend_temp_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_backend_temp_ctx *) smpl->ctx;

    if (ctx_data->temp <= 0.0f) {
        return;
    }

    struct ggml_tensor * scaled = ggml_scale(ctx, ggml_data->logits, 1.0f / ctx_data->temp);
    ggml_set_name(scaled, "temp_scaled");

    // Make sure the scaled tensor is contiguous for subsequent operations
    ggml_data->logits = ggml_cont(ctx, scaled);
    ggml_set_name(ggml_data->logits, "temp_scaled_logits");

    ggml_build_forward_expand(gf, ggml_data->logits);
}

static const char * llama_sampler_backend_temp_name(const struct llama_sampler *) {
    return "backend-temp";
}

static void llama_sampler_backend_temp_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_backend_temp_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_backend_temp_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_backend_temp_ctx *) smpl->ctx;
    return llama_sampler_backend_init_temp(ctx->temp);
}

struct llama_sampler * llama_sampler_backend_init_temp(float temp) {
    static const llama_sampler_i iface = {
        /*.name                =*/ llama_sampler_backend_temp_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_backend_temp_clone,
        /*.free                =*/ llama_sampler_backend_temp_free,
        /*.apply_ggml          =*/ llama_sampler_backend_temp_apply_ggml,
        /*.accept_ggml         =*/ nullptr,
        /*.set_input_ggml      =*/ nullptr,
        /*.set_backend_context =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_backend_temp_ctx {
        /*.temp    =*/ temp,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}


struct llama_sampler_backend_top_k_ctx {
    int32_t k;

    // Only required for checking operation support and can be removed later.
    ggml_backend_dev_t device;
};

static void llama_sampler_backend_top_k_init_ggml(
        struct llama_sampler           * smpl,
        ggml_backend_buffer_type_t       buft) {
    auto * ctx_data = (llama_sampler_backend_top_k_ctx *) smpl->ctx;
    ctx_data->device = ggml_backend_buft_get_device(buft);
}

static void llama_sampler_backend_top_k_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_backend_top_k_ctx *) smpl->ctx;

    struct ggml_tensor * top_k = ggml_top_k(ctx, ggml_data->logits, ctx_data->k);
    ggml_set_name(top_k, "top_k");

    // top_k is a view of argsort - check if backend supports the underlying argsort operation
    // by checking the source tensor (which is the argsort result)
    if (ctx_data->device && top_k->src[0] && !ggml_backend_dev_supports_op(ctx_data->device, top_k->src[0])) {
        fprintf(stderr, "Warning: backend does not support argsort operation required for top-k sampling\n");
        fprintf(stderr, "CPU backend will be used instead which defeats the purpose of having backend samplers\n");
    }

    // TODO: temporary cont until https://github.com/ggml-org/llama.cpp/pull/17365 is merged
    ggml_data->candidates = ggml_cont(ctx, top_k);

    struct ggml_tensor * logits_rows = ggml_reshape_2d(ctx, ggml_data->logits, 1, ggml_data->logits->ne[0]);
    struct ggml_tensor * top_k_rows = ggml_get_rows(ctx, logits_rows, top_k);
    ggml_set_name(top_k_rows, "top_k_rows");

    ggml_data->logits = ggml_reshape_1d(ctx, top_k_rows, ctx_data->k);
    ggml_build_forward_expand(gf, ggml_data->logits);
}

static const char * llama_sampler_backend_top_k_name(const struct llama_sampler *) {
    return "backend-top-k";
}

static void llama_sampler_backend_top_k_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_backend_top_k_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_backend_top_k_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_backend_top_k_ctx *) smpl->ctx;
    return llama_sampler_backend_init_top_k(ctx->k);
}

struct llama_sampler * llama_sampler_backend_init_top_k(int32_t k) {
    static const llama_sampler_i iface = {
        /*.name                =*/ llama_sampler_backend_top_k_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_backend_top_k_clone,
        /*.free                =*/ llama_sampler_backend_top_k_free,
        /*.apply_ggml          =*/ llama_sampler_backend_top_k_apply_ggml,
        /*.accept_ggml         =*/ nullptr,
        /*.set_input_ggml      =*/ nullptr,
        /*.init_ggml           =*/ llama_sampler_backend_top_k_init_ggml,
    };

    auto * ctx_data = new llama_sampler_backend_top_k_ctx {
        /*.k       =*/ k,
        /*.device  =*/ nullptr,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}


static uint32_t get_rng_seed(uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

struct llama_sampler_backend_dist_ctx {
    const uint32_t seed;
          uint32_t seed_cur;
    std::mt19937   rng;

    struct ggml_tensor   * uniform;
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buffer;

    // Only required for checking operation support and can be removed later.
    ggml_backend_dev_t device;
};

static void llama_sampler_backend_dist_init_ggml(
        struct llama_sampler      * smpl,
        ggml_backend_buffer_type_t  buft) {

    auto * sctx = (llama_sampler_backend_dist_ctx *) smpl->ctx;
    sctx->device = ggml_backend_buft_get_device(buft);
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sctx->ctx = ggml_init(params);

    // Create the uniform random scalar input tensor. This will be set by
    // llama_sampler_backend_dist_set_input_ggml after this graph is built.
    sctx->uniform = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_F32, 1);
    ggml_set_name(sctx->uniform, "uniform");
    ggml_set_input(sctx->uniform);
    ggml_set_output(sctx->uniform);

    // Allocate all tensors from our context to the backend
    sctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(sctx->ctx, buft);
}

static void llama_sampler_backend_dist_set_input_ggml(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_dist_ctx *) smpl->ctx;
    GGML_ASSERT(sctx->uniform != nullptr);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float rnd = dist(sctx->rng);
    ggml_backend_tensor_set(sctx->uniform, &rnd, 0, sizeof(float));
}

static void llama_sampler_backend_dist_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    auto * sctx = (llama_sampler_backend_dist_ctx *) smpl->ctx;

    struct ggml_tensor * probs = ggml_soft_max(ctx, ggml_data->logits);
    ggml_set_name(probs, "dist_probs");

    struct ggml_tensor * cumsum = ggml_cumsum(ctx, probs);
    if (sctx->device && !ggml_backend_dev_supports_op(sctx->device, cumsum)) {
        fprintf(stderr, "Warning: backend does not support cumsum operation required for dist sampling\n");
        fprintf(stderr, "CPU backend will be used instead which defeats the purpose of having backend samplers\n");
    }
    ggml_set_name(cumsum, "cumsum");

    // The uniform tensor has a random value and we subtract this tensor with
    // the cumsum tensor (the uniform tensor will be broadcasted by ggml_sub).
    // Recall that each entry in cumsum is the cumulative probability up to that
    // index so values stay negative while the cumulative total is below the
    // random value, and become zero/positive once the threshold is crossed.
    struct ggml_tensor * diff = ggml_sub(ctx, cumsum, sctx->uniform);
    ggml_set_name(diff, "dist_cumsum");

    // The ggml_step function produces a tensor where entries are 1 if the
    // corresponding entry in diff is > 0, and 0 otherwise. So all values up to
    // the index where the cumulative probability exceeds the random value are 0,
    // and all entries after that are 1.
    struct ggml_tensor * mask = ggml_step(ctx, diff);
    ggml_set_name(mask, "dist_mask");

    // Taking the sum of the mask gives us the sum of elements after the threshold
    // we are interested in.
    struct ggml_tensor * idxf = ggml_sum(ctx, mask);
    ggml_set_name(idxf, "dist_index_f32");

    // Use ggml_scale_bias to scale the index value by -1 and then add the size
    // of the mask to that value so we get the correct index ((-1 * idxf) + n).
    struct ggml_tensor * idx = ggml_cast(ctx, ggml_scale_bias(ctx, idxf, -1.0f, mask->ne[0]), GGML_TYPE_I32);
    ggml_set_name(idx, "dist_index_i32");

    // Map back to original vocab ids if a candidates tensor is available.
    struct ggml_tensor * sampled_token = idx;
    if (ggml_data->candidates != nullptr) {
        struct ggml_tensor * candidates = ggml_data->candidates;
        struct ggml_tensor * candidates_reshaped = ggml_view_2d(ctx, candidates, 1, ggml_nelements(candidates),
                ggml_type_size(candidates->type), 0);

        sampled_token = ggml_get_rows(ctx, candidates_reshaped, idx);
        ggml_set_name(sampled_token, "dist_sampled_token");
    }

    ggml_set_output(sampled_token);
    ggml_data->sampled = sampled_token;
}

static const char * llama_sampler_backend_dist_name(const struct llama_sampler *) {
    return "backend-dist";
}

static void llama_sampler_backend_dist_free(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_dist_ctx *) smpl->ctx;
    ggml_backend_buffer_free(sctx->buffer);
    ggml_free(sctx->ctx);
    delete sctx;
}

static struct llama_sampler * llama_sampler_backend_dist_clone(const struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_dist_ctx *) smpl->ctx;
    return llama_sampler_backend_init_dist(sctx->seed);
}


struct llama_sampler * llama_sampler_backend_init_dist(uint32_t seed) {
    static const llama_sampler_i iface = {
        /*.name           =*/ llama_sampler_backend_dist_name,
        /*.accept         =*/ nullptr,
        /*.apply          =*/ nullptr,
        /*.reset          =*/ nullptr,
        /*.clone          =*/ llama_sampler_backend_dist_clone,
        /*.free           =*/ llama_sampler_backend_dist_free,
        /*.apply_ggml     =*/ llama_sampler_backend_dist_apply_ggml,
        /*.accept_ggml    =*/ nullptr,
        /*.set_input_ggml =*/ llama_sampler_backend_dist_set_input_ggml,
        /*.init_ggml      =*/ llama_sampler_backend_dist_init_ggml,
    };

    auto seed_cur = get_rng_seed(seed);
    auto * ctx_data = new llama_sampler_backend_dist_ctx {
        /*.seed     =*/ seed,
        /*.seed_cur =*/ seed_cur,
        /*.rng      =*/ std::mt19937(seed_cur),
        /*.uniform  =*/ nullptr,
        /*.ctx      =*/ nullptr,
        /*.buffer   =*/ nullptr,
        /*.device   =*/ nullptr,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}

struct llama_sampler_backend_logit_bias_ctx {
    const int32_t n_vocab;

    const std::vector<llama_logit_bias> logit_bias;

    struct ggml_tensor * logit_bias_t;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static void llama_sampler_backend_logit_bias_init_ggml(
        struct llama_sampler      * smpl,
        ggml_backend_buffer_type_t  buft) {
    auto * sctx = (llama_sampler_backend_logit_bias_ctx *) smpl->ctx;
    if (sctx->logit_bias.empty()) {
        return;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * sctx->n_vocab * sizeof(float),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sctx->ctx = ggml_init(params);

    struct ggml_tensor * logit_bias = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_F32, sctx->n_vocab);
    sctx->logit_bias_t = logit_bias;
    ggml_set_name(sctx->logit_bias_t, "logit_bias");
    ggml_set_input(sctx->logit_bias_t);
    ggml_set_output(sctx->logit_bias_t);

    // Allocate all tensors from our context to the backend
    sctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(sctx->ctx, buft);
}

static void llama_sampler_backend_logit_bias_set_input_ggml(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_logit_bias_ctx *) smpl->ctx;
    if (sctx->logit_bias.empty()) {
        return;
    }
    GGML_ASSERT(sctx->logit_bias_t != nullptr);

    // Create a sparse logit_bias vector from the logit_bias entries.
    std::vector<float> logit_bias_sparse(sctx->n_vocab, 0.0f);
    for (const auto & lb : sctx->logit_bias) {
        GGML_ASSERT(lb.token >= 0 && lb.token < (int32_t) sctx->n_vocab);
        logit_bias_sparse[lb.token] = lb.bias;
    }

    ggml_backend_tensor_set(sctx->logit_bias_t, logit_bias_sparse.data(), 0, ggml_nbytes(sctx->logit_bias_t));
}

static void llama_sampler_backend_logit_bias_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ctx);

    auto * sctx = (llama_sampler_backend_logit_bias_ctx *) smpl->ctx;
    if (sctx->logit_bias_t == nullptr) {
        return;
    }

    // Add the sparse logit logit_bias to the logits
    struct ggml_tensor * logit_biased = ggml_add_inplace(sctx->ctx, ggml_data->logits, sctx->logit_bias_t);
    ggml_build_forward_expand(gf, logit_biased);
}

static const char * llama_sampler_backend_logit_bias_name(const struct llama_sampler *) {
    return "backend-logit_bias";
}

static void llama_sampler_backend_logit_bias_free(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_logit_bias_ctx *) smpl->ctx;
    ggml_backend_buffer_free(sctx->buffer);
    ggml_free(sctx->ctx);
    delete sctx;
}

static struct llama_sampler * llama_sampler_backend_logit_bias_clone(const struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_backend_logit_bias_ctx *) smpl->ctx;
    return llama_sampler_backend_init_logit_bias(sctx->n_vocab,
                                      sctx->logit_bias.size(),
                                      sctx->logit_bias.data());
}


struct llama_sampler * llama_sampler_backend_init_logit_bias(int32_t   n_vocab,
                                                   int32_t   n_logit_bias,
                                    const llama_logit_bias * logit_bias) {
    static const llama_sampler_i iface = {
        /*.name           =*/ llama_sampler_backend_logit_bias_name,
        /*.accept         =*/ nullptr,
        /*.apply          =*/ nullptr,
        /*.reset          =*/ nullptr,
        /*.clone          =*/ llama_sampler_backend_logit_bias_clone,
        /*.free           =*/ llama_sampler_backend_logit_bias_free,
        /*.apply_ggml     =*/ llama_sampler_backend_logit_bias_apply_ggml,
        /*.accept_ggml    =*/ nullptr,
        /*.set_input_ggml =*/ llama_sampler_backend_logit_bias_set_input_ggml,
        /*.init_ggml      =*/ llama_sampler_backend_logit_bias_init_ggml,
    };

    auto * ctx_data = new llama_sampler_backend_logit_bias_ctx {
        /*.n_vocab      =*/ n_vocab,
        /*.logit_bias   =*/ std::vector<llama_logit_bias>(logit_bias, logit_bias + n_logit_bias),
        /*.logit_bias_t =*/ nullptr,
        /*.ctx          =*/ nullptr,
        /*.buffer       =*/ nullptr,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}
