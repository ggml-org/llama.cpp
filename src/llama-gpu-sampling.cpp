#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <chrono>
#include <random>
#include <unordered_map>
#include <vector>

static void llama_sampler_gpu_greedy_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    GGML_UNUSED(smpl);
    struct ggml_tensor * argmax_result = ggml_argmax(ctx, ggml_data->logits);
    ggml_set_name(argmax_result, "argmax_result");
    ggml_data->sampled_token = argmax_result;
}

static const char * llama_sampler_gpu_greedy_sampler_name(const struct llama_sampler *) {
    return "test-ggml";
}

static struct llama_sampler * llama_sampler_gpu_greedy_clone(const struct llama_sampler * smpl) {
    (void) smpl;
    return llama_sampler_gpu_init_greedy();
}

struct llama_sampler * llama_sampler_gpu_init_greedy() {
    static const llama_sampler_i iface = {
        /*.name                =*/ llama_sampler_gpu_greedy_sampler_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_gpu_greedy_clone,
        /*.free                =*/ nullptr,
        /*.apply_ggml          =*/ llama_sampler_gpu_greedy_apply_ggml,
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

struct llama_sampler_gpu_temp_ctx {
    float temp;
};


static void llama_sampler_gpu_temp_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_temp_ctx *) smpl->ctx;

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

static const char * llama_sampler_gpu_temp_name(const struct llama_sampler *) {
    return "gpu-temp";
}

static void llama_sampler_gpu_temp_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_temp_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_temp_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_gpu_temp_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_temp(ctx->temp);
}

struct llama_sampler * llama_sampler_gpu_init_temp(float temp) {
    static const llama_sampler_i iface = {
        /*.name                = */ llama_sampler_gpu_temp_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_gpu_temp_clone,
        /*.free                =*/ llama_sampler_gpu_temp_free,
        /*.apply_ggml          =*/ llama_sampler_gpu_temp_apply_ggml,
        /*.accept_ggml         =*/ nullptr,
        /*.set_input_ggml      =*/ nullptr,
        /*.set_backend_context =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_gpu_temp_ctx {
        /*.temp    =*/ temp,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}


struct llama_sampler_gpu_top_k_ctx {
    int32_t k;

    // Only required for checking operation support and can be removed later.
    ggml_backend_dev_t device;
};

static void llama_sampler_gpu_top_k_init_ggml(
        struct llama_sampler           * smpl,
        ggml_backend_buffer_type_t       buft) {
    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    ctx_data->device = ggml_backend_buft_get_device(buft);
}

static void llama_sampler_gpu_top_k_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;

    struct ggml_tensor * top_k = ggml_top_k(ctx, ggml_data->logits, ctx_data->k);
    ggml_set_name(top_k, "top_k");

    // top_k is a view of argsort - check if backend supports the underlying argsort operation
    // by checking the source tensor (which is the argsort result)
    if (ctx_data->device && top_k->src[0] && !ggml_backend_dev_supports_op(ctx_data->device, top_k->src[0])) {
        fprintf(stderr, "Warning: backend does not support argsort operation required for top-k sampling\n");
        fprintf(stderr, "CPU backend will be used instead which defeats the purpose of having GPU samplers\n");
    }

    ggml_data->filtered_ids = top_k;

    struct ggml_tensor * logits_rows = ggml_reshape_2d(ctx, ggml_data->logits, 1, ggml_data->logits->ne[0]);
    struct ggml_tensor * top_k_rows = ggml_get_rows(ctx, logits_rows, top_k);
    ggml_set_name(top_k_rows, "top_k_rows");

    ggml_data->logits = ggml_reshape_1d(ctx, top_k_rows, ctx_data->k);
    ggml_build_forward_expand(gf, ggml_data->logits);
}

static const char * llama_sampler_gpu_top_k_name(const struct llama_sampler *) {
    return "gpu-top-k";
}

static void llama_sampler_gpu_top_k_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_top_k_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_top_k(ctx->k);
}

struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k) {
    static const llama_sampler_i iface = {
        /*.name                =*/ llama_sampler_gpu_top_k_name,
        /*.accept              =*/ nullptr,
        /*.apply               =*/ nullptr,
        /*.reset               =*/ nullptr,
        /*.clone               =*/ llama_sampler_gpu_top_k_clone,
        /*.free                =*/ llama_sampler_gpu_top_k_free,
        /*.apply_ggml          =*/ llama_sampler_gpu_top_k_apply_ggml,
        /*.accept_ggml         =*/ nullptr,
        /*.set_input_ggml      =*/ nullptr,
        /*.init_ggml           =*/ llama_sampler_gpu_top_k_init_ggml,
    };

    auto * ctx_data = new llama_sampler_gpu_top_k_ctx {
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

struct llama_sampler_gpu_dist_ctx {
    const uint32_t seed;
          uint32_t seed_cur;
    std::mt19937   rng;

    struct ggml_tensor   * uniform;
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buffer;

    // Only required for checking operation support and can be removed later.
    ggml_backend_dev_t device;
};

static void llama_sampler_gpu_dist_init_ggml(
        struct llama_sampler      * smpl,
        ggml_backend_buffer_type_t  buft) {

    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    sctx->device = ggml_backend_buft_get_device(buft);
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sctx->ctx = ggml_init(params);

    // Create the uniform random scalar input tensor. This will be set by
    // llama_sampler_gpu_dist_set_input_ggml after this graph is built.
    sctx->uniform = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_F32, 1);
    ggml_set_name(sctx->uniform, "uniform");
    ggml_set_input(sctx->uniform);
    ggml_set_output(sctx->uniform);

    // Allocate all tensors from our context to the backend
    sctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(sctx->ctx, buft);
}

static void llama_sampler_gpu_dist_set_input_ggml(struct llama_sampler * smpl,
        llama_context * ctx) {
    GGML_UNUSED(ctx);
    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    GGML_ASSERT(sctx->uniform != nullptr);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float rnd = dist(sctx->rng);
    ggml_backend_tensor_set(sctx->uniform, &rnd, 0, sizeof(float));
}

static void llama_sampler_gpu_dist_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;

    struct ggml_tensor * probs = ggml_soft_max(ctx, ggml_data->logits);
    ggml_set_name(probs, "dist_probs");

    struct ggml_tensor * cumsum = ggml_cumsum(ctx, probs);
    if (sctx->device && !ggml_backend_dev_supports_op(sctx->device, cumsum)) {
        fprintf(stderr, "Warning: backend does not support cumsum operation required for dist sampling\n");
        fprintf(stderr, "CPU backend will be used instead which defeats the purpose of having GPU samplers\n");
    }
    ggml_set_name(cumsum, "cumsum");

    // Broadcast the random uniform value to match cumsums’s shape
    struct ggml_tensor * rnd_rep = ggml_repeat(ctx, sctx->uniform, cumsum);
    ggml_set_name(rnd_rep, "dist_rand_rep");

    // Each entry in rnd_rep has the random value in it so we subtract this
    // tensor with the cumsum tensor. Recall that each entry in cumsum is the
    // cumulative probability up to that index. While the value is smaller than
    // the random value the difference is positive, but once we exceed the random
    // value the difference becomes zero or negative.
    struct ggml_tensor * diff = ggml_sub(ctx, rnd_rep, cumsum);
    ggml_set_name(diff, "dist_rnd_minus_cumsum");

    // The ggml_step function produces a tensor where entries are 1 if the
    // corresponding entry in diff is > 0, and 0 otherwise. So all values up to
    // the index where the cumulative probability exceeds the random value are 1,
    // and all entries after that are 0.
    struct ggml_tensor * mask = ggml_step(ctx, diff);
    ggml_set_name(mask, "dist_mask");

    // Taking the sum of the mask gives us the index entry where the cumulative
    // threshold is first exceeded and this is our sampled token index as a float.
    struct ggml_tensor * idxf = ggml_sum(ctx, mask);
    ggml_set_name(idxf, "dist_index_f32");

    // Cast the float index to integer so we can used it with ggml_get_rows later.
    struct ggml_tensor * idx = ggml_cast(ctx, idxf, GGML_TYPE_I32);
    ggml_set_name(idx, "dist_index_i32");

    // Map back to original vocab ids if a filtered id tensor is available.
    struct ggml_tensor * sampled_token = idx;
    if (ggml_data->filtered_ids != nullptr) {
        struct ggml_tensor * filtered_ids = ggml_cont(ctx, ggml_data->filtered_ids);
        ggml_set_name(filtered_ids, "dist_filtered_ids");

        struct ggml_tensor * filtered_ids_reshaped = ggml_reshape_2d(ctx, filtered_ids, 1, ggml_nelements(filtered_ids));

        struct ggml_tensor * gathered = ggml_get_rows(ctx, filtered_ids_reshaped, idx);
        ggml_set_name(gathered, "dist_sampled_token");

        sampled_token = ggml_reshape_1d(ctx, gathered, 1);
    }

    ggml_set_output(sampled_token);
    ggml_data->sampled_token = sampled_token;
}

static const char * llama_sampler_gpu_dist_name(const struct llama_sampler *) {
    return "gpu-dist";
}

static void llama_sampler_gpu_dist_free(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    ggml_backend_buffer_free(sctx->buffer);
    ggml_free(sctx->ctx);
    delete sctx;
}

static struct llama_sampler * llama_sampler_gpu_dist_clone(const struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_dist(sctx->seed);
}


struct llama_sampler * llama_sampler_gpu_init_dist(uint32_t seed) {
    static const llama_sampler_i iface = {
        /*.name           =*/ llama_sampler_gpu_dist_name,
        /*.accept         =*/ nullptr,
        /*.apply          =*/ nullptr,
        /*.reset          =*/ nullptr,
        /*.clone          =*/ llama_sampler_gpu_dist_clone,
        /*.free           =*/ llama_sampler_gpu_dist_free,
        /*.apply_ggml     =*/ llama_sampler_gpu_dist_apply_ggml,
        /*.accept_ggml    =*/ nullptr,
        /*.set_input_ggml =*/ llama_sampler_gpu_dist_set_input_ggml,
        /*.init_ggml      =*/ llama_sampler_gpu_dist_init_ggml,
    };

    auto seed_cur = get_rng_seed(seed);
    auto * ctx_data = new llama_sampler_gpu_dist_ctx {
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

struct llama_sampler_gpu_logit_bias_ctx {
    const int32_t n_vocab;

    const std::vector<llama_logit_bias> logit_bias;

    struct ggml_tensor * logit_bias_t;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static void llama_sampler_gpu_logit_bias_init_ggml(
        struct llama_sampler      * smpl,
        ggml_backend_buffer_type_t  buft) {
    auto * sctx = (llama_sampler_gpu_logit_bias_ctx *) smpl->ctx;
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

static void llama_sampler_gpu_logit_bias_set_input_ggml(struct llama_sampler * smpl,
        llama_context * ctx) {
    GGML_UNUSED(ctx);
    auto * sctx = (llama_sampler_gpu_logit_bias_ctx *) smpl->ctx;
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

static void llama_sampler_gpu_logit_bias_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ctx);

    auto * sctx = (llama_sampler_gpu_logit_bias_ctx *) smpl->ctx;
    if (sctx->logit_bias_t == nullptr) {
        return;
    }

    // Add the sparse logit logit_bias to the logits
    struct ggml_tensor * logit_biased = ggml_add_inplace(sctx->ctx, ggml_data->logits, sctx->logit_bias_t);
    ggml_build_forward_expand(gf, logit_biased);
}

static const char * llama_sampler_gpu_logit_bias_name(const struct llama_sampler *) {
    return "gpu-logit_bias";
}

static void llama_sampler_gpu_logit_bias_free(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_logit_bias_ctx *) smpl->ctx;
    ggml_backend_buffer_free(sctx->buffer);
    ggml_free(sctx->ctx);
    delete sctx;
}

static struct llama_sampler * llama_sampler_gpu_logit_bias_clone(const struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_logit_bias_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_logit_bias(sctx->n_vocab,
                                      sctx->logit_bias.size(),
                                      sctx->logit_bias.data());
}


struct llama_sampler * llama_sampler_gpu_init_logit_bias(int32_t   n_vocab,
                                                   int32_t   n_logit_bias,
                                    const llama_logit_bias * logit_bias) {
    static const llama_sampler_i iface = {
        /*.name           =*/ llama_sampler_gpu_logit_bias_name,
        /*.accept         =*/ nullptr,
        /*.apply          =*/ nullptr,
        /*.reset          =*/ nullptr,
        /*.clone          =*/ llama_sampler_gpu_logit_bias_clone,
        /*.free           =*/ llama_sampler_gpu_logit_bias_free,
        /*.apply_ggml     =*/ llama_sampler_gpu_logit_bias_apply_ggml,
        /*.accept_ggml    =*/ nullptr,
        /*.set_input_ggml =*/ llama_sampler_gpu_logit_bias_set_input_ggml,
        /*.init_ggml      =*/ llama_sampler_gpu_logit_bias_init_ggml,
    };

    auto * ctx_data = new llama_sampler_gpu_logit_bias_ctx {
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

struct llama_sampler_gpu_penalties_ctx {
    const int32_t n_vocab;
    const int32_t last_n;
    const float   repeat;
    const float   freq;
    const float   present;

    struct ggml_tensor    * history_t;  // I32 [last_n]
    struct ggml_tensor    * n_history_t;   // I32 [1]
    struct ggml_context   * ctx;
    ggml_backend_buffer_t   buffer;

    ggml_backend_dev_t device = nullptr;

    // CPU-side history tracking
    std::vector<int32_t> cpu_history;  // Circular buffer
    int32_t history_pos = 0;            // Current write position in circular buffer
    int32_t history_count = 0;          // Number of valid tokens (0 to last_n)
    int32_t batch_idx = -1;
    int32_t prev_batch_idx = -1;
};

static void llama_sampler_gpu_penalties_init_ggml(
        struct llama_sampler           * smpl,
        ggml_backend_buffer_type_t       buft) {

    auto * sctx = (llama_sampler_gpu_penalties_ctx *) smpl->ctx;

    if ((sctx->last_n == 0) ||
        (sctx->freq == 0.0f && sctx->present == 0.0f)) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead() + sctx->last_n * sizeof(int32_t) + sizeof(int32_t),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sctx->ctx = ggml_init(params);

    sctx->history_t = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_I32, sctx->last_n);
    ggml_set_name(sctx->history_t, "penalties.token_history");
    ggml_set_input(sctx->history_t);
    ggml_set_output(sctx->history_t);

    sctx->n_history_t = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_I32, 1);
    ggml_set_name(sctx->n_history_t, "penalties.history_size");
    ggml_set_input(sctx->n_history_t);
    ggml_set_output(sctx->history_t);

    sctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(sctx->ctx, buft);

    // Initialize history_size to 0
    int32_t init_size = 0;
    ggml_backend_tensor_set(sctx->n_history_t, &init_size, 0, sizeof(int32_t));

    // Initialize token_history to -1
    std::vector<int32_t> init_history(sctx->last_n, -1);
    ggml_backend_tensor_set(sctx->history_t, init_history.data(), 0, ggml_nbytes(sctx->history_t));

    // Initialize CPU-side history buffer
    sctx->cpu_history.resize(sctx->last_n, -1);
    sctx->history_pos = 0;
    sctx->history_count = 0;

    sctx->device = ggml_backend_buft_get_device(buft);
}

static void llama_sampler_gpu_penalties_set_input_ggml(struct llama_sampler * smpl,
        llama_context * ctx) {
    auto * sctx = (llama_sampler_gpu_penalties_ctx *) smpl->ctx;

    llama_token token = llama_get_sampled_token_ith(ctx, sctx->prev_batch_idx);
    sctx->prev_batch_idx = sctx->batch_idx;

    // Update CPU-side circular buffer if we have a valid token
    if (token >= 0 && token != LLAMA_TOKEN_NULL) {
        sctx->cpu_history[sctx->history_pos] = token;
        sctx->history_pos = (sctx->history_pos + 1) % sctx->last_n;

        // Increment count up to last_n
        if (sctx->history_count < sctx->last_n) {
            sctx->history_count++;
        }
    }

    // Copy CPU history to GPU tensor
    ggml_backend_tensor_set(sctx->history_t, sctx->cpu_history.data(), 0, sctx->cpu_history.size() * sizeof(int32_t));
    ggml_backend_tensor_set(sctx->n_history_t, &sctx->history_count, 0, sizeof(int32_t));
}

static void llama_sampler_gpu_penalties_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    auto * sctx = (llama_sampler_gpu_penalties_ctx *) smpl->ctx;

    if ((sctx->last_n == 0) ||
        (sctx->repeat == 1.0f && sctx->freq == 0.0f && sctx->present == 0.0f)) {
        return;
    }

    if (sctx->history_t == nullptr || sctx->n_history_t == nullptr) {
        return;
    }

    // Apply penalties using the ggml_penalties operation
    struct ggml_tensor * penalized_logits = ggml_penalties(
        ctx,
        ggml_data->logits,
        sctx->history_t,
        sctx->n_history_t,
        sctx->repeat,
        sctx->freq,
        sctx->present
    );

    ggml_set_name(penalized_logits, "penalties.penalized_logits");
    ggml_build_forward_expand(gf, penalized_logits);

    // Update the logits pointer to the penalized version
    ggml_data->logits = penalized_logits;
}

static void llama_sampler_gpu_penalties_accept_ggml(
        struct llama_sampler  * smpl,
        struct ggml_context   * ctx,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * selected_token) {
    GGML_UNUSED(smpl);
    GGML_UNUSED(ctx);
    GGML_UNUSED(gf);
    GGML_UNUSED(selected_token);

    // No-op: Token history is managed in set_input_ggml by reading
    // the sampled token from the context (already on CPU via async sync)
}

static void llama_sampler_gpu_penalties_free(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_penalties_ctx *) smpl->ctx;
    ggml_backend_buffer_free(sctx->buffer);
    ggml_free(sctx->ctx);
    delete sctx;
}

static const char * llama_sampler_gpu_penalties_name(const struct llama_sampler *) {
    return "gpu-penalties";
}

static struct llama_sampler * llama_sampler_gpu_penalties_clone(const struct llama_sampler * smpl) {
    const auto * sctx = (const llama_sampler_gpu_penalties_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_penalties(sctx->n_vocab, sctx->last_n, sctx->repeat, sctx->freq, sctx->present);
}

static struct llama_sampler_i llama_sampler_gpu_penalties_i = {
    /* .name                = */ llama_sampler_gpu_penalties_name,
    /* .accept              = */ nullptr,
    /* .apply               = */ nullptr,
    /* .reset               = */ nullptr,
    /* .clone               = */ llama_sampler_gpu_penalties_clone,
    /* .free                = */ llama_sampler_gpu_penalties_free,
    /* .apply_ggml          = */ llama_sampler_gpu_penalties_apply_ggml,
    /* .accept_ggml         = */ llama_sampler_gpu_penalties_accept_ggml,
    /* .set_input_ggml      = */ llama_sampler_gpu_penalties_set_input_ggml,
    /* .init_ggml           = */ llama_sampler_gpu_penalties_init_ggml,
};

struct llama_sampler * llama_sampler_gpu_init_penalties(
        int32_t n_vocab,
        int32_t last_n,
        float   repeat,
        float   freq,
        float   present) {
    last_n = std::max(last_n, 0);

    auto * sctx = new llama_sampler_gpu_penalties_ctx {
        /* .n_vocab               =*/ n_vocab,
        /* .last_n                =*/ last_n,
        /* .repeat                =*/ repeat,
        /* .freq                  =*/ freq,
        /* .present               =*/ present,
        /* .token_history_t       =*/ nullptr,
        /* .history_size_t        =*/ nullptr,
        /* .ctx                   =*/ nullptr,
        /* .buffer                =*/ nullptr,
        /* .device                =*/ nullptr,
        /* .cpu_history           =*/ {},
        /* .history_pos           =*/ 0,
        /* .history_count         =*/ 0,
        /* .batch_idx             =*/ -1,
        /* .prev_batch_idx        =*/ -1,
    };

    auto * sampler = new llama_sampler {
        /* .iface = */ &llama_sampler_gpu_penalties_i,
        /* .ctx   = */ sctx,
    };

    return sampler;
}
