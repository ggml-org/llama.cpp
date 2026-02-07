/* expected-attention.cpp */

/***********************************************************************************************

This program serves as a proof-of-concept for implementing _Expected Attention_ in llama.cpp.
We are just trying to see if this will be viable or not.
If we get it to work in this program, we will try to implement it into llama.cpp proper.
First we need to prove it can work.


NOTES
---

### Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution

> Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient
> large language model (LLM) inference. While attention-score-based KV cache pruning shows
> promise, it faces critical practical limitations: attention scores from future tokens are
> unavailable during compression, and modern implementations like Flash Attention do not
> materialize the full attention matrix, making past scores inaccessible. To overcome these
> challenges, we introduce *Expected Attention*, **a training-free compression method** that
> estimates KV pairs importance by predicting how future queries will attend to them. Our
> approach leverages the distributional properties of LLM activations to compute expected
> attention scores in closed form for each KV pair. These scores enable principled ranking and
> pruning of KV pairs with minimal impact on the residual stream, achieving effective
> compression without performance degradation. Importantly, our method operates seamlessly
> across both prefilling and decoding phases, consistently outperforming state-of-the-art
> baselines in both scenarios. Finally, we release KVPress, a comprehensive library to enable
> researchers to implement and benchmark KV cache compression methods, already including more
> than 20 techniques.

refs:
- [arXiv:2510.0063v1](https://arxiv.org/abs/2510.00636v1)
- https://github.com/NVIDIA/kvpress

***********************************************************************************************/

#include <cmath>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <charconv>
#include <stdint.h>
#include <stdexcept>
#include <algorithm>

#include "ggml.h"
#include "llama.h"

// path to GGUF file to load from (compile-time constant for PoC - change me!)
static constexpr const char * MODEL_PATH = "/home/dylan/gguf/Qwen3-14B-Q4_K_X.gguf";
static constexpr float SCORE_EPS = 0.02f; // TODO: added to attention scores for numerical stability

// this struct holds the statistics we accumulate during graph execution via the callback
struct expected_attn_stats {

    // callback index - track how many times the callback fires for both <ask, !ask>
    std::pair<size_t, size_t> idx = {0, 0};

    size_t n_runs   = 0; // number of distinct graph executions observed
    size_t n_tokens = 0; // number of tokens observed (incremented by n_ubatch for each run)

    std::vector<std::vector<size_t>> n_samples_per_head; // [layer][head]

    // we need to know these model hparams
    const int32_t n_embd    = 0; // native dimensionality of model
    const int32_t n_head    = 0; // number of query heads per layer
    const int32_t n_head_kv = 0; // number of KV heads per layer
    const int32_t n_layers  = 0; // number of layers in the model

    // these are computed at init based on the provided hparams
    const int32_t n_embd_head;    // dimensionality per query head
    const int32_t n_embd_head_kv; // dimensionality per KV head

    // a vector of vectors of pairs of vectors (of doubles)
    //
    // the primary vector corresponds to the observed layers of the model [n_layers].
    // the secondary vector corresponds to the number of query heads per layer [n_head].
    // for each query head, we store a pair of vectors <mean, m2> where:
    //  - the first vector `mean` is the running mean for this query head
    //  - the second vector `m2` is the running sum of squared differences from the mean for
    //    this query head
    //  - both vectors are of the same length [n_embd_head]
    std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> observed_data;

    // these vectors are reserved once and re-used for every call to expected_attn_stats.print()

    mutable std::vector<double> all_means;
    mutable std::vector<double> all_vars;

    // initialize stats
    expected_attn_stats(
        const int32_t n_embd,
        const int32_t n_head,
        const int32_t n_head_kv,
        const int32_t n_layers
    ) : n_embd(n_embd),
        n_head(n_head),
        n_head_kv(n_head_kv),
        n_layers(n_layers),
        n_embd_head(n_embd / n_head),
        n_embd_head_kv(n_embd / n_head_kv)
    {
        fprintf(stdout,
            "expected_attn_stats: init: n_embd = %d, n_head = %d, n_head_kv = %d, "
            "n_layers = %d, n_embd_head = %d, n_embd_head_kv = %d\n",
            n_embd, n_head, n_head_kv, n_layers, n_embd_head, n_embd_head_kv
        );

        // resize outer vector for layers
        observed_data.resize(n_layers);
        n_samples_per_head.resize(n_layers);

        // for each layer, resize for number of query heads
        for (int32_t il = 0; il < n_layers; ++il) {
            observed_data[il].resize(n_head);
            n_samples_per_head[il].resize(n_head, 0);

            // for each head, initialize mean and m2 vectors
            for (int32_t ih = 0; ih < n_head; ++ih) {
                observed_data[il][ih].first.resize(n_embd_head, 0.0);  // mean
                observed_data[il][ih].second.resize(n_embd_head, 0.0); // m2
            }
        }

        all_means.reserve(n_head * n_embd_head);
        all_vars.reserve(n_head * n_embd_head);
    }

    // reset stats for all query heads in all layers
    void reset() {
        idx.first  = 0;
        idx.second = 0;
        n_runs     = 0;
        n_tokens   = 0;

        for (int32_t il = 0; il < n_layers; ++il) {
            for (int32_t ih = 0; ih < n_head; ++ih) {
                auto & [mean, m2] = observed_data[il][ih];
                std::fill(mean.begin(), mean.end(), 0.0);
                std::fill(m2.begin(),   m2.end(),   0.0);
                n_samples_per_head[il][ih] = 0;
            }
        }
    }

    // compute the expected query distribution for all query heads in all layers based on the
    // currently accumulated statistics
    //
    // returns a pair of 3D vectors <mu_q, sigma_q>
    // the shape of each vector is [n_layers][n_head][n_embd_head]
    const std::pair<
        std::vector<std::vector<std::vector<double>>>, 
        std::vector<std::vector<std::vector<double>>>
    >
    compute() const {
        std::vector<std::vector<std::vector<double>>> mu_q(n_layers);
        std::vector<std::vector<std::vector<double>>> sigma_q(n_layers);

        for (int32_t il = 0; il < n_layers; ++il) {
            mu_q[il].resize(n_head);
            sigma_q[il].resize(n_head);

            for (int32_t ih = 0; ih < n_head; ++ih) {
                const auto & [mean, m2] = observed_data[il][ih];
                const size_t n = n_samples_per_head[il][ih];

                mu_q[il][ih] = mean;

                // compute variance from m2 (Welford's algorithm)
                sigma_q[il][ih].resize(n_embd_head, 0.0);
                if (n > 1) {
                    for (int32_t i = 0; i < n_embd_head; ++i) {
                        sigma_q[il][ih][i] = m2[i] / (n - 1);
                    }
                }
            }
        }
        return {mu_q, sigma_q};
    }

    // print captured query statistics
    void print() const {
        fprintf(stdout, "%s: ------------------------------------------------------------\n", __func__);
        fprintf(stdout, "%s:  captured query statistics\n", __func__);
        fprintf(stdout, "%s: ------------------------------------------------------------\n", __func__);
        fprintf(stdout, "%s:  idx: <%ld, %ld>, n_runs: %ld, n_tokens: %ld\n",
                __func__, idx.first, idx.second, n_runs, n_tokens);
        fprintf(stdout, "%s: ------------------------------------------------------------\n", __func__);

        for (int32_t il = 0; il < n_layers; ++il) {
            // collect all means and variances for this layer
            all_means.clear();
            all_vars.clear();
            for (int32_t ih = 0; ih < n_head; ++ih) {
                const auto & [mean, m2] = observed_data[il][ih];
                const size_t n = n_samples_per_head[il][ih];

                for (int32_t i = 0; i < n_embd_head; ++i) {
                    all_means.push_back(mean[i]);

                    if (n > 1) {
                        double var = m2[i] / (n - 1);
                        all_vars.push_back(var);
                    }
                }
            }

            if (!all_means.empty()) {
                // compute mean and stddev of means
                double mean_of_means = 0.0;
                for (double val : all_means) {
                    mean_of_means += val;
                }
                mean_of_means /= all_means.size();

                double stddev_of_means = 0.0;
                for (double val : all_means) {
                    double diff = val - mean_of_means;
                    stddev_of_means += diff * diff;
                }
                stddev_of_means = std::sqrt(stddev_of_means / all_means.size());

                // compute mean and stddev of variances
                double mean_of_vars = 0.0;
                double stddev_of_vars = 0.0;
                if (!all_vars.empty()) {
                    for (double val : all_vars) {
                        mean_of_vars += val;
                    }
                    mean_of_vars /= all_vars.size();

                    for (double val : all_vars) {
                        double diff = val - mean_of_vars;
                        stddev_of_vars += diff * diff;
                    }
                    stddev_of_vars = std::sqrt(stddev_of_vars / all_vars.size());
                }

                fprintf(stdout, "%s: - layer %3d: mean: %8.4f ±%3.1f,  variance: %8.4f ±%3.1f\n",
                        __func__, il, mean_of_means, stddev_of_means, mean_of_vars, stddev_of_vars);
            } else {
                fprintf(stdout, "%s: - layer %3d: [no data]\n", __func__, il);
            }
        }
    }

    // given a computed distribution, print stats about it
    void print_distribution(
        const std::pair<
            std::vector<std::vector<std::vector<double>>>,
            std::vector<std::vector<std::vector<double>>>
        > & dist
    ) const {
        auto [mu_q, sigma_q] = dist;

        fprintf(stdout, "%s: ------------------------------------------------------------\n", __func__);
        fprintf(stdout, "%s:  computed query distribution\n", __func__);
        fprintf(stdout, "%s: ------------------------------------------------------------\n", __func__);

        for (int32_t il = 0; il < n_layers; ++il) {
            if (!mu_q[il].empty() && !mu_q[il][0].empty()) {
                double min_mu    =  std::numeric_limits<double>::infinity();
                double max_mu    = -std::numeric_limits<double>::infinity();
                double min_sigma =  std::numeric_limits<double>::infinity();
                double max_sigma = -std::numeric_limits<double>::infinity();

                for (int32_t ih = 0; ih < n_head; ++ih) {
                    for (int32_t ie = 0; ie < n_embd_head; ++ie) {
                        min_mu    = std::min(min_mu,       mu_q[il][ih][ie]);
                        max_mu    = std::max(max_mu,       mu_q[il][ih][ie]);
                        min_sigma = std::min(min_sigma, sigma_q[il][ih][ie]);
                        max_sigma = std::max(max_sigma, sigma_q[il][ih][ie]);
                    }
                }

                fprintf(stdout, "%s: - layer %3d: mu [%8.3f, %8.3f], sigma [%8.3f, %8.3f]\n",
                        __func__, il, min_mu, max_mu, min_sigma, max_sigma);
            } else {
                fprintf(stdout, "%s: - layer %3d: [no data]\n", __func__, il);
            }
        }
    }

    // calculate the total number of samples observed across all query heads in all layers
    size_t n_samples() const {
        size_t total = 0;
        for (const auto& layer : n_samples_per_head) {
            for (size_t count : layer) {
                total += count;
            }
        }
        return total;
    }
};

// parse layer index from tensor name
static int32_t parse_layer_index(const char * name) {
    std::string_view sv(name);
    auto dash_pos = sv.rfind('-');
    if (dash_pos == std::string_view::npos) {
        return -1;
    }
    auto n_part = sv.substr(dash_pos + 1);
    if (n_part.empty()) {
        return -1;
    }
    if (!std::all_of(n_part.begin(), n_part.end(), [](char c)
        { return c >= '0' && c <= '9'; }))
    {
        return -1;
    }
    int32_t result{};
    auto [ptr, ec] = std::from_chars(n_part.data(), n_part.data() + n_part.size(), result);
    if (ec != std::errc{}) {
        return -1;
    }
    return result;
}

// check if this tensor name starts with "Qcur-" and is not a (view) or (permuted)
static bool tensor_name_match(const char * name) {
    if /* OVERRIDE TO MATCH ALL NAMES FOR DEBUG?: */ (false)  {
        return true;
    }
    if (strncmp(name, "Qcur-", 5) != 0) {
        return false;
    }
    if (strchr(name, ' ') != nullptr) {
        // spaces indicate suffixes like " (view)" or " (permuted)"
        return false;
    }
    return true;
}

// print tensor name, shape, and type
static void print_tensor_info(const ggml_tensor * t) {
    fprintf(stdout, "%s: name = %8s, shape = [ %6ld, %6ld, %6ld, %6ld ], type = %s\n",
            __func__, t->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

// expected attention eval callback function
static bool expected_attn_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * stats = static_cast<expected_attn_stats*>(user_data);
    if (ask) {
        // the scheduler wants to know if we want to observe this tensor. if the shape is what
        // we expect, and the tensor name matches, then yes, we do.
        ++stats->idx.first;
        return (
            // TODO: this check works for Qwen3 and likely many other models, but not all models
            t->ne[0] == stats->n_embd_head &&
            t->ne[1] == stats->n_head      &&
            tensor_name_match(t->name)
        );
    } else {
        // the scheduler is giving us a tensor to observe
        print_tensor_info(t);

        GGML_ASSERT(t->ne[0] == stats->n_embd_head && t->ne[1] == stats->n_head &&
                    "unexpected shape - this should not happen");

        const int64_t n_tokens = t->ne[2];
        const int32_t il = parse_layer_index(t->name);

        // increment stat counters
        ++stats->idx.second;
        if (il == 0) {
            ++stats->n_runs;
            // only increment the n_tokens counter once per graph execution (not every layer)
            // TODO: is there a more elegant way to check per-execution?
            stats->n_tokens += n_tokens;
        }

        // allocate buffer and get the tensor data from the backend
        const int64_t n_elements = stats->n_embd_head * stats->n_head * n_tokens;
        GGML_ASSERT(n_elements == ggml_nelements(t));
        std::vector<float> buffer(n_elements);
        ggml_backend_tensor_get(t, buffer.data(), 0, ggml_nbytes(t));

        //
        // accumulate statistics from the tensor data using Welford's algorithm
        //

        // iterate over all tokens
        for (int64_t it = 0; it < n_tokens; ++it) {
            // for each query head
            for (int64_t ih = 0; ih < stats->n_head; ++ih) {
                ++stats->n_samples_per_head[il][ih];
                const size_t n = stats->n_samples_per_head[il][ih];

                auto & mean = stats->observed_data[il][ih].first;
                auto & m2   = stats->observed_data[il][ih].second;

                // for each dimension in this head
                for (int64_t ie = 0; ie < stats->n_embd_head; ++ie) {
                    const size_t idx = ie + ih * stats->n_embd_head + it * stats->n_embd_head * stats->n_head;
                    const double value = static_cast<double>(buffer[idx]);

                    // Welford's online algorithm
                    const double delta = value - mean[ie];
                    mean[ie] += delta / n;
                    const double delta2 = value - mean[ie];
                    m2[ie] += delta * delta2;
                }
            }
        }

        return true; // return false to cancel graph computation
    }
}

int main() {

    // init llama_model

    llama_model_params model_params = llama_model_default_params();
    model_params.check_tensors = true;
    model_params.n_gpu_layers  = 999;
    model_params.use_mmap      = false;
    model_params.use_mlock     = false;
    model_params.use_direct_io = false;
    llama_model * model = llama_model_load_from_file(MODEL_PATH, model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }

    const int32_t n_embd    = llama_model_n_embd(model);
    const int32_t n_head    = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_layers  = llama_model_n_layer(model);

    // callback statistics
    expected_attn_stats cb_stats(n_embd, n_head, n_head_kv, n_layers);

    // init llama_context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.offload_kqv       = true;
    ctx_params.n_ubatch          = 2560;
    ctx_params.n_batch           = 5120;
    ctx_params.n_ctx             = 5120;
    ctx_params.kv_unified        = true;
    ctx_params.n_seq_max         = 1;
    ctx_params.n_threads         = 8;
    ctx_params.n_threads_batch   = 8;
    ctx_params.cb_eval           = expected_attn_eval_cb;
    ctx_params.cb_eval_user_data = &cb_stats;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED; // need to test flash attention both enabled and disabled
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    // prepare dummy prompt processing input (TODO: eventually need to use real text)
    llama_batch pp_batch = llama_batch_init(/* n_tokens */ ctx_params.n_batch, /* embd */ 0, /* n_seq_max */ ctx_params.n_seq_max);
    pp_batch.n_tokens = ctx_params.n_batch;
    for (int32_t i = 0; i < pp_batch.n_tokens; ++i) {
        pp_batch.token[i]     = (llama_token) i; // use position as token ID for now
        pp_batch.pos[i]       = (llama_pos)   i;
        pp_batch.n_seq_id[i]  = 1;
        pp_batch.seq_id[i][0] = 0;
    }

    // run dummy prompt processing
    int32_t return_code = llama_decode(ctx, pp_batch);
    if (return_code != GGML_STATUS_SUCCESS) {
        llama_batch_free(pp_batch);
        llama_free(ctx);
        llama_model_free(model);
        throw std::runtime_error("dummy PP failed");
    }

    // display accumulated statistics
    cb_stats.print();

    // compute query distribution
    auto & dist = cb_stats.compute();

    // print query distribution
    cb_stats.print_distribution(dist);

    //
    // TODO: calculate importance scores for all KV entries based on `dist`
    //

    //
    // TODO: evict the least important x% of KV entries
    //

    // cleanup
    llama_batch_free(pp_batch); // llama_batch
    llama_free(ctx);            // llama_context
    llama_model_free(model);    // llama_model
    return GGML_STATUS_SUCCESS;
}
