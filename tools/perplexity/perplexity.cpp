#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <chrono>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct results_perplexity {
    std::vector<llama_token> tokens;
    double                   ppl_value;
    std::vector<float>       logits;
    std::vector<float>       probs;
};

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static inline int nearest_int(float fval) {
    //assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *)log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history
) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + size_t(i)*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static void process_logits(std::ostream& out, int n_vocab, const float * logits, const int * tokens, int n_token,
        std::vector<std::thread> & workers, std::vector<uint16_t> & log_probs, double & nll, double & nll2) {
    std::mutex mutex;
    const int nv = 2*((n_vocab + 1)/2) + 4;
    int counter = 0;
    auto compute = [&mutex, &counter, &log_probs, &nll, &nll2, n_vocab, logits, tokens, n_token, nv] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const double v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, log_probs.data() + i*nv, tokens[i+1]);
            local_nll += v;
            local_nll2 += v*v;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
    out.write((const char *)log_probs.data(), n_token*nv*sizeof(uint16_t));
}

struct kl_divergence_result {
    double sum_nll          = 0.0;
    double sum_nll2         = 0.0;
    double sum_nll_base     = 0.0;
    double sum_nll_base2    = 0.0;
    double sum_nll_nll_base = 0.0;
    double sum_kld          = 0.0;
    double sum_kld2         = 0.0;
    double sum_p_diff       = 0.0;
    double sum_p_diff2      = 0.0;
    double sum_p_diff4      = 0.0;
    float  max_p_diff       = 0.0f;
    size_t n_same_top       = 0.0;
    size_t count            = 0.0;
};

static std::pair<double, float> log_softmax(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kl_divergence_result & kld) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float * d = (const float *)base_log_prob;
    const float scale = d[0];
    const float min_log_prob = d[1];
    base_log_prob += 4;

    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll  += nll;
    kld.sum_nll2 += nll*nll;

    const float nll_base = -(scale*base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base  += nll_base;
    kld.sum_nll_base2 += nll_base*nll_base;

    kld.sum_nll_nll_base += nll*nll_base;

    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale*base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld  += sum;
    kld.sum_kld2 += sum*sum;
    ++kld.count;
    if (imax == imax_base) {
        ++kld.n_same_top;
    }

    const float p_base = expf(-nll_base);
    const float p = expf(-nll);
    const float p_diff = p - p_base;
    kld.sum_p_diff  += p_diff;
    const double p_diff2 = p_diff*p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2*p_diff2;
    kld.max_p_diff = std::max(kld.max_p_diff, std::fabs(p_diff));

    return std::make_pair(sum, p_diff);
}

static void process_logits(int n_vocab, const float * logits, const int * tokens, int n_token,
        std::vector<std::thread> & workers, const std::vector<uint16_t> & base_log_probs, kl_divergence_result & kld,
        float * kld_values, float * p_diff_values) {
    std::mutex mutex;
    const int nv = 2*((n_vocab + 1)/2) + 4;
    int counter = 0;
    auto compute = [&mutex, &counter, &base_log_probs, &kld, n_vocab, logits, tokens, n_token, nv, kld_values, p_diff_values] () {
        kl_divergence_result local_kld;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                kld.sum_nll          += local_kld.sum_nll;
                kld.sum_nll2         += local_kld.sum_nll2;
                kld.sum_nll_base     += local_kld.sum_nll_base;
                kld.sum_nll_base2    += local_kld.sum_nll_base2;
                kld.sum_nll_nll_base += local_kld.sum_nll_nll_base;
                kld.sum_kld          += local_kld.sum_kld;
                kld.sum_kld2         += local_kld.sum_kld2;
                kld.sum_p_diff       += local_kld.sum_p_diff;
                kld.sum_p_diff2      += local_kld.sum_p_diff2;
                kld.sum_p_diff4      += local_kld.sum_p_diff4;
                kld.n_same_top       += local_kld.n_same_top;
                kld.max_p_diff        = std::max(kld.max_p_diff, local_kld.max_p_diff);
                kld.count            += local_kld.count;
                break;
            }
            lock.unlock();
            std::pair<double, float> v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, base_log_probs.data() + i*nv, tokens[i+1], local_kld);
            kld_values[i]    = (float)v.first;
            p_diff_values[i] = v.second;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static results_perplexity perplexity_v2(llama_context * ctx, const common_params & params) {
    // Download: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    logit_history.resize(tokens.size());
    prob_history.resize(tokens.size());

    if (params.ppl_stride <= 0) {
        LOG_ERR("%s: stride is %d but must be greater than zero!\n",__func__,params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int calc_chunk = n_ctx;

    LOG_INF("%s: have %zu tokens. Calculation chunk = %d\n", __func__, tokens.size(), calc_chunk);

    if (int(tokens.size()) <= calc_chunk) {
        LOG_ERR("%s: there are only %zu tokens, this is not enough for a context size of %d and stride %d\n",__func__,
                tokens.size(), n_ctx, params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int n_chunk_max = (tokens.size() - calc_chunk + params.ppl_stride - 1)  / params.ppl_stride;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int count = 0;
    double nll = 0.0;

    LOG_INF("%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.ppl_stride;
        const int end   = start + calc_chunk;

        const int num_batches = (calc_chunk + n_batch - 1) / n_batch;
        //LOG_DBG("%s: evaluating %d...%d using %d batches\n", __func__, start, end, num_batches);

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            common_batch_clear(batch);
            for (int i = 0; i < batch_size; i++) {
                common_batch_add(batch, tokens[batch_start + i], j*n_batch + i, {0}, true);
            }

            //LOG_DBG("    Batch %d: starts at %d, size is %d, n_past is %d\n",j,batch_start,batch_size,j * n_batch);
            if (llama_decode(ctx, batch)) {
                //LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return {tokens, -1, logit_history, prob_history};
            }

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_vocab_bos(vocab);
            }

            const auto * batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + size_t(batch_size) * n_vocab);

            if (j == 0) {
                tokens[batch_start] = token_org;
            }
        }

        llama_batch_free(batch);

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        //LOG_DBG("%s: using tokens %d...%d\n",__func__,params.n_ctx - params.ppl_stride + start, params.n_ctx + start);
        for (int j = n_ctx - params.ppl_stride - 1; j < n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + size_t(j + 0) * n_vocab,
                logits.begin() + size_t(j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[start + j + 1]];
            logit_history[start + j + 1] = tok_logits[tokens[start + j + 1]];
            prob_history[start + j + 1]  = prob;

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            LOG("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            LOG("%8d  %.4lf\n", i*params.ppl_stride, std::exp(nll / count));
        }
    }
    LOG("\n");

    return {tokens, std::exp(nll / count), logit_history, prob_history};
}

static results_perplexity perplexity(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    if (params.ppl_stride > 0) {
        return perplexity_v2(ctx, params);
    }

    // Download: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    // Run `./llama-perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    std::ofstream logits_stream;
    if (!params.logits_file.empty()) {
        logits_stream.open(params.logits_file.c_str(), std::ios::binary);
        if (!logits_stream.is_open()) {
            LOG_ERR("%s: failed to open %s for writing\n", __func__, params.logits_file.c_str());
            return {};
        }
        LOG_INF("%s: saving all logits to %s\n", __func__, params.logits_file.c_str());
        logits_stream.write("_logits_", 8);
        logits_stream.write(reinterpret_cast<const char *>(&n_ctx), sizeof(n_ctx));
    }

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    logit_history.resize(tokens.size());

    std::vector<float> prob_history;
    prob_history.resize(tokens.size());

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    LOG_INF("%s: calculating perplexity over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    std::vector<uint16_t> log_probs;
    if (!params.logits_file.empty()) {
        logits_stream.write((const char *)&n_vocab, sizeof(n_vocab));
        logits_stream.write((const char *)&n_chunk, sizeof(n_chunk));
        logits_stream.write((const char *)tokens.data(), n_chunk*n_ctx*sizeof(tokens[0]));
        const int nv = 2*((n_vocab + 1)/2) + 4;
        log_probs.resize(n_ctx * nv);
    }

    // We get the logits for all the tokens in the context window (params.n_ctx)
    // from llama_decode below.  Now, based on https://huggingface.co/docs/transformers/perplexity,
    // calculate the perplexity over the last half of the window (so the model always has
    // some context to predict the token).
    //
    // We rely on the fact that attention in the forward pass only looks at previous
    // tokens here, so the logits returned for each token are an accurate representation
    // of what the model would have predicted at that point.
    //
    // Example, we have a context window of 512, we will compute perplexity for each of the
    // last 256 tokens.  Then, we split the input up into context window size chunks to
    // process the entire prompt.
    const int first = n_ctx/2;

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            int n_outputs = 0;

            batch.n_tokens = 0;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after decode
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int idx = seq*n_ctx + k;
                    batch.token   [idx]    = tokens[seq_start + k];
                    batch.pos     [idx]    = j*n_batch + k;
                    batch.n_seq_id[idx]    = 1;
                    batch.seq_id  [idx][0] = seq;
                    batch.logits  [idx]    = batch.pos[idx] >= first ? 1 : 0;

                    n_outputs += batch.logits[idx] != 0;
                }
                batch.n_tokens += batch_size;

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_INF("%s : failed to decode\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            if (num_batches > 1 && n_outputs > 0) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total*n_chunk/n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        for (int seq = 0; seq < n_seq_batch; seq++) {
            const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);

            llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;
            if (!params.logits_file.empty()) {
                process_logits(logits_stream, n_vocab, all_logits,
                        tokens_data, n_ctx - 1 - first,
                        workers, log_probs, nll, nll2);
            } else {
                process_logits(n_vocab, all_logits,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);
            }
            count += n_ctx - first - 1;

            // perplexity is e^(average negative log-likelihood)
            if (params.ppl_output_type == 0) {
                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            } else {
                double av = nll/count;
                double av2 = nll2/count - av*av;
                if (av2 > 0) {
                    av2 = sqrt(av2/(count-1));
                }
                LOG("%8d  %.4lf  %4lf  %4lf\n", i*n_ctx, std::exp(nll / count), av, av2);
            }
        }

        logits.clear();
    }
    LOG("\n");

    nll2 /= count;
    nll /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        LOG_INF("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
    } else {
        LOG_ERR("Unexpected negative standard deviation of log(prob)\n");
    }

    llama_batch_free(batch);

    return {tokens, ppl, logit_history, prob_history};
}

static bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int n_batch, int n_vocab) {
    int prev_outputs = 0;
    for (int i = 0; i < (int) batch.n_tokens; i += n_batch) {
        const int n_tokens = std::min<int>(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
            return false;
        }

        int n_outputs = 0;
        for (int i = 0; i < n_tokens; ++i) {
            n_outputs += batch_view.logits[i] != 0;
        }

        memcpy(batch_logits.data() + size_t(prev_outputs)*n_vocab, llama_get_logits(ctx), size_t(n_outputs)*n_vocab*sizeof(float));

        prev_outputs += n_outputs;
    }

    return true;
}

#define K_TOKEN_CHUNK 4

static void compute_logprobs(const float * batch_logits, int n_vocab, std::vector<std::thread>& workers,
        const std::vector<std::pair<size_t, llama_token>>& eval_pairs, std::vector<float>& eval_results) {
    if (eval_results.size() != eval_pairs.size()) {
        eval_results.resize(eval_pairs.size());
    }
    if (eval_pairs.empty()) {
        return;
    }

    size_t max_threads = std::min((eval_pairs.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK, workers.size());

    std::atomic<int> counter(0);
    auto compute = [&counter, &eval_pairs, &eval_results, batch_logits, n_vocab] () {
        float local_logprobs[K_TOKEN_CHUNK];
        while (true) {
            const size_t first = counter.fetch_add(K_TOKEN_CHUNK, std::memory_order_relaxed);
            if (first >= eval_results.size()) {
                break;
            }
            const size_t last = std::min(first + K_TOKEN_CHUNK, eval_results.size());
            for (size_t i = first; i < last; ++i) {
                const auto * logits = batch_logits + eval_pairs[i].first * n_vocab;
                float max_logit = logits[0];
                for (int j = 1; j < n_vocab; ++j) {
                    max_logit = std::max(max_logit, logits[j]);
                }
                float sum_p = 0.f;
                for (int j = 0; j < n_vocab; ++j) {
                    sum_p += expf(logits[j] - max_logit);
                }
                local_logprobs[i - first] = logits[eval_pairs[i].second] - max_logit - std::log(sum_p);
            }
            std::memcpy(eval_results.data() + first, local_logprobs, (last - first)*sizeof(float));
        }
    };

    for (size_t it = 0; it < max_threads; ++it) {
        workers[it] = std::thread(compute);
    }
    for (size_t it = 0; it < max_threads; ++it) {
        workers[it].join();
    }
}

static void hellaswag_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Calculates hellaswag score (acc_norm) from prompt
    //
    // Data extracted from the HellaSwag validation dataset (MIT license) https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
    // All used data fields are preprocessed as in https://github.com/EleutherAI/lm-evaluation-harness/blob/df3da98c5405deafd519c2ddca52bb7c3fe36bef/lm_eval/tasks/hellaswag.py#L62-L68
    //
    // All 10042 tasks should be extracted to keep the results standardized like other implementations.
    //
    // Datafile layout:
    // ['??'] denotes json fields
    // 6 lines per task:
    // ['activity_label'] + ": " +['ctx']  - The first part of the query, the context
    // ['label'] - The index the best common sense ending aka gold ending
    // ['endings'][0] - Endings added to the first part of the query
    // ['endings'][1]
    // ['endings'][2]
    // ['endings'][3]

    std::vector<std::string> prompt_lines;
    std::istringstream strstream(params.prompt);
    std::string line;

    while (std::getline(strstream,line,'\n')) {
        prompt_lines.push_back(line);
    }

    if (prompt_lines.size() % 6 != 0) {
        LOG_ERR("%s : number of lines in prompt not a multiple of 6.\n", __func__);
        return;
    }

    size_t hs_task_count = prompt_lines.size()/6;
    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, hs_task_count);

    const bool is_spm = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    LOG_INF("================================= is_spm = %d\n", is_spm);

    // The tasks should be randomized so the score stabilizes quickly.
    bool randomize_tasks = true;

    // Number of tasks to use when computing the score
    if (params.hellaswag_tasks < hs_task_count) {
        hs_task_count = params.hellaswag_tasks;
    }

    // The random seed should not impact the final result if the computation is done over enough tasks, so kept hardcoded for now
    std::mt19937 rng(1);

    // Dataholder for hellaswag tasks
    struct hs_data_t {
        std::string context;
        size_t gold_ending_idx;
        std::string ending[4];
        size_t ending_logprob_count[4];
        double ending_logprob[4];

        size_t i_logits;        // starting index of logits in the llama_batch
        size_t common_prefix;   // max number of initial tokens that are the same in all sentences
        size_t required_tokens; // needed number of tokens to evaluate all 4 endings
        std::vector<llama_token> seq_tokens[4];
    };

    LOG_INF("%s : selecting %zu %s tasks.\n", __func__, hs_task_count, (randomize_tasks?"randomized":"the first")  );

    // Select and read data from prompt lines
    std::vector<hs_data_t> hs_data(hs_task_count);
    for (size_t i = 0; i < hs_task_count; i++) {
        size_t idx = i;

        auto & hs_cur = hs_data[i];

        // Select a random example of those left in the prompt
        if (randomize_tasks) {
            std::uniform_int_distribution<size_t> dist(0, prompt_lines.size()/6-1 ) ;
            idx = dist(rng);
        }

        hs_cur.context = prompt_lines[idx*6];
        hs_cur.gold_ending_idx = std::stoi( prompt_lines[idx*6+1] );
        for (size_t j = 0; j < 4; j++) {
            hs_cur.ending[j] = prompt_lines[idx*6+2+j];
            hs_cur.seq_tokens[j] = common_tokenize(ctx, hs_cur.context + " " + hs_cur.ending[j], true);
        }

        // determine the common prefix of the endings
        hs_cur.common_prefix = 0;
        for (size_t k = 0; k < hs_cur.seq_tokens[0].size(); k++) {
            if (hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[1][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[2][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[3][k]) {
                break;
            }
            hs_cur.common_prefix++;
        }
        hs_cur.required_tokens = hs_cur.common_prefix +
            hs_cur.seq_tokens[0].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[1].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[2].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[3].size() - hs_cur.common_prefix;

        //GGML_ASSERT(hs_cur.common_prefix >= ::llama_tokenize(ctx, hs_cur.context, true).size());

        // Delete the selected random example from the prompt
        if (randomize_tasks) {
            prompt_lines.erase( std::next(prompt_lines.begin(),idx*6)  , std::next(prompt_lines.begin(),idx*6+6) );
        }
    }

    LOG_INF("%s : calculating hellaswag score over selected tasks.\n", __func__);

    LOG("\ntask\tacc_norm\t95%% confidence interval\n");

    double acc = 0.0f;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 4);

    std::vector<float> tok_logits(n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    for (size_t i0 = 0; i0 < hs_task_count; i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        while (n_cur + (int) hs_data[i1].required_tokens <= n_ctx) {
            auto & hs_cur = hs_data[i1];
            int n_logits = 0;

            const int s0 = 4*(i1 - i0);
            if (s0 + 4 > max_seq) {
                break;
            }

            for (size_t i = 0; i < hs_cur.common_prefix; ++i) {
                common_batch_add(batch, hs_cur.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < 4; ++s) {
                const size_t seq_tokens_size = hs_cur.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = hs_cur.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, hs_cur.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            hs_cur.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += hs_data[i1].required_tokens;
            if (++i1 == hs_task_count) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, hs_data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];
            size_t li = 1; // skip the last logit of the common prefix (computed separately below)
            for (int s = 0; s < 4; ++s) {
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(hs_cur.i_logits + li++, hs_cur.seq_tokens[s][j + 1]);
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];

            // get the logits of the last token of the common prefix
            std::memcpy(tok_logits.data(), batch_logits.data() + hs_cur.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            for (int s = 0; s < 4; ++s) {
                hs_cur.ending_logprob_count[s] = 1;
                hs_cur.ending_logprob[s] = std::log(first_probs[hs_cur.seq_tokens[s][hs_cur.common_prefix]]);
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    hs_cur.ending_logprob[s] += eval_results[ir++];
                    hs_cur.ending_logprob_count[s]++;
                }
                hs_cur.ending_logprob[s] /= hs_cur.ending_logprob_count[s];
            }

            // Find the ending with maximum logprob
            size_t ending_logprob_max_idx = 0;
            double ending_logprob_max_val = hs_cur.ending_logprob[0];
            for (size_t s = 1; s < 4; s++) {
                if (hs_cur.ending_logprob[s] > ending_logprob_max_val) {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val =  hs_cur.ending_logprob[s];
                }
            }

            //LOG("max logprob ending idx %lu, gold ending idx %lu\n", ending_logprob_max_idx, hs_cur.gold_ending_idx);

            // If the gold ending got the maximum logprobe add one accuracy point
            if (ending_logprob_max_idx == hs_cur.gold_ending_idx) {
                acc += 1.0;
            }

            double freq = acc / double(i + 1);

            const double za = 1.95996398454;

            // // Wald normal approx
            // double conf =za*sqrt(freq*(1-freq)/double(i + 1));
            // LOG("%zu\t%.8lf +/- %.8lf\n", i + 1, freq*100.0, conf*100.0);

            // Wilson score interval, more accurate
            double z   = za * za / double(i + 1);
            double cnf = z * sqrt(double(i + 1) * (4.0 * freq * (1 - freq) + z)) / (za + za);
            double a   = (freq + z * 0.5 - cnf) / (1.0 + z);
            double b   = (freq + z * 0.5 + cnf) / (1.0 + z);

            // Print the accumulated accuracy mean x 100 and confidence interval
            LOG("%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]\n", i + 1, freq * 100.0, a * 100.0, b * 100.0);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    LOG("\n");
}

struct winogrande_entry {
    std::string first;
    std::string second;
    std::array<std::string, 2> choices;
    int answer;

    size_t i_logits;
    size_t common_prefix;
    size_t required_tokens;
    size_t n_base1; // number of tokens for context + choice 1
    size_t n_base2; // number of tokens for context + choice 2
    std::vector<llama_token> seq_tokens[2];
};

static std::vector<winogrande_entry> load_winogrande_from_csv(const std::string & prompt) {
    std::vector<winogrande_entry> result;
    std::istringstream in(prompt);
    std::string line;
    std::array<int, 4> comma_pos;
    while (true) {
        std::getline(in, line);
        if (in.fail() || in.eof()) break;
        int ipos = 0;
        bool quote_open = false;
        for (int i = 0; i < int(line.size()); ++i) {
            if (!quote_open) {
                if (line[i] == ',') {
                    comma_pos[ipos++] = i;
                    if (ipos == 4) break;
                }
                else if (line[i] == '"') {
                    quote_open = true;
                }
            }
            else {
                if (line[i] == '"') {
                    quote_open = false;
                }
            }
        }
        if (ipos != 4) {
            LOG_ERR("%s: failed to find comma separators in <%s>\n", __func__, line.c_str());
            continue;
        }
        auto sentence = line[comma_pos[0]+1] == '"' ? line.substr(comma_pos[0]+2, comma_pos[1] - comma_pos[0] - 3)
                                                    : line.substr(comma_pos[0]+1, comma_pos[1] - comma_pos[0] - 1);
        auto choice1 = line.substr(comma_pos[1]+1, comma_pos[2] - comma_pos[1] - 1);
        auto choice2 = line.substr(comma_pos[2]+1, comma_pos[3] - comma_pos[2] - 1);
        auto answer  = line.substr(comma_pos[3]+1, line.size() - comma_pos[3] - 1);
        auto index = line.substr(0, comma_pos[0]);
        int where = 0;
        for ( ; where < int(sentence.size()); ++where) {
            if (sentence[where] == '_') break;
        }
        if (where == int(sentence.size())) {
            LOG_ERR("%s: no _ in <%s>\n", __func__, sentence.c_str());
            continue;
        }
        std::istringstream stream(answer.c_str());
        int i_answer; stream >> i_answer;
        if (stream.fail() || i_answer < 1 || i_answer > 2) {
            LOG_ERR("%s: failed to parse answer <%s>\n", __func__, answer.c_str());
            continue;
        }
        result.emplace_back();
        auto& wg = result.back();
        wg.first = sentence.substr(0, where);
        wg.second = sentence.substr(where + 1, sentence.size() - where - 1);
        wg.choices[0] = std::move(choice1);
        wg.choices[1] = std::move(choice2);
        wg.answer = i_answer;
    }
    return result;
}

/*
 * Evaluates the Winogrande score.
 * Uses a CSV containing task index, dentence, choice 1, choice 2, answer (1 or 2)
 * You can get one such dataset from e.g. https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp
 * As an example, the 1st row in the above dataset is
 *
 *    0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2
 *
 */
static void winogrande_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    constexpr int k_min_trailing_ctx = 3;

    auto data = load_winogrande_from_csv(params.prompt);
    if (data.empty()) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }

    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, data.size());

    if (params.winogrande_tasks > 0 && params.winogrande_tasks < data.size()) {
        LOG_INF("%s : selecting %zu random tasks\n", __func__, params.winogrande_tasks);
        std::mt19937 rng(1);
        std::vector<int> aux(data.size());
        for (int i = 0; i < int(data.size()); ++i) {
            aux[i] = i;
        }
        float scale = 1/(1.f + (float)rng.max());
        std::vector<winogrande_entry> selected;
        selected.resize(params.winogrande_tasks);
        for (int i = 0; i < int(params.winogrande_tasks); ++i) {
            int j = int(scale*rng()*aux.size());
            selected[i] = std::move(data[aux[j]]);
            aux[j] = aux.back();
            aux.pop_back();
        }
        data = std::move(selected);
    }

    LOG_INF("%s : tokenizing selected tasks\n", __func__);

    for (auto & task : data) {
        task.seq_tokens[0] = common_tokenize(ctx, task.first + task.choices[0] + task.second, true);
        task.seq_tokens[1] = common_tokenize(ctx, task.first + task.choices[1] + task.second, true);

        task.common_prefix = 0;
        for (size_t k = 0; k < task.seq_tokens[0].size(); k++) {
            if (task.seq_tokens[0][k] != task.seq_tokens[1][k]) {
                break;
            }
            task.common_prefix++;
        }

        // TODO: the last token of each of the sequences don't need to be evaluated
        task.required_tokens = task.common_prefix +
            task.seq_tokens[0].size() - task.common_prefix +
            task.seq_tokens[1].size() - task.common_prefix;

        task.n_base1 = common_tokenize(ctx, task.first + task.choices[0], true).size();
        task.n_base2 = common_tokenize(ctx, task.first + task.choices[1], true).size();
    }

    LOG_INF("%s : calculating winogrande score over selected tasks.\n", __func__);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 128;
    const int max_seq = std::min(2*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 2);

    std::vector<float> tok_logits(n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    int n_correct = 0;
    int n_done    = 0;

    for (size_t i0 = 0; i0 < data.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0;

        common_batch_clear(batch);

        while (n_cur + (int) data[i1].required_tokens <= n_ctx) {
            int n_logits = 0;
            const int s0 = 2*(i1 - i0);
            if (s0 + 2 > max_seq) {
                break;
            }

            for (size_t i = 0; i < data[i1].common_prefix; ++i) {
                common_batch_add(batch, data[i1].seq_tokens[0][i], i, { s0 + 0, s0 + 1 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true;
            n_logits += 1;

            for (int s = 0; s < 2; ++s) {
                // TODO: end before the last token, no need to predict past the end of the sequences
                for (size_t i = data[i1].common_prefix; i < data[i1].seq_tokens[s].size(); ++i) {
                    common_batch_add(batch, data[i1].seq_tokens[s][i], i, { s0 + s }, true);
                    n_logits += 1;
                }
            }

            data[i1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += data[i1].required_tokens;
            if (++i1 == data.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            size_t li = n_base1 - task.common_prefix;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[0][j+1]);
            }
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            // FIXME: this uses the wrong first logits when not skipping the choice word
            li = task.seq_tokens[0].size() - task.common_prefix + n_base2 - task.common_prefix;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[1][j+1]);
            }
        }
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            float score_1st = 0;
            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                score_1st += eval_results[ir++];
            }
            score_1st /= (task.seq_tokens[0].size() - n_base1 - last_1st);

            float score_2nd = 0;
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                score_2nd += eval_results[ir++];
            }
            score_2nd /= (task.seq_tokens[1].size() - n_base2 - last_2nd);

            int result = score_1st > score_2nd ? 1 : 2;

            if (result == task.answer) {
                ++n_correct;
            }
            ++n_done;

            // print the accumulated accuracy mean x 100
            LOG("%zu\t%.4lf\t%10.6f  %10.6f  %d  %d\n", i+1, 100.0 * n_correct/n_done, score_1st, score_2nd, result, task.answer);
        }

        i0 = i1 - 1;
    }

    LOG("\n");

    if (n_done < 100) return;

    const float p = 1.f*n_correct/n_done;
    const float sigma = 100.f*sqrt(p*(1-p)/(n_done-1));

    LOG_INF("Final Winogrande score(%d tasks): %.4lf +/- %.4lf\n", n_done, 100*p, sigma);
}

static bool deserialize_string(std::istream & in, std::string & str) {
    uint32_t size;
    if (!in.read((char *)&size, sizeof(size)).fail()) {
        str.resize(size);
        if (!in.read((char *)&str[0], size).fail()) return true;
    }
    return false;
}

struct multiple_choice_answers {
    std::vector<std::string> answers;
    std::vector<int>         labels;
    bool deserialize(std::istream& in) {
        uint32_t n;
        in.read((char *)&n, sizeof(n));
        if (in.fail() || n > 100) return false; // 100 as max. number of answers should be good enough for any practical purpose
        answers.resize(n);
        labels.resize(n);
        for (auto& a : answers) {
            if (!deserialize_string(in, a)) return false;
        }
        in.read((char *)labels.data(), n*sizeof(int));
        return !in.fail();
    }
};

struct multiple_choice_task {
    std::string question;         // the question (or context that needs to be continued)
    multiple_choice_answers mc1;  // possible answers (continuations) with a single correct answer
    multiple_choice_answers mc2;  // possible answers (continuations) with multiple correct answers - not handled yet
    bool deserialize(std::istream& in) {
        if (!deserialize_string(in, question)) return false;
        return mc1.deserialize(in) && mc2.deserialize(in);
    }

    // For evaluation
    size_t i_logits;        // starting index of logits in the llama_batch
    size_t common_prefix;   // max number of initial tokens that are the same in all sentences
    size_t required_tokens; // needed number of tokens to evaluate all answers
    std::vector<std::vector<llama_token>> seq_tokens;
    std::vector<float> log_probs;
};

static bool multiple_choice_prepare_one_task(llama_context * ctx, multiple_choice_task& task, bool log_error) {
    if (task.question.empty() || task.mc1.answers.empty()) {
        if (log_error) {
            LOG_ERR("%s: found bad task with empty question and/or answers\n", __func__);
        }
        return false;
    }
    task.seq_tokens.reserve(task.mc1.answers.size());
    for (auto& answer : task.mc1.answers) {
        if (answer.empty()) {
            if (log_error) {
                LOG_ERR("%s: found empty answer\n", __func__);
            }
            return false;
        }
        task.seq_tokens.emplace_back(::common_tokenize(ctx, task.question + " " + answer, true));
    }
    auto min_len = task.seq_tokens.front().size();
    for (auto& seq : task.seq_tokens) {
        min_len = std::min(min_len, seq.size());
    }
    task.common_prefix = 0;
    for (size_t k = 0; k < min_len; ++k) {
        auto token = task.seq_tokens[0][k];
        bool all_same = true;
        for (size_t i = 1; i < task.seq_tokens.size(); ++i) {
            if (task.seq_tokens[i][k] != token) {
                all_same = false;
                break;
            }
        }
        if (!all_same) {
            break;
        }
        ++task.common_prefix;
    }
    task.required_tokens = task.common_prefix;
    for (auto& seq : task.seq_tokens) {
        task.required_tokens += seq.size() - task.common_prefix;
    }
    return true;
}

//
// Calculates score for multiple choice tasks with single correct answer from prompt.
// Commonly used LLM evaluation metrics of this type are
//   * ARC
//   * HellaSwag
//   * MMLU
//   * TruthfulQA
//
// Validation datasets for these 4 tests can be found at
//     https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp
// The data for these datasets was extracted from
//     git@hf.co:datasets/allenai/ai2_arc
//     https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
//     git@hf.co:datasets/Stevross/mmlu
//     https://huggingface.co/datasets/truthful_qa
//
static void multiple_choice_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::istringstream strstream(params.prompt);
    uint32_t n_task;
    strstream.read((char *)&n_task, sizeof(n_task));
    if (strstream.fail() || n_task == 0) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }
    LOG_INF("%s: there are %u tasks in prompt\n", __func__, n_task);
    std::vector<uint32_t> task_pos(n_task);
    strstream.read((char *)task_pos.data(), task_pos.size()*sizeof(uint32_t));
    if (strstream.fail()) {
        LOG_ERR("%s: failed to read task positions from prompt\n", __func__);
        return;
    }

    std::vector<multiple_choice_task> tasks;
    if (params.multiple_choice_tasks == 0 || params.multiple_choice_tasks >= (size_t)n_task) {
        // Use all tasks
        tasks.resize(n_task);
        LOG_INF("%s: reading tasks", __func__);
        int n_dot = std::max((int) n_task/100, 1);
        int i = 0;
        for (auto& task : tasks) {
            ++i;
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d of %u\n", __func__, i, n_task);
                return;
            }
            if (i%n_dot == 0) LOG(".");
        }
        LOG("done\n");
    }
    else {
        LOG_INF("%s: selecting %zu random tasks from %u tasks available\n", __func__, params.multiple_choice_tasks, n_task);
        std::mt19937 rng(1);
        std::vector<int> aux(n_task);
        for (uint32_t i = 0; i < n_task; ++i) aux[i] = i;
        float scale = 1.f/(1.f + (float)std::mt19937::max());
        tasks.resize(params.multiple_choice_tasks);
        for (auto& task : tasks) {
            int j = (int)(scale * rng() * aux.size());
            int idx = aux[j];
            aux[j] = aux.back();
            aux.pop_back();
            strstream.seekg(task_pos[idx], std::ios::beg);
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d at position %u\n", __func__, idx, task_pos[idx]);
                return;
            }
        }
        n_task = params.multiple_choice_tasks;
    }

    LOG_INF("%s: preparing task data", __func__);
    if (n_task > 500) {
        LOG("...");
        std::atomic<int> counter(0);
        std::atomic<int> n_bad(0);
        auto prepare = [&counter, &n_bad, &tasks, ctx] () {
            int num_tasks = tasks.size();
            int n_bad_local = 0;
            while (true) {
                int first = counter.fetch_add(K_TOKEN_CHUNK);
                if (first >= num_tasks) {
                    if (n_bad_local > 0) n_bad += n_bad_local;
                    break;
                }
                int last = std::min(first + K_TOKEN_CHUNK, num_tasks);
                for (int i = first; i < last; ++i) {
                    if (!multiple_choice_prepare_one_task(ctx, tasks[i], false)) ++n_bad_local;
                }
            }
        };
        size_t max_thread = std::thread::hardware_concurrency();
        max_thread = std::min(max_thread, (tasks.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK);
        std::vector<std::thread> workers(max_thread-1);
        for (auto& w : workers) w = std::thread(prepare);
        prepare();
        for (auto& w : workers) w.join();
        LOG("done\n");
        int nbad = n_bad;
        if (nbad > 0) {
            LOG_ERR("%s: found %d malformed tasks\n", __func__, nbad);
            return;
        }
    } else {
        int n_dot = std::max((int) n_task/100, 1);
        int i_task = 0;
        for (auto& task : tasks) {
            ++i_task;
            if (!multiple_choice_prepare_one_task(ctx, task, true)) {
                return;
            }
            if (i_task%n_dot == 0) {
                LOG(".");
            }
        }
        LOG("done\n");
    }

    LOG_INF("%s : calculating TruthfulQA score over %zu tasks.\n", __func__, tasks.size());

    LOG("\ntask\tacc_norm\n");

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, max_seq);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());
    std::vector<int> batch_indeces;

    int n_done = 0;
    int n_correct = 0;
    int n_tot_answers = 0;

    for (size_t i0 = 0; i0 < tasks.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        int s0 = 0;
        while (n_cur + (int) tasks[i1].required_tokens <= n_ctx) {
            auto& cur_task = tasks[i1];
            int n_logits = 0;

            int num_answers = cur_task.seq_tokens.size();
            if (s0 + num_answers > max_seq) {
                if (s0 == 0) {
                    LOG_ERR("%s : task %zu requires a higher -np|--parallel value (at least %d)\n", __func__, i0, num_answers);
                    return;
                }
                break;
            }

            if (int(batch_indeces.size()) != num_answers) {
                batch_indeces.resize(num_answers);
            }

            for (int s = 0; s < num_answers; ++s) {
                batch_indeces[s] = s0 + s;
            }

            for (size_t i = 0; i < cur_task.common_prefix; ++i) {
                //llama_batch_add(batch, cur_task.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3}, false);
                common_batch_add(batch, cur_task.seq_tokens[0][i], i, batch_indeces, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                const size_t seq_tokens_size = cur_task.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = cur_task.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, cur_task.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            s0 += num_answers;

            cur_task.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += cur_task.required_tokens;
            if (++i1 == tasks.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %lu tokens)\n", __func__, i0, tasks[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto& cur_task = tasks[i];
            size_t li = 1; // skip the last logit of the common prefix (computed separately below)
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(cur_task.i_logits + li++, cur_task.seq_tokens[s][j + 1]);
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & cur_task = tasks[i];
            //LOG("==== Evaluating <%s> with correct answer ", cur_task.question.c_str());
            //for (int j = 0; j < int(cur_task.mc1.labels.size()); ++j) {
            //    if (cur_task.mc1.labels[j] == 1) {
            //        LOG("%d", j+1);
            //    }
            //}
            //LOG("\n    common_prefix: %zu\n", cur_task.common_prefix);

            // get the logits of the last token of the common prefix
            std::memcpy(tok_logits.data(), batch_logits.data() + cur_task.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            cur_task.log_probs.resize(cur_task.seq_tokens.size());
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                size_t count = 1;
                float  log_prob  = std::log(first_probs[cur_task.seq_tokens[s][cur_task.common_prefix]]);
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    //LOG("        %zu  %g\n", ir, eval_results[ir]);
                    ++count;
                    log_prob += eval_results[ir++];
                }
                cur_task.log_probs[s] = log_prob / count;
                //LOG("        Final: %g\n", log_prob / count);
                //LOG("    <%s> : %g\n", cur_task.mc1.answers[s].c_str(), log_prob/count);
            }

            // Find the ending with maximum logprob
            size_t logprob_max_idx = 0;
            float  logprob_max_val = cur_task.log_probs[0];
            for (size_t s = 1; s < cur_task.log_probs.size(); s++) {
                if (cur_task.log_probs[s] > logprob_max_val) {
                    logprob_max_val = cur_task.log_probs[s];
                    logprob_max_idx = s;
                }
            }

            n_tot_answers += cur_task.log_probs.size();
            if (cur_task.mc1.labels[logprob_max_idx] == 1) {
                ++n_correct;
            }
            ++n_done;

            // Print the accumulated accuracy mean x 100
            LOG("%d\t%.8lf\n", n_done, 100.*n_correct/n_done);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    if (n_done < 100 && (params.multiple_choice_tasks != 0 && params.multiple_choice_tasks < (size_t)n_task)) return;

    float p = 1.f*n_correct/n_done;
    float sigma = sqrt(p*(1-p)/(n_done-1));
    LOG("\n");
    LOG_INF("Final result: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);
    p = 1.f*n_done/n_tot_answers;
    sigma = sqrt(p*(1-p)/(n_done-1));
    LOG_INF("Random chance: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);

    LOG_INF("\n");
}

static void kl_divergence(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.logits_file.empty()) {
        LOG_ERR("%s: you must provide a name of a file containing the log probabilities of the base model\n", __func__);
        return;
    }
    std::ifstream in(params.logits_file.c_str(), std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n", __func__, params.logits_file.c_str());
        return;
    }
    {
        char check[9]; check[8] = 0;
        in.read(check, 8);
        if (in.fail() || strncmp("_logits_", check, 8) != 0) {
            LOG_ERR("%s: %s does not look like a file containing log-probabilities\n", __func__, params.logits_file.c_str());
            return;
        }
    }

    uint32_t n_ctx;
    in.read((char *)&n_ctx, sizeof(n_ctx));
    if (n_ctx > llama_n_ctx(ctx)) {
        LOG_ERR("%s: %s has been computed with %u, while the current context is %d. Increase it with -c and retry\n",
                __func__, params.logits_file.c_str(), n_ctx, params.n_ctx);
    }

    int n_vocab;
    int n_chunk;
    in.read((char *)&n_vocab, sizeof(n_vocab));
    in.read((char *)&n_chunk, sizeof(n_chunk));
    if (in.fail()) {
        LOG_ERR("%s: failed reading n_vocab, n_chunk from %s\n", __func__, params.logits_file.c_str());
        return;
    }
    if (n_vocab != llama_vocab_n_tokens(vocab)) {
        LOG_ERR("%s: inconsistent vocabulary (%d vs %d)\n", __func__, n_vocab, llama_vocab_n_tokens(vocab));
    }

    std::vector<llama_token> tokens(size_t(n_ctx) * n_chunk);
    if (in.read((char *)tokens.data(), tokens.size()*sizeof(tokens[0])).fail()) {
        LOG_ERR("%s: failed reading evaluation tokens from %s\n", __func__, params.logits_file.c_str());
        return;
    }

    const int n_batch = params.n_batch;
    const int num_batches = (n_ctx + n_batch - 1)/n_batch;
    const int nv = 2*((n_vocab + 1)/2) + 4;
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    std::vector<uint16_t> log_probs_uint16(size_t(n_ctx - 1 - n_ctx/2) * nv);
    std::vector<float>    kld_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> p_diff_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    auto mean_and_uncertainty = [] (double sum, double sum2, size_t count) {
        if (count < 1) {
            return std::make_pair(0., 0.);
        }
        double f = sum/count;
        double df = sum2/count - f*f;
        df = df > 0 && count > 10 ? sqrt(df/(count-1)) : 0.;
        return std::make_pair(f, df);
    };
    auto covariance = [] (double suma, double sumb, double sumab, size_t count) {
        if (count < 10) {
            return 0.0;
        }
        double var = sumab/count - (suma/count)*(sumb/count);
        var /= count - 1;
        return var;
    };

    kl_divergence_result kld;
    auto    kld_ptr =    kld_values.data();
    auto p_diff_ptr = p_diff_values.data();

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const auto t_start = std::chrono::high_resolution_clock::now();

        if (in.read((char *)log_probs_uint16.data(), log_probs_uint16.size()*sizeof(uint16_t)).fail()) {
            LOG_ERR("%s: failed reading log-probs for chunk %d\n", __func__, i);
            return;
        }

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_vocab_bos(vocab);
            }

            common_batch_clear(batch);
            for (int i = 0; i < batch_size; i++) {
                common_batch_add(batch, tokens[batch_start + i], j*n_batch + i, {0}, true);
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            if (num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(batch_size) * n_vocab);
            }
        }

        llama_batch_free(batch);

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }
        LOG("\n");
        LOG("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p\n");

        const int first = n_ctx/2;
        const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits(ctx);
        process_logits(n_vocab, all_logits + size_t(first)*n_vocab, tokens.data() + start + first, n_ctx - 1 - first,
                workers, log_probs_uint16, kld, kld_ptr, p_diff_ptr);
        p_diff_ptr += n_ctx - 1 - first;
        kld_ptr    += n_ctx - 1 - first;

        LOG("%4d", i+1);

        auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
        const double ppl_val = exp(log_ppl.first);
        const double ppl_unc = ppl_val * log_ppl.second; // ppl_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl.second ** 2 )
        LOG("    %9.4lf ± %9.4lf", ppl_val, ppl_unc);

        auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
        const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
        const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
        const double log_ppl_ratio_unc = sqrt(log_ppl.second*log_ppl.second + log_ppl_base.second*log_ppl_base.second - 2.0*log_ppl_cov);
        LOG("    %10.5lf ± %10.5lf", log_ppl_ratio_val, log_ppl_ratio_unc);

        auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
        LOG("    %10.5lf ± %10.5lf", kl_div.first, kl_div.second);

        auto p_diff_mse   = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
        const double p_diff_rms_val = sqrt(p_diff_mse.first);
        const double p_diff_rms_unc = 0.5/p_diff_rms_val * p_diff_mse.second;
        LOG("    %6.3lf ± %6.3lf %%", 100.0*p_diff_rms_val, 100.0*p_diff_rms_unc);

        double p_top_val = 1.*kld.n_same_top/kld.count;
        double p_top_unc = sqrt(p_top_val*(1 - p_top_val)/(kld.count - 1));
        LOG("    %6.3lf ± %6.3lf %%", 100.0*p_top_val, 100.0*p_top_unc);

        LOG("\n");

        logits.clear();
    }
    LOG("\n");

    if (kld.count < 100) return; // we do not wish to do statistics on so few values

    std::sort(kld_values.begin(), kld_values.end());
    std::sort(p_diff_values.begin(), p_diff_values.end());

    LOG("====== Perplexity statistics ======\n");

    auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    const double ppl_val = exp(log_ppl.first);
    const double ppl_unc = ppl_val * log_ppl.second; // ppl_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl.second ** 2 )
    LOG("Mean PPL(Q)                   : %10.6lf ± %10.6lf\n", ppl_val, ppl_unc);

    auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double ppl_base_val = exp(log_ppl_base.first);
    const double ppl_base_unc = ppl_base_val * log_ppl_base.second; // ppl_base_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl_base.second ** 2 )
    LOG("Mean PPL(base)                : %10.6lf ± %10.6lf\n", ppl_base_val, ppl_base_unc);

    const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    // LOG("Cov(ln(PPL(Q)), ln(PPL(base))): %10.6lf\n", log_ppl_cov);
    const double log_ppl_cor = log_ppl_cov / (log_ppl.second*log_ppl_base.second);
    LOG("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100.0*log_ppl_cor);

    const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
    const double log_ppl_ratio_unc = sqrt(log_ppl.second*log_ppl.second + log_ppl_base.second*log_ppl_base.second - 2.0*log_ppl_cov);
    LOG("Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf\n", log_ppl_ratio_val, log_ppl_ratio_unc);

    const double ppl_ratio_val = exp(log_ppl_ratio_val);
    const double ppl_ratio_unc = ppl_ratio_val * log_ppl_ratio_unc; // ppl_ratio_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl_ratio.second ** 2 )
    LOG("Mean PPL(Q)/PPL(base)         : %10.6lf ± %10.6lf\n", ppl_ratio_val, ppl_ratio_unc);

    const double ppl_cov = ppl_val * ppl_base_val * log_ppl_cov;
    const double ppl_diff_val = ppl_val - ppl_base_val;
    const double ppl_diff_unc = sqrt(ppl_unc*ppl_unc + ppl_base_unc*ppl_base_unc - 2.0*ppl_cov);
    LOG("Mean PPL(Q)-PPL(base)         : %10.6lf ± %10.6lf\n", ppl_diff_val, ppl_diff_unc);

    LOG("\n");

    LOG("====== KL divergence statistics ======\n");
    auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    LOG("Mean    KLD: %10.6lf ± %10.6lf\n", kl_div.first, kl_div.second);
    auto kld_median = kld_values.size()%2 == 0 ? 0.5f*(kld_values[kld_values.size()/2] + kld_values[kld_values.size()/2-1])
                                               : kld_values[kld_values.size()/2];

    auto percentile = [] (std::vector<float> values, float fraction) {
        if (fraction <= 0) return values.front();
        if (fraction >= 1) return values.back();
        float p = fraction*(values.size() - 1);
        size_t ip = size_t(p); p -= ip;
        return (1 - p)*values[ip] + p*values[std::min(ip+1, values.size()-1)];
    };

    LOG("Maximum KLD: %10.6f\n", kld_values.back());
    LOG("99.9%%   KLD: %10.6f\n", percentile(kld_values, 0.999f));
    LOG("99.0%%   KLD: %10.6f\n", percentile(kld_values, 0.990f));
    LOG("90.0%%   KLD: %10.6f\n", percentile(kld_values, 0.900f));
    LOG("Median  KLD: %10.6f\n", kld_median);
    LOG("10.0%%   KLD: %10.6f\n", percentile(kld_values, 0.100f));
    LOG(" 5.0%%   KLD: %10.6f\n", percentile(kld_values, 0.050f));
    LOG(" 1.0%%   KLD: %10.6f\n", percentile(kld_values, 0.010f));
    LOG("Minimum KLD: %10.6f\n", kld_values.front());

    LOG("\n");

    LOG("====== Token probability statistics ======\n");

    auto p_diff = mean_and_uncertainty(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    LOG("Mean    Δp: %6.3lf ± %5.3lf %%\n",  100.0*p_diff.first, 100.0*p_diff.second);

    auto p_diff_median = p_diff_values.size()%2 == 0 ? 0.5f*(p_diff_values[p_diff_values.size()/2] + p_diff_values[p_diff_values.size()/2-1])
                                               : p_diff_values[p_diff_values.size()/2];

    LOG("Maximum Δp: %6.3lf%%\n",  100.0*p_diff_values.back());
    LOG("99.9%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.999f));
    LOG("99.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.990f));
    LOG("95.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.950f));
    LOG("90.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.900f));
    LOG("75.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.750f));
    LOG("Median  Δp: %6.3lf%%\n",  100.0*p_diff_median);
    LOG("25.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.250f));
    LOG("10.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.100f));
    LOG(" 5.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.050f));
    LOG(" 1.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.010f));
    LOG(" 0.1%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.001f));
    LOG("Minimum Δp: %6.3lf%%\n",  100.0*p_diff_values.front());

    auto p_diff_mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    // LOG("MSE Δp    : %10.6lf ± %10.6lf\n", p_diff_mse.first, p_diff_mse.second);

    const double p_diff_rms_val = sqrt(p_diff_mse.first);
    const double p_diff_rms_unc = 0.5/p_diff_rms_val * p_diff_mse.second;
    LOG("RMS Δp    : %6.3lf ± %5.3lf %%\n", 100.0*p_diff_rms_val, 100.0*p_diff_rms_unc);

    const double same_top_p = 1.0*kld.n_same_top/kld.count;
    LOG("Same top p: %6.3lf ± %5.3lf %%\n", 100.0*same_top_p, 100.0*sqrt(same_top_p*(1.0 - same_top_p)/(kld.count - 1)));
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 512;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init();

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: perplexity tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    const bool ppl = !params.hellaswag && !params.winogrande && !params.multiple_choice && !params.kl_divergence;

    if (ppl) {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    } else {
        params.n_batch = std::min(params.n_batch, params.n_ctx);
        if (params.kl_divergence) {
            params.n_parallel = 1;
        } else {
            // ensure there's at least enough seq_ids for HellaSwag
            params.n_parallel = std::max(4, params.n_parallel);
        }
    }

    if (params.ppl_stride > 0) {
        LOG_INF("Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);

    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    struct results_perplexity results;
    if (params.hellaswag) {
        hellaswag_score(ctx, params);
    } else if (params.winogrande) {
        winogrande_score(ctx, params);
    } else if (params.multiple_choice) {
        multiple_choice_score(ctx, params);
    } else if (params.kl_divergence) {
        kl_divergence(ctx, params);
    } else {
        results = perplexity(ctx, params, n_ctx);
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
