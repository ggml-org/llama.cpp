#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct ngram_data {
    bool active = false;

    llama_seq_id seq_id = -1;

    std::vector<int> i_batch;

    std::vector<llama_token> tokens;
};

// n-gram container
struct ngram_container {
    ngram_container(int n_vocab, int N, int G) {
        cnt.resize(n_vocab);
        head.resize(n_vocab);
        tokens.resize(n_vocab * G * (N - 1));
    }

    int n_total = 0;

    std::vector<int> cnt;
    std::vector<int> head;

    // [n_vocab][G][N - 1]
    // for each token of the vocab, keep a ring-buffer of capacity G of n-grams of size N - 1
    std::vector<llama_token> tokens;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    const int W = 15; // lookahead window
    const int N = 5;  // n-gram size
    const int G = 15; // max verification n-grams

    const bool dump_kv_cache = params.dump_kv_cache;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("lookahead", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    // Tokenize the prompt
    std::vector<llama_token> inp;
    std::vector<llama_token> all;

    inp = ::llama_tokenize(ctx, params.prompt, true, true);
    all = inp;

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt
    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

    for (int s = 1; s < W + G + 1; ++s) {
        llama_kv_cache_seq_cp(ctx, 0, s, -1, -1);
    }

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    llama_token id = 0;

    // used to determine end of generation
    bool has_eos = false;

    // for each decoded batch, we have at most W + G + 1 distinct sequences:
    // seq_id == 0           : the current input token
    // seq_id [1, W]         : tokens from the past N - 1 Jacobi iterations
    // seq_id [W + 1, W + G] : verification n-grams
    llama_batch batch = llama_batch_init(params.n_ctx, 0, W + G + 1);

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // verification n-grams
    std::vector<ngram_data> ngrams_cur(G);

    // tokens for the past N - 1 Jacobi iterations
    std::vector<llama_token> tokens_j_prev(W);
    std::vector<std::vector<llama_token>> tokens_j(N - 1);
    for (int j = 0; j < N - 1; j++) {
        tokens_j[j].resize(W);

        for (int i = 0; i < W; i++) {
            // there are different ways to init these tokens
            if (0) {
                // initialize randomly from the prompt tokens
                tokens_j[j][i] = all[1 + rand() % (all.size() - 1)];
            } else {
                // initialize with a sequence of increasing numbers
                tokens_j[j][i] = 100 + i;
            }
        }
    }

    std::vector<llama_seq_id> seq_id_look;

    // the input token belongs both to all sequences
    std::vector<llama_seq_id> seq_id_all(W + G + 1);
    for (int i = 0; i < W + G + 1; i++) {
        seq_id_all[i] = i;
    }

    // here we keep adding new n-grams as we go
    ngram_container ngrams_observed(llama_n_vocab(model), N, G);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, W + G + 1);

    const auto t_dec_start = ggml_time_us();

    // sample first token
    {
        id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);

        llama_sampling_accept(ctx_sampling, ctx, id, true);

        {
            const std::string token_str = llama_token_to_piece(ctx, id);

            printf("%s", token_str.c_str());
            fflush(stdout);
        }
    }

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            llama_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        // build the mask from https://lmsys.org/blog/2023-11-21-lookahead-decoding/
        //
        // Example for W = 5, N = 4, G = 2:
        // (I = input, L = lookahead, V = verification)
        //
        // Batch:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
        // T:        -2 -2 -2 -2 -1 -1 -1 -1 -1  0  0  0  0  0  0
        // Info:   I  L  L  L  L  L  L  L  L  L  L  L  L  L  L  V  V  V  V  V  V
        // Pos:    0  1  2  3  4  1  2  3  4  5  2  3  4  5  6  1  2  3  1  2  3   (+ n_past)
        // Logits: 1  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1
        // ---------------------------------------------------------------------
        // Seq:    0
        //         1              1              1
        //         2  2              2              2
        //         3  3  3              3              3
        //         4  4  4  4              4              4
        //         5  5  5  5  5              5              5
        //         6                                            6  6  6
        //         7                                                     7  7  7
        // ---------------------------------------------------------------------
        //                                       |  |  |  |  |  |  |  |  |  |  |
        //                                       V  V  V  V  V  |  |  |  |  |  |
        //                                         j_tokens     |  |  |  |  |  |
        //                                                      V  V  V  V  V  V
        //                                                             id
        {
            llama_batch_clear(batch);

            // current token - first token of the first level
            llama_batch_add(batch, id, n_past, seq_id_all, true);

            // verification n-grams - queue this before the lookahead tokens for less KV cache fragmentation
            {
                const int g_cur = ngrams_observed.cnt[id];

                ngrams_cur.resize(g_cur);
                for (int g = 0; g < g_cur; g++) {
                    ngrams_cur[g].active = true;
                    ngrams_cur[g].tokens.resize(N);
                    ngrams_cur[g].i_batch.resize(N);
                    ngrams_cur[g].seq_id = W + 1 + g;
                    ngrams_cur[g].i_batch[0] = 0;
                    ngrams_cur[g].tokens [0] = id;
                }

                for (int j = 0; j < N - 1; j++) {
                    for (int g = 0; g < g_cur; g++) {
                        const int idx = id*(N - 1)*G + g*(N - 1);

                        const llama_token t = ngrams_observed.tokens[idx + j];

                        ngrams_cur[g].tokens [j + 1] = t;
                        ngrams_cur[g].i_batch[j + 1] = batch.n_tokens;

                        llama_batch_add(batch, t, n_past + j + 1, { W + 1 + g }, true);
                    }
                }
            }

            // fill the remaining W - 1 tokens for the first level
            for (int i = 1; i < W; i++) {
                seq_id_look.resize(W - i);
                for (int j = 0; j < W - i; j++) {
                    seq_id_look[j] = i + j + 1;
                }

                llama_batch_add(batch, tokens_j[0][i], n_past + i, seq_id_look, false);
            }

            // fill the rest of the levels
            for (int j = 1; j < N - 1; j++) {
                for (int i = 0; i < W; i++) {
                    llama_batch_add(batch, tokens_j[j][i], n_past + j + i, { i + 1 }, j == N - 2);
                }
            }
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\n\n%s: error: llama_decode failed - increase KV cache size\n", __func__);
            return 1;
        }

        int seq_id_best = 0;

        for (int v = 0; v < N; ++v) {
            int i_batch = 0;

            // if no active ngrams are left, it means the sampled token does not pass the verification
            if (v > 0) {
                for (int g = 0; g < (int) ngrams_cur.size(); g++) {
                    if (ngrams_cur[g].active) {
                        i_batch = ngrams_cur[g].i_batch[v];
                        seq_id_best = ngrams_cur[g].seq_id;

                        ++n_accept;
                        break;
                    }
                }

                // no more matches -> create a new batch
                if (i_batch == 0) {
                    break;
                }
            }

            // sample the next token
            id = llama_sampling_sample(ctx_sampling, ctx, NULL, i_batch);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            // print
            {
                const std::string token_str = llama_token_to_piece(ctx, id);

                if (v == 0) {
                    printf("%s", token_str.c_str());
                } else {
                    // print light cyan
                    printf("\033[0;96m%s\033[0m", token_str.c_str());
                }
                fflush(stdout);

                if (llama_token_is_eog(model, id)) {
                    has_eos = true;
                }

                all.push_back(id);
            }

            ++n_predict;
            ++n_past;

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }

            // verify across active n-grams
            for (int g = 0; g < (int) ngrams_cur.size(); g++) {
                if (ngrams_cur[g].active) {
                    if (v == N - 1) {
                        ngrams_cur[g].active = false;
                    } else {
                        if (id != ngrams_cur[g].tokens[v + 1]) {
                            ngrams_cur[g].active = false;
                        }
                    }
                }
            }

            // print known n-grams starting with token id (debug)
            if (0 && v == 0) {
                if (ngrams_observed.cnt[id] > 0) {
                    printf("\n - %d n-grams starting with '%s'\n", ngrams_observed.cnt[id], llama_token_to_piece(ctx, id).c_str());
                }

                for (int i = 0; i < ngrams_observed.cnt[id]; i++) {
                    printf("   - ngram %2d: ", i);

                    const int idx = id*(N - 1)*G + i*(N - 1);

                    for (int j = 0; j < N - 1; j++) {
                        const std::string token_str = llama_token_to_piece(ctx, ngrams_observed.tokens[idx + j]);

                        printf("%s", token_str.c_str());
                    }

                    printf("\n");
                }
            }

            // update lookahead tokens
            {
                for (int i = 0; i < W; i++) {
                    tokens_j_prev[i] = tokens_j[0][i];
                }

                for (int j = 0; j < N - 2; j++) {
                    tokens_j[j] = tokens_j[j + 1];
                }

                if (v == 0) {
                    // sample from the last level
                    for (int i = 0; i < W; i++) {
                        tokens_j[N - 2][i] = llama_sampling_sample(ctx_sampling, ctx, NULL, ngrams_cur.size()*(N-1) + W*(N - 2) + i);
                    }
                } else {
                    for (int i = 0; i < W; i++) {
                        // there are different ways to init these tokens
                        if (0) {
                            // random init
                            tokens_j[N - 2][i] = all[1 + rand() % (all.size() - 1)];
                        } else {
                            // init from the previous level
                            tokens_j[N - 2][i] = tokens_j[0][i];
                        }
                    }
                }
            }

            // update observed ngrams
            if (v == 0) {
                // the first token of the n-gram is determined by the index in the container so it is not stored
                std::vector<llama_token> ngram(N - 1);

                // n-gram generation
                // ref: https://github.com/hao-ai-lab/LookaheadDecoding/issues/14#issuecomment-1826198518
                for (int f = 0; f < W; ++f) {
                    const int ft = tokens_j_prev[f]; // first token of the n-gram

                    for (int j = 0; j < N - 1; ++j) {
                        ngram[j] = tokens_j[j][f];
                    }

                    // filter-out repeating n-grams
                    {
                        bool is_unique = true;

                        for (int k = 0; k < ngrams_observed.cnt[ft]; ++k) {
                            const int idx = ft*(N - 1)*G + k*(N - 1);

                            bool is_match = true;
                            for (int j = 0; j < N - 1; ++j) {
                                if (ngrams_observed.tokens[idx + j] != ngram[j]) {
                                    is_match = false;
                                    break;
                                }
                            }

                            if (is_match) {
                                is_unique = false;
                                break;
                            }
                        }

                        if (!is_unique) {
                            continue;
                        }
                    }

                    const int head = ngrams_observed.head[ft];
                    const int idx  = ft*(N - 1)*G + head*(N - 1);

                    for (int i = 0; i < N - 1; i++) {
                        ngrams_observed.tokens[idx + i] = ngram[i];
                    }

                    ngrams_observed.cnt[ft]  = std::min(G, ngrams_observed.cnt[ft] + 1);
                    ngrams_observed.head[ft] = (head + 1) % G;

                    ngrams_observed.n_total++;
                }
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // if no verification token matched, we simply remove all cells from this batch -> no fragmentation
        llama_kv_cache_seq_rm(ctx, -1, n_past, -1);

        if (seq_id_best != 0) {
            // if a verification token matched, we keep the best sequence and remove the rest
            // this leads to some KV cache fragmentation
            llama_kv_cache_seq_keep(ctx, seq_id_best);
            llama_kv_cache_seq_cp  (ctx, seq_id_best, 0, -1, -1);
            llama_kv_cache_seq_rm  (ctx, seq_id_best,    -1, -1);

            for (int s = 1; s < W + G + 1; ++s) {
                llama_kv_cache_seq_cp(ctx, 0, s, -1, -1);
            }
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("W = %2d\n", W);
    LOG_TEE("N = %2d\n", N);
    LOG_TEE("G = %2d\n", G);
    LOG_TEE("\n");
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_accept  = %d\n", n_accept);

    llama_print_timings(ctx);

    llama_kv_cache_view_free(&kvc_view);
    llama_sampling_free(ctx_sampling);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
