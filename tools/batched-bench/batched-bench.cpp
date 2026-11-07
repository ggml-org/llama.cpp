#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#define KVU_CACHE_N_IDLE_SEQ 4
#define KVU_CACHE_SCATTER_CHUNK_MAX 2048

static bool invalid_shape(const std::vector<int> & values) {
    return values.empty() || *std::min_element(values.begin(), values.end()) <= 0;
}

// 64-bit to prevent 32-bit arithmetic overflow
static int64_t required_kv(bool is_pp_shared, bool kv_unified, int64_t pp, int64_t tg, int64_t pl) {
    if (is_pp_shared && kv_unified) {
        return pp + pl*tg;
    }
    return pl*(pp + tg);
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG("\n");
}

// satisfies -Wmissing-declarations
int llama_batched_bench(int argc, char ** argv);

int llama_batched_bench(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    int is_pp_shared   = params.is_pp_shared;
    int is_tg_separate = params.is_tg_separate;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;

    if (invalid_shape(n_pp) || invalid_shape(n_tg) || invalid_shape(n_pl)) {
        LOG_ERR("benchmark needs positive PP, TG, and parallel prompt counts\n");
        return 1;
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model_ptr model_owner(llama_model_load_from_file(params.model.path.c_str(), model_params));
    llama_model * model = model_owner.get();

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    // ensure enough sequences are available
    const int max_pl = *std::max_element(n_pl.begin(), n_pl.end());
    if (params.kv_unified && size_t(max_pl) + KVU_CACHE_N_IDLE_SEQ > llama_max_parallel_sequences()) {
        LOG_ERR("unified KV warming supports at most %zu timed sequences\n", llama_max_parallel_sequences() - KVU_CACHE_N_IDLE_SEQ);
        return 1;
    }
    ctx_params.n_seq_max = max_pl + (params.kv_unified ? KVU_CACHE_N_IDLE_SEQ : 0);

    llama_context_ptr ctx_owner(llama_init_from_model(model, ctx_params));
    llama_context * ctx = ctx_owner.get();

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    const auto get_token_rand = [n_vocab]() -> llama_token {
        return std::rand() % n_vocab;
    };

    auto * mem = llama_get_memory(ctx);

    const int32_t n_kv_max = llama_n_ctx(ctx);

    common_batch batch(ctx);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, common_batch & batch, int32_t n_batch, bool synchronize) {
        for (int32_t i = 0; i < batch.size(); i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - i);

            const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(i, n_tokens));
            if (ret != 0) {
                LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            batch.add(get_token_rand(), i, 0, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    if (!params.batched_bench_output_jsonl) {
        LOG("\n");
        LOG("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, is_tg_separate, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG("\n");
        LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s");
        LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");
    }

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int64_t n_kv = required_kv(is_pp_shared, params.kv_unified, pp, tg, pl);

                if (n_kv > n_kv_max) {
                    continue;
                }

                const int seq_offset = params.kv_unified ? KVU_CACHE_N_IDLE_SEQ : 0;
                llama_memory_clear(mem, false);
                if (params.kv_unified) {
                    const int n_idle = n_kv_max - int(n_kv);
                    const int chunk_size = std::max(1, std::min(KVU_CACHE_SCATTER_CHUNK_MAX, n_idle/16));
                    int positions[KVU_CACHE_N_IDLE_SEQ] = {};
                    std::mt19937 idle_rng(1);
                    std::uniform_int_distribution<int32_t> idle_token(0, n_vocab - 1);

                    LOG_INF("warming unified KV cache: %d idle tokens\n", n_idle);
                    for (int filled = 0; filled < n_idle;) {
                        for (int seq = 0; seq < KVU_CACHE_N_IDLE_SEQ && filled < n_idle; ++seq) {
                            const int count = std::min(chunk_size, n_idle - filled);
                            batch.clear();
                            for (int i = 0; i < count; ++i) {
                                batch.add(idle_token(idle_rng), positions[seq]++, seq, false);
                            }
                            if (!decode_helper(ctx, batch, ctx_params.n_batch, false)) {
                                LOG_ERR("failed to fill scattered idle KV cache\n");
                                return 1;
                            }
                            filled += count;
                        }
                    }
                    llama_synchronize(ctx);
                    for (int seq = 0; seq < KVU_CACHE_N_IDLE_SEQ; ++seq) {
                        if (llama_memory_seq_pos_max(mem, seq) != positions[seq] - 1) {
                            LOG_ERR("scattered idle sequence %d has an unexpected end position\n", seq);
                            return 1;
                        }
                    }
                }

                batch.clear();
                for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                    for (int i = 0; i < pp; ++i) {
                        batch.add(get_token_rand(), i, seq_offset + j, i == pp - 1);
                    }
                }

                const auto t_pp_start = ggml_time_us();

                if (!decode_helper(ctx, batch, ctx_params.n_batch, false)) {
                    LOG_ERR("%s: llama_decode() failed\n", __func__);
                    return 1;
                }

                llama_synchronize(ctx);

                const auto t_pp_end = ggml_time_us();

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_memory_seq_cp(mem, seq_offset, seq_offset + i, -1, -1);
                    }

                    if (!params.kv_unified) {
                        // run one dummy token to apply the memory copy
                        batch.clear();
                        batch.add(get_token_rand(), pp + 0, seq_offset, true);
                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                        llama_memory_seq_rm(mem, seq_offset, pp, -1);
                    }
                }

                const auto t_tg_start = ggml_time_us();

                if (is_tg_separate) {
                    // decode pattern:
                    // 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    for (int j = 0; j < pl; ++j) {
                        for (int i = 0; i < tg; ++i) {
                            batch.clear();

                            batch.add(get_token_rand(), pp + i, seq_offset + j, true);

                            if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                                LOG_ERR("%s: llama_decode() failed\n", __func__);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern:
                    // 0123 0123 0123 ...
                    for (int i = 0; i < tg; ++i) {
                        batch.clear();

                        for (int j = 0; j < pl; ++j) {
                            batch.add(get_token_rand(), pp + i, seq_offset + j, true);
                        }

                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;

                if(params.batched_bench_output_jsonl) {
                    LOG(
                        "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                        "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %" PRId64 ", \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}\n",
                        n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                        pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed
                    );
                } else {
                    LOG("|%6d | %6d | %4d | %6" PRId64 " | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                }
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);


    batch.batch.reset();
    ctx_owner.reset();
    model_owner.reset();
    llama_backend_free();

    return 0;
}
