#include "sampling.h"
#include "log.h"

#ifdef LLAMA_USE_LLGUIDANCE

#    include "llguidance.h"
#    include "../src/llama-ext.h"
#    include "ggml-backend.h"
#    include "ggml-cpp.h"
#    include <array>
#    include <cmath>
#    include <memory>
#    include <stdexcept>

struct llama_sampler_llg {
    const llama_vocab * vocab;
    std::string         grammar_kind;
    std::string         grammar_data;
    LlgTokenizer *      tokenizer;
    LlgMatcher *        grammar;

    struct mask_binding {
        ggml_tensor * payload = nullptr;
        ggml_tensor * desc = nullptr;
    };
    struct prepared_row {
        std::vector<uint32_t> words;
        bool usable = false;
    };
    std::vector<mask_binding> bindings = {};
    std::vector<prepared_row> rows = {};
};

static LlgMatcher * llama_sampler_llg_new(LlgTokenizer * tokenizer, const char * grammar_kind,
                                          const char * grammar_data) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    const char * log_level = getenv("LLGUIDANCE_LOG_LEVEL");
    if (log_level && *log_level) {
        cinit.log_stderr_level = atoi(log_level);
    }
    auto c = llg_new_matcher(&cinit, grammar_kind, grammar_data);
    if (llg_matcher_get_error(c)) {
        LOG_ERR("llg error: %s\n", llg_matcher_get_error(c));
        llg_free_matcher(c);
        return nullptr;
    }

    return c;
}

static const char * llama_sampler_llg_name(const llama_sampler * /*smpl*/) {
    return "llguidance";
}

static std::runtime_error llama_sampler_llg_error(LlgMatcher * grammar) {
    const char * error = llg_matcher_get_error(grammar);
    return std::runtime_error(std::string("llguidance: ") + (error ? error : "mask computation failed"));
}

static const uint32_t * llama_sampler_llg_mask(LlgMatcher * grammar, int32_t n_vocab) {
    if (llg_matcher_get_error(grammar)) {
        throw llama_sampler_llg_error(grammar);
    }
    const uint32_t * mask = llg_matcher_get_mask(grammar);
    if (!mask && llg_matcher_compute_mask(grammar) == 0) {
        mask = llg_matcher_get_mask(grammar);
    }
    if (!mask) {
        throw llama_sampler_llg_error(grammar);
    }
    const int64_t n_words = (int64_t(n_vocab) + 31)/32;
    for (int64_t i = 0; i < n_words; ++i) {
        uint32_t word = mask[i];
        if (i == n_words - 1 && n_vocab % 32) {
            word &= (UINT32_C(1) << (n_vocab % 32)) - 1;
        }
        if (word) {
            return mask;
        }
    }
    throw std::runtime_error("llguidance: empty allowed token set");
}

static void llama_sampler_llg_accept_impl(llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    ctx->rows.clear();
    if (ctx->grammar && llg_matcher_consume_token(ctx->grammar, token) != 0) {
        throw llama_sampler_llg_error(ctx->grammar);
    }
}

static void llama_sampler_llg_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        const int32_t n_vocab = llama_vocab_n_tokens(ctx->vocab);
        const uint32_t * mask = llama_sampler_llg_mask(ctx->grammar, n_vocab);
        for (size_t i = 0; i < cur_p->size; ++i) {
            const auto token = cur_p->data[i].id;
            if (token < 0 || token >= n_vocab || (mask[token/32] & (UINT32_C(1) << (token % 32))) == 0) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

static void llama_sampler_llg_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    ctx->rows.clear();
    if (ctx->grammar && llg_matcher_reset(ctx->grammar) != 0) {
        throw llama_sampler_llg_error(ctx->grammar);
    }
}

static llama_sampler * llama_sampler_llg_clone(const llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;

    auto * result = llama_sampler_init_llg(ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (llama_sampler_llg *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_kind = ctx->grammar_kind;
            result_ctx->grammar_data = ctx->grammar_data;
            result_ctx->grammar      = llg_clone_matcher(ctx->grammar);
            result_ctx->tokenizer    = llg_clone_tokenizer(ctx->tokenizer);
        }
    }

    return result;
}

static void llama_sampler_llg_free(llama_sampler * smpl) {
    const auto * ctx = (llama_sampler_llg *) smpl->ctx;

    if (ctx->grammar) {
        llg_free_matcher(ctx->grammar);
    }
    if (ctx->tokenizer) {
        llg_free_tokenizer(ctx->tokenizer);
    }

    delete ctx;
}

static void llama_sampler_llg_backend_apply(
        llama_sampler * smpl, ggml_context * ctx, ggml_cgraph * gf, llama_sampler_data * data) {
    GGML_UNUSED(gf);
    auto * sctx = (llama_sampler_llg *) smpl->ctx;
    if (!sctx->grammar) {
        return;
    }
    GGML_ASSERT(data->sampled == nullptr && "grammar must precede token selection");
    GGML_ASSERT(data->logits->ne[1] == 1 && ggml_is_matrix(data->logits));
    const size_t row = sctx->bindings.size();
    sctx->bindings.push_back({});
    if (row < sctx->rows.size() && !sctx->rows[row].usable) {
        data->skip = true;
        return;
    }
    auto & binding = sctx->bindings.back();
    binding.payload = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t(llama_vocab_n_tokens(sctx->vocab)) + 31)/32);
    ggml_set_input(binding.payload);
    ggml_set_name(binding.payload, "llguidance_mask_payload");
    binding.desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(binding.desc);
    ggml_set_name(binding.desc, "llguidance_mask_desc");
    data->logits = ggml_token_mask(ctx, data->logits, data->candidates, binding.payload, binding.desc);
    data->probs = nullptr;
}

static bool llama_sampler_llg_backend_init(llama_sampler * smpl, ggml_backend_buffer_type_t buft, uint32_t n_outputs) {
    GGML_UNUSED(n_outputs);
    auto * sctx = (llama_sampler_llg *) smpl->ctx;
    sctx->rows.clear();
    if (!sctx->grammar) {
        return true;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(sctx->vocab);
    if (n_vocab <= 0) {
        return false;
    }
    ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    if (!ctx) {
        return false;
    }
    auto * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_vocab);
    auto * payload = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t(n_vocab) + 31)/32);
    auto * desc = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    auto * device = ggml_backend_buft_get_device(buft);
    return !device || (ggml_backend_dev_supports_op(device, ggml_token_mask(ctx.get(), logits, nullptr, payload, desc)) &&
                      ggml_backend_dev_supports_op(device, ggml_token_mask(ctx.get(), logits, ids, payload, desc)));
}

static bool llama_sampler_llg_backend_prepare(llama_sampler * smpl, const llama_sampler_row * rows, size_t n_rows) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    ctx->rows.clear();
    if (!ctx->grammar) {
        return true;
    }
    if (n_rows && !rows) {
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(ctx->vocab);
    for (size_t i = 0; i < n_rows; ++i) {
        const auto & row = rows[i];
        if (row.type < LLAMA_SAMPLER_PREFIX_COMMITTED || row.type > LLAMA_SAMPLER_PREFIX_TREE ||
                (row.type == LLAMA_SAMPLER_PREFIX_COMMITTED && row.n_prefix != 0) ||
                (row.n_prefix && !row.prefix)) {
            ctx->rows.clear();
            return false;
        }
        std::unique_ptr<LlgMatcher, decltype(&llg_free_matcher)> state(llg_clone_matcher(ctx->grammar), llg_free_matcher);
        std::vector<uint32_t> words;
        bool usable = true;
        try {
            for (size_t j = 0; j < row.n_prefix; ++j) {
                const auto token = row.prefix[j];
                const auto * mask = llama_sampler_llg_mask(state.get(), n_vocab);
                if (token < 0 || token >= n_vocab || llama_vocab_is_eog(ctx->vocab, token) ||
                        !(mask[token/32] & (UINT32_C(1) << (token % 32)))) {
                    usable = false;
                    break;
                }
                if (llg_matcher_consume_token(state.get(), token) != 0) {
                    throw llama_sampler_llg_error(state.get());
                }
            }
            if (usable) {
                const auto * mask = llama_sampler_llg_mask(state.get(), n_vocab);
                words.assign(mask, mask + (int64_t(n_vocab) + 31)/32);
            }
        } catch (const std::runtime_error & e) {
            LOG_DBG("%s: unusable prefix: %s\n", __func__, e.what());
            usable = false;
        }
        ctx->rows.push_back({std::move(words), usable});
    }
    return true;
}

static bool llama_sampler_llg_backend_row_usable(const llama_sampler * smpl, size_t row) {
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;
    return row >= ctx->rows.size() || ctx->rows[row].usable;
}

static void llama_sampler_llg_backend_set_input(llama_sampler * smpl, ggml_backend_sched_t ) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (!ctx->grammar) {
        return;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(ctx->vocab);
    const int32_t n_words = (int64_t(n_vocab) + 31)/32;
    const std::array<int32_t, 4> desc = {GGML_TOKEN_MASK_BITSET, n_words, 0, n_vocab};
    const uint32_t * committed_mask = nullptr;
    for (size_t row = 0; row < ctx->bindings.size(); ++row) {
        const auto & binding = ctx->bindings[row];
        if (!binding.payload) {
            continue;
        }
        if (row >= ctx->rows.size() && !committed_mask) {
            committed_mask = llama_sampler_llg_mask(ctx->grammar, n_vocab);
        }
        const auto * mask = row < ctx->rows.size() ? ctx->rows[row].words.data() : committed_mask;
        ggml_backend_tensor_set(binding.payload, mask, 0, size_t(n_words)*sizeof(uint32_t));
        ggml_backend_tensor_set(binding.desc, desc.data(), 0, sizeof(desc));
    }
}

static void llama_sampler_llg_backend_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    ctx->bindings.clear();
}

static void llama_sampler_llg_copy_state(const llama_sampler * src, llama_sampler * dst) {
    const auto * sctx = (const llama_sampler_llg *) src->ctx;
    auto * dctx = (llama_sampler_llg *) dst->ctx;
    GGML_ASSERT(sctx->vocab == dctx->vocab && sctx->grammar_kind == dctx->grammar_kind && sctx->grammar_data == dctx->grammar_data);
    auto * grammar = sctx->grammar ? llg_clone_matcher(sctx->grammar) : nullptr;
    if (dctx->grammar) {
        llg_free_matcher(dctx->grammar);
    }
    dctx->grammar = grammar;
    dctx->rows.clear();
}

static llama_sampler_i llama_sampler_llg_i = {
    /* .name              = */ llama_sampler_llg_name,
    /* .accept            = */ llama_sampler_llg_accept_impl,
    /* .apply             = */ llama_sampler_llg_apply,
    /* .reset             = */ llama_sampler_llg_reset,
    /* .clone             = */ llama_sampler_llg_clone,
    /* .free              = */ llama_sampler_llg_free,
    /* .backend_init      = */ llama_sampler_llg_backend_init,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ llama_sampler_llg_backend_apply,
    /* .backend_set_input = */ llama_sampler_llg_backend_set_input,
    /* .backend_reset     = */ llama_sampler_llg_backend_reset,
    /* .copy_state        = */ llama_sampler_llg_copy_state,
    /* .backend_prepare   = */ llama_sampler_llg_backend_prepare,
    /* .backend_row_usable = */ llama_sampler_llg_backend_row_usable,
};

static size_t llama_sampler_llg_tokenize_fn(const void * user_data, const uint8_t * bytes, size_t bytes_len,
                                            uint32_t * output_tokens, size_t output_tokens_len) {
    const llama_vocab * vocab = (const llama_vocab *) user_data;
    int                 r     = 0;
    try {
        r = llama_tokenize(vocab, (const char *) bytes, bytes_len, (int32_t *) output_tokens, output_tokens_len, false,
                           true);
    } catch (const std::exception & e) {
        GGML_ABORT("llama_tokenize failed: %s\n", e.what());
    }
    if (r < 0) {
        return -r;
    }
    return r;
}

static LlgTokenizer * llama_sampler_llg_new_tokenizer(const llama_vocab * vocab) {
    // TODO store the tokenizer in the vocab somehow
    static const llama_vocab * vocab_cache;
    static LlgTokenizer *      tokenizer_cache;

    if (vocab_cache == vocab) {
        return llg_clone_tokenizer(tokenizer_cache);
    }

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL) {
        tok_eos = llama_vocab_eos(vocab);
    }

    size_t vocab_size = llama_vocab_n_tokens(vocab);

    auto token_lens       = new uint32_t[vocab_size];
    // we typically have ~7 bytes per token; let's go on the safe side here
    auto token_bytes_size = vocab_size * 16 + 1024 * 1024;
    auto token_bytes      = new uint8_t[token_bytes_size];

    size_t offset = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        size_t max_token = 1024;
        if (token_bytes_size - offset < max_token) {
            GGML_ABORT("token_bytes buffer too small\n");
        }

        llama_token token = i;
        auto        dp    = (char *) token_bytes + offset;
        auto        size  = llama_detokenize(vocab, &token, 1, dp, max_token, false, false);
        if (size < 0) {
            GGML_ABORT("llama_detokenize failed\n");
        }
        if (size == 0) {
            size = llama_detokenize(vocab, &token, 1, dp + 1, max_token - 1, false, true);
            if (size < 0) {
                GGML_ABORT("llama_detokenize failed\n");
            }
            if (size != 0) {
                *dp = '\xff';  // special token prefix marker
                size += 1;
            }
        }

        token_lens[i] = size;
        offset += size;
    }

    LlgTokenizerInit tinit = {
        /* .vocab_size                         = */ (uint32_t) vocab_size,
        /* .tok_eos                            = */ (uint32_t) tok_eos,
        /* .token_lens                         = */ token_lens,
        /* .token_bytes                        = */ token_bytes,
        /* .tokenizer_json                     = */ nullptr,
        /* .tokenize_assumes_string            = */ true,
        /* .tokenize_fn                        = */ llama_sampler_llg_tokenize_fn,
        /* .use_approximate_greedy_tokenize_fn = */ false,
        /* .tokenize_user_data                 = */ vocab,
        /* .slices                             = */ nullptr,
    };

    char           error_buffer[1024];
    LlgTokenizer * tokenizer = llg_new_tokenizer(&tinit, error_buffer, sizeof(error_buffer));

    delete[] token_bytes;
    delete[] token_lens;

    if (tokenizer == nullptr) {
        LOG_ERR("llg tokenizer error: %s\n", error_buffer);
        return tokenizer;
    }

    if (tokenizer_cache) {
        llg_free_tokenizer(tokenizer_cache);
    }
    vocab_cache     = vocab;
    tokenizer_cache = tokenizer;

    return llg_clone_tokenizer(tokenizer_cache);
}

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, const char * grammar_kind,
                                       const char * grammar_data) {
    auto * ctx = new llama_sampler_llg;

    if (grammar_kind != nullptr && grammar_kind[0] != '\0') {
        auto tokenizer = llama_sampler_llg_new_tokenizer(vocab);
        if (!tokenizer) {
            delete ctx;
            return nullptr;
        }
        *ctx           = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ grammar_kind,
            /* .grammar_data = */ grammar_data,
            /* .tokenizer    = */ tokenizer,
            /* .grammar      = */ llama_sampler_llg_new(tokenizer, grammar_kind, grammar_data),
        };
        if (!ctx->grammar) {
            llg_free_tokenizer(tokenizer);
            delete ctx;
            return nullptr;
        }
        GGML_ASSERT(((size_t) llama_vocab_n_tokens(vocab) + 31) / 32 * 4 ==
                    llg_matcher_get_mask_byte_size(ctx->grammar));
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ {},
            /* .grammar_data = */ {},
            /* .tokenizer    = */ nullptr,
            /* .grammar      = */ nullptr,
        };
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_llg_i,
        /* .ctx   = */ ctx);
}

#else

llama_sampler * llama_sampler_init_llg(const llama_vocab *, const char *, const char *) {
    LOG_WRN("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    return nullptr;
}

#endif  // LLAMA_USE_LLGUIDANCE
