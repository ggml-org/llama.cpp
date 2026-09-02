#include "server-context.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "kv-ssd-cache.h"
#include "kv-ssd-system-cache.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-context-page-manager.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-task.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>

// fix problem with std::min and std::max
#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
        (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
                                                       common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER,  // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,     // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot;  // forward declaration

struct server_batch {
    llama_batch batch;
    bool        batch_rendered = false;

    struct token {
        int32_t     id_slot;
        llama_token token;
        llama_pos   pos;
        bool        output;
    };

    std::vector<token> tokens;
    int32_t            n_tokens_alloc = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.token = nullptr;  // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.token != nullptr) {
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc) {
        this->n_tokens_alloc = n_tokens_alloc;
        batch                = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output) {
        GGML_ASSERT(batch.token != nullptr);
        if ((int32_t) tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output });
        return true;
    }

    void clear() {
        tokens.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
    }

    int32_t size() const { return (int32_t) tokens.size(); }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t) tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(batch.token != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.token != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        llama_batch view = {
            n_tokens,           batch.token + off,  nullptr, batch.pos + off, batch.n_seq_id + off,
            batch.seq_id + off, batch.logits + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context *  mctx   = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens             spec_draft;
    llama_tokens             spec_prompt;
    std::vector<int32_t>     spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev;  // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool        has_next_token            = true;
    bool        has_new_line              = false;
    bool        truncated                 = false;
    bool        deferred_final_checkpoint = false;  // create final checkpoint after first token
    bool        ssd_cold_start_used       = false;  // SSD cache restored for this slot on cold start
    uint64_t    conv_hash                 = 0;      // consistent conversation hash for all checkpoints
    std::string user_id_;                           // identity of the owning task (for scheduling/affinity)

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt = llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t                               alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled;  // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0;  // number of sent text character

    // TODO @ngxson : move all metrics to a sub-struct for clarity
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    int64_t t_print_last   = 0;
    int32_t n_decoded_last = 0;

    double t_prompt_processing = 0.0;  // ms
    double t_token_generation  = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t              n_draft_total       = 0;  // Total draft tokens generated
    int32_t              n_draft_accepted    = 0;  // Draft tokens actually accepted
    int32_t              n_draft_verif_steps = 0;  // Total draft token verification steps by the target model
    std::vector<int32_t> n_accepted_per_pos;       // Accepted tokens per draft position

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos               = 0;
        generated_text            = "";
        has_new_line              = false;
        truncated                 = false;
        deferred_final_checkpoint = false;
        ssd_cold_start_used       = false;
        conv_hash                 = 0;
        user_id_.clear();
        stop          = STOP_TYPE_NONE;
        stopping_word = "";
        n_sent_text   = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total       = 0;
        n_draft_accepted    = 0;
        n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return !task->need_embd() ||
               (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true;  // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0;  // no budget
    }

    bool is_processing() const { return state != SLOT_STATE_IDLE; }

    bool can_speculate() const { return !!spec; }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n", sampled, n_ctx,
                    prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n", sampled,
                    prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        = ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3 * 1000 * 1000) {
            return;
        }

        const double n_gen_second     = 1e3 / (t_token_generation) * (n_decoded);
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (n_decoded - n_decoded_last);

        t_print_last   = t_now;
        n_decoded_last = n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", n_decoded, n_gen_second,
                n_gen_second_win);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress      = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        = t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        = t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this, "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this, "      total time = %10.2f ms / %5d tokens\n", t_prompt_processing + t_token_generation,
                n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this, "   graphs reused = %10d\n", llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float  draft_ratio = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len =
                n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos +=
                        string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this, "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n", draft_ratio,
                    n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this, "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            { "id",            id              },
            { "n_ctx",         n_ctx           },
            { "speculative",   can_speculate() },
            { "is_processing", is_processing() },
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"]                   = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) (ptask ? ptask->tokens.size() : prompt.tokens.size());
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"]                    = ptask->params.to_json(only_metrics);
            res["next_token"]                = {
                {
                 { "has_next_token", has_next_token },
                 { "has_new_line", has_new_line },
                 { "n_remain", n_remaining },
                 { "n_decoded", n_decoded },
                 }
            };

            if (!only_metrics) {
                res["prompt"]    = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id, -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        const auto & chunk        = input_tokens.find_chunk(idx);
        int32_t      res          = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    void *      cb_data = spec;
                    static auto cb      = [](llama_batch batch, void * user_data) {
                        common_speculative * spec = static_cast<common_speculative *>(user_data);
                        if (!common_speculative_process(spec, batch)) {
                            return 1;
                        }
                        return 0;
                    };

                    llama_pos new_n_past;  // unused for now
                    res = mtmd_helper_decode_image_chunk(mctx, ctx_tgt, chunk.get(), embd, prompt.tokens.pos_next(), id,
                                                         llama_n_batch(ctx_tgt), &new_n_past, cb, cb_data);
                    if (res != 0) {
                        SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                        return -1;
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0;  // success
                }
            }
            return 1;  // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        }
        if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0);  // we should never have an empty batch

        // try batching as much as possible
        int    n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};

//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    uint64_t              n_draft_tokens_total      = 0;
    uint64_t              n_draft_accepted_total    = 0;
    uint64_t              n_draft_verif_steps_total = 0;
    std::vector<uint64_t> n_accepted_per_pos_total;

    void init() { t_start = ggml_time_us(); }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
        t_prompt_processing += slot.t_prompt_processing;
        t_prompt_processing_total += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total += slot.n_decoded;
        n_tokens_predicted += slot.n_decoded;
        t_tokens_generation += slot.t_token_generation;
        t_tokens_generation_total += slot.t_token_generation;

        n_draft_tokens_total += slot.n_draft_total;
        n_draft_accepted_total += slot.n_draft_accepted;
        n_draft_verif_steps_total += slot.n_draft_verif_steps;

        if (n_accepted_per_pos_total.size() < slot.n_accepted_per_pos.size()) {
            n_accepted_per_pos_total.resize(slot.n_accepted_per_pos.size(), 0);
        }
        for (size_t i = 0; i < slot.n_accepted_per_pos.size(); i++) {
            n_accepted_per_pos_total[i] += slot.n_accepted_per_pos[i];
        }
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

  public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context *      mctx  = nullptr;
    const llama_vocab * vocab = nullptr;

    // get the number of in-flight slots for a given user_id (used for
    // per-user concurrency cap enforcement at the HTTP layer). Returns
    // 0 for unknown users. Anonymous bucket uses the "_anonymous" key.
    int get_active_user_count(const std::string & user_id) const {
        if (params_base.max_concurrent_per_user <= 0) {
            return 0;
        }
        const std::string           bucket = user_id.empty() ? std::string("_anonymous") : user_id;
        std::lock_guard<std::mutex> lock(queue_tasks.mutex_tasks);
        auto                        it = user_counts_.find(bucket);
        return (it == user_counts_.end()) ? 0 : it->second;
    }

    server_queue    queue_tasks;
    server_response queue_results;
    server_response queue_slots_sse;
    bool            was_processing_slots = false;

    void broadcast_slots_sse(bool force = false) {
        if (!params_base.endpoint_slots) {
            return;
        }
        if (!force && !queue_slots_sse.has_waiting_tasks()) {
            return;
        }

        json slots_data         = json::array();
        int  n_idle_slots       = 0;
        int  n_processing_slots = 0;

        for (server_slot & slot : slots) {
            slots_data.push_back(slot.to_json(slots_debug == 0));
            if (slot.is_processing()) {
                n_processing_slots++;
            } else {
                n_idle_slots++;
            }
        }

        auto res                = std::make_unique<server_task_result_metrics>();
        res->id                 = -1;
        res->slots_data         = std::move(slots_data);
        res->n_idle_slots       = n_idle_slots;
        res->n_processing_slots = n_processing_slots;
        res->n_tasks_deferred   = queue_tasks.queue_tasks_deferred_size();
        res->t_start            = metrics.t_start;

        res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
        res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
        res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
        res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

        res->n_tokens_max = metrics.n_tokens_max;

        res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
        res->t_prompt_processing       = metrics.t_prompt_processing;
        res->n_tokens_predicted        = metrics.n_tokens_predicted;
        res->t_tokens_generation       = metrics.t_tokens_generation;

        res->n_decode_total     = metrics.n_decode_total;
        res->n_busy_slots_total = metrics.n_busy_slots_total;

        res->n_draft_tokens_total      = metrics.n_draft_tokens_total;
        res->n_draft_accepted_total    = metrics.n_draft_accepted_total;
        res->n_draft_verif_steps_total = metrics.n_draft_verif_steps_total;
        res->n_accepted_per_pos_total  = metrics.n_accepted_per_pos_total;

        queue_slots_sse.broadcast(std::move(res));
    }

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {
    };

    server_context_impl() { mtmd_helper_log_set(common_log_default_callback, nullptr); }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

  private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // Compute checkpoint memory budget: `max(2 GiB, n_ctx_checkpoints * 400 MiB)`.
    // This guarantees the configured checkpoint count always fits with 2x headroom
    // for growth, so the solver's auto-scaled count (8 base + 1 per 8K above 65K,
    // capped at 32) actually fires.  Previously this was 1% of cache_ram, which
    // capped checkpoints to 3-4 at typical cache_ram sizes and made the ring
    // buffer (P1) and auto-scaling (P2) dead code.
    size_t _ckpt_memory_budget() const {
        const size_t default_limit = (size_t) 2 * 1024 * 1024 * 1024;  // 2 GiB floor
        if (params_base.n_ctx_checkpoints <= 0) {
            return default_limit;
        }
        // 400 MiB per configured checkpoint = 200 MiB working set * 2 headroom.
        // Covers worst-case q8_0 KV at 262K context (~100-150 MiB per ckpt).
        const size_t per = (size_t) params_base.n_ctx_checkpoints * 400 * 1024 * 1024;
        return std::max(default_limit, per);
    }

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model_ptr   model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx;  // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace               = 0;
    int slots_debug         = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    // SSD-backed KV cache
    std::unique_ptr<llama::server_context_page_manager> ssd_page_manager;

    // Global system prompt KV cache (cross-conversation)
    std::unique_ptr<kv_ssd_system_cache> sys_cache;

    // Per-slot system prompt hash tracking (dedupe extraction)
    std::unordered_map<int, uint64_t> slot_sys_hash;

    // Monotonic turn counter for SSD cache tiering (incremented per slot release)
    uint32_t ssd_turn_counter = 0;

    // Per-user concurrency tracking (user_id -> active slot count)
    mutable std::unordered_map<std::string, int> user_counts_;

    server_metrics metrics;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string           model_name;     // name of the loaded model, to be used by API
    std::set<std::string> model_aliases;  // additional names for the model
    std::set<std::string> model_tags;     // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt   = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl *    ctx;
        std::string              stage;
        std::vector<std::string> stages;
        int64_t                  t_last_load_progress_ms = 0;

        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };

    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto &        t_last    = d->t_last_load_progress_ms;
            const int64_t t_now     = ggml_time_ms();
            const bool    first     = t_last == 0;
            const bool    done      = progress >= 1.0f;
            const bool    throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                                                             { "stages",  d->stages },
                                                             { "current", d->stage  },
                                                             { "value",   progress  },
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text(this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec(this, "spec_model");

        const bool is_resume = sleeping;

        params_base                       = params;
        const auto output_limits          = server_output_limits(params_base);
        params_base.n_outputs_max         = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft  = params.speculative.has_dft();
        const bool spec_mtp   = std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                                          COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec   = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = { "text_model" };
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }

        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string &       mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams     = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu                     = params_base.mmproj_use_gpu;
            mparams.print_timings               = false;
            mparams.n_threads                   = params_base.cpuparams.n_threads;
            mparams.flash_attn_type             = params_base.flash_attn_type;
            mparams.warmup                      = params_base.warmup;
            mparams.image_min_tokens            = params_base.image_min_tokens;
            mparams.image_max_tokens            = params_base.image_max_tokens;
            mparams.batch_max_tokens            = params_base.mtmd_batch_max_tokens;
            mparams.media_marker                = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start    = ggml_time_us();
            auto    mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed  = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n",
                        total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n",
                                        size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            if (has_spec) {
                common_params params_dft          = params_base;
                bool          measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec         = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t                        hp_ngl = 0;
                uint32_t                        hp_nct = 0;
                uint32_t                        hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                                                             devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                            tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context", total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n", has_draft ? "draft model" : "MTP context",
                            e.what());
                }
            }
        }

        // attach a progress callback
        {
            params_base.load_progress_callback           = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_draft) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_TRC("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            // progress callback
            mparams_dft.progress_callback           = load_progress_callback;
            mparams_dft.progress_callback_user_data = &load_progress_spec;

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq  = 0;
            cparams.ctx_other = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create draft context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (spec_mtp) {
            // no new model load, so we simply report 0.0 and 1.0 progress
            load_progress_callback(0.0f, &load_progress_spec);

            SRV_TRC("creating MTP draft context against the target model '%s'\n", params_base.model.path.c_str());

            auto cparams_mtp          = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.type_k        = params_base.speculative.draft.cache_type_k;
            cparams_mtp.type_v        = params_base.speculative.draft.cache_type_v;
            cparams_mtp.n_rs_seq      = 0;
            cparams_mtp.n_outputs_max = params_base.n_parallel;
            cparams_mtp.ctx_other     = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {
                                                         { "stage", "mmproj_model" }
                });
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot,
                    n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n", params_base.n_parallel, n_ctx_slot,
                params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.mem.init(ctx_tgt, ctx_dft.get());
            slot.spec  = spec.get();
            slot.n_ctx = n_ctx_slot;

            slot.mctx = mctx;
            // has_mtmd is set automatically when IMAGE/AUDIO chunks are pushed
            // to slot.prompt.tokens via push_back() (and to task.tokens via
            // process_mtmd_prompt's chunked constructor). Do NOT pre-flag
            // based on the model being multimodal: a multimodal model
            // routinely serves pure-text slots, and conflating capability with
            // content makes get_tokens()/context-shift/cache-reuse assert on
            // otherwise valid text-only requests. See issue #11.

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                // SSD turn completion tracking + per-user concurrency release
                for (auto & s : slots) {
                    if (s.id == id_slot) {
                        on_turn_complete(s);
                        // Decrement the per-user active-slot counter so the
                        // user can launch another request. Skip empty
                        // user_id_ (anonymous bucket not tracked here).
                        if (!s.user_id_.empty() && params_base.max_concurrent_per_user > 0) {
                            std::lock_guard<std::mutex> lock(queue_tasks.mutex_tasks);
                            auto                        it = user_counts_.find(s.user_id_);
                            if (it != user_counts_.end()) {
                                if (--it->second <= 0) {
                                    user_counts_.erase(it);
                                }
                            }
                        }
                        break;
                    }
                }
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace                    = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug                           = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel));
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n", params_base.n_ctx_checkpoints,
                    params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        // SSD-backed KV cache initialization
        if (!params_base.cache_ssd_path.empty()) {
            llama::kv_eviction_config cfg;
            cfg.max_hot_bytes     = params_base.cache_ssd_hot_ram_mib > 0 ?
                                        (size_t) params_base.cache_ssd_hot_ram_mib * 1024 * 1024 :
                                        6ULL * 1024 * 1024;
            cfg.max_warm_bytes    = params_base.cache_ssd_warm_ram_mib > 0 ?
                                        (size_t) params_base.cache_ssd_warm_ram_mib * 1024 * 1024 :
                                        2ULL * 1024 * 1024;
            // Explicit RAM caps are hard limits: disable auto-sizing so the per-conversation
            // cache in common/kv-ssd-cache.cpp does not override these caps with values
            // derived from sysinfo.freeram at conversation-create time. Both flags unset
            // (default) keeps auto-sizing on so existing setups are unaffected.
            cfg.auto_size         = (params_base.cache_ssd_hot_ram_mib == 0 && params_base.cache_ssd_warm_ram_mib == 0);
            cfg.hot_window_tokens = params_base.cache_ssd_hot_window_tokens;
            cfg.warm_window_tokens        = params_base.cache_ssd_warm_window_tokens;
            cfg.page_size_tokens          = params_base.cache_ssd_page_size_tokens;
            cfg.max_cold_checkpoints      = params_base.cache_ssd_max_cold;
            cfg.turn_inactivity_threshold = 2;

            ssd_page_manager = std::make_unique<llama::server_context_page_manager>(
                params_base.cache_ssd_path.c_str(), &cfg, (size_t) n_ctx, params_base.cache_ssd_max_checkpoints);
            ssd_page_manager->max_conversations   = params_base.cache_ssd_max_conversations;
            // Global cap on cold tier bytes across all conversation directories.
            // 0 = unlimited (legacy default). Specified in MiB for parity with
            // --cache-ssd-hot-ram / --cache-ssd-warm-ram.
            ssd_page_manager->cold_max_size_bytes = params_base.cache_ssd_cold_max_size_mib > 0 ?
                                                        (size_t) params_base.cache_ssd_cold_max_size_mib * 1024 * 1024 :
                                                        0;
            ssd_page_manager->set_no_fsync(params_base.cache_ssd_no_fsync);

            // Set model info after page manager exists
            ssd_page_manager->set_model_info(model_tgt, params_base.cache_type_k, params_base.cache_type_v);

            // Seed turn counter from max on disk (persistent across restarts)
            ssd_turn_counter = ssd_page_manager->get_max_turn_id() + 1;

            SRV_INF("SSD cache enabled: path=%s, hot=%d MiB, warm=%d MiB\n", params_base.cache_ssd_path.c_str(),
                    params_base.cache_ssd_hot_ram_mib, params_base.cache_ssd_warm_ram_mib);
        }

        // Initialize global system prompt cache (cross-conversation reuse)
        if (params_base.cache_ssd_system_prompts > 0 && !params_base.cache_ssd_path.empty()) {
            // Compute model compat_hash (same FNV-1a as set_model_info)
            char     desc_buf[2048];
            int      desc_len = llama_model_desc(model_tgt, desc_buf, sizeof(desc_buf));
            uint64_t compat_h = (desc_len > 0) ? 14695981039346656037ULL : 0;
            if (desc_len > 0) {
                for (int i = 0; i < desc_len; i++) {
                    compat_h ^= (uint64_t) (unsigned char) desc_buf[i];
                    compat_h *= 1099511628211ULL;
                }
                uint32_t tk = (uint32_t) params_base.cache_type_k;
                compat_h ^= (uint64_t) (tk & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tk >> 8) & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tk >> 16) & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tk >> 24) & 0xFF);
                compat_h *= 1099511628211ULL;
                uint32_t tv = (uint32_t) params_base.cache_type_v;
                compat_h ^= (uint64_t) (tv & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tv >> 8) & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tv >> 16) & 0xFF);
                compat_h *= 1099511628211ULL;
                compat_h ^= (uint64_t) ((tv >> 24) & 0xFF);
                compat_h *= 1099511628211ULL;
            }

            static auto hex64 = [](uint64_t h) {
                char buf[17];
                std::snprintf(buf, sizeof(buf), "%016lx", (unsigned long) h);
                return std::string(buf);
            };

            std::string sys_dir = params_base.cache_ssd_path + "/sys-" + hex64(compat_h);

            sys_cache                  = std::make_unique<kv_ssd_system_cache>();
            sys_cache->max_entries     = (size_t) params_base.cache_ssd_system_prompts;
            sys_cache->max_unused_days = params_base.cache_ssd_system_max_days;

            if (sys_cache->init(sys_dir, compat_h)) {
                SRV_INF("system prompt cache enabled: max_entries=%d, max_days=%d, path=%s\n",
                        params_base.cache_ssd_system_prompts, params_base.cache_ssd_system_max_days, sys_dir.c_str());
            } else {
                SRV_WRN("%s\n", "system prompt cache init failed, disabling");
                sys_cache.reset();
            }
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name      = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) { process_single_task(std::move(task)); });
        queue_tasks.on_update_slots([this]() { update_slots(); });
        queue_tasks.on_sleeping_state([this](bool sleeping) { handle_sleeping_state(sleeping); });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings   = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool                      enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                        common_chat_format_example(chat_templates.get(), params_base.use_jinja,
                                                   params_base.default_template_kwargs)
                            .c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking =
                    params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR(
                    "%s: please consider disabling jinja via --no-jinja, or use a custom chat template via "
                    "--chat-template\n",
                    __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = { /* use_jinja             */ params_base.use_jinja,
                            /* prefill_assistant     */ params_base.prefill_assistant,
                            /* reasoning_format      */ params_base.reasoning_format,
                            /* chat_template_kwargs  */ params_base.default_template_kwargs,
                            /* tmpls                 */ std::move(chat_templates),
                            /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                            /* allow_audio           */ mctx ? mtmd_support_audio(mctx) : false,
                            /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                            /* enable_thinking       */ enable_thinking,
                            /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                            /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                            /* media_path            */ params_base.media_path,
                            /* force_pure_content    */ params_base.force_pure_content_parser };

            {
                auto caps      = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it        = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled   = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF(
                        "%s",
                        "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN(
                        "%s",
                        "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // Per-user concurrency cap. If the requesting user_id (or the
        // _anonymous bucket for empty user_id) is already at the cap,
        // refuse the slot. Caller is expected to handle nullptr (defer
        // the task). The HTTP layer has a faster path that returns 429
        // synchronously when the user is also over cap at queue time.
        if (params_base.max_concurrent_per_user > 0) {
            const std::string           bucket = task.user_id.empty() ? std::string("_anonymous") : task.user_id;
            std::lock_guard<std::mutex> lock(queue_tasks.mutex_tasks);
            auto                        it  = user_counts_.find(bucket);
            const int                   cur = (it == user_counts_.end()) ? 0 : it->second;
            if (cur >= params_base.max_concurrent_per_user) {
                SRV_INF("per-user concurrency cap hit for user_id='%s' (cur=%d, cap=%d)\n", bucket.c_str(), cur,
                        params_base.max_concurrent_per_user);
                return nullptr;
            }
        }

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const int common_prefix = tokens.get_common_prefix(task.tokens);

                // Stable-prefix gate: if the caller told us how many leading
                // tokens form a stable prefix (system prompt + thread_summary),
                // reject any slot whose stored prompt does not share that
                // prefix. This lets CLIO keep summary positions byte-identical
                // across turns so the LCP match survives a context trim.
                // Legacy behavior (prompt_stable_prefix_tokens == 0): no gate.
                if (task.params.prompt_stable_prefix_tokens > 0 &&
                    common_prefix < task.params.prompt_stable_prefix_tokens) {
                    SLT_DBG(slot, "LCP match rejected: common_prefix=%d < stable_prefix=%d\n", common_prefix,
                            task.params.prompt_stable_prefix_tokens);
                    continue;
                }

                const float sim_cur = float(common_prefix) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best * task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    if (task.params.prompt_stable_prefix_tokens > 0) {
                        SLT_INF(*ret,
                                "selected slot by LCP similarity (stable prefix=%d), sim_best = %.3f (> %.3f thold), "
                                "f_keep = %.3f\n",
                                task.params.prompt_stable_prefix_tokens, sim_best, slot_prompt_similarity, f_keep);
                    } else {
                        SLT_INF(*ret,
                                "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                                sim_best, slot_prompt_similarity, f_keep);
                    }
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            // Slot affinity: prefer slots already owned by the requesting
            // user_id (cache-locality win) before falling back to LRU.
            // An empty slot.user_id_ is unowned and fair game.
            const bool          want_affinity = !task.user_id.empty();
            const std::string & want_user     = task.user_id;

            if (want_affinity) {
                for (server_slot & slot : slots) {
                    if (slot.is_processing()) {
                        continue;
                    }
                    if (!slot.user_id_.empty() && slot.user_id_ != want_user) {
                        continue;
                    }
                    if (slot.prompt.n_tokens() == 0) {
                        continue;
                    }
                    if (!ret || slot.t_last_used > t_last) {
                        t_last = slot.t_last_used;
                        ret    = &slot;
                    }
                }
            }

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // skip slots owned by a different user (per-user isolation)
                if (!slot.user_id_.empty() && !task.user_id.empty() && slot.user_id_ != task.user_id) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret    = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU%s, t_last = %" PRId64 "\n",
                        want_affinity && ret->user_id_ == want_user ? " (user_id affinity)" : "", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    // Defensive guard: do NOT clear slot.prompt.tokens here.
                    //
                    // The LCP match in get_available_slot() already validated
                    // this slot (passed the sim_best > slot_prompt_similarity
                    // gate above). prompt_cache.load() can return false for
                    // benign reasons: the just-saved entry's f_keep/f_sim are
                    // equal to the baseline (not strictly greater), the cache
                    // has no other entries to compare against, or the cache
                    // is disabled. Clearing slot.prompt.tokens here would wipe
                    // the very state the in-memory checkpoint LCP matcher
                    // needs at server-context.cpp ~line 4096
                    // (`n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`)
                    // and force a from-scratch prefill of the full prompt on
                    // every turn. The ring buffer checkpoints (dbedc9ec) and
                    // the deferred_create_final_checkpoint metadata fix
                    // (3cbc0516e) already provide a correct restore path; we
                    // must not destroy the in-memory prompt they depend on.
                    //
                    // Log a warning so this is visible in debug output, but
                    // leave the prompt and KV cache intact.
                    SLT_WRN(*ret, "%s",
                            "prompt_cache.load() returned false after save; preserving slot prompt for in-memory "
                            "checkpoint LCP match\n");
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }

            // Per-user concurrency accounting: only count tasks that bind
            // a real user_id (anonymous requests don't participate in the
            // cap; they're throttled by the global n_parallel pool).
            if (params_base.max_concurrent_per_user > 0 && !task.user_id.empty()) {
                std::lock_guard<std::mutex> lock(queue_tasks.mutex_tasks);
                user_counts_[task.user_id]++;
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters;  // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        SLT_DBG(slot,
                "[PROBE] launch_slot_with_task: slot.prompt=%zu is_processing=%d seq_pos_min=%lld seq_pos_max=%lld\n",
                slot.prompt.tokens.size(), (int) slot.is_processing(),
                (long long) llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                (long long) llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(),
                            task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens(lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task     = std::make_unique<const server_task>(std::move(task));
        slot.user_id_ = slot.task->user_id;  // remember owner for affinity + counter release

        // Compute conversation hash once from the full task tokens.
        // All checkpoints (mid-prompt and deferred) must use the same
        // hash to prevent splitting checkpoints across conversations.
        {
            const auto & task_tokens = slot.task->tokens.get_tokens();
            size_t       hash_len    = std::min(task_tokens.size(), (size_t) 1024);
            slot.conv_hash           = kv_ssd_hash_tokens((const uint32_t *) task_tokens.data(), hash_len);
        }

        slot.state = slot.task->is_child() ? SLOT_STATE_WAIT_OTHER  // wait for the parent to process prompt
                                             :
                                             SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled                = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test  = slot.generated_text.substr(pos);
            bool              send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(slot.generated_text.begin() + pos + stop_pos, slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok)) {
                stop_pos  = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot,
                    "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, "
                    "n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded,
                    slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() &&
                           (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded,
                                n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 &&
                (ggml_time_us() - slot.t_start_generation > 1000.0f * slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded,
                        (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining,
                result.tok, token_str.c_str());

        return slot.has_next_token;  // continue
    }

    void populate_token_probs(const server_slot &       slot,
                              completion_token_output & result,
                              bool                      post_sampling,
                              bool                      special,
                              int                       idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p     = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs   = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({ cur_p->data[i].id, common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                                         cur_p->data[i].p });
            }
        } else {
            std::vector<llama_token_data> cur       = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t                  max_probs = cur.size();
            const size_t                  n_probs   = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({ cur[i].id, common_token_to_piece(ctx_tgt, cur[i].id, special), cur[i].p });
            }
        }
    }

    void send_error(const server_task &   task,
                    const std::string &   error,
                    const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot &   slot,
                    const std::string &   error,
                    const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int             id_task,
                    const std::string &   error,
                    const enum error_type type            = ERROR_TYPE_SERVER,
                    const int32_t         n_prompt_tokens = 0,
                    const int32_t         n_ctx           = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res             = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot &                   slot,
                               const completion_token_output & tkn,
                               bool                            is_progress,
                               bool                            is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn;  // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content = "";
            res->tokens  = llama_tokens{};
        } else {
            res->content = std::move(slot.generated_text);
            res->tokens  = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output  = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(), slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin(),
                                                                         slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params;  // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res      = std::make_unique<server_task_result_embd>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();
        res->res_type = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i],
                        batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res      = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i],
                        batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot &                parent_slot,
                                       std::vector<server_slot *> & child_slots,
                                       server_task &&               parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent,
                parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (slot.task->id == id_parent || slot.task->id_parent == id_parent)) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // context checkpoints exist only in process memory and are not part of the
    // llama_state_seq file format. persist them in a sidecar file so that
    // action=restore in a fresh process can roll back mid-prompt (e.g. after a
    // BPE boundary re-tokenization of the prompt tail) instead of re-prefilling.
    static bool checkpoints_save_sidecar(const std::list<common_prompt_checkpoint> & checkpoints,
                                         const std::string &                         filepath) {
        FILE * f = fopen(filepath.c_str(), "wb");
        if (f == nullptr) {
            return false;
        }

        bool ok = true;

        const uint32_t magic   = 0x4C434B50;  // "PKCL"
        const uint32_t version = 1;
        const uint32_t count   = (uint32_t) checkpoints.size();

        ok = ok && fwrite(&magic, sizeof(magic), 1, f) == 1;
        ok = ok && fwrite(&version, sizeof(version), 1, f) == 1;
        ok = ok && fwrite(&count, sizeof(count), 1, f) == 1;

        for (const auto & cur : checkpoints) {
            const int64_t  n_tokens = cur.n_tokens;
            const int32_t  pos_min  = cur.pos_min;
            const int32_t  pos_max  = cur.pos_max;
            const uint64_t n_tgt    = cur.data_tgt.size();
            const uint64_t n_dft    = cur.data_dft.size();

            ok = ok && fwrite(&n_tokens, sizeof(n_tokens), 1, f) == 1;
            ok = ok && fwrite(&pos_min, sizeof(pos_min), 1, f) == 1;
            ok = ok && fwrite(&pos_max, sizeof(pos_max), 1, f) == 1;
            ok = ok && fwrite(&n_tgt, sizeof(n_tgt), 1, f) == 1;
            ok = ok && fwrite(&n_dft, sizeof(n_dft), 1, f) == 1;
            ok = ok && (n_tgt == 0 || fwrite(cur.data_tgt.data(), 1, n_tgt, f) == n_tgt);
            ok = ok && (n_dft == 0 || fwrite(cur.data_dft.data(), 1, n_dft, f) == n_dft);
        }

        fclose(f);
        return ok;
    }

    static bool checkpoints_load_sidecar(std::list<common_prompt_checkpoint> & checkpoints,
                                         const std::string &                   filepath) {
        FILE * f = fopen(filepath.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }

        uint32_t magic = 0, version = 0, count = 0;

        bool ok = fread(&magic, sizeof(magic), 1, f) == 1 && fread(&version, sizeof(version), 1, f) == 1 &&
                  fread(&count, sizeof(count), 1, f) == 1 && magic == 0x4C434B50 && version == 1 && count <= 1024;

        std::list<common_prompt_checkpoint> loaded;

        for (uint32_t i = 0; ok && i < count; ++i) {
            auto & cur = loaded.emplace_back();

            uint64_t n_tgt = 0;
            uint64_t n_dft = 0;

            ok = ok && fread(&cur.n_tokens, sizeof(cur.n_tokens), 1, f) == 1;
            ok = ok && fread(&cur.pos_min, sizeof(cur.pos_min), 1, f) == 1;
            ok = ok && fread(&cur.pos_max, sizeof(cur.pos_max), 1, f) == 1;
            ok = ok && fread(&n_tgt, sizeof(n_tgt), 1, f) == 1;
            ok = ok && fread(&n_dft, sizeof(n_dft), 1, f) == 1;

            // sanity: refuse absurd blob sizes (16 GiB per blob)
            ok = ok && n_tgt <= (1ull << 34) && n_dft <= (1ull << 34);

            if (ok) {
                cur.data_tgt.resize(n_tgt);
                cur.data_dft.resize(n_dft);
                ok = ok && (n_tgt == 0 || fread(cur.data_tgt.data(), 1, n_tgt, f) == n_tgt);
                ok = ok && (n_dft == 0 || fread(cur.data_dft.data(), 1, n_dft, f) == n_dft);
            }
        }

        fclose(f);

        if (!ok) {
            return false;
        }

        checkpoints = std::move(loaded);

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        // Evict least-useful checkpoints when at capacity.
        //
        // We use insertion-order eviction (drop the front/oldest entry, append
        // new at back) rather than the old "evict highest pos_min" policy.
        // Deferred finals all carry pos_min == 0, so a pos_min comparator would
        // always tie and pick begin() forever -- a single-slot FIFO with N-1
        // dead entries.  Insertion order gives true round-robin cycling (1..N,
        // 1..N, ...) and works for both mid-prompt and deferred-final entries.
        //
        // For hybrid/recurrent models, the LCP acceptance predicate (line ~4264)
        // still filters by pos_min / pos_max and applies its own n_swa > 0 check;
        // reverse iteration (rbegin/rend) picks the newest qualifying entry, so
        // insertion-order insertion doesn't affect which checkpoint is selected.
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot,
                        "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens "
                        "= %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        // Same insertion-order ring buffer as deferred_create_final_checkpoint():
        // drop the OLDEST entry by list position (front, since insert is at
        // back).  See the matching comment there for why "evict highest
        // pos_min" doesn't work once deferred finals survive across turns.
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            const auto & victim = slot.prompt.checkpoints.front();
            SLT_WRN(slot, "kv ring buffer, evicted oldest (size=%.3f MiB)\n", (float) victim.size() / 1024 / 1024);
            slot.prompt.checkpoints.pop_front();
        }

        // Size-based memory budget (P3): evict largest checkpoints if total
        // memory exceeds _ckpt_memory_budget().  This runs after count-based
        // eviction so that a high checkpoint count can be sustained as long
        // as total memory stays within limits.
        {
            const size_t ckpt_mem_limit = _ckpt_memory_budget();
            size_t       total_ckpt_mem = 0;
            for (const auto & c : slot.prompt.checkpoints) {
                total_ckpt_mem += c.size();
            }
            while (total_ckpt_mem > ckpt_mem_limit && slot.prompt.checkpoints.size() > 1) {
                auto largest = slot.prompt.checkpoints.begin();
                for (auto it = std::next(largest); it != slot.prompt.checkpoints.end(); ++it) {
                    if (it->size() > largest->size()) {
                        largest = it;
                    }
                }
                total_ckpt_mem -= largest->size();
                SLT_TRC(slot,
                        "evicting checkpoint by memory budget (size = %.3f MiB, total = %.3f MiB, limit = %zu MiB)\n",
                        (float) largest->size() / 1024 / 1024, (float) total_ckpt_mem / 1024 / 1024,
                        ckpt_mem_limit / 1024 / 1024);
                slot.prompt.checkpoints.erase(largest);
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // Note: for SWA models, pos_min/pos_max may not cover the full [0, pos_max]
        // range (see LCP acceptance predicate at line ~4264 which applies an
        // n_swa > 0 guard excluding checkpoints where cur.pos_max > pos_next).
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot, "kv ring buffer, pushed mid-prompt (%zu/%d full, pos_min=%d size=%.3f MiB)\n",
                slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                (float) cur.size() / 1024 / 1024);

        // SSD-backed KV cache: store checkpoint on disk
        if (ssd_page_manager) {
            const auto & prefix_tokens = slot.prompt.tokens;
            ssd_page_manager->store_checkpoint_with_tokens(
                slot.id, ctx_tgt, ctx_dft.get(), cur, prefix_tokens.get_tokens().data(),
                prefix_tokens.get_tokens().size(), ssd_turn_counter, slot.conv_hash,
                slot.task ? slot.task->user_id : std::string());
        }
    }

    // Deferred final checkpoint: captures full prompt state after the last
    // batch was processed and the first generation token has been sent.
    // Runs asynchronously relative to the client, so the ~670 MiB SSD write
    // does not block the first token.
    void deferred_create_final_checkpoint(server_slot & slot) {
        if (params_base.n_ctx_checkpoints <= 0) {
            return;
        }
        if (!slot.task || slot.task->type != SERVER_TASK_TYPE_COMPLETION) {
            return;
        }

        // Use prompt positions, not generation positions.
        // The generation positions (done_pos_min/done_pos_max) extend past
        // the prompt end and cause stale positions to persist after restore,
        // triggering "Invalid input batch" on the next turn (issue #8).
        // CRITICAL: Use slot.prompt.n_tokens() - slot.n_decoded + 1 (actual
        // prompt tokens processed) NOT slot.task->n_tokens() (which is the
        // full task token count, including unprocessed conversation history)
        // and NOT slot.prompt.n_tokens() alone (which includes generated
        // tokens added since the deferred flag was set). At this point the
        // first generation token has been sampled (n_decoded incremented)
        // but NOT yet pushed to prompt.tokens (handle_last_sampled_token
        // runs after the sampling loop, not inside it). So:
        //   prompt.tokens has N + (k-1) tokens for the k-th generation
        //   n_decoded = k
        //   prompt_n_tokens = (N + k - 1) - k + 1 = N  (the prompt boundary)
        // Subtracting only n_decoded (without the +1) gave N - 1, which
        // made pos_max cover positions [0, N-2] instead of [0, N-1] -- the
        // last prompt token's KV cache entry was stripped from the checkpoint.
        // On restore the model had to reprocess that 1 token every cold
        // start, and f_keep/f_sim metrics were off by 1 token.
        const int64_t prompt_n_tokens = slot.prompt.n_tokens() - slot.n_decoded + 1;
        if (prompt_n_tokens < 64) {
            return;
        }

        // Note: cold-start mid-prompts (create_checkpoint() emits one with
        // pos_min==0 on a fresh slot, since pos_min_thold==0 with no prior
        // context) intentionally share the (pos_min=0, pos_max=prompt_end)
        // signature of a deferred final.  They're valid LCP snapshots at
        // [0, batch_end] and consumed by the LCP acceptance predicate on
        // future turns.  The SWA-skip guard in get_available() preserves
        // them alongside real deferred finals, costing one of the N ring
        // buffer slots per cold start.  That's correct -- not a bug.
        //

        // Deferred checkpoint always captures final state. Skip the proximity
        // guard used for mid-prompt checkpoints — the deferred ckpt is never
        // "too close" to a prior ckpt; it's the most complete snapshot.

        // Ring buffer: when at capacity, overwrite the least-useful checkpoint
        // in place instead of erasing+emplacing.  This preserves vector
        // capacity (no reallocation churn), gives clean cycling checkpoint
        // numbers (1..N, 1..N, ...), and avoids erase() invalidating iterators.
        // Uses the same insertion-order policy as create_checkpoint(): drop the
        // front (oldest by insertion order) and splice it to the back.
        // See create_checkpoint's comment for why "evict highest pos_min"
        // doesn't work once deferred finals survive across turns.
        //
        // Size-based memory budget (P3): evict largest first if total exceeds
        // _ckpt_memory_budget(), before count-based eviction.
        {
            const size_t ckpt_mem_limit = _ckpt_memory_budget();
            size_t       total_ckpt_mem = 0;
            for (const auto & c : slot.prompt.checkpoints) {
                total_ckpt_mem += c.size();
            }
            while (ckpt_mem_limit > 0 && total_ckpt_mem > ckpt_mem_limit && slot.prompt.checkpoints.size() > 1) {
                auto largest = slot.prompt.checkpoints.begin();
                for (auto it = std::next(largest); it != slot.prompt.checkpoints.end(); ++it) {
                    if (it->size() > largest->size()) {
                        largest = it;
                    }
                }
                total_ckpt_mem -= largest->size();
                SLT_TRC(slot,
                        "evicting checkpoint by memory budget (size = %.3f MiB, total = %.3f MiB, limit = %zu MiB)\n",
                        (float) largest->size() / 1024 / 1024, (float) total_ckpt_mem / 1024 / 1024,
                        ckpt_mem_limit / 1024 / 1024);
                slot.prompt.checkpoints.erase(largest);
            }
        }

        common_prompt_checkpoint * cur_ptr  = nullptr;
        bool                       recycled = false;
        if (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // Ring buffer recycle: move the FRONT (oldest by insertion order)
            // to the BACK and clear it in place.  Splice preserves the std::list
            // node (no allocation); clear() preserves the data vector capacity
            // (no reallocation).  After this, the recycled entry is at the
            // back and ready to be refilled below with today's snapshot.
            //
            // Why insertion order: deferred finals all carry pos_min == 0
            // (the SWA-skip guard below preserves them across turns), so the
            // old "evict highest pos_min" comparison always tied to 0 and
            // picked begin() forever -- a single-slot FIFO with 15 dead
            // entries.  Insertion order breaks the tie.
            //
            // Why splice+clear instead of pop_front+emplace_back: keeps the
            // recycled entry's existing vector capacity instead of throwing
            // it away.  For a server that's been running for hours, this
            // avoids ~32 KB worth of reallocations on every checkpoint.
            const auto & victim = slot.prompt.checkpoints.front();
            SLT_INF(slot, "kv ring buffer, recycled (%zu/%d full, dropped pos_min=%d size=%.3f MiB)\n",
                    slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, victim.pos_min,
                    (float) victim.size() / 1024 / 1024);
            slot.prompt.checkpoints.splice(slot.prompt.checkpoints.end(), slot.prompt.checkpoints,
                                           slot.prompt.checkpoints.begin());
            // The spliced node is now at the back.  clear() preserves capacity.
            slot.prompt.checkpoints.back().clear();
            recycled = true;
        }
        if (!recycled) {
            cur_ptr = &slot.prompt.checkpoints.emplace_back();
        } else {
            cur_ptr = &slot.prompt.checkpoints.back();
        }
        auto & cur = *cur_ptr;

        // Save prompt boundaries: pos_min=0 (start of prompt), pos_max=prompt_n_tokens-1 (end of prompt)
        cur.update_pos(prompt_n_tokens, 0, (llama_pos) prompt_n_tokens - 1);

        // The checkpoint metadata says n_tokens=prompt_n_tokens (prompt only),
        // but update_tgt/update_dft save the FULL KV cache including generated
        // token entries.  A metadata/data mismatch causes hallucination on
        // restore: the model loads generated-token KV cache and attends to its
        // own previous output.  Fix: temporarily strip generated-token entries
        // from the live KV cache, save the prompt-only state, then restore the
        // full KV cache so continued generation is unaffected.
        {
            // --- ctx_tgt (main model KV cache) ---
            size_t full_size = llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> full_state;
            if (full_size > 0) {
                full_state.resize(full_size);
                llama_state_seq_get_data_ext(ctx_tgt, full_state.data(), full_size, slot.id,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            auto * mem_tgt = llama_get_memory(ctx_tgt);
            if (mem_tgt) {
                llama_memory_seq_rm_attn_only(mem_tgt, slot.id, (llama_pos) prompt_n_tokens, -1);
            }

            cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            if (!full_state.empty()) {
                llama_state_seq_set_data_ext(ctx_tgt, full_state.data(), full_state.size(), slot.id,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }

        // --- ctx_dft (draft/MTP model KV cache) ---
        if (ctx_dft) {
            size_t full_size = llama_state_seq_get_size_ext(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> full_state;
            if (full_size > 0) {
                full_state.resize(full_size);
                llama_state_seq_get_data_ext(ctx_dft.get(), full_state.data(), full_size, slot.id,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            auto * mem_dft = llama_get_memory(ctx_dft.get());
            if (mem_dft) {
                llama_memory_seq_rm_attn_only(mem_dft, slot.id, (llama_pos) prompt_n_tokens, -1);
            }

            cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            if (!full_state.empty()) {
                llama_state_seq_set_data_ext(ctx_dft.get(), full_state.data(), full_state.size(), slot.id,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }

        // Log the resulting entry.  The canonical "what just happened" message
        // is the "recycled" or "pushed" final line above; this line confirms
        // what was written into the (possibly recycled) back entry.
        SLT_INF(slot, "kv ring buffer, %s final checkpoint (%zu/%d full, pos_min=%d size=%.3f MiB)\n",
                recycled ? "recycled" : "pushed", slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints,
                cur.pos_min, (float) cur.size() / 1024 / 1024);

        if (ssd_page_manager) {
            const auto & prefix_tokens = slot.task ? slot.task->tokens.get_tokens() : slot.prompt.tokens.get_tokens();
            ssd_page_manager->store_checkpoint_with_tokens(slot.id, ctx_tgt, ctx_dft.get(), cur, prefix_tokens.data(),
                                                           prefix_tokens.size(), ssd_turn_counter, slot.conv_hash,
                                                           slot.task ? slot.task->user_id : std::string());
        }
    }

    // Try to restore the system prompt section from the global SSD KV cache.
    // Returns the n_past value after restoration (-1 on failure).
    // Check if this slot's current prompt represents a system prompt
    // that should be stored in the global cache.
    void maybe_extract_system_prompt(server_slot & slot) {
        if (!sys_cache || !ssd_page_manager) {
            return;
        }

        // Only extract once per slot
        auto it = slot_sys_hash.find(slot.id);
        if (it != slot_sys_hash.end() && it->second != 0) {
            return;
        }

        const auto & tokens = slot.task->tokens.get_tokens();
        if (tokens.empty()) {
            return;
        }

        int n_sys = kv_detect_system_prompt_boundary(
            llama_model_get_vocab(llama_get_model(ctx_tgt)), tokens.data(), (int32_t) tokens.size(),
            params_base.chat_template.empty() ? nullptr : params_base.chat_template.c_str());

        if (n_sys <= 0 || n_sys >= (int32_t) tokens.size()) {
            slot_sys_hash[slot.id] = 1;  // mark as checked, nothing to do
            return;
        }

        // MIN_USEFUL_SYS_TOKENS: don't cache trivial system sections
        // (chat template header with no actual system message). The
        // per-conversation SSD cache handles these cases. This also
        // prevents false-positive boundary detection from filling
        // the system cache with near-empty entries.
        const int32_t MIN_USEFUL_SYS_TOKENS = 16;
        if (n_sys < MIN_USEFUL_SYS_TOKENS) {
            slot_sys_hash[slot.id] = 1;
            return;
        }

        // Get the state at the system prompt boundary
        size_t state_size = llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (state_size == 0) {
            slot_sys_hash[slot.id] = 1;
            return;
        }

        std::vector<uint8_t> state_data(state_size);
        size_t               got = llama_state_seq_get_data_ext(ctx_tgt, state_data.data(), state_size, slot.id,
                                                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (got == 0) {
            slot_sys_hash[slot.id] = 1;
            return;
        }

        // Store in system prompt cache
        sys_cache->store((const uint32_t *) tokens.data(), (uint32_t) n_sys, state_data.data(), (size_t) got);

        slot_sys_hash[slot.id] = 1;
        SLT_INF(slot, "stored system prompt cache entry: n_sys=%d, size=%zu bytes\n", n_sys, (size_t) got);
    }

    // Called when a slot finishes processing (turn complete).
    // Bumps the turn counter for SSD cache tiering.
    void on_turn_complete(server_slot & slot) {
        ssd_turn_counter++;

        // Store the final state to SSD
        if (ssd_page_manager && slot.prompt.n_tokens() > 0) {
            ssd_page_manager->on_turn_complete(ssd_turn_counter);
        }

        // Clean up per-slot state
        slot_sys_hash.erase(slot.id);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t                     n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots   = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG(
                                "not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, "
                                "id_task = %d\n",
                                child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break;  // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break;  // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                }
                break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                }
                break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id  = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                }
                break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res                = std::make_unique<server_task_result_metrics>();
                    res->id                 = task.id;
                    res->slots_data         = std::move(slots_data);
                    res->n_idle_slots       = n_idle_slots;
                    res->n_processing_slots = n_processing_slots;
                    res->n_tasks_deferred   = queue_tasks.queue_tasks_deferred_size();
                    res->t_start            = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total     = metrics.n_decode_total;
                    res->n_busy_slots_total = metrics.n_busy_slots_total;

                    res->n_draft_tokens_total      = metrics.n_draft_tokens_total;
                    res->n_draft_accepted_total    = metrics.n_draft_accepted_total;
                    res->n_draft_verif_steps_total = metrics.n_draft_verif_steps_total;
                    res->n_accepted_per_pos_total  = metrics.n_accepted_per_pos_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    broadcast_slots_sse(true);
                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int     id_slot = task.slot_action.id_slot;
                    server_slot * slot    = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id, reinterpret_cast<const llama_token *>(packed.data()),
                        packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end     = ggml_time_us();
                    const double  t_save_ms = (t_end - t_start) / 1000.0;

                    // persist context checkpoints alongside the state file so that a
                    // restore in a fresh process can roll back mid-prompt (see restore path)
                    if (!slot->prompt.checkpoints.empty()) {
                        if (checkpoints_save_sidecar(slot->prompt.checkpoints, filepath + ".ckpt")) {
                            SLT_INF(*slot, "saved %zu context checkpoints to sidecar\n",
                                    slot->prompt.checkpoints.size());
                        } else {
                            SLT_WRN(*slot, "failed to write checkpoint sidecar %s\n", (filepath + ".ckpt").c_str());
                        }
                    }

                    auto res      = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int     id_slot = task.slot_action.id_slot;
                    server_slot * slot    = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t       n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(),
                                                              packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(),
                                   ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // reload the context checkpoints written at save time; without them the
                    // next request's rollback finds no usable cache data and forces a full
                    // re-prefill ("forcing full prompt re-processing due to lack of cache
                    // data"). if no sidecar exists (state saved by an older build), fall back
                    // to synthesizing a tip checkpoint from the just-restored state, which at
                    // least covers exact continuations.
                    if (params_base.n_ctx_checkpoints > 0 && !slot->prompt.tokens.empty()) {
                        if (checkpoints_load_sidecar(slot->prompt.checkpoints, filepath + ".ckpt")) {
                            SLT_INF(*slot, "restored %zu context checkpoints from sidecar\n",
                                    slot->prompt.checkpoints.size());
                        } else {
                            const llama_pos p_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot->id);
                            const llama_pos p_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot->id);
                            if (p_min >= 0 && p_max >= p_min) {
                                create_checkpoint(*slot, 0, p_min, p_max);
                            }
                        }
                    }

                    const int64_t t_end        = ggml_time_us();
                    const double  t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res      = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int     id_slot = task.slot_action.id_slot;
                    server_slot * slot    = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res      = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto   res   = std::make_unique<server_task_result_get_lora>();
                    res->id      = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto &         lora                    = loras[i];
                        std::string    alora_invocation_string = "";
                        const uint64_t n_alora_tokens          = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens   alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                }
                break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res                  = std::make_unique<server_task_result_apply_lora>();
                    res->id                   = task.id;
                    queue_results.send(std::move(res));
                }
                break;
        }
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t   t_start;

        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) { t_start = ggml_time_us(); }

        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}

        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev  = 0;
        int64_t        t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) {  // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                if (was_processing_slots) {
                    was_processing_slots = false;
                    broadcast_slots_sse(true);
                }
                SRV_TRC("%s", "all slots are idle\n");
                return;  // skip further processing

            } else {
                was_processing_slots = true;
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t     off_next = 0;
        int32_t     n_batch  = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok    = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break;  // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break;  // stop any further processing
            }
        }

        broadcast_slots_sse();
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left,
                        n_discard);

                slot.mem.seq_rm(slot.id, n_keep, n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    // Gate on actual slot content (has_media) rather than the
                    // model capability flag (has_mtmd): context shift is only
                    // safe when there are no image/audio tokens in the slot's
                    // prompt. See issue #11.
                    GGML_ASSERT(!slot.prompt.tokens.has_media());

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens();  // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(slot.prompt.n_tokens(),
                                                  llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                                  llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        { common_speculative_draft(spec.get()); }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft = (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                                           draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot,
                            "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f "
                            "MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(), (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) { slot.handle_last_sampled_token(batch); });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true;  // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return;  // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation     = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n", slot.n_ctx,
                                slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;
                        SLT_DBG(slot,
                                "[PROBE] prefill-init n_past=0 slot.prompt=%zu ssd_page_manager=%d cache_prompt=%d\n",
                                slot.prompt.tokens.size(), (int) (ssd_page_manager != nullptr),
                                (int) slot.task->params.cache_prompt);

                        // cold start: try per-conversation SSD checkpoint restore
                        // Must populate slot.prompt.tokens so get_common_prefix() finds the match
                        if (n_past == 0 && slot.prompt.n_tokens() == 0 && ssd_page_manager) {
                            const auto & task_tokens = slot.task->tokens.get_tokens();
                            if (!task_tokens.empty()) {
                                int32_t              ssd_pos_min = 0, ssd_pos_max = 0;
                                uint64_t             ssd_n_tokens        = 0;
                                int32_t              ssd_lcp             = 0;
                                float                ssd_overlap         = 0.0f;
                                bool                 ssd_is_continuation = false;
                                bool                 ssd_partial         = false;
                                std::vector<uint8_t> ssd_spec_data;
                                if (ssd_page_manager->find_and_load_checkpoint(
                                        task_tokens.data(), task_tokens.size(), ssd_turn_counter, ctx_tgt,
                                        ctx_dft.get(), (uint32_t) slot.id, ssd_pos_min, ssd_pos_max, ssd_n_tokens,
                                        &ssd_spec_data, slot.conv_hash, 0, (uint64_t) task_tokens.size(), &ssd_lcp,
                                        &ssd_overlap, &ssd_is_continuation, &ssd_partial, slot.task->user_id)) {
                                    // Safety check: reject checkpoints that cover more tokens than
                                    // the current task. This can happen if a deferred final
                                    // checkpoint was created after the first generation token
                                    // was added to the prompt (bug fixed in deferred_create_final_checkpoint
                                    // to use slot.task->n_tokens()), or if a checkpoint from a
                                    // longer conversation is incorrectly matched. Such checkpoints
                                    // would restore generation state as prompt state, causing
                                    // hallucination on cold start.
                                    // For partial LCP matches (ssd_partial), the checkpoint can
                                    // legitimately cover more tokens than the current task (agent
                                    // trimmed middle tokens) — the caller caps n_past to the LCP.
                                    if (ssd_n_tokens > task_tokens.size() && !ssd_partial) {
                                        SLT_WRN(slot,
                                                "cold-start: rejecting SSD checkpoint (n_tokens=%lu > task_tokens=%zu) "
                                                "- checkpoint appears to contain generated tokens\n",
                                                (unsigned long) ssd_n_tokens, task_tokens.size());
                                        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                        if (ctx_dft) {
                                            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), slot.id, -1, -1);
                                        }
                                        n_past = 0;
                                        slot.prompt.tokens.clear();
                                        ssd_n_tokens = 0;
                                    }

                                    // Hybrid model LCP validation. Recurrent state is
                                    // content-dependent - if the LCP is much smaller than
                                    // the checkpoint's n_tokens, the recurrent state beyond
                                    // the LCP is from a different conversation and will
                                    // produce garbage logits (all -inf, sampler crash).
                                    //
                                    // Three cases:
                                    //   1. lcp >= n_tokens: full coverage, recurrent state valid
                                    //   2. lcp >= PREFIX_MAX (4096) AND overlap >= 99%:
                                    //      same-conversation checkpoint with full prefix match
                                    //   3. lcp < PREFIX_MAX: partial coverage, cap n_past to LCP
                                    //
                                    // For dense models the recurrent layer is replaced by full
                                    // attention, so this gate is a no-op for them.
                                    if (ssd_n_tokens > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                                        ssd_lcp > 0 && ssd_n_tokens > 0) {
                                        const uint64_t validated_tokens =
                                            std::min(ssd_n_tokens, (uint64_t) KV_SSD_TOKEN_PREFIX_MAX);
                                        const float MIN_LCP_RATIO = 0.80f;
                                        float       lcp_ratio     = (float) ssd_lcp / (float) validated_tokens;
                                        // A partial LCP match (hash didn't match the full sequence)
                                        // is never "full coverage" — the state beyond the LCP is
                                        // from a different token sequence.
                                        bool        full_coverage =
                                            !ssd_partial &&
                                            (ssd_lcp >= (int32_t) ssd_n_tokens ||
                                             (ssd_lcp >= (int32_t) KV_SSD_TOKEN_PREFIX_MAX && ssd_overlap >= 0.99f));
                                        if (!full_coverage && lcp_ratio < MIN_LCP_RATIO) {
                                            SLT_WRN(slot,
                                                    "cold-start: rejecting SSD checkpoint for hybrid model "
                                                    "(lcp=%d < 80%% of validated=%lu/n_tokens=%lu, ratio=%.1f%%, "
                                                    "overlap=%.1f%%)\n",
                                                    ssd_lcp, (unsigned long) validated_tokens,
                                                    (unsigned long) ssd_n_tokens, lcp_ratio * 100.0f,
                                                    ssd_overlap * 100.0f);
                                            // Clear the loaded state and reset
                                            llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                            n_past = 0;
                                            slot.prompt.tokens.clear();
                                            ssd_n_tokens = 0;
                                        } else if (!full_coverage) {
                                            // Partial coverage: cap n_past to LCP so only
                                            // validated recurrent state is used.
                                            SLT_INF(slot,
                                                    "SSD hybrid model partial-coverage: "
                                                    "lcp=%d ssd_n_tokens=%lu cap to LCP\n",
                                                    ssd_lcp, (unsigned long) ssd_n_tokens);
                                            llama_memory_seq_rm_attn_only(llama_get_memory(ctx_tgt), slot.id, ssd_lcp,
                                                                          -1);
                                            // Attention state is only valid up to LCP for hybrid models
                                            n_past = ssd_lcp;
                                        }
                                    }

                                    // Partial LCP match (dense or hybrid): the restored state
                                    // is only valid up to the LCP position. Strip stale
                                    // attention KV beyond ssd_lcp and cap n_past.
                                    // The hybrid model block above already does this for
                                    // recurrent models; dense (non-SWA) models fall through
                                    // here because seq_rm_attn_only == seq_rm for them.
                                    if (ssd_partial && ssd_lcp > 0 && (uint64_t) ssd_lcp < ssd_n_tokens) {
                                        SLT_INF(slot,
                                                "SSD cache partial-LCP restore: "
                                                "lcp=%d ssd_n_tokens=%lu stripping post-LCP attention KV\n",
                                                ssd_lcp, (unsigned long) ssd_n_tokens);
                                        llama_memory_seq_rm_attn_only(llama_get_memory(ctx_tgt), slot.id, ssd_lcp, -1);
                                        const auto cur_pos_min =
                                            llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                        if (cur_pos_min == -1 || cur_pos_min > 0) {
                                            SLT_WRN(slot,
                                                    "SSD cache partial-LCP restore: prefix [0, %d] not in KV cache "
                                                    "(pos_min=%d) - rejecting SSD restore\n",
                                                    ssd_lcp, cur_pos_min);
                                            llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                            if (ctx_dft) {
                                                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), slot.id, -1, -1);
                                            }
                                            n_past = 0;
                                            slot.prompt.tokens.clear();
                                            ssd_n_tokens = 0;
                                        } else {
                                            n_past = ssd_lcp;
                                        }
                                    }

                                    if (ssd_n_tokens > 0) {
                                        // Push checkpoint's token count.
                                        // ssd_lcp from find_match is capped at
                                        // KV_SSD_TOKEN_PREFIX_MAX (4096), but
                                        // same-conversation checkpoints match
                                        // for all n_tokens. Use ssd_n_tokens
                                        // directly for full coverage.
                                        // For partial LCP matches, cap to ssd_lcp —
                                        // only the validated prefix positions are
                                        // safe to push.
                                        int32_t n_push =
                                            (int32_t) std::min((uint64_t) task_tokens.size(), ssd_n_tokens);
                                        if (ssd_partial) {
                                            n_push = std::min(n_push, ssd_lcp);
                                        }
                                        for (int32_t i = 0; i < n_push; i++) {
                                            slot.prompt.tokens.push_back(task_tokens[i]);
                                        }
                                        SLT_INF(slot,
                                                "SSD cache restore: lcp=%d ssd_n_tokens=%lu pos=[%d,%d] n_push=%d "
                                                "overlap=%.1f%% continuation=%d%s\n",
                                                ssd_lcp, (unsigned long) ssd_n_tokens, ssd_pos_min, ssd_pos_max, n_push,
                                                ssd_overlap * 100.0f, (int) ssd_is_continuation,
                                                ssd_partial ? " (partial LCP)" : "");

                                        // Create in-memory checkpoint so downstream
                                        // checkpoint search finds it. pos_min=0
                                        // triggers the cur.pos_min==0 match in
                                        // the lambda. Empty data vectors mean
                                        // load_tgt/load_dft are no-ops — the
                                        // SSD restore already loaded full state.
                                        auto & ckpt = slot.prompt.checkpoints.emplace_back();
                                        ckpt.update_pos(n_push, 0, (llama_pos) n_push);

                                        // Restore speculative impl state (pending_h for MTP)
                                        // so the first draft after cold-start is consistent.
                                        if (spec && !ssd_spec_data.empty()) {
                                            common_speculative_set_state(spec.get(), slot.id, ssd_spec_data);
                                        }

                                        // Flag that SSD cache restored this slot.
                                        slot.ssd_cold_start_used = true;
                                        // For hybrid models with partial coverage, n_past was already set to ssd_lcp above.
                                        // For dense models with partial LCP match, n_past was also set above.
                                        // For full coverage (hybrid or dense, non-partial), set n_past to n_push.
                                        if (n_past == 0) {
                                            n_past = n_push;
                                        }
                                    }  // ssd_n_tokens > 0 (not rejected by hybrid LCP check)
                                }
                            }
                        }

                        // cold start: try SSD system prompt cache restore (global, cross-conversation)
                        // Only cold starts (empty slot) need system prompt cache.
                        // Warm slots (n_tokens > 0) already have full context from
                        // the in-memory prompt cache LCP restore or previous turn.
                        if (n_past == 0 && slot.prompt.n_tokens() == 0 && sys_cache && ssd_page_manager) {
                            const auto & task_tokens = slot.task->tokens.get_tokens();
                            int          n_sys       = kv_detect_system_prompt_boundary(
                                llama_model_get_vocab(llama_get_model(ctx_tgt)), task_tokens.data(),
                                (int32_t) task_tokens.size(),
                                params_base.chat_template.empty() ? nullptr : params_base.chat_template.c_str());

                            // MIN_USEFUL_SYS_TOKENS: skip system prompt cache when the
                            // detected boundary is too small (just a chat template header
                            // with no system message). The per-conversation SSD cache
                            // handles these cases with full prompt state restore.
                            const int32_t MIN_USEFUL_SYS_TOKENS = 16;
                            if (n_sys >= MIN_USEFUL_SYS_TOKENS && n_sys < (int32_t) task_tokens.size()) {
                                uint64_t sys_hash = kv_ssd_system_cache::hash_tokens(
                                    (const uint32_t *) task_tokens.data(), (size_t) n_sys);

                                std::vector<uint8_t> sys_data;
                                if (sys_cache->load((const uint32_t *) task_tokens.data(), (uint32_t) n_sys,
                                                    sys_data)) {
                                    // Restore system prompt state from cache
                                    // Match the save flag (PARTIAL_ONLY) used by
                                    // maybe_extract_system_prompt(). The system prompt
                                    // cache only stores recurrent memory state, not
                                    // attention KV cache. Loading with NONE would try
                                    // to read attention data that isn't there.
                                    if (llama_state_seq_set_data_ext(ctx_tgt, sys_data.data(), sys_data.size(), slot.id,
                                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0) {
                                        n_past = n_sys;

                                        // Populate slot.prompt.tokens so get_common_prefix()
                                        // finds the match and preserves n_past. Without this,
                                        // get_common_prefix() returns 0 on empty tokens and
                                        // the restored state is wiped by seq_rm(ctx, 0, -1).
                                        for (int32_t i = 0; i < n_sys; i++) {
                                            slot.prompt.tokens.push_back(task_tokens[i]);
                                        }

                                        SLT_INF(slot, "system prompt cache hit: hash=%016lx, n_sys=%d\n", sys_hash,
                                                n_sys);
                                    }
                                }
                            }
                        }

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping",
                                       ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                SLT_DBG(slot, "[PROBE] LCP n_past=%d slot.prompt=%zu task=%d\n", n_past,
                                        slot.prompt.tokens.size(), slot.task->n_tokens());

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot,
                                            "only caching to alora invocation start (n_past = %d, "
                                            "alora_invocation_start = %d)\n",
                                            n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse = llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                                             // KV-cache reuse via shifting is only safe
                                                             // for slots with no image/audio tokens.
                                                             // has_media() reflects actual slot content;
                                                             // has_mtmd() can also be true on a pure-text
                                                             // slot on a multimodal model (issue #11).
                                                             !slot.prompt.tokens.has_media();

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n",
                                            n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    // See comment above (issue #11).
                                    GGML_ASSERT(!slot.prompt.tokens.has_media());

                                    size_t head_c = n_past;  // cache
                                    size_t head_p = n_past;  // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse,
                                            n_past);

                                    while (head_c < slot.prompt.tokens.size() && head_p < input_tokens.size()) {
                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size() &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot,
                                                    "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> "
                                                    "[%zu, %zu)\n",
                                                    n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm(slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i,
                                                                             slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                                SLT_DBG(slot, "[PROBE] cache_prompt=FALSE -> n_past=0 (probe=%d)\n", 1);
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
                            SLT_DBG(slot,
                                    "[PROBE] pre-reset n_past=%d pos_next=%d n_swa=%d has_new=%d pos_min_thold=%d "
                                    "n_ckpts=%zu ssd_cold=%d\n",
                                    n_past, (int) pos_next, n_swa, (int) has_new_tokens, pos_min_thold,
                                    slot.prompt.checkpoints.size(), (int) slot.ssd_cold_start_used);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                SLT_DBG(slot, "[PROBE] in-block n_past=%d pos_min=%d pos_min_thold=%d\n", n_past,
                                        pos_min, pos_min_thold);
                                if (pos_min == -1) {
                                    SLT_WRN(slot,
                                            "pos_min == -1 but n_past = %d (seq_id = %d) - KV cache is empty, "
                                            "resetting n_past to 0\n",
                                            n_past, slot.id);
                                    n_past = 0;
                                    slot.prompt.tokens.clear();
                                    slot.prompt.checkpoints.clear();
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(
                                        n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ?
                                                                   common_token_to_piece(ctx_tgt, token) :
                                                                   "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ?
                                                                   common_token_to_piece(ctx_tgt, token) :
                                                                   "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min <= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(), slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n",
                                                    cur.pos_min, cur.pos_max, pos_min_thold);
                                            // Workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]:
                                            // for SWA models, pos_min/pos_max can be incorrect
                                            // so the saved state may not actually contain positions
                                            // past pos_next. Exclude those checkpoints. For non-SWA
                                            // (KV cache + bounded rec window like Qwen3.6 hybrid),
                                            // pos_max is correct and the checkpoint fully covers
                                            // [pos_min, pos_max] — the deferred-final pattern
                                            // (pos_min=0, pos_max=prompt_end) is a valid snapshot
                                            // at any LCP position <= pos_max.
                                            if (n_swa > 0 && cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        });

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next),
                                                            (size_t) it->n_tokens);
                                        SLT_DBG(slot, "[PROBE] ckpt-restored n_past=%d ckpt=[%d,%d]\n", n_past,
                                                it->pos_min, it->pos_max);
                                        SLT_TRC(slot,
                                                "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = "
                                                "%" PRId64 ", n_past = %d, size = %.3f MiB)\n",
                                                it->pos_min, it->pos_max, it->n_tokens, n_past,
                                                (float) it->size() / 1024 / 1024);

                                        // after restoring a checkpoint, the recurrent state positions
                                        // may not align with token indices on hybrid models (MoE/SSM).
                                        // use seq_rm_attn_only instead of full seq_rm to avoid crash
                                        slot.ssd_cold_start_used = true;
                                    }

                                    if (do_reset) {
                                        SLT_DBG(slot, "[PROBE] do_reset=true -> n_past=0 (no usable ckpt, probe=%d)\n",
                                                1);
                                        SLT_TRC(
                                            slot,
                                            "forcing full prompt re-processing due to lack of cache data (likely due "
                                            "to SWA or hybrid/recurrent memory, see %s)\n",
                                            "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past   = 0;
                                    }
                                }
                            }

                            // do_reset fallback for warm slots: when LCP matched deeply
                            // through the system prompt but no in-memory checkpoint covers
                            // the LCP position (hybrid/MoE models with bounded rec window),
                            // the system prompt cache may still hold the recurrent state
                            // for tokens [0, n_sys). Without this fallback, every turn that
                            // grows past the rec window forces a full reprocess of the
                            // system prompt on the next user input. The cold-start path
                            // already does this, but only when slot.prompt is empty.
                            //
                            // Tries prefix-match if exact load fails: scans stored entries
                            // for one whose first K tokens match the new task's first K
                            // tokens. This handles the common case where the chat template
                            // inserts a few dynamic tokens between the system section and
                            // the first user message, shifting the boundary by tens of
                            // tokens between turns.
                            if (n_past == 0 && slot.prompt.n_tokens() > 0 && sys_cache && ssd_page_manager) {
                                const auto & task_tokens = slot.task->tokens.get_tokens();
                                int          n_sys       = kv_detect_system_prompt_boundary(
                                    llama_model_get_vocab(llama_get_model(ctx_tgt)), task_tokens.data(),
                                    (int32_t) task_tokens.size(),
                                    params_base.chat_template.empty() ? nullptr : params_base.chat_template.c_str());

                                const int32_t  MIN_USEFUL_SYS_TOKENS = 16;
                                const uint32_t MIN_PREFIX_MATCH      = 64;

                                int                  recovered_n_sys = 0;
                                std::vector<uint8_t> sys_data;
                                bool                 recovered = false;

                                if (n_sys >= MIN_USEFUL_SYS_TOKENS && n_sys < (int32_t) task_tokens.size()) {
                                    if (sys_cache->load((const uint32_t *) task_tokens.data(), (uint32_t) n_sys,
                                                        sys_data)) {
                                        recovered_n_sys = n_sys;
                                        recovered       = true;
                                    } else if ((uint32_t) n_sys >= MIN_PREFIX_MATCH &&
                                               sys_cache->load_prefix((const uint32_t *) task_tokens.data(),
                                                                      (uint32_t) n_sys, MIN_PREFIX_MATCH, sys_data)) {
                                        // Prefix fallback matched. Use the boundary as
                                        // n_past - the loaded state's recurrent state covers
                                        // up to the stored entry's n_tokens, but for hybrid
                                        // models the state at position N is computed from
                                        // tokens [0, N). If the first 64+ tokens match we
                                        // accept the approximate match; any model state drift
                                        // is bounded by the small divergent region at the
                                        // boundary.
                                        recovered_n_sys = n_sys;
                                        recovered       = true;
                                        SLT_DBG(slot,
                                                "[PROBE] sys-cache-fallback prefix-match recovered at n_sys=%d (exact "
                                                "match failed)\n",
                                                n_sys);
                                    }
                                }

                                if (recovered) {
                                    if (llama_state_seq_set_data_ext(ctx_tgt, sys_data.data(), sys_data.size(), slot.id,
                                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0) {
                                        n_past   = recovered_n_sys;
                                        pos_next = recovered_n_sys;
                                        // Replace stale tokens from the previous turn with
                                        // the new task's first n_sys tokens. Stale tokens
                                        // would corrupt the prefill loop after n_past jumps.
                                        slot.prompt.tokens.keep_first(0);
                                        for (int32_t i = 0; i < recovered_n_sys; i++) {
                                            slot.prompt.tokens.push_back(task_tokens[i]);
                                        }
                                        // Route seq_rm through seq_rm_attn_only so the loaded
                                        // recurrent state is preserved (attention is empty for
                                        // the system section and will be filled as prefill
                                        // processes the remaining tokens).
                                        slot.ssd_cold_start_used = true;
                                        SLT_DBG(slot,
                                                "[PROBE] sys-cache-fallback n_past=%d (n_sys=%d) after do_reset\n",
                                                n_past, recovered_n_sys);
                                    }
                                }
                            }

                            {
                                // Erase checkpoints whose pos_max is past pos_next.
                                //
                                // Exception: deferred-final snapshots (pos_min == 0)
                                // are baked from a known-good state at end-of-prompt.
                                // SWA invalidation should not erase them on every turn
                                // because they were valid at capture time, and the load
                                // predicate below already filters out checkpoints that
                                // no longer cover pos_next (n_swa > 0 && cur.pos_max >
                                // pos_next in the acceptance loop).
                                //
                                // Without this guard, the SWA step would erase every
                                // older deferred final, defeating the ring buffer
                                // (P1) and the auto-scaled checkpoint count (P2).
                                // Both checkpoints shrink to a stale pair and the
                                // ring buffer never accumulates past 2 entries.
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur                     = *it;
                                    const bool   deferred_final_snapshot = (cur.pos_min == 0);
                                    if (!deferred_final_snapshot && cur.pos_max > pos_next) {
                                        SLT_TRC(slot,
                                                "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, "
                                                "n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n",
                                                cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next,
                                                (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot,
                                    "need to evaluate at least 1 token for each active slot (n_past = %d, "
                                    "task.n_tokens() = %d)\n",
                                    n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }
                        SLT_DBG(slot, "[PROBE] FINAL n_past=%d task.n=%d slot.prompt=%zu\n", n_past,
                                slot.task->n_tokens(), slot.prompt.tokens.size());

                        slot.n_prompt_tokens_cache     = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    }  // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    const int64_t t_now      = ggml_time_us();
                    slot.t_prompt_processing = (t_now - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot.
                    // skip when a checkpoint restored state (SSD cache or in-memory).
                    // a full common_context_seq_rm is unsafe on hybrid because it
                    // routes through mem_recr->seq_rm, whose n_rs_seq rollback path
                    // can fail when the saved checkpoint's position range extends
                    // past n_past (e.g., previous turn's generation tokens beyond
                    // LCP). the seq_rm_attn_only path handles both caches safely.
                    // See: https://github.com/fewtarius/llama-ai/issues/8
                    if (!slot.ssd_cold_start_used) {
                        const llama_pos p0 = slot.prompt.tokens.pos_next();

                        SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                        slot.mem.seq_rm(slot.id, p0, -1);
                    } else {
                        // For SSD/in-memory restored slots on hybrid models:
                        // clear attn cells AND stale position tracking in mem_recr.
                        //
                        // full common_context_seq_rm is unsafe on hybrid because
                        // seq_rm's n_rs_seq rollback path can fail when rollback
                        // exceeds n_rs_seq (mamba/GDN models). the seq_rm_attn_only
                        // path calls mem_attn->seq_rm (regular, frees cells) AND
                        // mem_recr->seq_rm_positions_only (clears seq_id without
                        // touching data - no rollback path).
                        //
                        // both are required: without clearing mem_recr's stale
                        // positions, hybrid seq_pos_max (min of attn+recr) would
                        // still report the stale value, and llama_batch_init
                        // validation fails on the next batch. See issue #8.
                        const llama_pos p0 = slot.prompt.tokens.pos_next();
                        SLT_DBG(slot, "SSD used, seq_rm_attn_only [%d, end)\n", p0);
                        auto * mem = llama_get_memory(ctx_tgt);
                        if (mem) {
                            llama_memory_seq_rm_attn_only(mem, slot.id, p0, -1);
                        }
                        if (ctx_dft) {
                            mem = llama_get_memory(ctx_dft.get());
                            if (mem) {
                                llama_memory_seq_rm_attn_only(mem, slot.id, p0, -1);
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot,
                                "processing pre-alora tokens without the adapter (n_tokens = %d, "
                                "alora_invocation_start = %d)\n",
                                slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale                       = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id                 = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint =
                        do_checkpoint && (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                          ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS || n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL  // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t  n_tokens_out = 0;
                        int32_t res          = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get());  // copy
                        }

                        has_mtmd = true;
                    }

                    const auto & spans         = slot.task->params.message_spans;
                    const auto   last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break;  // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n",
                                    slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id, cur_tok, slot.prompt.tokens.pos_next(), slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto   pos         = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() ||
                                pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = { 4 + n_ubatch, 4 };

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start        = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        // Extract system prompt for cross-conversation SSD cache
                        maybe_extract_system_prompt(slot);

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !(params_base.checkpoint_near_end && near_prompt_end)) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || is_last_user_message ||
                                                      (params_base.checkpoint_near_end && near_prompt_end) ||
                                                      n_tokens_start > slot.prompt.checkpoints.back().n_tokens +
                                                                           params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no",
                            pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n",
                           "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true;  // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        const int ret = llama_decode(ctx_tgt, batch_view);

        metrics.on_decoded(slots);

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN(
                "failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, "
                "ret = %d\n",
                off, n_batch, ret);

            return false;  // retry with the updated n_batch
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(
                        string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off,
                                      off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special || slot.task->params.sampling.preserved_tokens.find(token) !=
                                              slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                // Defer final checkpoint SSD I/O to after first token.
                // The inline mid-prompt checkpoints were created during
                // prompt processing; this final checkpoint captures the
                // full prompt state after the last batch was processed,
                // so warm restarts can skip nearly all prompt tokens.
                slot.deferred_final_checkpoint = true;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return;  // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.n_decoded += 1;

            if (slot.n_decoded == 1) {
                slot.t_start_generation  = t_now;
                slot.t_print_last        = t_now;
                slot.n_decoded_last      = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            completion_token_output result;
            result.tok = id;
            result.text_to_send =
                common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob = 1.0f;  // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                if (slot.deferred_final_checkpoint) {
                    slot.deferred_final_checkpoint = false;
                    deferred_create_final_checkpoint(slot);
                }
                slot.print_timings();
                send_final_response(slot);
                metrics.on_prediction(slot);
                slot.release();

                return;
            }

            // Defer final checkpoint after first token is sent.
            // The inline mid-prompt checkpoints in create_checkpoint()
            // were created before the last batch; this captures the
            // full prompt state so warm restarts skip nearly all tokens.
            if (slot.deferred_final_checkpoint) {
                slot.deferred_final_checkpoint = false;
                deferred_create_final_checkpoint(slot);
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty() ||
                slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch,
                                                                   slot.spec_draft);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1,
                                    slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n",
                                ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        {
                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);
                        }

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            // seq_rm already handled by slot.mem above (covers both ctx_tgt and ctx_dft)
                        }

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const int64_t t_now = ggml_time_us();

            const auto ids = std::move(slot.spec_draft);

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += ids.size() - 1;
            slot.n_draft_verif_steps += 1;

            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < ids.size() - 1 && i < slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ ids.begin(), ids.end() - 1 });

            slot.sampled = ids.back();  // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(),
                    n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok = ids[i];
                result.text_to_send =
                    common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob = 1.0f;  // set later

                // TODO: set result.probs

                slot.n_decoded += 1;

                if (!process_token(result, slot)) {
                    if (slot.deferred_final_checkpoint) {
                        slot.deferred_final_checkpoint = false;
                        deferred_create_final_checkpoint(slot);
                    }
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    return;
                }
            }

            // Defer final checkpoint after first token(s) sent
            if (slot.deferred_final_checkpoint) {
                slot.deferred_final_checkpoint = false;
                deferred_create_final_checkpoint(slot);
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft,
                    slot.prompt.n_tokens());
        });
    }

    int get_slot_n_ctx() { return slots.back().n_ctx; }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}

server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id        = llama_vocab_bos(impl->vocab);
    auto eos_id        = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta{
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;

    server_res_generator(server_queue &    queue_tasks,
                         server_response & queue_results,
                         int               sleep_idle_seconds,
                         bool              bypass_sleep = false) :
        rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }

    void ok(const json & response_data) {
        status = 200;
        data   = safe_json_to_str(response_data);
    }

    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data   = safe_json_to_str({
            { "error", error_data }
        });
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(const server_http_req &         req,
                                                                             server_task_type                type,
                                                                             const json &                    data,
                                                                             const std::vector<raw_buffer> & files,
                                                                             task_response_type              res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto   res           = create_response();
    auto   completion_id = gen_chatcmplid();
    auto & rd            = res->rd;
    auto & params        = this->params;

    std::vector<server_task> tasks;
    res->set_req(&req);  // will also set spipe if needed
    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        tasks.clear();

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) /
                                   string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(ctx_server.vocab, params, meta->logit_bias_eog, data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot      = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // Operator-supplied user identity. The HTTP layer may also
            // synthesize a tenant/user label via metadata->user_id; the
            // raw field is the canonical CachyLLama form. Empty value is
            // valid (anonymous bucket, no per-user cap).
            try {
                task.user_id = server_task::validate_user_id(json_value(data, "llama_user_id", std::string()));
            } catch (const std::invalid_argument & e) {
                throw std::invalid_argument(std::string("invalid llama_user_id: ") + e.what());
            }

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Per-user concurrency cap fast-fail (HTTP 429). After the request is
    // queued, we can immediately tell if the user is over cap. The slow
    // path (queue->get_available_slot returning nullptr) also enforces
    // this, but the fast path saves a queue round-trip and a log of
    // deferred tasks. We only check tasks carrying a non-empty user_id;
    // anonymous requests are throttled only by the global n_parallel.
    if (params.max_concurrent_per_user > 0) {
        std::string first_user_id;
        for (const auto & t : tasks) {
            if (!t.user_id.empty()) {
                first_user_id = t.user_id;
                break;
            }
        }
        if (!first_user_id.empty()) {
            const int cur = ctx_server.get_active_user_count(first_user_id);
            if (cur >= params.max_concurrent_per_user) {
                res->error(format_error_response("user '" + first_user_id + "' is at the per-user concurrency cap of " +
                                                     std::to_string(params.max_concurrent_per_user) + " (currently " +
                                                     std::to_string(cur) + " in-flight). " +
                                                     "Retry after in-flight requests complete.",
                                                 ERROR_TYPE_RATE_LIMIT));
                return res;
            }
        }
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res;  // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final *>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res;  // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(first_result.get()) != nullptr ||
                    dynamic_cast<server_task_result_cmpl_final *>(first_result.get()) != nullptr);

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = "";  // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status       = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        { "event", "error"  },
                        { "data",  res_json },
                    });
                } else {
                    return format_oai_sse(json{
                        { "error", res_json }
                    });
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false;  // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false;  // no more data, terminate
                }

                // receive subsequent results
                bool    timeout    = false;
                int64_t start_time = ggml_time_ms();
                auto    result     = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true;  // should_stop condition met
                    } else if (sse_ping_interval > 0 &&
                               ggml_time_ms() - start_time > (int64_t) sse_ping_interval * 1000) {
                        timeout = true;
                        return true;  // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false;  // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output        = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false;  // terminate on error
                } else {
                    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(result.get()) != nullptr ||
                                dynamic_cast<server_task_result_cmpl_final *>(result.get()) != nullptr);
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output          = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server) :
    params(params),
    ctx_server(*ctx_server.impl),
    queue_tasks(ctx_server.impl->queue_tasks),
    queue_results(ctx_server.impl->queue_results),
    queue_slots_sse(ctx_server.impl->queue_slots_sse) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server;  // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({
            { "status", "ok" }
        });
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);  // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics *>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json{
            { "counter",
             { { { "name", "prompt_tokens_total" },
                  { "help", "Number of prompt tokens processed." },
                  { "value", (uint64_t) res_task->n_prompt_tokens_processed_total } },
                { { "name", "prompt_seconds_total" },
                  { "help", "Prompt process time" },
                  { "value", (uint64_t) res_task->t_prompt_processing_total / 1.e3 } },
                { { "name", "tokens_predicted_total" },
                  { "help", "Number of generation tokens processed." },
                  { "value", (uint64_t) res_task->n_tokens_predicted_total } },
                { { "name", "tokens_predicted_seconds_total" },
                  { "help", "Predict process time" },
                  { "value", (uint64_t) res_task->t_tokens_generation_total / 1.e3 } },
                { { "name", "n_decode_total" },
                  { "help", "Total number of llama_decode() calls" },
                  { "value", res_task->n_decode_total } },
                { { "name", "n_tokens_max" },
                  { "help", "Largest observed n_tokens." },
                  { "value", res_task->n_tokens_max } },
                { { "name", "spec_decode_num_draft_tokens_total" },
                  { "help", "Total draft tokens generated" },
                  { "value", res_task->n_draft_tokens_total } },
                { { "name", "spec_decode_num_accepted_tokens_total" },
                  { "help", "Total draft tokens accepted by the target model" },
                  { "value", res_task->n_draft_accepted_total } },
                { { "name", "spec_decode_num_drafts_total" },
                  { "help", "Total speculative decoding verification steps" },
                  { "value", res_task->n_draft_verif_steps_total } } }                                           },
            { "gauge",
             { { { "name", "prompt_tokens_seconds" },
                  { "help", "Average prompt throughput in tokens/s." },
                  { "value", res_task->n_prompt_tokens_processed ?
                                 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed :
                                 0. } },
                { { "name", "predicted_tokens_seconds" },
                  { "help", "Average generation throughput in tokens/s." },
                  { "value", res_task->n_tokens_predicted ?
                                 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted :
                                 0. } },
                { { "name", "requests_processing" },
                  { "help", "Number of requests processing." },
                  { "value", (uint64_t) res_task->n_processing_slots } },
                { { "name", "requests_deferred" },
                  { "help", "Number of requests deferred." },
                  { "value", (uint64_t) res_task->n_tasks_deferred } },
                { { "name", "n_busy_slots_per_decode" },
                  { "help", "Average number of busy slots per llama_decode() call" },
                  { "value",
                    (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f) } } } }
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help << "\n"
                           << "# TYPE llamacpp:" << name << " " << type << "\n"
                           << "llamacpp:" << name << " " << value << "\n";
            }
        }

        // labeled counter: one time series per draft position
        if (!res_task->n_accepted_per_pos_total.empty()) {
            prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                          " Accepted tokens per draft position\n"
                       << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
            for (size_t i = 0; i < res_task->n_accepted_per_pos_total.size(); i++) {
                prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"" << i << "\"} "
                           << res_task->n_accepted_per_pos_total[i] << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type                       = "text/plain; version=0.0.4";
        res->status                             = 200;
        res->data                               = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);  // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics *>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->get_slots_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        if (!params.endpoint_slots) {
            res->status       = 501;
            res->content_type = "application/json";
            res->data         = safe_json_to_str(format_error_response(
                "This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        res->status       = 200;
        res->content_type = "text/event-stream";

        auto sse_client = std::make_shared<server_sse_client>(queue_slots_sse);

        // Request initial metrics/slots broadcast
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = queue_tasks.get_new_id();
        queue_tasks.post(std::move(task), true);

        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() { return req.should_stop(); });
            if (result == nullptr) {
                return false;
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true;
        };
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(
                format_error_response("This server does not support slots action. Start it with `--slot-save-path`",
                                      ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server;  // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling                           = params.sampling;
        json default_generation_settings_for_props = json{
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx      },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        std::string primary_backend     = "cpu";
        std::string primary_device_name = "CPU";
        json        devices_json        = json::array();
        size_t      dev_count           = ggml_backend_dev_count();
        for (size_t i = 0; i < dev_count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev) {
                const char *               name  = ggml_backend_dev_name(dev);
                const char *               desc  = ggml_backend_dev_description(dev);
                enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
                std::string                stype = "cpu";
                if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    stype = "gpu";
                } else if (dtype == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                    stype = "accel";
                }
                std::string        reg_name = "cpu";
                ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
                if (reg) {
                    const char * rname = ggml_backend_reg_name(reg);
                    if (rname) {
                        std::string rstr(rname);
                        for (char & c : rstr) {
                            c = std::tolower(c);
                        }
                        reg_name = rstr;
                    }
                }
                devices_json.push_back({
                    { "name",        name ? name : "" },
                    { "description", desc ? desc : "" },
                    { "type",        stype            },
                    { "backend",     reg_name         },
                });
                if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU ||
                    dtype == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                    primary_backend     = reg_name;
                    primary_device_name = desc ? desc : (name ? name : "GPU");
                }
            }
        }

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel                     },
            { "model_alias",                 meta->model_name                      },
            { "model_path",                  meta->model_path                      },
            { "backend",                     primary_backend                       },
            { "device_name",                 primary_device_name                   },
            { "devices",                     devices_json                          },
            { "modalities",
             json{
                  { "vision", meta->has_inp_image },
                  { "video", meta->has_inp_video },
                  { "audio", meta->has_inp_audio },
              }                                                                    },
            { "media_marker",                get_media_marker()                    },
            { "endpoint_slots",              params.endpoint_slots                 },
            { "endpoint_props",              params.endpoint_props                 },
            { "endpoint_metrics",            params.endpoint_metrics               },
            { "ui",                          params.ui                             },
            { "ui_settings",                 meta->json_ui_settings                },
            { "chat_template",               tmpl_default                          },
            { "chat_template_caps",          meta->chat_template_caps              },
            { "bos_token",                   meta->bos_token_str                   },
            { "eos_token",                   meta->eos_token_str                   },
            { "build_info",                  meta->build_info                      },
            { "is_sleeping",                 queue_tasks.is_sleeping()             },
            { "cors_proxy_enabled",          params.ui_mcp_proxy                   },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response(
                "This server does not support changing global properties. Start it with `--props`",
                ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({
            { "success", true }
        });
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto        res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()),
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(
                format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
                                      ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(
                    format_error_response("extra_context chunk must contain a \"text\" field with a string value",
                                          ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string",
                                                 ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra;  // default to empty array if it's not exist

        std::string                prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts =
            tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab, data.at("input_prefix"), data.at("input_suffix"), data.at("input_extra"), params.n_batch,
            params.n_predict, meta->slot_n_ctx, params.spm_infill,
            tokenized_prompts[0].get_tokens()  // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files;                            // dummy
        return handle_completions_impl(req, SERVER_TASK_TYPE_INFILL, data, files,
                                       TASK_RESPONSE_TYPE_NONE);  // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;  // dummy
        const json              body = json::parse(req.body);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body, files, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;  // dummy
        const json              body = json::parse(req.body);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body, files, TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;
        json                    body        = json::parse(req.body);
        json                    body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body_parsed, files,
                                       TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto       res  = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;
        json                    body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body_parsed, files,
                                       TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(
                format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json                    body =
            convert_transcriptions_to_chatcmpl(json::parse(req.body), meta->chat_params.tmpls.get(), req.files, files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body_parsed, files,
                                       TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;
        json                    body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body_parsed, files,
                                       TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto                    res = create_response();
        std::vector<raw_buffer> files;  // dummy, unused
        json                    body = json::parse(req.body);
        json                    data = oaicompat_chat_params_parse(body, meta->chat_params, files);
        res->ok({
            { "prompt", std::move(data.at("prompt")) }
        });
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server;  // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            { "models",
             { { { "name", meta->model_name },
                  { "model", meta->model_name },
                  { "modified_at", "" },
                  { "size", "" },
                  { "digest", "" },  // dummy value, llama.cpp does not support managing model file's hash
                  { "type", "model" },
                  { "description", "" },
                  { "tags", { "" } },
                  { "capabilities", meta->has_mtmd ? json({ "completion", "multimodal" }) : json({ "completion" }) },
                  { "parameters", "" },
                  { "details",
                    { { "parent_model", "" },
                      { "format", "gguf" },
                      { "family", "" },
                      { "families", { "" } },
                      { "parameter_size", "" },
                      { "quantization_level", "" } } } } } },
            { "object", "list"                             },
            { "data",
             {
                  get_model_info(),
              }                                            }
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto       res             = create_response();
        const json body            = json::parse(req.body);
        json       tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special   = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces   = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto & token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json        piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        { "id",    token      },
                        { "piece", piece_json }
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{
            { "tokens", std::move(tokens_response) }
        });
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto       res  = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content                   = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{
            { "content", std::move(content) }
        });
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents =
            json_value(body, "documents", json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(
                format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int) documents.size());

        // create and queue the task
        json   responses = json::array();
        auto & rd        = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp =
                    format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id          = rd.get_new_id();
                task.tokens      = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res;  // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(body, meta->model_name, responses, is_tei_format, documents, top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto       res  = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id       = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    std::string primary_backend     = "cpu";
    std::string primary_device_name = "CPU";
    json        devices_json        = json::array();
    size_t      dev_count           = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev) {
            const char *               name  = ggml_backend_dev_name(dev);
            const char *               desc  = ggml_backend_dev_description(dev);
            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            std::string                stype = "cpu";
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                stype = "gpu";
            } else if (dtype == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                stype = "accel";
            }
            std::string        reg_name = "cpu";
            ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
            if (reg) {
                const char * rname = ggml_backend_reg_name(reg);
                if (rname) {
                    std::string rstr(rname);
                    for (char & c : rstr) {
                        c = std::tolower(c);
                    }
                    reg_name = rstr;
                }
            }
            devices_json.push_back({
                { "name",        name ? name : "" },
                { "description", desc ? desc : "" },
                { "type",        stype            },
                { "backend",     reg_name         },
            });
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU ||
                dtype == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                primary_backend     = reg_name;
                primary_device_name = desc ? desc : (name ? name : "GPU");
            }
        }
    }

    return json{
        { "id",          meta->model_name    },
        { "aliases",     meta->model_aliases },
        { "tags",        meta->model_tags    },
        { "object",      "model"             },
        { "created",     std::time(0)        },
        { "owned_by",    "llamacpp"          },
        { "backend",     primary_backend     },
        { "device_name", primary_device_name },
        { "devices",     devices_json        },
        { "meta",
         {
              { "vocab_type", meta->model_vocab_type },
              { "n_vocab", meta->model_vocab_n_tokens },
              { "n_ctx", meta->slot_n_ctx },
              { "n_ctx_train", meta->model_n_ctx_train },
              { "n_embd", meta->model_n_embd_inp },
              { "n_params", meta->model_n_params },
              { "size", meta->model_size },
              { "ftype", meta->model_ftype },
              { "backend", primary_backend },
              { "device_name", primary_device_name },
          }                                  },
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto        res          = create_response();
    const json  request_data = json::parse(req.body);
    std::string filename     = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id                   = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto        res          = create_response();
    const json  request_data = json::parse(req.body);
    std::string filename     = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id                   = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto   res = create_response();
    auto & rd  = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id                  = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req,
                                                                            task_response_type      res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`",
                                         ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(
            format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type",
                                  ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE;  // "content" field is not OAI compatible
        prompt   = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64",
                                             ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json   responses = json::array();
    auto & rd        = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type       = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res;  // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd *>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD ?
                    format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64) :
                    json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab *     vocab,
                                                                         mtmd_context *          mctx,
                                                                         const server_http_req & req,
                                                                         task_response_type      res_type) {
    auto                    res = create_response();
    std::vector<raw_buffer> files;
    json                    body   = json::parse(req.body);
    bool                    is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            }
            break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body   = server_chat_convert_responses_to_chatcmpl(body);
            }
            break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            }
            break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
    json prompt      = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {
        { "input_tokens", static_cast<int64_t>(n_tokens) }
    };
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
