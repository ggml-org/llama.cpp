#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "mtmd.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...) LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

// get a random marker; note: each time the server restarts, the marker will be different
const char * get_media_marker();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
    llama_pos pos_next(int64_t n_tokens = -1) const;

    // number of tokens with position < max_pos
    size_t size_up_to_pos(llama_pos max_pos) const;

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    // find next media chunk after idx
    // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
    std::pair<const mtmd::input_chunk_ptr *, size_t> find_next_media_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // same as push_back, but media chunks are stored as placeholders (no image/audio data)
    // only use this if the chunk will never be encoded again (e.g. it is already in the KV cache)
    void push_back_placeholder(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift
    const llama_tokens & get_tokens() const;

    llama_tokens get_text_tokens() const;

    std::vector<char> serialize() const;
    static server_tokens deserialize(const llama_tokens & packed, bool has_mtmd);

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // split the tokens into message spans, skipping over media chunks
    common_chat_msg_spans find_message_spans(const common_chat_msg_delimiters & delims) const;

    // check text token IDs and the mapping between media chunks and token ranges
    bool validate(const struct llama_context * ctx) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
// if is_placeholder is true, the media chunk will be treated as placeholder for counting tokens; the output tokens are not usable for actual inference (e.g. for submitting a task to server_queue)
server_tokens process_mtmd_prompt(mtmd_context * mctx, const std::string & prompt, const std::vector<raw_buffer> & files, bool is_placeholder = false);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool allow_video;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    std::string media_path;
    bool force_pure_content = false;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files);

// TODO: move it to server-task.cpp
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false);

// TODO: move it to server-task.cpp
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// stats and metrics
//

// shared between server_slot and server_task_result_*
struct server_slot_stats {
    uint64_t n_prompt_cached    = 0;
    uint64_t n_prompt_processed = 0;
    uint64_t n_gen              = 0;

    // speculative decoding stats
    // note: the per-position breakdown lives in server_slot, it is not needed in a task result
    uint64_t n_draft_tokens      = 0;
    uint64_t n_draft_accepted    = 0;
    uint64_t n_draft_verif_steps = 0;

    // these are absolute timestamps (in us)
    // note: must be signed - they are subtracted before the later ones are set
    int64_t t_start       = 0;
    int64_t t_prompt_last = 0;
    int64_t t_gen_last    = 0;

    // can only move one direction: start -> prompt -> gen
    void update_prompt_start() {
        GGML_ASSERT(t_start == 0);
        t_start = ggml_time_us();
    }
    void set_prompt_last(int64_t t_us) {
        GGML_ASSERT(t_start > 0);
        t_prompt_last = t_us;
    }
    void update_prompt_last() {
        set_prompt_last(ggml_time_us());
    }
    void update_gen_last() {
        GGML_ASSERT(t_prompt_last > 0);
        t_gen_last = ggml_time_us();
    }

    // these are time durations
    int64_t t_elapsed_us() const {
        return ggml_time_us() - t_start;
    }
    double t_prompt_ms() const {
        if (t_prompt_last == 0) {
            return 0.0; // the prompt is not processed yet
        }
        return (t_prompt_last - t_start) / 1000.0;
    }
    int64_t t_gen_us() const {
        if (t_gen_last == 0) {
            return 0; // the generation is not started yet
        }
        // clamp to 1 us, the first token can land in the same us as t_prompt_last
        return std::max<int64_t>(1, t_gen_last - t_prompt_last);
    }
    double t_gen_ms() const {
        return t_gen_us() / 1000.0;
    }

    // number of decode steps spent on generation
    // the first token is free, it comes from the logits of the last prompt batch
    uint64_t n_gen_steps() const {
        return n_gen > 0 ? n_gen - 1 : 0;
    }

    // other derived metrics
    // note: all of them return 0.0 if the divisor is not known yet
    double t_prompt_per_token_ms() const {
        return n_prompt_processed > 0 ? t_prompt_ms() / n_prompt_processed : 0.0;
    }
    double t_gen_per_token_ms() const {
        return n_gen_steps() > 0 ? t_gen_ms() / n_gen_steps() : 0.0;
    }
    double n_prompt_tps() const {
        const double t_ms = t_prompt_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_prompt_processed : 0.0;
    }
    double n_gen_tps() const {
        const double t_ms = t_gen_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_gen_steps() : 0.0;
    }

    // false if the slot never started, i.e. the task result carries no stats
    bool is_set() const {
        return t_start > 0;
    }

    json to_json() const;
};

// shared between server_context_impl and server_task_result_*
// unlike server_slot_stats, server_metrics is server-global and cumulative, not tied to a slot
struct server_metrics {
    int64_t t_start = 0;

    struct bucket {
        uint64_t count = 0; // number of tokens
        uint64_t steps = 0; // number of decode steps,
                            // this excludes first generated token (logits from prompt batch)
        uint64_t time  = 0; // in microseconds

        // the rate uses the decode steps, so that "free" tokens do not inflate it
        double n_per_second() const {
            return time > 0 ? (double) steps / (double) time * 1e6 : 0.0;
        }

        void add(uint64_t n, uint64_t n_steps, uint64_t t_us) {
            count += n;
            steps += n_steps;
            time  += t_us;
        }
    };

    // these are reset by reset_bucket(), only the rate is read from them
    bucket prompt_bucket;
    bucket predict_bucket;

    // metrics below are cumulative since the server started
    bucket prompt; // only processed tokens, cached ones are counted separately below
    bucket predict;

    // tokens reused from the cache need no decode, so they only have a count
    uint64_t n_prompt_cached = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_decode     = 0;
    uint64_t n_busy_slots = 0;

    uint64_t n_draft_tokens      = 0; // Total draft tokens generated
    uint64_t n_draft_accepted    = 0; // Draft tokens actually accepted
    uint64_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<uint64_t> n_accepted_per_pos; // Accepted tokens per draft position

    void init() {
        t_start = ggml_time_us();
    }

    void reset_bucket() {
        prompt_bucket  = {};
        predict_bucket = {};
    }

    void add_prompt(uint64_t n_tokens, uint64_t t_us) {
        prompt       .add(n_tokens, n_tokens, t_us);
        prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    void add_prompt_cached(uint64_t n_tokens) {
        n_prompt_cached += n_tokens;
    }
};

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx, size_t n_top);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc);

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct server_pipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    // 0 = unbounded (default)
    // > 0, write() drops the oldest item once the queue is full
    size_t max_size = 0;

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
    // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
    bool read(T & output, const std::function<bool()> & should_stop, bool close_on_stop = true) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop && should_stop()) { // a null should_stop means "never stop"
                if (close_on_stop) {
                    close_read(); // signal broken pipe to writer
                }
                return false; // cancelled / deadline reached
            }
            cv.wait_for(lk, poll_interval);
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        if (max_size > 0) {
            while (queue.size() >= max_size) {
                queue.pop(); // drop oldest to stay bounded
            }
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};
