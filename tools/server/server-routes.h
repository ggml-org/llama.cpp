#pragma once

#include "engine-context.h"
#include "server-http.h"

#include "json.h"

#include <memory>
#include <mutex>

// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, engine_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const engine_context & ctx_server) {
        this->meta = std::make_unique<engine_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t post_systemone;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            engine_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session = {});
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const engine_context_meta> meta;

    const common_params & params;
    engine_context & ctx_server;

    engine_queue & queue_tasks;
    engine_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    engine_metrics cached_metrics;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};
