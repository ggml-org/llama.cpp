#pragma once

#include "server-http.h"
#include "server-queue.h"

#include <string>

struct server_context_meta;
struct common_params_sampling;
struct mtmd_context;
struct mtmd_helper_cap;
struct llama_vocab;

struct server_realtime_deps {
    server_queue &                 queue_tasks;
    server_response &              queue_results;
    const server_context_meta &    meta;
    const common_params_sampling & sampling;
    mtmd_context *                 mctx;
    const mtmd_helper_cap &        cap;
    const llama_vocab *            vocab;
};

// a realtime conversation over websocket, for models that answer after the audio (e.g. LFM3-Audio):
// the audio is prefilled as it arrives, responses follow output_modalities or the model's own switching
void server_realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn);
