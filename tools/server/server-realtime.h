#pragma once

#include "server-http.h"
#include "server-queue.h"
#include "server-task.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <deque>
#include <memory>
#include <set>
#include <string>
#include <vector>

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

// the input audio of a realtime session: PCM16 mono, resampled to the model rate
// the rate defaults to 24 kHz like OpenAI's audio/pcm, session.audio.input.format.rate sets it before the first audio
struct server_realtime_audio_in {
    explicit server_realtime_audio_in(int32_t model_rate) : model_rate(model_rate) {}

    int32_t rate = 24000;

    // reads the rate of a session.update, false with err if it can't be used
    bool update(const json & session, std::string & err);

    // PCM16 bytes to PCM F32 at the model rate
    std::vector<float> decode(const std::string & pcm);

private:
    struct resampler_deleter {
        void operator()(mtmd_helper_resampler * rs) const { mtmd_helper_resampler_free(rs); }
    };

    int32_t model_rate;
    bool    started = false;
    std::unique_ptr<mtmd_helper_resampler, resampler_deleter> resampler;
};

// a realtime conversation over websocket, for models that answer after the audio (e.g. LFM3-Audio):
// the audio is prefilled as it arrives, responses follow output_modalities or the model's own switching
void server_realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn);

// one control of a realtime conversation
struct server_realtime_op {
    server_control_action   action;
    std::string             text;
    std::vector<float>      audio;
    server_control_response response;
};

// the conversation of a slot that holds a realtime session (SERVER_TASK_TYPE_REALTIME), it runs on the server thread
// the controls run in order; the sequence is decoded by its own calls, not in the shared batch
struct server_realtime_slot {
    server_realtime_slot(llama_context * ctx, mtmd_context * mctx, llama_seq_id seq_id, int32_t n_ctx,
                         const common_params_sampling & sampling, server_response & queue_results, int id_task, int index);

    void push(server_realtime_op && op);
    void cancel(); // stops the response being generated

    // runs the queued controls, or one step of the response; false on error, see error and error_type
    bool update();

    // nothing to run until the next control
    bool is_idle() const;

    std::string error;
    enum error_type error_type = ERROR_TYPE_SERVER;

private:
    llama_context *        ctx;
    const llama_model *    model;
    const llama_vocab *    vocab;
    mtmd_context *         mctx;
    llama_seq_id           seq_id;
    int32_t                n_ctx;
    common_params_sampling sampling;
    server_response &      queue_results;
    int                    id_task;
    int                    index;

    std::deque<server_realtime_op> ops;
    llama_pos   n_past     = 0;     // positions of the conversation in the sequence
    bool        responding = false;
    bool        cancelled  = false;
    int32_t     n_predict  = 0;
    int32_t     n_decoded  = 0;
    llama_token sampled    = LLAMA_TOKEN_NULL; // next text token of the response, sampled right after the slot's own decode

    mtmd::audio_stream_ptr stream;   // the audio of the open turn
    mtmd_helper::gen_audio gen;      // set if the model generates audio
    std::vector<float>     h_state;  // hidden state that the gen-audio pipeline steps from
    common_sampler_ptr     smpl;
    std::set<llama_token>  preserved; // special tokens of the response that go out as text (tool calls)

    bool fail(const std::string & msg, enum error_type type = ERROR_TYPE_SERVER);
    void send_result(std::string text, std::vector<float> audio, bool response_done = false);
    bool decode(const llama_token * tokens, const float * rows, int32_t n, bool logits_last);
    bool audio_rows();
    void sample_next();
    std::string piece(llama_token tok) const;
    bool response_start(const server_realtime_op & op);
    bool response_step();
};
