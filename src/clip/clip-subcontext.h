#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <map>
#include <memory>
#include <random>
#include <vector>

// libllama internal: runs the graphs of one clip_model of a llama_context (see llama_context::clip_exec)

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

// a CPU-only scheduler for the preprocessing graphs (log-mel), apart from the scheduler of the encoder
struct clip_cpu_compute {
    ggml_backend_t       backend = nullptr;
    ggml_backend_sched_t sched   = nullptr;

    explicit clip_cpu_compute(int n_threads);
    ~clip_cpu_compute();

    clip_cpu_compute(const clip_cpu_compute &) = delete;
    clip_cpu_compute & operator=(const clip_cpu_compute &) = delete;
};

// compute the log-mel of n_frames frames, the waveform must already be padded and window must hold one whole frame
// out has the layout [n_frames, n_mel]; must not run at the same time as another pass
struct clip_audio_mel_config;
bool clip_audio_log_mel(const clip_audio_mel_config & config, const float * samples, const float * window, int64_t n_frames, float * out);

enum clip_subctx_gen_process {
    CLIP_SUBCTX_GEN_PROCESS_GEN_UNKNOWN,
    CLIP_SUBCTX_GEN_PROCESS_GEN_CODE, // h_state to codes
    CLIP_SUBCTX_GEN_PROCESS_GEN_WAV,  // codes to raw PCM audio
    CLIP_SUBCTX_GEN_PROCESS_TTS,        // full utterance of codes to raw PCM audio
    CLIP_SUBCTX_GEN_PROCESS_TTS_VOCODE, // internal: mel + source stft to istft input
    CLIP_SUBCTX_GEN_PROCESS_TOKENIZE,   // raw PCM audio to semantic speech tokens
    CLIP_SUBCTX_GEN_PROCESS_LATENT_IN,  // internal: flow latent rows to backbone embd rows
    CLIP_SUBCTX_GEN_PROCESS_LATENT_OUT, // internal: backbone hidden rows to flow velocity rows
};

// persistent state of a causal audio encoder, for streaming input (one per stream)
// it holds the sliding window of past keys/values, so each encode() call only takes the new frames
struct clip_subctx_enc_state {
    int32_t n_ctx  = 0; // ring slots: attention window + frames per call
    int32_t n_past = 0; // absolute position of the next frame

    std::vector<int32_t> slot_pos;  // absolute position held by each slot, -1 if empty
    std::vector<float>   stem_tail; // last conv stem output frame, empty on a cold start

    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> k; // per layer [n_embd_kv, n_ctx]
    std::vector<ggml_tensor *> v;

    // lfm3a: same order as list_lfm3a_state_slots(), F32 [ne0, ne1], updated in place
    std::vector<ggml_tensor *> slots;

    // nemotron transcribe: the window's normalized log-mel, cache and FIFO rows in; its speaker rows and predictions out
    std::vector<float> mel_norm;
    std::vector<float> spk_prev;
    std::vector<float> spk_chunk, spk_preds;
};

// per-sequence state of audio generation: RNG and the slots kept between GEN_WAV calls
struct clip_subctx_gen_state {
    std::mt19937 rng;

    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> slots; // same order as list_gen_state_slots(), F32 [ne0, ne1]

    // clears the slots and keeps the RNG stream
    void reset() {
        if (buf) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
};

struct clip_subctx_encode_params {
    int n_threads = 1;
    const clip_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    clip_subctx_gen_process gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_UNKNOWN;

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    int32_t code0 = 0; // semantic code sampled by the backbone
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> * out_codes = nullptr; // this frame's 16 sampled codes
    std::vector<float> * out_feats = nullptr; // continuous counterpart of out_codes
    float   temp = 0.0f;                      // sampling temperature, noise scale for flow-matching decoders
    bool * out_is_eos = nullptr;

    // TOKENIZE: speech embedding rows of the produced codes
    std::vector<float> * out_code_embd = nullptr;

    // GEN_WAV
    const std::vector<int32_t> * codes = nullptr;     // this frame's 16 RVQ codes
    const std::vector<float> *   feats = nullptr;     // continuous counterpart of codes
    std::vector<float> * out_audio = nullptr;         // decoded PCM samples, F32
    const clip_gen_ref * ref = nullptr; // TTS reference conditioning of the voice
    // TTS_VOCODE internal stage inputs
    const std::vector<float> * mel_in   = nullptr;
    const std::vector<float> * sstft_in = nullptr;
    int vocode_n_mel  = 0;
    int vocode_n_stft = 0;
    std::vector<float> * out_spec = nullptr;

    // audio gen: state of one sequence, encode() updates it in place
    struct clip_subctx_gen_state * gen_state = nullptr;

    // streaming audio input, updated in place by encode()
    struct clip_subctx_enc_state * enc_state = nullptr;
};

struct clip_subcontext {
    const clip_model & model;

    int max_nodes = 8192;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;

    bool debug_output_embeddings = false;

    bool support_batch = false;

    bool input_ok = true; // result of the last set_inputs()

    // where enc_state_init() and gen_state_init() allocate when no buffer type is given
    ggml_backend_buffer_type_t state_buft = nullptr;

    // graph under construction, owned by the llm_graph_context that builds it
    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    // runs one graph of this model: build(), set_inputs(), compute, read_outputs()
    std::function<bool(clip_subctx_encode_params *)> exec;

    // model must outlive the subcontext
    clip_subcontext(const clip_model & model, bool flash_attn);

    bool encode(clip_subctx_encode_params * params);

    // the steps of one exec(), ctx0 and gf must be set for build()
    ggml_cgraph * build(const clip_subctx_encode_params * params);
    bool set_inputs(ggml_cgraph * gf, clip_subctx_encode_params * params);
    // embeddings is the last node of the graph as built, nullptr if params->out_embd is not set
    bool read_outputs(ggml_cgraph * gf, ggml_tensor * embeddings, clip_subctx_encode_params * params);

    // TODO: remove encode_image() and always use batched version
    bool encode_image(int n_threads, const clip_image_f32 * img, std::vector<float> & out_vec);
    bool encode_batch(int n_threads, const clip_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

    // n_frames_max is the maximum number of encoder frames per encode() call, free the result with delete
    clip_subctx_enc_state * enc_state_init(int32_t n_frames_max, ggml_backend_buffer_type_t buft = nullptr);

    // seed UINT32_MAX means random, free the result with delete
    clip_subctx_gen_state * gen_state_init(uint32_t seed, ggml_backend_buffer_type_t buft = nullptr);

    // the batch the compute buffers are reserved for
    // placeholder: shapes only, for building a graph that is not computed (reservation)
    clip_image_f32_batch get_dummy_batch(bool placeholder = false) const;
};
