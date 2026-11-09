#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-impl.h"
#include "mtmd-model.h"

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <random>
#include <vector>

// !!! Internal header, to be used by mtmd only !!!

// a subcontext runs the graphs of one mtmd_model (one modality)
// it owns the backends and the compute buffers, the weights are owned by the model

enum mtmd_flash_attn_type {
    MTMD_FLASH_ATTN_TYPE_AUTO     = -1,
    MTMD_FLASH_ATTN_TYPE_DISABLED = 0,
    MTMD_FLASH_ATTN_TYPE_ENABLED  = 1,
};

struct mtmd_subctx_params {
    ggml_backend_dev_t device; // nullptr means CPU, see mtmd_select_device()
    std::shared_ptr<ggml_threadpool> threadpool; // CPU threadpool shared by the subcontexts, can be nullptr
    enum mtmd_flash_attn_type flash_attn_type;
    bool warmup;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

// the device the weights and the graphs are placed on, nullptr means CPU
ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device);

// nullptr if n_threads < 2 or the CPU backend has no threadpool support
std::shared_ptr<ggml_threadpool> mtmd_threadpool_init(int n_threads);

// compute the log-mel of n_frames frames, the waveform must already be padded and window must hold one whole frame
// out has the layout [n_frames, n_mel]; like the encoders, must not run at the same time as another pass on config.sched
struct mtmd_audio_mel_config;
bool mtmd_audio_log_mel(const mtmd_audio_mel_config & config, const float * samples, const float * window, int64_t n_frames, float * out);

enum mtmd_subctx_gen_process {
    MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN,
    MTMD_SUBCTX_GEN_PROCESS_GEN_CODE, // h_state to codes
    MTMD_SUBCTX_GEN_PROCESS_GEN_WAV,  // codes to raw PCM audio
    MTMD_SUBCTX_GEN_PROCESS_TTS,        // full utterance of codes to raw PCM audio
    MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE, // internal: mel + source stft to istft input
    MTMD_SUBCTX_GEN_PROCESS_TOKENIZE,   // raw PCM audio to semantic speech tokens
};

// persistent state of a causal audio encoder, for streaming input (one per stream)
// it holds the sliding window of past keys/values, so each encode() call only takes the new frames
struct mtmd_subctx_enc_state {
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
struct mtmd_subctx_gen_state {
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

struct mtmd_subctx_encode_params {
    int n_threads = 1;
    const mtmd_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    mtmd_subctx_gen_process gen_process = MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN;

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
    const mtmd_gen_ref * ref = nullptr; // TTS reference conditioning of the voice
    // TTS_VOCODE internal stage inputs
    const std::vector<float> * mel_in   = nullptr;
    const std::vector<float> * sstft_in = nullptr;
    int vocode_n_mel  = 0;
    int vocode_n_stft = 0;
    std::vector<float> * out_spec = nullptr;

    // audio gen: state of one sequence, encode() updates it in place
    struct mtmd_subctx_gen_state * gen_state = nullptr;

    // streaming audio input, updated in place by encode()
    struct mtmd_subctx_enc_state * enc_state = nullptr;
};

struct mtmd_subcontext {
    const mtmd_model & model;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    std::shared_ptr<ggml_threadpool> threadpool; // shared by the subcontexts of a model

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    mtmd_flash_attn_type flash_attn_type = MTMD_FLASH_ATTN_TYPE_AUTO;
    bool is_allocated = false;

    bool debug_output_embeddings = false;

    // for measuring memory usage
    std::map<ggml_backend_dev_t, size_t> mem_compute;

    bool support_batch = false;

    // the model must outlive the subcontext
    mtmd_subcontext(const mtmd_model & model, const mtmd_subctx_params & params);
    ~mtmd_subcontext();

    // reserve the compute buffers for the given batch
    void warmup(const mtmd_image_f32_batch & batch);

    bool encode(mtmd_subctx_encode_params * params);

    // TODO: remove encode_image() and always use batched version
    bool encode_image(int n_threads, const mtmd_image_f32 * img, std::vector<float> & out_vec);
    bool encode_batch(int n_threads, const mtmd_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

    // n_frames_max is the maximum number of encoder frames per encode() call, free the result with delete
    mtmd_subctx_enc_state * enc_state_init(int32_t n_frames_max);

    // seed UINT32_MAX means random, free the result with delete
    mtmd_subctx_gen_state * gen_state_init(uint32_t seed);

    // weights and compute buffers
    std::map<ggml_backend_dev_t, size_t> get_mem_usage() const;

private:
    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    mtmd_image_f32_batch get_dummy_batch() const;

    // only initialize backend buffers, but do not allocate them yet
    support_info_graph reserve_compute_meta(const mtmd_image_f32_batch & batch);
};
