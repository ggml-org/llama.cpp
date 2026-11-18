#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// the callback of libllama's own messages (llama_log_get() returns the ggml one, which llama_log_set() sets too)
LLAMA_API void llama_log_get_llama(ggml_log_callback * log_callback, void ** user_data);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);

//
// media (mmproj) models: encoder context, preprocessing, layout and media rows
// staging API for retiring libmtmd, see docs/mtmd-libllama-api-changes.md
//

// context role for mmproj models: runs the vision / audio input towers
// n_ubatch of such a context bounds the output rows of one encode, and sets its compute reservation
static const enum llama_context_type LLAMA_CONTEXT_TYPE_ENCODER = (enum llama_context_type) 16;

// context role for mmproj models with an audio generation stage: per-frame head on backbone rows (code predictor)
// a batch entry carries a token (the code the backbone sampled) and a state embedding (the backbone hidden state)
// the codes are sampled in the graph by the sampler of the sequence (llama_set_sampler), greedy without one
// outputs: llama_get_sampled_tokens_ith() for the codes, llama_get_embeddings_ith() for the feedback embedding
static const enum llama_context_type LLAMA_CONTEXT_TYPE_GEN_CODE = (enum llama_context_type) 17;

// context role for mmproj models with an audio generation stage: codes to PCM (codec decoder, vocoder)
// one entry per frame (llama_batch_ext_add_codes), the entries of a call are consecutive frames of one sequence
// the decoder state of each sequence (conv left-contexts, k/v window, position) is kept in the context memory;
// llama_memory_seq_rm(mem, seq, -1, -1) resets it; output row i (llama_get_embeddings_ith) is the PCM of frame i
static const enum llama_context_type LLAMA_CONTEXT_TYPE_GEN_WAV = (enum llama_context_type) 18;

LLAMA_API bool llama_model_has_context_type(const struct llama_model * model, enum llama_context_type type);

// width of the embedding rows the context outputs (it depends on the context type for mmproj models)
LLAMA_API int32_t llama_n_embd_out(const struct llama_context * ctx);

// an entry carrying one code per codebook (GEN_WAV context); returns the batch index or llama_batch_ext_add() errors
LLAMA_API int32_t llama_batch_ext_add_codes(struct llama_batch_ext * batch, llama_seq_id seq_id, const llama_token * codes, int32_t n_codes);

// several sampled tokens for output row i (the codes of one audio frame); returns their count, copies up to n_max into out
LLAMA_API int32_t llama_get_sampled_tokens_ith(struct llama_context * ctx, int32_t i, llama_token * out, int32_t n_max);

struct llama_media_proc;
struct llama_media_layout;
struct llama_media;

enum llama_media_type {
    LLAMA_MEDIA_TYPE_IMAGE = 0,
    LLAMA_MEDIA_TYPE_AUDIO = 1,
};

struct llama_media_proc_params {
    int32_t n_span_max; // largest atomic span the decoder takes in one ubatch, 0 = no limit
};

LLAMA_API struct llama_media_proc_params llama_media_proc_default_params(void);

// preprocessing and layout for an (encoder context, decoder model) pair; model_dec may be null for the queries only
// ctx_enc is an ENCODER context, or a GEN context of the same mmproj: it encodes the media of its pipeline
// (speaker references) on the compute buffers of the generation stages
// the user limits of the image token count are kv overrides of the mmproj model: clip.vision.image_min_tokens and
// clip.vision.image_max_tokens (int)
// audio preprocessing computes the log-mel on a CPU scheduler of the proc, under its own lock
LLAMA_API struct llama_media_proc * llama_media_proc_init(
        struct llama_context     * ctx_enc,
        const struct llama_model * model_dec,
        struct llama_media_proc_params params);

LLAMA_API void    llama_media_proc_free       (struct llama_media_proc * proc);
LLAMA_API bool    llama_media_proc_has        (const struct llama_media_proc * proc, enum llama_media_type type);
LLAMA_API int32_t llama_media_proc_sample_rate(const struct llama_media_proc * proc); // -1 without audio
LLAMA_API int32_t llama_media_proc_n_frame_merge(const struct llama_media_proc * proc); // frames merged into one span, 1 = none

// the most rows one media of this type takes (-1: no fixed bound, 0: no tower), atomic: its span sits in one decoder ubatch
LLAMA_API int32_t llama_model_media_n_tokens_max(const struct llama_model * model, enum llama_media_type type, bool * atomic);
// the frame size the model takes for a video of n_frames frames of src_w x src_h; false if the model has no video sizing
LLAMA_API bool    llama_media_proc_video_frame_size(const struct llama_media_proc * proc, int32_t n_frames,
                                                    int32_t src_w, int32_t src_h, int32_t * w, int32_t * h);
// the text that opens and closes a video (empty strings if none), with special tokens
LLAMA_API void    llama_media_proc_video_markers(const struct llama_media_proc * proc, const char ** beg, const char ** end);
// how the frames of a video are sampled and marked in the prompt
enum llama_media_video_template {
    LLAMA_MEDIA_VIDEO_TEMPLATE_DEFAULT,   // frames as images
    LLAMA_MEDIA_VIDEO_TEMPLATE_QWEN3VL,   // Qwen3-VL: 2 fps, timestamps
    LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3,      // LFM3-VL
    LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3_OMNI, // LFM3-VL, and the soundtrack in pieces after the frames, see llama_media_layout_begin_track()
};
LLAMA_API enum llama_media_video_template llama_media_proc_video_template(const struct llama_media_proc * proc);

// debugging of model ports, unstable: run the encoder on preprocessed input and print its rows
// image: nx * ny * 3 normalized RGB floats; audio: nx samples broadcast to every mel bin
LLAMA_API int32_t llama_media_debug_encode    (const struct llama_media_proc * proc, enum llama_media_type type,
                                               int32_t nx, int32_t ny, const float * data, size_t n_data);
// debugging of model ports, unstable: print the preprocessed input; image: nx * ny RGB bytes, audio: nx PCM samples
LLAMA_API int32_t llama_media_debug_preprocess(const struct llama_media_proc * proc, enum llama_media_type type,
                                               int32_t nx, int32_t ny, const void * data);

// builder: pieces in prompt order; pos_0 is the position of the first piece
// one layout per prompt: images are numbered and aligned within it
// data == nullptr gives a placeholder: exact counts, cannot be encoded
// returns 0 on success, 1 if the model does not take this media, 2 on invalid input
LLAMA_API struct llama_media_layout * llama_media_layout_init(const struct llama_media_proc * proc, llama_pos pos_0);
LLAMA_API void    llama_media_layout_free      (struct llama_media_layout * lay);
LLAMA_API int32_t llama_media_layout_add_tokens(struct llama_media_layout * lay, const llama_token * tokens, size_t n);
LLAMA_API int32_t llama_media_layout_add_image (struct llama_media_layout * lay, uint32_t nx, uint32_t ny, const uint8_t * rgb, const char * id);
// up to llama_media_proc_n_frame_merge() frames of equal size that become one span; is_video: frames of a video
LLAMA_API int32_t llama_media_layout_add_frames(struct llama_media_layout * lay, uint32_t nx, uint32_t ny,
                                                const uint8_t * const * rgb, int32_t n_frames, bool is_video, const char * id);
// mono PCM at llama_media_proc_sample_rate()
LLAMA_API int32_t llama_media_layout_add_audio (struct llama_media_layout * lay, const float * pcm, size_t n_samples, const char * id);

// an audio track cut in pieces that go into the prompt one by one, each in its own place (the soundtrack of a video,
// each piece after its frames, LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3_OMNI only: returns 1 for other models); it is encoded
// once, as one clip, and every piece takes its rows of it, without audio markers
// begin_track copies the pieces; pcm[i] can be NULL, the media of the track are then placeholders (laid out, not
// encodable); add_track_piece lays out the next one; a piece too short for a row adds no media
LLAMA_API int32_t llama_media_layout_begin_track    (struct llama_media_layout * lay, const float * const * pcm, const size_t * n_samples, int32_t n_pieces);
LLAMA_API int32_t llama_media_layout_add_track_piece(struct llama_media_layout * lay, const char * id);

// realtime audio input: audio fed as it arrives, the encoder state is kept between calls
// how text tokens are mixed into the rows is a model detail hidden inside the stream
// a stream encodes on the ENCODER context of its proc; streams of one proc are serialized
struct llama_audio_stream;

struct llama_audio_stream_params {
    int32_t step_ms; // audio encoded per encoder call, rounded up to the model step (default: 320)

    // tokens that open the stream, they go out with the first rows (default: none)
    // with the prompt of its "asr" chat template, LFM3-Audio transcribes live
    const llama_token * prompt;
    size_t              n_prompt;
};

LLAMA_API struct llama_audio_stream_params llama_audio_stream_default_params(void);

LLAMA_API bool    llama_media_proc_has_audio_stream(const struct llama_media_proc * proc);

// returns nullptr on failure
LLAMA_API struct llama_audio_stream * llama_audio_stream_init(const struct llama_media_proc * proc, struct llama_audio_stream_params params);
LLAMA_API void    llama_audio_stream_free (struct llama_audio_stream * s);
// PCM F32 at llama_media_proc_sample_rate(); 0 on success, 1 if the stream has ended
LLAMA_API int32_t llama_audio_stream_push (struct llama_audio_stream * s, const float * pcm, size_t n);
// pad the stream so that all pending rows can be produced; end: no more push() is accepted
LLAMA_API int32_t llama_audio_stream_flush(struct llama_audio_stream * s, bool end);
// the next decoder rows; last_token is the token sampled from the previous row, LLAMA_TOKEN_NULL before the first
// returns > 0 rows ready (llama_audio_stream_get_output_embd), 0 more audio is needed, -1 end of stream, -2 error
LLAMA_API int32_t llama_audio_stream_step (struct llama_audio_stream * s, llama_token last_token);
// rows of the last step, n_rows * llama_model_n_embd_inp(model_dec) floats
LLAMA_API const float * llama_audio_stream_get_output_embd(const struct llama_audio_stream * s);
// the rows of the last step start a new sequence: the caller drops its sequence before decoding them
LLAMA_API bool    llama_audio_stream_is_seq_start     (const struct llama_audio_stream * s);
// audio rows whose text has been sampled (prefix rows included)
LLAMA_API int64_t llama_audio_stream_get_n_rows       (const struct llama_audio_stream * s);
// audio rows up to the end of the last flush
LLAMA_API int64_t llama_audio_stream_get_n_rows_flushed(const struct llama_audio_stream * s);

// audio generation pipelines of an mmproj model, moved from mtmd: one step of one sequence per batch entry
// a GEN_CODE or GEN_WAV context of the mmproj runs both stages; its memory holds the decoder state of each sequence
// (llama_memory_seq_rm() of a whole sequence is a cold start)
enum llama_gen_audio_type {
    LLAMA_GEN_AUDIO_TYPE_NONE, // not supported
    LLAMA_GEN_AUDIO_TYPE_QWEN3TTS,
    LLAMA_GEN_AUDIO_TYPE_POCKETTTS,
    LLAMA_GEN_AUDIO_TYPE_SOPRANO,
    LLAMA_GEN_AUDIO_TYPE_LFM3A,
    LLAMA_GEN_AUDIO_TYPE_CHATTERBOX,
    LLAMA_GEN_AUDIO_TYPE_DRAMA3,
    LLAMA_GEN_AUDIO_TYPE_ZONOS3,
    LLAMA_GEN_AUDIO_TYPE_YUE2,
    LLAMA_GEN_AUDIO_TYPE_FISH_S2,
};

struct llama_gen_info {
    enum llama_gen_audio_type type;
    int32_t sample_rate;        // in Hz
    int32_t n_channels;         // 1 = mono, 2 = stereo (llama_gen_out::audio is interleaved)
    const char * model_variant; // name of the weight variant, can be nullptr
    int32_t n_voices;           // predefined voices, see llama_model_gen_voice_name()
};

enum llama_gen_stage {
    LLAMA_GEN_STAGE_CODE, // backbone hidden state to codes / features and the feedback row
    LLAMA_GEN_STAGE_WAV,  // codes / features to PCM
};

struct llama_gen_inp {
    enum llama_gen_stage type;

    // LLAMA_GEN_STAGE_CODE
    int32_t code0;      // codebook 0 entry sampled by the backbone
    const float * embd; // backbone hidden state, llama_model_n_embd_inp(backbone) values
    int32_t top_k;
    float   top_p;
    float   temp;           // sampling temperature, or noise scale for flow-matching decoders (0 = no noise)
    float   penalty_repeat; // repetition penalty of the sampled codes, <= 1 is off
    int32_t penalty_last_n; // frames the repetition penalty looks back on

    // LLAMA_GEN_STAGE_WAV: codes (discrete) or feats (continuous), depending on the pipeline
    // lfm3a CODE also takes codes: the codes revealed so far, -1 for a masked codebook
    // zonos3 WAV: -1 for a residual code not known yet
    const int32_t * codes;
    size_t          n_codes;
    const float   * feats;
    size_t          n_feats;

    // yue2 CODE: n_codes is the frame count; the backbone state of the backbone_n_past positions
    const uint8_t * backbone_state;
    size_t          backbone_state_size;
    int32_t         backbone_n_past;

    // flow-matching CODE: steps of the ODE solver, <= 0 = model default
    int32_t n_steps;
};

// owned by the context, valid until its next llama_process()
struct llama_gen_out {
    // LLAMA_GEN_STAGE_CODE
    const int32_t * codes;
    size_t          n_codes;
    const float   * feats; // continuous counterpart of codes
    size_t          n_feats;
    const float   * embd;  // feedback row for the backbone (ZONOS3: the logits of the codebook)
    size_t          n_embd;
    bool is_eos;           // pipelines with an EOS head in the mmproj

    // LLAMA_GEN_STAGE_WAV
    const float * audio;
    size_t        n_samples; // over all channels
};

LLAMA_API struct llama_gen_info llama_model_gen_info(const struct llama_model * model);
// defaults tuned for the pipeline
LLAMA_API struct llama_gen_inp  llama_gen_inp_default(const struct llama_model * model);
// a named tensor of the gen tower as F32, returns the element count (0 if not found), out may be null
LLAMA_API size_t        llama_model_gen_read_tensor(const struct llama_model * model, const char * name, float * out, size_t n_max);
// the idx-th predefined voice, idx 0 is the default; nullptr out of range
LLAMA_API const char *  llama_model_gen_voice_name (const struct llama_model * model, int32_t idx);
// a predefined voice as an encoded AUDIO media (backbone rows and the gen-side reference), matched case-insensitively;
// a voice the chat template names (lfm3a) has no rows
LLAMA_API struct llama_media * llama_media_init_voice(const struct llama_model * model, const char * name);

// one step of seq_id; returns the batch index, or -1 full, -3 bad seq_id, -6 the context does not run gen steps
LLAMA_API int32_t llama_batch_ext_add_gen(struct llama_batch_ext * batch, llama_seq_id seq_id, const struct llama_gen_inp * inp);
// the output of batch index i of the last llama_process()
LLAMA_API bool    llama_get_gen_out_ith (struct llama_context * ctx, int32_t i, struct llama_gen_out * out);
// the RNG of a sequence; UINT32_MAX = random; resets the sequence
LLAMA_API void    llama_gen_set_seed    (struct llama_context * ctx, llama_seq_id seq_id, uint32_t seed);
// a new utterance on a sequence: clears its decoder state, keeps its RNG position and its reference
// (llama_memory_seq_rm of the whole sequence drops all three)
LLAMA_API void    llama_gen_reset       (struct llama_context * ctx, llama_seq_id seq_id);
// the gen-side reference (speaker) of a sequence, from an encoded media of the same mmproj; non-zero if it has none
LLAMA_API int32_t llama_gen_set_ref     (struct llama_context * ctx, llama_seq_id seq_id, const struct llama_media * media);

LLAMA_API size_t              llama_media_layout_n_pieces  (const struct llama_media_layout * lay);
LLAMA_API const llama_token * llama_media_layout_get_tokens(const struct llama_media_layout * lay, size_t i, size_t * n); // nullptr for media
LLAMA_API struct llama_media * llama_media_layout_get_media(const struct llama_media_layout * lay, size_t i);             // new reference

LLAMA_API enum llama_media_type llama_media_get_type      (const struct llama_media * m);
LLAMA_API uint32_t              llama_media_n_tokens      (const struct llama_media * m); // KV cells it takes
LLAMA_API llama_pos             llama_media_n_pos         (const struct llama_media * m); // positions it advances
LLAMA_API bool                  llama_media_is_atomic     (const struct llama_media * m); // must sit in one decoder ubatch
LLAMA_API const char *          llama_media_get_id        (const struct llama_media * m);
LLAMA_API bool                  llama_media_is_placeholder(const struct llama_media * m);
LLAMA_API bool                  llama_media_is_encoded    (const struct llama_media * m);
LLAMA_API struct llama_media *  llama_media_ref           (struct llama_media * m);
LLAMA_API void                  llama_media_free          (struct llama_media * m); // drops one reference
LLAMA_API void                  llama_media_release_rows  (struct llama_media * m); // keep the handle, free the rows
// read back encoded rows; returns the row width, 0 if not encoded
LLAMA_API int64_t               llama_media_get_rows      (const struct llama_media * m, int64_t row0, int64_t n_rows, float * out);
// TRANSITIONAL: decoder positions of one row (n_pos values, [t, y, x, z] for M-RoPE), for callers that decode host rows
// with llama_batch_ext_add_embd instead of llama_batch_ext_add_media
LLAMA_API bool                  llama_media_get_row_pos   (const struct llama_media * m, llama_pos pos_0, uint32_t row, llama_pos * pos, uint32_t n_pos);
// the token grid of an image as the clip engine counts it (M-RoPE: tokens per row and rows; else n_tokens x 1); false otherwise
LLAMA_API bool                  llama_media_get_grid      (const struct llama_media * m, uint32_t * nx, uint32_t * ny);

// meaning depends on the context type of the batch:
// context of an mmproj (ENCODER, GEN): queue the media for encoding (row0, n_rows and pos_0 are ignored); llama_process(ENCODE) writes the rows into the media
// decoder context: add rows [row0, row0 + n_rows) of an encoded media to seq_id, the media starting at pos_0 (n_rows = 0: all)
//                  libllama sets the positions and the attention span of the rows
// returns the batch index of the first row, or: -1 full, -3 bad seq_id, -4 not encoded, -5 would split an atomic media,
//                                               -6 the context cannot take this media
LLAMA_API int32_t llama_batch_ext_add_media(struct llama_batch_ext * batch, llama_seq_id seq_id,
                                            struct llama_media * media, uint32_t row0, uint32_t n_rows, llama_pos pos_0);
