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

LLAMA_API bool llama_model_has_context_type(const struct llama_model * model, enum llama_context_type type);

struct llama_media_proc;
struct llama_media_layout;
struct llama_media;

enum llama_media_type {
    LLAMA_MEDIA_TYPE_IMAGE = 0,
    LLAMA_MEDIA_TYPE_AUDIO = 1,
};

struct llama_media_proc_params {
    int32_t image_min_tokens; // <= 0: model default
    int32_t image_max_tokens; // <= 0: model default, capped by n_ubatch of the encoder context
    int32_t n_span_max;       // largest atomic span the decoder takes in one ubatch, 0 = no limit
};

LLAMA_API struct llama_media_proc_params llama_media_proc_default_params(void);

// preprocessing and layout for an (encoder context, decoder model) pair
// thread-safe, reads only constant state, allocates no compute resources
LLAMA_API struct llama_media_proc * llama_media_proc_init(
        const struct llama_context * ctx_enc,
        const struct llama_model   * model_dec,
        struct llama_media_proc_params params);

LLAMA_API void    llama_media_proc_free       (struct llama_media_proc * proc);
LLAMA_API bool    llama_media_proc_has        (const struct llama_media_proc * proc, enum llama_media_type type);
LLAMA_API int32_t llama_media_proc_sample_rate(const struct llama_media_proc * proc); // -1 without audio

// builder: pieces in prompt order; pos_0 is the position of the first piece
// rgb == nullptr gives a placeholder: exact counts, cannot be encoded
LLAMA_API struct llama_media_layout * llama_media_layout_init(const struct llama_media_proc * proc, llama_pos pos_0);
LLAMA_API void    llama_media_layout_free      (struct llama_media_layout * lay);
LLAMA_API int32_t llama_media_layout_add_tokens(struct llama_media_layout * lay, const llama_token * tokens, size_t n);
LLAMA_API int32_t llama_media_layout_add_image (struct llama_media_layout * lay, uint32_t nx, uint32_t ny, const uint8_t * rgb, const char * id);
// a video frame pair that the model merges into one span (temporal merge); rgb1 may be null for a lone last frame
LLAMA_API int32_t llama_media_layout_add_frames(struct llama_media_layout * lay, uint32_t nx, uint32_t ny,
                                                const uint8_t * rgb0, const uint8_t * rgb1, const char * id);

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

// meaning depends on the context type of the batch:
// ENCODER context: queue the media for encoding (row0, n_rows and pos_0 are ignored); llama_process(ENCODE) writes the rows into the media
// decoder context: add rows [row0, row0 + n_rows) of an encoded media to seq_id, the media starting at pos_0 (n_rows = 0: all)
//                  libllama sets the positions and the attention span of the rows
// returns the batch index of the first row, or: -1 full, -3 bad seq_id, -4 not encoded, -5 would split an atomic media,
//                                               -6 the context cannot take this media
LLAMA_API int32_t llama_batch_ext_add_media(struct llama_batch_ext * batch, llama_seq_id seq_id,
                                            struct llama_media * media, uint32_t row0, uint32_t n_rows, llama_pos pos_0);
