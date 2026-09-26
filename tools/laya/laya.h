#pragma once

// Self-contained laya model: a ModernBERT encoder + a typed decision head for
// non-autoregressive decision making (choice / score / noul).
//
// This is a standalone ggml implementation following the tools/mtmd pattern:
// the model loads its own GGUF via gguf_init_from_file and builds the
// inference graph with ggml directly. It links the public ggml/llama
// libraries, but calls no llama inference / graph / batch API.
//
// WARNING: This API is experimental and subject to change.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// number of marker (option) slots per sequence; must be >= max options per
// question. The reference collates marker_pos / marker_mask to this width.
#define LAYA_MAX_MARKERS 16

struct laya_hparams {
    int32_t n_embd            = 0;   // hidden size (768)
    int32_t n_layer           = 0;   // encoder layers (22)
    int32_t n_head            = 0;   // attention heads (12)
    int32_t n_embd_head       = 0;   // head dim (64)
    int32_t n_ff              = 0;   // encoder intermediate size (1152)
    int32_t n_vocab           = 0;   // token embedding rows
    int32_t n_qtype           = 0;   // question types (choice/score/noul = 3)
    int32_t marker_token_id   = 0;   // <mask> token id
    int32_t max_len           = 0;   // max sequence length
    int32_t head_max_len      = 0;   // max option-region length
    int32_t n_act             = 0;   // action head classes
    int32_t n_head_layers     = 0;   // decision head transformer layers
    int32_t n_swa             = 0;   // encoder sliding window (128)
    int32_t swa_pattern       = 0;   // global_attn_every_n_layers (3)
    float   norm_eps          = 1e-5f;
    float   rope_freq_base    = 10000.0f;
    float   rope_freq_base_swa = 10000.0f;
    std::vector<float> temperature;  // per-qtype temperature (padded to n_qtype)
};

// One batched forward pass: n_tokens packed tokens across n_seqs sequences.
// Sequences are packed without padding; separation is handled by an internal
// block-diagonal attention mask built from seq_id/positions.
struct laya_batch {
    int32_t           n_tokens = 0;   // total tokens across all sequences
    int32_t           n_seqs   = 0;   // number of sequences (questions)
    const int32_t *   tokens   = nullptr;   // [n_tokens] token ids
    const int32_t *   positions= nullptr;   // [n_tokens] per-token position
    const int32_t *   seq_id   = nullptr;   // [n_tokens] sequence id per token
    const int32_t *   qtype    = nullptr;   // [n_tokens] qtype id per token
    const int32_t *   marker_pos  = nullptr; // [n_markers_max, n_seqs] absolute token indices
    const int32_t *   marker_mask = nullptr; // [n_markers_max, n_seqs] 1 = valid marker
    const int32_t *   seq_start   = nullptr; // [n_seqs] first global token index of each seq
};

// Outputs of one forward pass.
struct laya_result {
    std::vector<float> logits;       // [n_markers_max, n_seqs] scorer logits at marker positions
    std::vector<float> act_logits;   // [n_act, n_seqs] raw action head logits
    int32_t n_markers_max = 0;
    int32_t n_seqs        = 0;
    int32_t n_act         = 0;
};

struct laya_model;
struct laya_context;

// load a laya GGUF (F16 or quantized; dequantization happens inside ggml mul_mat)
laya_model * laya_model_load_from_file(const char * fname);

void laya_model_free(laya_model * model);

const laya_hparams & laya_model_hparams(const laya_model * model);

// ---- tokenizer (modern-bert / mmBERT: Metaspace pre-tokenizer + BPE) ----
// The reference checkpoint uses a Metaspace-style pre-tokenizer with the
// mmBERT BPE vocabulary, which llama.cpp's built-in GPT-2 pre-tokenizer cannot
// reproduce for this vocab. This is a self-contained port of the HF fast
// tokenizer: normalize " " -> U+2581, prepend U+2581, split into words, then
// greedy byte-level BPE with the GGUF vocab/merges. Matches
// `tokenizer(text, add_special_tokens=False)["input_ids"]` from the reference.

// tokenize with add_special_tokens=False (no <bos>/<eos>)
std::vector<int32_t> laya_tokenize(const laya_model * model, const std::string & text);

// special token ids from the GGUF tokenizer metadata
int32_t laya_vocab_bos (const laya_model * model);
int32_t laya_vocab_sep (const laya_model * model);
int32_t laya_vocab_mask(const laya_model * model);

laya_context * laya_init(const laya_model * model, int n_threads);


void laya_free(laya_context * ctx);

// run one forward pass; returns 0 on success, non-zero on failure
int laya_encode(laya_context * ctx, const laya_batch & batch, laya_result & result);
