#pragma once

//
// tessera-lk-train-data.h
//
// Input stage of the native LK drafter-training driver (tessera-train-lk):
// turns raw llama.tessera.spec.v1 acceptance traces (from llama-imatrix
// --telemetry-out --telemetry-topk K) into the (tokens, dense-labels)
// datapoints that GGML_OPT_LOSS_TYPE_LK consumes.
//
// Datapoint contract - fixed by the llama-layer LK path in src/llama-context.cpp
// (opt_epoch / opt_epoch_iter), not chosen here:
//   one datapoint = a token sequence of length n_ctx = block_size + 1, paired
//   with n_ctx dense verifier distributions of width n_vocab, laid out
//   position-major as labels[pos*n_vocab + tok].
//
// Why the input is the DRAFT trajectory: the trace's verifier_topk[j] is the
// verifier distribution conditioned on prime + drafted[0..j-1] (speculative
// decoding scores the drafter's own proposed tokens). The only input prefix
// consistent with that label is the draft prefix, so:
//   tokens[0]   = prime_token
//   tokens[j]   = drafted_tokens[j-1]          (j >= 1)
//   labels[j]   = densify(verifier_topk[j])    (j in 0..block_size)
// verifier_topk has exactly block_size + 1 entries, so every position is a real
// training position: no padding, no masking. This is on-policy distillation of
// the verifier into the drafter, and it is the unique label/input pairing with
// no train-time prefix mismatch.
//
// Pure logic: no llama/ggml dependency. nlohmann/json for parsing,
// tessera-lk-loss for densification.
//

#include <cstdint>

// Decide whether a JSON line is a usable LK training example: a
// llama.tessera.spec.v1 record whose drafted count equals block_size and that
// carries verifier_topk_tokens/probs for all block_size + 1 positions. Returns
// 1 if usable, 0 otherwise. Cheap (parse only, no densify); used to size the
// dataset in a first pass before densifying in a second pass.
int ts_lk_train_line_usable(const char * line, int block_size);

// Parse one llama.tessera.spec.v1 line and, if usable, write a single training
// example into the caller's buffers:
//   out_tokens[0..block_size]              length block_size + 1 (int32 tokens)
//   out_labels[0..(block_size+1)*n_vocab)  position-major dense distributions
// Returns 1 if an example was written, 0 if the line is not usable (skipped),
// -1 on a parse or densify error. Buffers are untouched when returning 0.
int ts_lk_train_example_from_line(const char * line, int block_size, int n_vocab,
                                  int32_t * out_tokens, float * out_labels);

// Auto-detect the modal drafted count across a trace file, so the driver can
// default --block-size to whatever the collection run actually used. Returns
// the modal n_dft (> 0), or -1 if no llama.tessera.spec.v1 record is found.
int ts_lk_train_detect_block_size(const char * traces_path);
