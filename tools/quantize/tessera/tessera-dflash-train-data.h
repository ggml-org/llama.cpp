#pragma once

//
// tessera-dflash-train-data.h
//
// Input stage of the native DFlash / D-PACE drafter-training driver
// (tessera-train-dflash). Turns raw llama.tessera.dflash-block.v1 records
// (from `tessera-dataset --tessera-dataset-mode dflash`) into the
// (tokens, sparse labels, per-position CE weights) datapoints that the
// weighted-CE label fill consumes, where the D-PACE weight replaces the
// cross-entropy "1.0" at the llama-layer label fill (the additive change
// to src/llama-context.cpp).
//
// Datapoint contract - fixed by the llama-layer cross-entropy path
// (opt_epoch / opt_epoch_iter), not chosen here:
//   one datapoint = n_ctx = (n_dft + 1) tokens, where position 0 is the
//   anchor (committed last token) and positions 1..n_dft are the drafted
//   (MASK) positions.
//   tokens[0]                 = anchor (any in-vocab token; the model sees
//                                it but we never backprop from it)
//   tokens[1 + j]             = drafted token at position j (dataset record's
//                                drafted_tokens[j], the on-policy prefix)
//   sparse_labels[0]          = sentinel (e.g. 0); weight is 0 so it is
//                                never backprop'd
//   sparse_labels[1 + j]      = dataset.target_tokens[j] (the verifier's
//                                argmax at drafted position j)
//   weights[0]                = 0.0f  (anchor: not a target)
//   weights[1 + j]            = dataset.dpace_weights[j] (smoothed + normalized,
//                                baked in by the dataset prep)
//
// Block layout, off-by-one (from docs/tessera-dflash-training-design.md 0b):
// the dflash model's block_drafts (dflash.block_size in GGUF metadata) INCLUDES
// the anchor, so max drafted = block_drafts - 1. The dataset record's n_dft
// indexes DRAFTED positions only. The driver maps dataset pos j -> model
// pos j+1 to add the anchor, with a zero weight on the anchor (zero gradient,
// free).
//
// D-PACE math lives in tessera-dpace.{h,cpp}; the driver only reads the baked
// weights. Switching between D-PACE and DFlash-decay is a data-side swap
// (--weight-scheme dpace | decay) - the loss graph is unchanged. This is the
// design separation: loss-side change = llama-context.cpp label fill (one
// write site); algorithm-side = this module + the dataset prep.
//
// Pure logic: no llama/ggml dependency. nlohmann/json for parsing.
//

#include <cstdint>

// Decide whether a JSON line is a usable DFlash training example: a
// llama.tessera.dflash-block.v1 record whose n_dft equals block_size. Returns
// 1 if usable, 0 otherwise. Cheap (parse only, no weight read); used to size
// the dataset in a first pass before filling in a second pass.
int ts_dflash_train_line_usable(const char * line, int block_size);

// Parse one dflash-block.v1 line and, if usable, write a single training
// example into the caller's buffers:
//   out_tokens[0..block_size]              length block_size + 1 (int32 tokens)
//   out_labels_sparse[0..block_size]       length block_size + 1 (int32 sparse
//                                          target token ids; pos 0 is a
//                                          sentinel since its weight is 0)
//   out_weights[0..block_size]             length block_size + 1 (float; the
//                                          D-PACE / decay weight, pos 0 = 0)
// weight_scheme: 0 = dpace (use dpace_weights), 1 = decay (use decay_weights).
// Returns 1 if an example was written, 0 if the line is not usable (skipped),
// -1 on a parse error. Buffers are untouched when returning 0.
int ts_dflash_train_example_from_line(const char * line,
                                      int block_size,
                                      int weight_scheme,
                                      int32_t * out_tokens,
                                      int32_t * out_labels_sparse,
                                      float   * out_weights);

// Auto-detect the modal n_dft across a dflash-block.v1 file, so the driver
// can default --block-size to whatever the dataset prep run produced. Returns
// the modal n_dft (> 0), or -1 if no dflash-block.v1 record is found.
int ts_dflash_train_detect_block_size(const char * dflash_jsonl_path);
