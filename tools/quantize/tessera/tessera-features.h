#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Offline trunk-feature capture format for DFlash drafter training.
//
// The DFlash drafter is feature-conditioned (EAGLE-style): its encoder
// consumes the trunk's hidden states at a set of target layers, fused into
// a row of n_target_layers * n_embd floats per token (layers concatenated
// in target_layer_ids order). Training the drafter offline therefore needs
// those hidden states captured once, while the trunk runs forward over a
// calibration corpus, and stored for reuse at train time (Path 1 in
// docs/tessera-dflash-training-design.md).
//
// This module owns ONLY the on-disk format, not the capture loop. The
// capture loop lives in tools/imatrix (a dedicated trunk-only forward pass);
// the training driver reads the files back. Keeping the format here means it
// is unit-testable without a model and shared by writer and reader.
//
// A capture is two files sharing a prefix:
//   <prefix>.bin   raw feature blob, row-major [n_tokens, n_layers*n_embd]
//   <prefix>.json  header: schema, shape, target layers, dtype, blob name
//
// Row t of the blob is token t's fused feature vector: layer order matches
// header.target_layers, so the encoder's FC input is a flat read of
// row_floats = n_layers * n_embd consecutive floats starting at
// t * row_floats * bytes_per_float.
//
// Schema: llama.tessera.features.v1

enum ts_features_dtype {
    TS_FEATURES_F32 = 0,
    // F16 / Q8_0 are the production storage choices (the dominant cost is
    // feature size). The header carries the dtype so the format is stable;
    // only F32 is implemented for now.
    TS_FEATURES_F16 = 1,
};

struct ts_features_header {
    int32_t n_tokens = 0;   // number of captured tokens (blob rows)
    int32_t n_embd   = 0;   // trunk hidden width per layer
    int32_t n_layers = 0;   // number of target layers fused per token
    std::vector<int32_t> target_layers;      // concatenation order
    ts_features_dtype dtype = TS_FEATURES_F32;

    // Window layout the capture was produced with. The trunk forward clears its
    // KV per window, so the first `warmup` tokens of each window are processed
    // for context but NOT emitted (their hidden states lack a full left
    // window). Each window therefore contributes (chunk_tokens - warmup)
    // emitted rows. chunk_tokens == 0 means "no window layout recorded"
    // (treat the blob as one contiguous sequence).
    int32_t chunk_tokens = 0;   // full decode window size, warmup included
    int32_t warmup       = 0;   // context-primer tokens skipped per window

    // Window advance. Overlap mode strides windows by `stride` tokens
    // (stride == chunk_tokens - warmup), so consecutive windows overlap by
    // `warmup` tokens: each window re-decodes the previous window's tail to
    // prime its KV, and the emitted rows form ONE contiguous corpus sequence
    // starting at token `warmup`. stride == 0 marks a legacy capture that
    // advanced by a full chunk_tokens per window and discarded a warmup prefix
    // per window (gappy output); the row mapping falls back to
    // stride = chunk_tokens for those files.
    int32_t stride = 0;

    int32_t row_floats() const { return n_layers * n_embd; }
    int32_t bytes_per_float() const { return dtype == TS_FEATURES_F16 ? 2 : 4; }
    int32_t rows_per_chunk() const { return chunk_tokens - warmup; }
    int32_t effective_stride() const { return stride > 0 ? stride : chunk_tokens; }
};

// Streaming writer. Feed one token at a time; the blob is written
// incrementally so memory stays flat regardless of corpus size. The JSON
// header is written at close() once the final token count is known.
struct ts_features_writer {
    // Open <prefix>.bin for writing and record the header fields. layer_order
    // is the target layer ids in the order they must be concatenated (i.e. the
    // drafter's target_layer_ids). Returns false on IO error or unsupported
    // dtype.
    bool open(const std::string & prefix,
              int32_t n_embd,
              const std::vector<int32_t> & layer_order,
              ts_features_dtype dtype = TS_FEATURES_F32);

    // Append one token from a pre-fused row of n_layers*n_embd floats.
    bool append_token(const float * fused);

    // Append one token from per-layer pointers: layers[i] points to n_embd
    // floats for target_layers[i]. The row is written layer by layer, so the
    // on-disk order matches layer_order. This is the capture-loop path.
    bool append_token_layers(const float * const * layers);

    int64_t n_tokens_written() const { return n_written; }

    // Flush the blob and write <prefix>.json. Returns false on IO error.
    bool close();

    ts_features_header header;

    // internal
    FILE * fp_bin = nullptr;
    std::string prefix;
    int64_t n_written = 0;
};

// Read <prefix>.json back into a header. Returns false if the file is
// missing, unparseable, or fails schema/shape validation. On success the blob
// filename is derivable as <prefix>.bin (also recorded in the header JSON).
bool ts_features_read_header(const std::string & prefix, ts_features_header & out);

// Map an emitted feature row index to its corpus token index, accounting for
// the window layout. In overlap mode (stride > 0) the emitted rows are
// contiguous and this reduces to `warmup + row`. In legacy mode (stride == 0)
// each window discarded a warmup prefix, so the mapping has a per-window gap.
// With no window layout recorded (chunk_tokens == 0) the mapping is the
// identity when warmup == 0. Returns -1 if the layout is inconsistent (warmup
// set without chunk_tokens, or a stride that would double-emit/skip tokens),
// the row is out of range ([0, n_tokens)), or a window has no emitted rows.
int64_t ts_features_row_to_token(const ts_features_header & h, int64_t row);
