#pragma once

//
// tessera-gguf-writer.h
//
// Writes the 6-component Tessera tensor cluster and tessera.* GGUF
// metadata KV pairs. G3 phase of the C++ port.
//

#include <string>
#include <cstdint>

struct gguf_context;
struct ggml_context;

struct ts_gguf_writer_params {
    uint32_t    seed;
    float       alpha;
    float       clip;
    float       outlier_frac;
    std::string policy_summary;   // tensor_families string
    std::string policy_sha256;    // hex
    std::string build_info;
    std::string main_tip;
    // S9 W4A4 activation quantization metadata (written when w4a4_enabled)
    bool        w4a4_enabled = false;
    int         w4a4_activation_bits = 4;
    std::string w4a4_scale_mode;  // "per_token" / "per_tensor"
    float       w4a4_outlier_thresh = 6.0f;
};

// Write tessera.* metadata KV pairs to an open gguf_context.
void ts_gguf_write_metadata(struct gguf_context * ctx, const ts_gguf_writer_params * params);

// Write the 6-component tensor cluster for one quantized weight.
// ctx: open gguf_context for writing.
// gctx: caller-owned ggml_context used to allocate the transient tensor
//       descriptors. gguf_add_tensor copies each descriptor by value, so the
//       descriptors themselves do not outlive this call; gctx is caller-owned
//       only to keep this function allocation-free.
// base_name: e.g. "blk.0.attn_q" (components get suffixed).
// result: the ts_quant_result_2d from tessera-quant.h. Its backing buffers
//       (packed / page_scales / lane_scales / outliers / act_scale) are stored
//       by data pointer, so they MUST remain valid until after the eventual
//       gguf_write_to_file call, which writes the payload from those pointers.
// out_dim, in_dim: tensor dimensions.
void ts_gguf_write_tensor_cluster(struct gguf_context * ctx,
                                  struct ggml_context * gctx,
                                  const char * base_name,
                                  const void * result,
                                  int64_t out_dim, int64_t in_dim);

// Repoint the data pointers of an existing tensor cluster (written by an
// earlier ts_gguf_write_tensor_cluster call) at the buffers of `result`.
// Used by the L5 adaptive-requantize loop, which re-quantizes a tensor
// in place into the same ts_quant_result_2d and must refresh the GGUF
// descriptors so the eventual gguf_write_to_file emits the refined bytes.
// Tensors that are not found in gctx are skipped (silent: the cluster is
// assumed to have been written already). Returns the number of tensors
// repointed.
int ts_gguf_repoint_tensor_cluster(struct ggml_context * gctx,
                                   const char * base_name,
                                   const void * result);
