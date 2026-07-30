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

struct ts_gguf_writer_params {
    uint32_t    seed;
    float       alpha;
    float       clip;
    float       outlier_frac;
    std::string policy_summary;   // tensor_families string
    std::string policy_sha256;    // hex
    std::string build_info;
    std::string main_tip;
};

// Write tessera.* metadata KV pairs to an open gguf_context.
void ts_gguf_write_metadata(struct gguf_context * ctx, const ts_gguf_writer_params * params);

// Write the 6-component tensor cluster for one quantized weight.
// ctx: open gguf_context for writing.
// base_name: e.g. "blk.0.attn_q" (components get suffixed).
// result: the ts_quant_result_2d from tessera-quant.h.
// out_dim, in_dim: tensor dimensions.
void ts_gguf_write_tensor_cluster(struct gguf_context * ctx,
                                  const char * base_name,
                                  const void * result,
                                  int64_t out_dim, int64_t in_dim);
