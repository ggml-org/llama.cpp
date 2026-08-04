#pragma once

//
// tessera-unified-writer.h
//
// Unified GGUF writer for the Gemma4 12B + dspark + dflash + MTP arch.
//
// The LLM_ARCH_GEMMA4_ASSISTANT loader (src/llama-arch.cpp:135, see also
// src/models/gemma4-assistant.cpp) handles n_layer (trunk) + n_layer_nextn
// (MTP) + DFlash + DSPark + MTP tensors in a single arch. The writer
// produces a gemma4-assistant GGUF in one quantization pass from 4+
// per-component GGUFs (trunk, dflash, dspark, mtp_nextn, optional
// shared_embd).
//
// The writer is a thin C++ class that knows the gemma4-assistant tensor
// layout. It does NOT do quantization itself; it copies tensors from the
// per-component source GGUFs to the destination, optionally re-tagging
// their qtype via the per-tensor calibration policy. The Tile640
// (GGML_TYPE_TESSERA_T640) cluster format is preserved end-to-end: if a
// source tensor is tile640-encoded, the writer copies the 6 component
// tensors (weight_packed / weight_page_scales / weight_lane_scales /
// weight_outlier_row_offsets / weight_outlier_cols / weight_outlier_vals
// + optional weight_act_scale) by data pointer to the destination.
// Standard qtypes (F16, Q4_K, Q5_K, ...) are also copied by data pointer
// after gguf_set_tensor_type repoints the destination descriptor.
//
// The per-tensor qtype lookup is the calibration side's verdict (the
// "tensor_families" map from unified_calibrate.py's output). When the
// dispatch's tessera_db is open, ts_tessera_db_read_unified_policy
// provides the per-(model_hash, model_role, name) qtype; otherwise the
// writer reads the qtype from a sidecar JSON (the same shape that
// unified_calibrate.py emits).
//
// Phase 16: this file is the C++ counterpart to the Python
// unified_calibrate.py. The orchestrator's per-component calibration
// runs feed the policy; the writer consumes it. The dispatch's
// --tessera-db path is the production data path (DB overrides the
// sidecar JSON; the sidecar is a debugging affordance).
//

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Forward decls: we do not need to drag ggml/gguf into every TU that
// uses the writer.
struct gguf_context;
struct ggml_context;
struct ggml_tensor;

// One source GGUF + the role tag that tells the writer which
// gemma4-assistant slot the source's tensors belong to. The role
// matches the model_role column in tensor_stats; see
// ts_tessera_db_unified_policy_entry.model_role.
//
// The path is the path to a GGUF on disk. The writer opens it via
// gguf_init_from_file and reads its tensors by data pointer.
struct ts_unified_component {
    std::string path;
    std::string model_role;   // "trunk" / "dflash" / "dspark" / "mtp_nextn" / "shared_embd"
};

// Per-tensor qtype override from the calibration policy. The map is
// keyed by (model_role, name). When the policy is empty for a
// tensor, the writer copies the source tensor's qtype as-is.
//
// dtype values are the same string names the calibration pipeline
// emits ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32",
// "TESSERA_T640"). The writer maps these to ggml_type via
// ts_unified_qtype_from_string.
//
// When the source tensor is tile640, dtype must be "TESSERA_T640"
// (or empty, in which case the tile640 type is auto-preserved).
struct ts_unified_policy_entry {
    std::string model_role;
    std::string name;
    std::string dtype;
};
struct ts_unified_policy {
    std::vector<ts_unified_policy_entry> entries;
};

// gemma4-assistant hparams. The writer needs the trunk-only fields
// the loader reads (load_arch_hparams in src/models/gemma4-assistant.cpp:41):
//   * n_layer (BLOCK_COUNT; trunk)
//   * n_embd, n_head, n_head_kv (attention dims)
//   * n_embd_head_k, n_embd_head_v
//   * n_ff (FEED_FORWARD_LENGTH)
//   * n_vocab, n_embd_out (EMBEDDING_LENGTH_OUT for the target's hidden)
//   * n_swa (sliding-window pattern)
//   * n_embd_head_k_swa, n_embd_head_v_swa
//   * f_norm_rms_eps (RMS norm epsilon)
//   * rope_freq_base_train_swa
//   * is_swa_impl (per-layer SWA pattern: uint8 array of length n_layer)
//   * n_kv_shared_layers (ATTENTION_SHARED_KV_LAYERS)
//
// The MTP side is symmetric: n_layer_nextn == n_layer (the unified
// arch's current constraint; see gemma4-assistant.cpp:53).
struct ts_unified_hparams {
    uint32_t    n_layer               = 0;
    uint32_t    n_embd                = 0;
    uint32_t    n_head                = 0;
    uint32_t    n_head_kv             = 0;
    uint32_t    n_embd_head_k         = 0;
    uint32_t    n_embd_head_v         = 0;
    uint32_t    n_embd_head_k_swa     = 0;
    uint32_t    n_embd_head_v_swa     = 0;
    uint32_t    n_ff                  = 0;
    uint32_t    n_vocab               = 0;
    uint32_t    n_embd_out            = 0;  // the target trunk's hidden (== n_embd_inp of the drafter)
    uint32_t    n_swa                 = 0;
    uint32_t    n_kv_shared_layers    = 0;
    float       rope_freq_base_train_swa = 10000.0f;
    float       f_norm_rms_eps        = 1e-6f;
    // Per-layer SWA pattern. 1 = full attention, 0 = sliding-window.
    // Length is n_layer; gemma4-assistant.cpp:45 reads it as
    // hparams.is_swa_impl.
    std::vector<uint8_t> is_swa_impl;
};

// Optional dflash-side hparams. The dflash drafter is a separate
// GGUF in the unified pipeline; the trunk-side gemma4-assistant
// graph borrows the drafter's tok_embd/output via ctx_other (see
// gemma4-assistant.cpp:150-159). The writer needs the drafter's
// n_layer / n_embd / n_vocab / n_ff to validate the tensor
// shapes when it copies them.
struct ts_unified_dflash_hparams {
    uint32_t    n_layer  = 0;
    uint32_t    n_embd   = 0;
    uint32_t    n_head   = 0;
    uint32_t    n_head_kv = 0;
    uint32_t    n_embd_head_k = 0;
    uint32_t    n_ff     = 0;
    uint32_t    n_vocab  = 0;
};

// Optional dspark-side hparams. The DSPark extension (markov_w1 /
// markov_w2 / conf_proj.*) is read off the dflash drafter GGUF in
// the unified pipeline; the writer uses the rank to validate the
// conf_proj dimensions.
struct ts_unified_dspark_hparams {
    int32_t     markov_rank = 0;   // -1 when not present
};

// The destination metadata written by the writer (beyond the
// per-tensor qtype). The keys match the loader's get_key calls in
// gemma4-assistant.cpp:41-60.
struct ts_unified_meta {
    std::string build_info;     // "tessera-unified-writer <sha> @ <date>"
    std::string main_tip;       // main branch HEAD when this writer was built
};

// The writer. Open on the destination path, then call write_all.
// Errors are reported via the out `err` parameter; the destructor
// closes the destination GGUF.
class ts_unified_writer {
public:
    // Open a writer on the destination GGUF. Reads the source
    // GGUFs (one per component), reads the calibration policy
    // (from the sidecar JSON or from the dispatch's tessera_db),
    // and prepares the destination GGUF in memory. Returns 0 on
    // success, non-zero on failure (message in *err).
    ts_unified_writer(const std::string & dst_path,
                      const std::vector<ts_unified_component> & components,
                      const ts_unified_policy & policy,
                      const ts_unified_hparams & hparams,
                      const ts_unified_dflash_hparams & dflash_hparams,
                      const ts_unified_dspark_hparams & dspark_hparams,
                      const ts_unified_meta & meta,
                      std::string * err);
    ~ts_unified_writer();

    // Disable copy / move. The writer owns a ggml_context + multiple
    // gguf_contexts and a lot of dangling pointers; copy semantics
    // would dangle.
    ts_unified_writer(const ts_unified_writer &) = delete;
    ts_unified_writer & operator=(const ts_unified_writer &) = delete;
    ts_unified_writer(ts_unified_writer &&) = delete;
    ts_unified_writer & operator=(ts_unified_writer &&) = delete;

    // Write the gemma4-assistant arch header (KV pairs) and every
    // tensor from every component into the destination GGUF, then
    // flush the GGUF to disk. Returns 0 on success, non-zero on
    // failure (message in *err).
    //
    // Tensor routing: each source tensor's name is the gemma4
    // gemma4-assistant slot name (e.g. "blk.0.attn_q.weight" for
    // trunk, "dflash.encoder.fc.weight" is NOT used -- the writer
    // maps "fc.weight" to "dflash.encoder.fc.weight" via the
    // per-component routing table; "markov_w1.weight" maps to the
    // dspark slot; "blk.0.nextn.eh_proj.weight" maps to the
    // mtp_nextn slot; "token_embd.weight" / "output.weight" are
    // shared_embd slots with the worst-of-trunk-and-dflash qtype).
    int write_all(std::string * err);

    // Stats on what was written. Populated by write_all; useful
    // for the CLI's stderr summary.
    struct stats {
        int32_t  n_tensors_trunk       = 0;
        int32_t  n_tensors_dflash      = 0;
        int32_t  n_tensors_dspark      = 0;
        int32_t  n_tensors_mtp_nextn   = 0;
        int32_t  n_tensors_shared_embd = 0;
        int32_t  n_tensors_skipped     = 0;   // unknown / unsupported names
        int32_t  n_qtype_overrides     = 0;   // policy changed source qtype
        int64_t  total_bytes           = 0;
    };
    const stats & get_stats() const { return stats_; }

private:
    struct impl;
    impl * p_;
    stats  stats_;
};

// --- helpers used by the CLI ---

// Map a calibration-side dtype string ("Q4_K", "TESSERA_T640", ...)
// to a ggml_type. Returns GGML_TYPE_COUNT (== 45) on unknown;
// the writer treats that as a hard error.
int ts_unified_qtype_from_string(const std::string & s);

// Map a ggml_type to its canonical string name (the inverse of
// ts_unified_qtype_from_string). Returns "" when the type has no
// canonical name (the writer never needs to call this; the helper
// exists for log/diagnostic output).
std::string ts_unified_qtype_to_string(int qtype);

// Read a calibration policy from a sidecar JSON file. The JSON
// shape mirrors unified_calibrate.py's output: a top-level
// "tensor_families" array of {model_role, name, dtype} triples.
// Returns 0 on success, non-zero on failure (message in *err).
//
// The dispatch's tessera_db path is the production data path; this
// helper is for the CLI's --policy FILE flag and for tests. The
// writer's constructor takes a ts_unified_policy directly so the
// caller (CLI / test) can choose between the DB and the JSON.
int ts_unified_policy_load_json(const std::string & path,
                                 ts_unified_policy * out,
                                 std::string * err);

// Serialize a ts_unified_policy to a JSON file. Symmetric to
// ts_unified_policy_load_json; the writer does not need it but
// the CLI uses it to dump the effective policy for debugging.
int ts_unified_policy_save_json(const std::string & path,
                                 const ts_unified_policy & policy,
                                 std::string * err);
