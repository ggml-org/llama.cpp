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
// The 8 recognized values are:
//   * "trunk"        - the gemma4-12B trunk (the dominant compute).
//   * "dflash"       - the dflash drafter (EAGLE-style feature-conditioned
//                      block drafter). The writer prefixes tensors with
//                      "dflash." to disambiguate from identically-named
//                      trunk tensors (the drafter is loaded as ctx_other
//                      at runtime; the prefix is forward-looking for the
//                      case where the drafter reads from the same
//                      unified GGUF).
//   * "dspark"       - the DSPark heads (markov_w1 / markov_w2 /
//                      conf_proj.*) read off the dflash drafter.
//   * "mtp_nextn"    - the MTP nextn_predict_layers' blk.{i}.nextn.*.
//   * "shared_embd"  - shared token_embd / output across trunk + dflash.
//   * "vision_tower" - the multimodal vision encoder (gemma4-assistant
//                      mtmd companion). Source tensors already carry
//                      the "v." prefix per tools/mtmd/clip.cpp:1831;
//                      the writer does not add a second prefix.
//   * "audio_tower"  - the multimodal audio encoder. Source tensors
//                      already carry the "a." prefix.
//   * "mm_projector" - the multimodal projector (multi_modal_projector /
//                      mm.* weights). Source tensors already carry
//                      the "mm." prefix. A "mm.*" tensor appearing in
//                      BOTH vision_tower and mm_projector components
//                      uses the same name-based worst-of lookup that
//                      token_embd.weight uses across trunk / dflash /
//                      shared_embd.
//
// The path is the path to a GGUF on disk. The writer opens it via
// gguf_init_from_file and reads its tensors by data pointer.
struct ts_unified_component {
    std::string path;
    std::string model_role;
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

// Phase 16.8: per-role size budget for cross-role relaxation.
//
// The "did this requant plan reduce error?" loop's producer
// (l5_retune.py) computes a requant_budget_bits per (model, role,
// family). When the writer reconciles a SHARED tensor (token_embd /
// output) across roles via worst-of, one role's budget can be pushed
// over by another role's more-conservative verdict. The writer needs
// the budget expressed as BITS PER ELEMENT (the same unit as
// ts_unified_writer_qtype_bits) so it is directly comparable to a
// qtype's bit cost, plus a dynamic weight that says how much that
// role's constraint/need should be honored in the relaxation decision.
//
// budget_bits: max bits/element the role's size envelope allows for a
//   shared tensor. -1 = unconstrained (no budget). 0 is a valid budget
//   (forces the lowest-bit verdict).
// weight: dynamic weighting, >= 0. The Python producer derives it from
//   n_samples, hit_rate, and coupling_score; the writer only COMPARES
//   weights. A role whose weight is >= the constrained role's weight
//   protects its more-conservative verdict (the constraint is relaxed).
struct ts_unified_role_budget {
    std::string model_role;
    int64_t     budget_bits = -1;   // bits/element; -1 = unconstrained
    double      weight      = 0.0;  // dynamic weighting for relaxation
};

struct ts_unified_policy {
    std::vector<ts_unified_policy_entry> entries;
    // Phase 16.8: optional per-role budgets. Empty = the writer applies
    // plain worst-of everywhere (the pre-16.8 contract). Carried in the
    // policy sidecar JSON ("role_budgets") and/or built by the CLI from
    // the dispatch's tessera_db (l5_weights).
    std::vector<ts_unified_role_budget> role_budgets;
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

// Optional multimodal-projector hparams. When non-zero, the writer
// emits the corresponding gemma4-assistant.vision.* /
// gemma4-assistant.audio.* / gemma4-assistant.mm.* KV pairs into the
// destination GGUF. Default-constructed (all zero / empty) = no
// mmproj metadata, which is the pre-M0a contract. The vision / audio
// hparams are independent so a vision-only or audio-only
// configuration is supported (the absent side stays at the default).
struct ts_unified_mmproj_hparams {
    int32_t     vision_n_embd    = 0;   // vision tower embedding dim
    int32_t     audio_n_embd     = 0;   // audio tower embedding dim
    int32_t     projector_dim    = 0;   // mm_projector output dim (== n_embd)
    std::string vision_arch;            // e.g. "clip" / "siglip" / "gemma4-vision"
    std::string audio_arch;             // e.g. "whisper" / "gemma4-audio-conformer"
};

// The destination metadata written by the writer (beyond the
// per-tensor qtype). The keys match the loader's get_key calls in
// gemma4-assistant.cpp:41-60.
struct ts_unified_meta {
    std::string build_info;     // "tessera-unified-writer <sha> @ <date>"
    std::string main_tip;       // main branch HEAD when this writer was built
};

// One budget event the writer recorded while writing (Phase 16.8).
// Surfaced via ts_unified_writer::get_budget_events and mirrored into
// the destination GGUF metadata (tessera.unified.budget_events) so the
// relaxation audit trail travels with the artifact.
struct ts_unified_budget_event {
    std::string tensor;          // shared tensor name
    std::string action;          // "relaxed" | "enforced"
    std::string role;            // the constrained role
    std::string other_role;      // protected role (relaxed) / "" (enforced)
    int         qtype = 45;      // resolved qtype (45 == GGML_TYPE_COUNT)
    std::string reason;          // evidence line
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
                      const ts_unified_mmproj_hparams & mmproj_hparams,
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
        int32_t  n_tensors_trunk        = 0;
        int32_t  n_tensors_dflash       = 0;
        int32_t  n_tensors_dspark       = 0;
        int32_t  n_tensors_mtp_nextn    = 0;
        int32_t  n_tensors_shared_embd  = 0;
        int32_t  n_tensors_vision_tower = 0;   // Phase M0a: multimodal
        int32_t  n_tensors_audio_tower  = 0;   // Phase M0a: multimodal
        int32_t  n_tensors_mm_projector = 0;   // Phase M0a: multimodal
        int32_t  n_tensors_skipped      = 0;   // unknown / unsupported names
        int32_t  n_qtype_overrides      = 0;   // policy changed source qtype
        int32_t  n_budget_relaxed       = 0;   // Phase 16.8: constraints relaxed
        int32_t  n_budget_enforced      = 0;   // Phase 16.8: budgets capped qtype
        int64_t  total_bytes            = 0;
    };
    const stats & get_stats() const { return stats_; }

    // Phase 16.8: the per-shared-tensor budget relaxation/enforcement
    // evidence recorded by write_all. Empty when the policy carries no
    // role budgets (or no shared tensor hit a budget). Also serialized
    // into the destination GGUF metadata (tessera.unified.budget_events).
    const std::vector<ts_unified_budget_event> & get_budget_events() const;

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

// --- Phase 16.6: worst-of-trunk-and-dflash qtype resolution ---
//
// The unified Gemma4 12B + dspark + dflash + MTP arch has shared
// token_embd / output tensors between the trunk and the dflash
// drafter (the drafter borrows them via ctx_other, frozen at
// train time; see tessera-train-dflash.cpp:72). When the per-
// component calibration runs, the trunk's and dflash's verdicts
// can disagree on the same shared tensor. The writer must pick
// ONE qtype per tensor. The architect's call: take the MORE
// CONSERVATIVE option (the qtype with more bits = less
// precision loss). The drafter is more sensitive to the
// embedding than the trunk, so when in doubt, use more bits.

// Bit cost of a ggml_type for the worst-of ordering. F32 is the
// "no quantization" anchor (0 bits quantized); F16 / BF16 are
// the "fully unquantized" anchors at 16 bits per element.
// Q2_K..Q8_0 are 2..8 bits per element.
//
// The qtype is passed as an int (matching the
// ts_unified_qtype_from_string convention) so the header
// does not need to drag ggml.h into every TU.
//
// Returns 0 for unknown / unsupported qtypes; the worst_of
// caller treats 0 as the lowest precision (F32 anchor), so an
// unknown qtype in the policy degrades safely to the more
// conservative partner.
int ts_unified_writer_qtype_bits(int qtype);

// Returns the more conservative of two qtypes (max(bits)).
// When both qtypes are equal, returns `a`. When one is unknown
// (bits == 0), the other is always more conservative and is
// returned. GGML_TYPE_COUNT is NOT handled here; the caller's
// sentinel for "no entry" is GGML_TYPE_COUNT and must be
// checked before calling worst_of.
int ts_unified_writer_worst_of(int a, int b);

// Sentinel for "no qtype verdict" that matches GGML_TYPE_COUNT without
// dragging ggml.h into this header (the writer's .cpp static_asserts
// the two agree).
static const int TS_UNIFIED_QTYPE_NONE = 45;

// --- Phase 16.8: budget-aware cross-role reconciliation --------------
//
// Worst-of is the correctness default: a shared tensor (token_embd /
// output) gets the more conservative qtype across the roles' verdicts.
// But a role's size budget (ts_unified_role_budget.budget_bits) can be
// pushed over by another role's more-conservative verdict. The
// architect's rule: when enforcing the constrained role's budget would
// compromise a needier role, RELAX THE CONSTRAINT - keep the
// conservative qtype - rather than drop bits. The decision is weighted
// dynamically: the constrained role only gets to pull bits down when it
// outweighs every role it would compromise.
//
// ts_unified_writer_resolve_shared implements that rule as a PURE
// function (no writer / ggml state) so the algorithm is unit-testable
// in isolation. The writer calls it per shared tensor; everything else
// falls back to plain worst-of.

// One role's qtype verdict + budget + weight for a single shared
// tensor. qtype is an int ggml_type (GGML_TYPE_COUNT = no verdict).
// budget_bits / weight match ts_unified_role_budget for that role
// (-1 / 0.0 when the role has no budget entry).
struct ts_unified_role_verdict {
    std::string model_role;
    int         qtype       = TS_UNIFIED_QTYPE_NONE;
    int64_t     budget_bits = -1;
    double      weight      = 0.0;
};

// Outcome of budget-aware reconciliation for one shared tensor.
struct ts_unified_shared_resolution {
    int         qtype          = TS_UNIFIED_QTYPE_NONE;  // resolved qtype
    bool        budget_applied = false;  // any budget considered
    bool        relaxed        = false;  // a constraint was relaxed
    bool        enforced       = false;  // a budget capped the qtype
    std::string relaxed_role;            // role whose constraint was relaxed
    std::string protected_role;          // role protected by the relaxation
    std::string enforced_role;           // role whose budget capped the qtype
    std::string reason;                  // human-readable evidence line
};

// Reconcile one shared tensor across per-role verdicts with budgets.
// Returns the resolved qtype + relaxation evidence. Pure; no I/O.
ts_unified_shared_resolution ts_unified_writer_resolve_shared(
    const std::vector<ts_unified_role_verdict> & verdicts);

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
