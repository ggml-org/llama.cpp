#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_context;
struct llama_batch;

struct common_ane_mtp_program;
struct common_ane_prefill_request;

using common_ane_mtp_program_ptr = std::shared_ptr<common_ane_mtp_program>;
// The stateless prefill runtime shares the warm multifunction implementation
// with MTP but is loaded only from the Tessera prefill artifact namespace.
using common_ane_prefill_program_ptr = common_ane_mtp_program_ptr;
using common_ane_prefill_request_ptr = std::shared_ptr<common_ane_prefill_request>;

struct common_ane_mtp_boundary_stats {
    uint64_t direct_input_views = 0;
    uint64_t direct_output_backings = 0;
    uint64_t arena_input_bytes = 0;
    uint64_t iosurface_arena_bytes = 0;
    uint64_t copied_output_bytes = 0;
    uint64_t async_prefill_submissions = 0;
    uint64_t async_prefill_completions = 0;
    uint64_t async_prefill_failures = 0;
};

struct common_ane_compute_function {
    std::string name;
    std::string role;
    uint32_t bucket = 0;
    bool warm = false;
};

// Phase 0 profile: per-function microsecond-level timing of the four
// dispatch phases the ANE heterogeneous backend drives per call.
// The phases are measured inside dispatch_pinned_function_locked
// (input_prep / ane_dispatch / output_read) and inside the pump's
// signal_fn ane_signal_slot_events (signal). All times are in
// microseconds; both totals and per-call max are tracked so the
// caller can see both throughput (total / count) and tail latency
// (max) at a glance. The counters are monotonic across the
// program's lifetime; common_ane_mtp_program_phase_stats returns
// a snapshot of the current values, organized as a per-function
// table for easy rendering.
//
// The struct is a SNAPSHOT type: the per-function program state
// is kept in an internal atomic-backed struct (single-writer on
// the E-core thread, multi-reader on the host), and
// common_ane_mtp_program_phase_stats copies the atomics into
// this POD for return. Plain uint64_t fields make the struct
// copyable and trivially destructible; the snapshot is the
// stable, copy-friendly view.
struct common_ane_phase_stats {
    // input_prep: building the input MLFeatureProvider (the
    // per-input slot loop + extra_inputs merge + MLDictionaryFeatureProvider
    // construction). Most of the cost is feature-dict boxing.
    uint64_t input_prep_us_total = 0;
    uint64_t input_prep_us_max = 0;
    // ane_dispatch: the Core ML predictionFromFeatures call. This
    // is the actual ANE work (or CPU fallback). It dominates the
    // total dispatch time for the gemma4 prefill bundle.
    uint64_t ane_dispatch_us_total = 0;
    uint64_t ane_dispatch_us_max = 0;
    // output_read: reading the outputs from the feature provider
    // + zero-copy verification (dataPointer == pinned.dataPointer
    // for every output slot in the function).
    uint64_t output_read_us_total = 0;
    uint64_t output_read_us_max = 0;
    // signal: the per-slot MTLSharedEvent signals emitted by the
    // pump's signal_fn. Each call is a single
    // ggml_mtl_shared_event_signal so this is normally <1us, but
    // it sums up under many outputs.
    uint64_t signal_us_total = 0;
    uint64_t signal_us_max = 0;
    // Number of successful dispatches the stats were collected
    // over. Failed dispatches (Core ML error, output not zero-
    // copy, pump not ready) are excluded.
    uint64_t count = 0;
};

// One row in the per-function phase table returned by
// common_ane_mtp_program_phase_stats. The row pairs a function
// (name + role + bucket) with its accumulated phase stats.
struct common_ane_program_phase_stats_row {
    std::string function_name;
    std::string role;
    uint32_t bucket = 0;
    common_ane_phase_stats stats;
};

struct common_ane_prefill_manifest {
    uint32_t abi_version = 0;
    uint32_t hidden_size = 0;
    uint32_t layer_first = 0;
    uint32_t layer_last = 0;
    std::string architecture;
    // Scheduler-selectable functions replace inclusive transformer layers and
    // emit the manifest-declared KV rows; lookup-only diagnostics are not a
    // valid prefill stage.
    std::string execution_stage;
    std::string hidden_layout;
    std::string kv_layout;
    std::string cache_requirement;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t batch_size = 0;
    // When true, a causal layer slab may run a shorter prompt in the next
    // larger sequence bucket.  Only the real prompt prefix is imported into
    // llama.cpp; right-padding must never affect causal prefix rows.
    bool causal_right_padding = false;
    std::vector<uint32_t> sequence_buckets;
};

// Materialize, load, warm, and retain an embedded self-contained MTP
// .mlmodelc program. Returns nullptr when the GGUF has no ANE payload or
// when Core ML cannot load/warm it; callers must retain the Metal fallback.
common_ane_mtp_program_ptr common_ane_mtp_program_load(
        const std::string & gguf_path,
        uint32_t batch_hint);

common_ane_prefill_program_ptr common_ane_prefill_program_load(
        const std::string & gguf_path,
        uint32_t sequence_hint = 0);

bool common_ane_prefill_manifest_load(
        const std::string & gguf_path,
        common_ane_prefill_manifest * manifest);

// Select the smallest warmable fixed-shape sequence bucket for `n_tokens`.
// Exact matches are always legal.  A larger bucket is legal only when the
// artifact explicitly declares causal right-padding support.  Zero means the
// caller must retain the ordinary backend path.
uint32_t common_ane_prefill_select_bucket(
        const common_ane_prefill_manifest & manifest,
        uint32_t n_tokens);

bool common_ane_mtp_program_is_warm(const common_ane_mtp_program_ptr & program);
const char * common_ane_mtp_program_cache_path(const common_ane_mtp_program_ptr & program);
common_ane_mtp_boundary_stats common_ane_mtp_program_boundary_stats(
        const common_ane_mtp_program_ptr & program);
std::vector<common_ane_compute_function> common_ane_compute_functions(
        const common_ane_mtp_program_ptr & program);

// Phase 0 profile: per-function phase table snapshot. Returns
// one row per warm function in the program; rows are sorted by
// (role, bucket) to match common_ane_compute_functions. The
// returned vector is a snapshot: subsequent dispatches update
// the program's internal counters; callers that need a
// consistent view should pin via a barrier (call twice and take
// the diff, or call once at the end of a benchmark loop).
std::vector<common_ane_program_phase_stats_row> common_ane_mtp_program_phase_stats(
        const common_ane_mtp_program_ptr & program);

// Phase 0 profile streaming. When the host sets a non-empty path
// via common_ane_phase_profile_set_output, the dispatch path
// (dispatch_pinned_function_locked) appends one NDJSON line per
// phase per dispatch to that path. Each line has the shape:
//
//   {"phase":"<phase>","us":<microseconds>,"n_tokens":<n>,"ts":<iso8601>}
//
// where <phase> is one of "input_prep", "ane_dispatch",
// "output_read" (matching the C++ struct common_ane_phase_stats
// field names) and <n> is the function's first non-batch input
// dim (the lane-active dim for prefill, MTP, and DFlash). The
// emit is opt-in: when the path is empty (the default) the
// dispatch path makes a single empty-string check per phase
// and returns. The file is opened lazily on the first emit
// and held until common_ane_phase_profile_set_output is called
// with a different path (or empty) or the process exits.
//
// The flag is read by both the host (synchronous dispatch) and
// the E-core pump (the submit_fn runs on the E-core thread), so
// the path is process-global. Multi-program dispatch paths
// share the same output file; the ts field is the per-line
// timestamp and the function name is implicit (the program is
// the one whose dispatch_pinned_function_locked is running).
//
// The default-empty path keeps the production dispatch path
// branch-free when profiling is off; the cost of the profile
// flag is one branch + one file-write per dispatch phase.
void common_ane_phase_profile_set_output(const char * path);
const char * common_ane_phase_profile_get_output();

// Test-only internal API. The dispatch path is the only legit
// caller of phase_profile_emit; this declaration exists so
// tests/test-ane-phase-profile-emit.cpp can drive a synthetic
// emit without spinning up a real .mlmodelc. The function is
// declared in the public header (rather than a private header)
// because the .mm file's anonymous-namespace helpers aren't
// visible to external translation units. Production code
// should not call this directly.
void common_ane_phase_profile_emit_test_only(
        const char * phase, uint64_t us, uint32_t n_tokens);

// Execute a fixed-sequence prefill function named prefill_sN. The function
// consumes I32 token_ids and positions shaped [lane_bucket, N], and writes
// hidden_states shaped [lane_bucket, N, hidden_size].
bool common_ane_compute_prefill(
        const common_ane_mtp_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        float * hidden_states);

// Execute a true transformer slab.  K/V are emitted token-major as
// [active_lanes, sequence_length, kv_heads, head_dim] and must be passed into
// llama.cpp's existing row writer for the current memory context; callers must
// not treat these arrays as a standalone cache.
bool common_ane_compute_prefill_slab(
        const common_ane_prefill_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        uint32_t kv_heads,
        uint32_t head_dim,
        float * hidden_states,
        float * key_states,
        float * value_states);

// Attempts an atomic ANE-prefill plus llama.cpp continuation for one sealed
// prompt bucket. A false return means no context mutation occurred and callers
// must use the ordinary Metal decode path. A true return means `result` is the
// real llama_decode status; callers must not retry that batch through Metal.
bool common_ane_prefill_decode(
        const common_ane_prefill_program_ptr & program,
        const common_ane_prefill_manifest & manifest,
        struct llama_context * ctx,
        const struct llama_batch & batch,
        int32_t * result);

// Fast-start path for prompts longer than the largest available bucket.  Runs
// the ANE slab over the first max_bucket tokens, imports its K/V into the
// cache, then continues with an ordinary llama_decode over the remaining
// tail tokens.  When `result` is non-null and the function returned true,
// it carries the final llama_decode status; when it returned false, no
// context mutation occurred and the caller must use the Metal fallback.
bool common_ane_prefill_decode_chunked(
        const common_ane_prefill_program_ptr & program,
        const common_ane_prefill_manifest & manifest,
        struct llama_context * ctx,
        const struct llama_batch & batch,
        int32_t * result);

// Start stateless prefill without blocking the llama.cpp scheduler.  The
// caller retains the input and output spans until completion; the request
// retains the compiled program and exposes completion/failure explicitly.
common_ane_prefill_request_ptr common_ane_compute_prefill_async(
        const common_ane_prefill_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        float * hidden_states);

bool common_ane_prefill_request_is_complete(
        const common_ane_prefill_request_ptr & request);
bool common_ane_prefill_request_wait(
        const common_ane_prefill_request_ptr & request);

// Execute a parallel DFlash draft function named dflash_bN. The function
// consumes target_features [lane_bucket, feature_width], token_ids and
// positions, and writes N token/confidence pairs per active lane.
bool common_ane_compute_dflash(
        const common_ane_mtp_program_ptr & program,
        uint32_t block_size,
        const float * target_features,
        uint32_t n_active,
        uint32_t feature_width,
        const int32_t * token_ids,
        const int32_t * positions,
        int32_t * draft_tokens,
        float * confidence);

// Arbitrate fixed-width DFlash and MTP candidate blocks with a multifunction
// entry named hybrid_bN. Candidate tensors are lane-major [lane_bucket, N].
// selected_source is 0 for no candidate, 1 for DFlash, and 2 for MTP.
bool common_ane_compute_hybrid(
        const common_ane_mtp_program_ptr & program,
        uint32_t block_size,
        const int32_t * dflash_tokens,
        const float * dflash_confidence,
        const int32_t * dflash_counts,
        const int32_t * mtp_tokens,
        const float * mtp_confidence,
        const int32_t * mtp_counts,
        uint32_t n_active,
        float dflash_cutoff,
        int32_t * selected_source,
        int32_t * agreement);

// Execute one fixed-bucket MTP step. Inactive lanes are padded with zeroes.
// The embedded program contract is:
//   token_ids   I32 [bucket]
//   h_nextn     F32 [bucket, hidden_size]
//   top_token   I32 [bucket]
//   confidence  F32 [bucket]
//   next_hidden F32 [bucket, hidden_size]
// Returns false without modifying outputs when the program is incompatible;
// callers should then use the regular llama.cpp draft context.
bool common_ane_mtp_program_predict(
        const common_ane_mtp_program_ptr & program,
        const int32_t * token_ids,
        const float * h_nextn,
        uint32_t n_active,
        uint32_t hidden_size,
        const int32_t * positions,
        int32_t * top_token,
        float * confidence,
        float * next_hidden);

// Synchronize target Gemma K/V deltas into a stateful multifunction program.
// Rows are packed per lane with `row_stride` rows allocated for each lane.
bool common_ane_mtp_program_sync_kv(
        const common_ane_mtp_program_ptr & program,
        uint32_t n_active,
        uint32_t row_stride,
        const uint32_t * row_counts,
        const int32_t * positions,
        const float * base_keys,
        const float * base_values,
        uint32_t base_width,
        const float * swa_keys,
        const float * swa_values,
        uint32_t swa_width);

// Clear persistent K/V state for selected fixed-bucket lanes before those
// server slots are reused for a new sequence.
bool common_ane_mtp_program_reset(
        const common_ane_mtp_program_ptr & program,
        uint32_t n_lanes,
        const int32_t * active);
