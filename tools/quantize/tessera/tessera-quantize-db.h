#pragma once

//
// tessera-quantize-db.h
//
// DuckDB-backed persistent store for the Tessera quantize pipeline.
//
// The quantize pipeline has historically been ephemeral: every run starts the
// GA from scratch, family warm-start lives only in process memory, and the
// only durable artifact is a flat policy JSON. ts_tessera_db wraps a DuckDB
// connection so the pipeline can:
//   - record one row per run, tensor, GA result, and acceptance comparison
//   - stream per-candidate GA evaluations to disk via the Appender API
//   - reload family-optimal alpha/clip on restart to warm-start the GA
//   - skip tensors that already converged in a prior (interrupted) run
//
// The whole store is optional: when the dispatch is constructed with a null
// ts_tessera_db*, every method is a no-op (guarded by the caller's null
// check). This keeps the existing ephemeral path unchanged.
//
// DuckDB's C++ API throws on errors; this wrapper catches and reports via the
// out `err` parameter, returning non-zero. The dispatch treats a DB error as
// non-fatal (logs + continues) so a corrupt DB never blocks quantization.
//

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declaration so the public header does not require duckdb.hpp.
// (The .cpp includes the amalgamation; translation units that only need the
// API surface stay clean of DuckDB's 2 MB header.)
namespace duckdb { class DuckDB; class Connection; }

struct ts_tessera_db {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> conn;
    // Phase 16.7: stores the on-disk path passed to
    // ts_tessera_db_open (or "" for ":memory:"). Used by
    // ts_tessera_db_migrate_model_role to write the
    // model_role_migration.json sidecar next to the duckdb
    // file. The C++ struct used to discard the path; the
    // sidecar needs it.
    std::string                        db_path;

    ts_tessera_db() = default;
    ~ts_tessera_db();
};

// Open (or create) the database at `path` and ensure the schema exists.
// Returns nullptr on failure (message in *err). An in-memory DB (":memory:")
// is supported for tests.
ts_tessera_db * ts_tessera_db_open(const std::string & path,
                                     std::string * err);

// Run-lifecycle hooks. begin_run inserts a new row and returns the run_id
// (a hash of model_hash + config + timestamp). complete_run / fail_run flip
// the status. The run_id is reused across all per-tensor inserts.
std::string ts_tessera_db_begin_run(ts_tessera_db * db,
                                     const std::string & model_path,
                                     const std::string & model_hash,
                                     const std::string & tessera_commit,
                                     const std::string & config_json,
                                     std::string * err);
int ts_tessera_db_complete_run(ts_tessera_db * db,
                                const std::string & run_id,
                                const std::string & status,   // "completed" / "failed"
                                std::string * err);

// Tensor registry. One row per quantizable 2D/3D weight, captured during the
// ga-prep walk. layer_depth is the block index (0 for non-block tensors).
struct ts_tessera_db_tensor {
    std::string  run_id;
    std::string  name;
    std::string  family;
    int32_t      layer_depth = 0;
    int64_t      out_dim     = 0;
    int64_t      in_dim      = 0;
    int64_t      n_elements  = 0;
    float        kurtosis    = 0.0f;
    float        eff_rank    = 0.0f;
    std::string  source_type;   // "f16", "f32", "q8_0", ...
};
int ts_tessera_db_insert_tensor(ts_tessera_db * db,
                                 const ts_tessera_db_tensor & t,
                                 std::string * err);

// GA results. One row per converged tensor (the summary). Re-inserting a
// (run_id, tensor_name) pair replaces the row (PRIMARY KEY conflict -> upsert).
struct ts_tessera_db_ga_result {
    std::string  run_id;
    std::string  tensor_name;
    std::string  family;
    float        best_alpha   = 0.0f;
    float        best_clip    = 0.0f;
    float        best_composite = 0.0f;
    float        best_mse     = 0.0f;
    int32_t      generations_run = 0;
    int64_t      n_evaluations    = 0;
    bool         converged   = false;
    bool         warm_started   = false;
};
int ts_tessera_db_insert_ga_result(ts_tessera_db * db,
                                    const ts_tessera_db_ga_result & r,
                                    std::string * err);

// Acceptance-gate comparison row (one per tensor).
struct ts_tessera_db_acceptance {
    std::string  run_id;
    std::string  tensor_name;
    std::string  family;
    float        composite_t2 = 0.0f;
    float        awq_t2       = 0.0f;
    float        rotation_t2  = 0.0f;
    float        lowrank_t2   = 0.0f;
    float        hessian_t2   = 0.0f;
    std::string  verdict;        // "pass" / "fail"
};
int ts_tessera_db_insert_acceptance(ts_tessera_db * db,
                                     const ts_tessera_db_acceptance & a,
                                     std::string * err);

// L5 adaptive requantize fixup row.
struct ts_tessera_db_l5_fixup {
    std::string  run_id;
    std::string  tensor_name;
    int32_t      generation   = 0;
    std::string  strategy;       // "A" (alpha/clip) or "B" (outlier)
    float        before_frob   = 0.0f;
    float        after_frob    = 0.0f;
};
int ts_tessera_db_insert_l5_fixup(ts_tessera_db * db,
                                   const ts_tessera_db_l5_fixup & f,
                                   std::string * err);

// --- tensor_stats upsert (cross-pipeline feature table) ---
//
// One row per (model_hash, name). The C++ GA-prep walk writes
// kurtosis / eff_rank / dtype here (alongside the legacy `tensors`
// per-run table); the Python calibration pipeline writes
// rms / mean_abs / tail_ratio. PRIMARY KEY (model_hash, name) +
// ON CONFLICT DO UPDATE makes this an upsert target.
//
// The `source` field records which pipeline last wrote the row
// ("cpp_quant" for the C++ side, "py_cal" for Python). It is
// informational; the upsert overwrites regardless.
//
// `recommended_action` is the per-tensor verdict produced by the
// calibration side (see tools/tessera/l5_action.py). It is a
// derived string ("protect" / "requant_up" / "requant_down" /
// "monitor" / "noop") that summarizes how the calibration pipeline
// should treat this tensor given the orchestrator's feedback
// (miscalibration_score, hit_rate, plan_accepted, delta_mse) for
// its (model, family). The C++ side does not write this field;
// the Python calibration_to_tensor_stats.py upserts it from
// l5_weights via the rules in l5_action.py. The COALESCE
// preservation in the upsert makes this a one-way Python write
// without disturbing the C++ side's other columns.
struct ts_tessera_db_tensor_stat {
    std::string  model_hash;
    std::string  model_role;   // Phase 16: "trunk" / "dflash" / "dspark" /
                               // "mtp_nextn" / "shared_embd". Disambiguates
                               // tensors with the same name in the unified
                               // Gemma4 12B + dspark + dflash + MTP arch
                               // (e.g. "blk.0.attn_q.weight" exists in both
                               // the trunk and the dflash encoder). Default
                               // "trunk" preserves the pre-Phase-16 contract.
    std::string  name;
    std::string  family;
    int32_t      layer_depth = 0;
    int64_t      out_dim     = 0;
    int64_t      in_dim      = 0;
    int64_t      n_elements  = 0;
    std::string  dtype;       // "f16", "f32", "q8_0", ...
    double       kurtosis    = 0.0;
    double       eff_rank    = 0.0;
    double       rms         = 0.0;
    double       mean_abs    = 0.0;
    double       tail_ratio  = 0.0;
    std::string  source;      // "cpp_quant" / "py_cal"
    std::string  recommended_action;  // "protect" / "requant_up" /
                                      // "requant_down" / "monitor" / "noop"
                                      // (Python-side only; default empty)
};
int ts_tessera_db_upsert_tensor_stat(ts_tessera_db * db,
                                     const ts_tessera_db_tensor_stat & row,
                                     std::string * err);

// --- Per-component qtype reader (Phase 16 unified writer) ---
//
// The unified Gemma4 12B + dspark + dflash + MTP arch shares tensor
// names across components. The per-tensor qtype the writer needs is
// keyed on (model_hash, model_role, name). The reader pulls every
// tensor_stats row for a model (or a single role) so the writer can
// pick the right qtype for each component tensor in a single round
// trip.
//
// Empty `role` returns all roles. Rows are returned in
// (model_role, name) order so the writer's per-component scan is
// cache-friendly. dtype is the per-tensor qtype the calibration
// pipeline recommended (or "f16" / "f32" for the no-quantize
// passthrough); the writer filters on this when assigning qtypes to
// gemma4-assistant tensor slots.
struct ts_tessera_db_unified_policy_entry {
    std::string  model_role;
    std::string  name;
    std::string  family;
    std::string  dtype;
    std::string  source;
    std::string  recommended_action;
};
struct ts_tessera_db_unified_policy {
    std::vector<ts_tessera_db_unified_policy_entry> entries;
};
int ts_tessera_db_read_unified_policy(
    ts_tessera_db * db,
    const std::string & model_hash,
    const std::string & role,    // empty = all roles
    ts_tessera_db_unified_policy * out,
    std::string * err);

// --- Phase 16 migration: in-place model_role column add ---
//
// Performs the standard DuckDB PK-rebuild dance on tensor_stats:
// CREATE new -> INSERT FROM old -> DROP old -> RENAME. Idempotent
// via information_schema.columns check. The writer branch only
// needs the model_role column on tensor_stats; the schema branch
// (evolve/unified-schema) extends the same dance to the other 6
// affected tables. Called from ts_tessera_db_open on every open.
int ts_tessera_db_migrate_model_role(
    ts_tessera_db * db,
    std::string * err);

// --- L5 weights: per-(model, family) retuned scoring weights ---
//
// The "did this requant plan reduce error?" feedback loop lands
// its consumer at l5_outcome. The next consumer is l5_retune.py,
// which fits a closed-form OLS model of delta_mse on
// sensitivity_score per (model, family) and projects the result
// onto the (w_imatrix, w_gradient, w_layer) simplex. The
// orchestrator's next generation reads these weights as the
// starting point for SensitivityScorer, closing the loop.
//
// PRIMARY KEY (model_hash, family). bias is the OLS intercept;
// n_samples is the count of l5_outcome rows that fed the fit;
// in_sample_loss is the post-fit mean abs residual. retune_source
// records which algorithm produced the row.
struct ts_tessera_db_l5_weight {
    std::string  model_hash;
    std::string  model_role;   // Phase 16: same enum as tensor_stats.
                               // The l5_weights table is per-(model, role,
                               // family) so the dflash family's retuned
                               // weights don't collide with the trunk
                               // family's. Default "trunk" preserves the
                               // pre-Phase-16 contract.
    std::string  family;
    double       w_imatrix    = 0.0;
    double       w_gradient   = 0.0;
    double       w_layer      = 0.0;
    double       bias         = 0.0;
    int32_t      n_samples    = 0;
    double       in_sample_loss = 0.0;
    double       hit_rate     = 0.0;
    // requant_budget_bits is the dispatch-side budget the orchestrator's
    // l5_retune recommends for this family in the next requant pass.
    // Computed by l5_retune.py as
    //   budget = family_storage_bits * (1 - hit_rate) * base_budget_fraction
    // and NULL when the family has too few samples to project a budget
    // (or the family has no tensor_stats storage rows). The dispatch's
    // L5 loop consumes it: budgeted families pick their A/B winner on
    // the Lagrangian score frob + lambda * violation (measured storage
    // bits), with subgradient ascent on lambda while the family stays
    // over budget. See tessera-dispatch.cpp (ts_dispatch_run_l5_loop).
    int64_t      requant_budget_bits = -1;   // -1 = NULL (sentinel for C++)
    std::string  retune_source;
};
int ts_tessera_db_upsert_l5_weight(ts_tessera_db * db,
                                   const ts_tessera_db_l5_weight & row,
                                   std::string * err);

// --- L5 weights typed list reader ---
//
// One entry per (model_hash, family) row in l5_weights. Used by the
// dispatch's GA-prep walk warm-start to bias the GA's (alpha, clip)
// initial population per family when the orchestrator's retune has
// already characterized the family. Empty `family` means "all families
// for this model_hash"; the dispatch passes empty when it wants the
// full list at open time.
//
// Mirrors ts_tessera_db_list_converged_for_model. Returned in hit_rate
// DESC order so the dispatch sees the most-converged family first.
struct ts_tessera_db_l5_weight_list_entry {
    std::string  model_hash;
    std::string  model_role;   // Phase 16: echoes the row's role so the
                               // dispatch's GA-prep walk can filter on
                               // (model_hash, model_role) without a
                               // separate read. Default "trunk" for
                               // pre-Phase-16 rows.
    std::string  family;
    double       w_imatrix    = 0.0;
    double       w_gradient   = 0.0;
    double       w_layer      = 0.0;
    double       bias         = 0.0;
    int32_t      n_samples    = 0;
    double       in_sample_loss = 0.0;
    double       hit_rate     = 0.0;
    int64_t      requant_budget_bits = -1;   // -1 = NULL
    std::string  retune_source;
};
struct ts_tessera_db_l5_weight_list {
    std::vector<ts_tessera_db_l5_weight_list_entry> entries;
};
int ts_tessera_db_list_l5_weights(ts_tessera_db * db,
                                  const std::string & model_hash,
                                  const std::string & family,
                                  ts_tessera_db_l5_weight_list * out);

// --- L5 outcome stats: per-(model, family) verdict aggregates ---
//
// Used by the dispatch's converged-fast early-exit: if a family has
// hit_rate > 0.95 across prior l5_outcome rows AND the current tensor's
// MSE is already within epsilon of the expected MSE at the next-rung
// qtype, skip the requant. The reader aggregates the verdict
// (plan_accepted) over the (model, family) group: hit_rate is the
// fraction of l5_outcome rows where plan_accepted = TRUE.
//
// Empty `family` means "all families" (caller picks the most-converged
// one to use as the gate). n_rows = 0 means "no l5_outcome for this
// model yet"; the dispatch treats that as no-op.
struct ts_tessera_db_l5_outcome_stats {
    int32_t      n_rows       = 0;
    int32_t      n_accepted   = 0;
    double       hit_rate     = 0.0;
    double       mean_delta_mse = 0.0;
    double       mean_sensitivity = 0.0;
    std::string  family;       // empty when aggregating across families
};
int ts_tessera_db_l5_outcome_stats_for(ts_tessera_db * db,
                                       const std::string & model_hash,
                                       const std::string & family,
                                       ts_tessera_db_l5_outcome_stats * out);

// --- L4 plan outcome (the feedback loop audit trail) ---
//
// One row per (model_hash, name, iteration, plan_id) recording the
// L4 forward-pass measurement AFTER a requant plan was applied.
// The C++ adaptive_requantize loop writes one row per (tensor, gen)
// with strategy = "A" (alpha/clip multiplier) or "B" (outlier_thresh
// bump). The Python l5_orchestrator's per-iteration runs also land
// here when the apply step re-quantizes and re-probes.
//
// The C++ side writes through a ts_db_buffer (parallel workers
// funnel into one flusher thread). The buffer is opened in
// ts_dispatch_db_open alongside the ga_evaluations buffer; the
// helper below is a thin shim that the dispatch's L5 loop calls
// per (tensor, iteration).
struct ts_tessera_db_l4_outcome {
    std::string model_hash;        // empty when hashing failed
    std::string  model_role;       // Phase 16: "trunk" / "dflash" / "dspark" /
                                   // "mtp_nextn" / "shared_embd". The
                                   // l4_plan_outcome table is the
                                   // feedback-loop audit trail; rows
                                   // for dflash / dspark / mtp_nextn
                                   // tensors must be distinguishable.
                                   // Default "trunk" preserves the
                                   // pre-Phase-16 contract.
    std::string  name;             // tensor name (drafter-local for
                                   // non-trunk roles)
    int32_t      layer             = 0;
    int32_t      iteration         = 0;
    std::string  plan_id;          // "cpp_quant_gen{N}_stage{S}" or "py_orch_..."
    std::string  strategy;         // "A" (alpha/clip) or "B" (outlier_thresh)
    double       alpha_before      = 0.0;
    double       alpha_after       = 0.0;
    double       clip_before       = 0.0;
    double       clip_after        = 0.0;
    double       outlier_thresh_before = 0.0;
    double       outlier_thresh_after  = 0.0;
    double       mse_before        = 0.0;   // rel_frob before the requant
    double       mse_after         = 0.0;   // rel_frob after the requant
    double       frob_before       = 0.0;   // absolute ||w - w_hat||^2 / ||w||^2
    double       frob_after        = 0.0;
    std::string  family;
};

// Push one row into the per-table write buffer. Thread-safe; the
// buffer's MPSC queue + flusher thread serializes the actual SQL.
// Returns 0 on success, non-zero on format / argument failure
// (the buffer's stats are bumped for rows that fail to flush at
// SQL time, but argument failures are returned synchronously).
//
// `buffer` is the ts_db_buffer* for the l4_plan_outcome table;
// typically obtained via ts_db_buffer_open() in ts_dispatch_db_open.
int ts_tessera_db_append_l4_outcome(
    struct ts_db_buffer * buffer,
    const ts_tessera_db_l4_outcome & row);

// --- Appender: bulk GA evaluation logging (the hot path) ---
//
// The GA evaluates 64+ candidates per generation x 100 generations x 254
// tensors = ~1.6M rows. Individual INSERTs would dominate runtime. The
// Appender buffers rows in memory and flushes in one bulk insert per
// generation. One appender is owned per active tensor; BeginRow/EndRow
// bracket each candidate, Flush commits the batch.
struct ts_tessera_db_eval_row {
    std::string  run_id;
    std::string  tensor_name;
    int32_t      generation   = 0;
    int32_t      island       = 0;
    int32_t      candidate_idx = 0;
    float        alpha        = 0.0f;
    float        clip         = 0.0f;
    float        composite    = 0.0f;
    float        mse          = 0.0f;
    float        relative_frob = 0.0f;
};
struct ts_tessera_db_appender;
ts_tessera_db_appender * ts_tessera_db_appender_open(ts_tessera_db * db,
                                                       const std::string & run_id,
                                                       const std::string & tensor_name,
                                                       std::string * err);
int ts_tessera_db_appender_row(ts_tessera_db_appender * ap,
                                const ts_tessera_db_eval_row & row);
int ts_tessera_db_appender_flush(ts_tessera_db_appender * ap);
void ts_tessera_db_appender_close(ts_tessera_db_appender * ap);

// --- Warm-start query ---
//
// Look up the best-known alpha/clip for a family across all prior runs
// (excluding the current one). Used by the dispatch to seed the GA from
// history instead of from scratch. Returns false if no row exists.
struct ts_tessera_db_family_seed {
    float best_alpha     = 0.0f;
    float best_clip      = 0.0f;
    float best_composite = 0.0f;
    std::string tensor_name;   // the prior run's tensor that produced this
};
bool ts_tessera_db_lookup_family_seed(ts_tessera_db * db,
                                       const std::string & family,
                                       const std::string & exclude_run_id,
                                       ts_tessera_db_family_seed * out);

// --- Resumability: list which tensors already converged for this run ---
//
// Populates `out` with tensor_name for every ga_results row of `run_id` whose
// converged flag is true. The dispatch uses this to skip already-finished
// tensors when re-running against an existing DB.
int ts_tessera_db_list_converged(ts_tessera_db * db,
                                  const std::string & run_id,
                                  std::vector<std::string> * out);

// Resume set across runs: list tensor_name for every converged ga_results
// row of any run of `model_hash`. Used by the dispatch to skip tensors that
// already converged in a prior (interrupted) run of the same model. Distinct
// from list_converged (single-run) so the caller does not need to know the
// prior run_id; the model_hash is the cross-run key.
int ts_tessera_db_list_converged_for_model(ts_tessera_db * db,
                                            const std::string & model_hash,
                                            std::vector<std::string> * out);

// Load the full ga_results row for one tensor in this run. Returns false if
// no row exists. Used by the resume path to reconstruct a candidate without
// re-running the GA.
bool ts_tessera_db_load_ga_result(ts_tessera_db * db,
                                   const std::string & run_id,
                                   const std::string & tensor_name,
                                   ts_tessera_db_ga_result * out);

// Resume lookup: find the most recent converged ga_result row for
// (model_hash, tensor_name) across all runs of this model. Returns false if
// no such row exists. Used by the dispatch's layer_skip hook so a re-launch
// after a crash picks up from the last completed tensor of any prior run.
bool ts_tessera_db_load_ga_result_for_model(ts_tessera_db * db,
                                             const std::string & model_hash,
                                             const std::string & tensor_name,
                                             ts_tessera_db_ga_result * out);

// Test-only: run an arbitrary SELECT COUNT(*)... query and return the int64
// result. Used by the e2e test to verify rows landed. Returns -1 on error.
// Not for production code paths (which should use the typed helpers above).
int64_t ts_tessera_db_debug_count(ts_tessera_db * db,
                                   const std::string & query);

// --- Phase 16: model_role migration ---
//
// The unified Gemma4 12B + dspark + dflash + MTP arch has tensors
// with the same name in both the trunk and the drafter. Phase 16
// disambiguates them with a `model_role` column on 7 of the
// unified-schema tables (tensor_stats, l3_outlier_summary,
// l4_probe_summary, l4_plan_outcome, l5_plan_summary, l5_outcome,
// l5_weights). The PK changes to include model_role.
//
// This function is the C++ side of the migration. It is:
//   * Idempotent: re-running on an already-migrated DB is a no-op
//     (each affected table is checked for the model_role column;
//     if present, the rebuild is skipped).
//   * Forward-compatible: a fresh DB created with the new schema
//     (CREATE TABLE IF NOT EXISTS includes model_role) is detected
//     as already-migrated and the function returns 0 without any
//     DDL.
//   * PK-changing: when the table lacks model_role, the function
//     does the standard DuckDB PK-rebuild dance (CREATE TABLE
//     new_name with the new schema, INSERT INTO new_name SELECT
//     *, 'trunk' AS model_role FROM old_name, DROP old_name,
//     ALTER new_name RENAME TO old_name).
//
// Called automatically by ts_tessera_db_open on every open. The
// Python-side tools/tessera/migrate_model_role.py is the
// equivalent migration for Python-only-opened DBs; both are
// idempotent so calling either or both is safe.
int ts_tessera_db_migrate_model_role(ts_tessera_db * db,
                                      std::string * err);

// Test-only: insert one synthetic l5_outcome row. The l5_outcome
// table is normally Python-written (tools/tessera/l5_outcome.py),
// but the C++ converged-fast test in test_l5_dispatch needs a way
// to populate it. Returns 0 on success, non-zero on error. The
// row is keyed on (model_hash, name, iteration, plan_id) so the
// dispatch's ts_tessera_db_l5_outcome_stats_for query can find it.
struct ts_tessera_db_l5_outcome_row {
    std::string  model_hash;
    std::string  model_role;    // Phase 16: same enum as tensor_stats. The
                                // l5_outcome table is per-(model, role,
                                // tensor, iter, plan_id) so the dflash /
                                // dspark / mtp_nextn verdicts don't
                                // collide with the trunk's. Default
                                // "trunk" for pre-Phase-16 rows.
    std::string  name;
    int32_t      layer          = 0;
    int32_t      iteration      = 0;
    std::string  plan_id;
    std::string  family;
    double       sensitivity_score = 0.0;
    double       recommended_alpha = 0.0;
    double       recommended_clip  = 0.0;
    double       mse_before        = 0.0;
    double       mse_after         = 0.0;
    double       delta_mse         = 0.0;
    double       delta_frob        = 0.0;
    bool         plan_accepted     = false;
    double       accept_threshold  = 0.0;
    double       residual          = 0.0;
    double       imatrix_magnitude = 0.0;   // nullable; 0 -> NULL
    double       gradient_proxy    = 0.0;   // nullable; 0 -> NULL
    double       layer_position_prior = 0.0; // nullable; 0 -> NULL
};
int ts_tessera_db_test_insert_l5_outcome(ts_tessera_db * db,
                                          const ts_tessera_db_l5_outcome_row & row);

// --- helpers used by the dispatch ---

// Extract the block index from a tensor name like "blk.12.ffn_gate.weight"
// -> 12. Returns 0 for non-block tensors (embeddings, norm, output).
int32_t ts_tessera_db_layer_depth(const std::string & name);

// SHA256 of the head + tail of a GGUF file (1 MB each). Full-file hashing of
// multi-GB models is too slow on every run; the head+tail fingerprint is
// unique enough for warm-start keying and runs in milliseconds. Returns an
// empty string on failure (the dispatch treats empty hash as "no warm-start").
std::string ts_tessera_db_hash_gguf(const std::string & path);
