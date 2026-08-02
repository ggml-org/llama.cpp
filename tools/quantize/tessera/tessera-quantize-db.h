#pragma once

//
// tessera-quantize-db.h
//
// DuckDB-backed persistent store for the Tessera quantize pipeline.
//
// The quantize pipeline has historically been ephemeral: every run starts the
// GA from scratch, family warm-start lives only in process memory, and the
// only durable artifact is a flat policy JSON. ts_quantize_db wraps a DuckDB
// connection so the pipeline can:
//   - record one row per run, tensor, GA result, and acceptance comparison
//   - stream per-candidate GA evaluations to disk via the Appender API
//   - reload family-optimal alpha/clip on restart to warm-start the GA
//   - skip tensors that already converged in a prior (interrupted) run
//
// The whole store is optional: when the dispatch is constructed with a null
// ts_quantize_db*, every method is a no-op (guarded by the caller's null
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

struct ts_quantize_db {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> conn;

    ts_quantize_db() = default;
    ~ts_quantize_db();
};

// Open (or create) the database at `path` and ensure the schema exists.
// Returns nullptr on failure (message in *err). An in-memory DB (":memory:")
// is supported for tests.
ts_quantize_db * ts_quantize_db_open(const std::string & path,
                                     std::string * err);

// Run-lifecycle hooks. begin_run inserts a new row and returns the run_id
// (a hash of model_hash + config + timestamp). complete_run / fail_run flip
// the status. The run_id is reused across all per-tensor inserts.
std::string ts_quantize_db_begin_run(ts_quantize_db * db,
                                     const std::string & model_path,
                                     const std::string & model_hash,
                                     const std::string & tessera_commit,
                                     const std::string & config_json,
                                     std::string * err);
int ts_quantize_db_complete_run(ts_quantize_db * db,
                                const std::string & run_id,
                                const std::string & status,   // "completed" / "failed"
                                std::string * err);

// Tensor registry. One row per quantizable 2D/3D weight, captured during the
// ga-prep walk. layer_depth is the block index (0 for non-block tensors).
struct ts_quantize_db_tensor {
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
int ts_quantize_db_insert_tensor(ts_quantize_db * db,
                                 const ts_quantize_db_tensor & t,
                                 std::string * err);

// GA results. One row per converged tensor (the summary). Re-inserting a
// (run_id, tensor_name) pair replaces the row (PRIMARY KEY conflict -> upsert).
struct ts_quantize_db_ga_result {
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
int ts_quantize_db_insert_ga_result(ts_quantize_db * db,
                                    const ts_quantize_db_ga_result & r,
                                    std::string * err);

// Acceptance-gate comparison row (one per tensor).
struct ts_quantize_db_acceptance {
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
int ts_quantize_db_insert_acceptance(ts_quantize_db * db,
                                     const ts_quantize_db_acceptance & a,
                                     std::string * err);

// L5 adaptive requantize fixup row.
struct ts_quantize_db_l5_fixup {
    std::string  run_id;
    std::string  tensor_name;
    int32_t      generation   = 0;
    std::string  strategy;       // "A" (alpha/clip) or "B" (outlier)
    float        before_frob   = 0.0f;
    float        after_frob    = 0.0f;
};
int ts_quantize_db_insert_l5_fixup(ts_quantize_db * db,
                                   const ts_quantize_db_l5_fixup & f,
                                   std::string * err);

// --- Appender: bulk GA evaluation logging (the hot path) ---
//
// The GA evaluates 64+ candidates per generation x 100 generations x 254
// tensors = ~1.6M rows. Individual INSERTs would dominate runtime. The
// Appender buffers rows in memory and flushes in one bulk insert per
// generation. One appender is owned per active tensor; BeginRow/EndRow
// bracket each candidate, Flush commits the batch.
struct ts_quantize_db_eval_row {
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
struct ts_quantize_db_appender;
ts_quantize_db_appender * ts_quantize_db_appender_open(ts_quantize_db * db,
                                                       const std::string & run_id,
                                                       const std::string & tensor_name,
                                                       std::string * err);
int ts_quantize_db_appender_row(ts_quantize_db_appender * ap,
                                const ts_quantize_db_eval_row & row);
int ts_quantize_db_appender_flush(ts_quantize_db_appender * ap);
void ts_quantize_db_appender_close(ts_quantize_db_appender * ap);

// --- Warm-start query ---
//
// Look up the best-known alpha/clip for a family across all prior runs
// (excluding the current one). Used by the dispatch to seed the GA from
// history instead of from scratch. Returns false if no row exists.
struct ts_quantize_db_family_seed {
    float best_alpha     = 0.0f;
    float best_clip      = 0.0f;
    float best_composite = 0.0f;
    std::string tensor_name;   // the prior run's tensor that produced this
};
bool ts_quantize_db_lookup_family_seed(ts_quantize_db * db,
                                       const std::string & family,
                                       const std::string & exclude_run_id,
                                       ts_quantize_db_family_seed * out);

// --- Resumability: list which tensors already converged for this run ---
//
// Populates `out` with tensor_name for every ga_results row of `run_id` whose
// converged flag is true. The dispatch uses this to skip already-finished
// tensors when re-running against an existing DB.
int ts_quantize_db_list_converged(ts_quantize_db * db,
                                  const std::string & run_id,
                                  std::vector<std::string> * out);

// Resume set across runs: list tensor_name for every converged ga_results
// row of any run of `model_hash`. Used by the dispatch to skip tensors that
// already converged in a prior (interrupted) run of the same model. Distinct
// from list_converged (single-run) so the caller does not need to know the
// prior run_id; the model_hash is the cross-run key.
int ts_quantize_db_list_converged_for_model(ts_quantize_db * db,
                                            const std::string & model_hash,
                                            std::vector<std::string> * out);

// Load the full ga_results row for one tensor in this run. Returns false if
// no row exists. Used by the resume path to reconstruct a candidate without
// re-running the GA.
bool ts_quantize_db_load_ga_result(ts_quantize_db * db,
                                   const std::string & run_id,
                                   const std::string & tensor_name,
                                   ts_quantize_db_ga_result * out);

// Resume lookup: find the most recent converged ga_result row for
// (model_hash, tensor_name) across all runs of this model. Returns false if
// no such row exists. Used by the dispatch's layer_skip hook so a re-launch
// after a crash picks up from the last completed tensor of any prior run.
bool ts_quantize_db_load_ga_result_for_model(ts_quantize_db * db,
                                             const std::string & model_hash,
                                             const std::string & tensor_name,
                                             ts_quantize_db_ga_result * out);

// Test-only: run an arbitrary SELECT COUNT(*)... query and return the int64
// result. Used by the e2e test to verify rows landed. Returns -1 on error.
// Not for production code paths (which should use the typed helpers above).
int64_t ts_quantize_db_debug_count(ts_quantize_db * db,
                                   const std::string & query);

// --- helpers used by the dispatch ---

// Extract the block index from a tensor name like "blk.12.ffn_gate.weight"
// -> 12. Returns 0 for non-block tensors (embeddings, norm, output).
int32_t ts_quantize_db_layer_depth(const std::string & name);

// SHA256 of the head + tail of a GGUF file (1 MB each). Full-file hashing of
// multi-GB models is too slow on every run; the head+tail fingerprint is
// unique enough for warm-start keying and runs in milliseconds. Returns an
// empty string on failure (the dispatch treats empty hash as "no warm-start").
std::string ts_quantize_db_hash_gguf(const std::string & path);
