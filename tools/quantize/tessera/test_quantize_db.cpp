//
// test_quantize_db.cpp
//
// Standalone test for the DuckDB-backed persistent store. Exercises:
//   - open / schema creation (runs + tensors + ga_evaluations + ga_results
//     + acceptance + l5_fixups)
//   - begin_run / complete_run lifecycle
//   - tensor insert
//   - per-evaluation Appender bulk insert + flush (the GA hot path)
//   - ga_result upsert (ON CONFLICT path)
//   - warm-start family lookup (cross-run)
//   - resumability: list_converged_for_model + load_ga_result
//   - layer_depth extraction
//
// Builds standalone against duckdb-amalgamation + tessera-quantize-db.cpp.
// Run with no args; uses a tmp file. Exit 0 on success, non-zero on failure.
//

#include "tessera-quantize-db.h"
#include "tessera-db-buffer.h"

#ifdef _WIN32
#  define NOMINMAX
#endif
#include "duckdb.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        failures++; \
    } \
} while (0)

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/tmp/tessera-quantize-db-test.db";
    // Always start clean so the schema-create path runs every time.
    std::remove(path);

    std::string err;
    ts_tessera_db * db = ts_tessera_db_open(path, &err);
    CHECK(db != nullptr, ("open failed: " + err).c_str());
    if (db == nullptr) return 1;

    // Begin run
    std::string run_id = ts_tessera_db_begin_run(db, "/tmp/fake.gguf",
                                                  "deadbeef", "test-build",
                                                  "{\"k\":1}", &err);
    CHECK(!run_id.empty(), ("begin_run failed: " + err).c_str());
    CHECK(run_id.size() == 16, "run_id should be 16 hex chars");

    // Insert a tensor
    ts_tessera_db_tensor t;
    t.run_id      = run_id;
    t.name        = "blk.12.ffn_gate.weight";
    t.family      = "ffn";
    t.layer_depth = ts_tessera_db_layer_depth(t.name);
    t.out_dim     = 4096;
    t.in_dim      = 11008;
    t.n_elements  = (int64_t)t.out_dim * t.in_dim;
    t.kurtosis    = 7.5f;
    t.eff_rank    = 0.42f;
    t.source_type = "f16";
    CHECK(ts_tessera_db_insert_tensor(db, t, &err) == 0, "insert_tensor failed");
    CHECK(t.layer_depth == 12, "layer_depth(blk.12.) should be 12");

    // layer_depth edge cases
    CHECK(ts_tessera_db_layer_depth("blk.0.attn_q") == 0, "blk.0 -> 0");
    CHECK(ts_tessera_db_layer_depth("blocks.99.ffn") == 99, "blocks.99 -> 99");
    CHECK(ts_tessera_db_layer_depth("h.7.") == 7, "h.7 -> 7");
    CHECK(ts_tessera_db_layer_depth("output.weight") == 0, "non-block -> 0");
    CHECK(ts_tessera_db_layer_depth("token_embd.weight") == 0, "embd -> 0");

    // Appender: bulk GA eval rows (the hot path)
    ts_tessera_db_appender * ap = ts_tessera_db_appender_open(
        db, run_id, "blk.12.ffn_gate.weight", &err);
    CHECK(ap != nullptr, "appender open failed");
    if (ap) {
        // Simulate 4 generations x 8 candidates = 32 rows.
        for (int32_t g = -1; g < 4; g++) {
            for (int32_t isl = 0; isl < 2; isl++) {
                for (int32_t ci = 0; ci < 4; ci++) {
                    ts_tessera_db_eval_row row;
                    row.run_id        = run_id;
                    row.tensor_name   = "blk.12.ffn_gate.weight";
                    row.generation    = g;
                    row.island        = isl;
                    row.candidate_idx = ci;
                    row.alpha         = 0.5f + 0.01f * ci;
                    row.clip          = 1.0f;
                    row.composite     = -0.001f * (ci + 1);
                    row.mse           = 0.001f * (ci + 1);
                    row.relative_frob = 0.001f * (ci + 1);
                    CHECK(ts_tessera_db_appender_row(ap, row) == 0,
                          "appender row failed");
                }
            }
        }
        CHECK(ts_tessera_db_appender_flush(ap) == 0, "appender flush failed");
        ts_tessera_db_appender_close(ap);
    }

    // Insert ga_result with converged=true
    ts_tessera_db_ga_result gr;
    gr.run_id          = run_id;
    gr.tensor_name     = "blk.12.ffn_gate.weight";
    gr.family          = "ffn";
    gr.best_alpha      = 0.53f;
    gr.best_clip       = 1.0f;
    gr.best_composite  = -0.0012f;
    gr.best_mse        = 0.0012f;
    gr.generations_run = 12;
    gr.n_evaluations   = 96;
    gr.converged       = true;
    gr.warm_started    = false;
    CHECK(ts_tessera_db_insert_ga_result(db, gr, &err) == 0,
          "insert_ga_result failed");

    // Re-insert should upsert (not duplicate). Same PK -> row replaced.
    gr.generations_run = 15;
    CHECK(ts_tessera_db_insert_ga_result(db, gr, &err) == 0,
          "insert_ga_result upsert failed");

    // Insert acceptance row
    ts_tessera_db_acceptance ar;
    ar.run_id       = run_id;
    ar.tensor_name  = "blk.12.ffn_gate.weight";
    ar.family       = "ffn";
    ar.composite_t2 = 0.0012f;
    ar.awq_t2       = 0.0015f;
    ar.rotation_t2  = 0.0014f;
    ar.lowrank_t2   = 0.0013f;
    ar.hessian_t2   = 0.0016f;
    ar.verdict      = "pass";
    CHECK(ts_tessera_db_insert_acceptance(db, ar, &err) == 0,
          "insert_acceptance failed");

    // Insert L5 fixup
    ts_tessera_db_l5_fixup fx;
    fx.run_id       = run_id;
    fx.tensor_name  = "blk.12.ffn_gate.weight";
    fx.generation   = 0;
    fx.strategy     = "A";
    fx.before_frob  = 0.0020f;
    fx.after_frob   = 0.0015f;
    CHECK(ts_tessera_db_insert_l5_fixup(db, fx, &err) == 0,
          "insert_l5_fixup failed");

    // Complete the run
    CHECK(ts_tessera_db_complete_run(db, run_id, "completed", &err) == 0,
          "complete_run failed");

    // Reload ga_result
    ts_tessera_db_ga_result loaded;
    CHECK(ts_tessera_db_load_ga_result(db, run_id, "blk.12.ffn_gate.weight",
                                        &loaded),
          "load_ga_result should find the row");
    CHECK(loaded.generations_run == 15, "upserted generations_run should be 15");
    CHECK(loaded.converged, "loaded converged flag should be true");
    CHECK(loaded.best_alpha > 0.52f && loaded.best_alpha < 0.54f,
          "loaded alpha matches");

    // Resumability: list converged for the model
    std::vector<std::string> done;
    CHECK(ts_tessera_db_list_converged_for_model(db, "deadbeef", &done) == 0,
          "list_converged_for_model failed");
    CHECK(done.size() == 1, "should have 1 converged tensor for this model");
    CHECK(!done.empty() && done[0] == "blk.12.ffn_gate.weight",
          "converged tensor name matches");

    // Resumability: list converged for THIS run_id
    std::vector<std::string> done_run;
    CHECK(ts_tessera_db_list_converged(db, run_id, &done_run) == 0,
          "list_converged failed");
    CHECK(done_run.size() == 1, "should have 1 converged tensor for this run");

    // Warm-start: open a SECOND run for the same model, look up the family seed.
    std::string run_id2 = ts_tessera_db_begin_run(db, "/tmp/fake.gguf",
                                                   "deadbeef", "test-build",
                                                   "{\"k\":2}", &err);
    CHECK(!run_id2.empty(), "begin_run #2 failed");
    CHECK(run_id2 != run_id, "run_id should differ across runs");
    ts_tessera_db_family_seed seed;
    CHECK(ts_tessera_db_lookup_family_seed(db, "ffn", run_id2, &seed),
          "warm-start lookup should find a seed for 'ffn'");
    CHECK(seed.best_alpha > 0.52f && seed.best_alpha < 0.54f,
          "seed alpha matches the prior run");
    CHECK(seed.tensor_name == "blk.12.ffn_gate.weight",
          "seed tensor_name matches");

    // No seed for unknown family
    ts_tessera_db_family_seed seed_x;
    CHECK(!ts_tessera_db_lookup_family_seed(db, "router", run_id2, &seed_x),
          "warm-start lookup should miss for unknown family");

    // Mark run2 failed (exercises the failed status path)
    CHECK(ts_tessera_db_complete_run(db, run_id2, "failed", &err) == 0,
          "complete_run(failed) failed");

    // l4_plan_outcome: the feedback-loop audit trail. Verifies the
    // ts_tessera_db_append_l4_outcome helper writes one row per
    // (tensor, iteration, plan_id), with before/after fields intact
    // and the buffer's MPSC flusher landing them.
    {
        std::vector<std::string> cols = {
            "model_hash", "model_role", "name", "layer", "iteration",
            "plan_id", "strategy", "alpha_before", "alpha_after",
            "clip_before", "clip_after", "outlier_thresh_before",
            "outlier_thresh_after", "mse_before", "mse_after",
            "frob_before", "frob_after", "family", "updated_at",
        };
        ts_db_buffer * buf = ts_db_buffer_open(
            db, "l4_plan_outcome", cols,
            /*flush_threshold=*/16, std::chrono::milliseconds(50));
        CHECK(buf != nullptr, "l4 buffer open");
        if (buf != nullptr) {
            // Write 3 rows: tensor A at iter 0/1/2, strategy A.
            for (int it = 0; it < 3; it++) {
                ts_tessera_db_l4_outcome row;
                row.model_hash   = "hash_l4";
                row.name         = "blk.5.attn_q.weight";
                row.layer        = 5;
                row.iteration    = it;
                row.plan_id      = "cpp_quant_gen" + std::to_string(it) + "_stageA";
                row.strategy     = "A";
                row.alpha_before = 0.5f;
                row.alpha_after  = 0.25f;
                row.clip_before  = 1.0f;
                row.clip_after   = 0.5f;
                row.outlier_thresh_before = 0.05f;
                row.outlier_thresh_after  = 0.05f;
                row.mse_before   = 0.012f;
                row.mse_after    = 0.012f - 0.001f * (it + 1);
                row.frob_before  = 0.020f;
                row.frob_after   = 0.020f - 0.002f * (it + 1);
                row.family       = "attn_q";
                CHECK(ts_tessera_db_append_l4_outcome(buf, row) == 0,
                      "append_l4_outcome failed");
            }
            // One row at iter 0, strategy B, different tensor.
            {
                ts_tessera_db_l4_outcome row;
                row.model_hash   = "hash_l4";
                row.name         = "blk.5.ffn_gate.weight";
                row.layer        = 5;
                row.iteration    = 0;
                row.plan_id      = "cpp_quant_gen0_stageB";
                row.strategy     = "B";
                row.mse_before   = 0.025f;
                row.mse_after    = 0.022f;
                row.frob_before  = 0.040f;
                row.frob_after   = 0.035f;
                row.family       = "ffn_gate";
                CHECK(ts_tessera_db_append_l4_outcome(buf, row) == 0, "append B");
            }
            ts_db_buffer_close(&buf);
            CHECK(buf == nullptr, "l4 close nulled the handle");
        }
        int64_t n = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM l4_plan_outcome WHERE model_hash = 'hash_l4'");
        CHECK(n == 4, "4 outcome rows landed");
        int64_t stage_b = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM l4_plan_outcome WHERE plan_id = 'cpp_quant_gen0_stageB'");
        CHECK(stage_b == 1, "stage B row findable by plan_id");
    }

    // ---- l5_weights: the per-(model, family) retuned scoring weights ----
    // Verifies the ts_tessera_db_upsert_l5_weight helper:
    //   * writes a row for (model_hash, family)
    //   * on a re-write, overwrites all columns (PRIMARY KEY is the
    //     upsert target)
    //   * bias / n_samples / in_sample_loss / hit_rate are preserved
    //   * retune_source records the algorithm tag
    {
        ts_tessera_db_l5_weight row;
        row.model_hash      = "hash_weights";
        row.family          = "attn_q";
        row.w_imatrix       = 0.42;
        row.w_gradient      = 0.33;
        row.w_layer         = 0.25;
        row.bias            = -0.001;
        row.n_samples       = 50;
        row.in_sample_loss  = 0.0008;
        row.hit_rate        = 0.80;
        row.retune_source   = "ols_slope_v1";
        std::string err;
        CHECK(ts_tessera_db_upsert_l5_weight(db, row, &err) == 0,
              ("upsert_l5_weight failed: " + err).c_str());
        // Re-write with different weights -> upsert overwrites.
        row.w_imatrix  = 0.30;
        row.w_gradient = 0.50;
        row.w_layer    = 0.20;
        row.n_samples  = 75;
        row.hit_rate   = 0.85;
        CHECK(ts_tessera_db_upsert_l5_weight(db, row, &err) == 0,
              ("upsert_l5_weight re-write failed: " + err).c_str());
        // Second family, same model.
        ts_tessera_db_l5_weight ffn;
        ffn.model_hash      = "hash_weights";
        ffn.family          = "ffn_gate";
        ffn.w_imatrix       = 0.50;
        ffn.w_gradient      = 0.30;
        ffn.w_layer         = 0.20;
        ffn.bias            = 0.002;
        ffn.n_samples       = 30;
        ffn.in_sample_loss  = 0.0015;
        ffn.hit_rate        = 0.55;
        ffn.retune_source   = "ols_slope_v1";
        CHECK(ts_tessera_db_upsert_l5_weight(db, ffn, &err) == 0,
              ("upsert_l5_weight ffn failed: " + err).c_str());

        int64_t n = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM l5_weights WHERE model_hash = 'hash_weights'");
        CHECK(n == 2, "2 weight rows landed (one per family)");
        // attn_q got the second-write values.
        int64_t n_im = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM l5_weights WHERE model_hash = 'hash_weights' "
                "AND family = 'attn_q' AND w_gradient = 0.50 AND n_samples = 75");
        CHECK(n_im == 1, "attn_q row reflects the upsert (w_gradient=0.50, n_samples=75)");
        int64_t n_source = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM l5_weights WHERE retune_source = 'ols_slope_v1'");
        CHECK(n_source == 2, "retune_source tag is preserved across both rows");

        // ---- requant_budget_bits: round-trip the new BIGINT column ----
        // Phase 14 contract: C++ writes a budget when l5_retune.py
        // projects one; the C++ side reads it but does not act on it
        // yet. The -1 sentinel in the struct maps to NULL in DuckDB;
        // a real value round-trips as itself.
        {
            // First: NULL round-trip via the -1 sentinel.
            ts_tessera_db_l5_weight null_row = ffn;   // copy the ffn row
            null_row.family = "ffn_gate";
            null_row.requant_budget_bits = -1;        // NULL
            CHECK(ts_tessera_db_upsert_l5_weight(db, null_row, &err) == 0,
                  ("upsert with NULL budget failed: " + err).c_str());
            int64_t n_null = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights WHERE "
                    "model_hash = 'hash_weights' AND family = 'ffn_gate' "
                    "AND requant_budget_bits IS NULL");
            CHECK(n_null == 1, "ffn_gate requant_budget_bits is NULL after upsert");

            // Second: real value round-trip.
            null_row.requant_budget_bits = 4096;
            CHECK(ts_tessera_db_upsert_l5_weight(db, null_row, &err) == 0,
                  ("upsert with real budget failed: " + err).c_str());
            int64_t n_val = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights WHERE "
                    "model_hash = 'hash_weights' AND family = 'ffn_gate' "
                    "AND requant_budget_bits = 4096");
            CHECK(n_val == 1, "ffn_gate requant_budget_bits = 4096 after upsert");

            // Third: attn_q stays at the second-write (no budget set, NULL).
            int64_t n_attn_null = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights WHERE "
                    "model_hash = 'hash_weights' AND family = 'attn_q' "
                    "AND requant_budget_bits IS NULL");
            CHECK(n_attn_null == 1, "attn_q requant_budget_bits is NULL (untouched)");

            // Fourth: zero is a valid budget (zero-budget edge case),
            // distinguishable from NULL.
            null_row.requant_budget_bits = 0;
            CHECK(ts_tessera_db_upsert_l5_weight(db, null_row, &err) == 0,
                  "upsert with zero budget");
            int64_t n_zero = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights WHERE "
                    "model_hash = 'hash_weights' AND family = 'ffn_gate' "
                    "AND requant_budget_bits = 0");
            CHECK(n_zero == 1, "zero budget is preserved (not collapsed to NULL)");
        }
    }

    // ---- ts_tessera_db_list_l5_weights: typed reader for the GA-prep ----
    // The dispatch's GA-prep walk uses this to bias the GA's initial
    // (alpha, clip) seed per family. Verifies:
    //   * empty family arg returns all families for the model
    //   * non-empty family arg returns just that family
    //   * entries are ordered by hit_rate DESC
    //   * requant_budget_bits round-trips (-1 sentinel -> NULL -> -1)
    //   * unknown model_hash returns an empty list (no rows)
    {
        // The ffn_gate row above has requant_budget_bits = 0 (set
        // just above). Re-upsert both rows with explicit values
        // for a clean list-ordering assertion.
        ts_tessera_db_l5_weight a;
        a.model_hash     = "hash_list";
        a.family         = "attn_q";
        a.w_imatrix      = 0.40;
        a.w_gradient     = 0.40;
        a.w_layer        = 0.20;
        a.bias           = -0.0005;
        a.n_samples      = 100;
        a.in_sample_loss = 0.0008;
        a.hit_rate       = 0.85;
        a.requant_budget_bits = 2048;
        a.retune_source  = "ols_slope_v1";
        CHECK(ts_tessera_db_upsert_l5_weight(db, a, &err) == 0, "list upsert a");

        ts_tessera_db_l5_weight b;
        b.model_hash     = "hash_list";
        b.family         = "ffn_gate";
        b.w_imatrix      = 0.50;
        b.w_gradient     = 0.30;
        b.w_layer        = 0.20;
        b.bias           = 0.0010;
        b.n_samples      = 60;
        b.in_sample_loss = 0.0012;
        b.hit_rate       = 0.97;   // higher than attn_q
        b.requant_budget_bits = -1; // NULL
        b.retune_source  = "ols_slope_v1";
        CHECK(ts_tessera_db_upsert_l5_weight(db, b, &err) == 0, "list upsert b");

        // Empty family = all families.
        ts_tessera_db_l5_weight_list all;
        CHECK(ts_tessera_db_list_l5_weights(db, "hash_list", "", &all) == 0,
              "list_l5_weights (all) failed");
        CHECK(all.entries.size() == 2, "list returns 2 entries");
        // hit_rate DESC: ffn_gate (0.97) first, attn_q (0.85) second.
        CHECK(all.entries[0].family == "ffn_gate",
              "first entry is the higher-hit_rate family");
        CHECK(all.entries[0].hit_rate > all.entries[1].hit_rate,
              "ordering is hit_rate DESC");
        CHECK(all.entries[0].requant_budget_bits == -1,
              "ffn_gate round-trips NULL -> -1");
        CHECK(all.entries[1].requant_budget_bits == 2048,
              "attn_q round-trips 2048");
        CHECK(all.entries[1].n_samples == 100,
              "attn_q n_samples round-trips");
        CHECK(all.entries[1].retune_source == "ols_slope_v1",
              "attn_q retune_source round-trips");

        // Non-empty family = single entry.
        ts_tessera_db_l5_weight_list one;
        CHECK(ts_tessera_db_list_l5_weights(db, "hash_list", "attn_q", &one) == 0,
              "list_l5_weights (single) failed");
        CHECK(one.entries.size() == 1, "single-family filter returns 1");
        CHECK(one.entries[0].family == "attn_q", "filter is exact");

        // Unknown model_hash -> empty.
        ts_tessera_db_l5_weight_list none;
        CHECK(ts_tessera_db_list_l5_weights(db, "no_such_model", "", &none) == 0,
              "list_l5_weights (unknown) failed");
        CHECK(none.entries.empty(), "unknown model returns 0 entries");

        // Unknown family on a known model -> empty.
        ts_tessera_db_l5_weight_list none_fam;
        CHECK(ts_tessera_db_list_l5_weights(db, "hash_list", "router", &none_fam) == 0,
              "list_l5_weights (unknown family) failed");
        CHECK(none_fam.entries.empty(), "unknown family on known model -> 0");

        // Empty model_hash is a no-op (returns 0, no error).
        ts_tessera_db_l5_weight_list empty_mh;
        CHECK(ts_tessera_db_list_l5_weights(db, "", "", &empty_mh) == 0,
              "list_l5_weights (empty model_hash) failed");
        CHECK(empty_mh.entries.empty(), "empty model_hash -> 0 entries");
    }

    // ---- ts_tessera_db_l5_outcome_stats_for: the converged-fast gate ----
    // The dispatch's early-exit uses this to ask "does l5_outcome
    // already say this model converges?" The l5_outcome table is
    // Python-written (tools/tessera/l5_outcome.py), so this C++
    // test exercises the production path on a model that has no
    // l5_outcome rows yet: n_rows=0, n_accepted=0, hit_rate=0.
    // The non-empty case (hit_rate > 0.95) is covered by
    // test_tessera_l5_outcome.py on the Python side and by the
    // new test_l5_dispatch warm-start path.
    {
        ts_tessera_db_l5_outcome_stats s;
        CHECK(ts_tessera_db_l5_outcome_stats_for(db, "no_l5o_yet", "attn_q",
                                                 &s) == 0,
              "l5_outcome_stats_for (no rows) failed");
        CHECK(s.n_rows == 0, "n_rows=0 for empty model");
        CHECK(s.n_accepted == 0, "n_accepted=0 for empty model");
        CHECK(s.hit_rate == 0.0, "hit_rate=0 for empty model");
        CHECK(s.family == "attn_q", "family echo");

        // Empty family = all-families aggregate (still 0 rows).
        ts_tessera_db_l5_outcome_stats s_all;
        CHECK(ts_tessera_db_l5_outcome_stats_for(db, "no_l5o_yet", "",
                                                 &s_all) == 0,
              "l5_outcome_stats_for (empty family) failed");
        CHECK(s_all.n_rows == 0, "n_rows=0 across families");
        CHECK(s_all.family.empty(), "empty family arg echoes empty");

        // Empty model_hash is a no-op (returns 0, no error).
        ts_tessera_db_l5_outcome_stats s_empty;
        CHECK(ts_tessera_db_l5_outcome_stats_for(db, "", "attn_q",
                                                 &s_empty) == 0,
              "l5_outcome_stats_for (empty model) failed");
        CHECK(s_empty.n_rows == 0, "empty model -> 0 rows");
    }

    // ---- tensor_stats recommended_action round-trip ----------------
    // The recommended_action column is the per-tensor verdict the
    // Python calibration_to_tensor_stats.py writes from
    // l5_weights via the l5_action rules. The C++ side just
    // carries the value through the upsert; the test confirms:
    //   * the column accepts the documented string values
    //   * on a re-write, the new value overwrites (same contract
    //     as the other Python-side columns: rms / mean_abs /
    //     tail_ratio)
    //   * a fresh write with an empty string is a no-op (the
    //     C++ GA-prep walk never sets this field).
    {
        ts_tessera_db_tensor_stat row;
        row.model_hash         = "hash_action";
        row.name               = "blk.0.attn_q.weight";
        row.family             = "attn_q";
        row.layer_depth        = 0;
        row.out_dim            = 4096;
        row.in_dim             = 4096;
        row.n_elements         = 16777216;
        row.dtype              = "f16";
        row.kurtosis           = 5.0;
        row.eff_rank           = 0.85;
        row.rms                = 0.10;
        row.mean_abs           = 0.08;
        row.tail_ratio         = 4.0;
        row.source             = "py_cal";
        row.recommended_action = "protect";
        std::string err;
        CHECK(ts_tessera_db_upsert_tensor_stat(db, row, &err) == 0,
              ("upsert_tensor_stat w/ recommended_action failed: " + err).c_str());
        // Second tensor: requant_up verdict.
        ts_tessera_db_tensor_stat row2 = row;
        row2.name               = "blk.0.ffn_gate.weight";
        row2.family             = "ffn_gate";
        row2.recommended_action = "requant_up";
        CHECK(ts_tessera_db_upsert_tensor_stat(db, row2, &err) == 0,
              ("upsert_tensor_stat row2 failed: " + err).c_str());
        // Verify via direct SQL: protect landed, requant_up landed.
        int64_t n_protect = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'hash_action' "
                "AND name = 'blk.0.attn_q.weight' "
                "AND recommended_action = 'protect'");
        CHECK(n_protect == 1,
              "recommended_action='protect' round-trip on the first write");
        int64_t n_requant = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'hash_action' "
                "AND name = 'blk.0.ffn_gate.weight' "
                "AND recommended_action = 'requant_up'");
        CHECK(n_requant == 1,
              "recommended_action='requant_up' round-trip on the second write");
        // Re-write: new value overwrites. The column is part of
        // the upsert target; the new value wins.
        row.recommended_action = "monitor";
        CHECK(ts_tessera_db_upsert_tensor_stat(db, row, &err) == 0,
              ("upsert_tensor_stat re-write failed: " + err).c_str());
        int64_t n_monitor = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'hash_action' "
                "AND name = 'blk.0.attn_q.weight' "
                "AND recommended_action = 'monitor'");
        CHECK(n_monitor == 1,
              "recommended_action overwrite on re-write (protect -> monitor)");
        // Empty string is a no-op semantic: it does not get
        // NULL'd by the upsert, but it does not break the row
        // either. The C++ side never sets this field; an empty
        // value is a no-op equivalent of NULL.
        row.recommended_action = "";
        CHECK(ts_tessera_db_upsert_tensor_stat(db, row, &err) == 0,
              ("upsert_tensor_stat empty action failed: " + err).c_str());
        int64_t n_empty = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'hash_action' "
                "AND name = 'blk.0.attn_q.weight' "
                "AND recommended_action = ''");
        CHECK(n_empty == 1,
              "recommended_action='' round-trip is a no-op semantic");
    }

    // ---- Phase 16: model_role round-trip on the 4 C++ structs ----
    // The unified Gemma4 12B + dspark + dflash + MTP arch has
    // tensors with the same name in both the trunk and the
    // drafter (e.g. blk.0.attn_q.weight). The model_role column
    // disambiguates them on the (model_hash, model_role, ...)
    // PKs. The C++ side carries model_role through the 4 row
    // structs (ts_tessera_db_tensor_stat,
    // ts_tessera_db_l4_outcome, ts_tessera_db_l5_outcome_row,
    // ts_tessera_db_l5_weight) and the upsert / append / test
    // helpers. The default "trunk" preserves the pre-Phase-16
    // contract (an empty model_role in the struct -> "trunk"
    // in the SQL).
    //
    // The test exercises:
    //   * default "trunk" on a struct that leaves model_role
    //     empty (preserves the pre-Phase-16 contract)
    //   * explicit "dflash" / "mtp_nextn" values round-trip
    //   * the same (model_hash, name) with two different
    //     model_role values coexist on the new PK
    //   * the 4 helpers (upsert_tensor_stat, append_l4_outcome,
    //     test_insert_l5_outcome, upsert_l5_weight) all carry
    //     model_role through
    //   * the list_l5_weights reader echoes model_role in
    //     ts_tessera_db_l5_weight_list_entry
    {
        // ---- tensor_stat: default "trunk" on empty model_role ----
        ts_tessera_db_tensor_stat def_row;
        def_row.model_hash = "p16_default";
        def_row.model_role = "";   // -> "trunk" in the SQL
        def_row.name       = "blk.0.attn_q.weight";
        def_row.family     = "attn_q";
        def_row.layer_depth = 0;
        def_row.kurtosis   = 3.0;
        def_row.eff_rank   = 0.85;
        def_row.source     = "py_cal";
        std::string err;
        CHECK(ts_tessera_db_upsert_tensor_stat(db, def_row, &err) == 0,
              ("upsert_tensor_stat default role failed: " + err).c_str());
        int64_t n_default = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'p16_default' "
                "AND name = 'blk.0.attn_q.weight' "
                "AND model_role = 'trunk'");
        CHECK(n_default == 1,
              "default model_role=empty -> 'trunk' on tensor_stat");

        // ---- tensor_stat: explicit "dflash" --------------------
        ts_tessera_db_tensor_stat dflash_row = def_row;
        dflash_row.model_role = "dflash";
        dflash_row.kurtosis   = 4.0;   // different value
        dflash_row.source     = "py_cal_dflash";
        CHECK(ts_tessera_db_upsert_tensor_stat(db, dflash_row, &err) == 0,
              ("upsert_tensor_stat dflash failed: " + err).c_str());
        int64_t n_dflash = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'p16_default' "
                "AND name = 'blk.0.attn_q.weight' "
                "AND model_role = 'dflash'");
        CHECK(n_dflash == 1,
              "explicit model_role='dflash' on tensor_stat");
        // The two rows coexist: same (model_hash, name),
        // different model_role.
        int64_t n_total = ts_tessera_db_debug_count(
            db, "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'p16_default' "
                "AND name = 'blk.0.attn_q.weight'");
        CHECK(n_total == 2,
              "trunk + dflash coexist on the new PK");

        // ---- tensor_stat: mtp_nextn ----------------------------
        ts_tessera_db_tensor_stat mtp_row = def_row;
        mtp_row.model_role = "mtp_nextn";
        mtp_row.name       = "blk.0.nextn_proj.weight";
        mtp_row.family     = "nextn";
        mtp_row.kurtosis   = 5.0;
        CHECK(ts_tessera_db_upsert_tensor_stat(db, mtp_row, &err) == 0,
              ("upsert_tensor_stat mtp_nextn failed: " + err).c_str());

        // ---- l4_outcome: default + dflash ----------------------
        // The l4_outcome struct is written via the per-table
        // buffer (ts_db_buffer). Open a buffer for the test
        // and append two rows (one trunk, one dflash).
        {
            std::vector<std::string> cols = {
                "model_hash", "model_role", "name", "layer", "iteration",
                "plan_id", "strategy", "alpha_before", "alpha_after",
                "clip_before", "clip_after", "outlier_thresh_before",
                "outlier_thresh_after", "mse_before", "mse_after",
                "frob_before", "frob_after", "family", "updated_at",
            };
            ts_db_buffer * buf = ts_db_buffer_open(
                db, "l4_plan_outcome", cols,
                /*flush_threshold=*/16, std::chrono::milliseconds(50));
            CHECK(buf != nullptr, "l4 buffer open (p16)");
            if (buf != nullptr) {
                // Row 1: default trunk (empty model_role).
                ts_tessera_db_l4_outcome l4_trunk;
                l4_trunk.model_hash = "p16_l4";
                l4_trunk.model_role = "";   // -> "trunk"
                l4_trunk.name       = "blk.0.attn_q.weight";
                l4_trunk.layer      = 0;
                l4_trunk.iteration  = 0;
                l4_trunk.plan_id    = "p0";
                l4_trunk.strategy   = "A";
                l4_trunk.mse_before = 0.012f;
                l4_trunk.mse_after  = 0.010f;
                l4_trunk.family     = "attn_q";
                CHECK(ts_tessera_db_append_l4_outcome(buf, l4_trunk) == 0,
                      "append_l4_outcome trunk");
                // Row 2: explicit dflash.
                ts_tessera_db_l4_outcome l4_dflash = l4_trunk;
                l4_dflash.model_role = "dflash";
                l4_dflash.mse_before = 0.020f;
                l4_dflash.mse_after  = 0.018f;
                CHECK(ts_tessera_db_append_l4_outcome(buf, l4_dflash) == 0,
                      "append_l4_outcome dflash");
                ts_db_buffer_close(&buf);
            }
            int64_t n_l4 = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l4_plan_outcome "
                    "WHERE model_hash = 'p16_l4'");
            CHECK(n_l4 == 2, "l4_outcome: 2 rows (trunk + dflash)");
            int64_t n_l4_dflash = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l4_plan_outcome "
                    "WHERE model_hash = 'p16_l4' "
                    "AND model_role = 'dflash'");
            CHECK(n_l4_dflash == 1,
                  "l4_outcome: dflash row landed");
            // Default -> "trunk" on the SQL side.
            int64_t n_l4_trunk = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l4_plan_outcome "
                    "WHERE model_hash = 'p16_l4' "
                    "AND model_role = 'trunk'");
            CHECK(n_l4_trunk == 1,
                  "l4_outcome: empty model_role -> 'trunk'");
        }

        // ---- l5_outcome_row: default + dflash (test-only INSERT) ----
        {
            // Trunk row.
            ts_tessera_db_l5_outcome_row l5_trunk;
            l5_trunk.model_hash  = "p16_l5";
            l5_trunk.model_role  = "";   // -> "trunk"
            l5_trunk.name        = "blk.0.attn_q.weight";
            l5_trunk.layer       = 0;
            l5_trunk.iteration   = 0;
            l5_trunk.plan_id     = "p0";
            l5_trunk.family      = "attn_q";
            l5_trunk.sensitivity_score = 0.5;
            l5_trunk.mse_before  = 0.012;
            l5_trunk.mse_after   = 0.010;
            l5_trunk.delta_mse   = -0.002;
            l5_trunk.plan_accepted = true;
            CHECK(ts_tessera_db_test_insert_l5_outcome(db, l5_trunk) == 0,
                  "test_insert_l5_outcome trunk");
            // Dflash row.
            ts_tessera_db_l5_outcome_row l5_dflash = l5_trunk;
            l5_dflash.model_role  = "dflash";
            l5_dflash.sensitivity_score = 0.6;
            l5_dflash.mse_before  = 0.020;
            l5_dflash.mse_after   = 0.018;
            l5_dflash.delta_mse   = -0.002;
            CHECK(ts_tessera_db_test_insert_l5_outcome(db, l5_dflash) == 0,
                  "test_insert_l5_outcome dflash");
            int64_t n_l5 = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_outcome "
                    "WHERE model_hash = 'p16_l5'");
            CHECK(n_l5 == 2, "l5_outcome: 2 rows (trunk + dflash)");
            int64_t n_l5_dflash = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_outcome "
                    "WHERE model_hash = 'p16_l5' "
                    "AND model_role = 'dflash'");
            CHECK(n_l5_dflash == 1,
                  "l5_outcome: dflash row landed");
            int64_t n_l5_trunk = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_outcome "
                    "WHERE model_hash = 'p16_l5' "
                    "AND model_role = 'trunk'");
            CHECK(n_l5_trunk == 1,
                  "l5_outcome: empty model_role -> 'trunk'");
        }

        // ---- l5_weight: trunk + dflash + reader echoes model_role ----
        {
            ts_tessera_db_l5_weight w_trunk;
            w_trunk.model_hash = "p16_w";
            w_trunk.model_role = "trunk";
            w_trunk.family     = "attn_q";
            w_trunk.w_imatrix  = 0.4;
            w_trunk.w_gradient = 0.3;
            w_trunk.w_layer    = 0.3;
            w_trunk.hit_rate   = 0.7;
            w_trunk.n_samples  = 10;
            CHECK(ts_tessera_db_upsert_l5_weight(db, w_trunk, &err) == 0,
                  ("upsert_l5_weight trunk: " + err).c_str());
            // Dflash: same model + family, different model_role.
            ts_tessera_db_l5_weight w_dflash;
            w_dflash.model_hash = "p16_w";
            w_dflash.model_role = "dflash";
            w_dflash.family     = "attn_q";
            w_dflash.w_imatrix  = 0.2;
            w_dflash.w_gradient = 0.5;
            w_dflash.w_layer    = 0.3;
            w_dflash.hit_rate   = 0.6;
            w_dflash.n_samples  = 5;
            CHECK(ts_tessera_db_upsert_l5_weight(db, w_dflash, &err) == 0,
                  ("upsert_l5_weight dflash: " + err).c_str());
            int64_t n_w = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights "
                    "WHERE model_hash = 'p16_w'");
            CHECK(n_w == 2, "l5_weights: 2 rows (trunk + dflash)");
            // Re-write on the same PK overwrites; the other
            // (model_role, family) row is untouched.
            w_dflash.w_gradient = 0.6;
            w_dflash.n_samples  = 8;
            CHECK(ts_tessera_db_upsert_l5_weight(db, w_dflash, &err) == 0,
                  ("upsert_l5_weight re-write: " + err).c_str());
            // The trunk row is still 0.3 (untouched by the
            // dflash re-write; different PK).
            int64_t n_trunk_unchanged = ts_tessera_db_debug_count(
                db, "SELECT COUNT(*) FROM l5_weights "
                    "WHERE model_hash = 'p16_w' AND model_role = 'trunk' "
                    "AND w_gradient = 0.3");
            CHECK(n_trunk_unchanged == 1,
                  "trunk row untouched by dflash re-write (different PK)");

            // list_l5_weights echoes model_role.
            ts_tessera_db_l5_weight_list all;
            CHECK(ts_tessera_db_list_l5_weights(db, "p16_w", "", &all) == 0,
                  "list_l5_weights (p16) failed");
            CHECK(all.entries.size() == 2, "list returns 2 entries");
            // hit_rate DESC: dflash (0.6) first, trunk (0.7) wait...
            // hit_rate = 0.7 > 0.6, so trunk is first. Just check
            // both model_role values echo.
            bool seen_trunk = false, seen_dflash = false;
            for (const auto & e : all.entries) {
                if (e.model_role == "trunk")  seen_trunk  = true;
                if (e.model_role == "dflash") seen_dflash = true;
            }
            CHECK(seen_trunk,
                  "list echoes model_role='trunk'");
            CHECK(seen_dflash,
                  "list echoes model_role='dflash'");
        }

        // ---- migration: fresh DB is no-op; old-shape DB is migrated ----
        // Open a separate DB on disk and seed a pre-Phase-16
        // (no model_role) schema by hand, then verify the
        // migration rebuilds the PK and backfills model_role.
        // The C++ open path calls ts_tessera_db_migrate_model_role
        // on every open; here we test the function directly.
        //
        // The pre-Phase-16 schema is created via raw SQL
        // (model_role is part of the new PK, so we cannot
        // DROP COLUMN on a freshly-opened DB; we must
        // create the old shape from scratch).
        {
            const std::string pre_path = std::string(path) + ".pre16";
            std::remove(pre_path.c_str());
            // Create a fresh DuckDB file with the
            // pre-Phase-16 schema (no model_role, original
            // PKs). We bypass the C++ open path because
            // ts_tessera_db_open always runs the new schema
            // setup, which would already give us the
            // model_role column. Instead, we open a raw
            // duckdb::DuckDB / duckdb::Connection and apply
            // the pre-Phase-16 DDL ourselves.
            std::unique_ptr<duckdb::DuckDB>     pre_duck;
            std::unique_ptr<duckdb::Connection> pre_conn;
            try {
                pre_duck.reset(new duckdb::DuckDB(pre_path));
                pre_conn.reset(new duckdb::Connection(*pre_duck));
                // Pre-Phase-16 schema: no model_role, original
                // PK (model_hash, name). The full column list
                // matches the migration's INSERT ... SELECT
                // (every column the migration carries through
                // must exist on the source table).
                pre_conn->Query(
                    "CREATE TABLE tensor_stats ("
                    "    model_hash         TEXT NOT NULL,"
                    "    name               TEXT NOT NULL,"
                    "    family             TEXT,"
                    "    layer_depth        INTEGER,"
                    "    out_dim            BIGINT,"
                    "    in_dim             BIGINT,"
                    "    n_elements         BIGINT,"
                    "    dtype              TEXT,"
                    "    kurtosis           DOUBLE,"
                    "    eff_rank           DOUBLE,"
                    "    rms                DOUBLE,"
                    "    mean_abs           DOUBLE,"
                    "    tail_ratio         DOUBLE,"
                    "    source             TEXT,"
                    "    recommended_action TEXT,"
                    "    updated_at         TIMESTAMP,"
                    "    PRIMARY KEY (model_hash, name)"
                    ")");
                pre_conn->Query(
                    "INSERT INTO tensor_stats "
                    "(model_hash, name, family, layer_depth, "
                    " out_dim, in_dim, n_elements, dtype, "
                    " kurtosis, eff_rank, source) "
                    "VALUES "
                    "('pre_model', 'blk.0.attn_q.weight', 'attn_q', 0, "
                    " 4096, 4096, 16777216, 'f16', "
                    " 5.0, 0.85, 'py_cal')");
            } catch (const std::exception & e) {
                CHECK(false,
                      ("pre-Phase-16 setup exception: " +
                       std::string(e.what())).c_str());
            } catch (...) {
                CHECK(false, "pre-Phase-16 setup unknown exception");
            }
            // Now open via the C++ wrapper. ts_tessera_db_open
            // runs the new schema (no-op on existing tables
            // because of CREATE TABLE IF NOT EXISTS) and then
            // runs ts_tessera_db_migrate_model_role, which
            // detects the missing column on tensor_stats and
            // runs the rebuild.
            std::string mig_err;
            ts_tessera_db * pre_db = ts_tessera_db_open(pre_path, &mig_err);
            CHECK(pre_db != nullptr,
                  ("pre-Phase-16 DB open failed: " + mig_err).c_str());
            if (pre_db != nullptr) {
                // The migration should have run on open
                // (the C++ open path calls
                // ts_tessera_db_migrate_model_role). Verify
                // the row has model_role='trunk' and the
                // new PK is in place.
                int64_t n_pre_trunk = ts_tessera_db_debug_count(
                    pre_db, "SELECT COUNT(*) FROM tensor_stats "
                            "WHERE model_hash = 'pre_model' "
                            "AND model_role = 'trunk'");
                CHECK(n_pre_trunk == 1,
                      "pre-Phase-16 row backfilled to model_role='trunk' "
                      "by open-path migration");

                // Explicit re-run is a no-op.
                std::string mig_err2;
                CHECK(ts_tessera_db_migrate_model_role(pre_db, &mig_err2) == 0,
                      ("re-migration failed: " + mig_err2).c_str());
                int64_t n_pre_trunk2 = ts_tessera_db_debug_count(
                    pre_db, "SELECT COUNT(*) FROM tensor_stats "
                            "WHERE model_hash = 'pre_model' "
                            "AND model_role = 'trunk'");
                CHECK(n_pre_trunk2 == 1,
                      "re-migration: row count unchanged");

                // The original data survived the rebuild
                // (kurtosis=5.0 is preserved).
                int64_t n_kurtosis = ts_tessera_db_debug_count(
                    pre_db, "SELECT COUNT(*) FROM tensor_stats "
                            "WHERE model_hash = 'pre_model' "
                            "AND kurtosis = 5.0");
                CHECK(n_kurtosis == 1,
                      "pre-Phase-16 rebuild preserves kurtosis");

                delete pre_db;
            }
        }
    }

    // ---- Crash-safe shutdown: ~ts_tessera_db() must CHECKPOINT
    //      before tearing down the connection so a SIGKILL on
    //      llama-quantize exit does not leave a stale .wal
    //      blocking subsequent read-only opens. We verify by
    //      writing a row, deleting the db (which triggers the
    //      explicit CHECKPOINT in the destructor), and then
    //      opening the path again to confirm the data is in
    //      the main file (not pending in a stale WAL).
    {
        const std::string ckpt_path = std::string(path) + ".ckpt";
        std::string ckpt_err;
        ts_tessera_db * ckpt_db = ts_tessera_db_open(
            ckpt_path.c_str(), &ckpt_err
        );
        CHECK(ckpt_db != nullptr,
              ("CHECKPOINT test: open failed: " + ckpt_err).c_str());
        if (ckpt_db != nullptr) {
            ts_tessera_db_tensor_stat row;
            row.model_hash         = "ckpt_model";
            row.model_role         = "trunk";
            row.name               = "blk.0.attn_q.weight";
            row.family             = "attn_q";
            row.layer_depth        = 0;
            row.out_dim            = 4096;
            row.in_dim             = 4096;
            row.n_elements         = 16777216;
            row.dtype              = "f16";
            row.kurtosis           = 3.0f;
            row.eff_rank           = 0.8f;
            row.rms                = 0.10f;
            row.mean_abs           = 0.08f;
            row.tail_ratio         = 4.0f;
            row.source             = "py_cal";
            row.recommended_action = "protect";
            std::string ins_err;
            CHECK(ts_tessera_db_upsert_tensor_stat(
                ckpt_db, row, &ins_err) == 0,
                ("CHECKPOINT test: upsert failed: " + ins_err).c_str());
            // Destructor: must CHECKPOINT before tearing down.
            delete ckpt_db;
        }
        // After the destructor, the .wal must be gone (CHECKPOINT
        // flushed it; on a SIGKILL between here and the next
        // open this is what would be missing and would block
        // a subsequent read-only open).
        const std::string wal_path = ckpt_path + ".wal";
        const bool wal_exists = (std::ifstream(wal_path).good());
        CHECK(!wal_exists,
              ("CHECKPOINT test: stale WAL left on disk: " + wal_path).c_str());
        // Reopen the path and confirm the row is in the main
        // file (not stranded in a WAL that the destructor
        // failed to flush).
        std::string reopen_err;
        ts_tessera_db * ro_db = ts_tessera_db_open(ckpt_path.c_str(),
                                                   &reopen_err);
        CHECK(ro_db != nullptr,
              ("CHECKPOINT test: reopen failed: " + reopen_err).c_str());
        if (ro_db != nullptr) {
            int64_t n = ts_tessera_db_debug_count(
                ro_db, "SELECT COUNT(*) FROM tensor_stats "
                       "WHERE model_hash = 'ckpt_model'"
            );
            CHECK(n == 1,
                  "CHECKPOINT test: post-destructor reopen row count");
            delete ro_db;
        }
        // Clean up
        std::remove(ckpt_path.c_str());
        std::remove(wal_path.c_str());
    }

    if (failures == 0) {
        printf("OK: all tessera-quantize-db tests passed (db=%s)\n", path);
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion(s) failed\n", failures);
    return 1;
}
