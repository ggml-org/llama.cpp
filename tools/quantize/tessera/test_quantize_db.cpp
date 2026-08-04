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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            "model_hash", "name", "layer", "iteration", "plan_id",
            "strategy", "alpha_before", "alpha_after", "clip_before",
            "clip_after", "outlier_thresh_before", "outlier_thresh_after",
            "mse_before", "mse_after", "frob_before", "frob_after",
            "family", "updated_at",
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

    if (failures == 0) {
        printf("OK: all tessera-quantize-db tests passed (db=%s)\n", path);
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion(s) failed\n", failures);
    return 1;
}
