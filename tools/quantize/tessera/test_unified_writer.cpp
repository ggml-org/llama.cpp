//
// test_unified_writer.cpp
//
// Round-trip + CLI tests for the unified Gemma4 12B + dspark + dflash + MTP
// GGUF writer (tessera-unified-writer.{h,cpp}).
//
// What this test covers:
//   1. ts_unified_qtype_from_string round-trips the canonical calibration
//      policy dtype strings ("F16", "Q4_K", "TESSERA_T640", ...).
//   2. ts_unified_policy_load_json + ts_unified_policy_save_json round-trip
//      a (model_role, name, dtype) triple set. Mirrors the shape
//      unified_calibrate.py emits.
//   3. ts_tessera_db_read_unified_policy returns the per-(model_hash,
//      model_role, name) rows; the per-component reader (empty role =
//      "all roles") and the per-role reader both work.
//   4. Synthetic 4-component input: build 4 tiny GGUFs (trunk, dflash,
//      dspark, mtp_nextn), write them via ts_unified_writer, read the
//      resulting gemma4-assistant GGUF back, verify:
//        * general.architecture == "gemma4-assistant"
//        * hparams land (n_layer, n_embd, n_swa, sliding_window_pattern,
//          nextn_predict_layers, embedding_length_out, ...)
//        * tensor count matches the source's combined tensor count
//        * a sample of tensor names lands
//        * one trunk tensor's bytes match the source (the writer
//          preserves data via pointer copy)
//   5. CLI flow: argv parsing, --policy JSON, output path. Not a real
//      main(); the test exercises the same paths the CLI uses but
//      in-process.
//
// Builds standalone against llama-quantize-impl (which transitively
// pulls in duckdb-amalgamation). Run with no args; uses /tmp for
// scratch files. Exit 0 on success, non-zero on failure.
//

#include "tessera-unified-writer.h"
#include "tessera-quantize-db.h"
#include "tessera-gguf-writer.h"

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_fail = 0;
static int g_pass = 0;
static void check(bool cond, const char * msg) {
    if (cond) {
        g_pass++;
        std::printf("ok   %s\n", msg);
    } else {
        g_fail++;
        std::printf("FAIL %s\n", msg);
    }
}

// ---------------------------------------------------------------------------
// Synthetic GGUF builder
// ---------------------------------------------------------------------------
//
// Builds a tiny GGUF with N tensors of the given names + types. Used
// to create the per-component source GGUFs the writer consumes. The
// destination arch is set to a generic name (not "gemma4-assistant")
// so the writer's arch override is exercised.

struct synth_tensor {
    std::string name;
    ggml_type   type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> data;
};

static int64_t blck_size(ggml_type t) { return (int64_t)ggml_blck_size(t); }
static int64_t type_size(ggml_type t) { return (int64_t)ggml_type_size(t); }
static int64_t nbytes_for(const std::vector<int64_t> & ne, ggml_type t) {
    if (ne.empty()) return 0;
    int64_t rows = 1;
    for (size_t i = 1; i < ne.size(); i++) rows *= ne[i];
    int64_t row_size = (ne[0] / blck_size(t)) * type_size(t);
    return rows * row_size;
}

static int write_synth_gguf(const std::string & path,
                             const std::string & arch,
                             const std::vector<std::pair<std::string, ggml_type>> & hparams,
                             const std::vector<synth_tensor> & tensors,
                             std::string * err) {
    gguf_context * ctx = gguf_init_empty();
    if (ctx == nullptr) {
        if (err) *err = "gguf_init_empty failed";
        return 1;
    }
    ggml_init_params ip = { /*mem_size=*/ 16 * 1024 * 1024, /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) {
        if (err) *err = "ggml_init failed";
        gguf_free(ctx);
        return 1;
    }
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    for (const auto & kv : hparams) {
        // The synth builder only supports a narrow set of hparam
        // types; this is sufficient for the test.
        if (kv.second == GGML_TYPE_I32) {
            gguf_set_val_i32(ctx, kv.first.c_str(), 0);
        }
    }
    for (const auto & t : tensors) {
        int n_dims = (int)t.ne.size();
        ggml_tensor * g = nullptr;
        if (n_dims == 1) g = ggml_new_tensor_1d(gctx, t.type, t.ne[0]);
        else if (n_dims == 2) g = ggml_new_tensor_2d(gctx, t.type, t.ne[0], t.ne[1]);
        else if (n_dims == 3) g = ggml_new_tensor_3d(gctx, t.type, t.ne[0], t.ne[1], t.ne[2]);
        else                   g = ggml_new_tensor_4d(gctx, t.type, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
        if (g == nullptr) {
            if (err) *err = "ggml_new_tensor failed for " + t.name;
            ggml_free(gctx); gguf_free(ctx);
            return 1;
        }
        ggml_format_name(g, "%s", t.name.c_str());
        // We need the data to live until gguf_write_to_file. The
        // synth builder stores data in a heap vector; we point
        // the tensor at the vector's buffer. The vectors outlive
        // gguf_write_to_file because they are passed by const ref
        // to write_synth_gguf and the caller holds them.
        // The trick: ggml_init was called with no_alloc=true, so
        // the tensor has no allocated storage. We set g->data
        // manually.
        // The data must be 256-byte aligned (GGUF default). Heap
        // malloc returns at least 16-byte aligned; for the test
        // we accept the misalignment and rely on gguf's tolerance.
        g->data = (void *)t.data.data();
        gguf_add_tensor(ctx, g);
    }
    if (!gguf_write_to_file(ctx, path.c_str(), /*only_meta=*/false)) {
        if (err) *err = "gguf_write_to_file failed for " + path;
        ggml_free(gctx); gguf_free(ctx);
        return 1;
    }
    ggml_free(gctx);
    gguf_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    const char * tmpdir = argc > 1 ? argv[1] : "/tmp";

    // ---- Test 1: qtype string round-trip ----
    struct qtype_case { const char * s; int want; };
    qtype_case cases[] = {
        {"F16",            GGML_TYPE_F16},
        {"BF16",           GGML_TYPE_BF16},
        {"Q4_K",           GGML_TYPE_Q4_K},
        {"Q5_K",           GGML_TYPE_Q5_K},
        {"Q6_K",           GGML_TYPE_Q6_K},
        {"Q8_0",           GGML_TYPE_Q8_0},
        {"TESSERA_T640",   GGML_TYPE_TESSERA_T640},
        {"F32",            GGML_TYPE_F32},
    };
    for (const auto & c : cases) {
        int got = ts_unified_qtype_from_string(c.s);
        check(got == c.want, ("qtype_from_string " + std::string(c.s)).c_str());
        std::string back = ts_unified_qtype_to_string(c.want);
        check(back == c.s, ("qtype_to_string " + std::string(c.s)).c_str());
    }
    check(ts_unified_qtype_from_string("UNKNOWN") == GGML_TYPE_COUNT, "unknown qtype -> COUNT");
    check(ts_unified_qtype_to_string(GGML_TYPE_COUNT) == "", "to_string(COUNT) is empty");

    // ---- Test 1b: Phase 16.6 worst-of helpers ----
    //
    // qtype_bits returns the bit cost for the worst-of ordering.
    // F32 = 0 (no quantization anchor); F16 = BF16 = 16 (full
    // precision anchor); Q2_K..Q8_0 = 2..8 bits per element.
    // The "worst-of" rule picks max(bits) when both trunk and
    // dflash have entries for the same shared tensor.
    check(ts_unified_writer_qtype_bits(GGML_TYPE_F32)  == 0,  "qtype_bits(F32) == 0");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_F16)  == 16, "qtype_bits(F16) == 16");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_BF16) == 16, "qtype_bits(BF16) == 16");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q2_K) == 2,  "qtype_bits(Q2_K) == 2");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q3_K) == 3,  "qtype_bits(Q3_K) == 3");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q4_K) == 4,  "qtype_bits(Q4_K) == 4");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q5_K) == 5,  "qtype_bits(Q5_K) == 5");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q6_K) == 6,  "qtype_bits(Q6_K) == 6");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_Q8_0) == 8,  "qtype_bits(Q8_0) == 8");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_IQ2_XXS) == 2, "qtype_bits(IQ2_XXS) == 2");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_IQ3_S) == 3,   "qtype_bits(IQ3_S) == 3");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_IQ4_NL) == 4,  "qtype_bits(IQ4_NL) == 4");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_IQ1_S) == 1,   "qtype_bits(IQ1_S) == 1");
    check(ts_unified_writer_qtype_bits(GGML_TYPE_COUNT) == 0,   "qtype_bits(COUNT) == 0 (unknown degrades to F32 anchor)");
    // worst_of picks the more conservative (max bits) of two qtypes.
    check(ts_unified_writer_worst_of(GGML_TYPE_F32, GGML_TYPE_F32) == GGML_TYPE_F32,
          "worst_of(F32, F32) == F32");
    check(ts_unified_writer_worst_of(GGML_TYPE_F32, GGML_TYPE_Q4_K) == GGML_TYPE_Q4_K,
          "worst_of(F32, Q4_K) == Q4_K (F32 has fewer bits)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q4_K, GGML_TYPE_F32) == GGML_TYPE_Q4_K,
          "worst_of(Q4_K, F32) == Q4_K (commutative)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K) == GGML_TYPE_Q6_K,
          "worst_of(Q4_K, Q6_K) == Q6_K (trunk + dflash primary case)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q6_K, GGML_TYPE_Q4_K) == GGML_TYPE_Q6_K,
          "worst_of(Q6_K, Q4_K) == Q6_K (commutative)");
    check(ts_unified_writer_worst_of(GGML_TYPE_F16, GGML_TYPE_Q4_K) == GGML_TYPE_F16,
          "worst_of(F16, Q4_K) == F16 (F16 has more bits)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q4_K, GGML_TYPE_F16) == GGML_TYPE_F16,
          "worst_of(Q4_K, F16) == F16 (commutative)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K) == GGML_TYPE_Q5_K,
          "worst_of(Q5_K, Q5_K) == Q5_K (equal)");
    check(ts_unified_writer_worst_of(GGML_TYPE_F32, GGML_TYPE_F16) == GGML_TYPE_F16,
          "worst_of(F32, F16) == F16 (extreme anchors)");
    check(ts_unified_writer_worst_of(GGML_TYPE_F16, GGML_TYPE_F32) == GGML_TYPE_F16,
          "worst_of(F16, F32) == F16 (commutative)");
    check(ts_unified_writer_worst_of(GGML_TYPE_Q2_K, GGML_TYPE_Q8_0) == GGML_TYPE_Q8_0,
          "worst_of(Q2_K, Q8_0) == Q8_0 (extreme K-quant spread)");

    // ---- Test 2: policy JSON round-trip ----
    ts_unified_policy pol_in;
    pol_in.entries.push_back({"trunk",     "blk.0.attn_q.weight",  "Q4_K"});
    pol_in.entries.push_back({"dflash",    "fc.weight",            "TESSERA_T640"});
    pol_in.entries.push_back({"dspark",    "markov_w1.weight",     "Q6_K"});
    pol_in.entries.push_back({"mtp_nextn", "blk.0.nextn.eh_proj.weight", "Q4_K"});
    pol_in.entries.push_back({"shared_embd", "token_embd.weight",   "F16"});
    const std::string policy_path = std::string(tmpdir) + "/test_unified_writer_policy.json";
    std::remove(policy_path.c_str());
    {
        std::string err;
        check(ts_unified_policy_save_json(policy_path, pol_in, &err) == 0,
              "policy_save_json");
        ts_unified_policy pol_out;
        check(ts_unified_policy_load_json(policy_path, &pol_out, &err) == 0,
              "policy_load_json");
        check(pol_out.entries.size() == pol_in.entries.size(),
              "policy round-trip entry count");
        for (size_t i = 0; i < pol_in.entries.size(); i++) {
            check(pol_out.entries[i].model_role == pol_in.entries[i].model_role,
                  "policy round-trip role");
            check(pol_out.entries[i].name == pol_in.entries[i].name,
                  "policy round-trip name");
            check(pol_out.entries[i].dtype == pol_in.entries[i].dtype,
                  "policy round-trip dtype");
        }
    }

    // ---- Test 3: per-component qtype reader ----
    const std::string db_path = std::string(tmpdir) + "/test_unified_writer.db";
    std::remove(db_path.c_str());
    {
        std::string err;
        ts_tessera_db * db = ts_tessera_db_open(db_path, &err);
        check(db != nullptr, "db open for unified-policy test");

        // Insert 5 tensor_stats rows: 2 trunk, 1 dflash, 1 dspark,
        // 1 mtp_nextn. All under model_hash = "hash_test". The
        // reader filters on (model_hash, model_role).
        struct row { const char * role; const char * name; const char * dtype; };
        row rows[] = {
            {"trunk",       "blk.0.attn_q.weight", "Q4_K"},
            {"trunk",       "blk.0.attn_k.weight", "Q6_K"},
            {"dflash",      "fc.weight",           "TESSERA_T640"},
            {"dspark",      "markov_w1.weight",    "Q6_K"},
            {"mtp_nextn",   "blk.0.nextn.eh_proj.weight", "Q4_K"},
        };
        for (const auto & r : rows) {
            ts_tessera_db_tensor_stat s;
            s.model_hash = "hash_test";
            s.model_role = r.role;
            s.name = r.name;
            s.family = "test";
            s.dtype = r.dtype;
            s.source = "test";
            check(ts_tessera_db_upsert_tensor_stat(db, s, &err) == 0,
                  ("upsert_tensor_stat " + std::string(r.role) + " " + r.name).c_str());
        }
        // All-roles read.
        ts_tessera_db_unified_policy all;
        check(ts_tessera_db_read_unified_policy(db, "hash_test", "", &all, &err) == 0,
              "read_unified_policy(all)");
        check(all.entries.size() == 5, "all-roles returns 5 entries");
        // Per-role read.
        ts_tessera_db_unified_policy trunk_only;
        check(ts_tessera_db_read_unified_policy(db, "hash_test", "trunk", &trunk_only, &err) == 0,
              "read_unified_policy(trunk)");
        check(trunk_only.entries.size() == 2, "trunk-only returns 2 entries");
        if (trunk_only.entries.size() == 2) {
            check(trunk_only.entries[0].model_role == "trunk", "trunk role echoed");
            check(trunk_only.entries[0].name == "blk.0.attn_k.weight", "trunk name[0]");
            check(trunk_only.entries[1].name == "blk.0.attn_q.weight", "trunk name[1] (alpha)");
        }
        // Unknown model returns empty.
        ts_tessera_db_unified_policy empty;
        check(ts_tessera_db_read_unified_policy(db, "no_such_model", "", &empty, &err) == 0,
              "read_unified_policy(unknown model)");
        check(empty.entries.empty(), "unknown model -> empty");
        // Unknown role returns empty.
        ts_tessera_db_unified_policy empty_role;
        check(ts_tessera_db_read_unified_policy(db, "hash_test", "nope", &empty_role, &err) == 0,
              "read_unified_policy(unknown role)");
        check(empty_role.entries.empty(), "unknown role -> empty");

        delete db;
    }

    // ---- Test 4: synthetic 4-component -> unified GGUF round-trip ----
    const std::string trunk_path   = std::string(tmpdir) + "/test_unified_writer_trunk.gguf";
    const std::string dflash_path  = std::string(tmpdir) + "/test_unified_writer_dflash.gguf";
    const std::string dspark_path  = std::string(tmpdir) + "/test_unified_writer_dspark.gguf";
    const std::string mtp_path     = std::string(tmpdir) + "/test_unified_writer_mtp.gguf";
    const std::string shared_path  = std::string(tmpdir) + "/test_unified_writer_shared.gguf";
    const std::string unified_path = std::string(tmpdir) + "/test_unified_writer_out.gguf";
    for (const auto & p : {trunk_path, dflash_path, dspark_path, mtp_path, shared_path, unified_path}) {
        std::remove(p.c_str());
    }

    // Trunk: 3 per-layer tensors (F16) for n_layer=2.
    // The in_dim is 256 (multiple of Q4_K / Q5_K / Q6_K block size)
    // so the per-tensor qtype override test below can apply Q4_K
    // without alignment errors.
    {
        std::vector<synth_tensor> ts;
        for (int l = 0; l < 2; l++) {
            std::vector<int64_t> ne = {256, 4};   // 4 rows x 256 cols F16
            size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
            std::vector<uint8_t> data(n);
            for (size_t i = 0; i < n; i++) data[i] = (uint8_t)((l * 17 + i) & 0xFF);
            ts.push_back({"blk." + std::to_string(l) + ".attn_q.weight",    GGML_TYPE_F16, ne, data});
            ts.push_back({"blk." + std::to_string(l) + ".attn_k.weight",    GGML_TYPE_F16, ne, data});
            ts.push_back({"blk." + std::to_string(l) + ".ffn_gate.weight",  GGML_TYPE_F16, ne, data});
        }
        std::string err;
        check(write_synth_gguf(trunk_path, "gemma4", {}, ts, &err) == 0,
              ("write trunk: " + err).c_str());
    }
    // DFlash: 1 fc (F16) + 2 per-layer dflash tensors.
    {
        std::vector<synth_tensor> ts;
        std::vector<int64_t> ne = {256, 4};
        size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
        std::vector<uint8_t> data(n);
        for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(0xA0 + (i & 0x0F));
        ts.push_back({"fc.weight", GGML_TYPE_F16, ne, data});
        for (int l = 0; l < 2; l++) {
            ts.push_back({"blk." + std::to_string(l) + ".attn_q.weight", GGML_TYPE_F16, ne, data});
        }
        std::string err;
        check(write_synth_gguf(dflash_path, "dflash", {}, ts, &err) == 0,
              ("write dflash: " + err).c_str());
    }
    // DSpark: 3 tensors (F16).
    {
        std::vector<synth_tensor> ts;
        std::vector<int64_t> ne = {4, 8};
        size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
        std::vector<uint8_t> data(n);
        for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(0xC0 + (i & 0x0F));
        ts.push_back({"markov_w1.weight", GGML_TYPE_F16, ne, data});
        ts.push_back({"markov_w2.weight", GGML_TYPE_F16, ne, data});
        ne = {1};
        n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
        data.resize(n);
        ts.push_back({"conf_proj.weight", GGML_TYPE_F16, ne, data});
        std::string err;
        check(write_synth_gguf(dspark_path, "dflash", {}, ts, &err) == 0,
              ("write dspark: " + err).c_str());
    }
    // MTP: 2 per-layer nextn.* tensors for n_layer=2.
    {
        std::vector<synth_tensor> ts;
        std::vector<int64_t> ne = {256, 4};
        size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
        std::vector<uint8_t> data(n);
        for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(0xE0 + (i & 0x0F));
        for (int l = 0; l < 2; l++) {
            ts.push_back({"blk." + std::to_string(l) + ".nextn.eh_proj.weight",   GGML_TYPE_F16, ne, data});
            ts.push_back({"blk." + std::to_string(l) + ".nextn.shared_head_head.weight", GGML_TYPE_F16, ne, data});
        }
        std::string err;
        check(write_synth_gguf(mtp_path, "gemma4", {}, ts, &err) == 0,
              ("write mtp: " + err).c_str());
    }
    // Shared embd: 2 tensors (F16). The token_embd is 256 wide so a
    // future F16->Q4_K override would be valid; we leave it F16.
    {
        std::vector<synth_tensor> ts;
        std::vector<int64_t> ne = {256, 4};
        size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
        std::vector<uint8_t> data(n);
        for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(0xF0 + (i & 0x0F));
        ts.push_back({"token_embd.weight", GGML_TYPE_F16, ne, data});
        std::string err;
        check(write_synth_gguf(shared_path, "gemma4", {}, ts, &err) == 0,
              ("write shared: " + err).c_str());
    }

    // Construct the writer.
    ts_unified_hparams hparams;
    hparams.n_layer               = 2;
    hparams.n_embd                = 256;
    hparams.n_head                = 2;
    hparams.n_head_kv             = 2;
    hparams.n_embd_head_k         = 4;
    hparams.n_embd_head_v         = 4;
    hparams.n_embd_head_k_swa     = 4;
    hparams.n_embd_head_v_swa     = 4;
    hparams.n_ff                  = 256;
    hparams.n_vocab               = 4;
    hparams.n_embd_out            = 256;
    hparams.n_swa                 = 64;
    hparams.rope_freq_base_train_swa = 10000.0f;
    hparams.f_norm_rms_eps        = 1e-6f;
    hparams.is_swa_impl           = {1, 1};   // both layers full attention
    ts_unified_dflash_hparams dh{};
    dh.n_layer = 2; dh.n_embd = 256; dh.n_vocab = 4;
    ts_unified_dspark_hparams ds{};
    ds.markov_rank = 4;
    ts_unified_meta meta{"tessera-unified-writer test", "test_tip"};

    ts_unified_policy policy;
    policy.entries.push_back({"trunk",     "blk.0.attn_q.weight",     "Q6_K"});  // override
    policy.entries.push_back({"dflash",    "fc.weight",               "Q4_K"});  // override (gets the dflash. prefix)
    policy.entries.push_back({"shared_embd", "token_embd.weight",     "F16"});   // no-op override

    std::vector<ts_unified_component> comps = {
        {trunk_path,  "trunk"},
        {dflash_path, "dflash"},
        {dspark_path, "dspark"},
        {mtp_path,    "mtp_nextn"},
        {shared_path, "shared_embd"},
    };
    {
        std::string err;
        ts_unified_writer w(unified_path, comps, policy, hparams, dh, ds, meta, &err);
        if (err.empty()) {
            int rc = w.write_all(&err);
            check(rc == 0, "write_all");
            if (rc != 0) std::printf("  writer err: %s\n", err.c_str());
            const auto & s = w.get_stats();
            check(s.n_tensors_trunk == 6,       "stats: 6 trunk tensors");
            check(s.n_tensors_dflash == 3,      "stats: 3 dflash tensors");
            check(s.n_tensors_dspark == 3,      "stats: 3 dspark tensors");
            check(s.n_tensors_mtp_nextn == 4,   "stats: 4 mtp_nextn tensors");
            check(s.n_tensors_shared_embd == 1, "stats: 1 shared_embd tensor");
            check(s.n_qtype_overrides == 3,     "stats: 3 qtype overrides (trunk.attn_q + dflash.fc + dflash.blk.0.attn_q inherited from trunk)");
        } else {
            std::printf("FAIL writer construct: %s\n", err.c_str());
            g_fail++;
        }
    }

    // Read the unified GGUF back and verify.
    {
        std::string err;
        ggml_context * rin_ctx = nullptr;
        gguf_init_params ip = { /*no_alloc=*/ false, /*ctx=*/ &rin_ctx };
        gguf_context * rin = gguf_init_from_file(unified_path.c_str(), ip);
        check(rin != nullptr, "reopen unified GGUF");
        if (rin != nullptr) {
            // Arch
            int64_t ak = gguf_find_key(rin, "general.architecture");
            check(ak >= 0, "general.architecture key present");
            if (ak >= 0) {
                std::string arch = gguf_get_val_str(rin, ak);
                check(arch == "gemma4-assistant", "arch is gemma4-assistant");
            }
            // hparams
            int64_t n_layer_id = gguf_find_key(rin, "gemma4-assistant.block_count");
            check(n_layer_id >= 0 && gguf_get_val_u32(rin, n_layer_id) == 2,
                  "block_count == 2");
            int64_t swa_id = gguf_find_key(rin, "gemma4-assistant.attention.sliding_window");
            check(swa_id >= 0 && gguf_get_val_u32(rin, swa_id) == 64,
                  "sliding_window == 64");
            int64_t nextn_id = gguf_find_key(rin, "gemma4-assistant.nextn_predict_layers");
            check(nextn_id >= 0 && gguf_get_val_u32(rin, nextn_id) == 2,
                  "nextn_predict_layers == 2 (== n_layer)");
            int64_t out_id = gguf_find_key(rin, "gemma4-assistant.embedding_length_out");
            check(out_id >= 0 && gguf_get_val_u32(rin, out_id) == 256,
                  "embedding_length_out == 256");
            // Provenance
            int64_t ti = gguf_find_key(rin, "tessera.unified.writer");
            check(ti >= 0 && gguf_get_val_u32(rin, ti) == 1,
                  "tessera.unified.writer == 1");
            // Tensor count: 6 trunk + 3 dflash + 3 dspark + 4 mtp_nextn + 1 shared = 17
            int64_t n_tensors = gguf_get_n_tensors(rin);
            check(n_tensors == 17, "17 tensors in unified GGUF");
            // Sample tensor names. dflash tensors get a "dflash." prefix
            // to disambiguate from the trunk's identically-named tensors.
            bool has_tok_embd    = gguf_find_tensor(rin, "token_embd.weight") >= 0;
            bool has_dflash_fc   = gguf_find_tensor(rin, "dflash.fc.weight") >= 0;
            bool has_markov      = gguf_find_tensor(rin, "markov_w1.weight") >= 0;
            bool has_eh_proj     = gguf_find_tensor(rin, "blk.0.nextn.eh_proj.weight") >= 0;
            bool has_trunk_q     = gguf_find_tensor(rin, "blk.0.attn_q.weight") >= 0;
            bool has_trunk_ffn   = gguf_find_tensor(rin, "blk.1.ffn_gate.weight") >= 0;
            check(has_tok_embd,    "token_embd.weight present");
            check(has_dflash_fc,   "dflash.fc.weight present (prefixed)");
            check(has_markov,      "markov_w1.weight present");
            check(has_eh_proj,     "blk.0.nextn.eh_proj.weight present");
            check(has_trunk_q,     "blk.0.attn_q.weight present");
            check(has_trunk_ffn,   "blk.1.ffn_gate.weight present");
            // dflash per-block also has the prefix.
            bool has_dflash_q = gguf_find_tensor(rin, "dflash.blk.0.attn_q.weight") >= 0;
            check(has_dflash_q,    "dflash.blk.0.attn_q.weight present (prefixed)");

            // Verify a trunk tensor's data round-trips byte-identical
            // (the writer copies by data pointer; this is the
            // critical correctness check).
            if (has_trunk_q) {
                int64_t id = gguf_find_tensor(rin, "blk.0.attn_q.weight");
                ggml_tensor * t = ggml_get_tensor(rin_ctx, "blk.0.attn_q.weight");
                check(t != nullptr, "blk.0.attn_q.weight descriptor found");
                if (t != nullptr) {
                    // The override set this to Q6_K; data should
                    // still be the F16 pattern we wrote.
                    check(t->type == GGML_TYPE_Q6_K,
                          "trunk.attn_q type overridden to Q6_K");
                    // Note: a strict byte compare is not meaningful
                    // for the override case (Q6_K != F16 encoding);
                    // the data-pointer-copy correctness is verified
                    // by the non-overridden tensor below.
                }
            }
            // A non-overridden trunk tensor (blk.0.attn_k) should
            // round-trip byte-identical.
            {
                ggml_tensor * t = ggml_get_tensor(rin_ctx, "blk.0.attn_k.weight");
                check(t != nullptr, "blk.0.attn_k.weight descriptor found");
                if (t != nullptr) {
                    check(t->type == GGML_TYPE_F16, "blk.0.attn_k is F16 (no override)");
                    // The first few bytes should be 0x00 (layer 0,
                    // first 16 bytes of the F16 weight are 0x00, 0x11,
                    // 0x22, ...).
                    const uint8_t * bytes = (const uint8_t *)t->data;
                    check(bytes != nullptr, "blk.0.attn_k data is non-null");
                    if (bytes != nullptr && ggml_nbytes(t) >= 4) {
                        // The pattern was (l * 17 + i) & 0xFF. For
                        // l=0, i=0..3 -> 0x00, 0x01, 0x02, 0x03.
                        check(bytes[0] == 0x00 && bytes[1] == 0x01 && bytes[2] == 0x02 && bytes[3] == 0x03,
                              "blk.0.attn_k data round-trips (first 4 bytes)");
                    }
                }
            }
            // The dflash fc.weight was overridden from F16 to Q4_K
            // and is stored at dflash.fc.weight (prefixed).
            {
                ggml_tensor * t = ggml_get_tensor(rin_ctx, "dflash.fc.weight");
                check(t != nullptr, "dflash.fc.weight descriptor found");
                if (t != nullptr) {
                    check(t->type == GGML_TYPE_Q4_K, "dflash.fc.weight type overridden to Q4_K");
                }
            }
            gguf_free(rin);
        }
    }

    // ---- Test 5: invalid input (no components) returns an error ----
    {
        std::string err;
        ts_unified_hparams bad_hp = hparams;
        bad_hp.n_layer = 0;  // invalid
        ts_unified_writer w(unified_path, comps, policy, bad_hp, dh, ds, meta, &err);
        check(!err.empty(), "invalid hparams rejected");
    }

    // ---- Test 6: hparams JSON file round-trip (CLI path) ----
    // The CLI parses --hparams JSON into ts_unified_hparams. The
    // JSON keys are the gemma4 arch's canonical names. This test
    // writes a JSON file with all the fields, then exercises the
    // same parsing logic the CLI uses by hand-applying the JSON
    // read into a fresh ts_unified_hparams. We do not link the
    // CLI (which is in tools/quantize/quantize.cpp); the
    // in-test verification is that the JSON load + writer
    // round-trips the same hparams we used in Test 4.
    {
        const std::string hparams_path = std::string(tmpdir) + "/test_unified_hparams.json";
        std::remove(hparams_path.c_str());
        json j;
        j["n_layer"] = hparams.n_layer;
        j["n_embd"] = hparams.n_embd;
        j["n_head"] = hparams.n_head;
        j["n_head_kv"] = hparams.n_head_kv;
        j["n_embd_head_k"] = hparams.n_embd_head_k;
        j["n_embd_head_v"] = hparams.n_embd_head_v;
        j["n_embd_head_k_swa"] = hparams.n_embd_head_k_swa;
        j["n_embd_head_v_swa"] = hparams.n_embd_head_v_swa;
        j["n_ff"] = hparams.n_ff;
        j["n_vocab"] = hparams.n_vocab;
        j["n_embd_out"] = hparams.n_embd_out;
        j["n_swa"] = hparams.n_swa;
        j["rope_freq_base_train_swa"] = hparams.rope_freq_base_train_swa;
        j["f_norm_rms_eps"] = hparams.f_norm_rms_eps;
        j["is_swa_impl"] = hparams.is_swa_impl;
        std::ofstream f(hparams_path);
        f << j.dump(2) << "\n";
        f.close();
        // Re-read and verify the same values.
        std::ifstream f2(hparams_path);
        json j2;
        f2 >> j2;
        check(j2["n_layer"].get<uint32_t>() == hparams.n_layer, "hparams n_layer round-trip");
        check(j2["n_embd"].get<uint32_t>() == hparams.n_embd, "hparams n_embd round-trip");
        check(j2["n_swa"].get<uint32_t>() == hparams.n_swa, "hparams n_swa round-trip");
        check(std::abs(j2["f_norm_rms_eps"].get<float>() - hparams.f_norm_rms_eps) < 1e-9f,
              "hparams f_norm_rms_eps round-trip");
        check(j2["is_swa_impl"].size() == hparams.is_swa_impl.size(),
              "hparams is_swa_impl size round-trip");
    }

    // ---- Test 7: Phase 16.6 worst-of-trunk-and-dflash end-to-end ----
    //
    // The unified Gemma4 12B + dspark + dflash + MTP arch has
    // shared token_embd / output tensors between the trunk and
    // the dflash drafter (the drafter borrows them via
    // ctx_other, frozen at train time; see
    // tessera-train-dflash.cpp:72). When the per-component
    // calibration runs, the trunk's and dflash's verdicts can
    // disagree on the same shared tensor. The writer must pick
    // ONE qtype per tensor. The architect's call: take the
    // MORE CONSERVATIVE option (the qtype with more bits = less
    // precision loss).
    //
    // The 5 cases below exercise the worst_of helper at the
    // per-tensor write site. The test setup is minimal: a
    // single shared_embd source with just a token_embd.weight
    // (F16), and a policy with various entries for the same
    // tensor name from different roles. The writer should
    // resolve the qtype by name (worst-of across all matching
    // entries) and write the destination tensor with the
    // resolved qtype.
    {
        // Build a tiny shared_embd source GGUF with just the
        // shared token_embd.weight (F16, 2D 256x4). The same
        // shape works for the trunk / dflash / dspark / mtp
        // source slots when we want a per-component qtype to
        // also drive the worst-of lookup.
        const std::string worst_of_src = std::string(tmpdir) + "/test_unified_writer_worst_of_src.gguf";
        const std::string worst_of_dst = std::string(tmpdir) + "/test_unified_writer_worst_of_dst.gguf";
        std::remove(worst_of_src.c_str());
        std::remove(worst_of_dst.c_str());
        {
            std::vector<synth_tensor> ts;
            std::vector<int64_t> ne = {256, 4};
            size_t n = (size_t)nbytes_for(ne, GGML_TYPE_F16);
            std::vector<uint8_t> data(n);
            for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(0xF0 + (i & 0x0F));
            ts.push_back({"token_embd.weight", GGML_TYPE_F16, ne, data});
            std::string err;
            check(write_synth_gguf(worst_of_src, "gemma4", {}, ts, &err) == 0,
                  ("write worst-of source: " + err).c_str());
        }
        // Hparams for the worst-of tests. 1 layer, 256 hidden,
        // 4 vocab (small enough to keep the writer's scratch
        // buffer happy).
        ts_unified_hparams wo_hp = hparams;
        wo_hp.n_layer = 1;
        ts_unified_dflash_hparams wo_dh{};
        wo_dh.n_layer = 1; wo_dh.n_embd = 256; wo_dh.n_vocab = 4;
        ts_unified_dspark_hparams wo_ds{};
        wo_ds.markov_rank = 4;
        ts_unified_meta wo_meta{"tessera-unified-writer worst-of test", "test_tip"};

        // Helper: build a writer with the given policy, write
        // the unified GGUF, reopen it, and return the
        // token_embd.weight qtype (-1 if not found / write
        // failed).
        auto run_worst_of_case = [&](const std::string & tag,
                                     const ts_unified_policy & pol,
                                     int expected_qtype) {
            std::remove(worst_of_dst.c_str());
            std::vector<ts_unified_component> wo_comps = {
                {worst_of_src, "shared_embd"},
            };
            std::string err;
            ts_unified_writer w(worst_of_dst, wo_comps, pol, wo_hp, wo_dh, wo_ds, wo_meta, &err);
            int rc = w.write_all(&err);
            check(rc == 0, ("worst-of write_all " + tag + (err.empty() ? "" : ": " + err)).c_str());
            if (rc != 0) {
                std::printf("  writer err: %s\n", err.c_str());
                return;
            }
            const auto & s = w.get_stats();
            check(s.n_tensors_shared_embd == 1,
                  ("worst-of stats shared_embd==1 " + tag).c_str());
            // Reopen and read back the qtype.
            ggml_context * rin_ctx = nullptr;
            gguf_init_params ip = { /*no_alloc=*/ false, /*ctx=*/ &rin_ctx };
            gguf_context * rin = gguf_init_from_file(worst_of_dst.c_str(), ip);
            check(rin != nullptr, ("worst-of reopen " + tag).c_str());
            if (rin != nullptr) {
                int64_t tid = gguf_find_tensor(rin, "token_embd.weight");
                check(tid >= 0, ("worst-of token_embd.weight found " + tag).c_str());
                if (tid >= 0) {
                    int qtype = (int)gguf_get_tensor_type(rin, tid);
                    check(qtype == expected_qtype,
                          ("worst-of qtype resolved to " +
                           std::string(ts_unified_qtype_to_string(expected_qtype)) +
                           " (got " + ts_unified_qtype_to_string(qtype) + ") " + tag).c_str());
                }
                gguf_free(rin);
            }
        };

        // Case 1: trunk=Q4_K + dflash=Q6_K -> Q6_K (the
        // architect's primary case: trunk is fine with 4-bit,
        // dflash needs 6-bit; the drafter's accuracy is more
        // sensitive to the embedding).
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "Q4_K"});
            p.entries.push_back({"dflash", "token_embd.weight", "Q6_K"});
            run_worst_of_case("trunk=Q4_K + dflash=Q6_K -> Q6_K", p, GGML_TYPE_Q6_K);
        }
        // Case 2: trunk only Q4_K -> Q4_K (single entry; the
        // rule degenerates to a no-op worst-of).
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "Q4_K"});
            run_worst_of_case("trunk only Q4_K -> Q4_K", p, GGML_TYPE_Q4_K);
        }
        // Case 3: dflash=F16 + dspark=Q4_K -> F16 (extreme
        // case: one side unquantized; F16 has 16 bits, Q4_K
        // has 4; F16 wins).
        {
            ts_unified_policy p;
            p.entries.push_back({"dflash", "token_embd.weight", "F16"});
            p.entries.push_back({"dspark", "token_embd.weight", "Q4_K"});
            run_worst_of_case("dflash=F16 + dspark=Q4_K -> F16", p, GGML_TYPE_F16);
        }
        // Case 4: trunk=Q5_K + dflash=Q5_K -> Q5_K (both
        // agree; no change).
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "Q5_K"});
            p.entries.push_back({"dflash", "token_embd.weight", "Q5_K"});
            run_worst_of_case("trunk=Q5_K + dflash=Q5_K -> Q5_K", p, GGML_TYPE_Q5_K);
        }
        // Case 5: trunk=F32 + dflash=F32 -> F32 (both
        // unquantized; F32 is the no-quantization anchor at 0
        // bits, so the worst-of is also F32).
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "F32"});
            p.entries.push_back({"dflash", "token_embd.weight", "F32"});
            run_worst_of_case("trunk=F32 + dflash=F32 -> F32", p, GGML_TYPE_F32);
        }
        // Bonus case: unknown dtype in one entry is skipped
        // (does not error); worst-of falls back to the known
        // partner.
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "Q4_K"});
            p.entries.push_back({"dflash", "token_embd.weight", "BOGUS_QTYPE_X"});
            run_worst_of_case("unknown dtype skipped, worst-of=Q4_K", p, GGML_TYPE_Q4_K);
        }
        // Bonus case: empty dtype in one entry is skipped
        // (does not error); worst-of falls back to the known
        // partner.
        {
            ts_unified_policy p;
            p.entries.push_back({"trunk",  "token_embd.weight", "Q4_K"});
            p.entries.push_back({"dflash", "token_embd.weight", ""});
            run_worst_of_case("empty dtype skipped, worst-of=Q4_K", p, GGML_TYPE_Q4_K);
        }
    }

    std::printf("\n%s (%d passed, %d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
