//
// test_policy.cpp
//
// Tests for tessera-policy.cpp:
//   1. round-trip: write a canonical-shape policy, read it back, verify fields
//   2. read a real Python-emitted fixture (tools/quantize/tessera/fixtures)
//   3. match/exact glob semantics + longest-fragment tie-break
//   4. backward-compat with the legacy top-level `tensors` shape
//   5. policy is applied to a per-tensor params struct (production wiring)
//

#include "tessera-policy.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++;                                    \
        }                                                    \
    } while (0)

static bool feq(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

// Resolve a path relative to this source file (so the test works regardless
// of the CWD the harness invokes it from).
static std::string src_relative(const std::string & tail) {
    std::string f = __FILE__;
    size_t slash = f.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : f.substr(0, slash);
    return dir + "/" + tail;
}

// --- helper: find a family by key in a ts_policy ---
static const ts_policy_tensor * find_family(const ts_policy & p, const std::string & key) {
    for (const auto & kv : p.tensors) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

int main(void) {
    const std::string rwp = "/tmp/test_policy.json";

    // -------------------------------------------------------------------
    // 1. round-trip (canonical shape)
    // -------------------------------------------------------------------
    ts_policy p;
    p.seed        = 42;
    p.generations = 8;
    p.islands     = 4;
    p.population  = 16;
    p.timestamp   = "2026-07-30T14:00:00Z";
    p.build_info  = "llama.cpp build 640";
    p.main_tip    = "abc123";
    p.search_schema = "llama.tessera.awq-evolution.v1";
    p.draft_type    = "hybrid";

    ts_policy_tensor q;
    q.family = "attention";
    q.match  = {"attn_q", "attn_k", "attn_v", "attn_output"};
    q.exact  = false;
    q.genes.alpha             = 0.65f;
    q.genes.clip              = 0.95f;
    q.genes.outlier_fraction  = 0.0040f;
    q.genes.moment_mix        = 0.20f;
    q.genes.tail_guard        = 0.25f;
    q.genes.ternary_threshold = 0.90f;
    p.tensors.emplace_back("attention", q);

    ts_policy_tensor ffn;
    ffn.family = "ffn";
    ffn.match  = {"ffn_gate_inp"};
    ffn.exact  = true;
    ffn.genes.alpha             = 0.50f;
    ffn.genes.clip              = 0.90f;
    ffn.genes.outlier_fraction  = 0.0020f;
    ffn.metrics.expert          = "dartquant";
    ffn.metrics.mse             = 0.0020f;
    ffn.metrics.relative_frob   = 0.0041f;
    p.tensors.emplace_back("router", ffn);

    ts_policy_archive_entry e;
    e.cell[0] = 2; e.cell[1] = 3; e.cell[2] = 0;
    e.alpha   = 0.70f;
    e.clip    = 0.90f;
    e.expert  = "dartquant";
    e.mse     = 0.0010f;
    p.archive.push_back(e);

    CHECK(ts_policy_write(rwp.c_str(), &p) == 0, "write succeeds");

    ts_policy r;
    std::string err;
    CHECK(ts_policy_read(rwp.c_str(), &r, &err) == 0, "read succeeds");
    if (!err.empty()) {
        std::printf("read error: %s\n", err.c_str());
    }

    CHECK(r.seed          == p.seed,          "seed round-trips");
    CHECK(r.generations   == p.generations,   "generations round-trips");
    CHECK(r.islands       == p.islands,       "islands round-trips");
    CHECK(r.population    == p.population,    "population round-trips");
    CHECK(r.timestamp     == p.timestamp,     "timestamp round-trips");
    CHECK(r.build_info    == p.build_info,    "build_info round-trips");
    CHECK(r.main_tip      == p.main_tip,      "main_tip round-trips");
    CHECK(r.search_schema == p.search_schema, "search_schema round-trips");
    CHECK(r.draft_type    == p.draft_type,    "draft_type round-trips");

    CHECK(r.tensors.size() == p.tensors.size(), "family count matches");
    const ts_policy_tensor * rq = find_family(r, "attention");
    CHECK(rq != nullptr, "attention family present");
    if (rq != nullptr) {
        CHECK(rq->family == "attention", "family name round-trips");
        CHECK(rq->match.size() == 4, "match list round-trips");
        CHECK(rq->exact == false, "exact round-trips");
        CHECK(feq(rq->genes.alpha,             0.65f),  "genes.alpha round-trips");
        CHECK(feq(rq->genes.clip,              0.95f),  "genes.clip round-trips");
        CHECK(feq(rq->genes.outlier_fraction,  0.0040f), "genes.outlier_fraction round-trips");
        CHECK(feq(rq->genes.moment_mix,        0.20f),  "genes.moment_mix round-trips");
        CHECK(feq(rq->genes.tail_guard,        0.25f),  "genes.tail_guard round-trips");
        CHECK(feq(rq->genes.ternary_threshold, 0.90f),  "genes.ternary_threshold round-trips");
    }
    const ts_policy_tensor * rr = find_family(r, "router");
    CHECK(rr != nullptr, "router family present");
    if (rr != nullptr) {
        CHECK(rr->exact == true, "router exact round-trips");
        CHECK(rr->metrics.expert == "dartquant", "metrics.expert round-trips");
        CHECK(feq(rr->metrics.mse, 0.0020f), "metrics.mse round-trips");
        CHECK(feq(rr->metrics.relative_frob, 0.0041f), "metrics.relative_frob round-trips");
    }

    CHECK(r.archive.size() == p.archive.size(), "archive count matches");
    if (r.archive.size() == p.archive.size()) {
        for (size_t i = 0; i < p.archive.size(); i++) {
            const ts_policy_archive_entry & a = p.archive[i];
            const ts_policy_archive_entry & b = r.archive[i];
            CHECK(b.cell[0] == a.cell[0] &&
                  b.cell[1] == a.cell[1] &&
                  b.cell[2] == a.cell[2], "archive.cell round-trips");
            CHECK(feq(b.alpha,  a.alpha),  "archive.alpha round-trips");
            CHECK(feq(b.clip,   a.clip),   "archive.clip round-trips");
            CHECK(b.expert      == a.expert, "archive.expert round-trips");
            CHECK(feq(b.mse,    a.mse),    "archive.mse round-trips");
        }
    }

    // -------------------------------------------------------------------
    // 2. real Python-emitted fixture
    // -------------------------------------------------------------------
    const std::string fxp = src_relative("fixtures/policy_python_sample.json");
    ts_policy fx;
    std::string ferr;
    CHECK(ts_policy_read(fxp.c_str(), &fx, &ferr) == 0, "fixture reads");
    if (!ferr.empty()) {
        std::printf("fixture read error: %s\n", ferr.c_str());
    }
    if (ts_policy_read(fxp.c_str(), &fx, &ferr) == 0) {
        // schema provenance
        CHECK(fx.search_schema == "llama.tessera.awq-evolution.v1", "fixture search_schema");
        CHECK(fx.draft_type    == "hybrid", "fixture draft_type");
        CHECK(fx.seed          == 640,      "fixture seed");
        CHECK(fx.generations   == 24,       "fixture generations");
        CHECK(fx.islands       == 4,        "fixture islands");
        CHECK(fx.population    == 16,       "fixture population");
        CHECK(fx.main_tip      == "abc123", "fixture main_tip");
        // fixture families: attention, ffn, norm, override:..., router
        CHECK(fx.tensors.size() == 5, "fixture family count");
        const ts_policy_tensor * att = find_family(fx, "attention");
        CHECK(att != nullptr, "fixture has attention");
        if (att != nullptr) {
            CHECK(att->exact == false, "fixture attention exact");
            CHECK(att->match.size() == 4, "fixture attention match list");
            CHECK(feq(att->genes.alpha,            0.72f),  "fixture awq_alpha");
            CHECK(feq(att->genes.clip,             0.95f),  "fixture awq_clip");
            CHECK(feq(att->genes.outlier_fraction, 0.0042f), "fixture outlier_fraction");
            CHECK(feq(att->genes.moment_mix,       0.21f),  "fixture moment_mix");
            CHECK(feq(att->genes.tail_guard,       0.27f),  "fixture tail_guard");
            // awq-evolve's policy_entry does NOT emit ternary_threshold, so
            // the reader must default it to 1.0 (legacy).
            CHECK(feq(att->genes.ternary_threshold, 1.0f), "fixture ternary default 1.0");
        }
        const ts_policy_tensor * norm = find_family(fx, "norm");
        CHECK(norm != nullptr, "fixture has norm");
        if (norm != nullptr) {
            CHECK(norm->exact == true, "fixture norm exact");
            CHECK(feq(norm->genes.alpha, 0.0f), "fixture norm alpha");
            CHECK(feq(norm->genes.outlier_fraction, 1.0f), "fixture norm fraction");
        }
        // the override family carries an awq_alpha that distinguishes it
        const ts_policy_tensor * ov = find_family(fx, "override:blk.3.ffn_gate.expert-7");
        CHECK(ov != nullptr, "fixture has override");
        if (ov != nullptr) {
            CHECK(feq(ov->genes.alpha, 0.55f), "fixture override alpha");
            CHECK(feq(ov->genes.outlier_fraction, 0.003f), "fixture override fraction");
        }
    }

    // -------------------------------------------------------------------
    // 3. match/exact semantics + longest-fragment tie-break
    // -------------------------------------------------------------------
    ts_policy_tensor sub;
    sub.match = {"attn_q", "attn_k"};
    sub.exact = false;
    CHECK(ts_policy_match(sub, "blk.0.attn_q"), "substring match hits");
    CHECK(!ts_policy_match(sub, "blk.0.ffn_gate"), "substring match rejects unrelated");
    CHECK(ts_policy_match(sub, "blk.0.attn_q_bias"), "substring match is not anchored");

    ts_policy_tensor ex;
    ex.match = {"ffn_gate_inp"};
    ex.exact = true;
    CHECK(ts_policy_match(ex, "ffn_gate_inp"), "exact match hits");
    CHECK(!ts_policy_match(ex, "blk.0.ffn_gate_inp"), "exact match rejects substring");

    // longest-fragment tie-break: both families match, the longer fragment wins
    ts_policy_tensor shortfrag;
    shortfrag.match = {"attn"};
    shortfrag.exact = false;
    ts_policy_tensor longfrag;
    longfrag.match = {"attn_q_weight"};
    longfrag.exact = false;
    std::vector<std::pair<std::string, ts_policy_tensor>> tie;
    tie.emplace_back("short", shortfrag);
    tie.emplace_back("long",  longfrag);
    const ts_policy_tensor * sel = ts_policy_select(tie, "blk.0.attn_q_weight");
    CHECK(sel != nullptr, "select returns a match");
    CHECK(sel == &tie[1].second, "longer fragment wins");

    // exact beats substring even when the substring is longer
    ts_policy_tensor exact_short;
    exact_short.match = {"x"};
    exact_short.exact = true;
    ts_policy_tensor substr_long;
    substr_long.match = {"x_bias_long_fragment"};
    substr_long.exact = false;
    std::vector<std::pair<std::string, ts_policy_tensor>> tie2;
    tie2.emplace_back("substr_long", substr_long);
    tie2.emplace_back("exact_short", exact_short);
    const ts_policy_tensor * sel2 = ts_policy_select(tie2, "x");
    CHECK(sel2 == &tie2[1].second, "exact rank beats longer substring");
    CHECK(ts_policy_select(tie2, "no_match_here") == nullptr, "no-match returns null");

    // end-to-end selection over the round-trip policy
    const ts_policy_tensor * as = ts_policy_select(r.tensors, "blk.5.attn_v");
    CHECK(as != nullptr && as->family == "attention", "select picks attention for attn_v");

    // -------------------------------------------------------------------
    // 4. backward-compat: legacy top-level `tensors` + `provenance` shape
    // -------------------------------------------------------------------
    const std::string legacy = "/tmp/test_policy_legacy.json";
    {
        std::ofstream lf(legacy);
        lf << "{\n"
           << "  \"schema\": \"llama.speculative.calibration-policy.v1\",\n"
           << "  \"provenance\": { \"seed\": 7, \"generations\": 2, \"islands\": 1,\n"
           << "    \"population\": 4, \"timestamp\": \"T\", \"build_info\": \"b\", \"main_tip\": \"t\" },\n"
           << "  \"tensors\": {\n"
           << "    \"blk.0.attn_q\": { \"family\": \"attn_q\", \"alpha\": 0.6, \"clip\": 0.9,\n"
           << "                       \"expert\": \"awq\", \"mse\": 0.001, \"relative_frob\": 0.002 }\n"
           << "  },\n"
           << "  \"archive\": []\n"
           << "}\n";
    }
    ts_policy lp;
    std::string lerr;
    CHECK(ts_policy_read(legacy.c_str(), &lp, &lerr) == 0, "legacy reads without error");
    if (ts_policy_read(legacy.c_str(), &lp, &lerr) == 0) {
        CHECK(lp.seed == 7, "legacy provenance parsed");
        CHECK(lp.tensors.size() == 1, "legacy tensor count");
        const ts_policy_tensor * lt = find_family(lp, "blk.0.attn_q");
        CHECK(lt != nullptr, "legacy tensor present");
        if (lt != nullptr) {
            // legacy alpha/clip mapped onto genes; expert/mse onto metrics
            CHECK(feq(lt->genes.alpha, 0.6f), "legacy alpha -> genes.alpha");
            CHECK(feq(lt->genes.clip,  0.9f), "legacy clip -> genes.clip");
            CHECK(lt->metrics.expert == "awq", "legacy expert -> metrics.expert");
            CHECK(feq(lt->metrics.mse, 0.001f), "legacy mse -> metrics.mse");
            // a legacy entry with no match list falls back to the key as the
            // substring matcher, so selection still works.
            CHECK(ts_policy_match(*lt, "blk.0.attn_q"), "legacy matcher fallback");
        }
    }

    // -------------------------------------------------------------------
    // 5. production wiring: a policy gene payload maps onto quant params
    // Mirrors how tessera-dispatch.cpp applies ts_policy_select results.
    // -------------------------------------------------------------------
    {
        ts_policy_tensor g;
        g.family = "attention";
        g.match  = {"attn_q"};
        g.genes.alpha             = 0.72f;
        g.genes.clip              = 0.95f;
        g.genes.outlier_fraction  = 0.0042f;
        g.genes.ternary_threshold = 1.1f;
        std::vector<std::pair<std::string, ts_policy_tensor>> fams;
        fams.emplace_back("attention", g);
        const ts_policy_tensor * hit = ts_policy_select(fams, "blk.2.attn_q");
        CHECK(hit != nullptr, "wiring: family resolved");
        if (hit != nullptr) {
            // the same mapping tessera-dispatch.cpp performs
            float applied_alpha  = hit->genes.alpha;
            float applied_clip   = hit->genes.clip;
            float applied_thresh = hit->genes.outlier_fraction;
            CHECK(feq(applied_alpha, 0.72f),  "wiring: alpha applied");
            CHECK(feq(applied_clip,  0.95f),  "wiring: clip applied");
            CHECK(feq(applied_thresh, 0.0042f), "wiring: outlier_thresh applied");
        }
        CHECK(ts_policy_select(fams, "blk.2.ffn_gate") == nullptr,
              "wiring: non-matching tensor is left untouched");
    }

    // -------------------------------------------------------------------
    // sha256: deterministic and non-zero
    // -------------------------------------------------------------------
    uint8_t h1[32], h2[32];
    ts_policy_sha256(rwp.c_str(), h1);
    ts_policy_sha256(rwp.c_str(), h2);
    CHECK(memcmp(h1, h2, 32) == 0, "sha256 deterministic");
    uint8_t zero[32] = {0};
    CHECK(memcmp(h1, zero, 32) != 0, "sha256 non-zero");

    if (g_failures == 0) {
        std::printf("test_policy: all checks passed\n");
    } else {
        std::printf("test_policy: %d failure(s)\n", g_failures);
    }
    return g_failures;
}
