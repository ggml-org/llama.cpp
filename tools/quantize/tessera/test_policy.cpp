//
// test_policy.cpp
//
// Smoke test for tessera-policy.cpp: write a policy to JSON, read it
// back, and verify every field round-trips.
//

#include "tessera-policy.h"

#include <cmath>
#include <cstdio>
#include <cstring>

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

int main(void) {
    const char * path = "/tmp/test_policy.json";

    // --- build ---
    ts_policy p;
    p.seed        = 42;
    p.generations = 8;
    p.islands     = 4;
    p.population  = 16;
    p.timestamp   = "2026-07-30T14:00:00Z";
    p.build_info  = "llama.cpp build 640";
    p.main_tip    = "abc123";

    ts_policy_tensor q;
    q.family        = "attn_q";
    q.alpha         = 0.65f;
    q.clip          = 0.95f;
    q.expert        = "awq";
    q.mse           = 0.0012f;
    q.relative_frob = 0.0034f;
    p.tensors["blk.0.attn_q"] = q;

    ts_policy_tensor ffn;
    ffn.family        = "ffn_gate";
    ffn.alpha         = 0.50f;
    ffn.clip          = 0.90f;
    ffn.expert        = "dartquant";
    ffn.mse           = 0.0020f;
    ffn.relative_frob = 0.0041f;
    p.tensors["blk.1.ffn_gate"] = ffn;

    ts_policy_archive_entry e;
    e.cell[0] = 2; e.cell[1] = 3; e.cell[2] = 0;
    e.alpha   = 0.70f;
    e.clip    = 0.90f;
    e.expert  = "dartquant";
    e.mse     = 0.0010f;
    p.archive.push_back(e);

    // --- write ---
    CHECK(ts_policy_write(path, &p) == 0, "write succeeds");

    // --- read ---
    ts_policy r;
    std::string err;
    CHECK(ts_policy_read(path, &r, &err) == 0, "read succeeds");
    if (!err.empty()) {
        std::printf("read error: %s\n", err.c_str());
    }

    // --- verify provenance ---
    CHECK(r.seed        == p.seed,        "seed round-trips");
    CHECK(r.generations == p.generations, "generations round-trips");
    CHECK(r.islands     == p.islands,     "islands round-trips");
    CHECK(r.population  == p.population,  "population round-trips");
    CHECK(r.timestamp   == p.timestamp,   "timestamp round-trips");
    CHECK(r.build_info  == p.build_info,  "build_info round-trips");
    CHECK(r.main_tip    == p.main_tip,    "main_tip round-trips");

    // --- verify tensors ---
    CHECK(r.tensors.size() == p.tensors.size(), "tensor count matches");
    for (const auto & kv : p.tensors) {
        auto it = r.tensors.find(kv.first);
        CHECK(it != r.tensors.end(), "tensor key present");
        if (it == r.tensors.end()) {
            continue;
        }
        const ts_policy_tensor & a = kv.second;
        const ts_policy_tensor & b = it->second;
        CHECK(b.family        == a.family,        "tensor.family round-trips");
        CHECK(feq(b.alpha,     a.alpha),          "tensor.alpha round-trips");
        CHECK(feq(b.clip,      a.clip),           "tensor.clip round-trips");
        CHECK(b.expert        == a.expert,        "tensor.expert round-trips");
        CHECK(feq(b.mse,       a.mse),            "tensor.mse round-trips");
        CHECK(feq(b.relative_frob, a.relative_frob), "tensor.relative_frob round-trips");
    }

    // --- verify archive ---
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

    // --- sha256: deterministic and non-zero ---
    uint8_t h1[32], h2[32];
    ts_policy_sha256(path, h1);
    ts_policy_sha256(path, h2);
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
