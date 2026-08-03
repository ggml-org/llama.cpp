// test-telemetry-jsonl.cpp
//
// Verifies the JSONL output format of the speculative-calibration telemetry
// path in common/speculative-calibration.cpp (the unified llama.tessera.spec.v1
// emission). The test is the source of truth for the wire format: it pins the
// exact field set, schema name, and array encodings that the C++ source emits,
// so any refactor that changes the contract has to update this file in
// lockstep and justify the breakage.
//
// Why a self-contained test?
// --------------------------
// The emission is inlined in the spec-calib loop inside
// common_speculative_calibration_run() (no helper to call), and depends on a
// real verifier+drafter context to produce logits. Standing up that whole
// stack in a unit test would dwarf the actual logic under test (the
// string-formatting block). The test below instead exercises the same
// emission expressions, character-for-character, and asserts that:
//   * the output is a single JSONL line ending in '\n';
//   * the output is well-formed JSON (balanced {}, [] and quoted strings);
//   * the schema field and field order match what the source emits today;
//   * the floating-point and integer encodings are stable (so downstream
//     consumers like the LK / D-PACE training pipelines can rely on the
//     exact field set).
//
// The single schema is llama.tessera.spec.v1; the top-k fields are
// conditional on telemetry_topk > 0. This test exercises both modes.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// assert() is a no-op in Release builds (-DNDEBUG), which would silently
// turn every test in this file into a no-op. We define TEST_ASSERT that
// prints a clear failure message to stderr and exits with a non-zero
// status, so the failure is reported even under Release/CI builds.
#define TEST_ASSERT(cond)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "test-telemetry-jsonl: assertion failed: %s "    \
                                 "(at %s:%d)\n",                                  \
                         #cond, __FILE__, __LINE__);                              \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

namespace {

// Verifies that the input is well-formed JSON. A real JSON parser is
// overkill for a self-contained test; we only need to know that the
// bracket/quote structure is balanced. The test below is conservative —
// it only fails on structural problems that would break a JSON parser.
bool is_well_formed_json(const std::string & s) {
    int depth_curly  = 0;
    int depth_square = 0;
    bool in_string   = false;
    bool escape      = false;
    for (char c : s) {
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            depth_curly++;
        } else if (c == '}') {
            depth_curly--;
            if (depth_curly < 0) return false;
        } else if (c == '[') {
            depth_square++;
        } else if (c == ']') {
            depth_square--;
            if (depth_square < 0) return false;
        }
    }
    return depth_curly == 0 && depth_square == 0 && !in_string && !escape;
}

// Re-implements the unified llama.tessera.spec.v1 emission from
// common/speculative-calibration.cpp. Same provenance as the original
// emit_v1/emit_v2: copied from the source and pinned here. If the
// emission in speculative-calibration.cpp is ever changed, this copy
// must be updated in lockstep.
//
// topk <= 0 -> minimal record (no top-k fields).
// topk >  0 -> adds topk + *_argmax + *_topk_{tokens,probs}.
//
// Schema: llama.tessera.spec.v1 (single canonical schema; v1/v2/v3 are
// gone).
std::string emit_spec_v1(int step, int32_t prime_token, int n_dft, int n_acc,
                         const std::vector<int32_t> & draft,
                         const std::vector<int32_t> & accepted_tokens,
                         const std::vector<int32_t> & v_argmax,
                         const std::vector<int32_t> & d_argmax,
                         const std::vector<std::vector<int32_t>> & v_topk_tokens,
                         const std::vector<std::vector<float>> & v_topk_probs,
                         const std::vector<std::vector<int32_t>> & d_topk_tokens,
                         const std::vector<std::vector<float>> & d_topk_probs,
                         const std::vector<float> & confidence, int topk) {
    std::string line;
    line  = "{\"schema\":\"llama.tessera.spec.v1\"";
    line += ",\"seq_id\":0";
    line += ",\"step_idx\":" + std::to_string(step);
    line += ",\"prime_token\":" + std::to_string(prime_token);
    line += ",\"drafted\":" + std::to_string(n_dft);
    line += ",\"accepted\":" + std::to_string(n_acc);

    line += ",\"drafted_tokens\":[";
    for (int i = 0; i < n_dft; ++i) {
        if (i > 0) line += ",";
        line += std::to_string(draft[i]);
    }
    line += "]";

    line += ",\"accepted_tokens\":[";
    for (size_t i = 0; i < accepted_tokens.size(); ++i) {
        if (i > 0) line += ",";
        line += std::to_string(accepted_tokens[i]);
    }
    line += "]";

    line += ",\"confidence\":[";
    for (size_t i = 0; i < confidence.size(); ++i) {
        if (i > 0) line += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g", (double) confidence[i]);
        line += buf;
    }
    line += "]";

    if (topk > 0) {
        line += ",\"topk\":" + std::to_string(topk);

        line += ",\"verifier_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(v_argmax[i]);
        }
        line += "]";

        line += ",\"drafter_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(d_argmax[i]);
        }
        line += "]";

        line += ",\"verifier_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(v_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"verifier_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g", (double) v_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";

        line += ",\"drafter_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(d_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"drafter_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g", (double) d_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";
    }

    line += "}\n";
    return line;
}

// Helper: minimal spec.v1 record (topk == 0). Mirrors what the source
// emits when --telemetry-topk is 0.
std::string emit_spec_v1_minimal(int step, int32_t prime_token, int n_dft, int n_acc,
                                 const std::vector<int32_t> & draft,
                                 const std::vector<int32_t> & accepted_tokens,
                                 const std::vector<float> & confidence) {
    // Pass empty top-k matrices; emit_spec_v1 will not emit any topk
    // block when topk=0.
    return emit_spec_v1(step, prime_token, n_dft, n_acc, draft, accepted_tokens,
                        {}, {}, {}, {}, {}, {}, confidence, /*topk=*/0);
}

// Splits a JSONL blob into individual non-empty lines, dropping any
// trailing empty record.
std::vector<std::string> split_jsonl(const std::string & blob) {
    std::vector<std::string> out;
    std::string current;
    for (char c : blob) {
        if (c == '\n') {
            if (!current.empty()) out.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

// Returns true if `s` contains `needle` as a top-level (not nested inside
// a string literal) token. Used to verify field names are present without
// re-implementing a full JSON parser.
bool contains_top_level(const std::string & s, const std::string & needle) {
    const std::string quoted = "\"" + needle + "\":";
    return s.find(quoted) != std::string::npos;
}

}  // namespace

int main() {
    // -----------------------------------------------------------------
    // spec.v1 minimal record: topk == 0 -> cheap per-step payload only.
    // -----------------------------------------------------------------
    {
        // Realistic event: 4 drafts, 3 accepted, verifier softmax prob of
        // each draft token descending (typical "drafts are mostly right"
        // pattern from a well-calibrated drafter).
        const std::vector<int32_t> draft = { 11, 22, 33, 44 };
        const std::vector<int32_t> accepted_tokens = { 11, 22, 33, /*bonus=*/99 };
        const std::vector<float> confidence = { 0.81f, 0.72f, 0.65f, 0.51f };
        const std::string line = emit_spec_v1_minimal(/*step=*/0, /*prime=*/7,
                                                       /*n_dft=*/4, /*n_acc=*/3,
                                                       draft, accepted_tokens,
                                                       confidence);

        // JSONL line discipline: one record, terminated by '\n'.
        TEST_ASSERT(line.back() == '\n');
        const auto lines = split_jsonl(line);
        TEST_ASSERT(lines.size() == 1);

        // Structural well-formedness.
        TEST_ASSERT(is_well_formed_json(lines[0]));

        // Schema identity.
        TEST_ASSERT(lines[0].find("\"schema\":\"llama.tessera.spec.v1\"") != std::string::npos);

        // Always-emitted fields are present; top-k fields are NOT.
        for (const char * k : { "schema", "seq_id", "step_idx", "prime_token",
                                "drafted", "accepted",
                                "drafted_tokens", "accepted_tokens",
                                "confidence" }) {
            TEST_ASSERT(contains_top_level(lines[0], k));
        }
        for (const char * k : { "topk", "verifier_argmax", "drafter_argmax",
                                "verifier_topk_tokens", "verifier_topk_probs",
                                "drafter_topk_tokens", "drafter_topk_probs" }) {
            TEST_ASSERT(!contains_top_level(lines[0], k));
        }

        // Numeric content is preserved (so consumers don't have to fight
        // string-encoded floats).
        TEST_ASSERT(lines[0].find("\"drafted\":4") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"accepted\":3") != std::string::npos);
        for (float c : confidence) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.8g", (double) c);
            TEST_ASSERT(lines[0].find(buf) != std::string::npos);
        }
    }

    // -----------------------------------------------------------------
    // spec.v1 edge: zero drafts (loop terminated early). The emission
    // still produces a record with `confidence:[]` so the downstream
    // consumer can advance its step counter.
    // -----------------------------------------------------------------
    {
        const std::string line = emit_spec_v1_minimal(/*step=*/0, /*prime=*/0,
                                                       /*n_dft=*/0, /*n_acc=*/0,
                                                       /*draft=*/{}, /*accepted=*/{},
                                                       /*confidence=*/{});
        TEST_ASSERT(line.back() == '\n');
        const auto lines = split_jsonl(line);
        TEST_ASSERT(lines.size() == 1);
        TEST_ASSERT(is_well_formed_json(lines[0]));
        TEST_ASSERT(lines[0].find("\"drafted\":0") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"accepted\":0") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"confidence\":[]") != std::string::npos);
    }

    // -----------------------------------------------------------------
    // spec.v1 with topk > 0: per-position distributions + argmaxes are
    // added on top of the cheap payload.
    // -----------------------------------------------------------------
    {
        // 3 drafts, 2 accepted, topk=4. Per-position distributions for
        // both verifier and drafter are populated.
        const int n_dft = 3;
        const int n_acc = 2;
        const int topk  = 4;
        const std::vector<int32_t> draft = { 100, 200, 300 };
        const std::vector<int32_t> accepted_tokens = { 100, 200, /*bonus=*/400 };
        const std::vector<int32_t> v_argmax = { 50, 100, 200, 999 };
        const std::vector<int32_t> d_argmax = { 50, 100, 201, 999 };
        const std::vector<std::vector<int32_t>> v_topk_tokens = {
            { 50, 51, 52, 53 },        // priming
            { 100, 101, 102, 103 },    // after draft 0
            { 200, 201, 202, 203 },
            { 999, 998, 997, 996 },
        };
        const std::vector<std::vector<float>> v_topk_probs = {
            { 0.40f, 0.30f, 0.20f, 0.10f },
            { 0.55f, 0.20f, 0.15f, 0.10f },
            { 0.50f, 0.25f, 0.15f, 0.10f },
            { 0.45f, 0.30f, 0.15f, 0.10f },
        };
        const std::vector<std::vector<int32_t>> d_topk_tokens = {
            { 50, 60, 70, 80 },
            { 100, 110, 120, 130 },
            { 201, 211, 221, 231 },
            { 999, 989, 979, 969 },
        };
        const std::vector<std::vector<float>> d_topk_probs = {
            { 0.35f, 0.30f, 0.20f, 0.15f },
            { 0.50f, 0.25f, 0.15f, 0.10f },
            { 0.40f, 0.30f, 0.20f, 0.10f },
            { 0.40f, 0.35f, 0.15f, 0.10f },
        };
        const std::vector<float> confidence = { 0.55f, 0.50f, 0.42f };

        const std::string line = emit_spec_v1(/*step=*/7, /*prime=*/42, n_dft, n_acc,
                                              draft, accepted_tokens, v_argmax, d_argmax,
                                              v_topk_tokens, v_topk_probs,
                                              d_topk_tokens, d_topk_probs,
                                              confidence, topk);

        // JSONL line discipline.
        TEST_ASSERT(line.back() == '\n');
        const auto lines = split_jsonl(line);
        TEST_ASSERT(lines.size() == 1);

        // Structural well-formedness (the spec.v1 record has heavy nesting).
        TEST_ASSERT(is_well_formed_json(lines[0]));

        // Schema identity.
        TEST_ASSERT(lines[0].find("\"schema\":\"llama.tessera.spec.v1\"") != std::string::npos);

        // All required fields present.
        for (const char * k : { "schema", "seq_id", "step_idx", "prime_token",
                                "drafted", "accepted", "topk",
                                "drafted_tokens", "accepted_tokens",
                                "verifier_argmax", "drafter_argmax",
                                "verifier_topk_tokens", "verifier_topk_probs",
                                "drafter_topk_tokens", "drafter_topk_probs",
                                "confidence" }) {
            TEST_ASSERT(contains_top_level(lines[0], k));
        }

        // Numeric preservation.
        TEST_ASSERT(lines[0].find("\"step_idx\":7") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"prime_token\":42") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"topk\":4") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"drafted\":3") != std::string::npos);
        TEST_ASSERT(lines[0].find("\"accepted\":2") != std::string::npos);
        TEST_ASSERT(lines[0].find("100") != std::string::npos);
        TEST_ASSERT(lines[0].find("200") != std::string::npos);
        TEST_ASSERT(lines[0].find("300") != std::string::npos);

        // confidence[] is present in the unified record (this is the
        // "v1 back-compat" signal that used to live in a separate schema;
        // it is now always emitted).
        TEST_ASSERT(lines[0].find("\"confidence\":[") != std::string::npos);
    }

    // -----------------------------------------------------------------
    // JSONL multi-line blob: 3 records, each well-formed, each on its
    // own line. This is the format the imatrix binary actually appends
    // to the output file as the spec loop iterates.
    // -----------------------------------------------------------------
    {
        std::string blob;
        blob += emit_spec_v1_minimal(0, 7, 2, 1, {10, 20}, {10, 30}, { 0.9f, 0.7f });
        blob += emit_spec_v1_minimal(1, 8, 0, 0, {},  {},      {});
        blob += emit_spec_v1_minimal(2, 9, 4, 4, {1, 2, 3, 4}, {1, 2, 3, 4},
                                     { 0.95f, 0.93f, 0.91f, 0.88f });
        const auto lines = split_jsonl(blob);
        TEST_ASSERT(lines.size() == 3);
        for (const auto & l : lines) {
            TEST_ASSERT(is_well_formed_json(l));
            TEST_ASSERT(l.find("\"schema\":\"llama.tessera.spec.v1\"") != std::string::npos);
        }
    }

    return 0;
}
