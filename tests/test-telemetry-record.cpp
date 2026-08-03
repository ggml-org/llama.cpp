// Unit tests for the unified llama.tessera.spec.v1 telemetry record
// serializer. The serializer is the single source of truth for the schema
// name and field set; these tests pin the contract so any breaking change
// to the schema forces a CI failure.
//
// Run via the existing tests/CMake build (target name: test-telemetry-record).
// The test is dependency-free: it links against the imatrix target's
// telemetry-record object via the tests/CMakeLists.txt entry below.

#include "../tools/imatrix/telemetry-record.h"

#include <cassert>
#include <cstring>
#include <string>

namespace {

// Minimal JSON substring matcher. We don't want to pull in a JSON parser
// dependency for tests, and we only need to assert the field set is stable.
bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

void assert_true(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "ASSERTION FAILED: %s\n", msg);
        std::abort();
    }
}

// Build a representative spec.v1 record (with topk > 0) for the round-trip
// tests.
spec_calib::telemetry_record make_spec_v1_with_topk() {
    spec_calib::telemetry_record rec;
    rec.seq_id      = 0;
    rec.step_idx    = 7;
    rec.prime_token = 42;
    rec.drafted     = 3;
    rec.accepted    = 2;
    rec.confidence  = {0.91f, 0.74f, 0.31f};
    rec.drafted_tokens   = {100, 200, 300};
    rec.accepted_tokens  = {100, 200, 999};  // 2 drafts + bonus
    rec.verifier_argmax  = {400, 100, 200, 500};
    rec.drafter_argmax   = {401, 100, 999, 501};

    rec.verifier_topk.resize(4);
    rec.verifier_topk[0] = {{400, 401, 402}, {0.6f, 0.2f, 0.1f}};
    rec.verifier_topk[1] = {{100, 101, 102}, {0.5f, 0.3f, 0.1f}};
    rec.verifier_topk[2] = {{200, 201, 202}, {0.4f, 0.3f, 0.2f}};
    rec.verifier_topk[3] = {{500, 501, 502}, {0.7f, 0.2f, 0.05f}};

    rec.drafter_topk.resize(4);
    rec.drafter_topk[0] = {{401, 400, 403}, {0.5f, 0.3f, 0.1f}};
    rec.drafter_topk[1] = {{100, 200, 300}, {0.4f, 0.3f, 0.2f}};
    rec.drafter_topk[2] = {{999, 200, 201}, {0.6f, 0.2f, 0.1f}};
    rec.drafter_topk[3] = {{501, 500, 502}, {0.55f, 0.25f, 0.1f}};

    return rec;
}

// A minimal spec.v1 record (no topk fields, just the always-present fields).
spec_calib::telemetry_record make_spec_v1_minimal() {
    spec_calib::telemetry_record rec;
    rec.seq_id      = 0;
    rec.step_idx    = 1;
    rec.prime_token = 5;
    rec.drafted     = 2;
    rec.accepted    = 1;
    rec.confidence  = {0.8f, 0.2f};
    rec.drafted_tokens   = {10, 20};
    rec.accepted_tokens  = {10, 30};
    return rec;
}

void test_spec_v1_with_topk_emits_canonical_schema() {
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_spec_v1_with_topk(), /*topk=*/3);

    // Schema name is the canonical llama.tessera.spec.v1.
    assert_true(contains(line, "\"schema\":\"llama.tessera.spec.v1\""),
                "spec.v1 record must carry schema=llama.tessera.spec.v1");
    // Must NOT carry the legacy schema names (no leakage).
    assert_true(!contains(line, "llama.dflash.acceptance.v1"),
                "spec.v1 record must not mention the legacy v1 schema name");
    assert_true(!contains(line, "llama.spec_calib.v2"),
                "spec.v1 record must not mention the legacy v2 schema name");
    assert_true(!contains(line, "llama.spec_calib.v3"),
                "spec.v1 record must not mention the legacy v3 schema name");

    // Always-present spec.v1 fields.
    assert_true(contains(line, "\"seq_id\":0"),        "seq_id missing");
    assert_true(contains(line, "\"step_idx\":7"),      "step_idx missing");
    assert_true(contains(line, "\"prime_token\":42"),  "prime_token missing");
    assert_true(contains(line, "\"drafted\":3"),       "drafted missing");
    assert_true(contains(line, "\"accepted\":2"),      "accepted missing");
    assert_true(contains(line, "\"drafted_tokens\":[100,200,300]"),
                "drafted_tokens array missing or wrong");
    assert_true(contains(line, "\"accepted_tokens\":[100,200,999]"),
                "accepted_tokens array missing or wrong");
    assert_true(contains(line, "\"confidence\":["),
                "confidence[] must be present in spec.v1");
    // topk fields present.
    assert_true(contains(line, "\"topk\":3"),
                "topk field missing in spec.v1 with topk>0");
    assert_true(contains(line, "\"verifier_argmax\":[400,100,200,500]"),
                "verifier_argmax missing or wrong");
    assert_true(contains(line, "\"drafter_argmax\":[401,100,999,501]"),
                "drafter_argmax missing or wrong");
    assert_true(contains(line, "\"verifier_topk_tokens\":["),
                "verifier_topk_tokens missing");
    assert_true(contains(line, "\"verifier_topk_probs\":["),
                "verifier_topk_probs missing");
    assert_true(contains(line, "\"drafter_topk_tokens\":["),
                "drafter_topk_tokens missing");
    assert_true(contains(line, "\"drafter_topk_probs\":["),
                "drafter_topk_probs missing");

    // Trailing newline (JSONL contract).
    assert_true(!line.empty() && line.back() == '\n',
                "JSONL record must end with a newline");
}

void test_spec_v1_minimal_emits_no_topk_fields() {
    // topk=0 means we emit a spec.v1 record WITHOUT the topk-only fields.
    // This is the default behavior for users who only want the cheap
    // per-step payload.
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_spec_v1_minimal(), /*topk=*/0);

    assert_true(contains(line, "\"schema\":\"llama.tessera.spec.v1\""),
                "minimal spec.v1 must still use the spec.v1 schema name");
    assert_true(contains(line, "\"confidence\":["),
                "minimal spec.v1 must still emit confidence[]");
    assert_true(!contains(line, "\"topk\""),
                "minimal spec.v1 (topk=0) must not emit topk field");
    assert_true(!contains(line, "verifier_topk"),
                "minimal spec.v1 (topk=0) must not emit verifier_topk_*");
    assert_true(!contains(line, "drafter_topk"),
                "minimal spec.v1 (topk=0) must not emit drafter_topk_*");
    assert_true(!contains(line, "verifier_argmax"),
                "minimal spec.v1 (topk=0) must not emit verifier_argmax");
    assert_true(!contains(line, "drafter_argmax"),
                "minimal spec.v1 (topk=0) must not emit drafter_argmax");
}

// Schema contract: the public constant must match the string we actually
// emit. Catches accidental renames in either place.
void test_schema_name_constant_matches_emitted_string() {
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_spec_v1_minimal(), 0);
    assert_true(line.find(std::string("\"") + spec_calib::SCHEMA_SPEC_V1 + "\"") != std::string::npos,
                "SCHEMA_SPEC_V1 constant does not match the emitted spec.v1 schema name");
}

void test_confidence_always_present_in_spec_v1() {
    // Even with empty confidence, spec.v1 must always carry the field.
    spec_calib::telemetry_record rec = make_spec_v1_minimal();
    rec.confidence.clear();
    const std::string line = spec_calib::build_telemetry_jsonl(rec, 0);
    assert_true(contains(line, "\"confidence\":[]"),
                "spec.v1 must always include confidence[] even when empty");
}

void test_spec_v1_minimal_field_set_is_stable() {
    // The always-present field set of spec.v1 is the wire contract for
    // cheap-per-step consumers. Any addition or removal is a breaking
    // change.
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_spec_v1_minimal(), 0);
    // Strip trailing newline and count top-level commas. After the
    // opening `{` we are at depth 1, so we count commas at depth 1
    // (which excludes commas inside the confidence[] and *_tokens[]
    // arrays at depth 2).
    const std::string body = line.substr(0, line.size() - 1);
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    int commas_at_object_root = 0;
    for (size_t i = 0; i < body.size(); ++i) {
        const char c = body[i];
        if (escape) { escape = false; continue; }
        if (c == '\\' && in_string) { escape = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) continue;
        if (c == '{' || c == '[') depth++;
        else if (c == '}' || c == ']') depth--;
        else if (c == ',' && depth == 1) commas_at_object_root++;
    }
    // 8 commas at the object root => 9 fields: schema, seq_id, step_idx,
    // prime_token, drafted, accepted, drafted_tokens, accepted_tokens,
    // confidence. This is the spec.v1 minimal contract; any addition or
    // removal is a breaking change.
    assert_true(commas_at_object_root == 8,
                "spec.v1 minimal record must emit exactly 9 top-level fields "
                "(schema, seq_id, step_idx, prime_token, drafted, accepted, "
                "drafted_tokens, accepted_tokens, confidence)");
}

}  // namespace

int main() {
    test_spec_v1_with_topk_emits_canonical_schema();
    test_spec_v1_minimal_emits_no_topk_fields();
    test_schema_name_constant_matches_emitted_string();
    test_confidence_always_present_in_spec_v1();
    test_spec_v1_minimal_field_set_is_stable();
    return 0;
}
