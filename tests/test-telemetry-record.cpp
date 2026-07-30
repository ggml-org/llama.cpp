// Unit tests for the unified spec_calib.v3 telemetry record serializer and
// the v1-compat adapter. The serializer is the single source of truth for
// the schema name and field set; these tests pin the contract so any
// breaking change to the schema forces a CI failure.
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

// Build a representative v3 record (with topk > 0) for the round-trip tests.
spec_calib::telemetry_record make_v3_with_topk() {
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

// A minimal v3 record (no topk fields, just the always-present fields).
spec_calib::telemetry_record make_v3_minimal() {
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

void test_v3_with_topk_emits_canonical_schema() {
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_v3_with_topk(), /*topk=*/3, /*v1_compat=*/false);

    // Schema name is the canonical v3.
    assert_true(contains(line, "\"schema\":\"llama.spec_calib.v3\""),
                "v3 record must carry schema=llama.spec_calib.v3");
    // Must NOT carry the v1 or v2 names (no leakage between schemas).
    assert_true(!contains(line, "llama.dflash.acceptance.v1"),
                "v3 record must not mention v1 schema name");
    assert_true(!contains(line, "llama.spec_calib.v2"),
                "v3 record must not mention v2 schema name (use v3)");

    // Always-present v3 fields.
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
                "confidence[] must be present in v3");
    // topk fields present.
    assert_true(contains(line, "\"topk\":3"),
                "topk field missing in v3 with topk>0");
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

void test_v3_minimal_emits_no_topk_fields() {
    // topk=0 means we emit a v3 record WITHOUT the topk-only fields. This is
    // the default behavior for users who only want the v1-equivalent
    // minimal schema.
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_v3_minimal(), /*topk=*/0, /*v1_compat=*/false);

    assert_true(contains(line, "\"schema\":\"llama.spec_calib.v3\""),
                "minimal v3 must still use v3 schema name");
    assert_true(contains(line, "\"confidence\":["),
                "minimal v3 must still emit confidence[]");
    assert_true(!contains(line, "\"topk\""),
                "minimal v3 (topk=0) must not emit topk field");
    assert_true(!contains(line, "verifier_topk"),
                "minimal v3 (topk=0) must not emit verifier_topk_*");
    assert_true(!contains(line, "drafter_topk"),
                "minimal v3 (topk=0) must not emit drafter_topk_*");
    assert_true(!contains(line, "verifier_argmax"),
                "minimal v3 (topk=0) must not emit verifier_argmax");
    assert_true(!contains(line, "drafter_argmax"),
                "minimal v3 (topk=0) must not emit drafter_argmax");
}

void test_v1_compat_emits_legacy_schema() {
    // The legacy v1 schema carries only seq_id, drafted, accepted,
    // confidence[]. All other fields on the record are dropped.
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_v3_with_topk(), /*topk=*/3, /*v1_compat=*/true);

    assert_true(contains(line, "\"schema\":\"llama.dflash.acceptance.v1\""),
                "v1-compat must carry schema=llama.dflash.acceptance.v1");
    assert_true(contains(line, "\"seq_id\":0"),
                "v1-compat must carry seq_id");
    assert_true(contains(line, "\"drafted\":3"),
                "v1-compat must carry drafted");
    assert_true(contains(line, "\"accepted\":2"),
                "v1-compat must carry accepted");
    assert_true(contains(line, "\"confidence\":["),
                "v1-compat must carry confidence[]");

    // v1 does NOT have these fields. Ensure none leak into v1-compat output.
    assert_true(!contains(line, "step_idx"),
                "v1-compat must not include step_idx");
    assert_true(!contains(line, "prime_token"),
                "v1-compat must not include prime_token");
    assert_true(!contains(line, "drafted_tokens"),
                "v1-compat must not include drafted_tokens");
    assert_true(!contains(line, "accepted_tokens"),
                "v1-compat must not include accepted_tokens");
    assert_true(!contains(line, "topk"),
                "v1-compat must not include topk");
    assert_true(!contains(line, "verifier_"),
                "v1-compat must not include verifier_*");
    assert_true(!contains(line, "drafter_"),
                "v1-compat must not include drafter_*");
}

void test_v1_compat_ignores_topk_parameter() {
    // Even when topk > 0, v1-compat must not emit topk fields. The topk
    // parameter is only meaningful for the v3 schema.
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_v3_with_topk(), /*topk=*/64, /*v1_compat=*/true);

    assert_true(contains(line, "\"schema\":\"llama.dflash.acceptance.v1\""),
                "v1-compat must use v1 schema regardless of topk");
    assert_true(!contains(line, "topk"),
                "v1-compat must not emit topk field even when topk > 0");
    assert_true(!contains(line, "verifier_topk"),
                "v1-compat must not emit verifier_topk even when topk > 0");
    assert_true(!contains(line, "drafter_topk"),
                "v1-compat must not emit drafter_topk even when topk > 0");
}

// Schema contract: the public constants must match the strings we actually
// emit. Catches accidental renames in either place.
void test_schema_name_constants_match_emitted_strings() {
    const std::string v3 = spec_calib::build_telemetry_jsonl(
        make_v3_minimal(), 0, false);
    assert_true(v3.find(std::string("\"") + spec_calib::SCHEMA_V3 + "\"") != std::string::npos,
                "SCHEMA_V3 constant does not match the emitted v3 schema name");

    const std::string v1 = spec_calib::build_telemetry_jsonl(
        make_v3_minimal(), 0, true);
    assert_true(v1.find(std::string("\"") + spec_calib::SCHEMA_V1_COMPAT + "\"") != std::string::npos,
                "SCHEMA_V1_COMPAT constant does not match the emitted v1 schema name");
}

void test_confidence_always_present_in_v3() {
    // Even with empty confidence, v3 must always carry the field. Consumers
    // should be able to rely on `confidence` being present in v3 records.
    spec_calib::telemetry_record rec = make_v3_minimal();
    rec.confidence.clear();
    const std::string line = spec_calib::build_telemetry_jsonl(rec, 0, false);
    assert_true(contains(line, "\"confidence\":[]"),
                "v3 must always include confidence[] even when empty");
}

// Schema stability guarantee: the field set of v1-compat output must remain
// exactly {schema, seq_id, drafted, accepted, confidence} for one major
// version. This is the contract we promise to v1 consumers via the
// --telemetry-v1-compat adapter.
void test_v1_compat_field_set_is_stable() {
    const std::string line = spec_calib::build_telemetry_jsonl(
        make_v3_with_topk(), 64, true);
    // Strip trailing newline and count top-level commas. After the opening
    // `{` we are at depth 1, so we count commas at depth 1 (which excludes
    // commas inside the confidence[] array at depth 2).
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
    // 4 commas at the object root => 5 fields: schema, seq_id, drafted,
    // accepted, confidence. This is the v1 contract; any addition is a
    // breaking change for the v1-compat adapter.
    assert_true(commas_at_object_root == 4,
                "v1-compat must emit exactly 5 top-level fields "
                "(schema, seq_id, drafted, accepted, confidence)");
}

}  // namespace

int main() {
    test_v3_with_topk_emits_canonical_schema();
    test_v3_minimal_emits_no_topk_fields();
    test_v1_compat_emits_legacy_schema();
    test_v1_compat_ignores_topk_parameter();
    test_schema_name_constants_match_emitted_strings();
    test_confidence_always_present_in_v3();
    test_v1_compat_field_set_is_stable();
    return 0;
}
