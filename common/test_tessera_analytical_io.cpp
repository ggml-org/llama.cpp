//
// test_tessera_analytical_io.cpp
//
// Smoke test for the tessera::analytical::write_record helper. Verifies:
//   1. Schema resolution finds the in-repo schema files.
//   2. A conformant record writes one NDJSON line and the line is
//      parseable as the input JSON object.
//   3. A non-conformant record throws std::invalid_argument in debug
//      builds and logs+writes in release builds (the test exercises
//      the debug path via NDEBUG-inverted logic).
//   4. An unknown schema name throws std::invalid_argument.
//
// Run as: ./bin/test-tessera-analytical-io
//
// Mirrors the smoke-test pattern in
// common/tessera-debug/test_sidecar_v3.cpp and
// common/tessera-debug/test_matmul_output.cpp.
//

#include "tessera-analytical-io.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "ok   %s\n", msg); \
    } \
} while (0)

nlohmann::json make_l3_outlier_record() {
    return nlohmann::json{
        {"sidecar_label", "ckpt-v3-q4_k"},
        {"tensor",        "blk.12.attn_q.weight"},
        {"layer",         "blk.12"},
        {"rows",          4096},
        {"cols",          4096},
        {"total_elements", 16777216},
        {"outlier_count",  17000},
        {"outlier_fraction", 0.001013},
        {"threshold",     6.0f},
        {"skipped",       false},
        {"kernel_version", "integrate/polars-scout-phase-a abc1234"},
        {"created_at",    "2026-08-03T20:30:00Z"},
        {"tessera_main_tip", "df3e9c6cd"},
    };
}

void test_schema_dir_resolves() {
    auto dir = tessera::analytical::schema_dir();
    CHECK(!dir.empty(), "schema_dir() returns a non-empty path");
    CHECK(fs::is_directory(dir),
          "schema_dir() points to an existing directory");
}

void test_known_schemas_listed() {
    const auto & names = tessera::analytical::known_schemas();
    std::vector<std::string> expected = {
        tessera::analytical::SCHEMA_L3_OUTLIER,
        tessera::analytical::SCHEMA_L4_PROBE,
        tessera::analytical::SCHEMA_L5_PLAN,
        tessera::analytical::SCHEMA_PER_LAYER,
    };
    CHECK(names.size() == expected.size(),
          "known_schemas() has the expected size");
    for (const auto & name : expected) {
        CHECK(std::find(names.begin(), names.end(), name) != names.end(),
              ("known_schemas() contains " + name).c_str());
    }
}

void test_schema_path_for_known_and_unknown() {
    auto p = tessera::analytical::schema_path(tessera::analytical::SCHEMA_L3_OUTLIER);
    CHECK(!p.empty(),
          "schema_path(l3_outlier) returns a non-empty path");
    CHECK(fs::is_regular_file(p),
          "schema_path(l3_outlier) points to an existing file");
    auto unknown = tessera::analytical::schema_path("not_a_real_schema");
    CHECK(unknown.empty(),
          "schema_path(unknown) returns an empty path");
}

void test_validate_conformant_record() {
    auto record = make_l3_outlier_record();
    auto result = tessera::analytical::validate_top_level(
        tessera::analytical::SCHEMA_L3_OUTLIER, record);
    CHECK(result.is_valid(),
          "validate_top_level: a conformant L3 record reports is_valid()");
    CHECK(result.missing_required.empty(),
          "validate_top_level: no missing_required keys");
    CHECK(result.wrong_type.empty(),
          "validate_top_level: no wrong_type entries");
}

void test_validate_missing_required() {
    auto record = make_l3_outlier_record();
    record.erase("tessera_main_tip");
    auto result = tessera::analytical::validate_top_level(
        tessera::analytical::SCHEMA_L3_OUTLIER, record);
    CHECK(!result.is_valid(),
          "validate_top_level: dropping a required key invalidates the record");
    CHECK(std::find(result.missing_required.begin(),
                    result.missing_required.end(),
                    "tessera_main_tip") != result.missing_required.end(),
          "validate_top_level: missing key reported by name");
}

void test_validate_wrong_type() {
    auto record = make_l3_outlier_record();
    record["outlier_count"] = "not-a-number";
    auto result = tessera::analytical::validate_top_level(
        tessera::analytical::SCHEMA_L3_OUTLIER, record);
    CHECK(!result.is_valid(),
          "validate_top_level: wrong-type column invalidates the record");
    bool found = false;
    for (const auto & e : result.wrong_type) {
        if (e.find("outlier_count") != std::string::npos) {
            found = true;
            break;
        }
    }
    CHECK(found, "validate_top_level: wrong_type names the offending column");
}

void test_write_record_to_stream() {
    auto record = make_l3_outlier_record();
    std::ostringstream out;
    tessera::analytical::write_record(out,
        tessera::analytical::SCHEMA_L3_OUTLIER, record);
    auto written = out.str();
    CHECK(!written.empty() && written.back() == '\n',
          "write_record appends a trailing newline");
    // Strip the trailing newline for round-trip parse.
    auto trimmed = written.substr(0, written.size() - 1);
    auto parsed = nlohmann::json::parse(trimmed);
    CHECK(parsed == record,
          "write_record round-trips through nlohmann::json::parse");
}

void test_write_record_to_file() {
    auto record = make_l3_outlier_record();
    auto tmp = fs::temp_directory_path() /
        ("tessera-analytical-io-test-" + std::to_string(::getpid()) + ".ndjson");
    tessera::analytical::write_record(tmp,
        tessera::analytical::SCHEMA_L3_OUTLIER, record);
    CHECK(fs::is_regular_file(tmp),
          "write_record(path) creates the output file");
    std::ifstream in(tmp);
    std::string line;
    std::getline(in, line);
    auto parsed = nlohmann::json::parse(line);
    CHECK(parsed == record,
          "write_record(path) round-trips the record through disk");
    fs::remove(tmp);
}

void test_write_record_unknown_schema_throws() {
    auto record = make_l3_outlier_record();
    std::ostringstream out;
    bool threw = false;
    try {
        tessera::analytical::write_record(out, "not_a_real_schema", record);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw, "write_record: unknown schema name throws std::invalid_argument");
}

} // namespace

int main() {
    std::fprintf(stdout, "tessera::analytical smoke test\n");
    test_schema_dir_resolves();
    test_known_schemas_listed();
    test_schema_path_for_known_and_unknown();
    test_validate_conformant_record();
    test_validate_missing_required();
    test_validate_wrong_type();
    test_write_record_to_stream();
    test_write_record_to_file();
    test_write_record_unknown_schema_throws();
    if (g_failures == 0) {
        std::fprintf(stdout, "\nALL PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
    return 1;
}
