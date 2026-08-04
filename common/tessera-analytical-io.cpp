//
// tessera-analytical-io.cpp
//
// Implementation of the Phase-A polars-integration helper. See
// tessera-analytical-io.h for the design contract and
// docs/tessera-polars-integration-scout.md for the upstream context.
//

#include "tessera-analytical-io.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#ifndef TESSERA_SCHEMA_DIR
// In-repo build fallback. The CMakeLists wires the real path via a
// preprocessor define. The relative fallback is intentionally not
// useful at runtime; it exists so the helper compiles in isolation
// (e.g. the standalone test below) without a CMake configure step.
#define TESSERA_SCHEMA_DIR ""
#endif

namespace tessera::analytical {

namespace {

// Hard-coded list of known schema names. Mirrored on the Python side
// in tools/tessera/_analytical_io.py:KNOWN_SCHEMAS. Keep them in sync.
const std::vector<std::string> k_known_schemas = {
    SCHEMA_L3_OUTLIER,
    SCHEMA_L4_PROBE,
    SCHEMA_L5_PLAN,
    SCHEMA_PER_LAYER,
};

// Map from a polars dtype string (as recorded in the schema JSON's
// `polars_schema` field) to the JSON type we expect to see for that
// column. This is a pragmatic subset: it covers the dtypes that appear
// in the four shipped schemas. The mapping is intentionally permissive
// (Int64 may be served as a JSON integer or a JSON number that
// happens to be integral); strict typing is a future concern.
const std::unordered_map<std::string, std::string> k_polars_to_json_type = {
    {"String",  "string"},
    {"Int8",    "integer"},
    {"Int16",   "integer"},
    {"Int32",   "integer"},
    {"Int64",   "integer"},
    {"UInt8",   "integer"},
    {"UInt16",  "integer"},
    {"UInt32",  "integer"},
    {"UInt64",  "integer"},
    {"Float32", "number"},
    {"Float64", "number"},
    {"Boolean", "boolean"},
    {"Date",    "string"},
    {"Datetime","string"},
};

// Cached schema files. Reading a schema is a small file I/O; for a
// single write call it is irrelevant, but a batch emitter (e.g. one
// L3 outlier per tensor in a 600-tensor model) would otherwise read
// the schema 600 times. A process-wide cache keyed by schema name
// keeps the per-write overhead at a hashmap lookup.
std::unordered_map<std::string, nlohmann::json> & schema_cache() {
    static std::unordered_map<std::string, nlohmann::json> cache;
    return cache;
}

std::mutex & schema_cache_mutex() {
    static std::mutex m;
    return m;
}

const nlohmann::json & load_schema(const std::string & schema_name) {
    std::lock_guard<std::mutex> lock(schema_cache_mutex());
    auto & cache = schema_cache();
    auto it = cache.find(schema_name);
    if (it != cache.end()) {
        return it->second;
    }
    auto path = schema_path(schema_name);
    if (path.empty()) {
        throw std::invalid_argument(
            "tessera::analytical: unknown schema name '" + schema_name + "'");
    }
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument(
            "tessera::analytical: cannot open schema file: " + path.string());
    }
    nlohmann::json schema;
    try {
        in >> schema;
    } catch (const nlohmann::json::parse_error & e) {
        throw std::invalid_argument(
            "tessera::analytical: schema file is not valid JSON: " + path.string() +
            " (" + e.what() + ")");
    }
    auto [inserted_it, inserted] = cache.emplace(schema_name, std::move(schema));
    return inserted_it->second;
}

// Format a validation result as a one-line message suitable for
// log-and-continue in release builds.
std::string format_validation_message(
    const std::string & schema_name,
    const ValidationResult & result) {
    std::ostringstream os;
    os << "tessera::analytical: record does not conform to schema '"
       << schema_name << "': ";
    if (!result.missing_required.empty()) {
        os << "missing_required=[";
        for (size_t i = 0; i < result.missing_required.size(); ++i) {
            if (i) { os << ", "; }
            os << result.missing_required[i];
        }
        os << "] ";
    }
    if (!result.wrong_type.empty()) {
        os << "wrong_type=[";
        for (size_t i = 0; i < result.wrong_type.size(); ++i) {
            if (i) { os << ", "; }
            os << result.wrong_type[i];
        }
        os << "]";
    }
    return os.str();
}

} // namespace

std::filesystem::path schema_dir() {
    // 1. Environment variable override.
    if (const char * env = std::getenv("TESSERA_SCHEMA_DIR"); env != nullptr && env[0] != '\0') {
        return std::filesystem::path(env);
    }
    // 2. Compile-time define (set by CMake in in-repo builds).
    if (std::string(TESSERA_SCHEMA_DIR).size() > 0) {
        return std::filesystem::path(TESSERA_SCHEMA_DIR);
    }
    // 3. Fallback: relative to cwd. Not pretty; only fires for
    //    ad-hoc unit tests that have not been CMake-wired.
    return std::filesystem::path("../common/schemas");
}

std::filesystem::path schema_path(const std::string & schema_name) {
    if (std::find(k_known_schemas.begin(), k_known_schemas.end(), schema_name)
        == k_known_schemas.end()) {
        return {};
    }
    auto path = schema_dir() / (schema_name + ".schema.json");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return {};
    }
    return path;
}

const std::vector<std::string> & known_schemas() {
    return k_known_schemas;
}

ValidationResult validate_top_level(
    const std::string & schema_name,
    const nlohmann::json & record) {
    ValidationResult result;
    const auto & schema = load_schema(schema_name);

    if (!record.is_object()) {
        result.wrong_type.push_back("<record>: expected JSON object at the top level");
        return result;
    }

    // Required-keys check.
    if (schema.contains("required") && schema["required"].is_array()) {
        std::set<std::string> record_keys;
        for (auto it = record.begin(); it != record.end(); ++it) {
            record_keys.insert(it.key());
        }
        for (const auto & required : schema["required"]) {
            if (required.is_string() && record_keys.find(required.get<std::string>()) == record_keys.end()) {
                result.missing_required.push_back(required.get<std::string>());
            }
        }
    }

    // Top-level dtype check (against polars_schema).
    if (schema.contains("polars_schema") && schema["polars_schema"].is_object()) {
        const auto & polars_schema = schema["polars_schema"];
        for (auto it = polars_schema.begin(); it != polars_schema.end(); ++it) {
            const std::string & key = it.key();
            if (!it.value().is_string()) { continue; }
            const std::string & expected_dtype = it.value().get<std::string>();
            auto type_it = k_polars_to_json_type.find(expected_dtype);
            if (type_it == k_polars_to_json_type.end()) { continue; }
            const std::string & expected_json_type = type_it->second;
            if (!record.contains(key)) { continue; } // absence is a missing_required, not a type error
            const auto & value = record[key];
            if (value.is_null()) { continue; } // null is the polars null sentinel; no type claim
            bool ok = false;
            if (expected_json_type == "string") {
                ok = value.is_string();
            } else if (expected_json_type == "integer") {
                ok = value.is_number_integer();
            } else if (expected_json_type == "number") {
                ok = value.is_number();
            } else if (expected_json_type == "boolean") {
                ok = value.is_boolean();
            }
            if (!ok) {
                result.wrong_type.push_back(
                    key + ": expected " + expected_dtype +
                    " (json " + expected_json_type + ")");
            }
        }
    }

    return result;
}

void write_record(std::ostream & out,
                  const std::string & schema_name,
                  const nlohmann::json & record,
                  bool validate) {
    if (validate) {
        auto result = validate_top_level(schema_name, record);
        if (!result.is_valid()) {
#ifndef NDEBUG
            throw std::invalid_argument(format_validation_message(schema_name, result));
#else
            // Release: log to stderr and write anyway. Matches the
            // llama-common convention of log-and-continue on bad
            // input rather than abort.
            std::fprintf(stderr, "%s\n", format_validation_message(schema_name, result).c_str());
#endif
        }
    }
    out << record.dump() << '\n';
}

void write_record(const std::filesystem::path & path,
                  const std::string & schema_name,
                  const nlohmann::json & record,
                  bool validate) {
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        // ec is intentionally ignored: if mkdir fails, the open()
        // below will surface the underlying error.
    }
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out) {
        throw std::invalid_argument(
            "tessera::analytical: cannot open output file: " + path.string());
    }
    write_record(out, schema_name, record, validate);
}

} // namespace tessera::analytical
