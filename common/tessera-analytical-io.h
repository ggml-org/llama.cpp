//
// tessera-analytical-io.h
//
// Tessera analytical-output IO (Phase A of the polars-integration scout,
// see docs/tessera-polars-integration-scout.md).
//
// The Tessera calibration / quantization pipeline emits four analytical
// outputs (L3 outlier report, L4 E2E probe report, L5 requant plan, per-layer
// error table). Today each producer hand-rolls a JSON shape on the C++ or
// Python side, and each consumer hand-rolls a parser. The two drift over
// time and the architect has to maintain both surfaces.
//
// This header centralises the writer half of that contract: a single
// C++ helper that, given a schema name and a record, writes a single
// NDJSON line conformant to the schema. The companion Python wrapper
// (tools/tessera/_analytical_io.py) is the reader half: a polars-typed
// reader that consumes the same NDJSON files and returns DataFrames.
//
// The schema files live in common/schemas/<name>.schema.json. Each
// declares:
//   - `required`:   a list of top-level keys the C++ writer enforces
//   - `polars_schema`: a dict mapping column name to polars dtype
//                     (consumed only by the Python wrapper)
//
// No new third-party dependency. nlohmann::json is already a transitive
// dep via llama-common.
//

#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace tessera::analytical {

// Schema names. Hard-coded in the .cpp; the Python side mirrors them.
inline constexpr const char * SCHEMA_L3_OUTLIER = "l3_outlier";
inline constexpr const char * SCHEMA_L4_PROBE  = "l4_probe";
inline constexpr const char * SCHEMA_L5_PLAN   = "l5_plan";
inline constexpr const char * SCHEMA_PER_LAYER = "per_layer_error";

// Resolve the directory holding the schema files. Search order:
//
//   1. The environment variable TESSERA_SCHEMA_DIR (if set and non-empty).
//      This is the path installed-build / container users should set.
//
//   2. The preprocessor define TESSERA_SCHEMA_DIR (baked at build time
//      by CMake). For in-repo builds this is the absolute path to
//      common/schemas so the helper works regardless of cwd.
//
//   3. The fallback "../common/schemas" resolved relative to the
//      current working directory. This last-resort path keeps the
//      helper usable in unit-test contexts that chdir.
//
// The returned path is canonicalised when possible.
std::filesystem::path schema_dir();

// The list of known schema names. Useful for CLI tools that want to
// validate "did the user pass a real schema name?".
const std::vector<std::string> & known_schemas();

// Validation result for the top-level record shape. The C++ helper
// checks the schema's `required` list and the `polars_schema` dtypes
// at the top level only; nested objects are not recursively validated
// (the caller's responsibility).
struct ValidationResult {
    std::vector<std::string> missing_required;
    std::vector<std::string> wrong_type;
    bool is_valid() const { return missing_required.empty() && wrong_type.empty(); }
};

// Validate a record against a named schema's top-level `required`
// array and the per-column `polars_schema` types. Throws
// std::invalid_argument on schema-name-not-found or schema-file-not-found.
// Returns the populated ValidationResult otherwise; the caller decides
// what to do with a non-is_valid result.
ValidationResult validate_top_level(
    const std::string & schema_name,
    const nlohmann::json & record);

// Write a single record as one NDJSON line to `out`. The trailing
// newline is appended; the stream is NOT flushed (caller controls).
//
// If `validate` is true (default), the record is first checked against
// the schema. On validation failure:
//   - In debug builds (NDEBUG not defined), throws std::invalid_argument
//     with a message listing every missing key and every type mismatch.
//   - In release builds, logs a warning to stderr and writes the
//     record anyway. The release behaviour matches the convention
//     of the rest of llama-common: log-and-continue rather than abort.
//
// The schema_name is matched against the known schemas (case-sensitive);
// an unknown name throws std::invalid_argument.
void write_record(std::ostream & out,
                  const std::string & schema_name,
                  const nlohmann::json & record,
                  bool validate = true);

// Convenience: open `path` for append, write one record, close. Creates
// parent directories if needed. Same validation semantics as the
// stream overload.
void write_record(const std::filesystem::path & path,
                  const std::string & schema_name,
                  const nlohmann::json & record,
                  bool validate = true);

// Look up the full path to a schema file. Returns an empty path if
// the name is not in the known list OR if the file does not exist on
// disk under schema_dir().
std::filesystem::path schema_path(const std::string & schema_name);

} // namespace tessera::analytical
