#pragma once

//
// tessera-config: small INI-style loader for --tessera-* options.
//
// The config file is plain text, one `key = value` per line, with optional
// [section] headers. Lines starting with '#' or ';' are comments; blank
// lines are ignored. Keys are case-sensitive. Quoted values ("..." or
// '...') have their surrounding quotes stripped and embedded escapes
// resolved (\" and \'). Duplicate keys in the same section are rejected.
//
// Precedence at the CLI level (Tessera-2026-08):
//   1. defaults from common_tessera_params
//   2. config file values (loaded by tessera_config_load + applied by
//      tessera_config_apply)
//   3. environment variables (LLAMA_TESSERA_*, etc.)
//   4. explicit CLI flags (--tessera-*)
//
// The apply function writes the same fields the --tessera-* add_opt
// handlers write and runs the same per-key validators. On any error it
// returns false and writes a human-readable message to `err` that names
// the offending key and the source line.
//

#include "tessera-args.h"

#include <map>
#include <string>

struct tessera_config {
    // Values written before any [section] header, or under [general].
    std::map<std::string, std::string> global;
    // Values written under [section-name] headers.
    std::map<std::string, std::map<std::string, std::string>> sections;
};

// Parse the INI text in `text` and populate `out`. `source_label` is used
// only for error messages (typically a file path). On error returns false
// and writes a message to `err` naming the line number.
bool tessera_config_parse(const std::string & text,
                          const std::string & source_label,
                          tessera_config & out,
                          std::string & err);

// Open `path` and forward its contents to tessera_config_parse. On any
// I/O or parse error returns false and writes a message to `err`.
bool tessera_config_load(const std::string & path,
                         tessera_config & out,
                         std::string & err);

// Apply a parsed config to the shared Tessera params struct. Each key is
// dispatched to a typed setter that mirrors the validation performed by
// the corresponding --tessera-* add_opt handler. Returns false and writes
// a message to `err` on any unknown key, type mismatch, or validation
// failure. The caller is responsible for ordering: this function should be
// called BEFORE env-var and CLI parsing so that the latter naturally
// take precedence on later writes.
bool tessera_config_apply(const tessera_config & cfg,
                          common_tessera_params & tessera_params,
                          std::string & err);
