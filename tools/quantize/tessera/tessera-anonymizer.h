#pragma once

//
// tessera-anonymizer.h
//
// Tier-2 escalation anonymizer (docs/self-improving-loop-design.md section
// 4.1 / 4.5 and Phase 5). Scrubs a text payload (source code) so it keeps
// reasoning value but carries no identity/IP before it is sent to a cloud
// teacher. The transform is:
//   - consistent: the same symbol maps to the same pseudonym everywhere;
//   - type-preserving: an identifier's case-shape (ALL_CAPS / Title /
//     lower) is kept so naming-convention reasoning stays valid;
//   - public-aware: language keywords, preprocessor directives, and common
//     stdlib/libc identifiers are NOT renamed (a conservative whitelist);
//   - ephemeral: the de-anonymization map is produced for LOCAL use only
//     and is meant to be shredded after the answer is de-anonymized.
//
// The aggressiveness dial trades reasoning fidelity for privacy:
//   light      rename user symbols only;
//   balanced   + scrub string-literal and comment contents;
//   aggressive + scrub numeric constants and path-like tokens.
//
// This is a pure text transform: no file I/O. The CLI wrapper
// (ts_cli_anonymize in quantize.cpp) handles files. A Swift anonymizer
// service shells out to it in a later wave.
//

#include <stdbool.h>

enum ts_anon_level {
    TS_ANON_LIGHT,
    TS_ANON_BALANCED,
    TS_ANON_AGGRESSIVE,
};

struct ts_anon_params {
    enum ts_anon_level level;    // aggressiveness dial (privacy vs reasoning fidelity)
    bool               emit_map; // when true, ts_anonymize_run fills *map_json
};

void ts_anon_default_params(struct ts_anon_params * p);

// Map a level name ("light" / "balanced" / "aggressive") to its enum.
// Returns 0 on success, -1 on an unknown name.
int ts_anon_level_from_string(const char * s, enum ts_anon_level * out);

// Inverse of ts_anon_level_from_string; never returns NULL.
const char * ts_anon_level_to_string(enum ts_anon_level level);

// Anonymize input_text. On success returns 0 and sets *output_text to a
// freshly-allocated NUL-terminated string (caller frees with free()). When
// p->emit_map is true and map_json is non-NULL, *map_json is set to a
// freshly-allocated JSON string (caller frees with free()); otherwise
// *map_json is set to NULL. The map is schema-versioned and keyed
// pseudonym -> original for local de-anonymization; see the .cpp for the
// exact shape. Returns -1 on error (null params / input / output_text).
int ts_anonymize_run(const struct ts_anon_params * p,
                     const char * input_text,
                     char ** output_text,
                     char ** map_json);
