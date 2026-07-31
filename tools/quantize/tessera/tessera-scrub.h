#pragma once

//
// tessera-scrub.h
//
// Text-privacy counterpart to tessera-anonymizer. The anonymizer renames
// SYMBOLS (structure-preserving, reversible via a local map); this module
// redacts SECRETS - api keys, tokens, PEM private keys, emails, IP
// addresses, absolute filesystem paths, and secret-looking env values -
// replacing each with a stable, type-labelled placeholder such as
// <secret:api_key> or <secret:path>.
//
// Stateless and deterministic: same input -> same output, no map needed,
// because redaction is one-way by design (a secret, once seen, is not
// recoverable from the scrubbed text). Pair it with the anonymizer for full
// tier-2 egress scrubbing (secrets here, symbols there).
//
// This is a HEURISTIC scrubber: it catches common secret shapes with high
// precision and deliberately favors not mangling ordinary code over catching
// every exotic secret. Known limits (see the rule table in the .cpp):
//   - compressed IPv6 (fe80::1) is NOT matched, to avoid scrubbing the C++
//     scope operator std::foo; only full 8-group IPv6 forms are redacted;
//   - generic api keys require a HYPHEN after the prefix (sk-..., key-...),
//     since C/C++ identifiers cannot contain a hyphen - this keeps false
//     positives near zero but misses underscore forms like sk_live_...;
//   - absolute paths are only matched under a known top-level dir
//     (/Users, /home, /etc, ...) so URL paths and division are left alone.
//
// Pure text transform: no file I/O.
//

// Redact secrets in input_text. On success returns 0 and sets *output_text
// to a freshly-allocated NUL-terminated string (caller frees with free()).
// If n_redactions is non-NULL, *n_redactions is set to the number of secrets
// masked. Returns -1 on error (null input / output_text) or if the rule
// table fails to compile - fail-closed: nothing is emitted rather than
// risking a leak.
int ts_scrub_run(const char * input_text, char ** output_text, int * n_redactions);
