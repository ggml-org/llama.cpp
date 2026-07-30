// Per-step telemetry record emitted by the imatrix spec-decoding calibration
// path. The shape of the serialized record is selected at build time via the
// `v1_compat` flag on `build_telemetry_jsonl`.
//
// Production schema (v3) is a strict superset of v1 + v2:
//   - always: schema, seq_id, step_idx, prime_token, drafted, accepted,
//             drafted_tokens, accepted_tokens, confidence[]
//   - when topk > 0: topk, verifier_argmax, drafter_argmax,
//                    verifier_topk_tokens, verifier_topk_probs,
//                    drafter_topk_tokens, drafter_topk_probs
//
// v1-compat adapter (--telemetry-v1-compat):
//   { schema, seq_id, drafted, accepted, confidence[] }
//
// The schema name is part of the public contract: consumers branch on it. The
// two schema names emitted by this module are exactly:
//   - "llama.spec_calib.v3"
//   - "llama.dflash.acceptance.v1"
//
// Any change to the field set, the schema name, or the field types is a
// breaking change and requires a major version bump.
//
// Schema versioning rationale: see docs/audit-2026-07-29.md §5.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spec_calib {

// One per-position entry in the verifier/drafter top-k distributions. The
// `tokens` and `probs` arrays are parallel: probs[i] is the softmax
// probability of tokens[i] at that draft position.
struct topk_entry {
    std::vector<int32_t> tokens;
    std::vector<float>   probs;
};

// All data needed to serialize one JSONL telemetry record for a single
// spec-decoding step. Caller fills in the fields that are available; the
// serializer decides which to emit based on `topk` and `v1_compat`.
struct telemetry_record {
    int32_t  seq_id      = 0;
    int32_t  step_idx    = 0;
    int32_t  prime_token = 0;
    int32_t  drafted     = 0;   // number of draft tokens proposed this step
    int32_t  accepted    = 0;   // number of accepted drafts (longest prefix match)

    // Per-draft-position verifier softmax probability of the drafter's pick.
    // Always populated (v3 always emits confidence[]; v1-compat also emits it).
    std::vector<float> confidence;

    // Cheap v2 fields, always emitted in v3.
    std::vector<int32_t> drafted_tokens;
    std::vector<int32_t> accepted_tokens;

    // Verifier and drafter per-position argmaxes. Only emitted in v3 when
    // topk > 0 (i.e. when the rest of the top-k fields are also emitted).
    std::vector<int32_t> verifier_argmax;
    std::vector<int32_t> drafter_argmax;

    // Per-position top-k distributions. Parallel arrays per position.
    // Only emitted in v3 when topk > 0.
    std::vector<topk_entry> verifier_topk;
    std::vector<topk_entry> drafter_topk;
};

// Serialize `rec` to a single JSONL line (including trailing newline).
//
// Parameters:
//   - topk: if > 0 and !v1_compat, include the verifier/drafter top-k
//           distributions and the argmax arrays. Ignored when v1_compat
//           is true (v1 never carries topk).
//   - v1_compat: emit the legacy llama.dflash.acceptance.v1 schema with
//               only seq_id, drafted, accepted, confidence[]. All other
//               fields on `rec` are ignored.
std::string build_telemetry_jsonl(const telemetry_record & rec, int topk, bool v1_compat);

// Schema names emitted by this module. Exposed for tests and consumers that
// need to branch on the schema without hardcoding the string.
constexpr const char * SCHEMA_V3       = "llama.spec_calib.v3";
constexpr const char * SCHEMA_V1_COMPAT = "llama.dflash.acceptance.v1";

}  // namespace spec_calib
