// Per-step telemetry record emitted by the imatrix spec-decoding calibration
// path. The serialized record uses the single canonical schema
// llama.tessera.spec.v1; the field set is the union of what the previous
// v1 and v2 schemas carried, with the top-k fields added only when topk > 0.
//
//   - always: schema, seq_id, step_idx, prime_token, drafted, accepted,
//             drafted_tokens, accepted_tokens, confidence[]
//   - when topk > 0: topk, verifier_argmax, drafter_argmax,
//                    verifier_topk_tokens, verifier_topk_probs,
//                    drafter_topk_tokens, drafter_topk_probs
//
// The schema name is part of the public contract: consumers branch on it.
// The single schema name emitted by this module is exactly:
//   - "llama.tessera.spec.v1"
//
// Any change to the field set, the schema name, or the field types is a
// breaking change and requires updating every consumer in lockstep.

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
// serializer decides which to emit based on `topk` (topk == 0 suppresses
// the per-position top-k fields).
struct telemetry_record {
    int32_t  seq_id      = 0;
    int32_t  step_idx    = 0;
    int32_t  prime_token = 0;
    int32_t  drafted     = 0;   // number of draft tokens proposed this step
    int32_t  accepted    = 0;   // number of accepted drafts (longest prefix match)

    // Per-draft-position verifier softmax probability of the drafter's pick.
    // Always populated; the unified record always carries confidence[].
    std::vector<float> confidence;

    // Cheap payload fields, always emitted.
    std::vector<int32_t> drafted_tokens;
    std::vector<int32_t> accepted_tokens;

    // Verifier and drafter per-position argmaxes. Only emitted when topk > 0.
    std::vector<int32_t> verifier_argmax;
    std::vector<int32_t> drafter_argmax;

    // Per-position top-k distributions. Parallel arrays per position.
    // Only emitted when topk > 0.
    std::vector<topk_entry> verifier_topk;
    std::vector<topk_entry> drafter_topk;
};

// Serialize `rec` to a single JSONL line (including trailing newline).
//
// Parameter:
//   - topk: if > 0, additionally include the verifier/drafter top-k
//           distributions and the argmax arrays. If 0, emit only the
//           always-present cheap payload.
//
// The schema is always llama.tessera.spec.v1.
std::string build_telemetry_jsonl(const telemetry_record & rec, int topk);

// Schema name emitted by this module. Exposed for tests and consumers
// that need to branch on the schema without hardcoding the string.
constexpr const char * SCHEMA_SPEC_V1 = "llama.tessera.spec.v1";

}  // namespace spec_calib
