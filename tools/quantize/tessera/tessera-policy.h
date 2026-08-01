#pragma once

//
// tessera-policy.h
//
// Reader/writer for the Tessera calibration policy JSON
// (schema: llama.speculative.calibration-policy.v1).
//
// The canonical writer is tools/tessera/awq-evolve.py (build_policy /
// policy_entry), which emits a top-level `tensor_families` object whose
// values carry the AWQ + Tessera sparse-residual genes. per_tensor_calibrate
// extends the same entries with `ternary_threshold`. This reader consumes
// that shape natively and still accepts the older top-level `tensors` form
// (alpha/clip/expert/mse/relative_frob) for on-disk backward compatibility.
//

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// AWQ + Tessera sparse-residual gene payload shared by the policy reader and
// the per-tensor search. This is the canonical "policy gene payload": B1's
// ts_awq_candidate carries the same space, so keep it a plain aggregate.
//
// Field defaults mirror the Python writer (awq-evolve.py Candidate.clipped
// and per_tensor_calibrate.py): alpha in [0,1] (0 = no AWQ), clip in
// [0.7,1.0], outlier_fraction in [0.0001,0.05], moment_mix in [0,1],
// tail_guard in [0,2], ternary_threshold in [0.3,3.0] (1.0 = legacy).
struct ts_policy_genes {
    float alpha            = 0.0f;  // awq_alpha: AWQ exponent (0 = off)
    float clip             = 1.0f;  // awq_clip: outlier clip (1 = none)
    float outlier_fraction = 0.0f;  // sparse-residual fraction
    float moment_mix       = 0.0f;  // kurtosis moment weight
    float tail_guard       = 0.0f;  // tail-excess weight
    float ternary_threshold = 1.0f; // per-row mean(|W|) multiplier; 1 = legacy
};

// Metrics recorded for a family by the search archive. Kept separate from
// the gene payload so B1's candidate can reuse ts_policy_genes untouched.
struct ts_policy_metrics {
    std::string expert;         // "awq", "lrq", "dartquant", "flrq", "champq", "septq"
    float       mse           = 0.0f;
    float       relative_frob = 0.0f;
};

// One entry under `tensor_families` (new schema) or `tensors` (legacy).
// `match` is a list of substrings; `exact` selects full-name match else
// substring match (matches Python's tensor_policy in quantize_v3.py).
struct ts_policy_tensor {
    std::string        family;   // key under tensor_families / tensors
    std::vector<std::string> match;
    bool               exact = false;
    ts_policy_genes    genes;
    ts_policy_metrics  metrics;
};

struct ts_policy_archive_entry {
    int32_t     cell[3];    // kurtosis_bucket, eff_rank_bucket, family_bucket
    float       alpha;
    float       clip;
    std::string expert;
    float       mse;
};

struct ts_policy {
    uint64_t    seed        = 0;
    int64_t     generations = 0;
    int64_t     islands     = 0;
    int64_t     population  = 0;
    std::string timestamp;
    std::string build_info;
    std::string main_tip;
    std::string search_schema;     // llama.tessera.awq-evolution.v1
    std::string draft_type;        // "hybrid" when emitted
    // insertion-ordered families (the writer emits an ordered dict). The key
    // is the family name (e.g. "attention", "override:blk.3...").
    std::vector<std::pair<std::string, ts_policy_tensor>> tensors;
    std::vector<ts_policy_archive_entry> archive;
};

// Read policy from JSON file. Returns 0 on success, -1 on error.
// On error, err_msg (if non-null) receives a description. Accepts both the
// current tensor_families shape and the legacy top-level tensors shape.
int ts_policy_read(const char * path, ts_policy * out, std::string * err_msg);

// Write policy to JSON file. Returns 0 on success, -1 on error.
int ts_policy_write(const char * path, const ts_policy * policy);

// Compute SHA-256 of the policy file (for provenance).
// out must be 32 bytes.
void ts_policy_sha256(const char * path, uint8_t * out);

// Match `name` against a single family's match/exact semantics.
// exact=true  -> name must appear in match verbatim
// exact=false -> any substring in match must occur in name
bool ts_policy_match(const ts_policy_tensor & family, const std::string & name);

// Resolve the best-matching family for `name`. Mirrors Python's
// tensor_policy tie-break: prefer exact matches, then the longest matching
// fragment. Returns nullptr when no family matches.
const ts_policy_tensor * ts_policy_select(
    const std::vector<std::pair<std::string, ts_policy_tensor>> & families,
    const std::string & name);
