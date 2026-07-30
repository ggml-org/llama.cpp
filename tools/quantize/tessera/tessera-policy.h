#pragma once

//
// tessera-policy.h
//
// Reader/writer for the Tessera calibration policy JSON
// (schema: llama.speculative.calibration-policy.v1).
//

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ts_policy_tensor {
    std::string family;
    float       alpha;
    float       clip;
    std::string expert;     // "awq", "lrq", "dartquant", "flrq", "champq", "septq"
    float       mse;
    float       relative_frob;
};

struct ts_policy_archive_entry {
    int32_t     cell[3];    // kurtosis_bucket, eff_rank_bucket, family_bucket
    float       alpha;
    float       clip;
    std::string expert;
    float       mse;
};

struct ts_policy {
    uint64_t    seed;
    int64_t     generations;
    int64_t     islands;
    int64_t     population;
    std::string timestamp;
    std::string build_info;
    std::string main_tip;
    std::map<std::string, ts_policy_tensor> tensors;
    std::vector<ts_policy_archive_entry> archive;
};

// Read policy from JSON file. Returns 0 on success, -1 on error.
// On error, err_msg (if non-null) receives a description.
int ts_policy_read(const char * path, ts_policy * out, std::string * err_msg);

// Write policy to JSON file. Returns 0 on success, -1 on error.
int ts_policy_write(const char * path, const ts_policy * policy);

// Compute SHA-256 of the policy file (for provenance).
// out must be 32 bytes.
void ts_policy_sha256(const char * path, uint8_t * out);
