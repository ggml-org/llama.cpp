// SPDX-License-Identifier: MIT
#include "artifact_manifest.h"
#include "spec_sidecar.h"
#include "../common/speculative.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using spec_sidecar_artifact::ManifestParser;
using spec_sidecar_artifact::TensorDesc;

static bool parses(const char * json, std::vector<TensorDesc> & tensors) {
    const std::string text(json);
    std::string error;
    ManifestParser parser(text, error);
    return parser.parse(tensors);
}

static int require(bool condition, const char * label) {
    if (condition) return 0;
    std::fprintf(stderr, "FAILED: %s\n", label);
    return 1;
}

static void set_environment(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void unset_environment(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int main() {
    int failures = 0;

    // The master gate must run before target metadata, artifact, or library
    // inspection. The deliberately invalid model pointer makes an accidental
    // pre-gate dereference fail immediately instead of merely returning false.
    common_params_speculative gate_params;
    gate_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    set_environment("LLAMA_SPEC_HIP_SIDECAR", "/definitely/missing/spec_hip_sidecar.so");
    set_environment("LLAMA_SPEC_HIP_WEIGHTS", "/definitely/missing");
    unset_environment("SPEC_SIDECAR");
    std::string gate_error = "stale";
    const auto disabled = common_speculative_sidecar_preflight(
            gate_params, reinterpret_cast<const llama_model *>(static_cast<uintptr_t>(1)), 1, gate_error);
    failures += require(disabled == COMMON_SPECULATIVE_TYPE_NONE &&
                        !gate_params.draft.sidecar_only &&
                        gate_params.draft.sidecar_type == COMMON_SPECULATIVE_TYPE_NONE &&
                        gate_params.draft.sidecar_profile == nullptr && gate_error.empty(),
                        "unset SPEC_SIDECAR disables preflight before model or artifact inspection");

    set_environment("SPEC_SIDECAR", "0");
    gate_error = "stale";
    const auto zero = common_speculative_sidecar_preflight(
            gate_params, reinterpret_cast<const llama_model *>(static_cast<uintptr_t>(1)), 1, gate_error);
    failures += require(zero == COMMON_SPECULATIVE_TYPE_NONE && gate_error.empty(),
                        "SPEC_SIDECAR=0 disables preflight");

    for (const char * value : {"", "01", "true", "2"}) {
        set_environment("SPEC_SIDECAR", value);
        gate_error = "stale";
        const auto rejected = common_speculative_sidecar_preflight(
                gate_params, reinterpret_cast<const llama_model *>(static_cast<uintptr_t>(1)), 1, gate_error);
        failures += require(rejected == COMMON_SPECULATIVE_TYPE_NONE && gate_error.empty(),
                            "only the exact SPEC_SIDECAR=1 value enables preflight");
    }
    unset_environment("SPEC_SIDECAR");

    const auto profile_count = common_spec_sidecar_profile_count();
    failures += require(profile_count >= 2, "provider registry exposes multiple profiles");
    const auto * profile0 = common_spec_sidecar_profile_at(0);
    const auto * profile1 = common_spec_sidecar_profile_at(1);
    failures += require(profile0 != nullptr && profile1 != nullptr &&
                        profile0->kind != profile1->kind &&
                        ((profile0->kind == COMMON_SPEC_SIDECAR_KIND_MTP && profile1->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH) ||
                         (profile0->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH && profile1->kind == COMMON_SPEC_SIDECAR_KIND_MTP)),
                        "MTP and DFlash are selected as distinct provider profiles");
    const auto * mtp_profile = profile0->kind == COMMON_SPEC_SIDECAR_KIND_MTP ? profile0 : profile1;
    const auto * dflash_profile = profile0->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH ? profile0 : profile1;
    failures += require(mtp_profile->mtp_embedding_width == 5120 && mtp_profile->mtp_head_rows == 40960 &&
                        dflash_profile->dflash_encoded_width == 25600 && dflash_profile->dflash_block_size == 8 &&
                        dflash_profile->dflash_head_rows == 40960,
                        "provider profiles carry independent dimensions and head-row contracts");

    failures += require(sizeof(spec_sidecar_state) == 24,
                        "sidecar state ABI is a fixed 24-byte record");
    failures += require(SPEC_SIDECAR_STATE_VERSION == 1 &&
                        SPEC_SIDECAR_STATE_KIND_MTP == 1 && SPEC_SIDECAR_STATE_KIND_DFLASH == 2,
                        "sidecar state ABI constants are stable");
    failures += require(offsetof(spec_sidecar_state, pos_min) == 8 &&
                        offsetof(spec_sidecar_state, pos_max) == 12 &&
                        offsetof(spec_sidecar_state, epoch) == 16,
                        "sidecar state ABI field offsets are stable");
    failures += require(SPEC_SIDECAR_MTP_DRAFT_TOP_K == 32 &&
                        SPEC_SIDECAR_DFLASH_DRAFT_TOP_K == 16,
                        "sidecar stochastic top-k constants are stable");
    const double u0 = spec_sidecar_stochastic_uniform(UINT64_C(1234), 0);
    failures += require(u0 >= 0.0 && u0 < 1.0 &&
                        u0 == spec_sidecar_stochastic_uniform(UINT64_C(1234), 0) &&
                        u0 != spec_sidecar_stochastic_uniform(UINT64_C(1234), 1),
                        "sidecar proposal RNG is deterministic and bounded");

    std::vector<TensorDesc> tensors;
    const char * valid =
        "{\"schema\":1,\"generator\":{\"name\":\"ignored\"},\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],\"offset\":0,\"nbytes\":8}]}";
    failures += require(parses(valid, tensors), "valid manifest parses");
    std::string error;
    failures += require(tensors.size() == 1 &&
                        spec_sidecar_artifact::validate_blob_layout(tensors, 8, error),
                        "valid contiguous layout passes");

    tensors.clear();
    failures += require(parses("{\"tensors\":[]}", tensors),
                        "legacy manifest without schema remains supported");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"tensors\":[{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],\"offset\":0}]}",
        tensors), "missing required tensor field is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1.0,\"tensors\":[]}", tensors),
        "fractional schema is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":2,\"tensors\":[]}", tensors),
        "unsupported schema is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"schema\":1,\"tensors\":[]}", tensors),
        "duplicate schema field is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"tensors\":[{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],"
        "\"offset\":18446744073709551616,\"nbytes\":8}]}", tensors),
        "overflowing integer is rejected");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"metadata\":{\"name\":\"not-a-tensor\",\"offset\":999},\"tensors\":[]}",
        tensors) && tensors.empty(), "nested metadata is not scraped as a tensor");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":0,\"nbytes\":4},"
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":4,\"nbytes\":4}]}",
        tensors), "duplicate-name fixture parses structurally");
    error.clear();
    failures += require(!spec_sidecar_artifact::validate_blob_layout(tensors, 8, error),
                        "duplicate tensor name is rejected by layout validation");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":4,\"nbytes\":4}]}",
        tensors), "gapped-layout fixture parses structurally");
    error.clear();
    failures += require(!spec_sidecar_artifact::validate_blob_layout(tensors, 8, error),
                        "gapped tensor range is rejected");

    error.clear();
    failures += require(spec_sidecar_artifact::validate_remap({0, 3, 2}, 4, error),
                        "unique in-range remap passes");
    error.clear();
    failures += require(!spec_sidecar_artifact::validate_remap({0, 3, 3}, 4, error),
                        "duplicate remap id is rejected");
    error.clear();
    failures += require(!spec_sidecar_artifact::validate_remap({0, 4}, 4, error),
                        "out-of-range remap id is rejected");
    error.clear();
    failures += require(!spec_sidecar_artifact::validate_remap({0, -1}, 4, error),
                        "negative remap id is rejected");

    common_spec_sidecar_mtp mtp;
    error.clear();
    failures += require(!mtp.load("relative-sidecar.so", "/absolute/artifacts", "/absolute/ids.bin", 5120, 40960, 1, error) &&
                        error.find("absolute path") != std::string::npos,
                        "MTP loader rejects relative library paths");
    failures += require(!mtp.active(), "MTP loader remains inactive after path rejection");

    common_spec_sidecar_dflash dflash;
    error.clear();
    failures += require(!dflash.load("relative-sidecar.so", "/absolute/artifacts", 25600, 8, 1, error) &&
                        error.find("absolute path") != std::string::npos,
                        "DFlash loader rejects relative library paths");
    failures += require(!dflash.active(), "DFlash loader remains inactive after path rejection");

    if (failures == 0) std::puts("artifact_manifest_test: PASS");
    return failures == 0 ? 0 : 1;
}
