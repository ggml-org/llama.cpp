// SPDX-License-Identifier: MIT
#include "artifact_manifest.h"
#include "mtp/catchup_alignment.h"
#include "spec_sidecar.h"
#include "../common/speculative.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char ** argv) {
    int failures = 0;

    failures += require(spec_sidecar_mtp::hidden_source_row(0) == -1,
                        "first MTP token row consumes prior committed hidden state");
    failures += require(spec_sidecar_mtp::hidden_source_row(1) == 0 &&
                        spec_sidecar_mtp::hidden_source_row(7) == 6,
                        "MTP catch-up token rows consume the preceding target hidden row");
    failures += require(spec_sidecar_mtp::can_begin_catchup(0, false) &&
                        spec_sidecar_mtp::can_begin_catchup(7, true) &&
                        !spec_sidecar_mtp::can_begin_catchup(7, false),
                        "MTP catch-up requires retained hidden state away from BOS");
    failures += require(spec_sidecar_mtp::committed_hidden_matches_tip(7, 7) &&
                        !spec_sidecar_mtp::committed_hidden_matches_tip(7, 5),
                        "MTP restore retains hidden state only at the unchanged committed tip");

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
    failures += require(profile_count >= 4, "provider registry exposes all independent profiles");
    const common_spec_sidecar_profile * qwen35_mtp = nullptr;
    const common_spec_sidecar_profile * qwen35moe_mtp = nullptr;
    const common_spec_sidecar_profile * qwen35_dflash = nullptr;
    const common_spec_sidecar_profile * qwen4exp_mtp = nullptr;
    for (size_t i = 0; i < profile_count; ++i) {
        const auto * profile = common_spec_sidecar_profile_at(i);
        if (profile == nullptr || profile->name == nullptr) {
            continue;
        }
        if (std::strcmp(profile->name, "qwen35-mtp") == 0) qwen35_mtp = profile;
        if (std::strcmp(profile->name, "qwen35moe-mtp") == 0) qwen35moe_mtp = profile;
        if (std::strcmp(profile->name, "qwen35-dflash") == 0) qwen35_dflash = profile;
        if (std::strcmp(profile->name, "qwen4exp-mtp") == 0) qwen4exp_mtp = profile;
    }
    failures += require(qwen35_mtp != nullptr && qwen35moe_mtp != nullptr &&
                        qwen35_dflash != nullptr && qwen4exp_mtp != nullptr &&
                        qwen35_mtp->kind == COMMON_SPEC_SIDECAR_KIND_MTP &&
                        qwen35moe_mtp->kind == COMMON_SPEC_SIDECAR_KIND_MTP &&
                        qwen35_dflash->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH &&
                        qwen4exp_mtp->kind == COMMON_SPEC_SIDECAR_KIND_MTP,
                        "Qwen3.8-27B and Flash Next use distinct named provider profiles");
    failures += require(qwen35_mtp != nullptr && qwen35moe_mtp != nullptr && qwen35_dflash != nullptr &&
                        qwen35_mtp->mtp_embedding_width == 5120 && qwen35_mtp->mtp_head_rows == 40960 &&
                        std::strcmp(qwen35moe_mtp->target_architecture, "qwen35moe") == 0 &&
                        std::strcmp(qwen35moe_mtp->target_name, "Qwen3.6") == 0 &&
                        std::strcmp(qwen35moe_mtp->target_size_label, "35B-A3B") == 0 &&
                        qwen35moe_mtp->target_n_embd == 2048 &&
                        qwen35moe_mtp->target_n_embd_out == 2048 &&
                        qwen35moe_mtp->target_n_layer == 40 &&
                        qwen35moe_mtp->target_n_layer_nextn == 1 &&
                        qwen35moe_mtp->target_n_vocab == 248320 &&
                        qwen35moe_mtp->mtp_embedding_width == 2048 &&
                        qwen35moe_mtp->mtp_head_rows == 40960 &&
                        qwen35moe_mtp->explicit_paths_only &&
                        std::strcmp(qwen35moe_mtp->default_library_name,
                                    "spec_qwen35moe_mtp_sidecar.so") == 0 &&
                        qwen35_dflash->dflash_encoded_width == 25600 && qwen35_dflash->dflash_block_size == 8 &&
                        qwen35_dflash->dflash_head_rows == 40960,
                        "Qwen3.8-27B providers retain their independent contracts");
    failures += require(qwen4exp_mtp != nullptr &&
                        std::strcmp(qwen4exp_mtp->target_architecture, "qwen4exp") == 0 &&
                        std::strcmp(qwen4exp_mtp->target_name, "Qwen3.8 Flash Next") == 0 &&
                        qwen4exp_mtp->target_n_embd == 2560 &&
                        qwen4exp_mtp->target_n_embd_out == 10240 &&
                        qwen4exp_mtp->target_n_layer == 48 &&
                        qwen4exp_mtp->target_n_layer_nextn == 0 &&
                        qwen4exp_mtp->target_n_vocab == 248320 &&
                        qwen4exp_mtp->mtp_embedding_width == 10240 &&
                        qwen4exp_mtp->mtp_head_rows == 248320 &&
                        std::strcmp(qwen4exp_mtp->default_library_name,
                                    "spec_qwen4exp_mtp_sidecar.so") == 0,
                        "Flash Next profile carries its separate handoff, vocabulary, and library contract");

    // Optional external fixture paths keep the default host test hermetic while
    // allowing release validation against a real sharded base target and a
    // deliberately incompatible auxiliary-only GGUF.
    if (argc >= 2) {
        std::string fixture_error;
        const auto * fixture = common_spec_sidecar_profile_for_target_file(
                COMMON_SPEC_SIDECAR_KIND_MTP, argv[1], fixture_error);
        failures += require(fixture != nullptr && fixture->name != nullptr &&
                            std::strcmp(fixture->name, "qwen4exp-mtp") == 0,
                            "real Flash Next base target selects only qwen4exp-mtp");
    }
    if (argc >= 3) {
        std::string fixture_error;
        const auto * fixture = common_spec_sidecar_profile_for_target_file(
                COMMON_SPEC_SIDECAR_KIND_MTP, argv[2], fixture_error);
        failures += require(fixture == nullptr,
                            "Flash Next auxiliary-only GGUF is rejected as a base target");
    }

    failures += require(sizeof(spec_sidecar_state) == 24,
                        "sidecar state ABI is a fixed 24-byte record");
    failures += require(SPEC_SIDECAR_STATE_VERSION == 1 &&
                        SPEC_SIDECAR_STATE_KIND_MTP == 1 && SPEC_SIDECAR_STATE_KIND_DFLASH == 2,
                        "sidecar state ABI constants are stable");
    failures += require(offsetof(spec_sidecar_state, pos_min) == 8 &&
                        offsetof(spec_sidecar_state, pos_max) == 12 &&
                        offsetof(spec_sidecar_state, epoch) == 16,
                        "sidecar state ABI field offsets are stable");
    if (argc >= 7 && qwen35moe_mtp != nullptr) {
        set_environment(qwen35moe_mtp->library_env, "/explicit/qwen35moe-sidecar.so");
        set_environment(qwen35moe_mtp->artifact_env, "/explicit/qwen35moe-artifacts");
        std::string fixture_error;
        const auto * fixture = common_spec_sidecar_profile_for_target_file(
                COMMON_SPEC_SIDECAR_KIND_MTP, argv[6], fixture_error);
        failures += require(fixture != nullptr && fixture->name != nullptr &&
                            std::strcmp(fixture->name, "qwen35moe-mtp") == 0,
                            "real Qwen3.6 35B-A3B target selects qwen35moe-mtp with explicit paths");
        unset_environment(qwen35moe_mtp->library_env);
        unset_environment(qwen35moe_mtp->artifact_env);
    }

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
