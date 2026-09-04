// SPDX-License-Identifier: MIT
#include "artifact_manifest.h"
#include "dflash/stochastic_distribution.h"
#include "mtp/catchup_alignment.h"
#include "spec_sidecar.h"
#include "spec_sidecar_assets.h"
#include "../common/speculative.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include <algorithm>
#include <cmath>
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
    failures += require(spec_sidecar_mtp::draft_storage_required(16381, 4) == 16385 &&
                        spec_sidecar_mtp::draft_storage_required(32766, 3) == 32769,
                        "MTP draft storage includes lookahead across geometric KV boundaries");

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

    common_params_speculative candidate_gate_params;
    candidate_gate_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    failures += require(!common_speculative_sidecar_candidate(
                            candidate_gate_params, "/definitely/missing/model.gguf", 1) &&
                        !candidate_gate_params.draft.sidecar_prepare_attempted,
                        "disabled sidecar candidate performs no model or cache work");
    unset_environment("LLAMA_SPEC_HIP_SIDECAR");
    unset_environment("LLAMA_SPEC_HIP_WEIGHTS");

    std::vector<int32_t> builtin_ids;
    std::string builtin_error;
    failures += require(common_spec_sidecar_builtin_draft_vocab_ids(builtin_ids, builtin_error) &&
                        builtin_ids.size() == 40960 && builtin_ids.front() == 0 &&
                        builtin_ids.back() == 248076 &&
                        std::is_sorted(builtin_ids.begin(), builtin_ids.end()) &&
                        std::adjacent_find(builtin_ids.begin(), builtin_ids.end()) == builtin_ids.end(),
                        "built-in Apache-2.0 draft vocabulary passes its integrity check");

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
                        "model-specific providers use distinct named profiles");
    failures += require(qwen35_mtp != nullptr && qwen35moe_mtp != nullptr &&
                        qwen35_dflash != nullptr && qwen4exp_mtp != nullptr &&
                        common_spec_sidecar_profile_name_matches(*qwen35_mtp, "Qwen/Qwen3.8-27B") &&
                        common_spec_sidecar_profile_name_matches(*qwen35_mtp, "..") &&
                        common_spec_sidecar_profile_name_matches(*qwen35_dflash, "..") &&
                        !common_spec_sidecar_profile_name_matches(*qwen35_mtp, ".") &&
                        !common_spec_sidecar_profile_name_matches(*qwen35_mtp, "unrelated") &&
                        !common_spec_sidecar_profile_name_matches(*qwen35moe_mtp, "..") &&
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
                        !qwen35moe_mtp->explicit_paths_only &&
                        std::strcmp(qwen35moe_mtp->default_library_name,
                                    "spec_qwen35moe_mtp_sidecar.so") == 0 &&
                        qwen35_dflash->dflash_encoded_width == 25600 && qwen35_dflash->dflash_block_size == 8 &&
                        qwen35_dflash->dflash_head_rows == 40960 &&
                        std::strcmp(qwen4exp_mtp->target_architecture, "qwen4exp") == 0 &&
                        qwen4exp_mtp->target_n_embd == 2560 &&
                        qwen4exp_mtp->target_n_embd_out == 10240 &&
                        qwen4exp_mtp->target_n_layer == 48 &&
                        qwen4exp_mtp->target_n_layer_nextn == 0 &&
                        qwen4exp_mtp->target_n_vocab == 248320 &&
                        qwen4exp_mtp->mtp_embedding_width == 10240 &&
                        qwen4exp_mtp->mtp_head_rows == 248320,
                        "providers retain narrow identity and independent capability contracts");

    if (qwen4exp_mtp != nullptr) {
        if (const char * target = std::getenv("LLAMA_TEST_QWEN4EXP_TARGET")) {
            std::string fixture_error;
            const auto * fixture = common_spec_sidecar_profile_for_target_file(
                    COMMON_SPEC_SIDECAR_KIND_MTP, target, fixture_error);
            failures += require(fixture != nullptr && fixture->name != nullptr &&
                                std::strcmp(fixture->name, "qwen4exp-mtp") == 0,
                                "real Flash Next target selects qwen4exp-mtp");
            set_environment("SPEC_SIDECAR", "1");
            common_params_speculative probe_params;
            probe_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            failures += require(common_speculative_sidecar_candidate(probe_params, target, 8),
                                "Flash Next provider and artifact pass the eight-slot probe");
            unset_environment("SPEC_SIDECAR");
        }
    }

    if (qwen35moe_mtp != nullptr) {
        const char * provider = std::getenv("LLAMA_TEST_QWEN35MOE_PROVIDER");
        const char * bundle = std::getenv("LLAMA_TEST_QWEN35MOE_BUNDLE");
        if (provider != nullptr || bundle != nullptr) {
            std::string provider_error;
            const std::string weights = bundle != nullptr ? bundle : "";
            const std::string ids = weights.empty() ? "" : weights + "/draft_head_ids.bin";
            failures += require(provider != nullptr && bundle != nullptr &&
                            common_spec_sidecar_mtp_probe(provider, weights, ids,
                                qwen35moe_mtp->mtp_embedding_width,
                                qwen35moe_mtp->mtp_head_rows, 1, provider_error),
                        "Qwen35MoE provider exports the current MTP release ABI");
            if (!provider_error.empty()) {
                std::fprintf(stderr, "Qwen35MoE provider probe: %s\n", provider_error.c_str());
            }
        }
    }

    if (qwen35_mtp != nullptr) {
        set_environment(qwen35_mtp->library_env, "/definitely/missing/spec_hip_sidecar.so");
        std::string library;
        std::string library_error;
        failures += require(!common_spec_sidecar_get_library(*qwen35_mtp, library, library_error) &&
                            library_error.find("not readable") != std::string::npos,
                            "automatic preparation rejects a missing provider before writing assets");
        unset_environment(qwen35_mtp->library_env);
    }

    // Optional external fixture paths keep the default host test hermetic while
    // allowing release validation against a real sharded base target and a
    // deliberately incompatible auxiliary-only GGUF.
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

    failures += require(SPEC_SIDECAR_MTP_RELEASE_ABI == 6 &&
                        SPEC_SIDECAR_DFLASH_RELEASE_ABI == 7 &&
                        SPEC_SIDECAR_MTP_DRAFT_TOP_K == 32 &&
                        SPEC_SIDECAR_DFLASH_DRAFT_TOP_K == 16,
                        "sidecar release and stochastic top-k ABI constants match");
    const double u0 = spec_sidecar_stochastic_uniform(UINT64_C(1234), 0);
    failures += require(u0 >= 0.0 && u0 < 1.0 &&
                        u0 == spec_sidecar_stochastic_uniform(UINT64_C(1234), 0) &&
                        u0 != spec_sidecar_stochastic_uniform(UINT64_C(1234), 1),
                        "sidecar proposal RNG is deterministic and bounded");

    float proposal_probs[] = { 2.0f, 1.0f, 1.0f, 0.0f };
    const int proposal_selected = spec_sidecar_dflash::normalize_and_select(
            proposal_probs, 4, 4.0f, 0.8);
    failures += require(proposal_selected == 2 &&
                        std::abs(proposal_probs[0] - 0.5f) < 1e-6f &&
                        std::abs(proposal_probs[1] - 0.25f) < 1e-6f &&
                        std::abs(proposal_probs[2] - 0.25f) < 1e-6f &&
                        proposal_probs[3] == 0.0f,
                        "DFlash normalizes the complete q row before selection");

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
    failures += require(!mtp.load("relative-sidecar.so", "/absolute/artifacts", "/absolute/ids.bin", 5120, 40960, 1, 262144, error) &&
                        error.find("absolute path") != std::string::npos,
                        "MTP loader rejects relative library paths");
    failures += require(!mtp.active(), "MTP loader remains inactive after path rejection");
    error.clear();
    failures += require(!mtp.load("/absolute/sidecar.so", "/absolute/artifacts", "/absolute/ids.bin",
                                5120, 40960, 1, 0, error) &&
                        error.find("context must be positive") != std::string::npos,
                        "MTP loader rejects a non-positive target context");

    common_spec_sidecar_dflash dflash;
    error.clear();
    failures += require(!dflash.load("relative-sidecar.so", "/absolute/artifacts", 25600, 8, 1, 262144, error) &&
                        error.find("absolute path") != std::string::npos,
                        "DFlash loader rejects relative library paths");
    failures += require(!dflash.active(), "DFlash loader remains inactive after path rejection");
    error.clear();
    failures += require(!dflash.load("/absolute/sidecar.so", "/absolute/artifacts",
                                    25600, 8, 1, 0, error) &&
                        error.find("context must be positive") != std::string::npos,
                        "DFlash loader rejects a non-positive target context");

    if (failures == 0) std::puts("artifact_manifest_test: PASS");
    return failures == 0 ? 0 : 1;
}
