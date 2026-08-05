#include "ggml-cuda/comm-precision.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static ggml_cuda_allreduce_precision_input eligible_input() {
    ggml_cuda_allreduce_precision_input input;
    input.candidate_enabled  = true;
    input.candidate_topology = true;
    input.all_f32            = true;
    input.all_contiguous     = true;
    input.all_same_shape     = true;
    input.n_backends         = 4;
    input.nelements          = 7168;
    input.ne[0]              = 7168;
    input.ne[1]              = 1;
    input.ne[2]              = 1;
    input.ne[3]              = 1;
    return input;
}

int main() {
    require(ggml_cuda_parse_rdna2_bf16_hidden_option(nullptr) == ggml_cuda_rdna2_bf16_hidden_option::disabled,
            "unset option must disable");
    require(ggml_cuda_parse_rdna2_bf16_hidden_option("0") == ggml_cuda_rdna2_bf16_hidden_option::disabled,
            "exact 0 must disable");
    require(ggml_cuda_parse_rdna2_bf16_hidden_option("1") == ggml_cuda_rdna2_bf16_hidden_option::enabled,
            "exact 1 must enable");
    for (const char * invalid : { "", "01", "true", "-1", " 1", "1 " }) {
        require(ggml_cuda_parse_rdna2_bf16_hidden_option(invalid) == ggml_cuda_rdna2_bf16_hidden_option::invalid,
                "non-canonical option must be invalid");
    }

    const auto disabled = ggml_cuda_rdna2_bf16_hidden_option::disabled;
    const auto enabled = ggml_cuda_rdna2_bf16_hidden_option::enabled;
    const auto invalid = ggml_cuda_rdna2_bf16_hidden_option::invalid;
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(disabled, false, false, nullptr, false) ==
            ggml_cuda_rdna2_bf16_hidden_activation::disabled, "disabled option must bypass platform requirements");
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(invalid, true, true, "nccl", true) ==
            ggml_cuda_rdna2_bf16_hidden_activation::invalid_option, "invalid option must fail activation");
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(enabled, false, true, "nccl", true) ==
            ggml_cuda_rdna2_bf16_hidden_activation::requires_hip, "candidate must require HIP");
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(enabled, true, false, "nccl", true) ==
            ggml_cuda_rdna2_bf16_hidden_activation::requires_nccl, "candidate must require NCCL/RCCL");
    const char * invalid_backends[] = { nullptr, "", "internal", "none", "NCCL" };
    for (const char * backend : invalid_backends) {
        require(ggml_cuda_validate_rdna2_bf16_hidden_activation(enabled, true, true, backend, true) ==
                ggml_cuda_rdna2_bf16_hidden_activation::requires_explicit_nccl,
                "candidate must require exact explicit nccl backend");
    }
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(enabled, true, true, "nccl", false) ==
            ggml_cuda_rdna2_bf16_hidden_activation::requires_four_distinct_rdna2,
            "candidate must require exact topology");
    require(ggml_cuda_validate_rdna2_bf16_hidden_activation(enabled, true, true, "nccl", true) ==
            ggml_cuda_rdna2_bf16_hidden_activation::enabled, "valid candidate activation must enable");

    ggml_cuda_allreduce_topology_device topology[5] = {
        { 0, 0, 1, true }, { 1, 1, 1, true }, { 2, 2, 1, true }, { 3, 3, 1, true }, { 4, 4, 1, true },
    };
    require(ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5),
            "four distinct nonconsecutive-capable RDNA2 entries must pass");
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 3, 5, 5), "three devices must fail topology");
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 5, 5, 5), "five devices must fail topology");
    topology[3].physical_id = 2;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "duplicate physical device must fail");
    topology[3].physical_id = 3; topology[3].logical_id = 2;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "duplicate logical device must fail");
    topology[3].logical_id = 3; topology[3].share_count = 2;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "shared physical device must fail");
    topology[3].share_count = 1; topology[3].rdna2 = false;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "mixed architecture must fail");
    topology[3].rdna2 = true; topology[3].logical_id = 5;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "out-of-range logical id must fail");
    topology[3].logical_id = 3; topology[3].physical_id = 5;
    require(!ggml_cuda_is_four_distinct_rdna2_topology(topology, 4, 5, 5), "out-of-range physical id must fail");

    const uint32_t force_mask = 0x20;
    uint32_t flags[4] = { 0, 0, 0, 0 };
    require(!ggml_cuda_any_allreduce_force_flag(flags, 4, force_mask), "empty rank flags must not force FP32");
    for (size_t rank = 0; rank < 4; ++rank) {
        flags[rank] = force_mask;
        require(ggml_cuda_any_allreduce_force_flag(flags, 4, force_mask), "force flag on any rank must win");
        flags[rank] = 0;
    }

    auto input = eligible_input();
    require(ggml_cuda_is_rdna2_bf16_hidden_shape(input), "exact eligible shape must match");
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::candidate_bf16,
            "exact armed shape must select candidate BF16");

    input.candidate_enabled = false;
    require(ggml_cuda_is_rdna2_bf16_hidden_shape(input), "eligibility must be independent of activation");
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "disabled eligible shape must retain legacy FP32");

    input = eligible_input();
    input.force_fp32 = true;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::forced_fp32,
            "force FP32 must win over candidate eligibility");

    input = eligible_input();
    input.candidate_topology = false;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "topology miss must retain legacy FP32");

    input = eligible_input();
    input.n_backends = 3;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "three backends must miss candidate");
    input.n_backends = 5;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "five backends must miss candidate");

    input = eligible_input();
    input.nelements = 7167;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "7167 elements must miss candidate");
    input = eligible_input();
    input.nelements = 7169;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "7169 elements must miss candidate");

    input = eligible_input();
    input.ne[0] = 3584;
    input.ne[1] = 2;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "reshaped 7168 elements must miss candidate");

    input = eligible_input();
    input.all_f32 = false;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "non-F32 input must miss candidate");
    input = eligible_input();
    input.all_contiguous = false;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "noncontiguous input must miss candidate");
    input = eligible_input();
    input.all_same_shape = false;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_fp32,
            "rank shape mismatch must miss candidate");

    require(ggml_cuda_allreduce_is_small_by_default(2, 32767), "two-rank lower threshold");
    require(!ggml_cuda_allreduce_is_small_by_default(2, 32768), "two-rank threshold boundary");
    require(ggml_cuda_allreduce_is_small_by_default(3, 131071), "three-rank lower threshold");
    require(!ggml_cuda_allreduce_is_small_by_default(3, 131072), "three-rank threshold boundary");
    require(ggml_cuda_allreduce_is_small_by_default(4, 262143), "four-rank lower threshold");
    require(!ggml_cuda_allreduce_is_small_by_default(4, 262144), "four-rank threshold boundary");

    input = eligible_input();
    input.nelements = 262144;
    input.ne[0] = 262144;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::legacy_bf16,
            "large noncandidate tensor must retain legacy BF16");
    input.force_fp32 = true;
    require(ggml_cuda_select_allreduce_precision(input) == ggml_cuda_allreduce_precision::forced_fp32,
            "force FP32 must also win over legacy BF16");

    ggml_cuda_allreduce_audit_counters audit;
    ggml_cuda_audit_record_call(audit, true);
    ggml_cuda_audit_record_call(audit, false);
    ggml_cuda_audit_record_decision(audit, true, true, false, ggml_cuda_allreduce_precision::candidate_bf16);
    ggml_cuda_audit_record_call(audit, false);
    ggml_cuda_audit_record_decision(audit, true, false, false, ggml_cuda_allreduce_precision::legacy_fp32);
    ggml_cuda_audit_record_call(audit, false);
    ggml_cuda_audit_record_decision(audit, true, true, true, ggml_cuda_allreduce_precision::forced_fp32);
    ggml_cuda_audit_record_call(audit, false);
    ggml_cuda_audit_record_decision(audit, false, true, false, ggml_cuda_allreduce_precision::legacy_bf16);
    require(audit.allreduce_calls == 5 && audit.zero_element_calls == 1, "audit call partition mismatch");
    require(audit.candidate_eligible_calls == 3 && audit.candidate_bf16_calls == 1,
            "audit candidate partition mismatch");
    require(audit.candidate_disabled_calls == 1 && audit.force_fp32_calls == 1 &&
            audit.force_candidate_conflicts == 1, "audit force/disabled partition mismatch");
    require(audit.legacy_fp32_calls == 1 && audit.legacy_bf16_calls == 1,
            "audit legacy partition mismatch");
    require(ggml_cuda_audit_nonzero_decision_count(audit) ==
            audit.allreduce_calls - audit.zero_element_calls, "every nonzero call must have one precision decision");

    const std::string audit_path = "test-cuda-allreduce-precision-audit-" +
        std::to_string(reinterpret_cast<uintptr_t>(&audit)) + ".tmp";
    std::remove(audit_path.c_str());
    require(ggml_cuda_append_allreduce_audit_line(audit_path.c_str(), "{\"first\":1}\n"),
            "audit append must create a writable file");
    require(ggml_cuda_append_allreduce_audit_line(audit_path.c_str(), "{\"second\":2}\n"),
            "audit append must preserve an existing file");
    std::ifstream audit_file(audit_path);
    const std::string audit_text((std::istreambuf_iterator<char>(audit_file)), std::istreambuf_iterator<char>());
    require(audit_text == "{\"first\":1}\n{\"second\":2}\n", "audit append contents mismatch");
    audit_file.close();
    std::remove(audit_path.c_str());
#ifndef _WIN32
    require(!ggml_cuda_append_allreduce_audit_line("/dsv4/nonexistent/audit.jsonl", "x\n"),
            "audit open failure must propagate");
    require(!ggml_cuda_append_allreduce_audit_line("/dev/full", "x\n"),
            "audit write/flush failure must propagate");
#endif

    std::puts("PASS: CUDA AllReduce precision selector");
    return 0;
}