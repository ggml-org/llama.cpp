#include "rccl-tuner-policy.h"

#include <cassert>
#include <iostream>

int main() {
    for (size_t ranks = 1; ranks <= 8; ++ranks) {
        const bool expected_auto = ranks == 4;
        assert(ggml_rdna2_rccl_policy_eligible({
            ggml_rdna2_rccl_tune_mode::automatic, ranks, 1, true, true, false,
        }) == expected_auto);
        const bool expected_force = ranks >= 2;
        assert(ggml_rdna2_rccl_policy_eligible({
            ggml_rdna2_rccl_tune_mode::force, ranks, 1, true, true, false,
        }) == expected_force);
    }

    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::off, 4, 1, true, true, false}));
    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::automatic, 4, 2, true, true, false}));
    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::automatic, 4, 1, false, true, false}));
    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::automatic, 4, 1, true, false, false}));
    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::automatic, 4, 1, true, true, true}));
    assert(!ggml_rdna2_rccl_policy_eligible({ggml_rdna2_rccl_tune_mode::force, 9, 1, true, true, false}));

    std::cout << "RDNA2_RCCL_POLICY_TEST_OK\n";
    return 0;
}
