#include "../ggml/src/ggml-cuda/vmm-probe.h"
#include "testing.h"

int main() {
    testing t;
    t.test("supported_vmm_commits_driver_granularity", [](testing & t) {
        size_t granularity = 17;
        size_t queries = 0;
        const bool available = ggml_cuda_probe_vmm(
            [&](int & supported) { ++queries; supported = 1; return true; },
            [&](size_t & bytes) { ++queries; bytes = 65536; return true; }, granularity);
        t.assert_true(available);
        t.assert_equal(size_t(2), queries);
        t.assert_equal(size_t(65536), granularity);
    });
    t.test("absent_or_failed_optional_attribute_uses_legacy_pool", [](testing & t) {
        for (bool success : {false, true}) for (int reported : {0, 1}) {
            if (success && reported) continue;
            size_t granularity = 17;
            size_t queries = 0;
            const bool available = ggml_cuda_probe_vmm(
                [&](int & supported) { ++queries; supported = reported; return success; },
                [&](size_t &) { ++queries; return true; }, granularity);
            t.assert_true(!available);
            t.assert_equal(size_t(1), queries);
            t.assert_equal(size_t(17), granularity);
        }
    });
    t.test("failed_or_zero_granularity_does_not_enable_vmm", [](testing & t) {
        for (bool success : {false, true}) for (size_t reported : {size_t(0), size_t(65536)}) {
            if (success && reported) continue;
            size_t granularity = 17;
            const bool available = ggml_cuda_probe_vmm(
                [](int & supported) { supported = 1; return true; },
                [&](size_t & bytes) { bytes = reported; return success; }, granularity);
            t.assert_true(!available);
            t.assert_equal(size_t(17), granularity);
        }
    });
    return t.summary();
}
