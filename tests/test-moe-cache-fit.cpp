#include "fit.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static common_moe_cache_fit_shape_input shape(
        enum ggml_type type,
        size_t expert_size,
        size_t tensor_bytes,
        size_t scratch_bytes,
        size_t pool_bytes,
        bool cacheable = true) {
    return {type, expert_size, tensor_bytes, scratch_bytes, pool_bytes, cacheable};
}

int main() {
    constexpr size_t MiB = 1024*1024;

    const std::vector<common_moe_cache_fit_shape_input> one_pool = {
        shape(GGML_TYPE_Q4_0, MiB, 128*MiB, MiB/4, MiB),
    };

    {
        const std::vector<common_moe_cache_fit_device_input> devices = {
            {0, 860, 10*(int64_t)MiB, 2*MiB},
            {1, 860, 10*(int64_t)MiB, 2*MiB},
        };
        const common_moe_cache_fit_result result = common_moe_cache_plan_fit(
                devices, one_pool, MiB, 4*MiB, 2);
        expect(result.feasible, "two devices should satisfy the pool floor");
        expect(result.devices.size() == 2, "two physical devices should remain distinct");
        expect(result.minimum_device_bytes == 5*MiB/4, "scratch and pool bytes should be combined");
        expect(result.cache_bytes == 8*MiB, "fixed budget should cap each device");
    }

    {
        const size_t minimum = 5*MiB/4;
        const std::vector<common_moe_cache_fit_device_input> exact = {
            {0, 860, (int64_t)(2*MiB + MiB + minimum), 2*MiB},
        };
        const common_moe_cache_fit_result at_boundary = common_moe_cache_plan_fit(
                exact, one_pool, MiB, 0, 1);
        expect(at_boundary.feasible, "a cache equal to the complete pool floor should be usable");
        expect(at_boundary.cache_bytes == minimum, "the pool boundary should preserve all available bytes");

        std::vector<common_moe_cache_fit_device_input> below = exact;
        below[0].free_bytes--;
        expect(!common_moe_cache_plan_fit(below, one_pool, MiB, 0, 1).feasible,
                "one byte below the pool floor should be rejected");
    }

    {
        const std::vector<common_moe_cache_fit_device_input> devices = {
            {0, 860, 12*(int64_t)MiB, 0},
        };
        const common_moe_cache_fit_result result = common_moe_cache_plan_fit(
                devices, one_pool, 0, 0, 1, 8*MiB);
        expect(result.feasible, "an automatic slab floor should be usable at its boundary");
        expect(result.minimum_device_bytes == 8*MiB + MiB/4,
                "the automatic slab floor should replace a smaller pool inventory");

        std::vector<common_moe_cache_fit_device_input> below = devices;
        below[0].free_bytes = (int64_t)result.minimum_device_bytes - 1;
        expect(!common_moe_cache_plan_fit(below, one_pool, 0, 0, 1, 8*MiB).feasible,
                "one byte below the automatic slab floor should be rejected");
    }

    {
        const std::vector<common_moe_cache_fit_device_input> aliases = {
            {7, 860, 10*(int64_t)MiB, 2*MiB},
            {7, 750, 10*(int64_t)MiB, 3*MiB},
        };
        const common_moe_cache_fit_result result = common_moe_cache_plan_fit(
                aliases, one_pool, MiB, 0, 1);
        expect(result.feasible, "one physical device represented by aliases should be usable once");
        expect(result.devices.size() == 1, "logical aliases should be deduplicated");
        expect(result.devices[0].cache_bytes == 4*MiB, "aliases must not duplicate physical free memory");
        expect(result.devices[0].compute_capability == 750, "alias policy should use the conservative capability");
        expect(!common_moe_cache_plan_fit(aliases, one_pool, MiB, 0, 2).feasible,
                "aliases must not satisfy a two-device hardware policy");
    }

    {
        const std::vector<common_moe_cache_fit_shape_input> mixed = {
            shape(GGML_TYPE_Q4_0, MiB, 100*MiB, MiB, 2*MiB),
            shape(GGML_TYPE_Q4_0, MiB, 100*MiB, MiB/2, 3*MiB),
            shape(GGML_TYPE_Q8_0, 2*MiB, 100*MiB, MiB/2, 4*MiB),
        };
        const std::vector<common_moe_cache_fit_device_input> devices = {
            {0, 860, 10*(int64_t)MiB, MiB},
        };
        const common_moe_cache_fit_result result = common_moe_cache_plan_fit(devices, mixed, MiB, 0, 1);
        expect(result.feasible, "mixed complete pool shapes should be usable");
        expect(result.minimum_device_bytes == 8*MiB,
                "duplicate pool shapes should use their maximum floor instead of summing");
        expect(result.expert_bytes == 300*MiB, "all routed expert bytes should be inventoried");

        std::vector<common_moe_cache_fit_shape_input> unsupported = mixed;
        unsupported[2].cacheable = false;
        expect(!common_moe_cache_plan_fit(devices, unsupported, MiB, 0, 1).feasible,
                "partially unsupported routed weights should reject global cache placement");
    }

    {
        const std::vector<common_moe_cache_fit_device_input> devices = {
            {0, 860, 128*(int64_t)MiB, 0},
        };
        const std::vector<common_moe_cache_fit_shape_input> aggregated = {
            shape(GGML_TYPE_Q4_0, MiB, 16*MiB, MiB/4, 64*MiB),
            shape(GGML_TYPE_Q4_0, MiB, 49*MiB, MiB/2, 64*MiB),
        };
        const common_moe_cache_fit_result result = common_moe_cache_plan_fit(
                devices, aggregated, 0, 0, 1);
        expect(result.feasible, "small tensors should satisfy a shared pool floor in aggregate");
        expect(result.minimum_device_bytes == 64*MiB + MiB/2,
                "an aggregate pool should retain its largest scratch requirement");

        const std::vector<common_moe_cache_fit_shape_input> underfilled = {
            aggregated.front(),
        };
        expect(!common_moe_cache_plan_fit(devices, underfilled, 0, 0, 1).feasible,
                "a shape with too few aggregate entries should remain ineligible");
    }

    {
        const std::vector<common_moe_cache_fit_device_input> devices = {
            {0, 860, 16*(int64_t)MiB, MiB},
        };
        const std::vector<common_moe_cache_fit_shape_input> tensor_overflow = {
            shape(GGML_TYPE_Q4_0, MiB, std::numeric_limits<size_t>::max(), 0, MiB),
            shape(GGML_TYPE_Q4_0, MiB, 1, 0, MiB),
        };
        expect(!common_moe_cache_plan_fit(devices, tensor_overflow, MiB, 0, 1).feasible,
                "expert inventory overflow should fail closed");

        const std::vector<common_moe_cache_fit_shape_input> pool_overflow = {
            shape(GGML_TYPE_Q4_0, MiB, MiB, std::numeric_limits<size_t>::max(), 1),
        };
        expect(!common_moe_cache_plan_fit(devices, pool_overflow, MiB, 0, 1).feasible,
                "pool inventory overflow should fail closed");

        const std::vector<common_moe_cache_fit_device_input> used_overflow = {
            {0, 860, std::numeric_limits<int64_t>::max(), (size_t)std::numeric_limits<int64_t>::max()},
            {0, 860, std::numeric_limits<int64_t>::max(), 1},
        };
        expect(!common_moe_cache_plan_fit(used_overflow, one_pool, MiB, 0, 1).feasible,
                "aliased projected usage overflow should fail closed");
    }

    expect(!common_moe_cache_plan_fit({}, one_pool, MiB, 0, 1).feasible,
            "an empty eligible device set should be rejected");

    if (failures != 0) {
        std::fprintf(stderr, "%d MoE cache fit tests failed\n", failures);
        return 1;
    }
    std::printf("MoE cache fit tests passed\n");
    return 0;
}
