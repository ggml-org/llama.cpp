#include "tiered-memory.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    assert(common_tiered_memory_active_fraction("blk.0.attn_q.weight", 256, 8) == 1.0);
    assert(std::abs(common_tiered_memory_active_fraction("blk.0.ffn_up_exps.weight", 256, 8) - 0.03125) < 1e-12);

    const std::vector<common_tiered_memory_item> items = {
        {"dense-a", 100, 1.0},
        {"dense-b", 80, 1.0},
        {"expert", 200, 0.125},
        {"cold", 50, 0.0},
    };

    const auto plan = common_tiered_memory_make_plan(items, {180, 200});

    assert(plan.vram_bytes == 180);
    assert(plan.dram_bytes == 200);
    assert(plan.ssd_bytes == 50);
    assert(plan.active_vram_bytes == 180.0);
    assert(plan.active_dram_bytes == 25.0);
    assert(plan.active_ssd_bytes == 0.0);

    return 0;
}
