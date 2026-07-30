#include "kv_compaction_state.h"

#include <cassert>
#include <vector>

int main() {
    tessera::kv_compaction_metadata state = {
        { 0, 1, 2, 3, 4, 5 },
        { 0, 0, 0, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1 },
        { 0, 0, 0, 0, 0, 0 },
    };
    tessera::kv_remove(state, 0, 1, 2);
    assert((state.valid == std::vector<uint8_t>{ 1, 0, 1, 1, 1, 1 }));
    tessera::kv_shift(state, 1, 3, -1, -2);
    assert((state.positions == std::vector<int32_t>{ 0, 1, 2, 1, 2, 3 }));
    assert((state.pending_shifts == std::vector<int32_t>{ 0, 0, 0, -2, -2, -2 }));
    tessera::kv_divide(state, 1, 1, 3, 2);
    assert((state.positions == std::vector<int32_t>{ 0, 1, 1, 0, 1, 3 }));
    assert((state.pending_shifts == std::vector<int32_t>{ 0, 0, -1, -3, -3, -2 }));
    const auto order = tessera::kv_defragment_order(state);
    assert((order == std::vector<size_t>{ 0, 2, 3, 4, 5, 1 }));

    tessera::kv_compaction_metadata merge_state = {
        { 0, 1, 2, 3, 4, 5, 6, 7 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 1, 1, 1, 1, 1, 1, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const auto merge = tessera::kv_make_plan(merge_state, 4, 1, 1, true);
    assert((merge.destination_positions == std::vector<int32_t>{ 0, 3, 6, 7, -1, -1, -1, -1 }));
    assert((merge.source_to_destination == std::vector<int32_t>{ 0, 1, 1, 1, 2, 2, 2, 3 }));
    assert((merge.rope_deltas == std::vector<int32_t>{ 0, 2, 1, 0, 2, 1, 0, 0 }));

    const auto evict = tessera::kv_make_plan(merge_state, 5, 2, 2, false);
    assert((evict.source_to_destination == std::vector<int32_t>{ 0, 1, -1, -1, -1, 2, 3, 4 }));
    return 0;
}
