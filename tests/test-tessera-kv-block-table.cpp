#include "llama-kv-cache.h"

#include <cassert>

int main() {
    llama_kv_cache::slot_info sinfo;
    sinfo.resize(2);
    sinfo.idxs[0] = { 64, 65, 66, 67, 80, 81, 82, 83, 84 };
    sinfo.idxs[1] = { 96, 97, 98, 99, 100, 101, 102, 103, 104 };

    const auto table = llama_kv_cache::build_tessera_block_table(
        sinfo, 9, 4);
    assert(table.block_size == 4);
    assert(table.n_tokens == 9);
    assert(table.is_direct());
    assert(table.streams.size() == 2);

    // Logical block boundaries are preserved even when physical cells are
    // contiguous, while a physical discontinuity starts a new direct span.
    assert(table.streams[0].size() == 3);
    assert((table.streams[0][0].logical_p0 == 0 &&
            table.streams[0][0].cell_p0 == 64 &&
            table.streams[0][0].n_cells == 4));
    assert((table.streams[0][1].logical_p0 == 4 &&
            table.streams[0][1].cell_p0 == 80 &&
            table.streams[0][1].n_cells == 4));
    assert((table.streams[0][2].logical_p0 == 8 &&
            table.streams[0][2].cell_p0 == 84 &&
            table.streams[0][2].n_cells == 1));
    assert((table.streams[1].size() == 3 &&
            table.streams[1][1].logical_p0 == 4 &&
            table.streams[1][1].cell_p0 == 100 &&
            table.streams[1][1].n_cells == 4));

    // The page map preserves the discontinuity as indirection metadata. It
    // does not copy or repack any K/V values, which is the contract a direct
    // paged Metal attention reader needs.
    const auto page_map = table.make_page_map();
    assert(page_map.size() == 9);
    assert((page_map == std::vector<uint32_t>{ 64, 65, 66, 67, 80, 81, 82, 83, 84 }));

    return 0;
}
