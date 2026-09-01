#include "testing.h"

#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

static uint64_t ref_popcount(uint64_t x) {
    uint64_t c = 0;
    while (x) {
        x &= x - 1;
        ++c;
    }
    return c;
}

static int ref_countr_zero(uint64_t x) {
    int c = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        ++c;
    }
    return c;
}

static int ref_countl_zero(uint64_t x) {
    int c = 0;
    while ((x & (1ull << 63)) == 0) {
        x <<= 1;
        ++c;
    }
    return c;
}

static void test_bitops(testing & t) {
    t.test("popcount64", [&](testing & t) {
        t.assert_equal(0, llama_bits::popcount64(0));
        t.assert_equal(1, llama_bits::popcount64(1));
        t.assert_equal(2, llama_bits::popcount64(0x8000000000000001ull));
        t.assert_equal(8, llama_bits::popcount64(0xFF));
        t.assert_equal(64, llama_bits::popcount64(~0ull));

        for (int i = 0; i < 64; ++i) {
            t.assert_equal((int) ref_popcount(1ull << i), llama_bits::popcount64(1ull << i));
        }

        std::mt19937 rng(1);
        for (int i = 0; i < 1000; ++i) {
            const uint64_t v = ((uint64_t) rng() << 32) ^ rng();
            t.assert_equal((int) ref_popcount(v), llama_bits::popcount64(v));
        }
    });

    t.test("countr_zero64", [&](testing & t) {
        t.assert_equal(0, llama_bits::countr_zero64(1));
        t.assert_equal(1, llama_bits::countr_zero64(2));
        t.assert_equal(3, llama_bits::countr_zero64(0x8));
        t.assert_equal(63, llama_bits::countr_zero64(0x8000000000000000ull));
        t.assert_equal(2, llama_bits::countr_zero64(0x4));

        for (int i = 0; i < 64; ++i) {
            t.assert_equal(ref_countr_zero(1ull << i), llama_bits::countr_zero64(1ull << i));
        }

        std::mt19937 rng(2);
        for (int i = 0; i < 1000; ++i) {
            uint64_t v = ((uint64_t) rng() << 32) ^ rng();
            if (v == 0) {
                continue;
            }
            t.assert_equal(ref_countr_zero(v), llama_bits::countr_zero64(v));
        }
    });

    t.test("countl_zero64", [&](testing & t) {
        t.assert_equal(63, llama_bits::countl_zero64(1));
        t.assert_equal(62, llama_bits::countl_zero64(2));
        t.assert_equal(60, llama_bits::countl_zero64(0x8));
        t.assert_equal(0, llama_bits::countl_zero64(0x8000000000000000ull));

        for (int i = 0; i < 64; ++i) {
            t.assert_equal(ref_countl_zero(1ull << i), llama_bits::countl_zero64(1ull << i));
        }

        std::mt19937 rng(3);
        for (int i = 0; i < 1000; ++i) {
            uint64_t v = ((uint64_t) rng() << 32) ^ rng();
            if (v == 0) {
                continue;
            }
            t.assert_equal(ref_countl_zero(v), llama_bits::countl_zero64(v));
        }
    });
}

static void test_ext(testing & t) {
    t.test("is_2d_gt", [&](testing & t) {
        llama_kv_cell_ext a{/*x=*/1, /*y=*/2};

        // equal positions are not greater
        t.assert_true(!a.is_2d_gt(1, 2));
        // equal y: larger x is greater, smaller x is not
        t.assert_true(!a.is_2d_gt(3, 2));
        t.assert_true(a.is_2d_gt(0, 2));
        // y dominates x
        t.assert_true(!a.is_2d_gt(1, 3));
        t.assert_true(a.is_2d_gt(1, 1));
        t.assert_true(!a.is_2d_gt(0, 3));
        t.assert_true(!a.is_2d_gt(5, 3));
    });

    t.test("reset", [&](testing & t) {
        llama_kv_cell_ext e{/*x=*/7, /*y=*/9};
        e.reset();
        t.assert_equal((llama_pos) 0, e.x);
        t.assert_equal((llama_pos) 0, e.y);
    });
}

static void test_basic(testing & t) {
    t.test("resize_and_reset", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);

        t.assert_equal(10u, cells.size());
        for (uint32_t i = 0; i < 10; ++i) {
            t.assert_true(cells.is_empty(i));
        }
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(0u, cells.used_min());
        t.assert_equal(0u, cells.used_max_p1());
        t.assert_true(!cells.get_has_shift());
    });

    t.test("resize_resets_existing", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(10);

        cells.pos_set(3, 7);
        cells.pos_add(3, 2); // also sets has_shift
        t.assert_equal(1u, cells.get_used());
        t.assert_true(cells.get_has_shift());

        cells.resize(10);
        t.assert_equal(10u, cells.size());
        t.assert_true(cells.is_empty(3));
        t.assert_equal(0u, cells.get_used());
        t.assert_true(!cells.get_has_shift());
    });

    t.test("pos_set_get", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(3, 42);
        t.assert_equal(42, cells.pos_get(3));
        t.assert_true(!cells.is_empty(3));
        t.assert_equal(1u, cells.get_used());
        t.assert_equal(3u, cells.used_min());
        t.assert_equal(4u, cells.used_max_p1());
    });

    t.test("used_tracking_noncontiguous", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(130);

        cells.pos_set(63, 1);
        cells.pos_set(64, 2);
        cells.pos_set(128, 3);

        t.assert_equal(3u, cells.get_used());
        t.assert_equal(63u, cells.used_min());
        t.assert_equal(129u, cells.used_max_p1());

        cells.rm(63);
        t.assert_equal(2u, cells.get_used());
        t.assert_equal(64u, cells.used_min());
        t.assert_equal(129u, cells.used_max_p1());

        cells.rm(128);
        t.assert_equal(1u, cells.get_used());
        t.assert_equal(64u, cells.used_min());
        t.assert_equal(65u, cells.used_max_p1());

        cells.rm(64);
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(0u, cells.used_min());
        t.assert_equal(0u, cells.used_max_p1());
    });

    t.test("pos_in", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);
        cells.pos_set(1, 5);

        t.assert_true(cells.pos_in(1, 5, 6));
        t.assert_true(cells.pos_in(1, 0, 6));
        t.assert_true(!cells.pos_in(1, 6, 7));
        t.assert_true(!cells.pos_in(1, 0, 5));

        // empty cells never match ranges with p0 >= 0 (callers clamp p0)
        t.assert_true(!cells.pos_in(0, 0, 1000));
    });
}

static void test_seq(testing & t) {
    t.test("seq_add_has_count", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(1, 5);
        t.assert_true(!cells.seq_has(1, 0));
        t.assert_equal(0, cells.seq_count(1));

        cells.seq_add(1, 0);
        cells.seq_add(1, 2);
        cells.seq_add(1, 5);

        t.assert_true(cells.seq_has(1, 0));
        t.assert_true(cells.seq_has(1, 2));
        t.assert_true(cells.seq_has(1, 5));
        t.assert_true(!cells.seq_has(1, 1));
        t.assert_equal(3, cells.seq_count(1));

        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(5, cells.seq_pos_max(0));
        t.assert_equal(5, cells.seq_pos_min(2));
        t.assert_equal(5, cells.seq_pos_max(5));
    });

    t.test("seq_get", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 1);
        cells.seq_add(0, 7);
        t.assert_equal(7, cells.seq_get(0));

        cells.seq_add(0, 3);
        cells.seq_rm(0, 3);
        t.assert_equal(7, cells.seq_get(0));
    });

    t.test("seq_rm_partial", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        cells.seq_add(0, 1);

        // removing one of two seqs keeps the cell
        t.assert_true(!cells.seq_rm(0, 0));
        t.assert_true(!cells.is_empty(0));
        t.assert_equal(1, cells.seq_count(0));
        t.assert_true(!cells.seq_has(0, 0));
        t.assert_true(cells.seq_has(0, 1));
        t.assert_equal(1u, cells.get_used());

        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
        t.assert_equal(5, cells.seq_pos_min(1));
        t.assert_equal(5, cells.seq_pos_max(1));
    });

    t.test("seq_rm_full", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(2, 9);
        cells.seq_add(2, 3);

        t.assert_true(cells.seq_rm(2, 3));
        t.assert_true(cells.is_empty(2));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(0u, cells.used_min());
        t.assert_equal(0u, cells.used_max_p1());
        t.assert_equal(-1, cells.seq_pos_min(3));
        t.assert_equal(-1, cells.seq_pos_max(3));
    });

    t.test("seq_keep_only", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 1);
        cells.seq_add(0, 0);
        cells.seq_add(0, 1);
        cells.seq_add(0, 2);

        // keeping seq 1 drops the other seqs but keeps the cell
        t.assert_true(!cells.seq_keep(0, 1));
        t.assert_true(!cells.is_empty(0));
        t.assert_equal(1, cells.seq_count(0));
        t.assert_equal(1, cells.seq_get(0));

        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_min(2));
        t.assert_equal(1, cells.seq_pos_min(1));
        t.assert_equal(1, cells.seq_pos_max(1));
    });

    t.test("seq_keep_absent_empties", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(1, 3);
        cells.seq_add(1, 0);
        cells.seq_add(1, 2);

        // the kept seq is not present: the whole cell is cleared
        t.assert_true(cells.seq_keep(1, 5));
        t.assert_true(cells.is_empty(1));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_min(2));
    });

    t.test("seq_keep_empty_noop", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        t.assert_true(!cells.seq_keep(0, 3));
        t.assert_true(cells.is_empty(0));
        t.assert_equal(0u, cells.get_used());
    });

    t.test("seqS_add", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 7);

        llama_seq_id seqs[3] = {0, 3, 7};
        cells.seqS_add(0, 3, seqs);

        t.assert_equal(3, cells.seq_count(0));
        for (auto s : seqs) {
            t.assert_true(cells.seq_has(0, s));
            t.assert_equal(7, cells.seq_pos_min(s));
            t.assert_equal(7, cells.seq_pos_max(s));
        }
    });
}

static void test_seq_pos(testing & t) {
    t.test("min_max_across_cells", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(0, 10);
        cells.seq_add(0, 0);
        cells.pos_set(3, 5);
        cells.seq_add(3, 0);
        cells.pos_set(7, 20);
        cells.seq_add(7, 0);

        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(20, cells.seq_pos_max(0));

        cells.rm(3);
        t.assert_equal(10, cells.seq_pos_min(0));
        t.assert_equal(20, cells.seq_pos_max(0));

        cells.rm(7);
        t.assert_equal(10, cells.seq_pos_min(0));
        t.assert_equal(10, cells.seq_pos_max(0));

        cells.rm(0);
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
    });

    t.test("duplicate_positions", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        cells.pos_set(1, 5);
        cells.seq_add(1, 0);

        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(5, cells.seq_pos_max(0));

        // removing one of two cells at the same position keeps the position
        cells.rm(0);
        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(5, cells.seq_pos_max(0));

        cells.rm(1);
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
    });

    t.test("insert_before_base", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 10);
        cells.seq_add(0, 0);

        // a new cell with a smaller position exercises the prepend path
        cells.pos_set(1, 5);
        cells.seq_add(1, 0);

        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(10, cells.seq_pos_max(0));

        cells.rm(1);
        t.assert_equal(10, cells.seq_pos_min(0));
        t.assert_equal(10, cells.seq_pos_max(0));

        cells.rm(0);
        t.assert_equal(-1, cells.seq_pos_min(0));
    });

    t.test("compact", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(0, 10);
        cells.seq_add(0, 0);
        cells.pos_set(1, 11);
        cells.seq_add(1, 0);
        cells.pos_set(2, 12);
        cells.seq_add(2, 0);

        cells.rm(0);
        t.assert_equal(11, cells.seq_pos_min(0));
        cells.compact(0);
        t.assert_equal(11, cells.seq_pos_min(0));
        t.assert_equal(12, cells.seq_pos_max(0));

        // tracking still works after compaction: insert before the new base
        cells.pos_set(0, 9);
        cells.seq_add(0, 0);
        t.assert_equal(9, cells.seq_pos_min(0));
        t.assert_equal(12, cells.seq_pos_max(0));

        // and after the tail
        cells.pos_set(3, 13);
        cells.seq_add(3, 0);
        t.assert_equal(9, cells.seq_pos_min(0));
        t.assert_equal(13, cells.seq_pos_max(0));
    });

    t.test("absent_seq", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 1);
        cells.seq_add(0, 0);

        t.assert_equal(-1, cells.seq_pos_min(1));
        t.assert_equal(-1, cells.seq_pos_max(1));
    });
}

static void test_shift(testing & t) {
    t.test("pos_add", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 10);
        cells.seq_add(0, 0);
        cells.pos_set(1, 20);
        cells.seq_add(1, 0);

        t.assert_true(!cells.get_has_shift());

        t.assert_true(!cells.pos_add(0, 5));
        t.assert_equal(15, cells.pos_get(0));
        t.assert_equal(5, cells.get_shift(0));
        t.assert_true(cells.get_has_shift());
        t.assert_equal(15, cells.seq_pos_min(0));
        t.assert_equal(20, cells.seq_pos_max(0));

        // untouched cell keeps a zero shift
        t.assert_equal(0, cells.get_shift(1));

        // shifts accumulate
        t.assert_true(!cells.pos_add(0, -3));
        t.assert_equal(12, cells.pos_get(0));
        t.assert_equal(2, cells.get_shift(0));
        t.assert_equal(12, cells.seq_pos_min(0));
    });

    t.test("pos_add_removes", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 3);
        cells.seq_add(0, 0);
        cells.pos_set(1, 10);
        cells.seq_add(1, 1);

        // 3 - 4 < 0 -> the cell is removed
        t.assert_true(cells.pos_add(0, -4));
        t.assert_true(cells.is_empty(0));
        t.assert_equal(1u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
        t.assert_true(cells.get_has_shift());

        // the other cell is unaffected
        t.assert_equal(10, cells.seq_pos_min(1));
        t.assert_equal(10, cells.seq_pos_max(1));
    });

    t.test("pos_div", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 100);
        cells.seq_add(0, 0);

        cells.pos_div(0, 4);
        t.assert_equal(25, cells.pos_get(0));
        t.assert_equal(75, cells.get_shift(0)); // 100 - 25
        t.assert_true(cells.get_has_shift());
        t.assert_equal(25, cells.seq_pos_min(0));
        t.assert_equal(25, cells.seq_pos_max(0));

        // negative positions truncate toward zero
        cells.pos_set(1, -7);
        cells.seq_add(1, 0);
        cells.pos_div(1, 2);
        t.assert_equal(-3, cells.pos_get(1));
        t.assert_equal(-4, cells.get_shift(1)); // -7 - (-3)
        t.assert_equal(-3, cells.seq_pos_min(0));
        t.assert_equal(25, cells.seq_pos_max(0));
    });

    t.test("reset_shift", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 10);
        cells.seq_add(0, 0);
        cells.pos_add(0, 5);  // pos 15, shift 5
        cells.pos_div(0, 3);  // pos 5, shift 5 + (15 - 5) = 15

        t.assert_true(cells.get_has_shift());
        t.assert_equal(15, cells.get_shift(0));

        cells.reset_shift();
        t.assert_true(!cells.get_has_shift());
        t.assert_equal(0, cells.get_shift(0));

        // positions stay shifted
        t.assert_equal(5, cells.pos_get(0));
        t.assert_equal(5, cells.seq_pos_min(0));
    });
}

static void test_remove(testing & t) {
    t.test("rm", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(2, 7);
        cells.seq_add(2, 0);
        cells.ext_set(2, {/*x=*/3, /*y=*/4});

        cells.rm(2);
        t.assert_true(cells.is_empty(2));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
    });

    t.test("rm_single", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(1, 9);
        cells.seq_add(1, 2);

        cells.rm_single(1, 2);
        t.assert_true(cells.is_empty(1));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(2));
        t.assert_equal(-1, cells.seq_pos_max(2));
    });
}

static void test_save_restore(testing & t) {
    t.test("cp_range_roundtrip", [&](testing & t) {
        const uint32_t n = 16;
        llama_kv_cells cells;
        cells.resize(n);

        cells.pos_set(2, 10);
        cells.seq_add(2, 0);
        cells.seq_add(2, 1);
        cells.ext_set(2, {/*x=*/5, /*y=*/6});

        cells.pos_set(3, 11);
        cells.seq_add(3, 1);

        cells.pos_set(4, 12);
        cells.seq_add(4, 2);

        cells.pos_set(5, 13);
        cells.seq_add(5, 0);

        // save the state of cells [2, 2 + 4)
        const llama_kv_cells saved = cells.cp(2, 4);

        // the copy carries pos/ext/seq
        t.assert_equal(4u, saved.size());
        t.assert_equal(10, saved.pos_get(0));
        t.assert_equal(11, saved.pos_get(1));
        t.assert_equal(12, saved.pos_get(2));
        t.assert_equal(13, saved.pos_get(3));
        t.assert_equal(5, saved.ext_get(0).x);
        t.assert_equal(6, saved.ext_get(0).y);
        t.assert_true(saved.seq_has(0, 0));
        t.assert_true(saved.seq_has(0, 1));
        t.assert_equal(2, saved.seq_count(0));
        t.assert_equal(1, saved.seq_count(1));
        t.assert_equal(1, saved.seq_count(3));

        // wipe the original cells
        cells.rm(2);
        cells.rm(3);
        cells.rm(4);
        cells.rm(5);
        t.assert_equal(0u, cells.get_used());

        // restore
        const std::vector<uint32_t> idxs = {2, 3, 4, 5};
        cells.set(idxs, saved);

        t.assert_equal(4u, cells.get_used());
        t.assert_equal(2u, cells.used_min());
        t.assert_equal(6u, cells.used_max_p1());
        for (uint32_t j = 0; j < 4; ++j) {
            t.assert_equal(saved.pos_get(j), cells.pos_get(2 + j));
            t.assert_equal(saved.seq_has(j, 0), cells.seq_has(2 + j, 0));
            t.assert_equal(saved.seq_has(j, 1), cells.seq_has(2 + j, 1));
            t.assert_equal(saved.seq_has(j, 2), cells.seq_has(2 + j, 2));
            t.assert_equal(saved.seq_count(j), cells.seq_count(2 + j));
        }
        t.assert_equal(5, cells.ext_get(2).x);
        t.assert_equal(6, cells.ext_get(2).y);

        // sequence position tracking is rebuilt
        t.assert_equal(10, cells.seq_pos_min(0));
        t.assert_equal(13, cells.seq_pos_max(0));
        t.assert_equal(10, cells.seq_pos_min(1));
        t.assert_equal(11, cells.seq_pos_max(1));
        t.assert_equal(12, cells.seq_pos_min(2));
        t.assert_equal(12, cells.seq_pos_max(2));
        t.assert_equal(-1, cells.seq_pos_min(3));
    });

    t.test("cp_idxs_roundtrip", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(1, 5);
        cells.seq_add(1, 0);
        cells.pos_set(6, 9);
        cells.seq_add(6, 1);

        const std::vector<uint32_t> idxs = {1, 6};
        const llama_kv_cells saved = cells.cp(idxs);

        t.assert_equal(2u, saved.size());
        t.assert_equal(5, saved.pos_get(0));
        t.assert_equal(9, saved.pos_get(1));
        t.assert_true(saved.seq_has(0, 0));
        t.assert_true(saved.seq_has(1, 1));

        // restore into different cells (remap)
        cells.rm(1);
        cells.rm(6);
        const std::vector<uint32_t> dst = {3, 5};
        cells.set(dst, saved);

        t.assert_equal(2u, cells.get_used());
        t.assert_equal(5, cells.pos_get(3));
        t.assert_equal(9, cells.pos_get(5));
        t.assert_true(cells.seq_has(3, 0));
        t.assert_true(cells.seq_has(5, 1));
        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(9, cells.seq_pos_min(1));
    });

    t.test("set_replaces_existing", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(0, 1);
        cells.seq_add(0, 0);

        // "other" describes a single empty cell
        llama_kv_cells other;
        other.resize(1);

        const std::vector<uint32_t> idxs = {0};
        cells.set(idxs, other);

        t.assert_true(cells.is_empty(0));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(0));
    });

    t.test("set_fast_path_identical", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(4);

        cells.pos_set(1, 5);
        cells.seq_add(1, 0);

        // copy a cell with an identical state: set takes the fast path
        const llama_kv_cells saved = cells.cp(1, 1);

        const std::vector<uint32_t> idxs = {1};
        cells.set(idxs, saved);

        t.assert_equal(1u, cells.get_used());
        t.assert_equal(5, cells.pos_get(1));
        t.assert_true(cells.seq_has(1, 0));
        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(5, cells.seq_pos_max(0));
    });
}

static void test_next_head(testing & t) {
    t.test("removes_range_of_seq", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        cells.pos_set(1, 6);
        cells.seq_add(1, 0);
        cells.seq_add(1, 1); // shared cell
        cells.pos_set(2, 7);
        cells.seq_add(2, 1);
        cells.pos_set(3, 8);
        cells.seq_add(3, 0); // outside the range

        t.assert_equal(0u, cells.nextHead(0, 5, 8));

        t.assert_true(cells.is_empty(0));
        t.assert_equal(1, cells.seq_count(1)); // shared cell kept seq 1
        t.assert_true(cells.seq_has(2, 1));
        t.assert_true(cells.seq_has(3, 0));
        t.assert_equal(3u, cells.get_used());

        t.assert_equal(8, cells.seq_pos_min(0));
        t.assert_equal(8, cells.seq_pos_max(0));
        t.assert_equal(6, cells.seq_pos_min(1));
        t.assert_equal(7, cells.seq_pos_max(1));
    });

    t.test("returns_first_removed", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(2, 3);
        cells.seq_add(2, 0);
        cells.pos_set(4, 3);
        cells.seq_add(4, 0);

        // both cells are freed; the lowest index is returned
        t.assert_equal(2u, cells.nextHead(0, 0, 100));

        t.assert_true(cells.is_empty(2));
        t.assert_true(cells.is_empty(4));
        t.assert_equal(0u, cells.get_used());
        t.assert_equal(-1, cells.seq_pos_min(0));
        t.assert_equal(-1, cells.seq_pos_max(0));
    });

    t.test("no_match", [&](testing & t) {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        cells.pos_set(1, 7);
        cells.seq_add(1, 1);

        // ranges do not cover any seq-0 cell
        t.assert_equal(cells.size(), cells.nextHead(0, 0, 5));
        t.assert_equal(cells.size(), cells.nextHead(0, 8, 100));
        // no cell has seq 2
        t.assert_equal(cells.size(), cells.nextHead(2, 0, 100));
        t.assert_equal(2u, cells.get_used());
        t.assert_equal(5, cells.seq_pos_min(0));
        t.assert_equal(7, cells.seq_pos_min(1));
    });
}

// reference model for the randomized test: same operations, naive O(n) state
struct cells_ref {
    struct cell_t {
        llama_pos pos = -1;
        std::set<llama_seq_id> seqs;
        llama_pos shift = 0;
        llama_kv_cell_ext ext = {};
    };

    std::vector<cell_t> cells;
    bool has_shift = false;

    void resize(uint32_t n) {
        cells.assign(n, cell_t{});
        has_shift = false;
    }

    uint32_t size() const {
        return (uint32_t) cells.size();
    }

    uint32_t get_used() const {
        uint32_t c = 0;
        for (const auto & cl : cells) {
            if (cl.pos != -1) {
                ++c;
            }
        }
        return c;
    }

    uint32_t used_min() const {
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells[i].pos != -1) {
                return i;
            }
        }
        return 0;
    }

    uint32_t used_max_p1() const {
        for (uint32_t i = (uint32_t) cells.size(); i-- > 0;) {
            if (cells[i].pos != -1) {
                return i + 1;
            }
        }
        return 0;
    }

    llama_pos seq_pos_min(llama_seq_id s) const {
        llama_pos res = -1;
        for (const auto & cl : cells) {
            if (cl.pos != -1 && cl.seqs.count(s)) {
                res = res == -1 ? cl.pos : std::min(res, cl.pos);
            }
        }
        return res;
    }

    llama_pos seq_pos_max(llama_seq_id s) const {
        llama_pos res = -1;
        for (const auto & cl : cells) {
            if (cl.pos != -1 && cl.seqs.count(s)) {
                res = std::max(res, cl.pos);
            }
        }
        return res;
    }
};

static void test_random(testing & t) {
    t.test("ops_vs_reference_model", [&](testing & t) {
        std::mt19937 rng(1234);

        const uint32_t n     = 130;
        const uint32_t n_seq = 8;

        llama_kv_cells cells;
        cells.resize(n);

        cells_ref ref;
        ref.resize(n);

        auto check = [&](const std::string & msg) {
            t.assert_equal(ref.size(), cells.size());

            for (uint32_t i = 0; i < n; ++i) {
                const auto & cl = ref.cells[i];

                t.assert_equal(cl.pos == -1, cells.is_empty(i));

                if (cl.pos != -1) {
                    t.assert_equal(cl.pos, cells.pos_get(i));
                    t.assert_equal(cl.shift, cells.get_shift(i));
                    t.assert_equal(cl.ext.x, cells.ext_get(i).x);
                    t.assert_equal(cl.ext.y, cells.ext_get(i).y);
                    t.assert_equal((int) cl.seqs.size(), cells.seq_count(i));
                    if (cl.seqs.size() == 1) {
                        t.assert_equal((llama_seq_id) *cl.seqs.begin(), cells.seq_get(i));
                    }
                }

                for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                    t.assert_equal(cl.seqs.count(s) > 0, cells.seq_has(i, s));
                }
            }

            t.assert_equal(ref.get_used(), cells.get_used());
            t.assert_equal(ref.used_min(), cells.used_min());
            t.assert_equal(ref.used_max_p1(), cells.used_max_p1());
            t.assert_equal(ref.has_shift, cells.get_has_shift());

            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                t.assert_equal(msg + " seq_pos_min", ref.seq_pos_min(s), cells.seq_pos_min(s));
                t.assert_equal(msg + " seq_pos_max", ref.seq_pos_max(s), cells.seq_pos_max(s));
            }
        };

        for (uint32_t step = 0; step < 3000; ++step) {
            const uint32_t i = rng() % n;
            const llama_seq_id s = (llama_seq_id) (rng() % n_seq);

            switch (rng() % 12) {
                case 0: { // pos_set on an empty cell
                    if (ref.cells[i].pos == -1) {
                        // keep positions non-negative: is_empty asserts pos == -1 or pos >= 0
                        const llama_pos p = (llama_pos) (rng() % 100);
                        cells.pos_set(i, p);
                        ref.cells[i].pos = p;
                    }
                } break;

                case 1: { // seq_add
                    if (ref.cells[i].pos != -1 && !ref.cells[i].seqs.count(s)) {
                        cells.seq_add(i, s);
                        ref.cells[i].seqs.insert(s);
                    }
                } break;

                case 2: { // seq_rm
                    if (ref.cells[i].seqs.count(s)) {
                        const bool empty = cells.seq_rm(i, s);
                        ref.cells[i].seqs.erase(s);
                        if (ref.cells[i].seqs.empty()) {
                            ref.cells[i].pos   = -1;
                            ref.cells[i].shift = 0;
                            ref.cells[i].ext   = {};
                        }
                        t.assert_equal(ref.cells[i].seqs.empty(), empty);
                    }
                } break;

                case 3: { // seq_keep (skip used cells without seqs: the API asserts on them)
                    if (ref.cells[i].pos == -1 || !ref.cells[i].seqs.empty()) {
                        const bool empty = cells.seq_keep(i, s);
                        if (ref.cells[i].seqs.count(s)) {
                            ref.cells[i].seqs = {s};
                            t.assert_true(!empty);
                        } else if (!ref.cells[i].seqs.empty()) {
                            ref.cells[i] = cells_ref::cell_t{};
                            t.assert_true(empty);
                        } else {
                            t.assert_true(!empty);
                        }
                    }
                } break;

                case 4: { // seqS_add
                    if (ref.cells[i].pos != -1) {
                        std::vector<llama_seq_id> pick;
                        for (llama_seq_id k = 0; k < (llama_seq_id) n_seq; ++k) {
                            if (!ref.cells[i].seqs.count(k)) {
                                pick.push_back(k);
                            }
                        }
                        if (!pick.empty()) {
                            const size_t k = 1 + (rng() % pick.size());
                            cells.seqS_add(i, (int32_t) k, pick.data());
                            for (size_t j = 0; j < k; ++j) {
                                ref.cells[i].seqs.insert(pick[j]);
                            }
                        }
                    }
                } break;

                case 5: { // rm
                    if (ref.cells[i].pos != -1) {
                        cells.rm(i);
                        ref.cells[i] = cells_ref::cell_t{};
                    }
                } break;

                case 6: { // rm_single
                    if (ref.cells[i].seqs.size() == 1) {
                        cells.rm_single(i, *ref.cells[i].seqs.begin());
                        ref.cells[i] = cells_ref::cell_t{};
                    }
                } break;

                case 7: { // pos_add
                    if (ref.cells[i].pos != -1) {
                        const int d = (int) (rng() % 41) - 20;
                        const bool removed = cells.pos_add(i, d);
                        ref.has_shift = true;
                        if (ref.cells[i].pos + d < 0) {
                            // pos_add clears pos/shift/seq but not ext
                            ref.cells[i].seqs.clear();
                            ref.cells[i].pos   = -1;
                            ref.cells[i].shift = 0;
                            t.assert_true(removed);
                        } else {
                            ref.cells[i].pos   += d;
                            ref.cells[i].shift += d;
                            t.assert_true(!removed);
                        }
                    }
                } break;

                case 8: { // pos_div
                    if (ref.cells[i].pos != -1) {
                        const int d = 1 + (int) (rng() % 4);
                        const llama_pos p_old = ref.cells[i].pos;
                        cells.pos_div(i, d);
                        ref.cells[i].pos = p_old / d;
                        ref.cells[i].shift += p_old - ref.cells[i].pos;
                        ref.has_shift = true;
                    }
                } break;

                case 9: { // nextHead
                    const llama_pos p0 = (llama_pos) ((int) (rng() % 40) - 10);
                    const llama_pos p1 = p0 + (llama_pos) (rng() % 60);

                    uint32_t new_head = n;
                    for (uint32_t k = 0; k < n; ++k) {
                        auto & cl = ref.cells[k];
                        if (cl.pos == -1 || cl.pos < p0 || cl.pos >= p1 || !cl.seqs.count(s)) {
                            continue;
                        }
                        cl.seqs.erase(s);
                        if (cl.seqs.empty()) {
                            cl.pos   = -1;
                            cl.shift = 0;
                            cl.ext   = {};
                            if (new_head == n) {
                                new_head = k;
                            }
                        }
                    }

                    t.assert_equal(new_head, cells.nextHead(s, p0, p1));
                } break;

                case 10: { // reset_shift
                    cells.reset_shift();
                    for (auto & cl : ref.cells) {
                        cl.shift = 0;
                    }
                    ref.has_shift = false;
                } break;

                case 11: { // ext_set
                    const llama_kv_cell_ext e{
                        /*x=*/ (llama_pos) (rng() % 100),
                        /*y=*/ (llama_pos) (rng() % 100),
                    };
                    cells.ext_set(i, e);
                    ref.cells[i].ext = e;
                } break;
            }

            // periodic save/restore roundtrip of a random contiguous range
            if (step % 256 == 128) {
                cells.reset_shift();
                for (auto & cl : ref.cells) {
                    cl.shift = 0;
                }
                ref.has_shift = false;

                const uint32_t a = rng() % (n - 8);
                const uint32_t len = 1 + (rng() % 8);

                const llama_kv_cells saved = cells.cp(a, len);

                for (uint32_t k = a; k < a + len; ++k) {
                    if (ref.cells[k].pos != -1) {
                        cells.rm(k);
                        ref.cells[k] = cells_ref::cell_t{};
                    }
                }

                std::vector<uint32_t> idxs;
                for (uint32_t k = a; k < a + len; ++k) {
                    idxs.push_back(k);
                }
                cells.set(idxs, saved);

                // restore the reference model from the copy
                for (uint32_t j = 0; j < len; ++j) {
                    auto & cl = ref.cells[a + j];
                    if (saved.is_empty(j)) {
                        // empty cells are not touched by set(): it only copies ext from saved,
                        // which equals the current ext of the cell
                        continue;
                    }
                    cl.pos   = saved.pos_get(j);
                    cl.ext   = saved.ext_get(j);
                    cl.seqs.clear();
                    for (llama_seq_id s2 = 0; s2 < (llama_seq_id) n_seq; ++s2) {
                        if (saved.seq_has(j, s2)) {
                            cl.seqs.insert(s2);
                        }
                    }
                }
            }

            check("step " + std::to_string(step));
        }
    });
}

int main(int argc, char ** argv) {
    testing t;

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("bitops",      test_bitops);
    t.test("ext",         test_ext);
    t.test("basic",       test_basic);
    t.test("seq",         test_seq);
    t.test("seq_pos",     test_seq_pos);
    t.test("shift",       test_shift);
    t.test("remove",      test_remove);
    t.test("save_restore", test_save_restore);
    t.test("next_head",   test_next_head);
    t.test("random",      test_random);

    return t.summary();
}
