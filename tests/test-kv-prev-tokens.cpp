// unit test: llama_kv_cells::seq_pos_token_le vs brute-force ascending scan
//
// the reference implements exactly what llama_kv_cache::get_prev_tokens() does:
// scan all cells ascending (for_each_token_in visits `used` in ascending row order), build
// hist[(seq, pos)] = token with last-write-wins (= largest row), then resolve each need by
// walking q = p..0 and taking the first hit. compares against the O(log n) index lookup.
//
// (llama_kv_cells is header-inline; built as a standalone unit test via tests/CMakeLists.txt)

#include "llama-kv-cells.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

static int n_fail = 0;
static int n_checked = 0;

using hist_t = std::map<std::pair<int, llama_pos>, llama_token>;

static hist_t build_hist(const llama_kv_cells & cells, uint32_t p_max) {
    hist_t hist;
    std::bitset<LLAMA_MAX_SEQ> all;
    all.set();

    cells.for_each_token_in(all, 0, p_max + 1, [&](llama_seq_id s, llama_pos p, llama_token t) {
        hist[{s, p}] = t; // ascending rows: last write (largest row) wins
    });

    return hist;
}

// reference answer for (seq, p): token at the largest position <= p, per the hist
static llama_token ref_lookup(const hist_t & hist, llama_seq_id s, llama_pos p) {
    for (llama_pos q = p; q >= 0; --q) {
        auto it = hist.find({ s, q });
        if (it != hist.end()) {
            return it->second;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void check(const llama_kv_cells & cells, uint32_t p_max, const char * stage) {
    const hist_t hist = build_hist(cells, p_max);

    for (llama_seq_id s = 0; s < 4; ++s) {
        for (llama_pos p = -2; p <= (llama_pos) p_max + 2; ++p) {
            if (p < 0) {
                continue;
            }

            llama_token tok = LLAMA_TOKEN_NULL;
            const int rc = cells.seq_pos_token_le(s, p, &tok);

            const llama_token idx = rc > 0 ? tok : LLAMA_TOKEN_NULL;
            const llama_token ref = ref_lookup(hist, s, p);

            n_checked++;

            if (idx != ref) {
                printf("FAIL [%s] seq=%d p=%d: index=%d reference=%d\n", stage, s, p, idx, ref);
                n_fail++;
            }
        }
    }
}

static void add_cell(llama_kv_cells & cells, uint32_t row, llama_pos p, llama_seq_id s, llama_token t) {
    cells.pos_set(row, p);
    llama_kv_cell_ext e;
    e.tok = t;
    cells.ext_set(row, e);
    cells.seq_add(row, s);
}

int main() {
    // scenario 1: sequential append
    {
        llama_kv_cells cells;
        cells.resize(64);
        for (uint32_t i = 0; i < 20; ++i) {
            add_cell(cells, i, (llama_pos) i, 0, (llama_token) (100 + i));
        }
        check(cells, 19, "sequential");
    }

    // scenario 2: holes (removed middle cells) -> nearest <= fallback
    {
        llama_kv_cells cells;
        cells.resize(64);
        for (uint32_t i = 0; i < 20; ++i) {
            add_cell(cells, i, (llama_pos) i, 0, (llama_token) (100 + i));
        }
        // remove positions 7..12 via seq_rm + cleanup
        for (uint32_t i = 7; i <= 12; ++i) {
            if (cells.seq_rm(i, 0)) { /* becomes empty */ }
        }
        check(cells, 19, "holes");
    }

    // scenario 3: duplicate positions (checkpoint-style copies), incl. removal of copies
    {
        llama_kv_cells cells;
        cells.resize(64);
        for (uint32_t i = 0; i < 10; ++i) {
            add_cell(cells, i, (llama_pos) i, 0, (llama_token) (100 + i));
        }
        // "checkpoint" copy of rows 0..9 into rows 32..41, same positions, same seq, different tokens
        for (uint32_t i = 0; i < 10; ++i) {
            add_cell(cells, 32 + i, (llama_pos) i, 0, (llama_token) (900 + i));
        }
        check(cells, 15, "dup-positions");

        // erase the originals (rows 0..9) -> copies must still resolve
        for (uint32_t i = 0; i < 10; ++i) {
            cells.seq_rm(i, 0);
        }
        check(cells, 15, "dup-orig-removed");

        // erase the copies too
        for (uint32_t i = 32; i < 42; ++i) {
            cells.seq_rm(i, 0);
        }
        check(cells, 15, "dup-copies-removed");
    }

    // scenario 4: restore via set() (prompt-cache load), rows out of position order
    {
        llama_kv_cells cells;
        cells.resize(64);
        add_cell(cells, 0, 0, 0, 100);
        add_cell(cells, 1, 1, 0, 101);

        // simulate memory_seq_cp: write a state block at a LOW row range with HIGH positions
        llama_kv_cells src;
        src.resize(8);
        for (uint32_t j = 0; j < 8; ++j) {
            add_cell(src, j, (llama_pos) (100 + j), 1, (llama_token) (500 + j));
        }
        cells.set(2, src); // rows 2..9 now hold positions 100..107 for seq 1

        // decode continues at 108 on a high row
        add_cell(cells, 10, 108, 1, 600);
        check(cells, 110, "restore-low-rows");
    }

    // scenario 5: multi-seq shared cells
    {
        llama_kv_cells cells;
        cells.resize(32);
        for (uint32_t i = 0; i < 6; ++i) {
            add_cell(cells, i, (llama_pos) i, 0, (llama_token) (100 + i));
            cells.seq_add(i, 1);
        }
        for (uint32_t i = 6; i < 10; ++i) {
            add_cell(cells, i, (llama_pos) i, 1, (llama_token) (200 + i));
        }
        check(cells, 12, "multi-seq");
    }

    // scenario 6: randomized fuzz - random adds/removes/multi-seq vs reference
    // the mirror tracks (row -> seq set) so that seq_rm/seq_add are called only when valid
    {
        std::mt19937 rng(42);
        llama_kv_cells cells;
        cells.resize(256);

        std::map<uint32_t, std::set<llama_seq_id>> mirror;

        for (int step = 0; step < 3000; ++step) {
            const uint32_t row = rng() % 256;

            auto it = mirror.find(row);
            if (it == mirror.end()) {
                const llama_seq_id s = rng() % 3;
                add_cell(cells, row, (llama_pos) (rng() % 60), s, (llama_token) (1000 + rng() % 1000));
                mirror[row].insert(s);
            } else if (it->second.size() < 2 && rng() % 5 == 0) {
                const llama_seq_id s = rng() % 3;
                if (!it->second.count(s)) {
                    cells.seq_add(row, s);
                    it->second.insert(s);
                }
            } else {
                const llama_seq_id s = *it->second.begin();
                const bool empty = cells.seq_rm(row, s);
                it->second.erase(s);
                if (empty != it->second.empty()) {
                    printf("FAIL [fuzz-consistency] row=%u seq_rm returned %d, mirror left %zu\n", row, (int) empty, it->second.size());
                    n_fail++;
                }
                if (it->second.empty()) {
                    mirror.erase(it);
                }
            }

            if (step % 97 == 0) {
                check(cells, 64, "fuzz");
            }
        }
        check(cells, 64, "fuzz-final");
    }

    printf("%s: %d lookups checked, %d failures\n", n_fail ? "FAILED" : "PASSED", n_checked, n_fail);
    return n_fail ? 1 : 0;
}
