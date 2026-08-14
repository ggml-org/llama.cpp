#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
#  define LLAMA_HAS_STD_BITOPS 1
#elif defined(__has_include) && __has_include(<bit>) && __cplusplus >= 202002L
#  define LLAMA_HAS_STD_BITOPS 1
#else
#  define LLAMA_HAS_STD_BITOPS 0
#endif

namespace llama_bits {
inline int popcount64(uint64_t x) {
#if LLAMA_HAS_STD_BITOPS
    return std::popcount(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    return (int)__popcnt64(x);
#else
    // Hacker's Delight
    x = x - ((x >> 1) & UINT64_C(0x5555555555555555));
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    return (int)((x * UINT64_C(0x0101010101010101)) >> 56);
#endif
}

inline int countr_zero64(uint64_t x) {
    assert(x != 0);
#if LLAMA_HAS_STD_BITOPS
    return std::countr_zero(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    int r = 63;
    if (x & UINT64_C(0x00000000FFFFFFFF)) r -= 32; else x >>= 32;
    if (x & UINT64_C(0x000000000000FFFF)) r -= 16; else x >>= 16;
    if (x & UINT64_C(0x00000000000000FF)) r -=  8; else x >>=  8;
    if (x & UINT64_C(0x000000000000000F)) r -=  4; else x >>=  4;
    if (x & UINT64_C(0x0000000000000003)) r -=  2; else x >>=  2;
    if (x & UINT64_C(0x0000000000000001)) r -=  1;
    return r;
#endif
}

inline int countl_zero64(uint64_t x) {
    assert(x != 0);
#if LLAMA_HAS_STD_BITOPS
    return std::countl_zero(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 63 - (int)idx;
#else
    int r = 0;
    if (!(x & UINT64_C(0xFFFFFFFF00000000))) { r += 32; x <<= 32; }
    if (!(x & UINT64_C(0xFFFF000000000000))) { r += 16; x <<= 16; }
    if (!(x & UINT64_C(0xFF00000000000000))) { r +=  8; x <<=  8; }
    if (!(x & UINT64_C(0xF000000000000000))) { r +=  4; x <<=  4; }
    if (!(x & UINT64_C(0xC000000000000000))) { r +=  2; x <<=  2; }
    if (!(x & UINT64_C(0x8000000000000000))) { r +=  1; }
    return r;
#endif
}

} // namespace llama_bits

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        memset(this, 0, sizeof(*this));
    }
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        std::fill(used_bits.begin(), used_bits.end(), 0);
        used_cnt = 0;

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);

        used_bits.assign((n + 63) / 64, 0);
        used_cnt = 0;

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used_cnt;
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        for (size_t w = 0; w < used_bits.size(); ++w) {
            if (used_bits[w]) {
                return (uint32_t)(w * 64 + llama_bits::countr_zero64(used_bits[w]));
            }
        }
        return 0;
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        for (size_t w = used_bits.size(); w-- > 0;) {
            if (used_bits[w]) {
                return (uint32_t)(w * 64 + 64 - llama_bits::countl_zero64(used_bits[w]));
            }
        }
        return 0;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other);

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used_erase(i);
    }

    void rm_single(uint32_t i, llama_seq_id seq_id);

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) { // need compact after some seq_rm
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used_erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, pos[i]);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used_erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, pos[i]);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);
        for (int k = 0; k < N_SEQ_WORDS; ++k) {
            if (seq[i].w[k]) return k * 64 + llama_bits::countr_zero64(seq[i].w[k]);
        }
        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);
        const auto & v = seq_pos[seq_id];
        return v.total > 0 ? v.min() : -1;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);
        const auto & v = seq_pos[seq_id];
        return v.total > 0 ? v.max() : -1;
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used_insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used_erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

    const llama_pos * pos_data() const { return pos.data(); }
    void seqS_add(uint32_t i, int32_t n, llama_seq_id *_seq);
    void compact(llama_seq_id s);
    uint32_t nextHead(int32_t seq_id, llama_pos p0, llama_pos p1);
  private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::vector<uint64_t> used_bits;
    uint32_t             used_cnt = 0;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    static_assert(LLAMA_MAX_SEQ > 0 && (LLAMA_MAX_SEQ % 64) == 0,
                  "LLAMA_MAX_SEQ must be a multiple of 64");
    static constexpr int N_SEQ_WORDS = LLAMA_MAX_SEQ / 64;

    struct seq_set_t {
        uint64_t w[N_SEQ_WORDS]{};   // zero-init

        void reset()                { for (auto & x : w) x = 0; }
        void reset(int s)           { w[s >> 6] &= ~(1ull << (s & 63)); }
        void set(int s)             { w[s >> 6] |=  1ull << (s & 63); }
        bool test(int s) const      { return (w[s >> 6] >> (s & 63)) & 1; }
        bool none() const           { for (auto x : w) if (x) return false; return true; }
        bool any() const            { return !none(); }
        int  count() const          { int c = 0; for (auto x : w) c += llama_bits::popcount64(x); return c; }
        bool operator==(const seq_set_t & o) const {
            for (int k = 0; k < N_SEQ_WORDS; ++k) if (w[k] != o.w[k]) return false;
            return true;
        }
        bool operator!=(const seq_set_t & o) const { return !(*this == o); }
    };

    std::vector<seq_set_t> seq;

    struct seq_pos_t {
        llama_pos          base  = 0;
        std::vector<int32_t> cnt;
        int64_t            total = 0;
        uint32_t           head  = 0;
        uint32_t           tail  = 0;

        void clear() { base = 0; cnt.clear(); total = 0; head = 0; tail = 0; }
        llama_pos min() const { return base + (llama_pos)head; }
        llama_pos max() const { return base + (llama_pos)tail; }
    };

    seq_pos_t seq_pos[LLAMA_MAX_SEQ];

     void used_insert(uint32_t i) {
        assert(i < pos.size());
        const uint64_t bit = 1ull << (i & 63);
        if (!(used_bits[i >> 6] & bit)) {
            used_bits[i >> 6] |= bit;
            ++used_cnt;
        }
    }

    void used_erase(uint32_t i) {
        assert(i < pos.size());
        const uint64_t bit = 1ull << (i & 63);
        if (used_bits[i >> 6] & bit) {
            used_bits[i >> 6] &= ~bit;
            --used_cnt;
        }
    }

    // O(1)
    void seq_pos_inc(llama_seq_id s, llama_pos p);

    // O(1) amort
    void seq_pos_dec(llama_seq_id s, llama_pos p);

    // remove cell i
    void seq_pos_rm(uint32_t i) {
         for (int k = 0; k < N_SEQ_WORDS; ++k) {
             for (auto m = seq[i].w[k]; m; m &= m - 1) {
                 seq_pos_dec(k * 64 + llama_bits::countr_zero64(m), pos[i]);
             }
         }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
         for (int k = 0; k < N_SEQ_WORDS; ++k) {
             for (auto m = seq[i].w[k]; m; m &= m - 1) {
                 seq_pos_inc(k * 64 + llama_bits::countr_zero64(m), pos[i]);
             }
         }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
