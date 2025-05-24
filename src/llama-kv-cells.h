#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <bitset>
#include <cassert>
#include <vector>

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells_unified {
public:
    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            shift[i] =  0;
            seq[i].reset();
        }

        used      = 0;
        has_shift = false;
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
        shift.resize(n);
        seq.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    void mv(uint32_t isrc, uint32_t idst) {
        assert(isrc < pos.size());
        assert(idst < pos.size());

        pos  [idst] = pos  [isrc];
        shift[idst] = shift[isrc];
        seq  [idst] = seq  [isrc];

        pos  [isrc] = -1;
        shift[isrc] =  0;
        seq  [isrc].reset();
    }

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells_unified cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells_unified res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            res.pos[j] = pos[i + j];
            res.seq[j] = seq[i + j];

            assert(shift[i + j] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells_unified & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            if (pos[i + j] == -1 && other.pos[j] != -1) {
                used++;
            }

            if (pos[i + j] != -1 && other.pos[j] == -1) {
                used--;
            }

            pos[i + j] = other.pos[j];
            seq[i + j] = other.seq[j];

            assert(shift[i + j] == 0);
        }
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);

        if (seq[i].none()) {
            pos[i] = -1;

            used--;

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq[i].reset();
            seq[i].set(seq_id);

            return false;
        }

        if (seq[i].any()) {
            seq[i].reset();
            pos[i] = -1;

            used--;

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

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
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
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

        pos[i] = p;
        used++;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            pos[i] = -1;
            seq[i].reset();

            used--;

            return true;
        }

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        has_shift = true;
    }

private:
    uint32_t used = 0; // used cells (i.e. pos[i] != -1, allowed to not have any seq_id)

    bool has_shift = false;

    std::vector<llama_pos> pos;

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

    std::vector<std::bitset<LLAMA_MAX_PARALLEL_SEQUENCES>> seq;
};

