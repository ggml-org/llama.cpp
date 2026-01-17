// run:
//  g++ -std=c++17 -I ../src/ -I ../include -I ../ggml/include ../tests/test-kq-mask.cpp  && ./a.out

#include "llama-hparams.h"
#include "llama-batch.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

// Populate random data into cells before tests
static void populate_random_cells(std::vector<llama_kv_cells> & cells, std::mt19937 & gen) {
    std::uniform_int_distribution<int> pos_dist(0, cells.size() + 64);
    for (auto & c : cells) {
        // c is already resized to n_kv by caller
        for (uint32_t i = 0; i < c.size(); ++i) {
            // randomly decide if cell is used
            if (std::uniform_int_distribution<int>(0, 3)(gen) == 0) {
                continue;  // leave empty ~25%
            }
            llama_pos p = pos_dist(gen);
            c.pos_set(i, p);
            // random ext
            llama_kv_cell_ext ext{ (llama_pos) std::uniform_int_distribution<int>(0, 10)(gen),
                                   (llama_pos) std::uniform_int_distribution<int>(0, 10)(gen) };
            c.ext_set(i, ext);
            // assign a random seq id
            llama_seq_id sid =
                (llama_seq_id) std::uniform_int_distribution<int>(0, std::min(10, (int) LLAMA_MAX_SEQ - 1))(gen);
            c.seq_add(i, sid);
        }
    }
}

// Simplified llama_hparams structure for testing
// Simplified args_set_input_kq_mask structure
struct args_set_input_kq_mask {
          llama_hparams & hparams;
    const llama_ubatch *  ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t> &       seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

// Old implementation of set_input_kq_mask_impl
template <bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl_old(const args_set_input_kq_mask & args, float * data) {
    const auto & ubatch = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    std::fill(data, data + n_kv*ubatch->n_tokens, -INFINITY);

    // Use only the previous KV cells of the correct sequence for each token of the ubatch.
    // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
    // Example with a cache of 10 tokens, 2 tokens populated in cache and 3 tokens in batch:
    //   Causal mask:
    //      xxx-------
    //      xxxx------
    //      xxxxx-----
    //   Non-causal mask:
    //      xxxxx-----
    //      xxxxx-----
    //      xxxxx-----
    // To visualize the mask, see https://github.com/ggml-org/llama.cpp/pull/12615
    // TODO: optimize this section
    for (uint32_t h = 0; h < 1; ++h) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            for (uint32_t ii = 0; ii < n_tps; ++ii) {
                const uint32_t i = s*n_tps + ii;

                const llama_seq_id seq_id = ubatch->seq_id[i][0];

                const auto & cells = v_cells[seq_to_stream[seq_id]];

                const llama_pos p1 = ubatch->pos[i];

                // for M-RoPE
                const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
                const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

                const uint64_t idst = n_kv*(h*n_stream*n_tps + s*n_tps + ii);

                for (uint32_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j)) {
                        continue;
                    }

                    // mask the token if not the same sequence
                    if (!cells.seq_has(j, seq_id)) {
                        continue;
                    }

                    const llama_pos p0 = cells.pos_get(j);

                    // mask future tokens
                    if (causal && p0 > p1) {
                        continue;
                    }

                    // M-RoPE causal mask
                    if (causal && is_2d && p0 == p1) {
                        const auto & p0_ext = cells.ext_get(j);
                        if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                            continue;
                        }
                    }

                    // apply SWA if any
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        continue;
                    }

                    data[idst + j] = alibi ? -std::abs(p0 - p1) : 0.0f;
                }
            }
        }
    }
}

// New implementation of set_input_kq_mask_impl (with the optimization from PR #18842)
template <bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl_new(const args_set_input_kq_mask & args, float * data) {
    const auto & ubatch = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s * n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

            llama_pos       p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens * 2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens] : 0;

            const uint64_t idst = n_kv * i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv * srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = -std::abs(p0 - p1);
                } else {
                    data[idst + j] = 0.0f;
                }

                continue;
skip:
                data[idst + j] = -INFINITY;
            }
        }
    }
}

// Wrapper functions to call the implementations
template <bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl_old_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl_old<causal, swa, is_2d, true>(args, data);
    } else {
        set_input_kq_mask_impl_old<causal, swa, is_2d, false>(args, data);
    }
}

template <bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl_new_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl_new<causal, swa, is_2d, true>(args, data);
    } else {
        set_input_kq_mask_impl_new<causal, swa, is_2d, false>(args, data);
    }
}

template <bool causal, bool swa>
static void set_input_kq_mask_impl_old_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl_old_wrapper<causal, swa, true>(args, data);
    } else {
        set_input_kq_mask_impl_old_wrapper<causal, swa, false>(args, data);
    }
}

template <bool causal, bool swa>
static void set_input_kq_mask_impl_new_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl_new_wrapper<causal, swa, true>(args, data);
    } else {
        set_input_kq_mask_impl_new_wrapper<causal, swa, false>(args, data);
    }
}

template <bool causal>
static void set_input_kq_mask_impl_old_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl_old_wrapper<causal, true>(args, data);
    } else {
        set_input_kq_mask_impl_old_wrapper<causal, false>(args, data);
    }
}

template <bool causal>
static void set_input_kq_mask_impl_new_wrapper(const args_set_input_kq_mask & args, float * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl_new_wrapper<causal, true>(args, data);
    } else {
        set_input_kq_mask_impl_new_wrapper<causal, false>(args, data);
    }
}

// Simple test function
static void test_kq_mask_impl() {
    std::cout << "Testing set_input_kq_mask implementations...\n";

    // Parameter space (kept small for test speed)
    const std::vector<int> test_n_kv     = { 64, 512, 2048, 8192 };
    const std::vector<int> test_n_stream = { 1, 2, 4 };
    const std::vector<int> test_n_tokens = { 1, 8, 64, 128, 512 };

    // Random generator
    std::random_device rd;
    std::mt19937       gen(rd());

    int total_tests  = 0;
    int passed_tests = 0;

    // Helper to run a single configuration and compare old vs new
    auto run_case = [&](bool causal, bool alibi, llama_swa_type swa_type, int n_swa,
                        args_set_input_kq_mask args, std::vector<float> & data_old,
                        std::vector<float> & data_new) {
        args.hparams.use_alibi = alibi;
        args.swa_type = swa_type;
        args.n_swa    = n_swa;

        // call appropriate wrappers based on causal flag
        if (causal) {
            set_input_kq_mask_impl_old_wrapper<true>(args, data_old.data());
            set_input_kq_mask_impl_new_wrapper<true>(args, data_new.data());
        } else {
            set_input_kq_mask_impl_old_wrapper<false>(args, data_old.data());
            set_input_kq_mask_impl_new_wrapper<false>(args, data_new.data());
        }

        // compare
        bool match = true;
        for (size_t i = 0; i < data_old.size(); ++i) {
            if (data_old[i] != data_new[i]) {
                match = false;
                break;
            }
        }

        ++total_tests;
        if (match) {
            ++passed_tests;
            std::cout << "✓ Test passed: " << (causal ? "causal" : "non-causal") << ", "
                      << (alibi ? "ALIBI" : "no ALIBI") << ", " << (swa_type == LLAMA_SWA_TYPE_NONE ? "no SWA" : "SWA")
                      << ", n_kv=" << args.n_kv << ", n_stream=" << args.n_stream
                      << ", n_tokens=" << args.n_tps * args.n_stream << "\n";
        } else {
            std::cout << "✗ Test failed: " << (causal ? "causal" : "non-causal") << ", "
                      << (alibi ? "ALIBI" : "no ALIBI") << ", " << (swa_type == LLAMA_SWA_TYPE_NONE ? "no SWA" : "SWA")
                      << ", n_kv=" << args.n_kv << ", n_stream=" << args.n_stream
                      << ", n_tokens=" << args.n_tps * args.n_stream << "\n";
        }
    };

    // Main loops over dimensions
    for (int n_kv : test_n_kv) {
        for (int n_stream : test_n_stream) {
            for (int n_tokens : test_n_tokens) {
                for (int n_pos : {1, 3}) {
                    if (n_tokens > n_kv) {
                        continue;  // unrealistic
                    }
                    if (n_stream > 1 && n_tokens % n_stream != 0) {
                        continue;  // must divide evenly
                    }

                    // Prepare random test data
                    std::vector<llama_pos>      test_pos(n_tokens*n_pos);
                    std::vector<int32_t>        test_n_seq_id(n_tokens);
                    std::vector<llama_seq_id>   test_seq_id_data(n_tokens);
                    std::vector<llama_seq_id *> test_seq_id(n_tokens);

                    std::uniform_int_distribution<> pos_dist(std::max(0, n_kv - 2 * n_tokens), n_kv);
                    std::uniform_int_distribution<> seq_dist(0, std::min(LLAMA_MAX_SEQ, n_stream - 1));
                    for (int i = 0; i < n_tokens; ++i) {
                        for (int p = 0; p < n_pos; ++p) {
                            test_pos[i*n_pos + p] = pos_dist(gen);
                        }
                        test_n_seq_id[i]    = 1;
                        test_seq_id_data[i] = seq_dist(gen);
                        test_seq_id[i]      = &test_seq_id_data[i];
                    }

                    // Build ubatch
                    llama_ubatch ubatch{};
                    ubatch.n_tokens     = n_tokens;
                    ubatch.n_seq_tokens = n_tokens;
                    ubatch.n_seqs       = n_stream;
                    ubatch.n_seqs_unq   = n_stream;
                    ubatch.n_pos        = n_pos;
                    ubatch.pos          = test_pos.data();
                    ubatch.n_seq_id     = test_n_seq_id.data();
                    ubatch.seq_id       = test_seq_id.data();
                    ubatch.seq_id_unq   = test_seq_id_data.data();

                    // Dummy hparams (will be mutated per case)
                    llama_hparams hparams{};

                    // Cells per stream
                    std::vector<llama_kv_cells> cells(n_stream);
                    for (int s = 0; s < n_stream; ++s) {
                        cells[s].resize(n_kv);
                    }

                    // Populate random data into cells
                    static std::random_device rd;
                    static std::mt19937       gen(rd());
                    populate_random_cells(cells, gen);

                    // seq_to_stream mapping
                    std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);
                    for (int s = 0; s < n_stream; ++s) {
                        seq_to_stream[s] = s;
                    }

                    // Base args (will be copied/modified for each case)
                    args_set_input_kq_mask base_args = {
                        .hparams       = hparams,
                        .ubatch        = &ubatch,
                        .v_cells       = cells,
                        .seq_to_stream = seq_to_stream,
                        .n_swa         = 0,
                        .swa_type      = LLAMA_SWA_TYPE_NONE,
                        .n_kv          = n_kv,
                        .n_stream      = n_stream,
                        .n_tps         = n_tokens / n_stream,
                    };

                    // Output buffers
                    std::vector<float> data_old(n_tokens * n_kv);
                    std::vector<float> data_new(n_tokens * n_kv);

                    // 1) causal, no SWA, no ALIBI
                    run_case(true, false, LLAMA_SWA_TYPE_NONE, 0, base_args, data_old, data_new);
                    // 2) causal, SWA, no ALIBI
                    run_case(true, false, LLAMA_SWA_TYPE_STANDARD, 128, base_args, data_old, data_new);
                    // 3) non‑causal, no SWA, no ALIBI
                    run_case(false, false, LLAMA_SWA_TYPE_NONE, 0, base_args, data_old, data_new);
                    // 4) non‑causal, SWA, no ALIBI
                    run_case(false, false, LLAMA_SWA_TYPE_STANDARD, 128, base_args, data_old, data_new);
                    // 5) causal, ALIBI, no SWA
                    run_case(true, true, LLAMA_SWA_TYPE_NONE, 0, base_args, data_old, data_new);
                    // 6) non‑causal, ALIBI, no SWA
                    run_case(false, true, LLAMA_SWA_TYPE_NONE, 0, base_args, data_old, data_new);
                }
            }
        }
    }

    std::cout << "Test completed. Passed: " << passed_tests << "/" << total_tests << "\n";
}

int main() {
    test_kq_mask_impl();
    return 0;
}
