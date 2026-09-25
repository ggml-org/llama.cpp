#include "testing.h"

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-memory.h"
#include "../src/llama-vocab.h"

#include <cstdlib>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

// mock memory that only provides per-sequence position ranges
struct mock_memory : public llama_memory_i {
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges; // seq_id -> [pos_min, pos_max]

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_full() override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_update(llama_context *, bool) override { GGML_ASSERT(false && "not implemented"); }

    bool get_can_shift() const override { GGML_ASSERT(false && "not implemented"); }

    void clear(bool) override { GGML_ASSERT(false && "not implemented"); }

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_keep(llama_seq_id) override { GGML_ASSERT(false && "not implemented"); }
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override { GGML_ASSERT(false && "not implemented");  }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.first;
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.second;
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { GGML_ASSERT(false && "not implemented"); }
    void state_read (llama_io_read_i &,  llama_seq_id, llama_state_seq_flags) override { GGML_ASSERT(false && "not implemented"); }
};

// builds a llama_batch_ext without a llama_context
// n_vocab = 0 by default, so every token id is invalid and the tests use embeddings unless stated otherwise
struct batch_builder {
    const uint32_t n_embd;

    llama_batch_ext b;

    batch_builder(
            uint32_t n_embd = 2,
            llama_memory_i * mem = nullptr,
            llama_seq_id n_seq_max = 4,
            uint32_t n_pos_per_embd = 1,
            llama_token n_vocab = 0,
            uint32_t n_embd_inp_enc = 0)
        : n_embd(n_embd),
          b(/*n_tokens_max*/ 64, n_embd, n_embd_inp_enc > 0 ? n_embd_inp_enc : n_embd, n_seq_max, mem, n_vocab, n_pos_per_embd) {}

    // one embedding row for batch index i, values 100*i + k so ubatch contents can be traced back
    std::vector<float> row(int32_t i, uint32_t width) const {
        std::vector<float> r(width);
        for (uint32_t k = 0; k < width; ++k) {
            r[k] = 100.0f*i + k;
        }
        return r;
    }

    // embedding entry with full M-RoPE positions
    int32_t add_embd(const llama_pos * pos, std::initializer_list<llama_seq_id> seq_ids, bool output, uint32_t width = 0) {
        width = width > 0 ? width : n_embd;

        auto it = seq_ids.begin();
        const int32_t idx = b.add_token(*it);
        GGML_ASSERT(idx >= 0);
        for (++it; it != seq_ids.end(); ++it) {
            GGML_ASSERT(b.add_seq(idx, *it));
        }

        const auto r = row(idx, width);
        GGML_ASSERT(b.set_token_embd(idx, { r.data(), 1, width }));
        GGML_ASSERT(b.set_token_pos(idx, pos));
        GGML_ASSERT(b.set_output(idx, output));

        return idx;
    }

    // embedding entry with a single sequential position
    int32_t add(llama_pos p, std::initializer_list<llama_seq_id> seq_ids, bool output) {
        const llama_pos pos[GGML_MROPE_SECTIONS] = { p, 0, 0, 0 };
        return add_embd(pos, seq_ids, output);
    }
};

static void test_init(testing & t) {
    llama_vocab vocab;

    t.test("rejects_n_seq_max_too_large", [&](testing & t) {
        batch_builder bb(2, nullptr, LLAMA_MAX_SEQ + 1);
        bb.add(0, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.b, vocab, false));
    });

    t.test("rejects_invalid_token", [&](testing & t) {
        // n_vocab = 0 -> every token id is out of range
        // set_token_id() refuses such ids, so the token is poked directly to reach the init() check
        batch_builder bb;
        const int32_t idx = bb.b.add_token(0);
        const llama_pos pos = 0;
        bb.b.set_token_pos(idx, &pos);
        bb.b.set_output(idx, true);

        llama_batch_allocr ba(1);

        t.assert_true("set_token_id refuses out of range id", !bb.b.set_token_id(idx, 0));

        bb.b.tokens[idx].id = 0;
        t.assert_true("token id >= n_vocab", !ba.init(bb.b, vocab, false));

        bb.b.tokens[idx].id = -1;
        t.assert_true("negative token id", !ba.init(bb.b, vocab, false));
    });

    t.test("rejects_invalid_seq_id", [&](testing & t) {
        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            t.assert_true("add_token refuses seq_id >= n_seq_max", bb.b.add_token(4) == -3);
            t.assert_true("add_token refuses negative seq_id",   bb.b.add_token(-1) == -3);
        }
        {
            // poke the seq_ids directly to reach the init() check
            batch_builder bb;
            const int32_t idx = bb.add(0, {0}, true);
            bb.b.tokens[idx].seq_ids = { 4 };
            t.assert_true("seq_id >= n_seq_max", !ba.init(bb.b, vocab, false));
        }
        {
            batch_builder bb;
            const int32_t idx = bb.add(0, {0}, true);
            bb.b.tokens[idx].seq_ids = { -1 };
            t.assert_true("negative seq_id", !ba.init(bb.b, vocab, false));
        }
    });

    t.test("copies_pos_seq_output", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, i == 3);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        const llama_batch & batch = ba.get_batch();

        t.assert_equal(4u, ba.get_n_tokens());
        t.assert_true("embedding batch", batch.embd  != nullptr);
        t.assert_true("no token ids",    batch.token == nullptr);

        for (int i = 0; i < 4; ++i) {
            t.assert_equal(i, batch.pos[i]);
            t.assert_equal(1, batch.n_seq_id[i]);
            t.assert_equal(0, batch.seq_id[i][0]);
            t.assert_equal(100.0f*i, batch.embd[i*bb.n_embd]);
        }

        t.assert_equal("only the last token is an output", 1u, ba.get_n_outputs());
        t.assert_equal(0, (int) batch.logits[0]);
        t.assert_equal(1, (int) batch.logits[3]);

        t.assert_equal(0, ba.seq_pos_min(0));
        t.assert_equal(3, ba.seq_pos_max(0));
        t.assert_equal(-1, ba.seq_pos_min(1));
    });

    t.test("output_all", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, true));
        t.assert_equal(4u, ba.get_n_outputs());
    });

    t.test("explicit_logits", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);
        bb.add(1, {0}, false);
        bb.add(2, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
        t.assert_equal(2u, ba.get_n_outputs());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, (int) ub.output[0]);
        t.assert_equal(0, (int) ub.output[1]);
        t.assert_equal(1, (int) ub.output[2]);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 2, out_ids.size());
        t.assert_equal(0, out_ids[0]);
        t.assert_equal(2, out_ids[1]);
    });

    t.test("pos_after_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        batch_builder bb(2, &mem);
        for (int i = 0; i < 3; ++i) {
            bb.add(10 + i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        t.assert_equal("pos continues after memory", 10, ba.seq_pos_min(0));
        t.assert_equal(12, ba.seq_pos_max(0));
    });

    t.test("pos_continuity_with_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(1);

        {
            batch_builder bb(2, &mem);
            bb.add(10, {0}, false);
            bb.add(11, {0}, true);
            t.assert_true("pos_max + 1 is accepted", ba.init(bb.b, vocab, false));
        }
        {
            batch_builder bb(2, &mem);
            bb.add(11, {0}, false);
            bb.add(12, {0}, true);
            t.assert_true("gap after memory is rejected", !ba.init(bb.b, vocab, false));
        }
        {
            batch_builder bb(2, &mem);
            bb.add(9, {0}, false);
            bb.add(10, {0}, true);
            t.assert_true("overlap with memory is rejected", !ba.init(bb.b, vocab, false));
        }
    });

    t.test("rejects_non_continuous_positions", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(1, {0}, false);
        bb.add(3, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.b, vocab, false));
    });

    t.test("rejects_decreasing_positions", [&](testing & t) {
        batch_builder bb;
        const llama_pos    pos[7] = {4, 5, 0, 1, 6, 2, 3};
        const llama_seq_id seq[7] = {0, 0, 1, 1, 0, 1, 0};
        for (int i = 0; i < 7; ++i) {
            bb.add(pos[i], {seq[i]}, false);
        }
        // seq 0 sees positions 4,5,6,3 in batch order -> the trailing 3 decreases

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.b, vocab, false));
    });

    t.test("allows_equal_positions_in_seq", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(0, {0}, false);
        bb.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
    });

    t.test("rejects_coupled_diverged_seqs", [&](testing & t) {
        llama_batch_allocr ba(1);

        mock_memory mem;
        mem.ranges[0] = {0, 5};
        mem.ranges[1] = {2, 5}; // same pos_max, different pos_min -> diverged
        {
            batch_builder bb(2, &mem);
            bb.add(6, {0, 1}, true);
            t.assert_true(!ba.init(bb.b, vocab, false));
        }

        mem.ranges[1] = {0, 5};
        {
            batch_builder bb(2, &mem);
            bb.add(6, {0, 1}, true);
            t.assert_true(ba.init(bb.b, vocab, false));
        }
    });
}

static void test_content_types(testing & t) {
    llama_vocab vocab;

    t.test("token_and_embd_together", [&](testing & t) {
        // e.g. MTP hook batches: a token id and its embedding on the same entry
        batch_builder bb(2, nullptr, 4, 1, /*n_vocab*/ 10);

        const int32_t idx = bb.b.add_token(0);
        t.assert_true(bb.b.set_token_id(idx, 3));
        const auto r = bb.row(idx, bb.n_embd);
        t.assert_true(bb.b.set_token_embd(idx, { r.data(), 1, bb.n_embd }));
        const llama_pos pos = 0;
        bb.b.set_token_pos(idx, &pos);
        bb.b.set_output(idx, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        const llama_batch & batch = ba.get_batch();
        t.assert_true("token ids are kept",   batch.token != nullptr);
        t.assert_true("embeddings are kept",  batch.embd  != nullptr);
        t.assert_equal(3, batch.token[0]);
        t.assert_equal(0.0f, batch.embd[0]);
        t.assert_equal(1.0f, batch.embd[1]);

        llama_ubatch ub = ba.split_simple(1);
        t.assert_true(ub.token != nullptr && ub.embd != nullptr);
        t.assert_equal(3, ub.token[0]);
    });

    t.test("rejects_mixed_content_types", [&](testing & t) {
        batch_builder bb(2, nullptr, 4, 1, /*n_vocab*/ 10);

        // entry 0: token only, entry 1: token + embd
        const llama_pos p0 = 0;
        const llama_pos p1 = 1;

        int32_t i0 = bb.b.add_token(0);
        bb.b.set_token_id(i0, 1);
        bb.b.set_token_pos(i0, &p0);

        int32_t i1 = bb.b.add_token(0);
        bb.b.set_token_id(i1, 2);
        const auto r = bb.row(i1, bb.n_embd);
        bb.b.set_token_embd(i1, { r.data(), 1, bb.n_embd });
        bb.b.set_token_pos(i1, &p1);
        bb.b.set_output(i1, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.b, vocab, false));
    });

    t.test("rejects_neither_token_nor_embd", [&](testing & t) {
        batch_builder bb;
        const int32_t idx = bb.b.add_token(0);
        const llama_pos pos = 0;
        bb.b.set_token_pos(idx, &pos);
        bb.b.set_output(idx, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.b, vocab, false));
    });

    t.test("rejects_embd_size_mismatch", [&](testing & t) {
        batch_builder bb; // n_embd = 2, n_embd_inp_enc = 2
        const int32_t idx = bb.b.add_token(0);
        const auto r = bb.row(idx, 8);

        t.assert_true("too small", !bb.b.set_token_embd(idx, { r.data(), 1, 1 }));
        t.assert_true("too large", !bb.b.set_token_embd(idx, { r.data(), 1, 3 }));
        t.assert_true("zero rows", !bb.b.set_token_embd(idx, { r.data(), 0, 2 }));
        t.assert_true("null data", !bb.b.set_token_embd(idx, { nullptr,  1, 2 }));
        t.assert_true("same total via a different split is accepted", bb.b.set_token_embd(idx, { r.data(), 2, 1 }));
    });

    t.test("rejects_double_embd", [&](testing & t) {
        batch_builder bb;
        const int32_t idx = bb.add(0, {0}, true);
        const auto r = bb.row(idx, bb.n_embd);
        t.assert_true(!bb.b.set_token_embd(idx, { r.data(), 1, bb.n_embd }));
    });

    t.test("encoder_width", [&](testing & t) {
        // e.g. eagle3/dflash: extracted features are wider than the decoder input
        const uint32_t n_embd_enc = 6;
        batch_builder bb(2, nullptr, 4, 1, 0, n_embd_enc);

        const llama_pos p0 = 0;
        const llama_pos p1 = 1;
        bb.add_embd(&p0, {0}, false, n_embd_enc);
        bb.add_embd(&p1, {0}, true,  n_embd_enc);

        t.assert_equal("batch width follows the first embedding", (size_t) n_embd_enc, bb.b.n_embd);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        // the ubatch uses the encoder stride: token 1 starts at offset n_embd_enc
        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(100.0f, ub.embd[n_embd_enc]);
        t.assert_equal(105.0f, ub.embd[n_embd_enc + 5]);
    });

    t.test("rejects_mixing_widths", [&](testing & t) {
        batch_builder bb(2, nullptr, 4, 1, 0, /*n_embd_inp_enc*/ 6);

        const llama_pos p0 = 0;
        bb.add_embd(&p0, {0}, false, 2); // first entry fixes the batch width to 2

        const int32_t idx = bb.b.add_token(0);
        const auto r = bb.row(idx, 6);
        t.assert_true(!bb.b.set_token_embd(idx, { r.data(), 1, 6 }));
    });
}

static void test_split(testing & t) {
    llama_vocab vocab;

    t.test("split_simple_chunks", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 5; ++i) {
            bb.add(i, {0}, i == 4);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_true(!ub.equal_seqs());
        t.assert_equal(1u, ub.n_seqs_unq);
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        for (int i = 0; i < 2; ++i) {
            t.assert_equal(i, ub.pos[i]);
            t.assert_equal(1, ub.n_seq_id[i]);
            t.assert_equal(0, ub.seq_id[i][0]);
            t.assert_equal(100.0f*i, ub.embd[i*bb.n_embd]);
            t.assert_equal(100.0f*i + 1, ub.embd[i*bb.n_embd + 1]);
        }

        ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_simple(2);
        t.assert_equal(1u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(1, (int) ub.output[0]);

        t.assert_equal(5u, ba.get_n_used());

        ub = ba.split_simple(2);
        t.assert_equal("batch is consumed", 0u, ub.n_tokens);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 1, out_ids.size());
        t.assert_equal(4, out_ids[0]);
    });

    t.test("split_reset_allows_resplit", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(i, {0}, i == 2);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        while (ba.split_simple(1).n_tokens > 0) {
        }
        t.assert_equal(3u, ba.get_n_used());

        ba.split_reset();
        t.assert_equal(0u, ba.get_n_used());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
    });

    t.test("split_equal_unequal_lengths", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, i == 3);
        }
        for (int i = 0; i < 2; ++i) {
            bb.add(i, {1}, i == 1);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_equal(8, false, 0);
        t.assert_true(ub.equal_seqs());
        t.assert_equal("both seqs advance by the shorter length", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        // tokens are grouped per sequence set: [s0 s0 s1 s1]
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(0, ub.seq_id[1][0]);
        t.assert_equal(1, ub.seq_id[2][0]);
        t.assert_equal(1, ub.seq_id[3][0]);
        t.assert_equal(0, ub.pos[0]);
        t.assert_equal(1, ub.pos[1]);
        t.assert_equal(0, ub.pos[2]);
        t.assert_equal(1, ub.pos[3]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal("only seq 0 remains", 2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal(0u, ub.n_tokens);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("split_equal_coupled", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0, 1}, false);
        bb.add(1, {0, 1}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_equal(4, true, 0);
        t.assert_equal("sequential split rejects coupled seqs", 0u, ub.n_tokens);

        ub = ba.split_equal(4, false, 0);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal("one sequence set", 1u, ub.n_seqs);
        t.assert_equal("two unique seq ids", 2u, ub.n_seqs_unq);
        t.assert_equal(2, ub.n_seq_id[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        t.assert_equal(1, ub.seq_idx[1]);
    });

    t.test("split_seq_per_sequence", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id s = 0; s < 3; ++s) {
            bb.add(0, {s}, false);
            bb.add(1, {s}, true);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        for (llama_seq_id s = 0; s < 3; ++s) {
            llama_ubatch ub = ba.split_seq(8);
            t.assert_equal(2u, ub.n_tokens);
            t.assert_equal(1u, ub.n_seqs);
            t.assert_equal(s, ub.seq_id[0][0]);
            t.assert_equal(s, ub.seq_id_unq[0]);
        }

        t.assert_equal(0u, ba.split_seq(8).n_tokens);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("ubatch_reserve", [&](testing & t) {
        llama_batch_allocr ba(1);

        llama_ubatch ub = ba.ubatch_reserve(3, 2);
        t.assert_equal(6u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        t.assert_true(ub.equal_seqs());
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(1, ub.seq_id_unq[1]);
        t.assert_true(ub.token != nullptr);
        t.assert_true(ub.embd == nullptr);
    });
}

static void test_keep_tail(testing & t) {
    llama_vocab vocab;

    // batch with n_tokens[s] tokens for each seq s, output on the last token of each seq
    auto make_batch = [](batch_builder & bb, std::initializer_list<int> n_tokens) {
        llama_seq_id s = 0;
        for (int n : n_tokens) {
            for (int i = 0; i < n; ++i) {
                bb.add(i, {s}, i == n - 1);
            }
            ++s;
        }
    };

    t.test("noop_when_seqs_complete", [&](testing & t) {
        batch_builder bb;
        make_batch(bb, {2, 2});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("both seqs fit whole", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seq_tokens);

        t.assert_equal(0u, ba.split_equal(4, false, 2).n_tokens);
    });

    t.test("defers_seq_with_short_remainder", [&](testing & t) {
        batch_builder bb;
        make_batch(bb, {2, 3});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        // expansion stops at 2 tokens per seq: seq 0 completes, seq 1 would be left
        // with 1 < n_keep_tail remaining, so it is deferred entirely
        llama_ubatch ub = ba.split_equal(4, true, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2u, ba.get_n_used());

        ub = ba.split_equal(4, true, 2);
        t.assert_equal("deferred seq comes back whole", 3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(1, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }

        t.assert_equal(5u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(4, true, 2).n_tokens);
    });

    t.test("completes_first_seq_when_all_violate", [&](testing & t) {
        batch_builder bb;
        make_batch(bb, {3, 3});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        // expansion stops at 2 tokens per seq, leaving both with 1 < n_keep_tail remaining;
        // seq 0 still fits in n_ubatch, so it is extended to completion and emitted alone
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("truncates_to_preserve_tail", [&](testing & t) {
        batch_builder bb;
        make_batch(bb, {5});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        // 4 tokens would leave a remainder of 1, and the seq does not fit in n_ubatch,
        // so the ubatch is truncated until n_keep_tail tokens remain
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[2]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal("trailing tokens stay in one ubatch", 2u, ub.n_tokens);
        t.assert_equal(3, ub.pos[0]);
        t.assert_equal(4, ub.pos[1]);
        t.assert_equal(1, (int) ub.output[1]);

        t.assert_equal(5u, ba.get_n_used());
    });

    t.test("keeps_full_ubatch_with_sufficient_remainder", [&](testing & t) {
        batch_builder bb;
        make_batch(bb, {6});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("remainder >= n_keep_tail, no truncation", 4u, ub.n_tokens);

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(5, ub.pos[1]);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("multi_seq_prefix_kept", [&](testing & t) {
        batch_builder bb(2, nullptr, 6);
        make_batch(bb, {3, 4});

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));

        // expansion stops at 3 tokens per seq: seq 0 completes, seq 1 has 1 < n_keep_tail
        // remaining and is deferred even though its tokens were already gathered
        llama_ubatch ub = ba.split_equal(6, true, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(6, true, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(7u, ba.get_n_used());
    });
}

static void test_mrope(testing & t) {
    llama_vocab vocab;

    t.test("pos_layout_and_split", [&](testing & t) {
        const uint32_t n_pos  = 4;
        const uint32_t n_embd = 2;

        batch_builder bb(n_embd, nullptr, 4, n_pos);

        // M-RoPE positions per embedding: [temporal, y, x, other]
        const llama_pos pos0[n_pos] = { 10, 5, 7, 0 };
        const llama_pos pos1[n_pos] = { 11, 6, 8, 0 };
        bb.add_embd(pos0, {0}, false);
        bb.add_embd(pos1, {0}, true);

        llama_batch_allocr ba(n_pos);
        t.assert_true(ba.init(bb.b, vocab, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(n_pos, ub.n_pos);
        t.assert_true(ub.is_pos_2d());

        // the ubatch stores positions section-major: [n_pos][n_tokens]
        const llama_pos expected[8] = {10, 11, 5, 6, 7, 8, 0, 0};
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(expected[i], ub.pos[i]);
        }
    });

    t.test("pos_jump_allowed", [&](testing & t) {
        const uint32_t n_pos  = 4;
        const uint32_t n_embd = 2;

        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(n_pos);

        auto try_pos = [&](llama_pos p0) {
            batch_builder bb(n_embd, &mem, 4, n_pos);

            const llama_pos pos[n_pos] = { p0, 1, 1, 0 };
            bb.add_embd(pos, {0}, true);

            return ba.init(bb.b, vocab, false);
        };

        t.assert_true("gap after memory is allowed",     try_pos(15));
        t.assert_true("overlap is allowed for embd",     try_pos(9));
        t.assert_true("pos behind memory is rejected",  !try_pos(8));
    });
}

// conversion from the old llama_batch API (llama_batch_compat::init)
static void test_compat(testing & t) {
    llama_vocab vocab;

    t.test("token_batch_explicit_fields", [&](testing & t) {
        llama_token  token[3]    = { 5, 6, 7 };
        llama_pos    pos[3]      = { 3, 4, 5 };
        int32_t      n_seq_id[3] = { 1, 1, 2 };
        llama_seq_id s0[1]       = { 1 };
        llama_seq_id s1[1]       = { 1 };
        llama_seq_id s2[2]       = { 1, 2 };
        llama_seq_id * seq_id[4] = { s0, s1, s2, nullptr };
        int8_t       logits[3]   = { 0, 1, 0 };

        llama_batch lb = {};
        lb.n_tokens = 3;
        lb.token    = token;
        lb.pos      = pos;
        lb.n_seq_id = n_seq_id;
        lb.seq_id   = seq_id;
        lb.logits   = logits;

        batch_builder bb(2, nullptr, 4, 1, /*n_vocab*/ 100);
        llama_batch_compat::init(bb.b, lb);

        t.assert_equal((size_t) 3, bb.b.tokens.size());
        t.assert_true("no embeddings", bb.b.embd.empty() && bb.b.n_embd == 0);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(token[i], bb.b.tokens[i].id);
            t.assert_equal(pos[i],   bb.b.tokens[i].pos[0]);
            t.assert_true(!bb.b.tokens[i].has_embd);
            t.assert_equal(logits[i] != 0, bb.b.tokens[i].output);
        }
        t.assert_equal((size_t) 1, bb.b.tokens[0].seq_ids.size());
        t.assert_true(bb.b.tokens[0].seq_ids.count(1) == 1);
        t.assert_equal((size_t) 2, bb.b.tokens[2].seq_ids.size());
        t.assert_true(bb.b.tokens[2].seq_ids.count(1) == 1 && bb.b.tokens[2].seq_ids.count(2) == 1);

        // round trip through the allocator
        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
        const llama_batch & batch = ba.get_batch();
        t.assert_true(batch.token != nullptr && batch.embd == nullptr);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(token[i], batch.token[i]);
            t.assert_equal(pos[i],   batch.pos[i]);
        }
        t.assert_equal(1u, ba.get_n_outputs());
    });

    t.test("defaults_for_null_fields", [&](testing & t) {
        // llama_batch_get_one: only token and n_tokens are set
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_token token[3] = { 5, 6, 7 };
        llama_batch lb = llama_batch_get_one(token, 3);

        batch_builder bb(2, &mem, 4, 1, /*n_vocab*/ 100);
        llama_batch_compat::init(bb.b, lb);

        t.assert_equal((size_t) 3, bb.b.tokens.size());
        for (int i = 0; i < 3; ++i) {
            t.assert_equal("pos continues after memory",  10 + i, bb.b.tokens[i].pos[0]);
            t.assert_equal("seq_id defaults to 0",        (size_t) 1, bb.b.tokens[i].seq_ids.size());
            t.assert_true(bb.b.tokens[i].seq_ids.count(0) == 1);
        }
        t.assert_true("only the last token is an output", !bb.b.tokens[0].output && !bb.b.tokens[1].output && bb.b.tokens[2].output);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
        t.assert_equal(10, ba.seq_pos_min(0));
        t.assert_equal(12, ba.seq_pos_max(0));
    });

    t.test("auto_pos_starts_at_zero_without_memory", [&](testing & t) {
        llama_token token[2] = { 5, 6 };
        llama_batch lb = llama_batch_get_one(token, 2);

        batch_builder bb(2, nullptr, 4, 1, /*n_vocab*/ 100);
        llama_batch_compat::init(bb.b, lb);

        t.assert_equal(0, bb.b.tokens[0].pos[0]);
        t.assert_equal(1, bb.b.tokens[1].pos[0]);
    });

    t.test("auto_pos_is_tracked_per_seq", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9}; // seq 1 is empty

        llama_token  token[4]    = { 5, 6, 7, 8 };
        int32_t      n_seq_id[4] = { 1, 1, 1, 1 };
        llama_seq_id s0[1] = { 0 };
        llama_seq_id s1[1] = { 1 };
        llama_seq_id * seq_id[5] = { s0, s1, s0, s1, nullptr };

        llama_batch lb = {};
        lb.n_tokens = 4;
        lb.token    = token;
        lb.n_seq_id = n_seq_id;
        lb.seq_id   = seq_id;

        batch_builder bb(2, &mem, 4, 1, /*n_vocab*/ 100);
        llama_batch_compat::init(bb.b, lb);

        t.assert_equal("seq 0 continues after memory", 10, bb.b.tokens[0].pos[0]);
        t.assert_equal("seq 1 starts from 0",           0, bb.b.tokens[1].pos[0]);
        t.assert_equal(11, bb.b.tokens[2].pos[0]);
        t.assert_equal( 1, bb.b.tokens[3].pos[0]);
    });

    t.test("embd_batch_with_mrope_positions", [&](testing & t) {
        const uint32_t n_pos  = 4;
        const uint32_t n_embd = 2;

        float embd[2*n_embd] = { 0, 1, 100, 101 };
        // section-major layout: pos[j*n_tokens + i]
        llama_pos pos[n_pos*2] = {
            10, 11, // temporal
             5,  6, // y
             7,  8, // x
             0,  0,
        };

        llama_batch lb = {};
        lb.n_tokens = 2;
        lb.embd     = embd;
        lb.pos      = pos;

        batch_builder bb(n_embd, nullptr, 4, n_pos);
        llama_batch_compat::init(bb.b, lb);

        t.assert_equal((size_t) 2, bb.b.tokens.size());
        t.assert_equal("batch width", (size_t) n_embd, bb.b.n_embd);
        for (int i = 0; i < 2; ++i) {
            t.assert_true(bb.b.tokens[i].has_embd);
            t.assert_equal(LLAMA_TOKEN_NULL, bb.b.tokens[i].id);
            t.assert_equal((size_t) i*n_embd, bb.b.tokens[i].embd_off);
            for (uint32_t j = 0; j < n_pos; ++j) {
                t.assert_equal(pos[j*2 + i], bb.b.tokens[i].pos[j]);
            }
        }
        t.assert_equal(100.0f, bb.b.embd[2]);
        t.assert_equal(101.0f, bb.b.embd[3]);

        llama_batch_allocr ba(n_pos);
        t.assert_true(ba.init(bb.b, vocab, false));
        llama_ubatch ub = ba.split_simple(2);
        const llama_pos expected[8] = {10, 11, 5, 6, 7, 8, 0, 0};
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(expected[i], ub.pos[i]);
        }
    });

    t.test("token_and_embd_both_set", [&](testing & t) {
        // e.g. MTP hook batches
        llama_token token[2] = { 5, 6 };
        float       embd[4]  = { 0, 1, 100, 101 };
        llama_pos   pos[2]   = { 3, 4 };

        llama_batch lb = {};
        lb.n_tokens = 2;
        lb.token    = token;
        lb.embd     = embd;
        lb.pos      = pos;

        batch_builder bb(2, nullptr, 4, 1, /*n_vocab*/ 100);
        llama_batch_compat::init(bb.b, lb);

        for (int i = 0; i < 2; ++i) {
            t.assert_equal(token[i], bb.b.tokens[i].id);
            t.assert_true(bb.b.tokens[i].has_embd);
            t.assert_equal("one position per token", pos[i], bb.b.tokens[i].pos[0]);
        }
        t.assert_equal(100.0f, bb.b.embd[2]);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
        const llama_batch & batch = ba.get_batch();
        t.assert_true("both kept", batch.token != nullptr && batch.embd != nullptr);
    });

    t.test("embd_row_width_override", [&](testing & t) {
        // encoder input (e.g. eagle3/dflash) is wider than the decoder input
        const uint32_t n_embd_enc = 6;
        float embd[2*n_embd_enc];
        for (int i = 0; i < 2*6; ++i) {
            embd[i] = (float) i;
        }

        llama_batch lb = {};
        lb.n_tokens = 2;
        lb.embd     = embd;

        batch_builder bb(2, nullptr, 4, 1, 0, n_embd_enc);
        llama_batch_compat::init(bb.b, lb, n_embd_enc);

        t.assert_equal((size_t) n_embd_enc, bb.b.n_embd);
        t.assert_equal((size_t) 2*n_embd_enc, bb.b.embd.size());
        t.assert_equal((size_t) n_embd_enc, bb.b.tokens[1].embd_off);
        t.assert_equal(6.0f, bb.b.embd[n_embd_enc]);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.b, vocab, false));
        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal("ubatch uses the encoder stride", 6.0f, ub.embd[n_embd_enc]);
    });
}

static void test_mtp_embd_width(testing & t) {
    t.test("mtp_uses_n_embd_out", [&](testing & t) {
        llama_hparams hparams = {};
        hparams.n_embd             = 64;
        hparams.n_deepstack_layers = 2;   // makes n_embd_inp() = 64 + 64*2 = 192
        hparams.n_embd_out_impl    = 96;  // makes n_embd_out() = 96

        t.assert_equal("default context uses n_embd_inp (deepstack-aware)",
                (size_t) 192, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_DEFAULT, LLM_ARCH_LLAMA, hparams));

        t.assert_equal("MTP context uses n_embd_out instead (target-model hidden state width)",
                (size_t) 96, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_MTP, LLM_ARCH_LLAMA, hparams));
    });

    t.test("mtp_falls_back_to_n_embd_when_no_override", [&](testing & t) {
        llama_hparams hparams = {};
        hparams.n_embd = 64; // no deepstack, no n_embd_out_impl override

        t.assert_equal((size_t) 64, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_DEFAULT, LLM_ARCH_LLAMA, hparams));
        t.assert_equal((size_t) 64, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_MTP, LLM_ARCH_LLAMA, hparams));
    });

    t.test("dflash_uses_n_embd_inp_enc", [&](testing & t) {
        llama_hparams hparams = {};
        hparams.n_embd              = 64;
        hparams.n_embd_inp_enc_impl = 128; // makes n_embd_inp_enc() = 128
        hparams.n_embd_out_impl     = 96;  // makes n_embd_out() = 96

        t.assert_equal("DFlash uses the encoder input width",
                (size_t) 128, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_DEFAULT, LLM_ARCH_DFLASH, hparams));

        t.assert_equal("other archs ignore n_embd_inp_enc",
                (size_t) 64, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_DEFAULT, LLM_ARCH_LLAMA, hparams));

        t.assert_equal("MTP takes precedence over DFlash",
                (size_t) 96, llama_batch_ext_select_n_embd_inp(LLAMA_CONTEXT_TYPE_MTP, LLM_ARCH_DFLASH, hparams));
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("init",           test_init);
    t.test("content_types",  test_content_types);
    t.test("compat",         test_compat);
    t.test("split",          test_split);
    t.test("keep_tail",      test_keep_tail);
    t.test("mrope",          test_mrope);
    t.test("mtp_embd_width", test_mtp_embd_width);

    return t.summary();
}
