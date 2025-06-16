#pragma once

#include "llama.h"

#include <array>
#include <vector>
#include <set>

// very similar to llama_batch,
// but has more metadata about sequences
struct llama_ubatch {
    bool equal_seqs;
    // TODO: whole_seqs for embeddings?

    uint32_t n_tokens;     // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence
    uint32_t n_seqs;

    llama_token  *  token;    // [n_tokens]
    float        *  embd;     // [n_embd, n_tokens]
    llama_pos    *  pos;      // [n_tokens]
    int32_t      *  n_seq_id; // [n_seqs]
    llama_seq_id ** seq_id;   // [n_seqs]
    int8_t       *  output;   // [n_tokens]
};

struct llama_sbatch_seq {
    int32_t n_seq_id;

    llama_seq_id * seq_id;

    size_t offset;
    size_t length;
};

// sequence-length-aware batch splitting
struct llama_sbatch {
    // tokens left in this batch
    size_t n_tokens;

    size_t n_embd;

    // sorted indices into the batch
    std::vector<int64_t> ids;
    // batch indices of the output
    std::vector<int64_t> out_ids;
    std::vector<llama_sbatch_seq> seq;

    const llama_batch * batch = nullptr;

    // buffers for the ubatches
    // TODO: very hacky, this needs a complete rework
    struct ubatch_data {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;
        std::vector<int8_t>         output;
    };

    std::vector<ubatch_data> udatas;

    llama_ubatch reserve_ubatch(size_t n_ubatch, bool has_embd = false);

    void add_seq_to_ubatch(llama_ubatch & ubatch, llama_sbatch_seq & seq, size_t length);

    // simple split, unknown number of sequences of unequal lengths
    llama_ubatch split_simple(size_t n_ubatch);

    // make batches of equal-length sequences
    llama_ubatch split_equal(size_t n_ubatch);

    // sequence-wise split
    llama_ubatch split_seq(size_t n_ubatch);

    llama_sbatch() = default;
    llama_sbatch(const llama_batch & batch, size_t n_embd, bool simple_split = false);
};

// a helper for sanitizing and fulfilling a batch
class llama_batch_allocr {
public:
    llama_batch_allocr();

    // sanitize and auto-gen missing data in the input batch
    // memory is optional. if provided will be used to check for sequence continuity and to determine the positions
    bool init(
            const llama_batch & batch_inp,
            const llama_vocab & vocab,
            const llama_memory_i * memory,
            bool embd_all);

    const llama_batch & get_batch() const;

    uint32_t get_n_outputs() const;

    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

private:
    void clear();

    llama_batch batch;

    uint32_t n_outputs;

    std::array<llama_seq_id, 1> seq_id_0 = { 0 }; // default sequence id

    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t>         output;

    std::vector<std::set<llama_pos>> seq_pos; // seq_pos[s]: the set of positions in sequence s
    std::vector<std::vector<bool>>   seq_cpl; // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

    int debug;
};
