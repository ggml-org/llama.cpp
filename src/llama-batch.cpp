#include "llama-batch.h"

#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-vocab.h"
#include "llama-memory.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <sstream>

llama_ubatch llama_sbatch::reserve_ubatch(size_t n_ubatch, bool has_embd) {
    // clear empty sequences
    // the previous ubatch is assumed to be gone,
    // so nothing should refer to values in these sequences anymore.
    for (size_t i = seq.size(); i-- > 0;) {
        if (seq[i].length == 0) {
            seq.pop_back();
        } else {
            break;
        }
    }

    udatas.push_back({});

    auto & udata = udatas.back();

    udata.token.resize(!has_embd ? n_ubatch : 0);
    udata.embd.resize(has_embd ? n_embd * n_ubatch : 0);
    udata.pos.resize(n_ubatch);
    udata.n_seq_id.resize(n_ubatch);
    udata.seq_id.resize(n_ubatch);
    udata.output.resize(n_ubatch);

    llama_ubatch ubatch = {
        /*equal_seqs   =*/ true,
        /*n_tokens     =*/ 0,
        /*n_seq_tokens =*/ 0,
        /*n_seqs       =*/ 0,
        /*token        =*/ !has_embd ? udata.token.data() : nullptr,
        /*embd         =*/ has_embd  ? udata.embd.data()  : nullptr,
        /*pos          =*/ udata.pos.data(),
        /*n_seq_id     =*/ udata.n_seq_id.data(),
        /*seq_id       =*/ udata.seq_id.data(),
        /*output       =*/ udata.output.data(),
    };

    return ubatch;
}

void llama_sbatch::add_seq_to_ubatch(llama_ubatch & ubatch, llama_sbatch_seq & seq, size_t length) {
    GGML_ASSERT(batch != nullptr);
    GGML_ASSERT(length <= seq.length);
    // Can only add sequences of equal lengths to a batch,
    // otherwise it isn't clear to which sequence a token belongs
    GGML_ASSERT(seq.n_seq_id == 0 || ubatch.n_seqs == 0 || length == (size_t) ubatch.n_tokens / ubatch.n_seqs);
    GGML_ASSERT((seq.n_seq_id != 0) == ubatch.equal_seqs);
    // NOTE: loops are separated for cache-friendliness
    if (batch->token) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.token[ubatch.n_tokens + i] = batch->token[ids[seq.offset + i]];
            }
        } else {
            // simple split
            ubatch.token = batch->token + seq.offset;
        }
    } else {
        ubatch.token = nullptr;
    }
    if (batch->embd) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                memcpy(
                        ubatch.embd + (n_embd * (ubatch.n_tokens + i)),
                        batch->embd + (n_embd * ids[seq.offset + i]),
                        n_embd * sizeof(float)
                      );
            }
        } else {
            // simple split
            ubatch.embd = batch->embd + (n_embd * seq.offset);
        }
    } else {
        ubatch.embd = nullptr;
    }
    if (ubatch.equal_seqs) {
        for (size_t i = 0; i < length; ++i) {
            ubatch.pos[ubatch.n_tokens + i] = batch->pos[ids[seq.offset + i]];
        }
    } else {
        // simple split
        ubatch.pos = batch->pos + seq.offset;
    }
    if (ubatch.equal_seqs) {
        ubatch.n_seq_id[ubatch.n_seqs] = seq.n_seq_id;
        if (seq.seq_id) {
            ubatch.seq_id[ubatch.n_seqs] = seq.seq_id;
        }
    } else {
        // simple split
        if (batch->n_seq_id) {
            ubatch.n_seq_id = batch->n_seq_id + seq.offset;
        } else {
            for (size_t i = 0; i < length; ++i) {
                ubatch.n_seq_id[ubatch.n_seqs + i] = 1;
            }
        }
        if (batch->seq_id) {
            ubatch.seq_id = batch->seq_id + seq.offset;
        }
    }
    if (batch->logits) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                size_t id = ids[seq.offset + i];
                int8_t is_output = batch->logits[id];
                ubatch.output[ubatch.n_tokens + i] = is_output;
                if (is_output) { out_ids.push_back(id); }
            }
        } else {
            // simple split
            ubatch.output = batch->logits + seq.offset;
            for (size_t i = 0; i < length; ++i) {
                if (ubatch.output[i] != 0) { out_ids.push_back(seq.offset + i); }
            }
        }
    } else {
        // only get last output
        for (size_t i = 0; i < length; ++i) {
            size_t id = ids[seq.offset + i];
            int8_t is_last = id == ids.size() - 1;
            ubatch.output[ubatch.n_tokens + i] = is_last;
            if (is_last) { out_ids.push_back(id); }
        }
    }
    if (ubatch.n_tokens == 0 && ubatch.n_seqs == 0) {
        ubatch.n_seq_tokens = ubatch.equal_seqs ? length : 1;
    }
    ubatch.n_tokens += length;
    ubatch.n_seqs += ubatch.equal_seqs ? 1 : length; // virtual sequences for simple splits
    seq.offset += length;
    seq.length -= length;
    n_tokens -= length;
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);
}

llama_ubatch llama_sbatch::split_simple(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    ubatch.equal_seqs = false;
    if (!seq.empty()) {
        llama_sbatch_seq & s = seq[0];
        size_t length = s.length < n_ubatch ? s.length : n_ubatch;
        GGML_ASSERT(seq.size() == 1 && s.n_seq_id == 0); // don't mix with other splits
        add_seq_to_ubatch(ubatch, s, length);
    }
    return ubatch;
}

llama_ubatch llama_sbatch::split_equal(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    if (!seq.empty()) {
        size_t length = 0;
        size_t n_tokens_in_ubatch = 0;
        GGML_ASSERT(seq[0].n_seq_id > 0); // should not be mixed with simple splits
                                          // smallest first, because it's easier to split this way;
                                          // starting from the end to pop in constant time.
        for (size_t i = seq.size(); i-- > 0;) {
            llama_sbatch_seq & s = seq[i];
            GGML_ASSERT(s.length > 0);
            if (length == 0) {
                length = s.length < n_ubatch ? s.length : n_ubatch;
            }
            add_seq_to_ubatch(ubatch, s, length);
            n_tokens_in_ubatch += length;
            // shared prompts can't be mixed with any of their sequences,
            // so it's safer to compute them in their own ubatch
            if (s.n_seq_id > 1) { break; }
            // stop when there isn't enough space for another sequence
            if (length + n_tokens_in_ubatch > n_ubatch) { break; }
        }
    }
    return ubatch;
}

llama_ubatch llama_sbatch::split_seq(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    if (!seq.empty()) {
        llama_sbatch_seq & s = seq[seq.size() - 1];
        size_t length = s.length < n_ubatch ? s.length : n_ubatch;
        GGML_ASSERT(s.n_seq_id > 0); // should not be mixed with simple splits
        add_seq_to_ubatch(ubatch, s, length);
    }
    return ubatch;
}

llama_sbatch::llama_sbatch(const llama_batch & batch, size_t n_embd, bool simple_split) {
    GGML_ASSERT(batch.n_tokens >= 0);
    this->batch = &batch;
    this->n_embd = n_embd;

    n_tokens = batch.n_tokens;
    ids.resize(n_tokens);
    out_ids.clear();
    // TODO: reserve out_ids and seq

    for (size_t i = 0; i < n_tokens; ++i) {
        ids[i] = i;
    }

    if (simple_split) {
        seq.resize(1);
        llama_sbatch_seq & s = seq[0];
        s.n_seq_id = 0;
        s.seq_id = nullptr;
        s.offset = 0;
        s.length = n_tokens;
        return;
    }

    std::sort(ids.begin(), ids.end(),
            [&batch](size_t a, size_t b) {
                int32_t n_seq_a = batch.n_seq_id ? batch.n_seq_id[a] : 1;
                int32_t n_seq_b = batch.n_seq_id ? batch.n_seq_id[b] : 1;
                // sort by seq_id, then by pos
                if (n_seq_a == n_seq_b) {
                    if (batch.seq_id) {
                        for (int32_t i = 0; i < n_seq_a; ++i) {
                            llama_seq_id seq_id_a = batch.seq_id[a][i];
                            llama_seq_id seq_id_b = batch.seq_id[b][i];
                            // smaller seq_ids go first
                            if (seq_id_a != seq_id_b) {
                                return seq_id_a < seq_id_b;
                            }
                        }
                    }
                    // when all else is equal, sort by pos
                    if (batch.pos) {
                        return batch.pos[a] < batch.pos[b];
                    }
                    // no pos, sort by id
                    return a < b;
                }
                // shared prompts go first
                return n_seq_a > n_seq_b;
            }
    );

    // init seq
    llama_sbatch_seq * last_seq = nullptr;

    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t bi = ids[i];
        const int32_t n_seqs = batch.n_seq_id[bi];
        llama_seq_id * seq_ids = batch.seq_id[bi];
        if (last_seq != nullptr) {
            bool same = n_seqs == last_seq->n_seq_id;
            for (int32_t j = 0; same && j < n_seqs; ++j) {
                if (seq_ids[j] != last_seq->seq_id[j]) {
                    same = false;
                }
            }
            if (same) {
                last_seq->length += 1;
                continue;
            }
        }
        llama_sbatch_seq new_seq = {n_seqs, seq_ids, i, 1};
        seq.push_back(new_seq);
        last_seq = &seq.back();
    }

    // keep shared prompts first at the end, then sort by length descending.
    std::sort(seq.begin(), seq.end(),
            [](llama_sbatch_seq & a, llama_sbatch_seq & b) {
                if (a.n_seq_id == b.n_seq_id) {
                    return a.length > b.length;
                }
                return a.n_seq_id < b.n_seq_id;
            }
            );
}

llama_batch_allocr::llama_batch_allocr() {
    const char * LLAMA_BATCH_DEBUG = getenv("LLAMA_BATCH_DEBUG");
    debug = LLAMA_BATCH_DEBUG ? atoi(LLAMA_BATCH_DEBUG) : 0;

    seq_pos.resize(LLAMA_MAX_SEQ);
    seq_cpl.resize(LLAMA_MAX_SEQ);
    for (auto & cur : seq_cpl) {
        cur.resize(LLAMA_MAX_SEQ);
    }
}

bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        bool embd_all) {
    clear();

    batch = batch_inp;

    GGML_ASSERT(batch.n_tokens > 0);

    //
    // validate input batch
    //

    if (batch.token) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return false;
            }
        }
    }

    if (batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                if (batch.seq_id && (batch.seq_id[i][s] < 0 || batch.seq_id[i][s] >= LLAMA_MAX_SEQ)) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d > %d\n", __func__, i, s, batch.seq_id[i][s], LLAMA_MAX_SEQ);
                    return false;
                }
            }
        }
    }

    //
    // auto-generate missing fields
    //

    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }

    if (!batch.pos) {
        pos.resize(batch.n_tokens);

        // initialize the starting position for each sequence based on the positions in the memory
        llama_pos p0[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (!memory) {
                p0[s] = 0;
            } else {
                p0[s] = memory->seq_pos_max(s) + 1;
            }
        }

        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            pos[i] = p0[seq_id];

            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                p0[batch.seq_id[i][s]] = pos[i] + 1;
            }
        }

        batch.pos = pos.data();
    }

    if (!batch.logits) {
        if (embd_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (embd_all) {
        bool warn = false;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                warn = true;
            }
        }

        if (warn) {
            LLAMA_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", __func__);

            output.resize(batch.n_tokens, true);
            batch.logits = output.data();
        }
    }

    //
    // compute stats
    //

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            seq_pos[batch.seq_id[i][s]].insert(batch.pos[i]);

            if (s > 0) {
                const llama_seq_id s0 = batch.seq_id[i][0];
                const llama_seq_id s1 = batch.seq_id[i][s];

                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;

                // note: the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;
            }
        }
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: input batch info:\n", __func__);
        LLAMA_LOG_DEBUG("%s:   n_tokens  = %d\n", __func__,          batch.n_tokens);
        LLAMA_LOG_DEBUG("%s:   token     = %p\n", __func__, (void *) batch.token);
        LLAMA_LOG_DEBUG("%s:   embd      = %p\n", __func__, (void *) batch.embd);
        LLAMA_LOG_DEBUG("%s:   pos       = %p\n", __func__, (void *) batch.pos);
        LLAMA_LOG_DEBUG("%s:   n_seq_id  = %p\n", __func__, (void *) batch.n_seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id    = %p\n", __func__, (void *) batch.seq_id);
        LLAMA_LOG_DEBUG("%s:   logits    = %p\n", __func__, (void *) batch.logits);
        LLAMA_LOG_DEBUG("%s:   n_outputs = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, batch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLAMA_LOG_DEBUG("%s:   token     = [\n", __func__);
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                    seq_id[batch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                LLAMA_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                        __func__, i, batch.token[i], vocab.token_to_piece(batch.token[i]).c_str(),
                        batch.pos[i], batch.n_seq_id[i], ss.str().c_str(), batch.logits[i]);
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);

            LLAMA_LOG_DEBUG("%s:   seq       = [\n", __func__);
            for (int s0 = 0; s0 < (int) seq_pos.size(); ++s0) {
                if (seq_pos[s0].empty()) {
                    continue;
                }

                std::stringstream ss;
                for (int s1 = 0; s1 < (int) seq_cpl[s0].size(); ++s1) {
                    if (seq_cpl[s0][s1]) {
                        ss << s1 << " ";
                    }
                }

                LLAMA_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n",
                        __func__, s0, seq_pos_min(s0), seq_pos_max(s0), ss.str().empty() ? "-" : ss.str().c_str());
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
        }
    }

    //
    // consistency checks
    //

    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos[s].empty()) {
            continue;
        }

        if (memory && seq_pos_min(s) != memory->seq_pos_max(s) + 1) {
            LLAMA_LOG_ERROR("%s: sequence %d does not start from the last position stored in the memory\n", __func__, s);
            return false;
        }

        if (seq_pos_max(s) - seq_pos_min(s) + 1 > (int) seq_pos[s].size()) {
            LLAMA_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, s);
            return false;
        }
    }

    if (memory) {
        for (int32_t s0 = 0; s0 < LLAMA_MAX_SEQ; ++s0) {
            for (int32_t s1 = 0; s1 < LLAMA_MAX_SEQ; ++s1) {
                if (seq_cpl[s0][s1]) {
                    if (memory->seq_pos_min(s0) != memory->seq_pos_min(s1) ||
                        memory->seq_pos_max(s0) != memory->seq_pos_max(s1)) {
                        LLAMA_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, s0, s1);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

const llama_batch & llama_batch_allocr::get_batch() const {
    return batch;
}

uint32_t llama_batch_allocr::get_n_outputs() const {
    return n_outputs;
}

llama_pos llama_batch_allocr::seq_pos_min(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].begin();
}

llama_pos llama_batch_allocr::seq_pos_max(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].rbegin();
}

void llama_batch_allocr::clear() {
    n_outputs = 0;

    batch = {};
    pos.clear();
    n_seq_id.clear();
    seq_id.clear();
    output.clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }
}

//
// interface implementation
//

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {
        /*n_tokens       =*/ 0,
        /*tokens         =*/ nullptr,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
