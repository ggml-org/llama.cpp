#include "llama-kv-cache-dsv4.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;

static constexpr uint32_t DSV4_STATE_MAGIC         = 0x34565344; // DSV4
static constexpr uint32_t DSV4_STATE_VERSION       = 1;
static constexpr uint32_t DSV4_STATE_MODE_FULL     = 0;
static constexpr uint32_t DSV4_STATE_MODE_PARTIAL  = 1;
static constexpr uint32_t DSV4_K_CACHE_STATE_VER   = 1;
static constexpr uint32_t DSV4_COMP_STATE_VER      = 1;

static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static int64_t dsv4_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("DSV4 sequence id out of stream range");
    }

    return (int64_t) seq_id*size;
}

static bool dsv4_ubatch_has_coupled(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static bool dsv4_token_has_seq(const llama_ubatch & ubatch, uint32_t i, llama_seq_id seq_id) {
    for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
        if (ubatch.seq_id[i][s] == seq_id) {
            return true;
        }
    }

    return false;
}

static llama_ubatch dsv4_build_raw_write_ubatch(const llama_ubatch & ubatch) {
    if (!dsv4_ubatch_has_coupled(ubatch)) {
        return ubatch;
    }
    if (ubatch.embd) {
        throw std::runtime_error("DSV4 coupled embedding ubatches are not supported");
    }

    std::vector<uint32_t> counts(ubatch.n_seqs_unq, 0);
    uint32_t n_tokens = 0;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                ++counts[s];
                ++n_tokens;
            }
        }
    }

    if (n_tokens == 0) {
        return ubatch;
    }

    const uint32_t n_seq_tokens = counts[0];
    for (uint32_t s = 1; s < counts.size(); ++s) {
        if (counts[s] != n_seq_tokens) {
            throw std::runtime_error("DSV4 coupled raw writes require equal sequence lengths");
        }
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize((size_t) n_tokens*ubatch.n_pos);
    data->n_seq_id.reserve(n_tokens);
    data->seq_id.reserve(n_tokens);
    data->seq_id_data.reserve(n_tokens);
    data->seq_id_unq.assign(ubatch.seq_id_unq, ubatch.seq_id_unq + ubatch.n_seqs_unq);
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->output.assign(n_tokens, 0);
    if (ubatch.token) {
        data->token.reserve(n_tokens);
    }

    for (uint32_t s = 0; s < data->seq_id_unq.size(); ++s) {
        data->seq_idx[data->seq_id_unq[s]] = s;
    }

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (!dsv4_token_has_seq(ubatch, i, seq_id)) {
                continue;
            }

            const uint32_t dst = data->n_seq_id.size();
            if (ubatch.token) {
                data->token.push_back(ubatch.token[i]);
            }
            for (uint32_t p = 0; p < ubatch.n_pos; ++p) {
                data->pos[(size_t) p*n_tokens + dst] = ubatch.pos[(size_t) p*ubatch.n_tokens + i];
            }
            data->n_seq_id.push_back(1);
            data->seq_id_data.push_back(seq_id);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->seq_id.push_back(&data->seq_id_data[i]);
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ ubatch.n_seqs_unq,
        /*.n_seqs_unq   =*/ ubatch.n_seqs_unq,
        /*.n_pos        =*/ ubatch.n_pos,
        /*.token        =*/ data->token.empty() ? nullptr : data->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ data->pos.data(),
        /*.n_seq_id     =*/ data->n_seq_id.data(),
        /*.seq_id       =*/ data->seq_id.data(),
        /*.seq_id_unq   =*/ data->seq_id_unq.data(),
        /*.seq_idx      =*/ data->seq_idx.data(),
        /*.output       =*/ data->output.data(),
        /*.data         =*/ data,
    };

    return res;
}

static std::vector<llama_ubatch> dsv4_build_raw_write_ubatches(const std::vector<llama_ubatch> & ubatches) {
    std::vector<llama_ubatch> res;
    res.reserve(ubatches.size());
    for (const llama_ubatch & ubatch : ubatches) {
        res.push_back(dsv4_build_raw_write_ubatch(ubatch));
    }
    return res;
}

static bool dsv4_batch_has_coupled(const llama_batch & batch) {
    if (!batch.n_seq_id) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static int64_t dsv4_comp_graph_n_stream(const llama_ubatch & ubatch, uint32_t n_stream) {
    // Coupled sequence sets must stay in one graph stream because their
    // compressed state is shared. Independent per-seq state can fan out.
    if (n_stream <= 1 || ubatch.n_seqs_unq <= 1 || dsv4_ubatch_has_coupled(ubatch)) {
        return 1;
    }

    return ubatch.n_seqs_unq;
}

static void dsv4_state_src_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t     & s0,
        uint32_t     & ns) {
    if (seq_id >= 0 && n_stream > 1) {
        if ((uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = (uint32_t) seq_id;
        ns = 1;
        return;
    }

    s0 = 0;
    ns = seq_id >= 0 ? 1 : n_stream;
}

static void dsv4_state_dst_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t       ns,
        uint32_t     & s0) {
    if (seq_id >= 0) {
        if (ns != 1) {
            throw std::runtime_error("DSV4 sequence state stream count mismatch");
        }
        if (n_stream > 1 && (uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = n_stream > 1 ? (uint32_t) seq_id : 0;
        return;
    }

    if (ns != n_stream) {
        throw std::runtime_error("DSV4 full state stream count mismatch");
    }

    s0 = 0;
}

static void dsv4_state_write_tensor_streams(
        llama_io_write_i & io,
        ggml_tensor      * tensor,
        uint32_t           n_rows,
        uint32_t           s0,
        uint32_t           ns) {
    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    io.write(&type_i,   sizeof(type_i));
    io.write(&ne0,      sizeof(ne0));
    io.write(&rows,     sizeof(rows));
    io.write(&row_size, sizeof(row_size));

    const size_t offset = (size_t) s0*n_rows*row_size;
    const size_t size   = (size_t) ns*n_rows*row_size;

    io.write_tensor(tensor, offset, size);
}

static void dsv4_state_read_tensor_streams(
        llama_io_read_i & io,
        ggml_tensor     * tensor,
        uint32_t          n_rows,
        uint32_t          s0,
        uint32_t          ns) {
    int32_t  type_i_ref;
    uint64_t ne0_ref;
    uint64_t rows_ref;
    uint64_t row_size_ref;

    io.read(&type_i_ref,   sizeof(type_i_ref));
    io.read(&ne0_ref,      sizeof(ne0_ref));
    io.read(&rows_ref,     sizeof(rows_ref));
    io.read(&row_size_ref, sizeof(row_size_ref));

    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (type_i != type_i_ref || ne0 != ne0_ref || rows != rows_ref || row_size != row_size_ref) {
        throw std::runtime_error("DSV4 state tensor metadata mismatch");
    }

    const size_t offset = (size_t) s0*n_rows*row_size;
    const size_t size   = (size_t) ns*n_rows*row_size;

    io.read_tensor(tensor, offset, size);
}

static void dsv4_state_write_k_cache(
        llama_io_write_i    & io,
        const llama_kv_cache * kv,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(kv->get_n_stream(), seq_id, s0, ns);

    const uint32_t version = DSV4_K_CACHE_STATE_VER;
    const uint32_t kv_size = kv->get_size();
    const auto layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = layer_ids.size();

    io.write(&version, sizeof(version));
    io.write(&kv_size, sizeof(kv_size));
    io.write(&ns,      sizeof(ns));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il : layer_ids) {
        io.write(&il, sizeof(il));
        dsv4_state_write_tensor_streams(io, kv->get_k_storage(il), kv_size, s0, ns);
    }
}

static void dsv4_state_read_k_cache(
        llama_io_read_i  & io,
        llama_kv_cache   * kv,
        llama_seq_id       seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t kv_size_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,     sizeof(version));
    io.read(&kv_size_ref, sizeof(kv_size_ref));
    io.read(&ns,          sizeof(ns));
    io.read(&n_layer_ref, sizeof(n_layer_ref));

    if (version != DSV4_K_CACHE_STATE_VER) {
        throw std::runtime_error("DSV4 K-cache state version mismatch");
    }
    if (kv_size_ref != kv->get_size()) {
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(kv->get_n_stream(), seq_id, ns, s0);

    const auto layer_ids = kv->get_layer_ids();
    if (n_layer_ref != layer_ids.size()) {
        throw std::runtime_error("DSV4 K-cache layer count mismatch");
    }

    for (uint32_t il : layer_ids) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != il) {
            throw std::runtime_error("DSV4 K-cache layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, kv->get_k_storage(il), kv->get_size(), s0, ns);
    }
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);

    // n_stream is the persistent cache/state layout; plan.n_stream is the
    // graph view for this ubatch and can be a subset of those streams.
    if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
        throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
    }

    const int64_t state_rows = (int64_t) state_size*n_stream;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
    // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
    // the "previous-window" gather indices for every block, followed by the
    // "current-window" indices for every block. Collect them separately here and
    // append cur after prev once the loop has visited all completed blocks
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            curr_token_idx_map[std::make_pair(ubatch.seq_id[i][s], ubatch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return (int32_t) (state_rows + ubatch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
        return (int32_t) (stream_off + pos%state_size);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i];

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const int32_t state_idx = (int32_t) (stream_off + pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);

            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_pos.push_back((int32_t) source_start);

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    if (ratio == DSV4_CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty()) {
        // Non-boundary CSA steps still need a write op so their graph matches
        // boundary steps. Use a padded scratch row that is masked from attention.
        assert(kv_size > 0);

        uint32_t i = 0;
        while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
            ++i;
        }
        assert(i < ubatch.n_tokens);

        const llama_pos    pos    = ubatch.pos[i];
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
        const int32_t source_idx = state_source_idx(seq_id, pos);

        plan.state_write_idxs.push_back(cache_off + kv_size - 1);
        plan.state_write_pos .push_back(0);

        if (overlap) {
            for (uint32_t j = 0; j < ratio; ++j) {
                overlap_prev_reads.push_back(source_idx);
                overlap_cur_reads .push_back(source_idx);
            }
        } else {
            for (uint32_t j = 0; j < ratio; ++j) {
                plan.state_read_idxs.push_back(source_idx);
            }
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    plan.n_kv = GGML_PAD(plan.n_kv, 256u);

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) {
                return a.dst < b.dst;
            });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }

    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    if (debug) {
        LLAMA_LOG_INFO("%s: ratio=%u, n_tokens=%u, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, ratio, ubatch.n_tokens,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }

    return plan;
}

static std::vector<llama_kv_cache_dsv4_context::comp_plan> dsv4_build_comp_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    std::vector<llama_kv_cache_dsv4_context::comp_plan> plans;
    plans.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(ubatch, ratio, overlap, state_size, kv_size, n_stream));
    }

    return plans;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_comp_sinfos(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_stream) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
            throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
        }

        const uint32_t ns = (uint32_t) dsv4_comp_graph_n_stream(ubatch, n_stream);
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = n_stream > 1 ? LLAMA_MAX_SEQ : 0;
        sinfo.s1 = 0;
        sinfo.resize(ns);

        for (uint32_t s = 0; s < ns; ++s) {
            const llama_seq_id seq_id = n_stream > 1 ? ubatch.seq_id_unq[s] : 0;
            const uint32_t strm = (uint32_t) dsv4_stream_offset(n_stream, seq_id, 1);

            sinfo.s0 = std::min(sinfo.s0, strm);
            sinfo.s1 = std::max(sinfo.s1, strm);
            sinfo.strm[s] = strm;
            sinfo.idxs[s].resize(1, 0);
        }

        if (n_stream > 1 && sinfo.s1 - sinfo.s0 + 1 != ns) {
            throw std::runtime_error("DSV4 compressed streams are not contiguous in ubatch");
        }

        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_raw_read_sinfos(
        const llama_kv_cache::slot_info_vec_t & sinfos_write,
        const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (size_t i = 0; i < ubatches.size(); ++i) {
        const llama_ubatch & ubatch = ubatches[i];
        const auto & sinfo_write = sinfos_write[i];

        if (!dsv4_ubatch_has_coupled(ubatch)) {
            sinfos.push_back(sinfo_write);
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[0][0];
        uint32_t i_stream = 0;
        for (; i_stream < sinfo_write.n_stream(); ++i_stream) {
            if (sinfo_write.strm[i_stream] == seq_id) {
                break;
            }
        }
        if (i_stream == sinfo_write.n_stream()) {
            throw std::runtime_error("DSV4 raw write stream not found for coupled read");
        }

        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = sinfo_write.strm[i_stream];
        sinfo.s1 = sinfo_write.strm[i_stream];
        sinfo.resize(1);
        sinfo.strm[0] = sinfo_write.strm[i_stream];
        sinfo.idxs[0] = sinfo_write.idxs[i_stream];
        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_reserve_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);
    plan.n_kv = kv_size;

    if (ubatch.n_tokens == 0) {
        return plan;
    }

    const uint32_t n_seqs       = std::max<uint32_t>(1, ubatch.n_seqs);
    const uint32_t n_seq_tokens = std::max<uint32_t>(1, ubatch.n_seq_tokens);
    const uint64_t n_blocks_u64 = (uint64_t) n_seqs*((n_seq_tokens + ratio - 1)/ratio);
    const size_t n_blocks = (size_t) std::max<uint64_t>(1, n_blocks_u64);
    GGML_ASSERT((uint64_t) n_blocks == std::max<uint64_t>(1, n_blocks_u64));

    const uint64_t state_rows = (uint64_t) state_size*n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>(ubatch.n_tokens, state_rows);

    plan.state_pos .resize(ubatch.n_tokens);
    plan.state_persist_src_idxs.resize(n_persist);
    plan.state_persist_dst_idxs.resize(n_persist);
    plan.state_read_idxs .resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_pos .resize(n_blocks);

    return plan;
}

static void dsv4_make_k_only(llama_hparams & hparams) {
    // llama_kv_cache uses hparams.is_mla() to allocate K-only storage.
    hparams.n_embd_head_k_mla_impl = hparams.n_embd_head_k();
    hparams.n_embd_head_v_mla_impl = hparams.n_embd_head_k();
}

//
// llama_dsv4_comp_state
//

llama_dsv4_comp_state::llama_dsv4_comp_state(
        const llama_model & model,
                bool        offload,
                bool        unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
        const char    * name,
        const llama_memory_i::layer_filter_cb & filter) :
    ratio(ratio),
    state_size(state_size),
    n_embd_state(n_embd_state),
    n_stream(unified ? 1 : n_seq_max) {
    const llama_hparams & hparams = model.hparams;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*hparams.n_layer()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DSV4 compressor state");
        }

        ggml_tensor * kv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_stream);
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_stream);

        ggml_format_name(kv,    "dsv4_%s_state_kv_l%d",    name, il);
        ggml_format_name(score, "dsv4_%s_state_score_l%d", name, il);

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, kv, score });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for DSV4 compressor state");
        }

        ggml_backend_buffer_clear(buf, 0);

        LLAMA_LOG_INFO("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), name, ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, layers = %zu, size = %7.2f MiB\n",
            __func__, name, ratio, state_size, n_embd_state, n_stream, layers.size(), total_size()/1024.0/1024.0);
}

void llama_dsv4_comp_state::clear(bool data) {
    if (!data) {
        return;
    }

    for (auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

uint32_t llama_dsv4_comp_state::get_ratio() const {
    return ratio;
}

uint32_t llama_dsv4_comp_state::get_state_size() const {
    return state_size;
}

uint32_t llama_dsv4_comp_state::get_n_stream() const {
    return n_stream;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_dsv4_comp_state::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ret[buft] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

void llama_dsv4_comp_state::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(n_stream, seq_id, s0, ns);

    const uint32_t version      = DSV4_COMP_STATE_VER;
    const uint32_t n_layer      = layers.size();

    io.write(&version,      sizeof(version));
    io.write(&ratio,        sizeof(ratio));
    io.write(&state_size,   sizeof(state_size));
    io.write(&n_embd_state, sizeof(n_embd_state));
    io.write(&ns,           sizeof(ns));
    io.write(&n_layer,      sizeof(n_layer));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));

        dsv4_state_write_tensor_streams(io, layer.kv,    state_size, s0, ns);
        dsv4_state_write_tensor_streams(io, layer.score, state_size, s0, ns);
    }
}

void llama_dsv4_comp_state::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t ratio_ref;
    uint32_t state_size_ref;
    uint32_t n_embd_state_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,          sizeof(version));
    io.read(&ratio_ref,        sizeof(ratio_ref));
    io.read(&state_size_ref,   sizeof(state_size_ref));
    io.read(&n_embd_state_ref, sizeof(n_embd_state_ref));
    io.read(&ns,               sizeof(ns));
    io.read(&n_layer_ref,      sizeof(n_layer_ref));

    if (version != DSV4_COMP_STATE_VER) {
        throw std::runtime_error("DSV4 compressor state version mismatch");
    }
    if (ratio_ref != ratio || state_size_ref != state_size || n_embd_state_ref != n_embd_state) {
        throw std::runtime_error("DSV4 compressor state metadata mismatch");
    }
    if (n_layer_ref != layers.size()) {
        throw std::runtime_error("DSV4 compressor state layer count mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(n_stream, seq_id, ns, s0);

    for (const auto & layer : layers) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != layer.il) {
            throw std::runtime_error("DSV4 compressor state layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, layer.kv,    state_size, s0, ns);
        dsv4_state_read_tensor_streams(io, layer.score, state_size, s0, ns);
    }
}

ggml_tensor * llama_dsv4_comp_state::get_kv(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);

    ggml_tensor * state = layers[ids].kv;

    return ggml_reshape_2d(ctx, state, state->ne[0], state->ne[1]*state->ne[2]);
}

ggml_tensor * llama_dsv4_comp_state::get_score(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);

    ggml_tensor * state = layers[ids].score;

    return ggml_reshape_2d(ctx, state, state->ne[0], state->ne[1]*state->ne[2]);
}

ggml_tensor * llama_dsv4_comp_state::cpy_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_kv(ctx, il), cur, idxs);
}

ggml_tensor * llama_dsv4_comp_state::cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_score(ctx, il), cur, idxs);
}

size_t llama_dsv4_comp_state::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

//
// llama_kv_cache_dsv4
//

llama_kv_cache_dsv4::llama_kv_cache_dsv4(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    hparams_raw(model.hparams),
    hparams_csa(model.hparams),
    hparams_hca(model.hparams),
    hparams_lid(model.hparams),
    n_seq_max(n_seq_max) {

    const layer_filter_cb filter_raw = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return true;
    };

    GGML_UNUSED(unified);

    // Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
    const bool unified_raw = false;

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    dsv4_make_k_only(hparams_raw);

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr);

    dsv4_make_k_only(hparams_csa);
    dsv4_make_k_only(hparams_hca);

    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_k_swa  = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_swa  = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
    dsv4_make_k_only(hparams_lid);

    const layer_filter_cb filter_csa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_hca = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    const bool unified_compressed = false;

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_csa = std::make_unique<llama_kv_cache>(
            model, hparams_csa, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_HCA_RATIO));

    kv_hca = std::make_unique<llama_kv_cache>(
            model, hparams_hca, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_hca, nullptr, nullptr);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr);

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressor state\n", __func__);

    csa_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.n_embd_head_k(), "csa", filter_csa);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressor state\n", __func__);

    hca_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_HCA_RATIO, DSV4_HCA_RATIO,
            model.hparams.n_embd_head_k(), "hca", filter_hca);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer compressor state\n", __func__);

    lid_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.indexer_head_size, "lid", filter_csa);

    // DSV4 attention reads compressed-K / compressor-state rows that the current
    // graph does not necessarily overwrite; uninitialized buffer contents would
    // otherwise leak in (instance-specific garbage) and corrupt recall. Zero all
    // compressed buffers up front so reads of un-written rows are deterministic.
    clear_compressed(true);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    const bool raw_per_seq  = kv_raw->get_base()->get_n_stream() != 1;
    const bool comp_per_seq = csa_state->get_n_stream() > 1;
    const bool has_coupled = dsv4_batch_has_coupled(balloc.get_batch());

    const auto make_context = [&](std::vector<llama_ubatch> ubatches) -> llama_memory_context_ptr {
        auto ubatches_raw = dsv4_build_raw_write_ubatches(ubatches);

        auto sinfos_raw_base_write = kv_raw->get_base()->prepare(ubatches_raw);
        if (sinfos_raw_base_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_write = kv_raw->get_swa()->prepare(ubatches_raw);
        if (sinfos_raw_swa_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_read = dsv4_build_raw_read_sinfos(sinfos_raw_swa_write, ubatches);

        return std::make_unique<llama_kv_cache_dsv4_context>(
                this,
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                std::move(ubatches),
                std::move(ubatches_raw));
    };

    // Match llama_kv_cache_iswa splitting when DSV4 compressed state does not
    // require per-sequence graph layout.
    do {
        if (raw_per_seq || comp_per_seq) {
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    // When raw or compressed state is per-sequence, independent sequences can
    // share an equal-length ubatch. Coupled sequence sets still serialize until
    // DSV4 has explicit shared-state handling for compressed streams.
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (has_coupled) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                ubatch = balloc.split_equal(n_ubatch, raw_per_seq || comp_per_seq, 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_full() {
    return std::make_unique<llama_kv_cache_dsv4_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv4_context>(this, lctx, optimize);
}

bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}

void llama_kv_cache_dsv4::clear(bool data) {
    kv_raw->clear(data);
    clear_compressed(true); // DSV4 compressed buffers must never expose stale/uninit rows
}

bool llama_kv_cache_dsv4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        return false;
    }

    if (p0 > 0) {
        // DSV4 compressed cache rows are derived from running compressor state,
        // so arbitrary rollback is not reconstructible from the raw cache alone.
        // Allow the common prompt-cache cleanup no-op: remove [end, infinity).
        if (seq_id >= 0 && p0 > kv_raw->seq_pos_max(seq_id)) {
            return true;
        }

        return false;
    }

    const bool res = kv_raw->seq_rm(seq_id, p0, p1);

    if (res) {
        clear_compressed(true);
    }

    return res;
}

void llama_kv_cache_dsv4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_raw->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    clear_compressed(true);
}

void llama_kv_cache_dsv4::seq_keep(llama_seq_id seq_id) {
    kv_raw->seq_keep(seq_id);
    clear_compressed(true);
}

void llama_kv_cache_dsv4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_raw->seq_add(seq_id, p0, p1, shift);
    clear_compressed(true);
}

void llama_kv_cache_dsv4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_raw->seq_div(seq_id, p0, p1, d);
    clear_compressed(true);
}

llama_pos llama_kv_cache_dsv4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    // The raw SWA cache may contain a wider window, but the compressed DSV4
    // state cannot be rolled back within that window. Report only the current
    // boundary so server-context uses checkpoints for rollback.
    return kv_raw->seq_pos_max(seq_id);
}

llama_pos llama_kv_cache_dsv4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return kv_raw->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv4::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_raw->memory_breakdown();
    for (const auto & buft_size : kv_csa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_hca->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : csa_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : hca_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : lid_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_kv_cache_dsv4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool partial_only = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    const uint32_t magic   = DSV4_STATE_MAGIC;
    const uint32_t version = DSV4_STATE_VERSION;
    const uint32_t mode    = partial_only ? DSV4_STATE_MODE_PARTIAL : DSV4_STATE_MODE_FULL;

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&mode,    sizeof(mode));

    kv_raw->state_write(io, seq_id, flags);

    if (!partial_only) {
        dsv4_state_write_k_cache(io, kv_csa.get(), seq_id, flags);
        dsv4_state_write_k_cache(io, kv_hca.get(), seq_id, flags);
        dsv4_state_write_k_cache(io, kv_lid.get(), seq_id, flags);
    }

    csa_state->state_write(io, seq_id, flags);
    hca_state->state_write(io, seq_id, flags);
    lid_state->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsv4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    uint32_t magic;
    uint32_t version;
    uint32_t mode = DSV4_STATE_MODE_FULL;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));

    if (magic != DSV4_STATE_MAGIC) {
        throw std::runtime_error("DSV4 state magic mismatch");
    }
    if (version != DSV4_STATE_VERSION) {
        throw std::runtime_error("DSV4 state version mismatch");
    }

    io.read(&mode, sizeof(mode));
    if (mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL) {
        throw std::runtime_error("DSV4 state mode mismatch");
    }

    const bool partial_only = mode == DSV4_STATE_MODE_PARTIAL;
    if (partial_only != !!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("DSV4 state flags mismatch");
    }

    kv_raw->state_read(io, seq_id, flags);

    if (!partial_only) {
        dsv4_state_read_k_cache(io, kv_csa.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_hca.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_lid.get(), seq_id, flags);
    }

    csa_state->state_read(io, seq_id, flags);
    hca_state->state_read(io, seq_id, flags);
    lid_state->state_read(io, seq_id, flags);

}

llama_kv_cache_iswa * llama_kv_cache_dsv4::get_raw() const {
    return kv_raw.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_csa() const {
    return kv_csa.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_hca() const {
    return kv_hca.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_lid() const {
    return kv_lid.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_csa_state() const {
    return csa_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_hca_state() const {
    return hca_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_lid_state() const {
    return lid_state.get();
}

void llama_kv_cache_dsv4::clear_compressed(bool data) {
    kv_csa->clear(data);
    kv_hca->clear(data);
    kv_lid->clear(data);
    csa_state->clear(data);
    hca_state->clear(data);
    lid_state->clear(data);
}

//
// llama_kv_cache_dsv4_raw_context
//

static llama_kv_cache::slot_info dsv4_build_full_sinfo(const llama_kv_cache * kv) {
    const uint32_t n_stream = kv->get_n_stream();

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = n_stream - 1;
    sinfo.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfo.strm[s] = s;
        sinfo.idxs[s].resize(1, 0);
    }

    return sinfo;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(nullptr),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
    sinfos_read.push_back(dsv4_build_full_sinfo(kv_swa));
    sinfos_write = sinfos_read;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa_mem(kv->get_swa()->init_update(lctx, optimize)),
    n_kv(kv_swa->get_size()),
    status(llama_memory_status_combine(ctx_base_mem->get_status(), ctx_swa_mem->get_status())) {
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base_write,
        slot_info_vec_t sinfos_swa_write,
        slot_info_vec_t sinfos_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_write) :
    kv_swa(kv->get_swa()),
    sinfos_write(std::move(sinfos_swa_write)),
    sinfos_read(std::move(sinfos_swa_read)),
    ubatches(std::move(ubatches)),
    ubatches_write(std::move(ubatches_write)),
    ctx_base_mem(std::make_unique<llama_kv_cache_context>(
                kv->get_base(), std::move(sinfos_base_write), this->ubatches_write)),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_kv_cache_dsv4_raw_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (ctx_base_mem) {
        ctx_base_mem->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_raw_context::apply() {
    bool res = true;

    if (ctx_base_mem) {
        res = res & ctx_base_mem->apply();
    }
    if (ctx_swa_mem) {
        res = res & ctx_swa_mem->apply();
    }
    if (!ubatches_write.empty()) {
        kv_swa->apply_ubatch(sinfos_write[i_next], ubatches_write[i_next]);
        n_kv = kv_swa->get_n_kv(sinfos_read[i_next]);
    }

    return res;
}

llama_memory_status llama_kv_cache_dsv4_raw_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_raw_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_write() const {
    if (ubatches_write.empty()) {
        return 0;
    }

    return ubatches_write[i_next].n_tokens;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv_swa->get_k(ctx, il, n_kv, sinfos_read[i_next]);
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const auto & sinfo = sinfos_write[i_next];

    if (k_cur->ne[2] == k_idxs->ne[0]) {
        return kv_swa->cpy_k(ctx, k_cur, k_idxs, il, sinfo);
    }

    // k_idxs may be expanded to one block per stream while k_cur is only
    // the token block. Keep zero deps on all copies so each write executes.
    const int64_t n_fanout = (int64_t) sinfo.size()*sinfo.n_stream();

    GGML_ASSERT(sinfo.n_stream() > 1);
    GGML_ASSERT(k_cur->ne[2] == (int64_t) sinfo.size());
    GGML_ASSERT(k_idxs->ne[0] == n_fanout);

    ggml_tensor * res = nullptr;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        ggml_tensor * k_idxs_s = ggml_view_1d(ctx, k_idxs, sinfo.size(), s*sinfo.size()*ggml_element_size(k_idxs));
        ggml_tensor * cur = kv_swa->cpy_k(ctx, k_cur, k_idxs_s, il, sinfo);
        if (res == nullptr) {
            res = cur;
        } else {
            res = ggml_add(ctx, res, ggml_sub(ctx, cur, cur));
        }
    }

    return res;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatches_write.empty() ? ubatch.n_tokens : ubatches_write[i_next].n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_rot(ggml_context * ctx) const {
    return kv_swa->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_idxs(ggml_tensor * dst) const {
    kv_swa->set_input_k_idxs(dst, &ubatches_write[i_next], sinfos_write[i_next]);
}

void llama_kv_cache_dsv4_raw_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv_swa->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_rot(ggml_tensor * dst) const {
    kv_swa->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_comp_context
//

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv) : kv(kv), n_kv(kv->get_size()) {
    const uint32_t n_stream = kv->get_n_stream();

    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(
        llama_kv_cache * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) :
    kv(kv),
    sinfos(std::move(sinfos)),
    ubatches(std::move(ubatches)),
    n_kv(kv->get_size()) {
}

bool llama_kv_cache_dsv4_comp_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

uint32_t llama_kv_cache_dsv4_comp_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_comp_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_context
//

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv) :
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw())),
    ctx_csa_mem(kv->get_csa()->init_full()),
    ctx_hca_mem(kv->get_hca()->init_full()),
    ctx_lid_mem(kv->get_lid()->init_full()),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_csa())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_hca())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_lid())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    reserve_plans(true),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw(), lctx, optimize)),
    ctx_csa_mem(kv->get_csa()->init_update(lctx, optimize)),
    ctx_hca_mem(kv->get_hca()->init_update(lctx, optimize)),
    ctx_lid_mem(kv->get_lid()->init_update(lctx, optimize)),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_csa())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_hca())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_lid())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        slot_info_vec_t sinfos_raw_base_write,
        slot_info_vec_t sinfos_raw_swa_write,
        slot_info_vec_t sinfos_raw_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_raw) :
    ubatches(std::move(ubatches)),
    plans_csa(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_csa_state()->get_state_size(), kv->get_csa()->get_size(), kv->get_csa_state()->get_n_stream())),
    plans_hca(dsv4_build_comp_plans(this->ubatches, DSV4_HCA_RATIO, false,
                kv->get_hca_state()->get_state_size(), kv->get_hca()->get_size(), kv->get_hca_state()->get_n_stream())),
    plans_lid(plans_csa),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(
                kv->get_raw(),
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                this->ubatches,
                std::move(ubatches_raw))),
    ctx_csa_mem(nullptr),
    ctx_hca_mem(nullptr),
    ctx_lid_mem(nullptr),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_csa(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_csa()->get_n_stream()),
                this->ubatches)),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_hca(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_hca()->get_n_stream()),
                this->ubatches)),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_lid(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_lid()->get_n_stream()),
                this->ubatches)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(ctx_raw->get_status()) {
}

llama_kv_cache_dsv4_context::~llama_kv_cache_dsv4_context() = default;

bool llama_kv_cache_dsv4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_raw->next();
    ctx_csa->next();
    ctx_hca->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_raw->apply();

    return res;
}

llama_memory_status llama_kv_cache_dsv4_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsv4_raw_context * llama_kv_cache_dsv4_context::get_raw() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_raw.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_csa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_csa.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_hca() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_hca.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_lid() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_lid.get();
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_csa_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return csa_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_hca_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return hca_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_lid_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return lid_state;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_csa.empty()) {
        return empty;
    }

    return plans_csa[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_hca.empty()) {
        return empty;
    }

    return plans_hca[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_lid.empty()) {
        return empty;
    }

    return plans_lid[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_csa_plan();
    }

    reserve_plan_csa = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            csa_state->get_state_size(), get_csa()->get_n_kv(), csa_state->get_n_stream());

    return reserve_plan_csa;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_hca_plan();
    }

    reserve_plan_hca = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_HCA_RATIO, false,
            hca_state->get_state_size(), get_hca()->get_n_kv(), hca_state->get_n_stream());

    return reserve_plan_hca;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_lid_plan();
    }

    reserve_plan_lid = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            lid_state->get_state_size(), get_lid()->get_n_kv(), lid_state->get_n_stream());

    return reserve_plan_lid;
}
