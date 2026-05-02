#include "llama-memory-deepseek4.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// v1: every cache tensor was serialized with its full ggml_nbytes(), regardless of how
//     many slots were populated. With n_ctx in the millions this made each checkpoint
//     several GiB even for short conversations; the server's per-turn checkpoint restore
//     (triggered because DeepSeek4 only supports full-removal seq_rm) became dominant.
// v2: only the active row prefix of n_ctx-scaling tensors (attn_kv, indexer_kv) is
//     written. On read the active prefix bytes are restored and the remaining tail is
//     explicitly zeroed via ggml_backend_tensor_memset, preserving the
//     "untouched-slot == zero" invariant the compute graph relies on.
static constexpr uint32_t DEEPSEEK4_STATE_VERSION = 2;

static llama_ubatch make_dummy_ubatch() {
    llama_ubatch ubatch = {};
    ubatch.data = std::make_shared<llama_ubatch::data_t>();

    ubatch.b_equal_seqs = 1;
    ubatch.n_tokens = 1;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;

    ubatch.data->token = { 0 };
    ubatch.data->pos = { 0 };
    ubatch.data->n_seq_id = { 1 };
    ubatch.data->seq_id_unq = { 0 };
    ubatch.data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    ubatch.data->seq_idx[0] = 0;
    ubatch.data->output = { 0 };
    ubatch.data->seq_id_data = { 0 };
    ubatch.data->seq_id = { ubatch.data->seq_id_data.data() };

    ubatch.token = ubatch.data->token.data();
    ubatch.embd = nullptr;
    ubatch.pos = ubatch.data->pos.data();
    ubatch.n_seq_id = ubatch.data->n_seq_id.data();
    ubatch.seq_id = ubatch.data->seq_id.data();
    ubatch.seq_id_unq = ubatch.data->seq_id_unq.data();
    ubatch.seq_idx = ubatch.data->seq_idx.data();
    ubatch.output = ubatch.data->output.data();

    return ubatch;
}

static uint32_t deepseek4_compress_ratio(const llama_layer & layer) {
    return layer.attn_compress_ape ? static_cast<uint32_t>(layer.attn_compress_ape->ne[1]) : 0;
}

static uint32_t deepseek4_comp_slots(const ggml_tensor * ape, uint32_t head_dim) {
    if (!ape || head_dim == 0) {
        return 0;
    }

    return static_cast<uint32_t>(ape->ne[0] / head_dim);
}

static void deepseek4_fill_f32_tensor(ggml_tensor * tensor, float value) {
    if (!tensor) {
        return;
    }

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> data(ggml_nelements(tensor), value);
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

static void deepseek4_write_tensor(llama_io_write_i & io, const ggml_tensor * tensor, uint64_t active_bytes_override = UINT64_MAX) {
    const uint32_t present = tensor != nullptr;
    io.write(&present, sizeof(present));

    if (!present) {
        return;
    }

    const int32_t type = static_cast<int32_t>(tensor->type);
    const uint32_t n_dims = ggml_n_dims(tensor);
    int64_t ne[GGML_MAX_DIMS] = {};
    for (uint32_t i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i] = tensor->ne[i];
    }
    const uint64_t total_bytes  = ggml_nbytes(tensor);
    const uint64_t active_bytes = active_bytes_override == UINT64_MAX
                                      ? total_bytes
                                      : std::min<uint64_t>(active_bytes_override, total_bytes);

    io.write(&type,         sizeof(type));
    io.write(&n_dims,       sizeof(n_dims));
    io.write(ne,            sizeof(ne));
    io.write(&active_bytes, sizeof(active_bytes));
    io.write(&total_bytes,  sizeof(total_bytes));
    if (active_bytes > 0) {
        io.write_tensor(tensor, 0, active_bytes);
    }
}

static void deepseek4_read_tensor(llama_io_read_i & io, ggml_tensor * tensor) {
    uint32_t present;
    io.read_to(&present, sizeof(present));

    if (!present) {
        if (tensor != nullptr) {
            throw std::runtime_error("DeepSeek4 state is missing a runtime tensor");
        }
        return;
    }

    if (tensor == nullptr) {
        throw std::runtime_error("DeepSeek4 state contains an unexpected runtime tensor");
    }

    int32_t type_ref;
    uint32_t n_dims_ref;
    int64_t ne_ref[GGML_MAX_DIMS];
    uint64_t active_bytes_ref;
    uint64_t total_bytes_ref;

    io.read_to(&type_ref,         sizeof(type_ref));
    io.read_to(&n_dims_ref,       sizeof(n_dims_ref));
    io.read_to(ne_ref,            sizeof(ne_ref));
    io.read_to(&active_bytes_ref, sizeof(active_bytes_ref));
    io.read_to(&total_bytes_ref,  sizeof(total_bytes_ref));

    if (type_ref != static_cast<int32_t>(tensor->type)) {
        throw std::runtime_error("DeepSeek4 state tensor type mismatch");
    }
    if (n_dims_ref != static_cast<uint32_t>(ggml_n_dims(tensor))) {
        throw std::runtime_error("DeepSeek4 state tensor rank mismatch");
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; ++i) {
        if (ne_ref[i] != tensor->ne[i]) {
            throw std::runtime_error("DeepSeek4 state tensor shape mismatch");
        }
    }

    const uint64_t total_bytes = ggml_nbytes(tensor);
    if (total_bytes_ref != total_bytes) {
        throw std::runtime_error("DeepSeek4 state tensor size mismatch");
    }
    if (active_bytes_ref > total_bytes) {
        throw std::runtime_error("DeepSeek4 state tensor active range exceeds tensor size");
    }

    if (active_bytes_ref > 0) {
        ggml_backend_tensor_set(tensor, io.read(active_bytes_ref), 0, active_bytes_ref);
    }
    if (active_bytes_ref < total_bytes) {
        // Preserve the "untouched-slot == zero" invariant the compute graph relies on:
        // build_attn_v4 reads compressed/indexer prefixes by current batch end, which can
        // include rows beyond the restored prefix on the first batch after restore.
        ggml_backend_tensor_memset(tensor, 0, active_bytes_ref, total_bytes - active_bytes_ref);
    }
}

} // namespace

llama_memory_deepseek4::llama_memory_deepseek4(
        const llama_model & model,
                ggml_type   type_k,
                     bool   offload,
                 uint32_t   n_ctx_seq,
                 uint32_t   n_seq_max) :
    model(model),
    n_ctx_seq(n_ctx_seq),
    n_seq_max(n_seq_max),
    layers(model.hparams.n_layer),
    seq_pos_min_v(n_seq_max, -1),
    seq_pos_max_v(n_seq_max, -1) {
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params params = {
            /*.mem_size   =*/ size_t(16u * model.hparams.n_layer * ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            return nullptr;
        }

        ctx_map.emplace(buft, ctx);
        return ctx;
    };

    for (int32_t il = 0; il < (int32_t) model.hparams.n_layer; ++il) {
        const auto & layer_model = model.layers[il];
        auto & layer = layers[il];

        const uint32_t head_dim = model.hparams.n_embd_head_k(il);
        const uint32_t ratio = deepseek4_compress_ratio(layer_model);
        const uint32_t kv_size = model.hparams.n_swa + (ratio ? n_ctx_seq / ratio : 0);

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create DeepSeek4 state context");
        }

        layer.attn_kv = ggml_new_tensor_2d(ctx, type_k, head_dim, kv_size);
        ggml_format_name(layer.attn_kv, "deepseek4_attn_kv_l%d", il);

        if (ratio > 0) {
            const uint32_t attn_comp_slots = deepseek4_comp_slots(layer_model.attn_compress_ape, head_dim);
            layer.attn_comp_kv_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layer_model.attn_compress_ape->ne[0], attn_comp_slots * ratio);
            ggml_format_name(layer.attn_comp_kv_state, "deepseek4_attn_comp_kv_state_l%d", il);
            layer.attn_comp_score_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layer_model.attn_compress_ape->ne[0], attn_comp_slots * ratio);
            ggml_format_name(layer.attn_comp_score_state, "deepseek4_attn_comp_score_state_l%d", il);
        }

        if (layer_model.indexer_proj && layer_model.indexer_attn_q_b && layer_model.indexer_compress_ape) {
            const uint32_t idx_ratio = static_cast<uint32_t>(layer_model.indexer_compress_ape->ne[1]);
            const uint32_t idx_head_dim = model.hparams.indexer_head_size;
            const uint32_t idx_kv_size = idx_ratio ? n_ctx_seq / idx_ratio : 0;
            const uint32_t idx_comp_slots = deepseek4_comp_slots(layer_model.indexer_compress_ape, idx_head_dim);

            layer.indexer_kv = ggml_new_tensor_2d(ctx, type_k, idx_head_dim, idx_kv_size);
            ggml_format_name(layer.indexer_kv, "deepseek4_indexer_kv_l%d", il);

            layer.indexer_comp_kv_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layer_model.indexer_compress_ape->ne[0], idx_comp_slots * idx_ratio);
            ggml_format_name(layer.indexer_comp_kv_state, "deepseek4_indexer_comp_kv_state_l%d", il);
            layer.indexer_comp_score_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layer_model.indexer_compress_ape->ne[0], idx_comp_slots * idx_ratio);
            ggml_format_name(layer.indexer_comp_score_state, "deepseek4_indexer_comp_score_state_l%d", il);
        }
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate DeepSeek4 state buffer");
        }
        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    for (auto & layer : layers) {
        deepseek4_fill_f32_tensor(layer.attn_comp_score_state, -std::numeric_limits<float>::infinity());
        deepseek4_fill_f32_tensor(layer.indexer_comp_score_state, -std::numeric_limits<float>::infinity());
    }
}

llama_memory_context_ptr llama_memory_deepseek4::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);
    GGML_UNUSED(n_ubatch);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        // The compression-window indexer state is updated one position at a
        // time, so each ubatch carries exactly one token from one sequence.
        llama_ubatch ubatch = balloc.split_seq(1);
        if (ubatch.n_tokens == 0) {
            break;
        }

        if (ubatch.n_tokens != 1 || ubatch.n_seqs_unq != 1) {
            LLAMA_LOG_ERROR("%s: DeepSeek4 runtime currently supports a single token from a single sequence per ubatch\n",
                    __func__);
            return std::make_unique<llama_memory_deepseek4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i] < 0 || (uint32_t) ubatch.pos[i] >= n_ctx_seq) {
                LLAMA_LOG_ERROR("%s: DeepSeek4 runtime position %d exceeds the configured context length %u\n",
                        __func__, ubatch.pos[i], n_ctx_seq);
                return std::make_unique<llama_memory_deepseek4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
        }

        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_memory_deepseek4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_deepseek4_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_deepseek4::init_full() {
    std::vector<llama_ubatch> ubatches = { make_dummy_ubatch() };
    return std::make_unique<llama_memory_deepseek4_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_deepseek4::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_deepseek4_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_deepseek4::get_can_shift() const {
    return false;
}

void llama_memory_deepseek4::clear(bool data) {
    std::fill(seq_pos_min_v.begin(), seq_pos_min_v.end(), -1);
    std::fill(seq_pos_max_v.begin(), seq_pos_max_v.end(), -1);

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
        for (auto & layer : layers) {
            deepseek4_fill_f32_tensor(layer.attn_comp_score_state, -std::numeric_limits<float>::infinity());
            deepseek4_fill_f32_tensor(layer.indexer_comp_score_state, -std::numeric_limits<float>::infinity());
        }
    }
}

bool llama_memory_deepseek4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const llama_pos r0 = p0 < 0 ? 0 : p0;
    const llama_pos r1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    if (r0 >= r1) {
        return true;
    }

    llama_pos pos_min = -1;
    llama_pos pos_max = -1;
    if (seq_id < 0) {
        for (size_t i = 0; i < seq_pos_min_v.size(); ++i) {
            if (seq_pos_min_v[i] < 0) {
                continue;
            }
            pos_min = pos_min < 0 ? seq_pos_min_v[i] : std::min(pos_min, seq_pos_min_v[i]);
            pos_max = std::max(pos_max, seq_pos_max_v[i]);
        }
    } else {
        if (static_cast<size_t>(seq_id) >= seq_pos_min_v.size()) {
            return false;
        }
        pos_min = seq_pos_min_v[seq_id];
        pos_max = seq_pos_max_v[seq_id];
    }

    if (pos_min < 0 || r1 <= pos_min || r0 > pos_max) {
        return true;
    }

    if (r0 <= pos_min && r1 > pos_max) {
        clear(true);
        return true;
    }

    return false;
}

void llama_memory_deepseek4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(seq_id_src);
    GGML_UNUSED(seq_id_dst);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
}

void llama_memory_deepseek4::seq_keep(llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);
}

void llama_memory_deepseek4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(shift);
}

void llama_memory_deepseek4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);
}

llama_pos llama_memory_deepseek4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_pos_min_v.size()) {
        return -1;
    }
    return seq_pos_min_v[seq_id];
}

llama_pos llama_memory_deepseek4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_pos_max_v.size()) {
        return -1;
    }
    return seq_pos_max_v[seq_id];
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_deepseek4::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb;
    for (const auto & [_, buf] : ctxs_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return mb;
}

void llama_memory_deepseek4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    const bool seq_specific = seq_id != -1;
    const bool seq_valid = seq_id >= 0 && static_cast<size_t>(seq_id) < seq_pos_min_v.size();
    const bool seq_active = !seq_specific || (seq_valid && seq_pos_min_v[seq_id] >= 0);

    const uint32_t version = DEEPSEEK4_STATE_VERSION;
    const uint32_t n_layer = layers.size();
    const uint32_t seq_mode = seq_specific ? 1 : 0;
    const uint32_t has_data = seq_active ? 1 : 0;
    const uint32_t seq_count = seq_specific ? 1 : n_seq_max;

    io.write(&version,   sizeof(version));
    io.write(&n_ctx_seq, sizeof(n_ctx_seq));
    io.write(&n_seq_max, sizeof(n_seq_max));
    io.write(&n_layer,   sizeof(n_layer));
    io.write(&seq_mode,  sizeof(seq_mode));
    io.write(&has_data,  sizeof(has_data));
    io.write(&seq_count, sizeof(seq_count));

    if (seq_specific) {
        const llama_pos pos_min = seq_valid ? seq_pos_min_v[seq_id] : -1;
        const llama_pos pos_max = seq_valid ? seq_pos_max_v[seq_id] : -1;
        io.write(&pos_min, sizeof(pos_min));
        io.write(&pos_max, sizeof(pos_max));
    } else {
        for (uint32_t i = 0; i < n_seq_max; ++i) {
            const llama_pos pos_min = i < seq_pos_min_v.size() ? seq_pos_min_v[i] : -1;
            const llama_pos pos_max = i < seq_pos_max_v.size() ? seq_pos_max_v[i] : -1;
            io.write(&pos_min, sizeof(pos_min));
            io.write(&pos_max, sizeof(pos_max));
        }
    }

    if (!has_data) {
        return;
    }

    // Compute the highest populated position over the seqs we are about to serialize so
    // that n_ctx-scaling tensors can be trimmed to their active prefix.  The model only
    // supports n_seq_max == 1 in practice; for the broader (-1) save case we take the
    // union of all seqs to stay correct if that ever changes.
    llama_pos pos_max_global = -1;
    if (seq_specific) {
        if (seq_valid) {
            pos_max_global = seq_pos_max_v[seq_id];
        }
    } else {
        for (size_t i = 0; i < seq_pos_max_v.size(); ++i) {
            if (seq_pos_min_v[i] >= 0) {
                pos_max_global = std::max(pos_max_global, seq_pos_max_v[i]);
            }
        }
    }

    const uint32_t n_swa = model.hparams.n_swa;

    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & layer       = layers[il];
        const auto & layer_model = model.layers[il];

        // attn_kv: shape [head_dim, n_swa + n_ctx_seq/ratio]; rows used are
        // [0, n_swa) (SWA circular slots) plus [n_swa, n_swa + ceil((pos_max+1)/ratio)).
        // For ratio == 0 there is no compressed region and the tensor is sized for n_swa.
        uint64_t attn_active_bytes = UINT64_MAX;
        if (layer.attn_kv != nullptr) {
            const uint32_t ratio    = deepseek4_compress_ratio(layer_model);
            const uint64_t row_size = layer.attn_kv->nb[1];
            const uint64_t total_rows  = layer.attn_kv->ne[1];
            uint64_t       active_rows = std::min<uint64_t>(n_swa, total_rows);
            if (ratio > 0 && pos_max_global >= 0) {
                const uint64_t comp_rows = (uint64_t(pos_max_global) + ratio) / ratio; // ceil((pos_max+1)/ratio)
                active_rows = std::min<uint64_t>(uint64_t(n_swa) + comp_rows, total_rows);
            }
            attn_active_bytes = active_rows * row_size;
        }

        // indexer_kv: shape [idx_head_dim, n_ctx_seq/idx_ratio]; rows used are
        // [0, ceil((pos_max+1)/idx_ratio)).  No n_swa offset for the indexer.
        uint64_t indexer_active_bytes = UINT64_MAX;
        if (layer.indexer_kv != nullptr && layer_model.indexer_compress_ape != nullptr) {
            const uint32_t idx_ratio = static_cast<uint32_t>(layer_model.indexer_compress_ape->ne[1]);
            const uint64_t row_size  = layer.indexer_kv->nb[1];
            const uint64_t total_rows = layer.indexer_kv->ne[1];
            uint64_t       active_rows = 0;
            if (idx_ratio > 0 && pos_max_global >= 0) {
                active_rows = std::min<uint64_t>((uint64_t(pos_max_global) + idx_ratio) / idx_ratio, total_rows);
            }
            indexer_active_bytes = active_rows * row_size;
        }

        deepseek4_write_tensor(io, layer.attn_kv, attn_active_bytes);
        // attn_comp_*/indexer_comp_* are fixed-size compression state and must be
        // restored byte-for-byte (they encode incremental sums that the next batch
        // continues from).  Pass UINT64_MAX to keep the full-size write path.
        deepseek4_write_tensor(io, layer.attn_comp_kv_state);
        deepseek4_write_tensor(io, layer.attn_comp_score_state);
        deepseek4_write_tensor(io, layer.indexer_kv, indexer_active_bytes);
        deepseek4_write_tensor(io, layer.indexer_comp_kv_state);
        deepseek4_write_tensor(io, layer.indexer_comp_score_state);
    }
}

void llama_memory_deepseek4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t n_ctx_seq_ref;
    uint32_t n_seq_max_ref;
    uint32_t n_layer_ref;
    uint32_t seq_mode;
    uint32_t has_data;
    uint32_t seq_count;

    io.read_to(&version,       sizeof(version));
    io.read_to(&n_ctx_seq_ref, sizeof(n_ctx_seq_ref));
    io.read_to(&n_seq_max_ref, sizeof(n_seq_max_ref));
    io.read_to(&n_layer_ref,   sizeof(n_layer_ref));
    io.read_to(&seq_mode,      sizeof(seq_mode));
    io.read_to(&has_data,      sizeof(has_data));
    io.read_to(&seq_count,     sizeof(seq_count));

    if (version != DEEPSEEK4_STATE_VERSION) {
        throw std::runtime_error("DeepSeek4 state version mismatch");
    }
    if (n_ctx_seq_ref != n_ctx_seq) {
        throw std::runtime_error("DeepSeek4 state context length mismatch");
    }
    if (n_layer_ref != layers.size()) {
        throw std::runtime_error("DeepSeek4 state layer count mismatch");
    }

    if (seq_mode == 1) {
        if (seq_count != 1) {
            throw std::runtime_error("DeepSeek4 sequence state metadata mismatch");
        }

        llama_pos pos_min;
        llama_pos pos_max;
        io.read_to(&pos_min, sizeof(pos_min));
        io.read_to(&pos_max, sizeof(pos_max));

        if (seq_id < 0 || static_cast<size_t>(seq_id) >= seq_pos_min_v.size()) {
            throw std::runtime_error("DeepSeek4 sequence state destination is out of range");
        }

        seq_pos_min_v[seq_id] = has_data ? pos_min : -1;
        seq_pos_max_v[seq_id] = has_data ? pos_max : -1;
    } else if (seq_mode == 0) {
        const uint32_t n_read = std::min<uint32_t>(seq_count, n_seq_max);
        for (uint32_t i = 0; i < seq_count; ++i) {
            llama_pos pos_min;
            llama_pos pos_max;
            io.read_to(&pos_min, sizeof(pos_min));
            io.read_to(&pos_max, sizeof(pos_max));

            if (i < n_read) {
                seq_pos_min_v[i] = pos_min;
                seq_pos_max_v[i] = pos_max;
            }
        }
        for (uint32_t i = n_read; i < n_seq_max; ++i) {
            seq_pos_min_v[i] = -1;
            seq_pos_max_v[i] = -1;
        }
    } else {
        throw std::runtime_error("DeepSeek4 state sequence mode mismatch");
    }

    GGML_UNUSED(n_seq_max_ref);

    if (!has_data) {
        return;
    }

    for (auto & layer : layers) {
        deepseek4_read_tensor(io, layer.attn_kv);
        deepseek4_read_tensor(io, layer.attn_comp_kv_state);
        deepseek4_read_tensor(io, layer.attn_comp_score_state);
        deepseek4_read_tensor(io, layer.indexer_kv);
        deepseek4_read_tensor(io, layer.indexer_comp_kv_state);
        deepseek4_read_tensor(io, layer.indexer_comp_score_state);
    }
}

const llama_memory_deepseek4::layer_state & llama_memory_deepseek4::get_layer(int32_t il) const {
    return layers.at(il);
}

uint32_t llama_memory_deepseek4::get_n_ctx_seq() const {
    return n_ctx_seq;
}

llama_memory_deepseek4_context::llama_memory_deepseek4_context(llama_memory_status status) :
    status(status) {
}

llama_memory_deepseek4_context::llama_memory_deepseek4_context(
        llama_memory_deepseek4 * mem,
        std::vector<llama_ubatch> ubatches) :
    status(LLAMA_MEMORY_STATUS_SUCCESS),
    mem(mem),
    ubatches(std::move(ubatches)) {
}

bool llama_memory_deepseek4_context::next() {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
        return false;
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_deepseek4_context::apply() {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || mem == nullptr || ubatches.empty()) {
        return status != LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    }

    const auto & ubatch = ubatches[i_next];
    const llama_seq_id seq_id = ubatch.seq_id[0][0];
    if (seq_id < 0 || (size_t) seq_id >= mem->seq_pos_min_v.size()) {
        return false;
    }

    auto & pos_min = mem->seq_pos_min_v[seq_id];
    auto & pos_max = mem->seq_pos_max_v[seq_id];

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.seq_id[i][0] != seq_id) {
            return false;
        }

        const llama_pos pos = ubatch.pos[i];
        pos_min = pos_min < 0 ? pos : std::min(pos_min, pos);
        pos_max = std::max(pos_max, pos);
    }

    return true;
}

const llama_ubatch & llama_memory_deepseek4_context::get_ubatch() const {
    return ubatches.at(i_next);
}

llama_memory_status llama_memory_deepseek4_context::get_status() const {
    return status;
}

const llama_memory_deepseek4::layer_state & llama_memory_deepseek4_context::get_layer(int32_t il) const {
    return mem->get_layer(il);
}

uint32_t llama_memory_deepseek4_context::get_n_ctx_seq() const {
    return mem->get_n_ctx_seq();
}
