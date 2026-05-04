#include "llama-kv-cache-paged.h"

#include "llama-impl.h"

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max) :
    kv_type(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    gpu_backend(nullptr),
    cpu_backend(nullptr) {}

void llama_kv_cache_paged::init(ggml_backend_t backend_gpu,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");
    GGML_ASSERT(backend_gpu && "backend_gpu is nullptr");
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend_gpu);
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        LLAMA_LOG_WARN(
            "%s: no GPU device found, allocating KV block pool on CPU. "
            "This is valid for testing but it will be slow.\n",
            __func__);
    }

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    LLAMA_LOG_INFO(
        "%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f\n", __func__,
        n_gpu_blocks, n_cpu_blocks, block_size, watermark);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    kv_type        = type;
    gpu_backend    = backend_gpu;
    cpu_backend    = backend_cpu;
    block_bytes    = 2 * block_size * n_heads_kv * head_dim * ggml_type_size(kv_type);

    // Set up GPU context and tensor
    // Interleaved shape: [num_blocks, 2, n_heads_kv, block_size, head_dim] (5D)
    struct ggml_init_params gpu_params;
    gpu_params.mem_size   = ggml_tensor_overhead() * 5 * n_layers;
    gpu_params.mem_buffer = NULL;
    gpu_params.no_alloc   = true;

    struct ggml_context * ctx_gpu = ggml_init(gpu_params);

    for (uint32_t il = 0; il < n_layers; ++il) {
        // Since GGML_MAX_DIMS is set to 4, we flatten the layout to be 4D: [num_blocks, 2 * n_heads_kv, block_size, head_dim]
        ggml_tensor * kv_layer_gpu =
            ggml_new_tensor_4d(ctx_gpu, type, head_dim, block_size, 2 * n_heads_kv, n_gpu_blocks);
        kv_gpu_layers.push_back(kv_layer_gpu);
    }

    // Allocate on GPU backend
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer");
    ggml_backend_buffer_clear(buf_gpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
        GGML_ASSERT(kv_gpu_layers[il]->buffer && "GPU layer tensor has null buffer");
    }

    // For non CUDA backends, we would split views to allow for standard ggml operators to work out of the box

    // Set up CPU context and tensor (for swapping)
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * n_layers;
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    struct ggml_context * ctx_cpu = ggml_init(cpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * kv_layer_cpu =
            ggml_new_tensor_4d(ctx_cpu, type, head_dim, block_size, 2 * n_heads_kv, n_cpu_blocks);
        kv_cpu_layers.push_back(kv_layer_cpu);
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(buf_cpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
        GGML_ASSERT(kv_cpu_layers[il]->buffer && "CPU layer tensor has null buffer");
    }

    // Setting up our block accountant
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
}

bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    uint32_t total_num_tokens     = group.n_prompt + group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_gpu_blocks(num_requested_blocks)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_requested_blocks);
    concat_block_ids(group.block_table, new_ids);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (group.block_table.empty()) {
        return;
    }

    llama_block_ids blocks_to_free_gpu;
    llama_block_ids blocks_to_free_cpu;

    for (uint32_t block_id : group.block_table) {
        if (block_manager.is_gpu(block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    if (!blocks_to_free_gpu.empty()) {
        block_manager.release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }

    group.block_table.clear();
    seq_rm(group.request_id, llama_pos{}, llama_pos{});
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    const auto & src_layers = to_gpu ? kv_cpu_layers : kv_gpu_layers;
    const auto & dst_layers = to_gpu ? kv_gpu_layers : kv_cpu_layers;

    GGML_ASSERT(src_layers.size() == n_layers && "src layer count mismatch.");
    GGML_ASSERT(dst_layers.size() == n_layers && "src layer count mismatch.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        struct ggml_tensor * src_main = src_layers[il];
        struct ggml_tensor * dst_main = dst_layers[il];

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;

            // Put src tensor into HOST staging buffer
            ggml_backend_tensor_get(src_main, staging.data(), src_offset, block_bytes);
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_set(dst_main, staging.data(), dst_offset, block_bytes);
        }
    }
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    // A potential optimization to reduce thrashing is to have a heuristic to check if
    // if we can continue decoding after swap_in.
    if (!block_manager.has_free_gpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/true);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_cpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));

        // Do not use balloc's internal batch. It does not carry any paged metadata.
        GGML_ASSERT(last_paged_info && "no paged batch info set before init_batch was called.");
        ctx->set_batch_data(*last_paged_info);
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(num_gpu_blocks);  // every block could theoretically belong to one seq

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    std::vector<llama_ubatch> dummy_ubatch = {};
    auto                      ctx          = std::make_unique<llama_kv_cache_paged_context>(this, dummy_ubatch);
    // TODO maybe confirm block counts or clean up stale pointers
    return ctx;
}

struct ggml_tensor * llama_kv_cache_paged::get_kv_tensor(int layer_idx) const {
    return kv_gpu_layers[layer_idx];
}

void llama_kv_cache_paged::clear(bool /*data*/) {
    sequence_positions.clear();
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos /*p0*/, llama_pos /*p1*/) {
    sequence_positions.erase(seq_id);
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;
    const size_t                                 n_gpu_kvs = kv_gpu_layers.size();
    const size_t                                 n_cpu_kvs = kv_cpu_layers.size();

    for (size_t il = 0; il < n_layers; ++il) {
        auto * kv_gpu = (il < n_gpu_kvs) ? kv_gpu_layers[il] : nullptr;
        if (kv_gpu) {
            breakdown[ggml_backend_buffer_get_type(kv_gpu->buffer)] = ggml_nbytes(kv_gpu);
        }
        auto * kv_cpu = (il < n_cpu_kvs) ? kv_cpu_layers[il] : nullptr;
        if (kv_cpu) {
            breakdown[ggml_backend_buffer_get_type(kv_cpu->buffer)] = ggml_nbytes(kv_cpu);
        }
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    paged_write_slots   = info.write_slots;
    paged_block_table   = info.block_table;
    paged_context_lens  = info.context_lens;
    paged_batch_offsets = info.batch_offsets;
    paged_batch_lens    = info.batch_lens;
    n_tokens            = info.n_tokens;
    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}
