#include "llama_mmproj_pool.h"
#include "llama-impl.h"
#include "../src/llama-model.h"
#include <algorithm>
#include <chrono>
#include <thread>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static size_t calc_aligned_size(const std::vector<ggml_tensor *> & tensors, size_t align = 256) {
    size_t total = 0;
    for (ggml_tensor * t : tensors) {
        total = (total + align - 1) / align * align;
        total += ggml_nbytes(t);
    }
    return total;
}

static std::vector<ggml_tensor *> collect_evicted_tensors(struct llama_model * model, int n_swap_layers) {
    if (!model || n_swap_layers <= 0) return {};
    const int n_layer = llama_model_n_layer(model);
    const int first = std::max(0, n_layer - n_swap_layers);
    std::vector<ggml_tensor *> result;
    const auto & tensor_map = llama_internal_get_tensor_map(model);
    
    for (int il = first; il < n_layer; ++il) {
        const std::string prefix = "blk." + std::to_string(il) + ".";
        for (auto & [name, t] : tensor_map) {
            if (t && name.rfind(prefix, 0) == 0) {
                if (!t->buffer) continue;
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
                if (ggml_backend_buft_is_host(buft)) continue;
                result.push_back(t);
            }
        }
    }
    return result;
}

struct llama_mmproj_pool * llama_mmproj_pool_init(
    struct llama_model         * model,
    int                          n_swap_layers,
    std::vector<ggml_tensor *> & mmproj_tensors,
    size_t                       dynamic_overhead_bytes) {

    if (mmproj_tensors.empty()) return nullptr;

    size_t align = 256;
    size_t mmproj_host_size = calc_aligned_size(mmproj_tensors, align);

    // 1. Auto-calculation (-1) logic, combining precisely probed dynamic overhead
    if (n_swap_layers < 0) {
        int n_layer = llama_model_n_layer(model);
        size_t accumulated_size = 0;
        int calculated_layers = 0;
        
        // Target eviction size = Vision Weights + Compute Buffer
        // Reserve a 5% safety margin for VRAM fragmentation
        size_t target_eviction_size = (mmproj_host_size + dynamic_overhead_bytes) * 1.05;

        for (int il = n_layer - 1; il >= 0; --il) {
            calculated_layers++;
            auto evicted_tensors_tmp = collect_evicted_tensors(model, calculated_layers);
            
            accumulated_size = 0;
            for (auto * t : evicted_tensors_tmp) {
                accumulated_size += ggml_nbytes(t);
            }
            
            if (accumulated_size >= target_eviction_size) {
                break;
            }
        }
        n_swap_layers = calculated_layers;
        LLAMA_LOG_INFO("%s: auto mode: need %.0f MB (Weights) + %.0f MB (Overhead) for mmproj; will evict %d layers (target eviction %.0f MB)\n",
                    __func__, mmproj_host_size / 1e6, dynamic_overhead_bytes / 1e6, n_swap_layers, target_eviction_size / 1e6);
    }

    if (n_swap_layers <= 0) return nullptr;


    auto * pool = new llama_mmproj_pool();
    pool->evicted_tensors = collect_evicted_tensors(model, n_swap_layers);

    if (pool->evicted_tensors.empty()) {
        delete pool;
        return nullptr;
    }

    // Get the actual GPU Backend Dev to prepare for pinned memory
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            dev = d;
            break;
        }
    }

    // Allocate Host buffer
    size_t evicted_total_bytes = 0;
    for (auto * t : pool->evicted_tensors) {
        pool->evicted_offsets.push_back(evicted_total_bytes);
        evicted_total_bytes += ggml_nbytes(t);
    }

    pool->host_buf_size = evicted_total_bytes + mmproj_host_size;
    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (!host_buft) host_buft = ggml_backend_cpu_buffer_type();

    pool->host_buf = ggml_backend_buft_alloc_buffer(host_buft, pool->host_buf_size);
    if (!pool->host_buf) {
        delete pool;
        return nullptr;
    }
    pool->host_ptr = ggml_backend_buffer_get_base(pool->host_buf);
    char * host_mm = (char *)pool->host_ptr + evicted_total_bytes;

    // 2.Restore the robust "Bin-Packing" method to prevent any risk of data corruption
    struct Block {
        ggml_tensor * t;
        size_t used;
        size_t cap;
    };
    std::vector<Block> blocks;
    for (auto * t : pool->evicted_tensors) {
        blocks.push_back({t, 0, ggml_nbytes(t)});
    }
    std::sort(blocks.begin(), blocks.end(), [](const Block & a, const Block & b) { return a.cap > b.cap; });

    std::vector<ggml_tensor *> sorted_mmproj = mmproj_tensors;
    std::sort(sorted_mmproj.begin(), sorted_mmproj.end(), [](ggml_tensor * a, ggml_tensor * b) {
        return ggml_nbytes(a) > ggml_nbytes(b);
    });

    bool packing_failed = false;
    size_t current_host_offset = 0;

    for (ggml_tensor * vt : sorted_mmproj) {
        size_t vsize = ggml_nbytes(vt);
        
        current_host_offset = (current_host_offset + align - 1) / align * align;
        char * host_data = host_mm + current_host_offset;
        
        if (vt->data) {
            ggml_backend_tensor_get(vt, host_data, 0, vsize); // Backup vision model to host
        }
        current_host_offset += vsize;

        bool placed = false;
        for (auto & b : blocks) {
            size_t offset = (b.used + align - 1) / align * align;
            if (offset + vsize <= b.cap) {
                b.used = offset + vsize;
                char * gpu_data = (char *)b.t->data + offset;
                pool->mappings.push_back({vt, gpu_data, b.t->buffer, host_data, vsize});
                placed = true;
                break;
            }
        }
        if (!placed) {
            packing_failed = true;
            break;
        }
    }

    if (packing_failed) {
        LLAMA_LOG_ERROR("%s: Fragmentation prevents packing mmproj tensors. Increase --mmproj-swap-layers.\n", __func__);
        llama_mmproj_pool_free(pool);
        return nullptr;
    }

    // Redirect pointers, ready for execution
    for (const auto & m : pool->mappings) {
        m.vision_t->data   = m.host_data;
        m.vision_t->buffer = pool->host_buf;
    }

    pool->state = llama_pool_state::LLM_RESIDENT;
    LLAMA_LOG_INFO("%s: pool ready | %zu evicted (%.0f MB) | packed %zu mmproj (%.0f MB) | host_buft: %s\n",
                   __func__, pool->evicted_tensors.size(), evicted_total_bytes / 1e6, 
                   pool->mappings.size(), mmproj_host_size / 1e6, ggml_backend_buft_name(host_buft));
    return pool;
}




// Helper: Given the physical address of allocated gpu_data, deduce which evicted tensor (LLM layer) it maps to
static int find_evicted_idx(void * gpu_data, const std::vector<ggml_tensor*> & ev_tensors) {
    for (size_t i = 0; i < ev_tensors.size(); ++i) {
        char * base = (char *)ev_tensors[i]->data;
        size_t size = ggml_nbytes(ev_tensors[i]);
        // If the vision data falls within this evicted LLM tensor's address range
        if ((char *)gpu_data >= base && (char *)gpu_data < base + size) {
            return (int)i;
        }
    }
    return -1;
}

bool llama_mmproj_pool_swap_in(struct llama_mmproj_pool * pool, struct llama_context * ctx) {
    if (!pool) return false;
    std::lock_guard<std::mutex> guard(pool->mutex);
    if (pool->state == llama_pool_state::MMPROJ_RESIDENT) return true;
    if (pool->state == llama_pool_state::DISABLED || pool->state == llama_pool_state::CORRUPTED) return false;

    if (ctx) llama_synchronize(ctx);
    double t0 = now_ms();
    pool->state = llama_pool_state::SWAPPING_OUT;

    char * host_llm = (char *)pool->host_ptr;

    // 3. Use pipelining strategy to achieve PCIe full-duplex parallelism, completely preventing VRAM read/write pollution
    // First group vision tensors by the evicted LLM tensor they occupy
    std::vector<std::vector<llama_mmproj_pool::tensor_mapping>> grouped_mappings(pool->evicted_tensors.size());
    for (const auto & m : pool->mappings) {
        int idx = find_evicted_idx(m.gpu_data, pool->evicted_tensors);
        if (idx >= 0) {
            grouped_mappings[idx].push_back(m);
        }
    }

    std::thread prev_load_thread;

    for (size_t i = 0; i < pool->evicted_tensors.size(); ++i) {
        // Step A: Read the LLM weights of the current layer back to host (Device-to-Host)
        // This DMA copy is blocking in the main thread
        ggml_backend_tensor_get(
            pool->evicted_tensors[i], 
            host_llm + pool->evicted_offsets[i], 
            0, 
            ggml_nbytes(pool->evicted_tensors[i])
        );

        // Wait for the previous block's asynchronous write (H2D) to complete, preventing thread backlog
        if (prev_load_thread.joinable()) {
            prev_load_thread.join();
        }

        // Step B: Since the current layer (i-th) has been safely moved to host, its VRAM space can now be safely overwritten
        // Launch a background thread to write the corresponding vision tensors to that VRAM (Host-to-Device)
        // Key advantage: when the loop next executes D2H for layer i+1, it can run in full-duplex parallel with this H2D!
        prev_load_thread = std::thread([pool, i, &grouped_mappings]() {
            for (const auto & m : grouped_mappings[i]) {
                m.vision_t->data   = m.gpu_data;
                m.vision_t->buffer = m.gpu_buffer;
                ggml_backend_tensor_set(m.vision_t, m.host_data, 0, m.size); // Push to VRAM
            }
        });
    }

    // After the loop, ensure the final background write task has completed
    if (prev_load_thread.joinable()) {
        prev_load_thread.join();
    }

    pool->state = llama_pool_state::MMPROJ_RESIDENT;

    if (ctx) llama_synchronize(ctx);
    pool->total_swap_ms += (now_ms() - t0);
    ++pool->n_swaps;
    return true;
}






void llama_mmproj_pool_swap_back(struct llama_mmproj_pool * pool, struct llama_context * ctx) {
    if (!pool) return;
    std::lock_guard<std::mutex> guard(pool->mutex);
    if (pool->state != llama_pool_state::MMPROJ_RESIDENT) return;

    if (ctx) llama_synchronize(ctx);
    pool->state = llama_pool_state::SWAPPING_IN;

    // Vision -> Host (Adjust pointers only, no copy needed)
    for (const auto & m : pool->mappings) {
        m.vision_t->data   = m.host_data;
        m.vision_t->buffer = pool->host_buf;
    }

    // LLM -> GPU (Restore LLM)
    char * host_llm = (char *)pool->host_ptr;
    for (size_t i = 0; i < pool->evicted_tensors.size(); ++i) {
        ggml_backend_tensor_set(pool->evicted_tensors[i], host_llm + pool->evicted_offsets[i], 0, ggml_nbytes(pool->evicted_tensors[i]));
    }

    if (ctx) llama_synchronize(ctx);
    pool->state = llama_pool_state::LLM_RESIDENT;
}

void llama_mmproj_pool_free(struct llama_mmproj_pool * pool) {
    if (!pool) return;
    for (const auto & m : pool->mappings) {
        if (m.vision_t) {
            m.vision_t->data = nullptr;
            m.vision_t->buffer = nullptr;
        }
    }
    if (pool->host_buf) ggml_backend_buffer_free(pool->host_buf);
    delete pool;
}

void llama_mmproj_pool_log_stats(const struct llama_mmproj_pool * pool) {
    if (!pool) return;
    LLAMA_LOG_INFO("mmproj pool stats: n_swaps=%lld, avg_swap_ms=%.1f\n",
                   (long long)pool->n_swaps,
                   pool->n_swaps > 0 ? pool->total_swap_ms / pool->n_swaps : 0.0);
}
