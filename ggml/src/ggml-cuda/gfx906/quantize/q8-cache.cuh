#pragma once

// Q8_1 activation cache for GFX906 - True Multi-Consumer Detection
// Only caches tensors with 2+ MUL_MAT consumers (pre-computed via graph analysis)

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm> // for std::max
#include <cctype>    // for isdigit()
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // for atoi()
#include <cstring>   // for strrchr()

#include "../gfx906-config.h"

// Diagnostics level: 0=none, 1=summary only, 2=verbose (per-op logging)
#ifndef Q8_CACHE_DIAGNOSTICS
#define Q8_CACHE_DIAGNOSTICS 1
#endif

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define Q8_CACHE_CHECK(err) do { if ((err) != hipSuccess) { fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err)); abort(); } } while(0)
#define Q8_CACHE_MALLOC(ptr, size) Q8_CACHE_CHECK(hipMalloc(ptr, size))
#define Q8_CACHE_FREE(ptr) Q8_CACHE_CHECK(hipFree(ptr))
#else
#include <cuda_runtime.h>
#define Q8_CACHE_CHECK(err) do { if ((err) != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); abort(); } } while(0)
#define Q8_CACHE_MALLOC(ptr, size) Q8_CACHE_CHECK(cudaMalloc(ptr, size))
#define Q8_CACHE_FREE(ptr) Q8_CACHE_CHECK(cudaFree(ptr))
#endif

// Forward declarations
struct ggml_tensor;
struct ggml_cgraph;

// Cache key for hashmap lookups
// CRITICAL FIX: Use tensor name instead of pointer (pointers change between graph builds)
struct q8_cache_key {
    std::string tensor_name;  // Use name (stable) instead of pointer (unstable)
    int layout;
    int layer = -1;           // NEW: extracted layer number for slot routing
    
    bool operator==(const q8_cache_key& other) const {
        return layer == other.layer && 
               layout == other.layout && 
               tensor_name == other.tensor_name;
    }
};

struct q8_cache_key_hash {
    size_t operator()(const q8_cache_key& k) const {
        // Layer-first hashing for better cache locality
        return std::hash<int>{}(k.layer) ^ 
               (std::hash<std::string>{}(k.tensor_name) << 4) ^
               (std::hash<int>{}(k.layout) << 8);
    }
};

// Slot metadata with generation tracking (NEW)
struct q8_cache_slot {
    char* base_ptr = nullptr;      // Start in main arena
    char* bump_ptr = nullptr;      // Next allocation position
    int owner_group = -1;          // Current layer group
    uint32_t generation = 0;       // Incremented on ownership change
    size_t used_bytes = 0;
    size_t peak_bytes = 0;         // Current generation peak
    size_t max_peak_bytes = 0;     // Max peak across all generations
    bool is_active = false;
    
    // Free list for recyclable space within this slot
    std::vector<std::pair<void*, size_t>> free_list;
    
    // Reset bump pointer for new allocation (preserves free_list for recycling)
    void reset() {
        bump_ptr = base_ptr;
        used_bytes = 0;
        // NOTE: peak_bytes and free_list NOT reset - we track across generations
    }
    
    // Full reset (clear everything including free_list)
    void full_reset() {
        bump_ptr = base_ptr;
        used_bytes = 0;
        peak_bytes = 0;
        max_peak_bytes = 0;
        free_list.clear();
        owner_group = -1;
        generation = 0;
        is_active = false;
    }
    
    void change_ownership(int new_group) {
        if (owner_group != new_group) {
            // Save peak before cycling
            if (peak_bytes > max_peak_bytes) {
                max_peak_bytes = peak_bytes;
            }
            owner_group = new_group;
            generation++;
            reset();  // Only reset bump, keep free_list for reuse!
            peak_bytes = 0;  // Reset for new generation
            is_active = true;
        }
    }
    
    // Get the true maximum peak (current or historical)
    size_t get_true_peak() const {
        return std::max(peak_bytes, max_peak_bytes);
    }
};

// Cache entry with actual pointer
// USE-COUNT RECYCLING: Track consumption to enable slot reuse
struct q8_cache_entry {
    void* ptr;
    size_t size;                 // Aligned size
    int64_t ne10_padded;
    int64_t ne11;
    int64_t ne12;
    int64_t ne13;
    
    // Use-count tracking for eviction
    int use_count = 0;           // Current use count
    int expected_consumers = 2;  // Expected total consumers (from analysis, default 2)
    bool is_active = true;       // False = can be recycled
    
    // NEW: Layer-cycling validation
    uint32_t generation = 0;     // Must match slot.generation
    int slot_idx = -1;           // Source slot
};

// Layer-cycling cache arena with multi-consumer detection
struct q8_cache_arena {
    // Pre-allocated GPU memory
    void* base_ptr = nullptr;
    
    // Constants - layer-cycling configuration
    static constexpr size_t TOTAL_SIZE = GFX906_Q8_CACHE_TOTAL_SIZE;
    static constexpr int NUM_SLOTS = GFX906_Q8_CACHE_NUM_SLOTS;
    static constexpr int LAYERS_PER_SLOT = GFX906_Q8_CACHE_LAYERS_PER_SLOT;
    static constexpr size_t SLOT_SIZE = TOTAL_SIZE / NUM_SLOTS;
    static constexpr size_t ALIGNMENT = 256;  // 256-byte alignment
    
    // NEW: Fixed slots for layer-cycling
    q8_cache_slot slots[NUM_SLOTS];
    
    // NEW: Layer extraction cache (tensor ptr → layer number)
    std::unordered_map<const void*, int> layer_cache;
    
    // Hash map for tensor -> cached data lookup
    std::unordered_map<q8_cache_key, q8_cache_entry, q8_cache_key_hash> entries;
    
    // Consumer count map from analysis (populated by analyze_graph)
    std::unordered_map<std::string, int> consumer_counts;
    
    // Cache candidate names (for diagnostics)
    std::unordered_set<std::string> cache_candidate_names;
    
    // Statistics
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t cached_count = 0;
    size_t resets = 0;
    size_t fallbacks = 0;
    
    // Recycling statistics
    size_t slots_recycled = 0;
    size_t slots_reused = 0;
    size_t bytes_recycled = 0;
    size_t bytes_reused = 0;
    
    // NEW: Layer-cycling statistics
    size_t generation_mismatches = 0;
    int last_layer_accessed = -1;
    size_t out_of_order_accesses = 0;
    
#if Q8_CACHE_DIAGNOSTICS
    // DIAGNOSTICS: Counters for understanding cache behavior (not verbose logging)
    struct diag_counters {
        // Candidate checks
        size_t candidate_checked = 0;      // Total is_cache_candidate calls
        size_t candidate_passed = 0;       // Passed pattern matching
        size_t candidate_rejected = 0;     // Rejected (not norm tensor)
        
        // Allocation attempts
        size_t alloc_attempted = 0;        // Tensors we tried to cache
        size_t alloc_succeeded = 0;        // Successfully allocated
        size_t alloc_failed_full = 0;      // Failed - arena full
        
        // Lookup results
        size_t lookup_attempted = 0;       // Total lookups
        size_t lookup_hit = 0;             // Found in cache
        size_t lookup_miss_notfound = 0;   // Not in entries map
        size_t lookup_miss_dim = 0;        // Found but dimensions mismatch
        
        // Store results
        size_t store_called = 0;           // Total store calls
        
        // Unique tensor name patterns encountered (for pattern expansion)
        std::unordered_set<std::string> rejected_patterns;
    } diag;
    
    // PERSISTENT counters that survive reset() - track across all graphs
    size_t persistent_hits = 0;        // Total hits ever
    size_t persistent_lookups = 0;     // Total lookups ever
    size_t persistent_stores = 0;      // Total stores ever
#endif
    
    // Initialize the arena - single allocation
    void init() {
        if (base_ptr != nullptr) {
            return;  // Already initialized
        }
        
        // Single allocation for entire arena
        Q8_CACHE_MALLOC(&base_ptr, TOTAL_SIZE);
        
        // Initialize slots
        char* slot_base = static_cast<char*>(base_ptr);
        for (int i = 0; i < NUM_SLOTS; i++) {
            slots[i].base_ptr = slot_base;
            slots[i].bump_ptr = slot_base;
            slots[i].owner_group = -1;
            slots[i].generation = 0;
            slots[i].used_bytes = 0;
            slots[i].peak_bytes = 0;
            slots[i].max_peak_bytes = 0;
            slots[i].is_active = false;
            slots[i].free_list.clear();
            slot_base += SLOT_SIZE;
        }
        
        fprintf(stderr, "[Q8 Cache Arena] Layer-cycling: %zu MB, %d slots x %zu MB @ %p\n",
                TOTAL_SIZE / (1024*1024), NUM_SLOTS, SLOT_SIZE / (1024*1024), base_ptr);
    }
    
    // Cleanup
    void cleanup() {
        // Calculate total peak across all slots (using true peak)
        size_t total_peak = 0;
        for (const auto& slot : slots) {
            total_peak += slot.get_true_peak();
        }
        fprintf(stderr, "[Q8 Cache Arena] Stats: hits=%zu, misses=%zu, cached=%zu, candidates=%zu, fallbacks=%zu, peak=%.1f MB\n",
                cache_hits, cache_misses, cached_count, cache_candidate_names.size(), fallbacks, total_peak / (1024.0*1024.0));
        
        // Show recycling summary
        if (slots_recycled > 0 || generation_mismatches > 0) {
            fprintf(stderr, "[Q8 Cache Arena] RECYCLING: %zu slots recycled (%.1f MB), %zu reused (%.1f MB), gen_mismatch=%zu\n",
                    slots_recycled, bytes_recycled / (1024.0*1024.0),
                    slots_reused, bytes_reused / (1024.0*1024.0),
                    generation_mismatches);
        }
        
#if Q8_CACHE_DIAGNOSTICS
        print_diagnostics();
        
        // Final persistent stats
        fprintf(stderr, "\n[Q8 Cache FINAL PERSISTENT STATS]\n");
        fprintf(stderr, "  Total stores: %zu\n", persistent_stores);
        fprintf(stderr, "  Total lookups: %zu\n", persistent_lookups);
        fprintf(stderr, "  Total hits: %zu (%.1f%% hit rate)\n",
                persistent_hits,
                persistent_lookups > 0 ? (100.0 * persistent_hits / persistent_lookups) : 0.0);
#endif
        
        if (base_ptr != nullptr) {
            Q8_CACHE_FREE(base_ptr);
            base_ptr = nullptr;
        }
        entries.clear();
        cache_candidate_names.clear();
    }
    
    // Reset for new graph
    void reset() {
#if Q8_CACHE_DIAGNOSTICS
        // Print diagnostics before reset (if we have data)
        if (resets > 0 && diag.candidate_checked > 0) {
            print_diagnostics();
        }
#endif
        
        // Reset all slots (full reset including free_list)
        for (auto& slot : slots) {
            slot.full_reset();
            // NOTE: generation is NOT reset (distinguishes across graphs)
        }
        
        entries.clear();
        layer_cache.clear();  // Clear layer extraction cache
        cache_candidate_names.clear();
        cached_count = 0;
        resets++;
        
#if Q8_CACHE_DIAGNOSTICS
        // Reset diagnostic counters (per-graph)
        diag = diag_counters();
        // NOTE: persistent_* counters are NOT reset - they track across all graphs
#endif
    }
    
    // Align size to boundary
    static size_t align_size(size_t size) {
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
    
    // NEW: Layer number extraction from tensor name (e.g., "attn_norm-47" → 47)
    int get_layer_from_tensor(const ggml_tensor* tensor) {
        if (!tensor || tensor->name[0] == '\0') return -1;
        
        // Fast path: check cache
        auto it = layer_cache.find(tensor);
        if (it != layer_cache.end()) {
            return it->second;
        }
        
        // Parse: find last dash, extract number
        const char* name = tensor->name;
        const char* dash = strrchr(name, '-');
        if (!dash || !isdigit(dash[1])) {
            layer_cache[tensor] = -1;
            return -1;
        }
        
        int layer = atoi(dash + 1);
        layer_cache[tensor] = layer;
        return layer;
    }
    
    // NEW: Compute slot index from layer number
    static inline int get_slot_for_layer(int layer) {
        if (layer < 0) return 0;  // Non-layered tensors → slot 0
        int group = layer / LAYERS_PER_SLOT;
        return group % NUM_SLOTS;
    }
    
    // NEW: Layer-aware allocation with slot cycling
    void* allocate(const ggml_tensor* tensor, size_t size) {
        int layer = get_layer_from_tensor(tensor);
        int slot_idx = get_slot_for_layer(layer);
        q8_cache_slot& slot = slots[slot_idx];
        
        int current_group = (layer >= 0) ? layer / LAYERS_PER_SLOT : 0;
        
        // Check for slot ownership change (cycle detection)
        if (slot.is_active && slot.owner_group != current_group) {
            slot.change_ownership(current_group);
            slots_recycled++;
        }
        
        if (!slot.is_active) {
            slot.change_ownership(current_group);
        }
        
        size_t aligned_size = align_size(size);
        
        // STEP 1: Try slot's free-list first (BEST-FIT)
        auto& free_list = slot.free_list;
        size_t best_idx = free_list.size();
        size_t best_waste = SIZE_MAX;
        
        for (size_t i = 0; i < free_list.size(); ++i) {
            if (free_list[i].second >= aligned_size) {
                size_t waste = free_list[i].second - aligned_size;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_idx = i;
                    if (waste == 0) break;  // Perfect fit
                }
            }
        }
        
        if (best_idx < free_list.size()) {
            void* result = free_list[best_idx].first;
            // Swap-pop for O(1) removal
            if (best_idx != free_list.size() - 1) {
                free_list[best_idx] = free_list.back();
            }
            free_list.pop_back();
            
            slots_reused++;
            bytes_reused += aligned_size;
            return result;
        }
        
        // STEP 2: Bump allocate within slot
        if (slot.used_bytes + aligned_size > SLOT_SIZE) {
            return nullptr;  // Slot full
        }
        
        void* result = slot.bump_ptr;
        slot.bump_ptr += aligned_size;
        slot.used_bytes += aligned_size;
        
        if (slot.used_bytes > slot.peak_bytes) {
            slot.peak_bytes = slot.used_bytes;
        }
        
        return result;
    }
    
    // BACKWARD COMPATIBILITY: Simple size-based allocation (goes to slot 0)
    void* allocate(size_t size) {
        // Direct allocation from slot 0 for non-layered tensors
        size_t aligned_size = align_size(size);
        q8_cache_slot& slot = slots[0];
        
        // Try free-list first
        auto& free_list = slot.free_list;
        for (size_t i = 0; i < free_list.size(); ++i) {
            if (free_list[i].second >= aligned_size) {
                void* result = free_list[i].first;
                // Swap-pop
                if (i != free_list.size() - 1) {
                    free_list[i] = free_list.back();
                }
                free_list.pop_back();
                return result;
            }
        }
        
        // Bump allocate
        if (slot.used_bytes + aligned_size > SLOT_SIZE) {
            return nullptr;
        }
        
        void* result = slot.bump_ptr;
        slot.bump_ptr += aligned_size;
        slot.used_bytes += aligned_size;
        return result;
    }
    
    // NEW: Recycle a slot after final use (per-slot free_list)
    void recycle_slot(q8_cache_entry& entry, int slot_idx) {
        if (!entry.is_active) return;
        
        entry.is_active = false;
        
        // Add to slot's free_list
        if (slot_idx >= 0 && slot_idx < NUM_SLOTS) {
            slots[slot_idx].free_list.push_back({entry.ptr, entry.size});
        }
        
        slots_recycled++;
        bytes_recycled += entry.size;
    }
    
    // Check if a tensor is already cached
    const q8_cache_entry* lookup(const ggml_tensor* tensor, int layout,
                                  int64_t ne10p, int64_t ne11, int64_t ne12, int64_t ne13) {
#if Q8_CACHE_DIAGNOSTICS
        diag.lookup_attempted++;
#endif
        
        // NEW: Get layer and slot for validation
        int layer = get_layer_from_tensor(tensor);
        int slot_idx = get_slot_for_layer(layer);
        const q8_cache_slot& slot = slots[slot_idx];
        
        // Track out-of-order access
        if (layer >= 0) {
            if (last_layer_accessed > layer) {
                out_of_order_accesses++;
            }
            last_layer_accessed = layer;
        }
        
        // Lookup with layer in key
        auto it = entries.find({tensor->name, layout, layer});
        if (it != entries.end()) {
            q8_cache_entry& e = it->second;
            
            // NEW: Generation check - slot may have been recycled
            if (e.generation != slot.generation) {
                generation_mismatches++;
                cache_misses++;
#if Q8_CACHE_DIAGNOSTICS
                diag.lookup_miss_notfound++;
                persistent_lookups++;
#endif
                return nullptr;
            }
            
            // Verify dimensions match
            if (e.ne10_padded == ne10p && e.ne11 == ne11 && 
                e.ne12 == ne12 && e.ne13 == ne13) {
                // Skip recycled entries
                if (!e.is_active) {
                    cache_misses++;
#if Q8_CACHE_DIAGNOSTICS
                    diag.lookup_miss_notfound++;
                    persistent_lookups++;
#endif
                    return nullptr;
                }
                
                cache_hits++;
                e.use_count++;
                
                // Check if this was the final expected use
                if (e.use_count >= e.expected_consumers) {
                    recycle_slot(e, e.slot_idx);
                }
                
#if Q8_CACHE_DIAGNOSTICS
                diag.lookup_hit++;
                persistent_hits++;
                persistent_lookups++;
#endif
                return &e;
            }
#if Q8_CACHE_DIAGNOSTICS
            diag.lookup_miss_dim++;
            persistent_lookups++;
#endif
        } else {
#if Q8_CACHE_DIAGNOSTICS
            diag.lookup_miss_notfound++;
            persistent_lookups++;
#endif
        }
        cache_misses++;
        return nullptr;
    }
    
    // TRUE MULTI-CONSUMER: Analyze graph to find tensors with 2+ MUL_MAT consumers
    // This is called from ggml-cuda.cu where ggml headers are fully available
    void analyze_graph(const ggml_cgraph* cgraph);
    
    // Check if tensor is a cache candidate (has multiple consumers)
    // Uses tensor NAME pattern matching for MoE models
    // Caches ANY norm tensor that might be reused
    bool is_cache_candidate(const ggml_tensor* tensor, int layout) {
        if (!tensor || tensor->name[0] == '\0') {
            return false;
        }
        
        const char* name = tensor->name;
        
#if Q8_CACHE_DIAGNOSTICS
        diag.candidate_checked++;
#endif
        
        // Pattern match: various norm tensor naming conventions
        bool is_norm = 
            // Standard Qwen/DeepSeek naming
            (strncmp(name, "attn_norm-", 10) == 0 || 
             strncmp(name, "ffn_norm-", 9) == 0) ||
            // GPT-OSS naming
            (strncmp(name, "attn_post_norm-", 15) == 0 ||
             strncmp(name, "attn_pre_norm-", 14) == 0) ||
            // Generic norm patterns
            (strstr(name, "_norm") != nullptr && (
                strstr(name, "attn") != nullptr ||
                strstr(name, "ffn") != nullptr ||
                strstr(name, "input") != nullptr
            ));
        
#if Q8_CACHE_DIAGNOSTICS
        if (is_norm) {
            diag.candidate_passed++;
        } else {
            diag.candidate_rejected++;
            // Track rejected patterns that contain "norm" for pattern expansion
            if (strstr(name, "norm") != nullptr) {
                diag.rejected_patterns.insert(name);
            }
        }
#endif
        return is_norm;
    }
    
    // Store entry in cache
    // USE-COUNT RECYCLING: Set expected consumers from analysis
    void store(const ggml_tensor* tensor, int layout, void* ptr, size_t size,
               int64_t ne10p, int64_t ne11, int64_t ne12, int64_t ne13) {
        // NEW: Get layer and slot
        int layer = get_layer_from_tensor(tensor);
        int slot_idx = get_slot_for_layer(layer);
        q8_cache_slot& slot = slots[slot_idx];
        
        q8_cache_entry entry;
        entry.ptr = ptr;
        entry.size = align_size(size);  // Store aligned size for reuse
        entry.ne10_padded = ne10p;
        entry.ne11 = ne11;
        entry.ne12 = ne12;
        entry.ne13 = ne13;
        entry.use_count = 0;  // Will be incremented on first lookup
        entry.is_active = true;
        
        // NEW: Tag with slot info
        entry.generation = slot.generation;
        entry.slot_idx = slot_idx;
        
        // Get expected consumers from analysis
        auto it = consumer_counts.find(tensor->name);
        if (it != consumer_counts.end()) {
            entry.expected_consumers = it->second;
        } else {
            // Conservative default for pattern-matched tensors not in analysis
            entry.expected_consumers = 2;
        }
        
        // Store with layer in key
        entries[{tensor->name, layout, layer}] = entry;
        cached_count++;
        
#if Q8_CACHE_DIAGNOSTICS
        diag.store_called++;
        persistent_stores++;
#endif
    }
    
#if Q8_CACHE_DIAGNOSTICS
    // Print diagnostic summary (concise)
    void print_diagnostics() const {
        if (diag.candidate_checked == 0) return;
        
        fprintf(stderr, "\n[Q8 Cache DIAGNOSTICS]\n");
        
        // Slot info
        fprintf(stderr, "  Slots: %d x %zu MB, Layers/slot: %d\n", 
                NUM_SLOTS, SLOT_SIZE / (1024*1024), LAYERS_PER_SLOT);
        
        // Candidate analysis
        fprintf(stderr, "  Candidate checks: %zu passed, %zu rejected (of %zu total)\n",
                diag.candidate_passed, diag.candidate_rejected, diag.candidate_checked);
        
        // Allocation analysis
        if (diag.alloc_attempted > 0) {
            fprintf(stderr, "  Allocations: %zu succeeded, %zu failed (of %zu attempted)\n",
                    diag.alloc_succeeded, diag.alloc_failed_full, diag.alloc_attempted);
            if (diag.alloc_failed_full > 0) {
                fprintf(stderr, "    -> ARENA FULL: %zu tensors couldn't be cached!\n", diag.alloc_failed_full);
            }
        }
        
        // Lookup analysis
        if (diag.lookup_attempted > 0) {
            fprintf(stderr, "  Lookups: %zu hit, %zu miss (of %zu total)\n",
                    diag.lookup_hit, diag.lookup_miss_notfound + diag.lookup_miss_dim, 
                    diag.lookup_attempted);
            if (diag.lookup_miss_notfound > 0) {
                fprintf(stderr, "    -> Miss not found: %zu (tensor never cached)\n", diag.lookup_miss_notfound);
            }
            if (diag.lookup_miss_dim > 0) {
                fprintf(stderr, "    -> Miss dimension mismatch: %zu (cached but different dims)\n", diag.lookup_miss_dim);
            }
        }
        
        // Layer-cycling stats
        if (generation_mismatches > 0) {
            fprintf(stderr, "  Generation mismatches: %zu (slot cycles)\n", generation_mismatches);
        }
        if (out_of_order_accesses > 0) {
            fprintf(stderr, "  Out-of-order accesses: %zu\n", out_of_order_accesses);
        }
        
        // Store analysis
        fprintf(stderr, "  Store calls: %zu (should match alloc_succeeded)\n", diag.store_called);
        
        // Recycling analysis
        if (slots_recycled > 0) {
            fprintf(stderr, "  Recycling: %zu recycled, %zu reused\n",
                    slots_recycled, slots_reused);
        }
        
        // Rejected patterns
        if (!diag.rejected_patterns.empty()) {
            fprintf(stderr, "  Rejected 'norm' patterns (consider adding):\n");
            int count = 0;
            for (const auto& p : diag.rejected_patterns) {
                if (count++ >= 10) {
                    fprintf(stderr, "    ... and %zu more\n", diag.rejected_patterns.size() - 10);
                    break;
                }
                fprintf(stderr, "    - '%s'\n", p.c_str());
            }
        }
        
        // Health check
        fprintf(stderr, "\n  Health Check:\n");
        if (diag.lookup_hit == 0 && diag.store_called > 0) {
            fprintf(stderr, "    [WARNING] Stored %zu tensors but 0 hits!\n", diag.store_called);
        }
        if (diag.alloc_failed_full > 0) {
            fprintf(stderr, "    [WARNING] Arena too small - %zu allocations failed!\n", diag.alloc_failed_full);
        }
        
        // PERSISTENT counters
        fprintf(stderr, "\n  PERSISTENT (across all graphs):\n");
        fprintf(stderr, "    Stores: %zu, Lookups: %zu, Hits: %zu (%.1f%%)\n",
                persistent_stores, persistent_lookups, persistent_hits,
                persistent_lookups > 0 ? (100.0 * persistent_hits / persistent_lookups) : 0.0);
        
        fprintf(stderr, "\n");
    }
#endif
};

// Multi-consumer fusion info (unchanged)
struct prequantized_q8_info {
    char* buffer_ptr;
    int64_t ne10, ne11, ne12, ne13;
};
