#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;

// pre-load handoff between the model loader and the expert hot store.
//
// when the tier is active (--expert-hot-s set), the model loader does not
// allocate the expert weight tensors at all; instead it streams their data
// into two buffers owned here: a GPU store holding the first S experts of
// every layer, and a host buffer holding the remaining (cold) experts. this
// keeps the experts entirely outside the model's own buffer plan, so a model
// larger than RAM can load when the GPU store supplies the missing capacity.
// the hotstore adopts the GPU store at context init; the cold op reads the
// host buffer through a slice hook.
namespace llama_expert_preload {

    void set_slots(int s);
    int  get_slots();
    void set_model_path(const char * path); // for the debug disk hash
    void set_no_evict(bool no_evict);       // --expert-no-evict
    bool get_no_evict();

    // true when the tier will actually build a hot store: slots requested AND
    // (forced OR a CUDA device is present). mirrors the gate in llama-context.cpp.
    bool tier_will_engage();

    // backend-agnostic 256-byte alignment for the store's LUT region: some
    // backends (Vulkan) require minStorageBufferOffsetAlignment for get_rows
    // sources. returns the total store bytes (entries + padded LUTs).
    size_t store_total_bytes(size_t entries_bytes, int n_layers, int n_experts, int slots);
    size_t align_up256(size_t x);

    // true if `name` matches an exps weight tensor; sets layer_idx on match
    bool is_exps(const char * name, int & layer_idx);

    struct entry {
        const ggml_tensor * src;
        size_t plane_bytes;  // bytes of one expert slice in this tensor
        size_t gpu_offset;   // offset of this entry's planes in the store buffer
        size_t cpu_offset;   // offset of this entry's cold experts in the host buffer
        size_t file_off;     // absolute gguf data offset of this tensor
        int    fd;           // model file descriptor (for the debug disk read)
        int    n_experts;
        int    startup;      // first S experts (live on the GPU)
    };

    // loader side. begin() allocates the two buffers (single call, after the
    // loader computed the sizes); register_tensor() adds an entry; write_entry()
    // writes the startup slices to the store; write_cold() writes the rest to
    // the host buffer.
    void   begin(struct ggml_backend_buffer_type * buft, size_t gpu_bytes, size_t cpu_bytes, int max_entries);
    size_t register_tensor(const ggml_tensor * src, size_t plane_bytes, int n_experts, int startup,
                           size_t file_off, int fd);
    bool   write_entry(size_t idx, const uint8_t * data, size_t nbytes);
    bool   read_expert(size_t idx, int expert, void * out, size_t n); // from the gguf file
    bool   write_cold(size_t idx, const uint8_t * data, size_t nbytes);

    // hotstore + cold-op side.
    struct ggml_backend_buffer * take_buffer();
    size_t num_entries();
    const entry * entry_at(size_t idx);
    int    index_of(const ggml_tensor * src);
    size_t entries_size(); // bytes used by the entry regions in the store

    // cold expert slice access (nullptr when the slice is on the GPU)
    const uint8_t * cpu_slice(size_t idx, int expert);
    void free_cpu_slice(size_t idx, int expert);
    void set_cpu_slice(size_t idx, int expert, const uint8_t * data);

    // debug: ground-truth FNV-1a of the first 1024 bytes of an expert slice,
    // taken straight from the GGUF at load. compare against the hash of the
    // slice actually read to verify a copy/routing is correct.
    uint64_t expected_hash(const ggml_tensor * src, int expert);

    void clear();

} // namespace llama_expert_preload
