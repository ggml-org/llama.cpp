#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <memory>
#include <cstdint>
#include <vector>

struct llama_recurrent_spill_layer {
    ggml_type type_r = GGML_TYPE_F32;
    ggml_type type_s = GGML_TYPE_F32;
    uint64_t width_r = 0;
    uint64_t width_s = 0;
    bool enabled = false;
};

struct llama_recurrent_spill_layer_layout {
    bool enabled = false;
    ggml_type type_r = GGML_TYPE_F32, type_s = GGML_TYPE_F32;
    size_t r_row_bytes = 0, s_row_bytes = 0;
    uint64_t width_r = 0, width_s = 0;
    size_t r_offset = 0, s_offset = 0;
};

struct llama_recurrent_spill_layout {
    uint32_t cells = 0, depth = 0;
    std::vector<llama_recurrent_spill_layer_layout> layers;
    size_t device_current_bytes = 0;
    size_t host_snapshot_bytes = 0;
    size_t stock_device_bytes = 0;
    size_t device_stage_bytes = 0;
};

// Model and backend row types determine exact bytes; failure leaves output unchanged.
bool llama_recurrent_spill_layout_make(const std::vector<llama_recurrent_spill_layer> & layers,
        uint32_t cells, uint32_t depth, llama_recurrent_spill_layout & output);

struct llama_recurrent_stage_layer {
    bool enabled = false;
    size_t r_snapshot_bytes = 0;
    size_t s_snapshot_bytes = 0;
    size_t r_stride = 0;
    size_t s_stride = 0;
    size_t s_offset = 0;
};

struct llama_recurrent_stage_layout {
    uint32_t slots = 0;
    uint32_t depth = 0;
    size_t alignment = 1;
    size_t bytes_per_slot = 0;
    size_t total_bytes = 0;
    std::vector<llama_recurrent_stage_layer> layers;
};

bool llama_recurrent_stage_layout_make(const llama_recurrent_spill_layout & snapshots,
        uint32_t slots, size_t alignment, llama_recurrent_stage_layout & output);
bool llama_recurrent_stage_region(const llama_recurrent_stage_layout & layout,
        size_t layer, bool value, uint32_t snapshot, uint32_t slot, size_t & offset, size_t & bytes);


class llama_recurrent_stage_buffer {
public:
    static std::unique_ptr<llama_recurrent_stage_buffer> create(
            const llama_recurrent_spill_layout & snapshots, uint32_t slots,
            ggml_backend_buffer_type_t device_type);
    ~llama_recurrent_stage_buffer();
    llama_recurrent_stage_buffer(const llama_recurrent_stage_buffer &) = delete;
    llama_recurrent_stage_buffer & operator=(const llama_recurrent_stage_buffer &) = delete;

    ggml_tensor * tensor(size_t layer, bool value, uint32_t snapshot, uint32_t slot) const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    size_t bytes() const noexcept;
    const llama_recurrent_stage_layout & layout() const noexcept;
private:
    struct implementation;
    explicit llama_recurrent_stage_buffer(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};

struct llama_recurrent_spill_identity {
    uint64_t cache = 0;
    uint64_t sequence = 0;
    uint64_t generation = 0;
};

class llama_recurrent_spill_bank {
public:
    static std::unique_ptr<llama_recurrent_spill_bank> create(
            const llama_recurrent_spill_layout & layout, ggml_backend_buffer_type_t host_type,
            uint64_t cache, uint64_t sequence);
    ~llama_recurrent_spill_bank();
    llama_recurrent_spill_bank(const llama_recurrent_spill_bank &) = delete;
    llama_recurrent_spill_bank & operator=(const llama_recurrent_spill_bank &) = delete;

    bool begin_capture(llama_recurrent_spill_identity identity);
    bool enable_publication(ggml_backend_dev_t device, ggml_backend_buffer_type_t stage_type, uint32_t slots);
    bool staged_publication_enabled() const noexcept;
    size_t staged_device_bytes() const noexcept;
    ggml_tensor * publication_tensor(size_t layer, bool value, uint32_t snapshot) const noexcept;
    ggml_tensor * staging_tensor(size_t layer, bool value, uint32_t snapshot) const noexcept;
    bool begin_staged_capture(llama_recurrent_spill_identity identity, uint32_t tokens);
    bool stage_ready();

    bool capture(ggml_backend_t backend, size_t layer, bool value, uint32_t slot,
            const ggml_tensor * source, size_t source_offset);
    bool publish_capture();
    bool complete_capture();
    bool begin_restore(llama_recurrent_spill_identity identity, uint32_t slot);
    bool restore(ggml_backend_t backend, size_t layer, bool value,
            ggml_tensor * destination, size_t destination_offset);
    bool publish_restore();
    bool complete_restore();
    void cancel() noexcept;
    void reset(bool pristine) noexcept;
    size_t host_bytes() const noexcept;
    uint64_t snapshot_generation(size_t layer, bool value, uint32_t slot) const noexcept;
    bool snapshot_pristine(size_t layer, bool value, uint32_t slot) const noexcept;
    const void * snapshot_data(size_t layer, bool value, uint32_t slot, uint64_t generation) const noexcept;
private:
    struct implementation;
    explicit llama_recurrent_spill_bank(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};
