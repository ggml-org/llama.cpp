#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__) && defined(GGML_USE_METAL)

#include "ggml-metal.h"

#ifdef __cplusplus
extern "C" {
#endif

void * llama_moe_offloader_create_shared_event(ggml_backend_t backend);
void   llama_moe_offloader_signal_event(void * event, uint64_t value);
void   llama_moe_offloader_release_event(void * event);

#ifdef __cplusplus
}
#endif

// Shared memory layout (MTLStorageModeShared).
static constexpr int    MOE_MAX_IDS      = 4096;
static constexpr size_t MOE_OFF_REQ      = 0;                                   // atomic_uint: request seq
static constexpr size_t MOE_OFF_N        = 8;                                   // int32:       id count
static constexpr size_t MOE_OFF_SELECTED = 16;                                  // int32[n]:    expert ids (GPU writes)
static constexpr size_t MOE_OFF_REMAPPED = MOE_OFF_SELECTED + MOE_MAX_IDS * 4;  // int32[n]:    slot ids (CPU writes)
static constexpr size_t MOE_MSG_NBYTES   = MOE_OFF_REMAPPED + MOE_MAX_IDS * 4;

struct moe_pool {
    ggml_tensor *         tensor      = nullptr;
    ggml_context *        own_ctx     = nullptr;
    ggml_backend_buffer_t own_buf     = nullptr;
    uint64_t              file_offset = 0;
    int                   fd          = -1;
    size_t                stride      = 0;
};

struct moe_layer {
    int     layer_idx = -1;
    int64_t n_slots   = 0;
    int64_t n_expert  = 0;

    std::vector<moe_pool> pools;

    std::unordered_map<std::string, size_t> name_to_pool_idx;

    // lru state
    std::vector<int32_t>  expert_to_slot;
    std::vector<int32_t>  slot_to_expert;
    std::vector<uint32_t> lru_clock;
    uint32_t              lru_time = 0;
    std::vector<uint8_t>  in_use;

    // shmem message
    ggml_tensor *         msg_tensor = nullptr;
    ggml_context *        msg_ctx    = nullptr;
    ggml_backend_buffer_t msg_buf    = nullptr;
    uint8_t *             mapped     = nullptr;  // == msg_tensor->data

    // gpu-cpu sync
    void *                shared_event = nullptr;
    std::atomic<uint32_t> next_seq{ 0 };
    std::atomic<uint32_t> last_processed{ 0 };

    // reuse detection for gate_up/down etc.
    const ggml_tensor * last_src2      = nullptr;
    uint32_t            last_src2_seq  = 0;
    int                 last_src2_uses = 0;

    uint64_t total_hits   = 0;
    uint64_t total_misses = 0;

    ~moe_layer();
};

class llama_moe_offloader {
  public:
    llama_moe_offloader(ggml_backend_t backend);
    ~llama_moe_offloader();

    void start();
    void stop();

    ggml_tensor * bind_pool(int layer_idx, ggml_tensor * orig, int64_t n_slots, uint64_t file_offset, int fd);

    static bool hook(void *                     user_data,
                     const ggml_tensor *        src0,
                     const ggml_tensor *        src2,
                     ggml_metal_moe_intercept * out);

    void clear();

  private:
    static bool pread_pool(moe_layer & layer, size_t pool_idx, int32_t expert_id, int64_t dst_slot);

    static int64_t lru_evict(moe_layer & layer);

    static void resolve(moe_layer & layer, const int32_t * ids, int32_t * out, int n);

    void start_sidecar();
    void stop_sidecar();
    void sidecar_loop();

    ggml_backend_t backend;

    std::unordered_map<const ggml_tensor *, moe_layer *> pool_to_layer;

    std::mutex                              build_mtx;
    std::vector<std::unique_ptr<moe_layer>> layers;
    std::vector<moe_layer *>                sidecar_layers;

    std::thread       sidecar_thread;
    std::atomic<bool> sidecar_run{ false };
};

#else   // !(__APPLE__ && GGML_USE_METAL)

class llama_moe_offloader {
  public:
    llama_moe_offloader(ggml_backend_t) {}

    ~llama_moe_offloader() {}

    void start() {}

    void stop() {}

    ggml_tensor * bind_pool(int, ggml_tensor *, int64_t, uint64_t, int) { return nullptr; }

    void clear() {}
};

#endif  // __APPLE__ && GGML_USE_METAL
