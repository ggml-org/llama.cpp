#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct dist_comm_config {
    int         rank        = 0;
    int         world_size  = 1;
    std::string master_host;
    int         master_port = 0;
    int         listen_port = 0;
};

struct dist_comm_stats {
    uint64_t allreduce_calls = 0;
    uint64_t broadcast_calls = 0;
    uint64_t bytes_sent      = 0;
    uint64_t bytes_recv      = 0;
};

class dist_comm_context {
public:
    virtual ~dist_comm_context() = default;

    virtual bool exchange_ring_f32(int32_t src_rank, const float * send_data, size_t n,
            int32_t & recv_src_rank, float * recv_data, std::string & err) = 0;
    virtual bool broadcast_i32(int32_t & value, int root, std::string & err) = 0;
    virtual bool broadcast_vector_i32(std::vector<int32_t> & values, int root, std::string & err) = 0;
    virtual bool broadcast_vector_u8(std::vector<uint8_t> & values, int root, std::string & err) = 0;

    virtual dist_comm_stats get_stats() const = 0;
};

std::unique_ptr<dist_comm_context> dist_comm_create(const dist_comm_config & config, std::string & err);

struct common_dist_tp_env {
    bool        enabled     = false;
    int         world_size  = 1;
    int         world_rank  = 0;
    int         master_rank = 0;
    std::string master_host = "127.0.0.1";
    int         master_port = 0;
    int         listen_port = 0;
};

struct common_dist_tp_runtime;

// Reads distributed TP settings from environment variables:
// - LLAMA_DIST_TP_WORLD_SIZE : total participants (distributed mode when >= 2)
// - LLAMA_DIST_TP_WORLD_RANK : current world rank
// - LLAMA_DIST_TP_MASTER     : host:port for bootstrap service
// - LLAMA_DIST_TP_LISTEN_PORT: optional ring listener port for this rank
//                              (default: LLAMA_DIST_TP_MASTER port + rank)
// Optional toggles:
// - LLAMA_DIST_TP_MODEL_SPLIT_MODE: override model split mode (none|layer|row|tensor)
// - LLAMA_DIST_TP_MODEL_TENSOR_SPLIT: comma-separated tensor split values for model loading
bool common_dist_tp_env_load(common_dist_tp_env & out, std::string & err);
bool common_dist_tp_resolve(const common_params & params, common_dist_tp_env & out, std::string & err);

// Applies distributed TP environment overrides to model/context init parameters.
// Must run before common_init_from_params() so tensor placement and callbacks are set correctly.
bool common_dist_tp_apply_env_overrides(common_params & params, const common_dist_tp_env & env, std::string & err);

// Creates distributed TP runtime state (dist_comm backend + callback state) and binds eval callback into params.
// Must run before common_init_from_params().
common_dist_tp_runtime * common_dist_tp_runtime_create(common_params & params, const common_dist_tp_env & env, std::string & err);
void common_dist_tp_runtime_destroy(common_dist_tp_runtime * runtime);

// Attaches decode-batch synchronization callback after llama_context is created.
bool common_dist_tp_runtime_attach_context(common_dist_tp_runtime * runtime, llama_context * ctx, std::string & err);

// Runs non-master dist-peer worker loop: waits for rank0 batches via callback, executes llama_decode, and participates in allreduce.
int common_dist_tp_run_worker(common_init_result & llama_init, const common_dist_tp_env & env);
