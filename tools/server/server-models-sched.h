#pragma once

#include "server-common.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// when to load and unload model instances: request queue (--models-max) and memory accounting (--models-memory-margin)
// see "Memory-based unloading" in the README

struct server_models;

// memory of a model instance per device, in bytes; from a measure child or from the running instance
struct server_model_memory {
    struct device_t {
        std::string kind;    // "device", "host" or "buffer_type"
        std::string name;    // ggml device name, "Host" for host memory
        size_t total   = 0;  // device memory, 0 unless kind == "device"
        size_t free    = 0;  // free device memory when this was measured, 0 unless kind == "device"
        size_t model   = 0;  // held by the instance, by kind of buffer
        size_t context = 0;
        size_t compute = 0;

        bool   is_device() const { return kind == "device"; }
        size_t self()      const { return model + context + compute; }
    };

    uint32_t n_ctx = 0; // context size this was measured at
    std::vector<device_t> devices;

    server_model_memory() = default;
    server_model_memory(uint32_t n_ctx, const std::vector<struct common_memory_breakdown_row> & rows);

    json to_json() const;
    static server_model_memory from_json(const json & j); // throws on a malformed input
};

// memory measurement state of a model
enum server_model_memory_status {
    SERVER_MODEL_MEMORY_UNMEASURED, // queued for a measure child
    SERVER_MODEL_MEMORY_MEASURING,  // its measure child is running
    SERVER_MODEL_MEMORY_MEASURED,
    SERVER_MODEL_MEMORY_FAILED,     // measure failed, no memory limit for this model
};

static std::string server_model_memory_status_to_string(server_model_memory_status status) {
    switch (status) {
        case SERVER_MODEL_MEMORY_UNMEASURED: return "unmeasured";
        case SERVER_MODEL_MEMORY_MEASURING:  return "measuring";
        case SERVER_MODEL_MEMORY_MEASURED:   return "measured";
        case SERVER_MODEL_MEMORY_FAILED:     return "failed";
        default:                             return "unknown";
    }
}

struct server_model_memory_info {
    server_model_memory_status status = SERVER_MODEL_MEMORY_UNMEASURED;
    int n_attempts = 0; // to limit the retries
    std::optional<server_model_memory> min       = std::nullopt; // at the smallest context --fit may pick
    std::optional<server_model_memory> max       = std::nullopt; // at the largest context --fit picks, a snapshot
    std::optional<server_model_memory> effective = std::nullopt; // what the running instance allocated, reset when it exits

    bool is_settled() const {
        return status == SERVER_MODEL_MEMORY_MEASURED || status == SERVER_MODEL_MEMORY_FAILED;
    }

    json to_json() const;
};

// functions taking lk need models.mutex held by lk
struct server_models_sched {
    server_models_sched(server_models & models) : models(models) {}

    //
    // request queue (--models-max)
    //

    bool has_capacity(std::unique_lock<std::mutex> & lk);

    // returns "" if no model can be given up
    std::string pick_victim(std::unique_lock<std::mutex> & lk);

    // requests wanting the same model share one entry, so they all need only one slot
    // and all get unblocked by the single load that entry performs
    void join(std::unique_lock<std::mutex> & lk, const std::string & model_id);
    void leave(std::unique_lock<std::mutex> & lk, const std::string & model_id);
    bool queue_empty(std::unique_lock<std::mutex> & lk);

    // true if it is this model's turn to load, and nobody is loading it yet
    bool try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id);

    // on failure the entry is back in line; on success it stays until its waiters leave,
    // so the model coming up is never picked as a victim before they use it
    void claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok);

    // evict idle models while queued requests outnumber the slots that are free or being freed
    // never blocks, so it is safe from any thread
    void tick(std::unique_lock<std::mutex> & lk);

    //
    // loading a model
    //

    // true if loading name now exceeds models_max or the memory margin
    bool limits_exceeded(std::unique_lock<std::mutex> & lk, const std::string & name, bool verbose = true);

    // wait for the measure of name, then unload idle LRU models until it fits; call without models.mutex
    void make_room(const std::string & name);

    //
    // memory accounting (--models-memory-margin), the router never touches a GPU
    // track what each instance holds, child reports only calibrate the rest: works for per-process (Metal) and system-wide (CUDA) free memory
    //

    bool memory_enabled() const;

    // queue a measure child for every model not measured yet
    void measure_unmeasured(std::unique_lock<std::mutex> & lk);

    // count a new instance at its min until it reports
    void on_spawned(std::unique_lock<std::mutex> & lk, const std::string & name);

    // memory: what the instance allocated, nullopt on wake up from sleep
    void on_ready(std::unique_lock<std::mutex> & lk, const std::string & name, const std::optional<server_model_memory> & memory);
    // a sleeping instance frees its model and context
    void on_sleeping(std::unique_lock<std::mutex> & lk, const std::string & name);
    void on_unloaded(std::unique_lock<std::mutex> & lk, const std::string & name);

    // stop the measure child and drop the queue, for shutdown
    void cancel_measurements(std::unique_lock<std::mutex> & lk);
    bool is_measuring(std::unique_lock<std::mutex> & lk);

    // called from the monitor thread, take models.mutex themselves
    void handle_measured(const std::string & name, const json & payload);
    void on_measure_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, int exit_code);

  private:
    struct entry_t {
        std::string model_id;
        int  n_waiters; // requests waiting for this model
        bool loading;   // one of the waiters is doing the load right now
    };

    struct device_memory_t {
        int64_t total    = 0; // bytes
        int64_t external = 0; // bytes not tracked by the router: other processes, backend contexts
    };

    server_models & models;

    // requests waiting for a slot
    std::deque<entry_t> queue;

    // one measure child runs at a time
    std::deque<std::string> measure_queue;
    std::string measuring; // model whose measure child is running, empty when idle
    std::shared_ptr<server_subproc> measure_proc;

    std::map<std::string, device_memory_t> device_memory; // by device name, empty until the first child reports
    std::map<std::string, std::map<std::string, size_t>> held; // model -> device -> bytes counted for its instance

    entry_t * find(const std::string & model_id);
    void check_lock(std::unique_lock<std::mutex> & lk);
    size_t count_running();

    // all below need models.mutex
    void enqueue_measure(const std::string & name, bool front = false);
    void measure_next();
    // update device_memory from a child report; reporter: the instance that sent it, empty for a measure child
    void calibrate(const server_model_memory & mem, const std::string & reporter);
    void reserve(const std::string & name); // count the instance's memory (effective, else min)
    void release(const std::string & name);
    int64_t available(const std::string & dev) const; // total - external - what the instances hold
    // bytes missing per device to load name at its min, empty if it fits
    std::map<std::string, size_t> deficit(const std::string & name) const;
};
