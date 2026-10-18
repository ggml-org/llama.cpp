#pragma once

// live performance figures for GET /perf: a lazy sampler thread reads the machine, the process and the model
// counters about once a second while the endpoint is polled, and sleeps otherwise

#include "server-common.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// one prompt batch of a request: tokens and time (us)
struct server_perf_batch {
    int32_t n;
    int64_t us;
};

// the prompt batches of the running request, and of the earlier ones (the cost model of the projection)
struct server_perf_plan {
    std::vector<server_perf_batch> done;      // this request
    int32_t                        cur_n = 0; // the batch being decoded, 0: none
    std::vector<int32_t>           remaining; // after it, as the server will split the rest of the prompt
    std::vector<server_perf_batch> hist;      // earlier requests, oldest first
    bool                           valid = true; // false: the split is not known (media, several slots), no projection
};

// how the server splits the rest of a prompt into batches (mirrors the prompt loop of server_context)
struct server_prompt_plan_params {
    int32_t pos       = 0;     // first token of the next batch
    int32_t n_total   = 0;
    int32_t n_batch   = 0;
    int32_t n_ubatch  = 0;
    bool    checkpoints = false; // the request makes context checkpoints
    std::vector<int32_t> user_starts; // positions where user messages start, ascending
    int32_t last_user_pos = -1;
    int32_t last_ckpt = -1;    // n_tokens of the newest checkpoint, -1: none
    int32_t min_step  = 0;
    bool    mem_empty = false; // nothing in the memory before pos
};

std::vector<int32_t> server_prompt_plan(const server_prompt_plan_params & p);

// written by the server loop (relaxed atomics, no locks), read by the sampler
// the "current" fields describe the most recent request of any slot
struct server_perf_counters {
    std::atomic<uint64_t> n_gen_total{0};       // generated tokens, every slot
    std::atomic<uint64_t> n_prompt_total{0};    // processed prompt tokens, every slot
    std::atomic<uint64_t> prompt_busy_us{0};    // time spent in prompt batches

    std::atomic<bool>     generating{false};
    std::atomic<uint64_t> cur_gen_n{0};
    std::atomic<int64_t>  cur_gen_t_first_us{0}; // the first token comes from the last prompt batch
    std::atomic<int64_t>  cur_gen_t_last_us{0};

    std::atomic<bool>     processing{false};
    std::atomic<uint64_t> cur_prompt_n{0};
    std::atomic<int64_t>  cur_prompt_t_start_us{0};

    // the last finished request, as in its timings (steps exclude the first token)
    std::atomic<uint64_t> last_gen_n{0}, last_gen_steps{0}, last_gen_us{0};
    std::atomic<uint64_t> last_prompt_n{0}, last_prompt_us{0};

    // finished requests
    std::atomic<uint64_t> tot_gen_steps{0}, tot_gen_us{0}, tot_prompt_n{0}, tot_prompt_us{0};

    // plan and batch history, written once per batch (not on the graph path)
    mutable std::mutex plan_mtx;
    server_perf_plan plan;

    void on_prompt_start(int64_t t_start_us);
    // valid = false: the batches cannot be predicted, the live rate falls back to tokens done / elapsed
    void on_prompt_plan(int32_t n_cur, std::vector<int32_t> remaining, bool valid = true);
    void on_prompt_batch(const server_slot_stats & st, uint64_t n_tokens, int64_t t_batch_us);
    void on_gen(const server_slot_stats & st);
    void on_request_end(const server_slot_stats & st);
};

// one sample; has_* false when the source is not available
struct server_perf_tick {
    int64_t t_us  = 0; // ggml_time_us
    int64_t ts_ms = 0; // wall clock

    bool     has_cpu    = false;
    uint64_t cpu_total  = 0; // /proc/stat jiffies
    uint64_t cpu_idle   = 0; // idle + iowait
    uint64_t proc_ticks = 0; // utime + stime of this process

    bool     has_mem       = false;
    uint64_t mem_total     = 0;
    uint64_t mem_free      = 0;
    uint64_t mem_available = 0;
    uint64_t mem_cached    = 0;

    bool     has_gpu        = false;
    double   gpu_busy_pct   = 0.0;
    uint64_t gpu_mem_used   = 0;
    uint64_t gpu_mem_total  = 0;

    bool     has_disk     = false;
    uint64_t disk_sectors = 0; // sectors read, 512 bytes each

    bool     has_io        = false;
    uint64_t io_read_bytes = 0; // /proc/self/io

    // server counters
    uint64_t n_gen_total    = 0;
    uint64_t n_prompt_total = 0;
    uint64_t prompt_busy_us = 0;
    bool     generating     = false;
    uint64_t cur_gen_n      = 0;
    int64_t  cur_gen_t_first_us = 0;
    int64_t  cur_gen_t_last_us  = 0;
    bool     processing     = false;
    uint64_t cur_prompt_n   = 0;
    int64_t  cur_prompt_t_start_us = 0;
    uint64_t last_gen_n = 0, last_gen_steps = 0, last_gen_us = 0;
    uint64_t last_prompt_n = 0, last_prompt_us = 0;
    uint64_t tot_gen_steps = 0, tot_gen_us = 0, tot_prompt_n = 0, tot_prompt_us = 0;

    llama_decode_progress_data progress = {};
    server_perf_plan           plan;
    llama_moe_predict_data     predict  = {};
};

// totals count from here: taken right after the model is loaded
struct server_perf_baseline {
    bool     has_io        = false;
    uint64_t io_read_bytes = 0;
    llama_moe_predict_data predict = {};
};

struct server_perf_info {
    int         n_cpu   = 0;
    long        clk_tck = 100;
    bool        gpu_ok  = false;
    std::string gpu_name;
    bool        disk_ok = false;
    std::string disk_name;
};

// the /perf JSON from the samples (oldest first), pure for tests
json server_perf_compute(const std::deque<server_perf_tick> & ring, const server_perf_baseline & base,
        const server_perf_info & info, int64_t now_us);

struct server_perf {
    server_perf_counters counters;

    ~server_perf();

    // after the model is loaded: takes the baseline, starts the (idle) sampler thread
    void start(const llama_model * model, const llama_context * ctx, const std::string & model_path);
    // before the model is freed: joins the thread
    void stop();

    // from any HTTP thread: false when no model is loaded; a poll after idle waits for two fresh samples
    bool get(json & out);

private:
    void loop();
    server_perf_tick sample() const;

    const llama_model   * model = nullptr;
    const llama_context * ctx   = nullptr;

    server_perf_info     info;
    server_perf_baseline base;

    std::mutex              mtx;
    std::condition_variable cv;      // wakes the sampler
    std::condition_variable cv_tick; // a new sample is in the ring
    std::deque<server_perf_tick> ring;
    bool    running      = false;
    bool    nvml_checked = false; // sampler thread
    bool    stopping     = false;
    int64_t last_poll_us = 0;
    std::thread worker;
};
