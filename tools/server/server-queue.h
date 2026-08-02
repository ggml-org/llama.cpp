#pragma once

#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Overlap scheduler: a background worker that pre-stages CPU-only preparation
// for tasks while the main inference loop is busy on the GPU. This mirrors the
// spirit of TRT-LLM's overlap scheduler (prepare iteration N+1 while iteration
// N decodes) without splitting into a separate process: the main loop stays the
// single GPU-submission thread, and only pure-function work that touches no
// slot/llama_context state is allowed to run here.
//
// The first staged artifact is the stable cache-key vector used by the
// block-radix prefix cache and the serialized prompt cache. Computing it is
// O(prompt length) and previously ran on the inference thread inside
// get_available_slot(); staging it on the worker removes that cost from the
// admission critical path whenever a request waited in the queue.
//
// Fine-grained GPU/CPU overlap via Metal shared events or CUDA streams is left
// as a TODO: it requires double-buffered slot/batch state which is a larger
// change than is safe to make here.
struct server_prep_pool {
    // A single staged prep request. Holds only the data needed to derive
    // stable cache keys without referencing the (move-only, shared) task
    // object: the token ids and whether multimodal chunks are present. Text
    // prompts are fully cacheable here; multimodal prompts are skipped and
    // derived inline on the inference thread.
    struct request {
        int                      task_id  = -1;
        bool                     has_mtmd = false;
        std::vector<llama_token> tokens;
    };

    // Process staged requests. Called from the worker thread; the result is
    // typically stored by the callback in a thread-safe map keyed by task_id.
    using prep_fn = std::function<void(const request &)>;

    server_prep_pool() = default;
    ~server_prep_pool();

    // Start the worker thread. idempotent. Returns false if the thread could
    // not be started (caller falls back to inline prep).
    bool start(prep_fn callback);

    // Stop the worker and join. safe to call from any thread.
    void stop();

    // Submit a request for background staging. Moves the snapshot. If the
    // worker is not running, the request is dropped (caller will compute
    // inline on demand).
    void submit(request && req);

private:
    void loop();

    std::mutex              mutex;
    std::condition_variable cv;
    std::thread             worker;
    std::atomic<bool>       running { false };
    std::atomic<bool>       stop_requested { false };
    std::deque<request>     queue;
    prep_fn                 fn;
};

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
    //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
    void on_sleeping_state(std::function<void(bool)> callback) {
        if (callback_sleeping_state) {
            auto prev_callback = std::move(callback_sleeping_state);
            callback_sleeping_state = [prev_callback, callback](bool sleeping) {
                prev_callback(sleeping);
                callback(sleeping);
            };
        } else {
            callback_sleeping_state = std::move(callback);
        }
    }

private:
    void cleanup_pending_task(int id_target);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
