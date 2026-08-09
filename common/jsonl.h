#pragma once

#include "chat.h"

#include "nlohmann/json_fwd.hpp"

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum common_jsonl_empty_line_policy {
    COMMON_JSONL_EMPTY_LINE_SKIP,
    COMMON_JSONL_EMPTY_LINE_KEEP,
};

enum common_jsonl_chat_parse_mode {
    COMMON_JSONL_CHAT_PARSE_TEXT,
    COMMON_JSONL_CHAT_PARSE_OPTIONAL_TEXT,
    COMMON_JSONL_CHAT_PARSE_STRICT_REASONING,
};

struct common_jsonl_line {
    std::string text;
    int64_t number = 0;
};

class common_jsonl_worker_pool {
public:
    explicit common_jsonl_worker_pool(size_t n_workers);
    ~common_jsonl_worker_pool();

    void parallel_for(size_t n_tasks, const std::function<void(size_t, size_t)> & fn);

private:
    void worker_loop(size_t worker_index);

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    std::function<void(size_t, size_t)> task;
    std::atomic<size_t> next_task { 0 };
    size_t task_count = 0;
    size_t workers_pending = 0;
    size_t generation = 0;
    std::exception_ptr exception;
    bool stop = false;
};

size_t common_jsonl_worker_limit(int32_t requested);
size_t common_jsonl_worker_count(int32_t requested, size_t n_tasks);

std::vector<common_jsonl_line> common_jsonl_read_lines(
        const std::string & path,
        common_jsonl_empty_line_policy empty_line_policy);

common_chat_msg common_jsonl_parse_chat_message(
        const nlohmann::json & data,
        common_jsonl_chat_parse_mode mode);

common_chat_msg common_jsonl_parse_chat_message(
        const nlohmann::ordered_json & data,
        common_jsonl_chat_parse_mode mode);
