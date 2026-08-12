#include "jsonl.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

size_t physical_core_count() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::set<std::pair<int, int>> cores;
        int package_id = -1;
        int core_id = -1;
        std::string line;

        auto add_core = [&] {
            if (package_id >= 0 && core_id >= 0) {
                cores.insert({ package_id, core_id });
            }
            package_id = -1;
            core_id = -1;
        };

        while (std::getline(cpuinfo, line)) {
            if (line.empty()) {
                add_core();
                continue;
            }

            const size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            const std::string key = line.substr(0, colon);
            if (key.find("physical id") != std::string::npos) {
                package_id = std::stoi(line.substr(colon + 1));
            } else if (key.find("core id") != std::string::npos) {
                core_id = std::stoi(line.substr(colon + 1));
            }
        }
        add_core();

        if (!cores.empty()) {
            return cores.size();
        }
    }
#endif

    return std::max(1u, std::thread::hardware_concurrency());
}

template <typename Json>
std::string required_string(const Json & object, const char * key, bool required) {
    auto it = object.find(key);
    if (it == object.end()) {
        if (required) throw std::runtime_error(std::string("missing '") + key + "'");
        return {};
    }
    if (!it->is_string()) throw std::runtime_error(std::string("'") + key + "' must be a string");
    return it->template get<std::string>();
}

template <typename Json>
common_chat_msg parse_chat_message(const Json & data, common_jsonl_chat_parse_mode mode) {
    if (!data.is_object()) {
        if (mode == COMMON_JSONL_CHAT_PARSE_OPTIONAL_TEXT) return {};
        throw std::runtime_error("each message must be an object");
    }

    common_chat_msg message;
    if (mode == COMMON_JSONL_CHAT_PARSE_TEXT || mode == COMMON_JSONL_CHAT_PARSE_OPTIONAL_TEXT) {
        message.role = data.value("role", "user");
        if (mode == COMMON_JSONL_CHAT_PARSE_TEXT || (data.contains("content") && data["content"].is_string())) {
            common_chat_msg_content_part part;
            part.type = "text";
            part.text = data.value("content", "");
            message.content_parts.push_back(std::move(part));
        }
        return message;
    }

    message.role = required_string(data, "role", true);
    if (common_chat_role_from_string(message.role) == COMMON_CHAT_ROLE_UNKNOWN) {
        throw std::runtime_error("unsupported role: " + message.role);
    }
    message.content = required_string(data, "content", message.role != "assistant");
    message.reasoning_content = required_string(data, "reasoning", false);
    if (data.contains("reasoning_content")) {
        if (!message.reasoning_content.empty()) throw std::runtime_error("use only one of 'reasoning' and 'reasoning_content'");
        message.reasoning_content = required_string(data, "reasoning_content", false);
    }
    if (message.role != "assistant" && !message.reasoning_content.empty()) {
        throw std::runtime_error("reasoning is supported only for assistant messages");
    }
    if (message.role == "assistant" && message.content.empty() && message.reasoning_content.empty()) {
        throw std::runtime_error("assistant message must contain 'content' or 'reasoning'");
    }
    return message;
}

}

common_jsonl_worker_pool::common_jsonl_worker_pool(size_t n_workers) {
    if (n_workers <= 1) return;

    workers.reserve(n_workers);
    for (size_t i = 0; i < n_workers; ++i) {
        workers.emplace_back([this, i] { worker_loop(i); });
    }
}

common_jsonl_worker_pool::~common_jsonl_worker_pool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    work_ready.notify_all();
    for (std::thread & worker : workers) worker.join();
}

void common_jsonl_worker_pool::parallel_for(size_t n_tasks, const std::function<void(size_t, size_t)> & fn) {
    if (n_tasks == 0) return;
    if (workers.empty()) {
        for (size_t i = 0; i < n_tasks; ++i) fn(i, 0);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        task = fn;
        task_count = n_tasks;
        next_task.store(0);
        workers_pending = workers.size();
        exception = nullptr;
        ++generation;
    }
    work_ready.notify_all();

    std::unique_lock<std::mutex> lock(mutex);
    work_done.wait(lock, [this] { return workers_pending == 0; });
    if (exception) std::rethrow_exception(exception);
}

void common_jsonl_worker_pool::worker_loop(size_t worker_index) {
    size_t worker_generation = 0;
    for (;;) {
        std::function<void(size_t, size_t)> current_task;
        size_t current_task_count = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            work_ready.wait(lock, [&] { return stop || generation != worker_generation; });
            if (stop) return;
            worker_generation = generation;
            current_task = task;
            current_task_count = task_count;
        }

        for (;;) {
            const size_t i = next_task.fetch_add(1);
            if (i >= current_task_count) break;
            try {
                current_task(i, worker_index);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!exception) exception = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (--workers_pending == 0) {
                task = {};
                work_done.notify_one();
            }
        }
    }
}

size_t common_jsonl_worker_limit(int32_t requested) {
    return requested > 0 ? (size_t) requested : physical_core_count();
}

size_t common_jsonl_worker_count(int32_t requested, size_t n_tasks) {
    return std::min(n_tasks, common_jsonl_worker_limit(requested));
}

std::vector<common_jsonl_line> common_jsonl_read_lines(
        const std::string & path,
        common_jsonl_empty_line_policy empty_line_policy) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open JSONL file: " + path);

    std::vector<common_jsonl_line> result;
    std::string line;
    int64_t number = 0;
    while (std::getline(input, line)) {
        ++number;
        if (line.empty() && empty_line_policy == COMMON_JSONL_EMPTY_LINE_SKIP) continue;
        result.push_back({ std::move(line), number });
    }
    return result;
}

common_chat_msg common_jsonl_parse_chat_message(
        const nlohmann::json & data,
        common_jsonl_chat_parse_mode mode) {
    return parse_chat_message(data, mode);
}

common_chat_msg common_jsonl_parse_chat_message(
        const nlohmann::ordered_json & data,
        common_jsonl_chat_parse_mode mode) {
    return parse_chat_message(data, mode);
}
