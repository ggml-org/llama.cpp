#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

class [[nodiscard]] Status {
  public:
    Status() = default;

    Status(const Status &)             = delete;
    Status & operator=(const Status &) = delete;

    Status(Status &&) noexcept             = default;
    Status & operator=(Status &&) noexcept = default;

    bool success() const { return messages_.empty(); }

    const std::vector<std::string> & errors() const { return messages_; }

    void append(const Status & other) {
        messages_.insert(messages_.end(), other.messages_.begin(), other.messages_.end());
    }

    void log(const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_va(format, args);
        va_end(args);
    }

  private:
    void push(std::string message) { messages_.push_back(std::move(message)); }

    void log_va(const char * format, va_list args) {
        if (format == nullptr) {
            push("failed to format error message");
            return;
        }

        char    stack[256];
        va_list args_copy;
        va_copy(args_copy, args);
        const int written = std::vsnprintf(stack, sizeof(stack), format, args_copy);
        va_end(args_copy);
        if (written < 0) {
            push("failed to format error message");
            return;
        }
        if (static_cast<size_t>(written) < sizeof(stack)) {
            push(std::string(stack, static_cast<size_t>(written)));
            return;
        }

        std::vector<char> buffer(static_cast<size_t>(written) + 1);
        const int         rewritten = std::vsnprintf(buffer.data(), buffer.size(), format, args);
        if (rewritten < 0) {
            push("failed to format error message");
            return;
        }
        push(std::string(buffer.data(), static_cast<size_t>(rewritten)));
    }

    std::vector<std::string> messages_;
};

}  // namespace ggml::hrx
