#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

//
// ring buffer
//

// Ring buffer with fixed capacity (similar to std::deque)
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (capacity == 0) {
            throw std::runtime_error("ring buffer: capacity is zero");
        }

        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result(sz);
        const size_t first_part = std::min(sz, capacity - first);

        // Copy first segment: from 'first' to end of buffer
        std::copy(data.begin() + first, data.begin() + first + first_part, result.begin());

        // Copy second segment: from start of buffer (no-op if no wrap-around)
        std::copy(data.begin(), data.begin() + (sz - first_part), result.begin() + first_part);

        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to llama_format ?
LLAMA_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llama_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LLAMA_TENSOR_NAME_FATTN   "__fattn__"
#define LLAMA_TENSOR_NAME_FGDN_AR "__fgdn_ar__"
#define LLAMA_TENSOR_NAME_FGDN_CH "__fgdn_ch__"
