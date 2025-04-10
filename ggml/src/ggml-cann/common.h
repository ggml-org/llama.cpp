/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CANN_COMMON_H
#define CANN_COMMON_H

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>

#include "../include/ggml-cann.h"
#include "../include/ggml.h"

#define MATRIX_ROW_PADDING 512
#define GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, aclGetRecentErrMsg)

/**
 * @brief Contains information about CANN devices.
 */
struct ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_count;

    /**
     * @brief Information about a single CANN device.
     */
    struct cann_device_info {
        int cc;                 /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool vmm;               /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] =
        {}; /**< Array of CANN device information. */
};

const ggml_cann_device_info& ggml_cann_info();

void ggml_cann_set_device(int32_t device);
int32_t ggml_cann_get_device();

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
struct ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    virtual ~ggml_cann_pool() = default;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    virtual void* alloc(size_t size, size_t* actual_size) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    virtual void free(void* ptr, size_t size) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
struct ggml_cann_pool_alloc {
    ggml_cann_pool* pool = nullptr; /**< Pointer to the memory pool. */
    void* ptr = nullptr;    /**< Pointer to the allocated memory block. */
    size_t actual_size = 0; /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = default;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_cann_pool_alloc(ggml_cann_pool& pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_cann_pool_alloc(ggml_cann_pool& pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        return ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(ggml_cann_pool& pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    void* get() { return ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(const ggml_cann_pool_alloc&) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc&&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc& operator=(const ggml_cann_pool_alloc&) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc& operator=(ggml_cann_pool_alloc&&) = delete;
};

using aclnn_func_t = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using AnyAclResource = std::unique_ptr<void, std::function<void(void*)>>;

template<typename T>
struct AclResourceTraits;
template<>
struct AclResourceTraits<aclTensor> {
    static void destroy(void* p) {
        ACL_CHECK(aclDestroyTensor(static_cast<aclTensor*>(p)));
    }
};
template<>
struct AclResourceTraits<aclIntArray> {
    static void destroy(void* p) {
        ACL_CHECK(aclDestroyIntArray(static_cast<aclIntArray*>(p)));
    }
};
template<>
struct AclResourceTraits<aclScalar> {
    static void destroy(void* p) {
        ACL_CHECK(aclDestroyScalar(static_cast<aclScalar*>(p)));
    }
};
template<>
struct AclResourceTraits<aclTensorList> {
    static void destroy(void* p) {
        ACL_CHECK(aclDestroyTensorList(static_cast<aclTensorList*>(p)));
    }
};

template<typename T>
AnyAclResource make_acl_resource(T* ptr) {
    return AnyAclResource(
        static_cast<void*>(ptr),
        [](void* p) {
            AclResourceTraits<T>::destroy(p);
        }
    );
}

template<typename... Args>
void register_acl_resources(std::vector<AnyAclResource>& vec, Args*... args) {
    (vec.emplace_back(make_acl_resource(args)), ...);
}

class cann_task {
public:
    virtual void run_task() {}
};

class aclnn_task : public cann_task {
public:
    aclnn_task(aclnn_func_t aclnn_func, void * workspace_addr, uint64_t workspace_size, aclOpExecutor * executor,
               aclrtStream stream) :
        aclnn_func_(aclnn_func),
        workspace_addr_(workspace_addr),
        workspace_size_(workspace_size),
        executor_(executor),
        stream_(stream) {}
    virtual void run_task() override {
        ACL_CHECK(aclnn_func_(workspace_addr_, workspace_size_, executor_, stream_));
    }
private:
    aclnn_func_t aclnn_func_;
    void *          workspace_addr_;
    uint64_t        workspace_size_;
    aclOpExecutor * executor_;
    aclrtStream     stream_;
};

class resource_task : public cann_task {
public:
    resource_task(std::vector<AnyAclResource>&& resources){
        resource_ = std::move(resources);
    }

    virtual void run_task() override {
        resource_.clear();
    }
private:
    std::vector<AnyAclResource> resource_;
};

class free_ptr_task : public cann_task {
public:
    free_ptr_task(void* ptr) : ptr_(ptr) {}

    virtual void run_task() override {
        free(ptr_);
    }
private:
    void* ptr_;
};

class async_memcpy_task : public cann_task {
public:
    async_memcpy_task(void* dst, const void* src, size_t size, aclrtMemcpyKind kind, aclrtStream stream)
        : dst_(dst), src_(src), size_(size), kind_(kind), stream_(stream) {}

    virtual void run_task() override {
        
        ACL_CHECK(aclrtMemcpyAsync(dst_, size_, src_, size_, kind_, stream_));
    }
private:
    void* dst_;
    const void* src_;
    size_t size_;
    aclrtMemcpyKind kind_;
    aclrtStream stream_;
};

class async_memset_task : public cann_task {
    public:
    async_memset_task(void* buffer, size_t size, int32_t value, aclrtStream stream)
            : buffer_(buffer), size_(size), value_(value), stream_(stream) {}
    
        virtual void run_task() override {
            ACL_CHECK(aclrtMemsetAsync(buffer_, size_, value_, size_, stream_));
        }
    private:
        void* buffer_;
        size_t size_;
        int32_t value_;
        aclrtStream stream_;
    };

class cann_task_queue {
public:
    explicit cann_task_queue(size_t capacity, int32_t device)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0), running_(false), device_(device), consuming_(false) {
        GGML_ASSERT((capacity & (capacity - 1)) == 0 && "capacity must be power of 2");
        mask_ = capacity_ - 1;
    }

    bool enqueue(std::unique_ptr<cann_task>&& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) & mask_;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);

        cv_.notify_one();

        return true;
    }

    size_t dequeue_bulk(std::vector<std::unique_ptr<cann_task>>& output) {
        output.clear();
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_acquire);

        while (running_ && head == tail) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock);
            head = head_.load(std::memory_order_relaxed);
            tail = tail_.load(std::memory_order_acquire);
        }

        size_t count = 0;
        while (running_ && head != tail) {
            output.push_back(std::move(buffer_[head]));
            head = (head + 1) & mask_;
            ++count;
        }

        head_.store(head, std::memory_order_release);
        return count;
    }

    void submit_task(std::unique_ptr<cann_task>&& task) {
        while(!enqueue(std::move(task))) continue;
        
        if (!running_) {
            thread_ = std::thread(&cann_task_queue::execute, this);
            running_ = true;
        }
        
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    void wait() {
        if (!running_)
            return;

        while (!(empty() && consuming_)) {}
    }

    void stop() {
        running_ = false;
        wait();
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void execute() {
        std::vector<std::unique_ptr<cann_task>> tasks;
        ggml_cann_set_device(device_);

        while(running_) {
            consuming_ = true;
            int count = dequeue_bulk(tasks);
            consuming_ = false;
            if (count == 0)
                continue;
            
            for(auto &task : tasks) {
                task->run_task();
            }
        }
    }

    std::vector<std::unique_ptr<cann_task>> buffer_;
    const size_t capacity_;
    size_t mask_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool running_;
    std::thread thread_;
    int32_t device_;
    bool consuming_;
};

/**
 * @brief Context for managing CANN backend operations.
 */
struct ggml_backend_cann_context {
    int32_t device;                  /**< Device ID. */
    std::string name;                /**< Name of the device. */
    std::string description;         /**< Description of the device. */
    aclrtEvent copy_event = nullptr; /**< Event for managing copy operations. */
    cann_task_queue task_queue;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = {nullptr}; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(int device)
        : device(device), name("CANN" + std::to_string(device)), task_queue(1024, device) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        task_queue.stop();
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { return stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool>
        mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    static std::unique_ptr<ggml_cann_pool> new_pool_for_device(int device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool& pool() {
        if (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        return *mem_pool;
    }
};

#endif  // CANN_COMMON_H
