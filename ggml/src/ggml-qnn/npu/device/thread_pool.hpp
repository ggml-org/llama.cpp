#pragma once

#include <qurt.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "util.hpp"

namespace hexagon {

constexpr const size_t             kMaxThreadCount       = 4;
constexpr const size_t             kDefaultStackSize     = 1024 * 32;  // 32KB
constexpr const unsigned long long kThreadTaskPendingBit = 1;

template <size_t _stack_size> class qurt_thread {
  public:
    typedef void (*qurt_thread_func_type)(qurt_thread * thread, void * arg);

    explicit qurt_thread(const std::string & thread_name, qurt_thread_func_type thread_func, void * arg,
                         unsigned short priority) {
        DEVICE_LOG_DEBUG("qurt_thread.create: %s", thread_name.c_str());
        qurt_thread_attr_init(&_attributes);
        qurt_thread_attr_set_name(&_attributes, (char *) thread_name.c_str());
        qurt_thread_attr_set_stack_addr(&_attributes, _stack);
        qurt_thread_attr_set_stack_size(&_attributes, _stack_size);
        qurt_thread_attr_set_priority(&_attributes, priority);

        _func    = thread_func;
        _arg     = arg;
        auto ret = qurt_thread_create(
            &_tid, &_attributes, reinterpret_cast<void (*)(void *)>(&qurt_thread::thread_func_impl), (void *) this);
        if (ret != QURT_EOK) {
            DEVICE_LOG_ERROR("Failed to create thread: %d", (int) ret);
            _func = nullptr;
            _arg  = nullptr;
            return;
        }

        DEVICE_LOG_DEBUG("qurt_thread.created: %s, id: %d", thread_name.c_str(), (int) _tid);
    }

    ~qurt_thread() {
        DEVICE_LOG_DEBUG("qurt_thread.destroy: %d", (int) _tid);
        int  thread_exit_code = QURT_EOK;
        auto ret              = qurt_thread_join(_tid, &thread_exit_code);
        if (ret != QURT_EOK && ret != QURT_ENOTHREAD) {
            DEVICE_LOG_ERROR("Failed to join thread: %d", (int) ret);
            return;
        }

        if (thread_exit_code != QURT_EOK) {
            DEVICE_LOG_ERROR("Thread exit code: %d", (int) thread_exit_code);
        }
    }

    bool is_valid() const { return _tid != 0 && _func != nullptr; }

  private:
    static void thread_func_impl(qurt_thread * thread) {
        if (thread->_func) {
            thread->_func(thread, thread->_arg);
        }

        qurt_thread_exit(QURT_EOK);
    }

    uint8_t               _stack[_stack_size] = {};
    qurt_thread_t         _tid;
    qurt_thread_attr_t    _attributes;
    qurt_thread_func_type _func = nullptr;
    void *                _arg  = nullptr;

    DISABLE_COPY_AND_MOVE(qurt_thread);
};

using qurt_thread_ptr = std::unique_ptr<qurt_thread<kDefaultStackSize>>;

template <size_t _thread_count> class thread_pool {
    static_assert(_thread_count > 1, "Thread count must be greater than 1");
    constexpr const static size_t kMaxSubThreadCount = _thread_count - 1;

  public:
    typedef qurt_thread<kDefaultStackSize> thread_type;
    typedef void (*task_type)(thread_pool * pool, size_t thread_idx, size_t thread_count, void * arg);

    thread_pool() {
        std::string thread_name_base = "thread_pool_";
        qurt_barrier_init(&_pending, kMaxSubThreadCount + 1);
        qurt_barrier_init(&_completed, kMaxSubThreadCount + 1);
        const auto priority = qurt_thread_get_priority(qurt_thread_get_id());
        for (size_t i = 0; i < kMaxSubThreadCount; ++i) {
            auto & thread_arg     = _thread_args[i];
            thread_arg.pool       = this;
            thread_arg.thread_idx = i + 1;

            auto thread = std::make_unique<thread_type>(
                thread_name_base + std::to_string(i),
                reinterpret_cast<thread_type::qurt_thread_func_type>(&thread_pool::thread_func_impl), &thread_arg,
                priority);
            if (!thread->is_valid()) {
                DEVICE_LOG_ERROR("Failed to create thread: %zu", i);
                // destroy all barriers and threads at destructor
                return;
            }

            _threads[i] = std::move(thread);
        }
        DEVICE_LOG_DEBUG("thread_pool.created: %zu", kMaxSubThreadCount);
    }

    ~thread_pool() {
        DEVICE_LOG_DEBUG("thread_pool.destroy");
        _thread_exit = true;
        qurt_barrier_wait(&_pending);  // release all task threads

        for (auto & thread : _threads) {
            thread.reset();
        }

        qurt_barrier_destroy(&_completed);
        qurt_barrier_destroy(&_pending);
    }

    bool sync_execute(task_type task, void * arg) {
        if (!task) {
            DEVICE_LOG_ERROR("Invalid task");
            return false;
        }

        _task = task;
        _arg  = arg;
        qurt_barrier_wait(&_pending);

        task(this, 0, kMaxSubThreadCount + 1, arg);
        DEVICE_LOG_DEBUG("main_thread.task_completed: 0");

        qurt_barrier_wait(&_completed);

        _task = nullptr;
        _arg  = nullptr;
        return true;
    }

    void sync_thread() { qurt_barrier_wait(&_completed); }

  private:
    struct thread_pool_arg {
        thread_pool * pool       = nullptr;
        size_t        thread_idx = 0;
    };

    static void thread_func_impl(thread_type * thread, thread_pool_arg * arg) {
        NPU_UNUSED(thread);

        DEVICE_LOG_DEBUG("thread_func_impl.start: %zu", arg->thread_idx);

        auto & pool = *arg->pool;
        for (;;) {
            qurt_barrier_wait(&pool._pending);
            if (pool._thread_exit) {
                DEVICE_LOG_DEBUG("thread_func_impl.exit: %zu", arg->thread_idx);
                break;
            }

            auto task = pool._task;
            if (task) {
                task(arg->pool, arg->thread_idx, kMaxSubThreadCount + 1, pool._arg);
            }

            DEVICE_LOG_DEBUG("thread_func_impl.task_completed: %zu", arg->thread_idx);
            qurt_barrier_wait(&pool._completed);
        }

        DEVICE_LOG_DEBUG("thread_func_impl.end: %zu", arg->thread_idx);
    }

    std::atomic_bool                                _thread_exit = false;
    std::array<qurt_thread_ptr, kMaxSubThreadCount> _threads;
    thread_pool_arg                                 _thread_args[kMaxSubThreadCount] = {};
    qurt_barrier_t                                  _pending                         = {};
    qurt_barrier_t                                  _completed                       = {};
    task_type                                       _task                            = nullptr;
    void *                                          _arg                             = nullptr;

    DISABLE_COPY_AND_MOVE(thread_pool);
};

using default_thread_pool = thread_pool<kMaxThreadCount>;

}  // namespace hexagon
