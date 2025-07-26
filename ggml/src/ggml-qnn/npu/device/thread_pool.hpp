#pragma once

#include <qurt.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "util.hpp"
#include "vtcm_mem.hpp"

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

template <size_t _ThreadCount> class thread_pool {
    static_assert(_ThreadCount > 1, "Thread count must be greater than 1");
    constexpr const static size_t kMaxThreadCount    = _ThreadCount;
    constexpr const static size_t kMaxSubThreadCount = _ThreadCount - 1;

  public:
    typedef qurt_thread<kDefaultStackSize> thread_type;

    struct thread_params {
        size_t                         tidx;
        const size_t                   tcnt = kMaxThreadCount;
        thread_pool<kMaxThreadCount> * pool = nullptr;
        size_t                         vtcm_quota_size;

        std::unique_ptr<vtcm_mem>  vtcm_cache;
        std::unique_ptr<uint8_t[]> mem_cache;
        size_t                     mem_cache_size = 0;

        uint8_t * get_vtcm_cache(size_t size) {
            if (!vtcm_cache || vtcm_cache->get_size() < size) {
                DEVICE_SCOPED_PERFORMANCE_TRACKER("[thread_params]get_vtcm_cache, size: %zu, tidx: %zu", size, tidx);
                vtcm_cache.reset();  // reset the cache to create a new one
                vtcm_cache = std::make_unique<vtcm_mem>(size, false);
            }

            if (!vtcm_cache->is_valid()) {
                return nullptr;
            }

            return vtcm_cache->get_mem();
        }

        uint8_t * get_mem_cache(size_t size) {
            if (!mem_cache || mem_cache_size < size) {
                mem_cache.reset();  // reset the cache to create a new one
                mem_cache      = std::make_unique<uint8_t[]>(size + 256);
                mem_cache_size = mem_cache ? size : 0;
            }

            return mem_cache.get();
        }
    };

    typedef void (*task_type)(thread_pool * pool, thread_params * param, void * arg);

    thread_pool() {
        for (size_t i = 0; i < kMaxThreadCount; ++i) {
            _thread_params[i].tidx            = i;
            _thread_params[i].vtcm_quota_size = hexagon::vtcm_mem::get_avail_block_size() / kMaxThreadCount;
            _thread_params[i].pool            = this;
        }

        qurt_barrier_init(&_pending, kMaxSubThreadCount + 1);
        qurt_barrier_init(&_completed, kMaxSubThreadCount + 1);
        const auto  priority         = qurt_thread_get_priority(qurt_thread_get_id());
        std::string thread_name_base = "thread_pool_";
        for (size_t i = 0; i < kMaxSubThreadCount; ++i) {
            auto thread = std::make_unique<thread_type>(
                thread_name_base + std::to_string(i), &thread_pool::thread_func_impl, &_thread_params[i + 1], priority);
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

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
        _task_begin_cycles = HAP_perf_get_qtimer_count();
#endif

        _task = task;
        _arg  = arg;
        qurt_barrier_wait(&_pending);

        task(this, &_thread_params[0], arg);
        DEVICE_LOG_DEBUG("main_thread.task_completed: 0");

        qurt_barrier_wait(&_completed);

        _task = nullptr;
        _arg  = nullptr;

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
        _task_begin_cycles = 0;
#endif

        return true;
    }

    void sync_thread() { qurt_barrier_wait(&_completed); }

    static size_t get_per_thread_vtcm_quota() { return vtcm_mem::get_total_size() / kMaxThreadCount; }

  private:
    static void thread_func_impl(thread_type * thread, void * arg) {
        NPU_UNUSED(thread);

        auto * param = reinterpret_cast<thread_params *>(arg);

        DEVICE_LOG_DEBUG("thread_func_impl.start: %zu", param->tidx);

        auto & pool = *(param->pool);
        for (;;) {
            qurt_barrier_wait(&pool._pending);
            if (pool._thread_exit) {
                DEVICE_LOG_DEBUG("thread_func_impl.exit: %zu", param->tidx);
                break;
            }

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
            auto task_begin_cycles = pool._task_begin_cycles.load();
            DEVICE_LOG_WARN("[profiler]worker_thread, tidx: %zu, prepare: %lluus", param->tidx,
                            static_cast<unsigned long long>(
                                HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - task_begin_cycles)));
#endif

            auto task = pool._task;
            if (task) {
                task(param->pool, param, pool._arg);
            }

            DEVICE_LOG_DEBUG("thread_func_impl.task_completed: %zu", param->tidx);
            qurt_barrier_wait(&pool._completed);

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
            DEVICE_LOG_WARN("[profiler]worker_thread, tidx: %zu, task_end: %lluus", param->tidx,
                            static_cast<unsigned long long>(
                                HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - task_begin_cycles)));
#endif
        }

        DEVICE_LOG_DEBUG("thread_func_impl.end: %zu", param->tidx);
    }

    std::atomic_bool                                _thread_exit                    = false;
    std::array<qurt_thread_ptr, kMaxSubThreadCount> _threads                        = {};
    qurt_barrier_t                                  _pending                        = {};
    qurt_barrier_t                                  _completed                      = {};
    thread_params                                   _thread_params[kMaxThreadCount] = {};
    task_type                                       _task                           = nullptr;
    void *                                          _arg                            = nullptr;

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
    std::atomic<uint64_t> _task_begin_cycles = 0;
#endif

    DISABLE_COPY_AND_MOVE(thread_pool);
};

using default_thread_pool = thread_pool<kMaxThreadCount>;

}  // namespace hexagon
