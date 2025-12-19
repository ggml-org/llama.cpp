// SYCL Backend VTune Profiling Support
// Enable with cmake -DGGML_SYCL_PROFILE=ON and run with:
//   vtune -collect gpu-offload -knob analyze-user-tasks=true ./build/bin/llama-bench ...

#pragma once

// To enable profiling, define GGML_SYCL_PROFILE before including this header
// or add -DGGML_SYCL_PROFILE to compile flags

#ifdef GGML_SYCL_PROFILE

#include <ittnotify.h>

// Domains for different operation categories
namespace ggml_sycl_profile {

// Initialize profiling - call once at startup
inline void init() {
    // Domains are created lazily, nothing to do here
}

// Get or create domain (lazy initialization)
inline __itt_domain* get_domain_fa() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.FlashAttention");
    return domain;
}

inline __itt_domain* get_domain_mmvq() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.MMVQ");
    return domain;
}

inline __itt_domain* get_domain_gemm() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.GEMM");
    return domain;
}

inline __itt_domain* get_domain_elementwise() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.ElementWise");
    return domain;
}

inline __itt_domain* get_domain_memory() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.Memory");
    return domain;
}

inline __itt_domain* get_domain_graph() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.Graph");
    return domain;
}

inline __itt_domain* get_domain_compute() {
    static __itt_domain* domain = __itt_domain_create("GGML.SYCL.Compute");
    return domain;
}

// String handle cache for common task names
inline __itt_string_handle* get_handle(const char* name) {
    // Note: __itt_string_handle_create caches internally, so calling
    // it multiple times with the same string is efficient
    return __itt_string_handle_create(name);
}

// RAII task wrapper for automatic task end
class ScopedTask {
public:
    ScopedTask(__itt_domain* domain, const char* name)
        : m_domain(domain) {
        __itt_task_begin(domain, __itt_null, __itt_null, get_handle(name));
    }

    ~ScopedTask() {
        __itt_task_end(m_domain);
    }

    // Non-copyable
    ScopedTask(const ScopedTask&) = delete;
    ScopedTask& operator=(const ScopedTask&) = delete;

private:
    __itt_domain* m_domain;
};

} // namespace ggml_sycl_profile

// Convenience macros for common profiling patterns
#define GGML_SYCL_PROFILE_INIT() ggml_sycl_profile::init()

#define GGML_SYCL_PROFILE_START_FA(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_fa(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_FA() \
    __itt_task_end(ggml_sycl_profile::get_domain_fa())
#define GGML_SYCL_PROFILE_SCOPE_FA(name) \
    ggml_sycl_profile::ScopedTask _itt_task_fa(ggml_sycl_profile::get_domain_fa(), name)

#define GGML_SYCL_PROFILE_START_MMVQ(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_mmvq(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_MMVQ() \
    __itt_task_end(ggml_sycl_profile::get_domain_mmvq())
#define GGML_SYCL_PROFILE_SCOPE_MMVQ(name) \
    ggml_sycl_profile::ScopedTask _itt_task_mmvq(ggml_sycl_profile::get_domain_mmvq(), name)

#define GGML_SYCL_PROFILE_START_GEMM(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_gemm(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_GEMM() \
    __itt_task_end(ggml_sycl_profile::get_domain_gemm())
#define GGML_SYCL_PROFILE_SCOPE_GEMM(name) \
    ggml_sycl_profile::ScopedTask _itt_task_gemm(ggml_sycl_profile::get_domain_gemm(), name)

#define GGML_SYCL_PROFILE_START_ELEMENTWISE(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_elementwise(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_ELEMENTWISE() \
    __itt_task_end(ggml_sycl_profile::get_domain_elementwise())
#define GGML_SYCL_PROFILE_SCOPE_ELEMENTWISE(name) \
    ggml_sycl_profile::ScopedTask _itt_task_ew(ggml_sycl_profile::get_domain_elementwise(), name)

#define GGML_SYCL_PROFILE_START_MEMORY(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_memory(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_MEMORY() \
    __itt_task_end(ggml_sycl_profile::get_domain_memory())
#define GGML_SYCL_PROFILE_SCOPE_MEMORY(name) \
    ggml_sycl_profile::ScopedTask _itt_task_mem(ggml_sycl_profile::get_domain_memory(), name)

#define GGML_SYCL_PROFILE_START_GRAPH(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_graph(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_GRAPH() \
    __itt_task_end(ggml_sycl_profile::get_domain_graph())
#define GGML_SYCL_PROFILE_SCOPE_GRAPH(name) \
    ggml_sycl_profile::ScopedTask _itt_task_graph(ggml_sycl_profile::get_domain_graph(), name)

#define GGML_SYCL_PROFILE_START_COMPUTE(name) \
    __itt_task_begin(ggml_sycl_profile::get_domain_compute(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END_COMPUTE() \
    __itt_task_end(ggml_sycl_profile::get_domain_compute())
#define GGML_SYCL_PROFILE_SCOPE_COMPUTE(name) \
    ggml_sycl_profile::ScopedTask _itt_task_compute(ggml_sycl_profile::get_domain_compute(), name)

// Generic task macros
#define GGML_SYCL_PROFILE_START(domain_fn, name) \
    __itt_task_begin(ggml_sycl_profile::domain_fn(), __itt_null, __itt_null, \
                     ggml_sycl_profile::get_handle(name))
#define GGML_SYCL_PROFILE_END(domain_fn) \
    __itt_task_end(ggml_sycl_profile::domain_fn())

#else // GGML_SYCL_PROFILE not defined

// No-op stubs when profiling is disabled
#define GGML_SYCL_PROFILE_INIT()

#define GGML_SYCL_PROFILE_START_FA(name)
#define GGML_SYCL_PROFILE_END_FA()
#define GGML_SYCL_PROFILE_SCOPE_FA(name)

#define GGML_SYCL_PROFILE_START_MMVQ(name)
#define GGML_SYCL_PROFILE_END_MMVQ()
#define GGML_SYCL_PROFILE_SCOPE_MMVQ(name)

#define GGML_SYCL_PROFILE_START_GEMM(name)
#define GGML_SYCL_PROFILE_END_GEMM()
#define GGML_SYCL_PROFILE_SCOPE_GEMM(name)

#define GGML_SYCL_PROFILE_START_ELEMENTWISE(name)
#define GGML_SYCL_PROFILE_END_ELEMENTWISE()
#define GGML_SYCL_PROFILE_SCOPE_ELEMENTWISE(name)

#define GGML_SYCL_PROFILE_START_MEMORY(name)
#define GGML_SYCL_PROFILE_END_MEMORY()
#define GGML_SYCL_PROFILE_SCOPE_MEMORY(name)

#define GGML_SYCL_PROFILE_START_GRAPH(name)
#define GGML_SYCL_PROFILE_END_GRAPH()
#define GGML_SYCL_PROFILE_SCOPE_GRAPH(name)

#define GGML_SYCL_PROFILE_START_COMPUTE(name)
#define GGML_SYCL_PROFILE_END_COMPUTE()
#define GGML_SYCL_PROFILE_SCOPE_COMPUTE(name)

#define GGML_SYCL_PROFILE_START(domain_fn, name)
#define GGML_SYCL_PROFILE_END(domain_fn)

#endif // GGML_SYCL_PROFILE
