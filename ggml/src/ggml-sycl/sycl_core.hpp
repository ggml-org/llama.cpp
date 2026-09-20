// sycl_core.hpp - native SYCL primitives replacing the dpct (SYCLomatic)
// emulation layer.
//
// Rationale:
// - dpct::dev_mgr emulates cudaSetDevice with a thread->device map and wraps
//   every device in a mutex-protected device_ext holding in-order queues.
// - In-order queues serialize every submission and stall/deadlock with
//   Level Zero V2 (Immediate Command List) runtimes, and the dpct mutex
//   dance adds host-side latency on every queue access.
// - This header provides the modern replacement: plain sycl::device
//   enumeration, out-of-order sycl::queue creation, and explicit
//   sycl::event dependency chaining for out-of-order queues.

#ifndef GGML_SYCL_SYCL_CORE_HPP
#define GGML_SYCL_SYCL_CORE_HPP

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "ggml.h"
#include "base.hpp"

namespace ggml_sycl {

using queue_ptr = sycl::queue *;
using event_ptr = sycl::event *;

// ---------------------------------------------------------------------------
// Error codes - replaces dpct::err0/success/default_error.
using err0 = int;
constexpr err0 success       = 0;
constexpr err0 default_error = -1;

// ---------------------------------------------------------------------------
// Device registry - replaces dpct::dev_mgr.
//
// No thread->device map: the SYCL model binds every operation to an explicit
// sycl::queue, there is no "current device" at the runtime level. No
// in-order queues, no mutex-protected device_ext wrapper. Enumeration via
// sycl::device::get_devices() honors ONEAPI_DEVICE_SELECTOR /
// ZE_FLAT_DEVICE_HIERARCHY, matching dpct's enumeration semantics.
class device_registry {
  public:
    static device_registry & instance() {
        static device_registry reg;
        return reg;
    }

    int device_count() const { return (int) m_devices.size(); }

    sycl::device & get_device(int id) {
        GGML_ASSERT(id >= 0 && id < (int) m_devices.size());
        return m_devices[id];
    }

    // Lazily created out-of-order queue for the given device. All commands
    // submitted on this queue must go through ordered_submit() (or depend
    // on the queue's event chain) to preserve ordering.
    sycl::queue & queue(int id) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = m_queues.find(id);
        if (it == m_queues.end()) {
            // Default-constructed properties == out-of-order queue.
            it = m_queues.emplace(id, sycl::queue(get_device(id))).first;
        }
        return it->second;
    }

  private:
    device_registry() {
        auto gpu_devs = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (!gpu_devs.empty()) {
            m_devices = std::move(gpu_devs);
        } else {
            // No GPU: expose whatever is enumerated (CPU devices) so
            // non-SYCL tools keep working when SYCL is compiled.
            m_devices = sycl::device::get_devices();
        }
    }

    std::vector<sycl::device>            m_devices;
    std::unordered_map<int, sycl::queue> m_queues;
    std::mutex                           m_queue_mutex;
};

// Per-thread current device - replaces dpct's thread->device map with a
// plain thread_local int (no mutex, no map). Kept only for the
// current-device fast paths; all operation code addresses devices and
// queues explicitly via ggml_backend_sycl_context.
inline int & current_device_id() {
    thread_local int id = 0;
    return id;
}

inline void set_current_device(int id) { current_device_id() = id; }

inline int get_current_device_id() { return current_device_id(); }

inline sycl::queue & default_queue() {
    return device_registry::instance().queue(current_device_id());
}

// ---------------------------------------------------------------------------
// Event-based dependency control for out-of-order queues.
//
// Every command submitted on a queue depends on the previous command's
// event, reproducing in-order semantics without the in_order queue
// property. Dependencies are expressed with native sycl::event objects -
// no barriers at op boundaries and no host synchronization; the runtime
// releases each event automatically once its dependents complete.
inline sycl::event & stream_last_event(queue_ptr q) {
    static std::mutex m;
    static std::unordered_map<queue_ptr, sycl::event> ev;
    std::lock_guard<std::mutex> lock(m);
    return ev[q];
}

// Submits cgf on q with a native-event dependency on everything previously
// submitted on q. Returns the event of this submission.
template <typename CGF>
sycl::event ordered_submit(queue_ptr q, CGF && cgf) {
    sycl::event prev = stream_last_event(q);
    sycl::event e = q->submit([&](sycl::handler & h) {
        h.depends_on(prev);
        cgf(h);
    });
    stream_last_event(q) = e;
    return e;
}

// Kernel-launch form: ordered_parallel_for(q, nd_range<3>(...), lambda).
template <typename... Args>
sycl::event ordered_parallel_for(queue_ptr q, Args && ... args) {
    return ordered_submit(q, [&](sycl::handler & h) {
        h.parallel_for(std::forward<Args>(args)...);
    });
}

template <typename... Args>
sycl::event ordered_single_task(queue_ptr q, Args && ... args) {
    return ordered_submit(q, [&](sycl::handler & h) {
        h.single_task(std::forward<Args>(args)...);
    });
}

template <typename... Args>
sycl::event ordered_memcpy(queue_ptr q, Args && ... args) {
    return ordered_submit(q, [&](sycl::handler & h) {
        h.memcpy(std::forward<Args>(args)...);
    });
}

template <typename... Args>
sycl::event ordered_memset(queue_ptr q, Args && ... args) {
    return ordered_submit(q, [&](sycl::handler & h) {
        h.memset(std::forward<Args>(args)...);
    });
}

// Empty command with dependencies - native (portable) barrier replacement
// for queue::ext_oneapi_submit_barrier({events...}).
template <typename... Deps>
sycl::event ordered_barrier(queue_ptr q, Deps && ... deps) {
    return ordered_submit(q, [&](sycl::handler & h) {
        (h.depends_on(std::forward<Deps>(deps)), ...);
    });
}

// Waits for all commands submitted on the queue so far. Native replacement
// for device_ext::queues_wait_and_throw() (no dpct mutex dance).
inline void queue_wait(queue_ptr q) {
    q->wait_and_throw();
}

// ---------------------------------------------------------------------------
// Device properties - replaces dpct::device_info (sycl::get_info based).
inline void parse_device_version(const std::string & ver, int & major, int & minor) {
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    // c. <AmdGcnArchName> e.g gfx1030
    std::string::size_type i = 0;
    while (i < ver.size()) {
        if (std::isdigit((unsigned char) ver[i])) {
            break;
        }
        i++;
    }
    major = std::stoi(&(ver[i]));
    while (i < ver.size()) {
        if (ver[i] == '.') {
            break;
        }
        i++;
    }
    if (i < ver.size()) {
        // a. and b.
        i++;
        minor = std::stoi(&(ver[i]));
    } else {
        // c.
        minor = 0;
    }
}

class device_info {
  public:
    device_info() = default;
    explicit device_info(const sycl::device & dev) { init(dev); }

    const char * get_name() const { return m_name.c_str(); }
    int    get_major_version() const { return m_major; }
    int    get_minor_version() const { return m_minor; }
    int    get_max_compute_units() const { return m_max_compute_units; }
    int    get_max_work_group_size() const { return m_max_work_group_size; }
    int    get_max_sub_group_size() const { return m_max_sub_group_size; }
    size_t get_global_mem_size() const { return m_global_mem_size; }
    size_t get_max_mem_alloc_size() const { return m_max_mem_alloc_size; }
    size_t get_local_mem_size() const { return m_local_mem_size; }
    sycl::range<3> get_max_work_item_sizes() const { return m_max_work_item_sizes; }
    bool   get_host_unified_memory() const { return m_host_unified_memory; }

    void init(const sycl::device & dev) {
        m_name = dev.get_info<sycl::info::device::name>();
        parse_device_version(dev.get_info<sycl::info::device::version>(), m_major, m_minor);
        m_max_work_item_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
        m_host_unified_memory = dev.has(sycl::aspect::usm_host_allocations);
        m_max_compute_units   = dev.get_info<sycl::info::device::max_compute_units>();
        m_max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
        // device::max_sub_group_size was removed from the SYCL 2020 traits;
        // sub_group_sizes is the supported list, last entry is the maximum.
        const auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
        m_max_sub_group_size  = sgs.empty() ? 0 : (int) sgs.back();
        m_global_mem_size     = dev.get_info<sycl::info::device::global_mem_size>();
        m_max_mem_alloc_size  = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        m_local_mem_size      = dev.get_info<sycl::info::device::local_mem_size>();
    }

  private:
    std::string    m_name;
    int            m_major              = 0;
    int            m_minor              = 0;
    int            m_max_compute_units  = 0;
    int            m_max_work_group_size = 0;
    int            m_max_sub_group_size = 0;
    size_t         m_global_mem_size    = 0;
    size_t         m_max_mem_alloc_size = 0;
    size_t         m_local_mem_size     = 0;
    bool           m_host_unified_memory = false;
    sycl::range<3> m_max_work_item_sizes{1, 1, 1};
};

inline void get_device_info(device_info & out, const sycl::device & dev) {
    out = device_info(dev);
}

// ---------------------------------------------------------------------------
// Device type naming - replaces dpct::get_device_type_name /
// dpct::get_device_backend_and_type.
inline std::string get_device_type_name(const sycl::device & device) {
    auto device_type = device.get_info<sycl::info::device::device_type>();
    switch (device_type) {
    case sycl::info::device_type::cpu:
        return "cpu";
    case sycl::info::device_type::gpu:
        return "gpu";
    case sycl::info::device_type::host:
        return "host";
    case sycl::info::device_type::accelerator:
        return "acc";
    default:
        return "unknown";
    }
}

inline std::string get_device_backend_and_type(const sycl::device & device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type << backend << ":" << get_device_type_name(device);
    return device_type.str();
}

// ---------------------------------------------------------------------------
// Capability checks - replaces dpct::has_capability_or_fail (no
// <sycl/info/aspects.def> inclusion).
inline void has_capability_or_fail(const sycl::device & dev,
                                   const std::initializer_list<sycl::aspect> & props) {
    for (const auto & it : props) {
        if (!dev.has(it)) {
            throw std::runtime_error("required SYCL aspect not supported in '" +
                                     dev.get_info<sycl::info::device::name>() + "' device");
        }
    }
}

// ---------------------------------------------------------------------------
// 3-component launch dimensions - replaces dpct::dim3 (CUDA leftover).
class dim3 {
  public:
    unsigned x, y, z;

    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
        : x(x_), y(y_), z(z_) {}

    dim3(const sycl::id<3> & r) : dim3(r[2], r[1], r[0]) {}

    operator sycl::range<3>() const { return sycl::range<3>(z, y, x); }
};

inline dim3 operator*(const dim3 & a, const dim3 & b) {
    return dim3{a.x * b.x, a.y * b.y, a.z * b.z};
}

// ---------------------------------------------------------------------------
// Library data types - replaces dpct::library_data_t / matrix_info_t.
enum class library_data_t : unsigned char {
    complex_float,
    complex_double,
    real_float,
    real_double,
    real_half,
    real_bfloat16,
    real_int8,
    real_int32,
    real_int64,
    real_uint8,
    real_uint16,
    real_uint32,
    real_uint64,
    library_data_t_size
};

static_assert((unsigned char) library_data_t::library_data_t_size <= 255,
              "library_data_t size exceeds limit.");

template <typename Ts> struct matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     value_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

namespace detail {

template <typename tag, typename T>
class generic_error_type {
  public:
    generic_error_type() = default;
    generic_error_type(T value) : value{value} {}
    operator T() const { return value; }

  private:
    T value;
};

template <typename Ts> struct get_mkl_type { using type = Ts; };

template <typename T>
inline T * get_memory(const void * ptr) { return static_cast<T *>(const_cast<void *>(ptr)); }

} // namespace detail

template <typename T>
inline auto get_value(const T * s, sycl::queue & q) {
    detail::generic_error_type<T, T> value = 0;
    ordered_memcpy(&q, &value, s, sizeof(T)).wait();
    return (T) value;
}

// ---------------------------------------------------------------------------
// half x half -> float GEMM on oneMKL - replaces dpct::gemm. Only the
// combinations used by the backend are provided.
inline void gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                 int m, int n, int k, const void * alpha, const void * a, int lda,
                 const void * b, int ldb, const void * beta, void * c, int ldc) {
    float alpha_value = get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value  = get_value(reinterpret_cast<const float *>(beta), q);
    sycl::event e = oneapi::mkl::blas::column_major::gemm(
        q, a_trans, b_trans, m, n, k, alpha_value,
        detail::get_memory<const sycl::half>(a), lda,
        detail::get_memory<const sycl::half>(b), ldb, beta_value,
        detail::get_memory<float>(c), ldc);
}

// Batched half GEMM (stride variant) - replaces dpct::gemm_batch.
inline void gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                       int m, int n, int k, const void * alpha, const void * a, int lda,
                       long long int stride_a, const void * b, int ldb, long long int stride_b,
                       const void * beta, void * c, int ldc, long long int stride_c, int batch_size) {
    float alpha_value = get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value  = get_value(reinterpret_cast<const float *>(beta), q);
    oneapi::mkl::blas::column_major::gemm_batch(
        q, a_trans, b_trans, m, n, k, alpha_value,
        detail::get_memory<const sycl::half>(a), lda, stride_a,
        detail::get_memory<const sycl::half>(b), ldb, stride_b, beta_value,
        detail::get_memory<float>(c), ldc, stride_c, batch_size);
}

// Batched half GEMM (pointer-array variant) - replaces dpct::gemm_batch.
inline void gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                       int m, int n, int k, const void * alpha, const void * a[], int lda,
                       const void * b[], int ldb, const void * beta, void * c[],
                       int ldc, int batch_size, matrix_info_t<float> * matrix_info) {
    float alpha_value = get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value  = get_value(reinterpret_cast<const float *>(beta), q);

    matrix_info->transpose_info[0] = a_trans;
    matrix_info->transpose_info[1] = b_trans;
    matrix_info->value_info[0]     = alpha_value;
    matrix_info->value_info[1]     = beta_value;
    matrix_info->size_info[0]      = m;
    matrix_info->size_info[1]      = n;
    matrix_info->size_info[2]      = k;
    matrix_info->ld_info[0]        = lda;
    matrix_info->ld_info[1]        = ldb;
    matrix_info->ld_info[2]        = ldc;
    matrix_info->groupsize_info    = batch_size;

    sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
        q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
        matrix_info->size_info, matrix_info->size_info + 1, matrix_info->size_info + 2,
        reinterpret_cast<float *>(matrix_info->value_info),
        reinterpret_cast<const sycl::half **>(a), matrix_info->ld_info,
        reinterpret_cast<const sycl::half **>(b), matrix_info->ld_info + 1,
        reinterpret_cast<float *>(matrix_info->value_info + 1),
        reinterpret_cast<float **>(c), matrix_info->ld_info + 2, 1,
        &(matrix_info->groupsize_info));
}

// ---------------------------------------------------------------------------
// Memory copy - replaces dpct::memcpy_direction / async_dpct_memcpy.
enum class memcpy_direction {
    host_to_device,
    device_to_host,
    device_to_device,
    automatic,
    automatic_device,
    custom
};

// 1D async memcpy between USM pointers. USM pointers are all addressable
// from the queue's context, so a single ordered h.memcpy covers every
// direction - the kind argument is kept for call-site compatibility and
// ignored.
inline err0 memcpy_async(queue_ptr q, void * to_ptr, const void * from_ptr, size_t size,
                         memcpy_direction kind = memcpy_direction::automatic) {
    GGML_UNUSED(kind);
    try {
        ordered_memcpy(q, to_ptr, from_ptr, size);
        return success;
    } catch (sycl::exception const & exc) {
        std::cerr << exc.what() << "\nException caught at file:" << __FILE__
                  << ", line:" << __LINE__ << ", func:" << __func__ << std::endl;
        return default_error;
    }
}

// 2D pitched async memcpy (row-by-row loop, native equivalent of
// dpct::async_dpct_memcpy with pitch).
inline err0 memcpy_async(queue_ptr q, void * to_ptr, size_t to_pitch, const void * from_ptr,
                         size_t from_pitch, size_t x, size_t y,
                         memcpy_direction kind = memcpy_direction::automatic) {
    GGML_UNUSED(kind);
    try {
        char *       to_row   = (char *) to_ptr;
        const char * from_row = (const char *) from_ptr;
        for (size_t i = 0; i < y; ++i) {
            ordered_memcpy(q, to_row + i * to_pitch, from_row + i * from_pitch, x);
        }
        return success;
    } catch (sycl::exception const & exc) {
        std::cerr << exc.what() << "\nException caught at file:" << __FILE__
                  << ", line:" << __LINE__ << ", func:" << __func__ << std::endl;
        return default_error;
    }
}

// ---------------------------------------------------------------------------
// Subgroup shuffle - replaces dpct::permute_sub_group_by_xor.
// Exact port: select_from_group within a logical sub-subgroup of the given
// size (default 32 = the full subgroup for the existing call sites).
template <typename T>
inline T sub_group_shuffle_xor(sycl::sub_group g, T x, unsigned int mask,
                               unsigned int logical_sub_group_size = 32) {
    unsigned int id = g.get_local_linear_id();
    unsigned int start_index =
        id / logical_sub_group_size * logical_sub_group_size;
    unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
    return sycl::select_from_group(g, x,
                                   target_offset < logical_sub_group_size
                                       ? start_index + target_offset
                                       : id);
}

// ---------------------------------------------------------------------------
// Device-code primitives - replaces dpct::dp4a / sub_sat / vectorized_binary
// / byte_level_permute / min / max / atomic_fetch_add.
namespace detail {

template <typename T1, typename T2>
using dot_product_acc_t = std::conditional_t<
    std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
    uint32_t,
    int32_t>;

template <typename T>
sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
    return sycl::vec<T, 1>(val)
        .template as<sycl::vec<
            std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
            4>>()
        .template convert<T>();
}

template <typename VecT, class BinaryOperation, class = void>
class vectorized_binary {
  public:
    inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
        VecT v4;
        for (size_t i = 0; i < v4.size(); ++i) {
            v4[i] = binary_op(a[i], b[i]);
        }
        return v4;
    }
};

template <typename VecT, class BinaryOperation>
class vectorized_binary<
    VecT, BinaryOperation,
    std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>> {
  public:
    inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
        return binary_op(a, b).template as<VecT>();
    }
};

} // namespace detail

template <typename T1, typename T2, typename T3>
inline auto dp4a(T1 a, T2 b, T3 c) {
    detail::dot_product_acc_t<T1, T2> res = c;
    auto va = detail::extract_and_sign_or_zero_extend4(a);
    auto vb = detail::extract_and_sign_or_zero_extend4(b);
    res += va[0] * vb[0];
    res += va[1] * vb[1];
    res += va[2] * vb[2];
    res += va[3] * vb[3];
    return res;
}

struct sub_sat {
    template <typename T>
    auto operator()(const T x, const T y) const {
        return sycl::sub_sat(x, y);
    }
};

template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b, const BinaryOperation binary_op) {
    sycl::vec<unsigned, 1> v0{a}, v1{b};
    auto v2 = v0.as<VecT>();
    auto v3 = v1.as<VecT>();
    auto v4 = detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
    v0 = v4.template as<sycl::vec<unsigned, 1>>();
    return v0;
}

inline unsigned int byte_level_permute(unsigned int a, unsigned int b, unsigned int s) {
    unsigned int ret;
    ret = ((((std::uint64_t) b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
          (((((std::uint64_t) b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
          (((((std::uint64_t) b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
          (((((std::uint64_t) b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
    return ret;
}

inline std::uint64_t max(const std::int32_t a, const std::uint64_t b) {
    return sycl::max(static_cast<std::uint64_t>(a), b);
}

inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b) {
    return sycl::max(a, static_cast<std::uint64_t>(b));
}

inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b) {
    return sycl::max(static_cast<std::uint64_t>(a), b);
}

template <sycl::access::address_space addressSpace =
            sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    return atm.fetch_add(operand);
}

} // namespace ggml_sycl

#endif // GGML_SYCL_SYCL_CORE_HPP
