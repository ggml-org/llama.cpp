#ifndef GGML_SYCL_XMX_COMPAT_HPP
#define GGML_SYCL_XMX_COMPAT_HPP

// Every XMX instruction of the SYCL backend goes through this layer: the ESIMD dpas as
// ggml_sycl_xmx::esimd::dpas, the joint_matrix API as ggml_sycl_xmx::matrix. Code takes
//
//     namespace xmx = ggml_sycl_xmx::esimd;    // instead of sycl::ext::intel::esimd::xmx
//     namespace mx  = ggml_sycl_xmx::matrix;   // instead of sycl::ext::oneapi::experimental::matrix
//
// and is written as against the SYCL namespaces. A joint_matrix kernel of sub-group size SG also
// takes its sub-group and barrier from here (sub_group_of<SG>, group_barrier) and is launched with
// reqd_sub_group_size(hw_sub_group_size(SG)).
//
// By default all of it is the SYCL API itself. With GGML_SYCL_XMX_EMULATION (CMake option, off by
// default) it computes in software, so a kernel runs on a device without the XMX shape it asks for:
// the 8-wide dpas of Xe-HPG on Xe2, the 16-wide one on Xe-HPG, joint_matrix shapes a device does not
// list, on logical sub-groups of a size it lacks. That is for debugging; it is slow.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <type_traits>

namespace ggml_sycl_xmx {

#if defined(GGML_SYCL_XMX_EMULATION)
inline constexpr bool emulated = true;
#else
inline constexpr bool emulated = false;
#endif

#if !defined(GGML_SYCL_XMX_EMULATION)

namespace esimd  = sycl::ext::intel::esimd::xmx;
namespace matrix = sycl::ext::oneapi::experimental::matrix;

// The sub-group of a joint_matrix kernel of sub-group size SG: launched with
// reqd_sub_group_size(hw_sub_group_size(SG)), it takes sub_group_of<SG>(item) and group_barrier().
constexpr int hw_sub_group_size(int sg) {
    return sg;
}

template <int SG, int D> inline sycl::sub_group sub_group_of(const sycl::nd_item<D> & item) {
    return item.get_sub_group();
}

using sycl::group_barrier;

#else

// Emulated, the sub-group is logical: SG consecutive work-items of the work-group, so a kernel of a
// sub-group size the device lacks (8 on Xe2) runs. It is launched on hardware sub-groups of at least
// 16 work-items, each holding whole logical ones, and the emulated joint_matrix only splits stores
// over a sub-group. Two logical sub-groups in one hardware sub-group may call group_barrier() a
// different number of times (the K splits of the grouped GEMM differ by a step): on Intel GPUs a
// sub-group is one hardware thread and its barrier a fence, which that does not hang, although SYCL
// does not promise it.
constexpr int hw_sub_group_size(int sg) {
    return sg < 16 ? 16 : sg;
}

template <int SG> struct logical_sub_group {
    sycl::sub_group hw;
    int             id;
    int             lane;

    sycl::id<1>    get_group_id() const { return sycl::id<1>(id); }
    sycl::id<1>    get_local_id() const { return sycl::id<1>(lane); }
    size_t         get_local_linear_id() const { return lane; }
    sycl::range<1> get_local_range() const { return sycl::range<1>(SG); }
    size_t         get_local_linear_range() const { return SG; }
};

template <int SG, int D> inline logical_sub_group<SG> sub_group_of(const sycl::nd_item<D> & item) {
    const int l = (int) item.get_local_linear_id();
    return { item.get_sub_group(), l / SG, l % SG };
}

template <int SG> inline void group_barrier(const logical_sub_group<SG> & g) {
    sycl::group_barrier(g.hw);
}

inline void group_barrier(const sycl::sub_group & g) {
    sycl::group_barrier(g);
}

namespace esimd {
namespace detail {
template <typename E> constexpr bool supported() {
    return std::is_same_v<E, sycl::half> || std::is_same_v<E, sycl::ext::oneapi::bfloat16> ||
           std::is_same_v<E, int8_t> || std::is_same_v<E, uint8_t>;
}

// D = C + A B as xmx::dpas computes it: A row major [RC][K], B the VNNI operand [K / OPC][NW][OPC],
// C and D [RC][NW], K = SD * OPC with OPC the operand elements of a 32-bit channel. Integer operands
// sum in int32, floating ones in f32; the f32 sum order is not the hardware's.
template <int SD, int RC, typename T, typename BT, typename AT, int BN, int AN>
ESIMD_INLINE void mad(sycl::ext::intel::esimd::simd<T, RC * (BN / (SD * (4 / sizeof(BT))))> & D,
                      sycl::ext::intel::esimd::simd<BT, BN> B, sycl::ext::intel::esimd::simd<AT, AN> A) {
    using namespace sycl::ext::intel::esimd;
    static_assert(supported<BT>() && supported<AT>() && sizeof(AT) == sizeof(BT), "dpas operand types");
    constexpr int OPC = 4 / sizeof(BT);
    constexpr int K   = SD * OPC;
    constexpr int NW  = BN / K;
    static_assert(BN == K * NW && AN == RC * K, "dpas operand sizes");
    using W = std::conditional_t<std::is_integral_v<BT>, int32_t, float>;
    simd<W, BN> bw = B;
    simd<W, AN> aw = A;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        simd<W, NW> bk = bw.template select<NW, OPC>((k / OPC) * NW * OPC + k % OPC);
#pragma unroll
        for (int r = 0; r < RC; ++r) {
            D.template select<NW, 1>(r * NW) += simd<T, NW>(bk * (W) aw[r * K + k]);
        }
    }
}
} // namespace detail

template <int SD, int RC, typename T, typename CT, typename BT, typename AT, int N, int BN, int AN>
ESIMD_INLINE sycl::ext::intel::esimd::simd<T, N> dpas(sycl::ext::intel::esimd::simd<CT, N> C,
                                                      sycl::ext::intel::esimd::simd<BT, BN> B,
                                                      sycl::ext::intel::esimd::simd<AT, AN> A) {
    sycl::ext::intel::esimd::simd<T, N> D = C;
    detail::mad<SD, RC, T, BT, AT, BN, AN>(D, B, A);
    return D;
}

template <int SD, int RC, typename T, typename BT, typename AT, int BN, int AN>
ESIMD_INLINE auto dpas(sycl::ext::intel::esimd::simd<BT, BN> B, sycl::ext::intel::esimd::simd<AT, AN> A) {
    sycl::ext::intel::esimd::simd<T, RC * (BN / (SD * (4 / sizeof(BT))))> D = 0;
    detail::mad<SD, RC, T, BT, AT, BN, AN>(D, B, A);
    return D;
}
} // namespace esimd

// joint_matrix in software: every work-item of the group holds the whole matrix, loads all of it
// and computes all of every product; a store is split over the group's work-items. The enums,
// tf32 and the combination record are the SYCL ones.
namespace matrix {
namespace sm = sycl::ext::oneapi::experimental::matrix;
using sm::combination;
using sm::layout;
using sm::matrix_type;
using sm::use;
namespace precision = sm::precision;

namespace detail {
template <typename T> struct storage {
    using type = T;
};
template <> struct storage<precision::tf32> {
    using type = float;
};
} // namespace detail

template <typename Group, typename T, use Use, size_t Rows, size_t Cols, layout Layout = layout::dynamic>
struct joint_matrix {
    using storage_t = typename detail::storage<T>::type;
    storage_t data[Rows * Cols];
};

// G, the group a call passes, is the logical sub-group of a kernel (sub_group_of) or a sycl::sub_group
template <typename G, typename Group, typename T, use U, size_t R, size_t C, layout L, typename V>
inline void joint_matrix_fill(G, joint_matrix<Group, T, U, R, C, L> & m, const V & v) {
    using S = typename joint_matrix<Group, T, U, R, C, L>::storage_t;
#pragma unroll
    for (size_t i = 0; i < R * C; ++i) {
        m.data[i] = (S) v;
    }
}

namespace detail {
// element (r, c) of a matrix at p with the given layout and stride; ext_intel_packed is the VNNI
// layout of B: rows of K / V, each holding the V consecutive k of every column
template <size_t R, size_t C, typename S, typename P>
inline void load(S * d, P p, size_t stride, layout l) {
    constexpr size_t V = 4 / sizeof(S) > 0 ? 4 / sizeof(S) : 1;
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
            size_t o;
            if (l == layout::row_major) {
                o = r * stride + c;
            } else if (l == layout::col_major) {
                o = c * stride + r;
            } else {
                o = (r / V) * stride + c * V + (r % V);
            }
            d[r * C + c] = (S) p[o];
        }
    }
}
} // namespace detail

// a or b, layout in the type
template <typename G, typename Group, typename T, use U, size_t R, size_t C, layout L, typename P>
inline std::enable_if_t<L != layout::dynamic> joint_matrix_load(G, joint_matrix<Group, T, U, R, C, L> & m, P p,
                                                                size_t stride) {
    detail::load<R, C>(m.data, p, stride, L);
}

// accumulator, layout given
template <typename G, typename Group, typename T, size_t R, size_t C, typename P>
inline void joint_matrix_load(G, joint_matrix<Group, T, use::accumulator, R, C, layout::dynamic> & m, P p,
                              size_t stride, layout l) {
    detail::load<R, C>(m.data, p, stride, l);
}

template <typename G, typename Group, typename T, size_t R, size_t C, typename P>
inline void joint_matrix_store(G g, const joint_matrix<Group, T, use::accumulator, R, C, layout::dynamic> & m,
                               P p, size_t stride, layout l) {
    // every work-item holds the same values; each writes its share
    const size_t n = g.get_local_linear_range();
    for (size_t i = g.get_local_linear_id(); i < R * C; i += n) {
        const size_t r = i / C, c = i % C;
        p[l == layout::col_major ? c * stride + r : r * stride + c] = m.data[i];
    }
}

template <typename G, typename Group, typename Td, typename Ta, typename Tb, typename Tc, size_t M, size_t K,
          size_t N, layout La, layout Lb>
inline void joint_matrix_mad(G, joint_matrix<Group, Td, use::accumulator, M, N, layout::dynamic> & D,
                             const joint_matrix<Group, Ta, use::a, M, K, La> &                   A,
                             const joint_matrix<Group, Tb, use::b, K, N, Lb> &                   B,
                             const joint_matrix<Group, Tc, use::accumulator, M, N, layout::dynamic> & C) {
    using S = typename joint_matrix<Group, Td, use::accumulator, M, N>::storage_t;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            S s = (S) C.data[m * N + n];
            for (size_t k = 0; k < K; ++k) {
                s += (S) A.data[m * K + k] * (S) B.data[k * N + n];
            }
            D.data[m * N + n] = s;
        }
    }
}
} // namespace matrix

#endif // GGML_SYCL_XMX_EMULATION

} // namespace ggml_sycl_xmx

#endif // GGML_SYCL_XMX_COMPAT_HPP
