#include "mmvq_tier2.hpp"

#include "ggml-sycl/mmq-esimd.hpp"

namespace sycl_bench {
namespace {

bool validate_esimd_args(const ggml_sycl::mmvq_bench_args & args, std::string & error) {
    if (args.stream == nullptr) {
        error = "SYCL stream is null.";
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "ESIMD MMVQ kernels require AOS weight layout.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_Q4_0) {
        error = "ESIMD MMVQ kernels support Q4_0 only.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid dimensions for ESIMD MMVQ kernel.";
        return false;
    }
    if ((args.ncols % QK4_0) != 0) {
        error = "K dimension must be multiple of 32 for ESIMD MMVQ kernel.";
        return false;
    }
    if (args.src1_padded_col_size != args.ncols) {
        error = "ESIMD MMVQ kernel requires unpadded activations (src1_padded_col_size == ncols).";
        return false;
    }
    if (args.row_low != 0 || args.row_high != args.nrows) {
        error = "ESIMD MMVQ kernel requires full row span.";
        return false;
    }
    return true;
}

}  // namespace

bool run_mmvq_esimd_v2(const ggml_sycl::mmvq_bench_args & args,
                       std::vector<sycl::event> * events,
                       std::string & error) {
#if SYCL_ESIMD_MMQ_AVAILABLE
    if (!validate_esimd_args(args, error)) {
        return false;
    }

    sycl::event ev;
    const bool launched = launch_mmq_q4_0_esimd_v2(
        reinterpret_cast<const block_q4_0 *>(args.weights),
        reinterpret_cast<const block_q8_1 *>(args.activations),
        args.output,
        static_cast<int64_t>(args.nrows),
        static_cast<int64_t>(args.batch),
        static_cast<int64_t>(args.ncols),
        static_cast<int64_t>(args.dst_row_stride),
        *args.stream,
        events ? &ev : nullptr);

    if (!launched) {
        error = "ESIMD v2 kernel rejected launch (K too small or unsupported).";
        return false;
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
#else
    error = "SYCL ESIMD unavailable; ESIMD v2 kernel disabled.";
    return false;
#endif
}

bool run_mmvq_esimd_v3(const ggml_sycl::mmvq_bench_args & args,
                       std::vector<sycl::event> * events,
                       std::string & error) {
#if SYCL_ESIMD_MMQ_AVAILABLE
    if (!validate_esimd_args(args, error)) {
        return false;
    }

    sycl::event ev;
    const bool launched = launch_mmq_q4_0_esimd_v3(
        reinterpret_cast<const block_q4_0 *>(args.weights),
        reinterpret_cast<const block_q8_1 *>(args.activations),
        args.output,
        static_cast<int64_t>(args.nrows),
        static_cast<int64_t>(args.batch),
        static_cast<int64_t>(args.ncols),
        static_cast<int64_t>(args.dst_row_stride),
        *args.stream,
        events ? &ev : nullptr);

    if (!launched) {
        error = "ESIMD v3 kernel rejected launch (K too small or unsupported).";
        return false;
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
#else
    error = "SYCL ESIMD unavailable; ESIMD v3 kernel disabled.";
    return false;
#endif
}

}  // namespace sycl_bench
