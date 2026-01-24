#include "mmvq_tier2.hpp"
#include "mmvq_tier2_spec_common.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_tile_8x8(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    return mmvq_tier2::launch_xmx_tile_8x8(args, events, error);
}

bool run_mmvq_xmx_tile_16x16(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    return mmvq_tier2::launch_xmx_tile_16x16(args, events, error);
}

bool run_mmvq_xmx_aos_direct(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    return mmvq_tier2::launch_xmx_aos_direct(args, events, error);
}

bool run_mmvq_xmx_soa_direct(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    return mmvq_tier2::launch_xmx_soa_direct(args, events, error);
}

bool run_mmvq_xmx_double_buffer(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    return mmvq_tier2::launch_xmx_double_buffer(args, events, error);
}

bool run_mmvq_esimd_dpas_1x16x32(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error) {
    return mmvq_tier2::launch_esimd_dpas_1x16x32(args, events, error);
}

bool run_mmvq_esimd_dpas_8x16x32(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error) {
    return mmvq_tier2::launch_esimd_dpas_8x16x32(args, events, error);
}

bool run_mmvq_esimd_dpas_chained(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error) {
    return mmvq_tier2::launch_esimd_dpas_chained(args, events, error);
}

}  // namespace sycl_bench
