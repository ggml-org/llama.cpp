#include "mmvq_tier3.hpp"
#include "mmvq_tier3_spec_common.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_tile_64x64(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    return mmvq_tier3::launch_xmx_tile_64x64(args, events, error);
}

bool run_mmvq_xmx_register_accum(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error) {
    return mmvq_tier3::launch_xmx_register_accum(args, events, error);
}

bool run_mmvq_xmx_multi_wg(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    return mmvq_tier3::launch_xmx_multi_wg(args, events, error);
}

bool run_mmvq_xmx_persistent(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    return mmvq_tier3::launch_xmx_persistent(args, events, error);
}

bool run_mmvq_esimd_large_tile(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    return mmvq_tier3::launch_esimd_large_tile(args, events, error);
}

bool run_mmvq_esimd_persistent(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    return mmvq_tier3::launch_esimd_persistent(args, events, error);
}

bool run_mmvq_esimd_lsc_prefetch(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error) {
    return mmvq_tier3::launch_esimd_lsc_prefetch(args, events, error);
}

}  // namespace sycl_bench
