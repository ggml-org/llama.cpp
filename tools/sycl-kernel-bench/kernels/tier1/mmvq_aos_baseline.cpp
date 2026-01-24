#include "mmvq_tier1.hpp"
#include "mmvq_tier1_common.hpp"

namespace sycl_bench {

bool run_mmvq_aos_baseline(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    return mmvq_tier1::launch_aos_variant<mmvq_tier1::AosVariant::Baseline>(args, events, error);
}

}  // namespace sycl_bench

