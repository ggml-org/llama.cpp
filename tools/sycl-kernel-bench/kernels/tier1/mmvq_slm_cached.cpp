#include "mmvq_tier1.hpp"
#include "mmvq_tier1_common.hpp"

namespace sycl_bench {

bool run_mmvq_slm_cached(const ggml_sycl::mmvq_bench_args & args,
                         std::vector<sycl::event> * events,
                         std::string & error) {
    return mmvq_tier1::launch_slm_cached(args, events, error);
}

}  // namespace sycl_bench

