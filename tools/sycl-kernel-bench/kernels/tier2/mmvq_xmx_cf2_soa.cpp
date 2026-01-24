#include "mmvq_tier2.hpp"
#include "mmvq_tier2_common.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_cf2_soa(const ggml_sycl::mmvq_bench_args & args,
                          std::vector<sycl::event> * events,
                          std::string & error) {
    return mmvq_tier2::launch_xmx_cf2_soa(args, events, error);
}

}  // namespace sycl_bench
