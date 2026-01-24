#include "mmvq_tier4.hpp"
#include "mmvq_tier4_spec_common.hpp"

namespace sycl_bench {

bool run_mmvq_coalesced_xmx_aligned(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    return mmvq_tier4::launch_coalesced_xmx_aligned(args, events, error);
}

}  // namespace sycl_bench
