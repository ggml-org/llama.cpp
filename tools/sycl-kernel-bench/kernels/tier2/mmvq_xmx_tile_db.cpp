#include "mmvq_tier2.hpp"
#include "mmvq_tier2_common.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_tile_db(const ggml_sycl::mmvq_bench_args & args,
                          std::vector<sycl::event> * events,
                          std::string & error) {
    return mmvq_tier2::launch_xmx_tile_db(args, events, error);
}

}  // namespace sycl_bench
