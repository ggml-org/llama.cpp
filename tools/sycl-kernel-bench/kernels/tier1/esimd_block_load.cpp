#include "mmvq_tier1.hpp"
#include "esimd_common.hpp"

namespace sycl_bench {

bool run_mmvq_esimd_block_load(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    return mmvq_tier1_esimd::run_esimd_variant(args, events, error, false);
}

}  // namespace sycl_bench
