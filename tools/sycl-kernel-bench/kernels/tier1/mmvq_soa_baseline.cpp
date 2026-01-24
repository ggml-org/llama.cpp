#include "mmvq_tier1.hpp"

#include "ggml-sycl/ggml-sycl-bench.hpp"

namespace sycl_bench {

bool run_mmvq_soa_baseline(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    if (!ggml_sycl::ggml_sycl_mmvq_bench_launch(args, events)) {
        error = "MMVQ SoA baseline launch failed.";
        return false;
    }
    return true;
}

}  // namespace sycl_bench

