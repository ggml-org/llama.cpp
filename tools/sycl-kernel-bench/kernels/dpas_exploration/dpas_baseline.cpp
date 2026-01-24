#include "dpas_kernels.hpp"

namespace sycl_bench {

bool run_dpas_baseline(const DpasBenchArgs & args,
                       std::vector<sycl::event> * events,
                       std::string & error) {
    DpasBenchArgs tuned = args;
    tuned.memory_pattern = DpasMemoryPattern::DIRECT_GLOBAL;
    tuned.grf_mode = DpasGrfMode::GRF_128;
    return dispatch_dpas(tuned, events, error);
}

}  // namespace sycl_bench
