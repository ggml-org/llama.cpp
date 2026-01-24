#include "dpas_kernels.hpp"

namespace sycl_bench {

bool run_dpas_memory_patterns(const DpasBenchArgs & args,
                              std::vector<sycl::event> * events,
                              std::string & error) {
    return dispatch_dpas(args, events, error);
}

}  // namespace sycl_bench
