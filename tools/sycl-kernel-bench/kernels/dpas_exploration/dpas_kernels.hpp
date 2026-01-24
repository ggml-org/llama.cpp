#pragma once

#include <string>
#include <vector>

#include "dpas_common.hpp"

namespace sycl_bench {

bool run_dpas_baseline(const DpasBenchArgs & args,
                       std::vector<sycl::event> * events,
                       std::string & error);

bool run_dpas_sweep(const DpasBenchArgs & args,
                    std::vector<sycl::event> * events,
                    std::string & error);

bool run_dpas_memory_patterns(const DpasBenchArgs & args,
                              std::vector<sycl::event> * events,
                              std::string & error);

}  // namespace sycl_bench
