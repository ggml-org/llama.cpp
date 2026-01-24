#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-sycl/ggml-sycl-bench.hpp"

namespace sycl_bench {

bool run_mmvq_aos_baseline(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

bool run_mmvq_soa_baseline(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

bool run_mmvq_coalesced(const ggml_sycl::mmvq_bench_args & args,
                        std::vector<sycl::event> * events,
                        std::string & error);

bool run_mmvq_slm_cached(const ggml_sycl::mmvq_bench_args & args,
                         std::vector<sycl::event> * events,
                         std::string & error);

bool run_mmvq_prefetch(const ggml_sycl::mmvq_bench_args & args,
                       std::vector<sycl::event> * events,
                       std::string & error);

bool run_mmvq_wide_load(const ggml_sycl::mmvq_bench_args & args,
                        std::vector<sycl::event> * events,
                        std::string & error);

bool run_mmvq_esimd_block_load(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error);

bool run_mmvq_esimd_slm(const ggml_sycl::mmvq_bench_args & args,
                        std::vector<sycl::event> * events,
                        std::string & error);

}  // namespace sycl_bench
