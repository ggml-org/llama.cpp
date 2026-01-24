#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-sycl/ggml-sycl-bench.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_tile_64x64(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error);

bool run_mmvq_xmx_register_accum(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error);

bool run_mmvq_xmx_multi_wg(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

bool run_mmvq_xmx_persistent(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error);

bool run_mmvq_esimd_large_tile(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error);

bool run_mmvq_esimd_persistent(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error);

bool run_mmvq_esimd_lsc_prefetch(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error);

}  // namespace sycl_bench
