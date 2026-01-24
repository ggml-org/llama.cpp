#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-sycl/ggml-sycl-bench.hpp"

namespace sycl_bench {

bool run_mmvq_xmx_tile_8x8(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

bool run_mmvq_xmx_tile_16x16(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error);

bool run_mmvq_xmx_aos_direct(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error);

bool run_mmvq_xmx_soa_direct(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error);

bool run_mmvq_xmx_double_buffer(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error);

bool run_mmvq_esimd_dpas_1x16x32(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error);

bool run_mmvq_esimd_dpas_8x16x32(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error);

bool run_mmvq_esimd_dpas_chained(const ggml_sycl::mmvq_bench_args & args,
                                 std::vector<sycl::event> * events,
                                 std::string & error);

}  // namespace sycl_bench
