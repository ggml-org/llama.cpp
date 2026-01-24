#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-sycl/ggml-sycl-bench.hpp"

namespace sycl_bench {

bool run_mmvq_hybrid_adaptive(const ggml_sycl::mmvq_bench_args & args,
                              std::vector<sycl::event> * events,
                              std::string & error);

bool run_mmvq_xmx_fused(const ggml_sycl::mmvq_bench_args & args,
                        std::vector<sycl::event> * events,
                        std::string & error);

bool run_mmvq_coalesced_xmx_aligned(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error);

bool run_mmvq_esimd_hybrid(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

bool run_mmvq_esimd_cooperative(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error);

bool run_mmvq_q4_0_specialized(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error);

bool run_mmvq_q6_k_specialized(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error);

bool run_mmvq_mxfp4_native(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error);

}  // namespace sycl_bench
