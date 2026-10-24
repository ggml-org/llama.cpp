#pragma once

// NPU (NPUW) compile configuration. Everything that decides which OpenVINO properties an NPU graph is
// compiled with lives here, so "what does the NPU compile with" is answered in one place:
//   * the backend-wide base config (NPUW defaults, named env aliases, GGML_OPENVINO_NPU_CONFIG overrides)
//   * the per-graph variants of it (cacheless graphs, static prefill, static decode)

#include <openvino/core/any.hpp>
#include <string>

class GgmlOvDecoder;

// Base config for every NPU graph, built once at backend init. GGML_OPENVINO_NPU_CONFIG ("K=V,...") is
// applied last, so it overrides the defaults and the named aliases.
ov::AnyMap ggml_openvino_npu_build_compile_config(const char * cache_dir);

// Cacheless (encoder) graphs compile on the plain NPU plugin: drop every NPUW key.
ov::AnyMap ggml_openvino_npu_without_npuw(const ov::AnyMap & config);

// Static prefill graph. With GGML_OPENVINO_NPU_HFA_PREFILL, switches NPUW to the HFA attention pipeline.
ov::AnyMap ggml_openvino_npu_prefill_config(const ov::AnyMap & base, const GgmlOvDecoder & decoder);

// Static decode graph. With GGML_OPENVINO_NPU_DECODE_UNFOLD, unfolds NPUW function calls into separate infer
// requests, unless NPUW_UNFOLD_IREQS is already set.
ov::AnyMap ggml_openvino_npu_decode_config(const ov::AnyMap & base, const std::string & device);
