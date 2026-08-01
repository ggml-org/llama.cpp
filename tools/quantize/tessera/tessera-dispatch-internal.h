#pragma once

//
// tessera-dispatch-internal.h
//
// Internal types shared between tessera-dispatch.cpp and its integration
// test (test_l5_dispatch.cpp). Not part of the public Tessera API surface;
// consumers outside the dispatch implementation and its tests should not
// include this file.
//

#include <string>
#include <vector>
#include <cstdint>

#include "tessera-quant.h"   // ts_quant_result_2d, ts_quant_params_2d

// One captured 2D quantizable tensor from the step-7 walk. Stored once per
// tensor and indexed by name so the L5 refine loop can target tensors without
// re-walking the GGUF. act_scales_copy / inputGGUF index let the loop
// re-quantize without retaining every source weight in memory (sources are
// re-read from the input GGUF on demand, per the L5 memory budget).
struct ts_dispatch_refine_entry {
    std::string             name;
    std::string             family;
    int64_t                 gguf_idx = -1;     // index in in_ctx
    int64_t                 out_dim  = 0;
    int64_t                 in_dim   = 0;
    ts_quant_result_2d *    qr       = nullptr; // points into cluster_results
    ts_quant_params_2d      tqp{};             // baseline params applied at step 7
    std::vector<float>      act_scales_copy;   // owned copy (act_scales may alias imatrix memory)
};
