#pragma once

//
// tessera-coreml-mil.h
//
// Minimal MIL (Model Intermediate Language) op-graph builder for the Tessera
// CoreML conversion. Builds an SSA op graph in memory and serializes it to the
// protobuf-JSON mapping that coremlcompiler ingests (the JSON form of
// mil_spec.proto's Program message). See docs/tessera-coreml-conversion-design.md
// sections 1.1 (MilBuilder), 3 (dequant chain) and 4.5 (pipeline).
//
// v1 (stock ops, decision C1): the Tessera dequant runs as a pre-processing
// step OUTSIDE the model; the graph carries pre-dequantized fp16 weights as
// const ops feeding standard matmul + add + reshape. The tessera_t640_dequant
// custom op is emitted only for v2 (gated on dequant > 5% of inference time).
//

#include <cstdint>
#include <string>
#include <vector>

enum ts_mil_dtype {
    TS_MIL_FP16  = 0,
    TS_MIL_FP32  = 1,
    TS_MIL_INT8  = 2,
    TS_MIL_INT32 = 3,
    TS_MIL_UINT8 = 4,
    TS_MIL_BOOL  = 5,
};

// One SSA operation. `inputs` maps MIL named-arg -> SSA value reference (each
// value must resolve to a builder input or a previous op's output). `attrs` are
// string-encoded attributes (int/bool/float/string) written verbatim into JSON.
struct ts_mil_op {
    std::string op_type;
    std::string output;                         // SSA value this op defines
    std::vector<std::pair<std::string, std::string>> inputs;
    std::vector<std::pair<std::string, std::string>> attrs;
    ts_mil_dtype out_dtype;
    std::vector<int64_t> out_shape;
};

struct ts_mil_value {
    std::string name;
    ts_mil_dtype dtype;
    std::vector<int64_t> shape;
};

struct ts_mil_builder {
    std::string function_name;
    int         opset;                          // CoreML spec version (default 9)
    int         counter;                        // SSA name suffix counter
    std::vector<ts_mil_value> inputs;
    std::vector<ts_mil_value> outputs;
    std::vector<ts_mil_op>    ops;
};

// Lifecycle.
void ts_mil_builder_init(ts_mil_builder * b, const char * function_name);

// Declare a graph input; returns the input name (== name passed in).
std::string ts_mil_add_input(ts_mil_builder * b, const char * name,
                             ts_mil_dtype dtype, const int64_t * shape, int64_t rank);

// Const op backed by an external weight blob. `blob_name` is the weight file
// the mlpackage writer will emit; the MIL const references it by name so the
// graph stays decoupled from the binary payload. Returns the SSA output name.
std::string ts_mil_const(ts_mil_builder * b, const char * hint,
                         ts_mil_dtype dtype, const int64_t * shape, int64_t rank,
                         const char * blob_name);

// Stock ops. Each returns the fresh SSA output name.
std::string ts_mil_matmul(ts_mil_builder * b, const char * x, const char * w, bool transpose_y);
std::string ts_mil_add(ts_mil_builder * b, const char * x, const char * y);
std::string ts_mil_relu(ts_mil_builder * b, const char * x);
std::string ts_mil_reshape(ts_mil_builder * b, const char * x, const int64_t * shape, int64_t rank);

// v2 custom op (decision C1, section 3.4). Emits a tessera_t640_dequant op with
// the 7 cluster-component inputs; output is the dequantized fp16 weight. Only
// used when the runtime selects the custom-op path.
std::string ts_mil_tessera_dequant(ts_mil_builder * b,
                                   const char * packed,
                                   const char * page_scales,
                                   const char * lane_scales,
                                   const char * outlier_offsets,
                                   const char * outlier_cols,
                                   const char * outlier_vals,
                                   const char * act_scale,
                                   const int64_t * out_shape, int64_t rank);

// Mark an SSA value as a block output.
void ts_mil_add_output(ts_mil_builder * b, const char * name);

// Validate the SSA chain: every op input and every block output must resolve to
// a declared input or a prior op output. Returns 0 on success, -1 on error
// (err_msg set if non-null).
int ts_mil_build(const ts_mil_builder * b, std::string * err_msg);

// Serialize the (validated) graph to the protobuf-JSON mapping of
// mil_spec.Program and write it to `path`. Returns 0 on success.
int ts_mil_emit_json(const ts_mil_builder * b, const char * path, std::string * err_msg);

// Same as ts_mil_emit_json but returns the JSON text instead of writing a file.
std::string ts_mil_to_json(const ts_mil_builder * b);

//
// Weight serialization (pipeline step 2).
//

// Source cluster for one logical weight tensor. The three core components
// (packed/page_scales/lane_scales) match the ggml Tile640 per-row layout; the
// outlier triple and act_scale are the optional v1.5 components. All pointers
// are byte buffers; nullptr means "absent".
struct ts_coreml_weight_src {
    const char * name;
    int64_t out_dim;                    // rows
    int64_t in_dim;                     // cols (matmul K)
    const uint8_t * packed;             // [out * pages * 32] uint32 words
    const uint8_t * page_scales;        // [out * pages] fp16
    const uint8_t * lane_scales;        // [out * pages * 32] int8
    const uint8_t * outlier_row_offsets;// [out + 1] int32 (optional)
    const uint8_t * outlier_cols;       // [n_outliers] int32 (optional)
    const uint8_t * outlier_vals;       // [n_outliers] fp16 (optional)
    int64_t n_outliers;
    const uint8_t * act_scale;          // [in_dim] fp16 (optional)
};

struct ts_coreml_weight_out {
    std::string blob_name;              // weight file name matching the MIL const
    int64_t     n_bytes;                // bytes written
    bool        custom_op;              // true = raw cluster blobs (v2), false = fp16 (v1)
};

// Serialize one weight tensor next to `dir`. When `custom_op` is false (v1 stock
// ops) the cluster is dequantized to fp16 [out, in] and written as a single
// blob. When true (v2) the raw cluster components are written as separate blobs
// and `out->blob_name` is the packed-weights blob. Returns 0 on success.
int ts_coreml_serialize_weights(const ts_coreml_weight_src * src,
                                const char * dir,
                                bool custom_op,
                                ts_coreml_weight_out * out,
                                std::string * err_msg);

// Dequantize one Tile640 row cluster to fp32 (matches ggml
// dequantize_row_tessera_t640, plus outlier replacement and act_scale). Exposed
// for testing. `y` must hold out_dim * in_dim floats.
void ts_coreml_dequant_t640(const ts_coreml_weight_src * src, float * y);

// Weight-blob file name for a logical tensor. Shared naming convention between
// the MIL const ops and the serialized weight files so the graph and the blobs
// stay in lockstep. v1 (custom_op=false) -> "<stem>.fp16.bin"; v2 -> the packed
// component blob "<stem>.packed.bin".
std::string ts_coreml_weight_blob_name(const char * tensor_name, bool custom_op);
