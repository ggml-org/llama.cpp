#include "tessera-coreml-mil.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <set>

// Tile640 layout constants (mirror ggml/src/ggml-common.h; duplicated here so
// the conversion module stays standalone for the test harness).
#define TS_T640_PAGE_SIZE      640
#define TS_T640_LANE_SIZE      20
#define TS_T640_LANES_PER_PAGE 32
#define TS_T640_WORDS_PER_PAGE 32

//
// fp16 <-> fp32 (self-contained IEEE 754 half; ggml's macros live in
// ggml/src/ggml-impl.h which is not on the standalone test include path).
//

static float ts_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3ff;
            f = sign | ((uint32_t) exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((uint32_t) (exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static uint16_t ts_f32_to_f16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    uint32_t fexp = (f >> 23) & 0xff;
    uint32_t man  = f & 0x7fffff;
    if (fexp == 0) {
        return (uint16_t) sign;                 // zero / float subnormal -> half zero
    }
    if (fexp == 0xff) {
        return (uint16_t) (sign | 0x7c00 | (man ? 0x200 : 0));
    }
    int32_t exp = (int32_t) fexp - 127 + 15;
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00);      // overflow -> inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;             // underflow -> zero
        }
        man |= 0x800000;
        uint32_t shift = (uint32_t) (14 - exp);
        uint32_t hman  = man >> shift;
        uint32_t rem   = man & ((1u << shift) - 1);
        uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (hman & 1))) {
            hman++;
        }
        return (uint16_t) (sign | hman);
    }
    uint32_t hman = man >> 13;
    uint32_t rem  = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (hman & 1))) {
        hman++;
        if (hman > 0x3ff) { hman = 0; exp++; }
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | hman);
}

//
// small helpers
//

static std::string ts_mil_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static const char * ts_mil_dtype_name(ts_mil_dtype d) {
    switch (d) {
        case TS_MIL_FP16:  return "FP16";
        case TS_MIL_FP32:  return "FP32";
        case TS_MIL_INT8:  return "INT8";
        case TS_MIL_INT32: return "INT32";
        case TS_MIL_UINT8: return "UINT8";
        case TS_MIL_BOOL:  return "BOOL";
    }
    return "FP16";
}

static void ts_mil_set_err(std::string * err_msg, const std::string & msg) {
    if (err_msg) {
        *err_msg = msg;
    }
}

// Sanitize a tensor name into a weight-blob file stem (CoreML weight paths are
// flat; dots and slashes are not portable).
static std::string ts_mil_blob_stem(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        out += (std::isalnum((unsigned char) c) || c == '_') ? c : '_';
    }
    return out;
}

std::string ts_coreml_weight_blob_name(const char * tensor_name, bool custom_op) {
    const std::string stem = ts_mil_blob_stem(tensor_name ? tensor_name : "weight");
    return stem + (custom_op ? ".packed.bin" : ".fp16.bin");
}

//
// builder lifecycle
//

void ts_mil_builder_init(ts_mil_builder * b, const char * function_name) {
    b->function_name = function_name ? function_name : "main";
    b->opset         = 9;   // matches Prism specification_version (design 1.5)
    b->counter       = 0;
    b->inputs.clear();
    b->outputs.clear();
    b->ops.clear();
}

static std::string ts_mil_fresh(ts_mil_builder * b, const char * hint) {
    return std::string(hint ? hint : "v") + "_" + std::to_string(b->counter++);
}

// Look up the shape of an SSA value (graph input or prior op output). Returns
// nullptr if unknown.
static const std::vector<int64_t> * ts_mil_find_shape(const ts_mil_builder * b, const std::string & name) {
    for (const auto & in : b->inputs) {
        if (in.name == name) {
            return &in.shape;
        }
    }
    for (const auto & op : b->ops) {
        if (op.output == name) {
            return &op.out_shape;
        }
    }
    return nullptr;
}

static ts_mil_dtype ts_mil_find_dtype(const ts_mil_builder * b, const std::string & name, ts_mil_dtype def) {
    for (const auto & in : b->inputs) {
        if (in.name == name) {
            return in.dtype;
        }
    }
    for (const auto & op : b->ops) {
        if (op.output == name) {
            return op.out_dtype;
        }
    }
    return def;
}

std::string ts_mil_add_input(ts_mil_builder * b, const char * name,
                             ts_mil_dtype dtype, const int64_t * shape, int64_t rank) {
    ts_mil_value v;
    v.name  = name;
    v.dtype = dtype;
    v.shape.assign(shape, shape + rank);
    b->inputs.push_back(v);
    return v.name;
}

static std::string ts_mil_push_op(ts_mil_builder * b, const ts_mil_op & op) {
    b->ops.push_back(op);
    return op.output;
}

std::string ts_mil_const(ts_mil_builder * b, const char * hint,
                         ts_mil_dtype dtype, const int64_t * shape, int64_t rank,
                         const char * blob_name) {
    ts_mil_op op;
    op.op_type  = "const";
    op.output   = ts_mil_fresh(b, hint);
    op.out_dtype = dtype;
    op.out_shape.assign(shape, shape + rank);
    op.attrs.push_back({"val", std::string("@") + (blob_name ? blob_name : op.output)});
    return ts_mil_push_op(b, op);
}

std::string ts_mil_matmul(ts_mil_builder * b, const char * x, const char * w, bool transpose_y) {
    ts_mil_op op;
    op.op_type = "matmul";
    op.output  = ts_mil_fresh(b, "matmul");
    op.inputs.push_back({"x", x});
    op.inputs.push_back({"y", w});
    op.attrs.push_back({"transpose_x", "false"});
    op.attrs.push_back({"transpose_y", transpose_y ? "true" : "false"});

    // shape inference: x [M, K], w [K, N] (or [N, K] when transpose_y) -> [M, N]
    const std::vector<int64_t> * sx = ts_mil_find_shape(b, x);
    const std::vector<int64_t> * sw = ts_mil_find_shape(b, w);
    op.out_dtype = ts_mil_find_dtype(b, x, TS_MIL_FP16);
    if (sx && sw && sx->size() >= 2 && sw->size() >= 2) {
        int64_t M = (*sx)[sx->size() - 2];
        int64_t N = transpose_y ? (*sw)[sw->size() - 2] : (*sw)[sw->size() - 1];
        op.out_shape = {M, N};
    }
    return ts_mil_push_op(b, op);
}

std::string ts_mil_add(ts_mil_builder * b, const char * x, const char * y) {
    ts_mil_op op;
    op.op_type = "add";
    op.output  = ts_mil_fresh(b, "add");
    op.inputs.push_back({"x", x});
    op.inputs.push_back({"y", y});
    op.out_dtype = ts_mil_find_dtype(b, x, TS_MIL_FP16);
    const std::vector<int64_t> * sx = ts_mil_find_shape(b, x);
    if (sx) {
        op.out_shape = *sx;
    }
    return ts_mil_push_op(b, op);
}

std::string ts_mil_relu(ts_mil_builder * b, const char * x) {
    ts_mil_op op;
    op.op_type = "relu";
    op.output  = ts_mil_fresh(b, "relu");
    op.inputs.push_back({"x", x});
    op.out_dtype = ts_mil_find_dtype(b, x, TS_MIL_FP16);
    const std::vector<int64_t> * sx = ts_mil_find_shape(b, x);
    if (sx) {
        op.out_shape = *sx;
    }
    return ts_mil_push_op(b, op);
}

std::string ts_mil_reshape(ts_mil_builder * b, const char * x, const int64_t * shape, int64_t rank) {
    ts_mil_op op;
    op.op_type = "reshape";
    op.output  = ts_mil_fresh(b, "reshape");
    op.inputs.push_back({"x", x});
    std::string s;
    for (int64_t i = 0; i < rank; i++) {
        if (i) {
            s += ",";
        }
        s += std::to_string(shape[i]);
    }
    op.attrs.push_back({"shape", s});
    op.out_dtype = ts_mil_find_dtype(b, x, TS_MIL_FP16);
    op.out_shape.assign(shape, shape + rank);
    return ts_mil_push_op(b, op);
}

std::string ts_mil_tessera_dequant(ts_mil_builder * b,
                                   const char * packed,
                                   const char * page_scales,
                                   const char * lane_scales,
                                   const char * outlier_offsets,
                                   const char * outlier_cols,
                                   const char * outlier_vals,
                                   const char * act_scale,
                                   const int64_t * out_shape, int64_t rank) {
    ts_mil_op op;
    op.op_type = "tessera_t640_dequant";
    op.output  = ts_mil_fresh(b, "dequant");
    op.inputs.push_back({"packed", packed});
    op.inputs.push_back({"page_scales", page_scales});
    op.inputs.push_back({"lane_scales", lane_scales});
    op.inputs.push_back({"outlier_offsets", outlier_offsets});
    op.inputs.push_back({"outlier_cols", outlier_cols});
    op.inputs.push_back({"outlier_vals", outlier_vals});
    op.inputs.push_back({"act_scale", act_scale});
    // attributes per design section 3.4
    op.attrs.push_back({"page_size", std::to_string(TS_T640_PAGE_SIZE)});
    op.attrs.push_back({"lane_size", std::to_string(TS_T640_LANE_SIZE)});
    op.attrs.push_back({"lanes_per_page", std::to_string(TS_T640_LANES_PER_PAGE)});
    op.out_dtype = TS_MIL_FP16;
    op.out_shape.assign(out_shape, out_shape + rank);
    return ts_mil_push_op(b, op);
}

void ts_mil_add_output(ts_mil_builder * b, const char * name) {
    ts_mil_value v;
    v.name  = name;
    v.dtype = ts_mil_find_dtype(b, name, TS_MIL_FP16);
    const std::vector<int64_t> * s = ts_mil_find_shape(b, name);
    if (s) {
        v.shape = *s;
    }
    b->outputs.push_back(v);
}

//
// SSA validation
//

int ts_mil_build(const ts_mil_builder * b, std::string * err_msg) {
    std::set<std::string> defined;
    for (const auto & in : b->inputs) {
        defined.insert(in.name);
    }
    for (const auto & op : b->ops) {
        for (const auto & kv : op.inputs) {
            if (defined.find(kv.second) == defined.end()) {
                ts_mil_set_err(err_msg, "op '" + op.op_type + "' references undefined value '" +
                               kv.second + "' (arg '" + kv.first + "')");
                return -1;
            }
        }
        if (defined.count(op.output)) {
            ts_mil_set_err(err_msg, "duplicate SSA output '" + op.output + "'");
            return -1;
        }
        defined.insert(op.output);
    }
    if (b->outputs.empty()) {
        ts_mil_set_err(err_msg, "program has no block outputs");
        return -1;
    }
    for (const auto & out : b->outputs) {
        if (defined.find(out.name) == defined.end()) {
            ts_mil_set_err(err_msg, "block output '" + out.name + "' is undefined");
            return -1;
        }
    }
    return 0;
}

//
// protobuf-JSON emit (JSON mapping of mil_spec.Program)
//

static std::string ts_mil_type_json(ts_mil_dtype dtype, const std::vector<int64_t> & shape) {
    std::string s = "{\"tensorType\": {\"dataType\": \"";
    s += ts_mil_dtype_name(dtype);
    s += "\", \"shape\": [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) {
            s += ", ";
        }
        s += std::to_string(shape[i]);
    }
    s += "]}}";
    return s;
}

std::string ts_mil_to_json(const ts_mil_builder * b) {
    std::string s;
    s += "{\n";
    s += "  \"functions\": {\n";
    s += "    \"" + ts_mil_json_escape(b->function_name) + "\": {\n";

    // inputs
    s += "      \"inputs\": [";
    for (size_t i = 0; i < b->inputs.size(); i++) {
        const auto & in = b->inputs[i];
        s += (i ? ", " : "") + std::string("\n        {\"name\": \"") +
             ts_mil_json_escape(in.name) + "\", \"type\": " +
             ts_mil_type_json(in.dtype, in.shape) + "}";
    }
    s += b->inputs.empty() ? "],\n" : "\n      ],\n";

    // block specialization (opset)
    s += "      \"blockSpecializations\": {\n";
    s += "        \"CoreML" + std::to_string(b->opset) + "\": {\n";

    // operations
    s += "          \"operations\": [";
    for (size_t i = 0; i < b->ops.size(); i++) {
        const auto & op = b->ops[i];
        s += (i ? "," : "") + std::string("\n            {\n");
        s += "              \"type\": \"" + ts_mil_json_escape(op.op_type) + "\",\n";

        s += "              \"outputs\": [{\"name\": \"" + ts_mil_json_escape(op.output) +
             "\", \"type\": " + ts_mil_type_json(op.out_dtype, op.out_shape) + "}],\n";

        s += "              \"inputs\": {";
        for (size_t j = 0; j < op.inputs.size(); j++) {
            s += (j ? ", " : "") + std::string("\"") + ts_mil_json_escape(op.inputs[j].first) +
                 "\": {\"name\": \"" + ts_mil_json_escape(op.inputs[j].second) + "\"}";
        }
        s += "},\n";

        s += "              \"attributes\": {";
        for (size_t j = 0; j < op.attrs.size(); j++) {
            s += (j ? ", " : "") + std::string("\"") + ts_mil_json_escape(op.attrs[j].first) +
                 "\": \"" + ts_mil_json_escape(op.attrs[j].second) + "\"";
        }
        s += "}\n";
        s += "            }";
    }
    s += b->ops.empty() ? "],\n" : "\n          ],\n";

    // block outputs
    s += "          \"outputs\": [";
    for (size_t i = 0; i < b->outputs.size(); i++) {
        s += (i ? ", " : "") + std::string("{\"name\": \"") +
             ts_mil_json_escape(b->outputs[i].name) + "\"}";
    }
    s += "]\n";

    s += "        }\n";
    s += "      }\n";
    s += "    }\n";
    s += "  }\n";
    s += "}\n";
    return s;
}

int ts_mil_emit_json(const ts_mil_builder * b, const char * path, std::string * err_msg) {
    if (path == nullptr) {
        ts_mil_set_err(err_msg, "path is null");
        return -1;
    }
    if (ts_mil_build(b, err_msg) != 0) {
        return -1;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        ts_mil_set_err(err_msg, std::string("cannot open ") + path);
        return -1;
    }
    f << ts_mil_to_json(b);
    if (!f.good()) {
        ts_mil_set_err(err_msg, std::string("write failed for ") + path);
        return -1;
    }
    return 0;
}

//
// Tile640 dequant (matches ggml dequantize_row_tessera_t640 + outliers + act_scale)
//

void ts_coreml_dequant_t640(const ts_coreml_weight_src * src, float * y) {
    const int64_t out_dim = src->out_dim;
    const int64_t in_dim  = src->in_dim;
    const int pages = (int) ((in_dim + TS_T640_PAGE_SIZE - 1) / TS_T640_PAGE_SIZE);

    const uint32_t * packed      = (const uint32_t *) src->packed;
    const uint16_t * page_scales = (const uint16_t *) src->page_scales;
    const int8_t   * lane_scales = (const int8_t   *) src->lane_scales;

    for (int64_t r = 0; r < out_dim; r++) {
        const uint32_t * rp = packed      + r * pages * TS_T640_WORDS_PER_PAGE;
        const uint16_t * rs = page_scales + r * pages;
        const int8_t   * rl = lane_scales + r * pages * TS_T640_LANES_PER_PAGE;
        float * row = y + r * in_dim;

        for (int p = 0; p < pages; p++) {
            const float page_max = ts_f16_to_f32(rs[p]);
            for (int l = 0; l < TS_T640_LANES_PER_PAGE; l++) {
                const float scale = page_max * (rl[p * TS_T640_LANES_PER_PAGE + l] * (1.0f / 127.0f));
                const int   col0  = p * TS_T640_PAGE_SIZE + l * TS_T640_LANE_SIZE;

                uint32_t rem = rp[p * TS_T640_WORDS_PER_PAGE + l];
                for (int g = 0; g < 4; g++) {
                    uint32_t idx = rem % 243;
                    rem /= 243;
                    for (int d = 0; d < 5; d++) {
                        const int col = col0 + g * 5 + d;
                        if (col >= in_dim) {
                            break;
                        }
                        const uint32_t trit = idx % 3;
                        idx /= 3;
                        row[col] = trit == 1 ? scale : trit == 2 ? -scale : 0.0f;
                    }
                }
            }
        }

        // outlier replacement (CSR over the row)
        if (src->outlier_row_offsets && src->outlier_cols && src->outlier_vals) {
            const int32_t  * offs = (const int32_t  *) src->outlier_row_offsets;
            const int32_t  * cols = (const int32_t  *) src->outlier_cols;
            const uint16_t * vals = (const uint16_t *) src->outlier_vals;
            for (int32_t i = offs[r]; i < offs[r + 1]; i++) {
                row[cols[i]] = ts_f16_to_f32(vals[i]);
            }
        }

        // per-input-channel act_scale (folded only when present)
        if (src->act_scale) {
            const uint16_t * as = (const uint16_t *) src->act_scale;
            for (int64_t c = 0; c < in_dim; c++) {
                row[c] *= ts_f16_to_f32(as[c]);
            }
        }
    }
}

//
// weight serialization
//

int ts_coreml_serialize_weights(const ts_coreml_weight_src * src,
                                const char * dir,
                                bool custom_op,
                                ts_coreml_weight_out * out,
                                std::string * err_msg) {
    if (src == nullptr || dir == nullptr || out == nullptr) {
        ts_mil_set_err(err_msg, "null argument");
        return -1;
    }
    if (src->out_dim <= 0 || src->in_dim <= 0) {
        ts_mil_set_err(err_msg, "invalid tensor dims");
        return -1;
    }

    const std::string stem = ts_mil_blob_stem(src->name ? src->name : "weight");
    *out = {};

    if (custom_op) {
        // v2: store the raw cluster components as separate blobs. The packed
        // blob is the primary reference the MIL const points at.
        struct comp { const char * suffix; const uint8_t * data; int64_t bytes; };
        const int pages = (int) ((src->in_dim + TS_T640_PAGE_SIZE - 1) / TS_T640_PAGE_SIZE);
        const comp comps[] = {
            {"packed",      src->packed,      src->out_dim * pages * TS_T640_WORDS_PER_PAGE * 4},
            {"page_scales", src->page_scales, src->out_dim * pages * 2},
            {"lane_scales", src->lane_scales, src->out_dim * pages * TS_T640_LANES_PER_PAGE},
        };
        int64_t total = 0;
        for (const auto & c : comps) {
            if (!c.data) {
                continue;
            }
            std::string path = std::string(dir) + "/" + stem + "." + c.suffix + ".bin";
            std::ofstream f(path, std::ios::binary);
            if (!f) {
                ts_mil_set_err(err_msg, "cannot write " + path);
                return -1;
            }
            f.write((const char *) c.data, c.bytes);
            total += c.bytes;
        }
        out->blob_name = ts_coreml_weight_blob_name(src->name, true);
        out->n_bytes   = total;
        out->custom_op = true;
        return 0;
    }

    // v1 stock ops: dequantize to fp16 [out, in]. act_scale is NOT folded into
    // the weight blob (decision C8: act_scale stays a runtime-selectable axis);
    // dequantize on a view with act_scale cleared.
    if (!src->packed || !src->page_scales || !src->lane_scales) {
        ts_mil_set_err(err_msg, "v1 serialization requires packed/page_scales/lane_scales");
        return -1;
    }
    ts_coreml_weight_src view = *src;
    view.act_scale = nullptr;

    std::vector<float> dense((size_t) (src->out_dim * src->in_dim));
    ts_coreml_dequant_t640(&view, dense.data());

    std::vector<uint16_t> f16(dense.size());
    for (size_t i = 0; i < dense.size(); i++) {
        f16[i] = ts_f32_to_f16(dense[i]);
    }

    std::string path = std::string(dir) + "/" + stem + ".fp16.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        ts_mil_set_err(err_msg, "cannot write " + path);
        return -1;
    }
    f.write((const char *) f16.data(), (std::streamsize) (f16.size() * sizeof(uint16_t)));
    if (!f.good()) {
        ts_mil_set_err(err_msg, "write failed for " + path);
        return -1;
    }

    out->blob_name = ts_coreml_weight_blob_name(src->name, false);
    out->n_bytes   = (int64_t) f16.size() * 2;
    out->custom_op = false;
    return 0;
}
