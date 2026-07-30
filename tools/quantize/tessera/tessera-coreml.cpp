#include "tessera-coreml.h"

#include "ggml.h"

#include <fstream>
#include <string>

//
// helpers
//

static const char * TS_COREML_SCHEMA = "tessera.coreml.spec.v1";

static const char * ts_coreml_type_name(int32_t ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_TESSERA_T640:    return "TESSERA_T640";
        case GGML_TYPE_TESSERA_T640_3D: return "TESSERA_T640_3D";
        default:                        return "unknown";
    }
}

static std::string ts_coreml_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
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

static void ts_coreml_set_err(std::string * err_msg, const std::string & msg) {
    if (err_msg) {
        *err_msg = msg;
    }
}

//
// availability
//

bool ts_coreml_available() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

//
// stock ops validation (C1)
//

int ts_coreml_validate_stock_ops(const ts_coreml_tensor_desc * tensors,
                                 int64_t n_tensors,
                                 std::string * err_msg) {
    if (n_tensors > 0 && tensors == nullptr) {
        ts_coreml_set_err(err_msg, "tensors is null");
        return -1;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const ts_coreml_tensor_desc & t = tensors[i];
        if (t.ggml_type == GGML_TYPE_TESSERA_T640) {
            continue; // stock ops v1
        }
        if (t.ggml_type == GGML_TYPE_TESSERA_T640_3D) {
            ts_coreml_set_err(err_msg, "tensor '" + t.name +
                "' is TESSERA_T640_3D; 3D expert banks require custom ops (v2)");
            return -1;
        }
        ts_coreml_set_err(err_msg, "tensor '" + t.name +
            "' has ggml type " + std::to_string(t.ggml_type) +
            " which is not representable with stock CoreML ops v1");
        return -1;
    }

    return 0;
}

//
// spec generation (C2, C9)
//

int ts_coreml_generate_spec(const ts_coreml_tensor_desc * tensors,
                            int64_t n_tensors,
                            const ts_coreml_params * params,
                            const char * spec_output_path,
                            std::string * err_msg) {
    if (params == nullptr) {
        ts_coreml_set_err(err_msg, "params is null");
        return -1;
    }
    if (spec_output_path == nullptr) {
        ts_coreml_set_err(err_msg, "spec_output_path is null");
        return -1;
    }

    if (ts_coreml_validate_stock_ops(tensors, n_tensors, err_msg) != 0) {
        return -1;
    }

    std::ofstream f(spec_output_path, std::ios::binary);
    if (!f) {
        ts_coreml_set_err(err_msg, std::string("cannot open ") + spec_output_path);
        return -1;
    }

    f << "{\n";
    f << "  \"schema\": \"" << TS_COREML_SCHEMA << "\",\n";
    f << "  \"format\": \"mlprogram\",\n";
    f << "  \"opset\": \"stock-v1\",\n";

    f << "  \"params\": {\n";
    f << "    \"output_dir\": \""      << ts_coreml_json_escape(params->output_dir) << "\",\n";
    f << "    \"mmap_weights\": "       << (params->mmap_weights    ? "true" : "false") << ",\n";
    f << "    \"ram_activations\": "    << (params->ram_activations ? "true" : "false") << ",\n";
    f << "    \"compute_units\": "      << params->compute_units << ",\n";
    f << "    \"allow_fallback\": "     << (params->allow_fallback  ? "true" : "false") << "\n";
    f << "  },\n";

    f << "  \"tensors\": [\n";
    for (int64_t i = 0; i < n_tensors; i++) {
        const ts_coreml_tensor_desc & t = tensors[i];
        f << "    {\n";
        f << "      \"name\": \""           << ts_coreml_json_escape(t.name) << "\",\n";
        f << "      \"out_dim\": "          << t.out_dim << ",\n";
        f << "      \"in_dim\": "           << t.in_dim << ",\n";
        f << "      \"ggml_type\": "        << t.ggml_type << ",\n";
        f << "      \"ggml_type_name\": \"" << ts_coreml_type_name(t.ggml_type) << "\",\n";
        f << "      \"has_act_scale\": "    << (t.has_act_scale ? "true" : "false") << "\n";
        f << "    }" << (i + 1 < n_tensors ? "," : "") << "\n";
    }
    f << "  ],\n";

    f << "  \"n_tensors\": " << n_tensors << "\n";
    f << "}\n";

    if (!f.good()) {
        ts_coreml_set_err(err_msg, std::string("write failed for ") + spec_output_path);
        return -1;
    }

    return 0;
}

//
// defaults
//

ts_coreml_params ts_coreml_default_params() {
    ts_coreml_params p;
    p.output_dir      = "";
    p.mmap_weights    = true;
    p.ram_activations = true;
    p.compute_units   = 0; // all
    p.allow_fallback  = true;
    return p;
}
