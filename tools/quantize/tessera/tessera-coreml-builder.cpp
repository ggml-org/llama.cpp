#include "tessera-coreml-builder.h"

#include "tessera-coreml.h"
#include "tessera-coreml-mil.h"
#include "tessera-coreml-telemetry.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>

static bool ts_mkdir(const std::string & path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

static std::string ts_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

int64_t ts_coreml_total_weight_bytes(const ts_coreml_builder_tensor * tensors,
                                     int64_t n_tensors) {
    int64_t total = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        total += tensors[i].out_dim * tensors[i].in_dim * 2; // fp16
        if (tensors[i].has_act_scale && tensors[i].act_scale_f16) {
            total += tensors[i].in_dim * 2;
        }
    }
    return total;
}

int ts_coreml_build_package(const ts_coreml_builder_tensor * tensors,
                            int64_t n_tensors,
                            const ts_coreml_builder_params * params,
                            std::string * err_msg) {
    if (params == nullptr) {
        if (err_msg) *err_msg = "params is null";
        return -1;
    }
    if (n_tensors > 0 && tensors == nullptr) {
        if (err_msg) *err_msg = "tensors is null";
        return -1;
    }

    const std::string & root = params->output_path;
    const std::string data_dir = root + "/Data";
    const std::string meta_dir = root + "/Metadata";

    if (!ts_mkdir(root) || !ts_mkdir(data_dir) || !ts_mkdir(meta_dir)) {
        if (err_msg) *err_msg = "cannot create directory: " + root;
        return -1;
    }

    // --- Manifest.json ---
    {
        std::ofstream f(root + "/Manifest.json");
        if (!f) {
            if (err_msg) *err_msg = "cannot write Manifest.json";
            return -1;
        }
        f << "{\n";
        f << "  \"fileFormatVersion\": \"1.0.0\",\n";
        f << "  \"itemInfoEntries\": {\n";
        f << "    \"model\": {\n";
        f << "      \"author\": \"Tessera\",\n";
        f << "      \"description\": \"Tessera-quantized model (stock ops v1)\",\n";
        f << "      \"name\": \"model.mlmodel\",\n";
        f << "      \"path\": \"Data/model.mlmodel\"\n";
        f << "    }\n";
        f << "  }\n";
        f << "}\n";
    }

    // --- Metadata/model.json ---
    {
        std::ofstream f(meta_dir + "/model.json");
        if (!f) {
            if (err_msg) *err_msg = "cannot write Metadata/model.json";
            return -1;
        }
        f << "{\n";
        f << "  \"name\": \"" << ts_json_escape(params->model_name) << "\",\n";
        f << "  \"framework\": \"tessera\",\n";
        f << "  \"opset\": \"stock-v1\",\n";
        f << "  \"compute_units\": " << params->compute_units << ",\n";
        f << "  \"n_tensors\": " << n_tensors << ",\n";
        f << "  \"total_weight_bytes\": " << ts_coreml_total_weight_bytes(tensors, n_tensors) << "\n";
        f << "}\n";
    }

    // --- Data/model.mlmodel ---
    // JSON-based model spec with embedded weight file references.
    // The actual weight blobs are written as separate .bin files and
    // referenced by path. The Objective-C compilation layer reads this
    // spec and produces the protobuf .mlmodelc.
    {
        std::ofstream f(data_dir + "/model.mlmodel");
        if (!f) {
            if (err_msg) *err_msg = "cannot write Data/model.mlmodel";
            return -1;
        }
        f << "{\n";
        f << "  \"specificationVersion\": 7,\n";
        f << "  \"description\": {\n";
        f << "    \"metadata\": {\n";
        f << "      \"author\": \"Tessera\",\n";
        f << "      \"shortDescription\": \"Stock ops v1, pre-dequantized fp16\"\n";
        f << "    }\n";
        f << "  },\n";

        // Layer descriptions
        f << "  \"layers\": [\n";
        for (int64_t i = 0; i < n_tensors; i++) {
            const auto & t = tensors[i];
            f << "    {\n";
            f << "      \"name\": \"" << ts_json_escape(t.name) << "\",\n";
            f << "      \"type\": \"innerProduct\",\n";
            f << "      \"out_dim\": " << t.out_dim << ",\n";
            f << "      \"in_dim\": " << t.in_dim << ",\n";
            f << "      \"has_act_scale\": " << (t.has_act_scale ? "true" : "false") << ",\n";

            // Write weight blob
            std::string wname = "weights_" + std::to_string(i) + ".bin";
            {
                std::ofstream wf(data_dir + "/" + wname, std::ios::binary);
                if (!wf) {
                    if (err_msg) *err_msg = "cannot write " + wname;
                    return -1;
                }
                wf.write((const char *) t.weights_f16,
                         t.out_dim * t.in_dim * sizeof(uint16_t));
            }
            f << "      \"weight_file\": \"" << wname << "\",\n";

            if (t.has_act_scale && t.act_scale_f16) {
                std::string aname = "act_scale_" + std::to_string(i) + ".bin";
                std::ofstream af(data_dir + "/" + aname, std::ios::binary);
                if (af) {
                    af.write((const char *) t.act_scale_f16,
                             t.in_dim * sizeof(uint16_t));
                }
                f << "      \"act_scale_file\": \"" << aname << "\",\n";
            }

            f << "      \"quantization\": \"TESSERA_T640_dequantized_fp16\"\n";
            f << "    }" << (i + 1 < n_tensors ? "," : "") << "\n";
        }
        f << "  ]\n";
        f << "}\n";
    }

    return 0;
}

//
// conversion pipeline
//

ts_coreml_convert_params ts_coreml_convert_default_params() {
    ts_coreml_convert_params p;
    p.output_path   = "";
    p.model_name    = "tessera-model";
    p.compute_units = 3;    // cpu+ne: ANE-first (design 4.2 default)
    p.compile       = true;
    p.telemetry     = false;
    p.telemetry_path = "";
    return p;
}

int ts_coreml_convert(const ts_coreml_builder_tensor * tensors,
                      int64_t n_tensors,
                      const ts_coreml_convert_params * params,
                      ts_coreml_convert_result * result,
                      std::string * err_msg) {
    if (params == nullptr || result == nullptr) {
        if (err_msg) *err_msg = "null argument";
        return -1;
    }
    if (n_tensors > 0 && tensors == nullptr) {
        if (err_msg) *err_msg = "tensors is null";
        return -1;
    }
    *result = {};

    const std::string root = params->output_path;
    const std::string data_dir = root + "/Data";
    const std::string meta_dir = root + "/Metadata";
    if (!ts_mkdir(root) || !ts_mkdir(data_dir) || !ts_mkdir(meta_dir)) {
        if (err_msg) *err_msg = "cannot create directory: " + root;
        return -1;
    }
    result->mlpackage_path = root;

    // --- build MIL graph + serialize fp16 weight blobs (stock ops v1) ---
    ts_mil_builder mil;
    ts_mil_builder_init(&mil, "main");

    int64_t weight_bytes = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const ts_coreml_builder_tensor & t = tensors[i];
        const std::string blob = ts_coreml_weight_blob_name(t.name.c_str(), false);

        std::ofstream wf(data_dir + "/" + blob, std::ios::binary);
        if (!wf) {
            if (err_msg) *err_msg = "cannot write " + blob;
            return -1;
        }
        wf.write((const char *) t.weights_f16,
                 (std::streamsize) (t.out_dim * t.in_dim * sizeof(uint16_t)));
        weight_bytes += t.out_dim * t.in_dim * 2;

        // input [1, in_dim] -> const weight [out_dim, in_dim] -> matmul (y^T)
        const std::string in_name = "x_" + std::to_string(i);
        const int64_t in_shape[2] = {1, t.in_dim};
        ts_mil_add_input(&mil, in_name.c_str(), TS_MIL_FP16, in_shape, 2);

        const int64_t w_shape[2] = {t.out_dim, t.in_dim};
        const std::string w = ts_mil_const(&mil, blob.c_str(), TS_MIL_FP16,
                                           w_shape, 2, blob.c_str());

        const std::string y = ts_mil_matmul(&mil, in_name.c_str(), w.c_str(), true);
        ts_mil_add_output(&mil, y.c_str());
    }

    const std::string mil_path = data_dir + "/model.mlmodel";
    if (ts_mil_emit_json(&mil, mil_path.c_str(), err_msg) != 0) {
        return -1;
    }
    result->mil_json_path = mil_path;
    result->n_tensors     = n_tensors;
    result->weight_bytes  = weight_bytes;

    // --- Manifest.json ---
    {
        std::ofstream f(root + "/Manifest.json");
        if (!f) {
            if (err_msg) *err_msg = "cannot write Manifest.json";
            return -1;
        }
        f << "{\n";
        f << "  \"fileFormatVersion\": \"1.0.0\",\n";
        f << "  \"itemInfoEntries\": {\n";
        f << "    \"model\": {\n";
        f << "      \"author\": \"Tessera\",\n";
        f << "      \"description\": \"Tessera model (MIL stock ops v1)\",\n";
        f << "      \"name\": \"model.mlmodel\",\n";
        f << "      \"path\": \"Data/model.mlmodel\"\n";
        f << "    }\n";
        f << "  }\n";
        f << "}\n";
    }

    // --- Metadata/model.json ---
    {
        std::ofstream f(meta_dir + "/model.json");
        if (!f) {
            if (err_msg) *err_msg = "cannot write Metadata/model.json";
            return -1;
        }
        f << "{\n";
        f << "  \"name\": \"" << ts_json_escape(params->model_name) << "\",\n";
        f << "  \"framework\": \"tessera\",\n";
        f << "  \"opset\": \"mil-stock-v1\",\n";
        f << "  \"compute_units\": " << params->compute_units << ",\n";
        f << "  \"n_tensors\": " << n_tensors << ",\n";
        f << "  \"total_weight_bytes\": " << weight_bytes << "\n";
        f << "}\n";
    }

    // --- compile (best-effort: skipped when xcrun is absent) ---
    if (params->compile) {
        if (ts_coreml_xcrun_available()) {
            // compile into the package's parent dir -> <parent>/<base>.mlmodelc
            std::string pkg = root;
            while (pkg.size() > 1 && pkg.back() == '/') {
                pkg.pop_back();
            }
            const size_t slash = pkg.find_last_of('/');
            std::string parent = (slash == std::string::npos) ? "." : pkg.substr(0, slash);
            std::string base   = (slash == std::string::npos) ? pkg : pkg.substr(slash + 1);
            const size_t ext   = base.rfind(".mlpackage");
            if (ext != std::string::npos) {
                base = base.substr(0, ext);
            }
            if (ts_coreml_compile(root.c_str(), parent.c_str(), err_msg) == 0) {
                result->compiled      = true;
                result->mlmodelc_path = parent + "/" + base + ".mlmodelc";
            } else {
                return -1;  // xcrun present but compilation failed
            }
        } else {
            result->note = "compile skipped: xcrun not available";
        }
    }

    // --- telemetry (C4 per-session attribution scaffold) ---
    if (params->telemetry) {
        ts_coreml_telemetry_config cfg = ts_coreml_telemetry_default_config();
        cfg.enable      = true;
        cfg.output_path = params->telemetry_path.empty()
                              ? (root + "/telemetry.json")
                              : params->telemetry_path;

        ts_coreml_telemetry tel;
        if (ts_coreml_telemetry_start(&tel, &cfg, err_msg) != 0) {
            return -1;
        }
        ts_coreml_telemetry_sample_t s;
        for (int k = 0; k < 16; k++) {
            ts_coreml_telemetry_sample(&tel, &s);
        }
        if (ts_coreml_telemetry_write_summary(&tel, cfg.output_path.c_str(), err_msg) != 0) {
            ts_coreml_telemetry_stop(&tel);
            return -1;
        }
        ts_coreml_telemetry_stop(&tel);
        result->telemetry_path = cfg.output_path;
    }

    return 0;
}
