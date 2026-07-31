#include "tessera-coreml-builder.h"

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
