#include "tessera-coreml.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#endif

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
// compilation (C9 via the coremlcompiler CLI; design 1.6 / 4.5 step 8)
//

#ifndef _WIN32
static std::string ts_coreml_shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Run a command, capture combined stdout+stderr, return the exit code (-1 if it
// could not be launched).
static int ts_coreml_run_capture(const std::string & cmd, std::string * out) {
    std::string buf;
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        return -1;
    }
    char chunk[512];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), p)) > 0) {
        buf.append(chunk, n);
    }
    int status = pclose(p);
    if (out) {
        *out = buf;
    }
    if (status == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

static bool ts_coreml_is_dir(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool ts_coreml_dir_has_file(const std::string & dir) {
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return false;
    }
    bool any = false;
    struct dirent * e;
    while ((e = readdir(d)) != nullptr) {
        if (std::string(e->d_name) != "." && std::string(e->d_name) != "..") {
            any = true;
            break;
        }
    }
    closedir(d);
    return any;
}
#endif // _WIN32

bool ts_coreml_xcrun_available() {
#ifndef _WIN32
    return std::system("xcrun --version > /dev/null 2>&1") == 0;
#else
    return false;
#endif
}

int ts_coreml_compile(const char * mlpackage_path,
                      const char * output_dir,
                      std::string * err_msg) {
#ifdef _WIN32
    (void) mlpackage_path; (void) output_dir;
    ts_coreml_set_err(err_msg, "coremlcompiler is not available on Windows");
    return -1;
#else
    if (mlpackage_path == nullptr || output_dir == nullptr) {
        ts_coreml_set_err(err_msg, "null argument");
        return -1;
    }
    if (!ts_coreml_xcrun_available()) {
        ts_coreml_set_err(err_msg,
            "Xcode command-line tools not found (xcrun); cannot compile .mlpackage");
        return -1;
    }

    const std::string cmd = "xcrun coremlcompiler compile " +
        ts_coreml_shell_quote(mlpackage_path) + " " +
        ts_coreml_shell_quote(output_dir) + " 2>&1";

    std::string log;
    const int rc = ts_coreml_run_capture(cmd, &log);
    if (rc != 0) {
        ts_coreml_set_err(err_msg, "coremlcompiler failed (exit " + std::to_string(rc) +
            "): " + log);
        return -1;
    }

    // coremlcompiler writes <output_dir>/<base>.mlmodelc; derive the base name
    // from the package path (strip a trailing ".mlpackage" and any directory).
    std::string pkg(mlpackage_path);
    while (!pkg.empty() && pkg.back() == '/') {
        pkg.pop_back();
    }
    std::string base = pkg.substr(pkg.find_last_of('/') + 1);
    const size_t ext = base.rfind(".mlpackage");
    if (ext != std::string::npos) {
        base = base.substr(0, ext);
    }
    const std::string mlmodelc = std::string(output_dir) + "/" + base + ".mlmodelc";

    if (!ts_coreml_is_dir(mlmodelc) || !ts_coreml_dir_has_file(mlmodelc)) {
        ts_coreml_set_err(err_msg, "compilation produced no valid model at " + mlmodelc);
        return -1;
    }

    return 0;
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
