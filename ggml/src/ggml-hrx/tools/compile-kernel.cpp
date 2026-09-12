#include "../loom-jit.h"
#include "tool-utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using ggml::hrx::tool::report_status;
using ggml::hrx::tool::write_file;

int main(int argc, char ** argv) {
    std::string              target;
    std::string              source_path;
    std::string              root;
    std::string              output_path;
    std::vector<std::string> config_storage;
    std::vector<int64_t>     workload;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--target" && i + 1 < argc) {
            target = argv[++i];
        } else if (argument == "--source" && i + 1 < argc) {
            source_path = argv[++i];
        } else if (argument == "--root" && i + 1 < argc) {
            root = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argument == "--config" && i + 1 < argc) {
            config_storage.emplace_back(argv[++i]);
        } else if (argument == "--workload" && i + 1 < argc) {
            workload.push_back(std::stoll(argv[++i]));
        } else {
            std::cerr << "unknown or incomplete argument: " << argument << '\n';
            return 2;
        }
    }
    if (target.empty() || source_path.empty() || root.empty() || output_path.empty()) {
        std::cerr << "usage: ggml-hrx-compile-kernel --target gfx... --source linked.loom --root symbol "
                     "--output dir [--config key=value] [--workload value]\n";
        return 2;
    }
    const std::string source = ggml::hrx::tool::read_file(source_path);
    if (source.empty()) {
        std::cerr << "cannot read Loom source: " << source_path << '\n';
        return 2;
    }
    std::vector<ggml_hrx_loom_jit_config_binding> configs;
    std::vector<std::string>                      config_keys;
    std::vector<std::string>                      config_values;
    for (const std::string & item : config_storage) {
        const size_t equals = item.find('=');
        if (equals == std::string::npos) {
            std::cerr << "invalid config binding: " << item << '\n';
            return 2;
        }
        config_keys.push_back(item.substr(0, equals));
        config_values.push_back(item.substr(equals + 1));
    }
    for (size_t i = 0; i < config_keys.size(); ++i) {
        configs.push_back({ config_keys[i].c_str(), config_values[i].c_str() });
    }

    ggml_hrx_loom_jit_amdgpu_options jit_options;
    jit_options.processor             = target.c_str();
    jit_options.identifier            = target.c_str();
    ggml_hrx_loom_jit_amdgpu * jit    = nullptr;
    hrx_status_t               status = ggml_hrx_loom_jit_amdgpu_create(&jit_options, &jit);
    if (!report_status(status, "create JIT")) {
        return 1;
    }

    ggml_hrx_loom_jit_compile_options options;
    options.source_data             = source.data();
    options.source_size             = source.size();
    options.source_format           = GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
    options.source_identifier       = source_path.c_str();
    options.root_symbol             = root.c_str();
    options.module_name             = root.c_str();
    options.artifact_identifier     = root.c_str();
    options.config_bindings         = configs.data();
    options.config_binding_count    = configs.size();
    options.workload_arguments      = workload.data();
    options.workload_argument_count = workload.size();
    options.evaluate_launch_config  = !workload.empty();
    ggml_hrx_loom_jit_compile_result result;
    status = ggml_hrx_loom_jit_amdgpu_compile(jit, &options, &result);
    if (!hrx_status_is_ok(status)) {
        ggml_hrx_loom_jit_amdgpu_release(jit);
        report_status(status, "compile kernel");
        return 1;
    }
    std::filesystem::create_directories(output_path);
    write_file(std::filesystem::path(output_path) / "kernel.hsaco", result.hsaco_data, result.hsaco_size);
    write_file(std::filesystem::path(output_path) / "manifest.json", result.manifest_json, result.manifest_json_size);
    write_file(std::filesystem::path(output_path) / "compile-report.json", result.compile_report_json,
               result.compile_report_json_size);
    write_file(std::filesystem::path(output_path) / "final.loom", result.final_module_text,
               result.final_module_text_size);
    std::ofstream launch(std::filesystem::path(output_path) / "launch.txt", std::ios::trunc);
    launch << "target=" << target << '\n'
           << "root=" << root << '\n'
           << "workgroups=" << result.launch_config.workgroup_count[0] << ',' << result.launch_config.workgroup_count[1]
           << ',' << result.launch_config.workgroup_count[2] << '\n'
           << "workgroup_size=" << result.launch_config.workgroup_size[0] << ','
           << result.launch_config.workgroup_size[1] << ',' << result.launch_config.workgroup_size[2] << '\n';
    ggml_hrx_loom_jit_amdgpu_release(jit);
    return 0;
}
