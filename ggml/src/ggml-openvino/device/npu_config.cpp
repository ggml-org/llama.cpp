#include "npu_config.h"

#include "ggml-impl.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cstring>
#include <openvino/runtime/properties.hpp>

// The environment does not set the NPU compiler type. Set by ggml_openvino_npu_build_compile_config().
static bool npu_compiler_type_is_default = false;

ov::AnyMap ggml_openvino_npu_build_compile_config(const char * cache_dir) {
    ov::AnyMap compile_config = {
        {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
        {"NPU_USE_NPUW",                      "YES"   },
        {"NPUW_DEVICES",                      "NPU"   },
        {"NPUW_FOLD",                         "YES"   },
        {"NPUW_WEIGHTS_BANK",                 "shared"},
        {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
        {"NPUW_FUNCALL_ASYNC",                "YES"   },
        {"NPUW_DQ",                           "YES"   },
        {"NPUW_DQ_FULL",                      "NO"    },
    };
    if (cache_dir && strlen(cache_dir) > 0) {
        // Keep the default cache mode: CACHE_MODE=OPTIMIZE_SIZE makes the NPU plugin compile the NPUW submodels with
        // weights separation, which fails for them.
        compile_config["NPUW_CACHE_DIR"] = cache_dir;
    }
    // Copy an optional string env var into the compile config only when it is set and non-empty.
    auto set_compile_option_from_env = [&](const char * env_var, const char * config_key) {
        const char * value = ggml_openvino_getenv_str(env_var);
        if (value && strlen(value) > 0) {
            compile_config[config_key] = value;
        }
    };
    // Legacy alias for GGML_OPENVINO_NPU_CONFIG=NPU_COMPILATION_MODE_PARAMS=...
    set_compile_option_from_env("GGML_OPENVINO_NPU_COMPILE_CONFIG", "NPU_COMPILATION_MODE_PARAMS");
    // PLUGIN | DRIVER | PREFER_PLUGIN. The in-plugin compiler and the driver compiler
    // can differ substantially in generated code quality for the same op. On the Intel
    // NPU the driver compiler emits markedly faster prefill kernels (~1940 vs ~1540 t/s
    // pp1024 on phi-4-mini), matching OpenVINO GenAI, so default to DRIVER below.
    set_compile_option_from_env("GGML_OPENVINO_NPU_COMPILER_TYPE", "NPU_COMPILER_TYPE");
    // NPUW_FUNCALL_FOR_ALL=YES hangs the NPU (device lost via TDR) for context
    // lengths >= ~786 on the 2026.3 in-plugin compiler. Allow turning it off.
    set_compile_option_from_env("GGML_OPENVINO_NPUW_FUNCALL_FOR_ALL", "NPUW_FUNCALL_FOR_ALL");
    // Unfolds function calls into separate infer requests, trading memory for the
    // per-call dispatch overhead that repeated funcalls otherwise pay.
    set_compile_option_from_env("GGML_OPENVINO_NPUW_UNFOLD_IREQS", "NPUW_UNFOLD_IREQS");
    // The compiler runs one llvm worker per core by default; each carries its own
    // working set, so large graphs can exhaust host RAM. Capping the workers trades
    // compile time for peak memory.
    set_compile_option_from_env("GGML_OPENVINO_COMPILATION_NUM_THREADS", "COMPILATION_NUM_THREADS");
    // NPUW profiling dumps; need an OpenVINO build with NPU_PLUGIN_DEVELOPER_BUILD.
    set_compile_option_from_env("GGML_OPENVINO_NPUW_DUMP_SUBS", "NPUW_DUMP_SUBS");
    set_compile_option_from_env("GGML_OPENVINO_NPUW_DUMP_SUBS_DIR", "NPUW_DUMP_SUBS_DIR");
    set_compile_option_from_env("GGML_OPENVINO_NPUW_DUMP_SUBS_ON_FAIL", "NPUW_DUMP_SUBS_ON_FAIL");
    set_compile_option_from_env("GGML_OPENVINO_NPUW_DUMP_IO", "NPUW_DUMP_IO");
    // Comma-separated KEY=VALUE pairs appended last, so they override anything above.
    // Escape hatch for bisecting plugin options without a rebuild.
    ggml_openvino_apply_config_overrides(compile_config, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_CONFIG"), "NPU");
    npu_compiler_type_is_default = compile_config.find("NPU_COMPILER_TYPE") == compile_config.end();
    if (npu_compiler_type_is_default) {
        compile_config["NPU_COMPILER_TYPE"] = "DRIVER";
    }
    return compile_config;
}

bool ggml_openvino_npu_gdn_uses_plugin_compiler() {
    return ggml_openvino_is_npu() && npu_compiler_type_is_default;
}

ov::AnyMap ggml_openvino_npu_gdn_config(const ov::AnyMap & base) {
    ov::AnyMap gdn_config = base;
    if (ggml_openvino_npu_gdn_uses_plugin_compiler()) {
        gdn_config["NPU_COMPILER_TYPE"] = "PLUGIN";
    }
    return gdn_config;
}

ov::AnyMap ggml_openvino_npu_without_npuw(const ov::AnyMap & config) {
    ov::AnyMap out;
    for (const auto & kv : config) {
        if (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            continue;
        }
        out.insert(kv);
    }
    return out;
}

ov::AnyMap ggml_openvino_npu_prefill_config(const ov::AnyMap & base, const GgmlOvDecoder & decoder) {
    ov::AnyMap prefill_config = base;
    if (ggml_openvino_npu_hfa_prefill_enabled()) {
        prefill_config["NPUW_ATTN_HISTORY_SIZE_INPUT"] = decoder.get_hfa_history_input_name();
        prefill_config["NPUW_ONLINE_PIPELINE"] = "REP";
        prefill_config["NPUW_ONLINE_ISOLATE"] = "ATTN";
        prefill_config["NPUW_ONLINE_KEEP_BLOCK_SIZE"] = "4";
        prefill_config["NPUW_UNFOLD_IREQS"] = "NO";
        prefill_config["NPUW_ATTN"] = "HFA";
        prefill_config["NPUW_ATTN_HFA_FUSED"] = "YES";
    }
    return prefill_config;
}

ov::AnyMap ggml_openvino_npu_decode_config(const ov::AnyMap & base, const std::string & device) {
    ov::AnyMap decode_config = base;
    if (device == "NPU" && ggml_openvino_npu_decode_unfold_enabled() &&
        decode_config.find("NPUW_UNFOLD_IREQS") == decode_config.end()) {
        decode_config["NPUW_UNFOLD_IREQS"] = "YES";
    }
    return decode_config;
}
