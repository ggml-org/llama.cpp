#include "spec_sidecar.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "../src/spec_sidecar/artifact_manifest.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <filesystem>
#include <utility>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef _WIN32
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <dlfcn.h>
#endif

namespace {

static bool is_absolute_path(const std::string & path) {
#ifdef _WIN32
    return (path.size() >= 3 && std::isalpha(static_cast<unsigned char>(path[0])) &&
            path[1] == ':' && (path[2] == '\\' || path[2] == '/')) ||
            path.rfind("\\\\", 0) == 0 || path.rfind("//", 0) == 0;
#else
    return !path.empty() && path[0] == '/';
#endif
}

static void * open_library(const std::string & path, std::string & error) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) {
        error = "LoadLibrary failed for " + path;
        return nullptr;
    }
    return reinterpret_cast<void *>(handle);
#else
    dlerror();
    void * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char * detail = dlerror();
        error = "dlopen failed for " + path + (detail ? ": " + std::string(detail) : "");
        return nullptr;
    }
    return handle;
#endif
}

static void close_library(void * handle) {
    if (handle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

template<typename Fn>
static bool resolve_symbol(void * handle, const char * name, Fn & result, std::string & error) {
#ifdef _WIN32
    FARPROC symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
    if (symbol == nullptr) {
        error = std::string("missing sidecar export: ") + name;
        return false;
    }
    static_assert(sizeof(Fn) == sizeof(symbol), "function/data pointer size mismatch");
    std::memcpy(&result, &symbol, sizeof(result));
#else
    dlerror();
    void * symbol = dlsym(handle, name);
    const char * detail = dlerror();
    if (detail != nullptr || symbol == nullptr) {
        error = std::string("missing sidecar export: ") + name;
        return false;
    }
    static_assert(sizeof(Fn) == sizeof(symbol), "function/data pointer size mismatch");
    std::memcpy(&result, &symbol, sizeof(result));
#endif
    return true;
}

static bool require_absolute(const std::string & path, const char * label, std::string & error) {
    if (!is_absolute_path(path)) {
        error = std::string(label) + " must be an absolute path";
        return false;
    }
    return true;
}

static bool require_file(const std::string & path, const char * label, std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        error = std::string(label) + " is not readable: " + path;
        return false;
    }
    return true;
}

static std::string join_path(const std::string & dir, const char * name) {
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif
    return dir + separator + name;
}

static const int32_t QWEN35_DFLASH_TARGET_LAYER_IDS[] = { 6, 20, 34, 48, 62 };

static bool profile_mismatch(const common_spec_sidecar_profile & profile,
        const std::string & detail, std::string & error) {
    error = std::string(profile.name != nullptr ? profile.name : "sidecar") +
            " target mismatch: " + detail;
    return false;
}

static bool profile_name_matches(
        const common_spec_sidecar_profile & profile, const char * name) {
    if (profile.target_name == nullptr) {
        return true;
    }
    if (name != nullptr && std::strstr(name, profile.target_name) != nullptr) {
        return true;
    }

    // Some Quark MXFP4 exports retain only the source path's parent marker as
    // general.name. Keep this exception exact; all architecture, size, shape,
    // auxiliary-layer, and vocabulary checks below remain mandatory.
    return name != nullptr && std::strcmp(name, "..") == 0 &&
            profile.target_architecture != nullptr &&
            std::strcmp(profile.target_architecture, "qwen35") == 0 &&
            std::strcmp(profile.target_name, "Qwen3.8-27B") == 0 &&
            profile.target_size_label != nullptr &&
            std::strcmp(profile.target_size_label, "27B") == 0;
}

static bool profile_matches_model(const common_spec_sidecar_profile & profile,
        const llama_model * model, std::string & error) {
    if (model == nullptr) {
        return profile_mismatch(profile, "target model is null", error);
    }

    char arch[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 ||
            profile.target_architecture == nullptr ||
            std::strcmp(arch, profile.target_architecture) != 0) {
        return profile_mismatch(profile, std::string("architecture is not ") +
                (profile.target_architecture != nullptr ? profile.target_architecture : "the provider target"), error);
    }

    if (profile.target_name != nullptr) {
        char name[128] = {};
        // Quantizers may retain a path or add a backend suffix to general.name;
        // require the stable model identity token rather than an exact spelling.
        if (llama_model_meta_val_str(model, "general.name", name, sizeof(name)) < 0 ||
                !profile_name_matches(profile, name)) {
            return profile_mismatch(profile, "model name is not the provider target", error);
        }
    }
    if (profile.target_size_label != nullptr) {
        char label[64] = {};
        if (llama_model_meta_val_str(model, "general.size_label", label, sizeof(label)) < 0 ||
                std::strcmp(label, profile.target_size_label) != 0) {
            return profile_mismatch(profile, "model size label is not the provider target", error);
        }
    }
    if (llama_model_n_embd(model) != profile.target_n_embd ||
            llama_model_n_embd_out(model) != profile.target_n_embd_out ||
            llama_model_n_layer(model) != profile.target_n_layer ||
            llama_vocab_n_tokens(llama_model_get_vocab(model)) != profile.target_n_vocab) {
        return profile_mismatch(profile, "model dimensions or vocabulary differ", error);
    }
    if (profile.target_n_layer_nextn >= 0 &&
            llama_model_n_layer_nextn(model) != profile.target_n_layer_nextn) {
        return profile_mismatch(profile, "auxiliary-layer count differs", error);
    }
    return true;
}

static bool profile_matches_target_file(const common_spec_sidecar_profile & profile,
        const std::string & path, std::string & error) {
    if (path.empty()) {
        return profile_mismatch(profile, "target model path is empty", error);
    }
    const gguf_init_params init_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), init_params);
    if (ctx == nullptr) {
        return profile_mismatch(profile, "target GGUF metadata could not be opened", error);
    }

    bool ok = true;
    const auto has_u32 = [&](const char * key, uint32_t expected, const char * detail) {
        const int64_t id = gguf_find_key(ctx, key);
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32 ||
                gguf_get_val_u32(ctx, id) != expected) {
            ok = profile_mismatch(profile, detail, error);
            return false;
        }
        return true;
    };
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx, arch_id) != GGUF_TYPE_STRING ||
            profile.target_architecture == nullptr ||
            std::strcmp(gguf_get_val_str(ctx, arch_id), profile.target_architecture) != 0) {
        ok = profile_mismatch(profile, "target GGUF architecture differs", error);
    }
    if (ok && profile.target_name != nullptr) {
        const int64_t name_id = gguf_find_key(ctx, "general.name");
        if (name_id < 0 || gguf_get_kv_type(ctx, name_id) != GGUF_TYPE_STRING ||
                !profile_name_matches(profile, gguf_get_val_str(ctx, name_id))) {
            ok = profile_mismatch(profile, "target GGUF model identity differs", error);
        }
    }
    if (ok && profile.target_size_label != nullptr) {
        const int64_t label_id = gguf_find_key(ctx, "general.size_label");
        if (label_id < 0 || gguf_get_kv_type(ctx, label_id) != GGUF_TYPE_STRING ||
                std::strcmp(gguf_get_val_str(ctx, label_id), profile.target_size_label) != 0) {
            ok = profile_mismatch(profile, "target GGUF size label differs", error);
        }
    }
    if (ok) {
        const std::string prefix = std::string(profile.target_architecture) + ".";
        uint32_t auxiliary_layers = profile.target_n_layer_nextn > 0
                ? (uint32_t) profile.target_n_layer_nextn : 0;
        if (profile.target_n_layer_nextn < 0) {
            // The provider does not constrain auxiliary layers; a target GGUF
            // may still carry them (for example an MTP-bearing export used
            // with the DFlash provider), and block_count includes them.
            const int64_t nextn_id = gguf_find_key(ctx, (prefix + "nextn_predict_layers").c_str());
            if (nextn_id >= 0 && gguf_get_kv_type(ctx, nextn_id) == GGUF_TYPE_UINT32) {
                auxiliary_layers = gguf_get_val_u32(ctx, nextn_id);
            }
        }
        ok = has_u32((prefix + "block_count").c_str(),
                (uint32_t) profile.target_n_layer + auxiliary_layers,
                "target GGUF block count differs");
    }
    if (ok && profile.target_n_layer_nextn >= 0) {
        const std::string key = std::string(profile.target_architecture) + ".nextn_predict_layers";
        const int64_t id = gguf_find_key(ctx, key.c_str());
        // Base targets commonly omit a zero-valued auxiliary-layer key. A
        // positive contract remains mandatory and exact.
        if (profile.target_n_layer_nextn == 0 && id < 0) {
            // Accepted: absent means no auxiliary layers.
        } else if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32 ||
                gguf_get_val_u32(ctx, id) != (uint32_t) profile.target_n_layer_nextn) {
            ok = profile_mismatch(profile, "target GGUF auxiliary-layer count differs", error);
        }
    }
    if (ok) {
        const std::string prefix = std::string(profile.target_architecture) + ".";
        ok = has_u32((prefix + "embedding_length").c_str(),
                (uint32_t) profile.target_n_embd, "target GGUF embedding width differs");
    }
    const int64_t vocab_id = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (ok && (vocab_id < 0 || gguf_get_kv_type(ctx, vocab_id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_n(ctx, vocab_id) != (size_t) profile.target_n_vocab)) {
        ok = profile_mismatch(profile, "target GGUF vocabulary differs", error);
    }
    gguf_free(ctx);
    return ok;
}

static const common_spec_sidecar_profile QWEN35_MTP_PROFILE = {
    /* .name                  = */ "qwen35-mtp",
    /* .kind                  = */ COMMON_SPEC_SIDECAR_KIND_MTP,
    /* .target_architecture   = */ "qwen35",
    /* .target_name           = */ "Qwen3.8-27B",
    /* .target_size_label     = */ "27B",
    /* .target_n_embd         = */ 5120,
    /* .target_n_embd_out     = */ 5120,
    /* .target_n_layer        = */ 64,
    /* .target_n_layer_nextn  = */ 1,
    /* .target_n_vocab        = */ 248320,
    /* .mtp_embedding_width  = */ 5120,
    /* .mtp_head_rows         = */ 40960,
    /* .dflash_encoded_width  = */ 0,
    /* .dflash_decoder_width  = */ 0,
    /* .dflash_block_size     = */ 0,
    /* .dflash_selector_top_k = */ 0,
    /* .dflash_head_rows      = */ 0,
    /* .dflash_target_layer_ids = */ nullptr,
    /* .dflash_target_layer_ids_n = */ 0,
    /* .library_env           = */ "LLAMA_SPEC_HIP_SIDECAR",
    /* .artifact_env          = */ "LLAMA_SPEC_HIP_WEIGHTS",
    /* .ids_env               = */ "LLAMA_DRAFT_HEAD_IDS",
    /* .full_head_env         = */ nullptr,
    /* .default_library_name  = */ "spec_hip_sidecar.so",
    /* .default_artifact_dir_name = */ "spec-sidecar-mtp",
    /* .explicit_paths_only   = */ false,
    /* .matches_model         = */ profile_matches_model,
    /* .matches_target_file   = */ profile_matches_target_file,
};

static const common_spec_sidecar_profile QWEN35MOE_MTP_PROFILE = {
    /* .name                  = */ "qwen35moe-mtp",
    /* .kind                  = */ COMMON_SPEC_SIDECAR_KIND_MTP,
    /* .target_architecture   = */ "qwen35moe",
    /* .target_name           = */ "Qwen3.6",
    /* .target_size_label     = */ "35B-A3B",
    /* .target_n_embd         = */ 2048,
    /* .target_n_embd_out     = */ 2048,
    /* .target_n_layer        = */ 40,
    /* .target_n_layer_nextn  = */ 1,
    /* .target_n_vocab        = */ 248320,
    /* .mtp_embedding_width  = */ 2048,
    /* .mtp_head_rows         = */ 40960,
    /* .dflash_encoded_width  = */ 0,
    /* .dflash_decoder_width  = */ 0,
    /* .dflash_block_size     = */ 0,
    /* .dflash_selector_top_k = */ 0,
    /* .dflash_head_rows      = */ 0,
    /* .dflash_target_layer_ids = */ nullptr,
    /* .dflash_target_layer_ids_n = */ 0,
    /* .library_env           = */ "LLAMA_SPEC_QWEN35MOE_HIP_SIDECAR",
    /* .artifact_env          = */ "LLAMA_SPEC_QWEN35MOE_HIP_WEIGHTS",
    /* .ids_env               = */ "LLAMA_QWEN35MOE_DRAFT_HEAD_IDS",
    /* .full_head_env         = */ nullptr,
    /* .default_library_name  = */ "spec_qwen35moe_mtp_sidecar.so",
    /* .default_artifact_dir_name = */ "spec-sidecar-qwen35moe-mtp",
    /* .explicit_paths_only   = */ true,
    /* .matches_model         = */ profile_matches_model,
    /* .matches_target_file   = */ profile_matches_target_file,
};

static const common_spec_sidecar_profile QWEN35_DFLASH_PROFILE = {
    /* .name                  = */ "qwen35-dflash",
    /* .kind                  = */ COMMON_SPEC_SIDECAR_KIND_DFLASH,
    /* .target_architecture   = */ "qwen35",
    /* .target_name           = */ "Qwen3.8-27B",
    /* .target_size_label     = */ "27B",
    /* .target_n_embd         = */ 5120,
    /* .target_n_embd_out     = */ 5120,
    /* .target_n_layer        = */ 64,
    /* .target_n_layer_nextn  = */ -1,
    /* .target_n_vocab        = */ 248320,
    /* .mtp_embedding_width  = */ 0,
    /* .mtp_head_rows         = */ 0,
    /* .dflash_encoded_width  = */ 25600,
    /* .dflash_decoder_width  = */ 5120,
    /* .dflash_block_size     = */ 8,
    /* .dflash_selector_top_k = */ SPEC_SIDECAR_DFLASH_DRAFT_TOP_K,
    /* .dflash_head_rows      = */ 40960,
    /* .dflash_target_layer_ids = */ QWEN35_DFLASH_TARGET_LAYER_IDS,
    /* .dflash_target_layer_ids_n = */ 5,
    /* .library_env           = */ "LLAMA_SPEC_HIP_DFLASH",
    /* .artifact_env          = */ "LLAMA_SPEC_HIP_DFLASH_DIR",
    /* .ids_env               = */ nullptr,
    /* .full_head_env         = */ "LLAMA_SPEC_HIP_FULL_HEAD",
    /* .default_library_name  = */ "spec_dflash_sidecar.so",
    /* .default_artifact_dir_name = */ "spec-sidecar-dflash",
    /* .explicit_paths_only   = */ false,
    /* .matches_model         = */ profile_matches_model,
    /* .matches_target_file   = */ profile_matches_target_file,
};

static const common_spec_sidecar_profile * const ALL_PROFILES[] = {
    &QWEN35_MTP_PROFILE,
    &QWEN35MOE_MTP_PROFILE,
    &QWEN35_DFLASH_PROFILE,
};

static const char * env_value(const char * name) {
    return name != nullptr ? std::getenv(name) : nullptr;
}

static bool profile_paths_are_available(const common_spec_sidecar_profile & profile) {
    if (!profile.explicit_paths_only) {
        return true;
    }
    // The MoE provider is an explicit experimental compatibility path until
    // its sidecar arithmetic and throughput are independently qualified. Do
    // not let merely placing the DLL beside llama-server replace native MTP.
    return env_value(profile.library_env) != nullptr && env_value(profile.artifact_env) != nullptr;
}

static std::string normalize_path(const std::filesystem::path & path) {
    std::error_code ec;
    const auto normalized = std::filesystem::weakly_canonical(path, ec);
    return (ec ? path : normalized).string();
}

static bool is_regular_file(const std::string & path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::is_regular_file(path, ec) && !ec;
}

static bool is_directory(const std::string & path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::is_directory(path, ec) && !ec;
}

static std::string executable_directory() {
#ifdef _WIN32
    char buffer[4096] = {};
    const DWORD n = GetModuleFileNameA(nullptr, buffer, sizeof(buffer));
    if (n == 0 || n >= sizeof(buffer)) {
        return {};
    }
    return normalize_path(std::filesystem::path(std::string(buffer, n)).parent_path());
#else
    char buffer[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n <= 0 || n >= (ssize_t) sizeof(buffer)) {
        return {};
    }
    buffer[n] = '\0';
    return normalize_path(std::filesystem::path(buffer).parent_path());
#endif
}

static std::string home_directory() {
#ifdef _WIN32
    const char * home = env_value("USERPROFILE");
    if (home == nullptr) {
        home = env_value("HOMEDRIVE");
        const char * tail = env_value("HOMEPATH");
        if (home != nullptr && tail != nullptr) {
            return normalize_path(std::filesystem::path(std::string(home) + tail));
        }
    }
#else
    const char * home = env_value("HOME");
#endif
    return home != nullptr ? normalize_path(std::filesystem::path(home)) : std::string();
}

static std::string find_default_library(const common_spec_sidecar_profile & profile) {
    if (profile.default_library_name == nullptr) {
        return {};
    }
    const std::filesystem::path exe = executable_directory();
    const std::vector<std::filesystem::path> candidates = {
        exe / profile.default_library_name,
        exe.parent_path() / "lib" / profile.default_library_name,
        exe.parent_path() / "lib" / "llama.cpp" / profile.default_library_name,
        std::filesystem::path(".") / profile.default_library_name,
    };
    for (const auto & candidate : candidates) {
        const std::string path = normalize_path(candidate);
        if (is_regular_file(path)) {
            return path;
        }
    }
    return {};
}

static std::string find_default_artifact_directory(const common_spec_sidecar_profile & profile) {
    if (profile.default_artifact_dir_name == nullptr) {
        return {};
    }
    const std::filesystem::path exe = executable_directory();
    const std::string home = home_directory();
    const std::vector<std::filesystem::path> roots = {
        exe,
        exe / "spec-sidecar",
        exe.parent_path() / "share" / "llama.cpp" / "spec-sidecar",
        home.empty() ? std::filesystem::path() : std::filesystem::path(home) / ".cache" / "llama.cpp" / "spec-sidecar",
        home.empty() ? std::filesystem::path() : std::filesystem::path(home) / ".local" / "share" / "llama.cpp" / "spec-sidecar",
    };
    for (const auto & root : roots) {
        if (root.empty()) {
            continue;
        }
        const std::filesystem::path candidate = root / profile.default_artifact_dir_name;
        const std::string path = normalize_path(candidate);
        if (is_directory(path)) {
            return path;
        }
    }
    return {};
}

static bool get_file_size(const std::string & path, uint64_t & size, std::string & error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot stat sidecar file: " + path;
        return false;
    }
    const std::streamoff end = file.tellg();
    if (end < 0) {
        error = "cannot determine sidecar file size: " + path;
        return false;
    }
    size = static_cast<uint64_t>(end);
    return true;
}

static bool validate_manifest_blob(const std::string & manifest_path,
        const std::string & blob_path, const char * label, std::string & error) {
    std::vector<spec_sidecar_artifact::TensorDesc> tensors;
    if (!spec_sidecar_artifact::load_manifest(manifest_path.c_str(), tensors, error)) {
        error = std::string(label) + " manifest invalid: " + error;
        return false;
    }
    uint64_t blob_size = 0;
    if (!get_file_size(blob_path, blob_size, error)) {
        return false;
    }
    if (!spec_sidecar_artifact::validate_blob_layout(tensors, blob_size, error)) {
        error = std::string(label) + " weights invalid: " + error;
        return false;
    }
    return true;
}

static bool validate_id_table(const std::string & path, int32_t rows,
        int32_t vocab, std::string & error) {
    uint64_t size = 0;
    if (!get_file_size(path, size, error)) {
        return false;
    }
    if (rows <= 0 || size != (uint64_t) rows * sizeof(int32_t)) {
        error = "sidecar ID table has an unexpected size: " + path;
        return false;
    }
    std::vector<int32_t> ids((size_t) rows);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(ids.data()), (std::streamsize) size)) {
        error = "sidecar ID table is truncated: " + path;
        return false;
    }
    if (!spec_sidecar_artifact::validate_remap(ids, vocab, error)) {
        error = "sidecar ID table is invalid: " + error;
        return false;
    }
    return true;
}

static bool validate_profile_artifacts_impl(const common_spec_sidecar_profile & profile,
        const common_spec_sidecar_paths & paths, std::string & error) {
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_MTP) {
        if (!validate_manifest_blob(join_path(paths.artifact_dir, "drafter_manifest.json"),
                join_path(paths.artifact_dir, "drafter_weights.bin"), "MTP sidecar", error)) {
            return false;
        }
        return validate_id_table(paths.ids, profile.mtp_head_rows,
                profile.target_n_vocab, error);
    }

    if (!validate_manifest_blob(join_path(paths.artifact_dir, "dflash_manifest.json"),
            join_path(paths.artifact_dir, "dflash_weights.bin"), "DFlash sidecar", error) ||
        !validate_manifest_blob(join_path(paths.artifact_dir, "drafter_manifest.json"),
            join_path(paths.artifact_dir, "drafter_weights.bin"), "DFlash target embedding", error)) {
        return false;
    }
    const char * head_name = paths.dflash_full_head ? "target_head.bin" : "target_head_sliced.bin";
    const std::string head_path = join_path(paths.artifact_dir, head_name);
    if (!require_file(head_path, "DFlash target head", error)) {
        return false;
    }
    uint64_t head_size = 0;
    const int64_t head_rows = paths.dflash_full_head ? profile.target_n_vocab : profile.dflash_head_rows;
    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, profile.target_n_embd);
    const uint64_t expected_head_size = head_rows > 0 && row_size > 0
            ? static_cast<uint64_t>(head_rows) * row_size : 0;
    if (!get_file_size(head_path, head_size, error) ||
            expected_head_size == 0 || head_size != expected_head_size) {
        if (error.empty()) {
            error = "DFlash target head has an unexpected size: " + head_path;
        }
        return false;
    }
    if (!paths.dflash_full_head && !validate_id_table(
            join_path(paths.artifact_dir, "draft_head_ids.bin"),
            profile.dflash_head_rows, profile.target_n_vocab, error)) {
        return false;
    }
    return true;
}

} // namespace

bool common_spec_sidecar_profile_name_matches(
        const common_spec_sidecar_profile & profile, const char * name) {
    return profile_name_matches(profile, name);
}

size_t common_spec_sidecar_profile_count() {
    return sizeof(ALL_PROFILES) / sizeof(ALL_PROFILES[0]);
}

const common_spec_sidecar_profile * common_spec_sidecar_profile_at(size_t index) {
    return index < common_spec_sidecar_profile_count() ? ALL_PROFILES[index] : nullptr;
}

const common_spec_sidecar_profile * common_spec_sidecar_profile_for_model(
        common_spec_sidecar_kind kind, const llama_model * model, std::string & error) {
    error.clear();
    for (const auto * profile : ALL_PROFILES) {
        if (profile->kind != kind || profile->matches_model == nullptr) {
            continue;
        }
        std::string mismatch;
        if (profile->matches_model(*profile, model, mismatch)) {
            if (!profile_paths_are_available(*profile)) {
                error = std::string(profile->name != nullptr ? profile->name : "sidecar") +
                        " requires explicit library and artifact paths";
                continue;
            }
            return profile;
        }
        if (error.empty()) {
            error = mismatch;
        }
    }
    if (error.empty()) {
        error = "no provider is registered for the requested sidecar kind";
    }
    return nullptr;
}

const common_spec_sidecar_profile * common_spec_sidecar_profile_for_target_file(
        common_spec_sidecar_kind kind, const std::string & path, std::string & error) {
    error.clear();
    for (const auto * profile : ALL_PROFILES) {
        if (profile->kind != kind || profile->matches_target_file == nullptr) {
            continue;
        }
        std::string mismatch;
        if (profile->matches_target_file(*profile, path, mismatch)) {
            if (!profile_paths_are_available(*profile)) {
                error = std::string(profile->name != nullptr ? profile->name : "sidecar") +
                        " requires explicit library and artifact paths";
                continue;
            }
            return profile;
        }
        if (error.empty()) {
            error = mismatch;
        }
    }
    if (error.empty()) {
        error = "no provider is registered for the requested sidecar kind";
    }
    return nullptr;
}

bool common_spec_sidecar_get_library(const common_spec_sidecar_profile & profile,
        std::string & library, std::string & error) {
    error.clear();
    const char * library_env = env_value(profile.library_env);
    library = library_env != nullptr ? library_env : find_default_library(profile);
    if (library.empty()) {
        error = std::string(profile.name != nullptr ? profile.name : "sidecar") +
                " could not discover its provider library; rebuild with speculative sidecars enabled or set " +
                (profile.library_env != nullptr ? profile.library_env : "the provider library path");
        return false;
    }
    if (!is_absolute_path(library)) {
        error = std::string(profile.name != nullptr ? profile.name : "sidecar") +
                " provider library path must be absolute: " + library;
        return false;
    }
    library = normalize_path(library);
    if (!is_regular_file(library)) {
        error = std::string(profile.name != nullptr ? profile.name : "sidecar") +
                " provider library is not readable: " + library;
        return false;
    }
    return true;
}

bool common_spec_sidecar_get_paths(const common_spec_sidecar_profile & profile,
        common_spec_sidecar_paths & paths, std::string & error) {
    paths = {};
    const char * artifact_env = env_value(profile.artifact_env);
    if (!common_spec_sidecar_get_library(profile, paths.library, error)) {
        return false;
    }
    paths.artifact_dir = artifact_env != nullptr ? artifact_env : find_default_artifact_directory(profile);
    if (paths.artifact_dir.empty()) {
        error = std::string(profile.name != nullptr ? profile.name : "sidecar") +
                " could not discover its default artifact bundle; set the provider path explicitly";
        return false;
    }
    paths.artifact_dir = normalize_path(paths.artifact_dir);
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_MTP) {
        const char * ids = env_value(profile.ids_env);
        paths.ids = ids != nullptr ? ids : join_path(paths.artifact_dir, "draft_head_ids.bin");
    }
    paths.dflash_full_head = profile.full_head_env != nullptr &&
            env_value(profile.full_head_env) != nullptr;
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_DFLASH && !paths.dflash_full_head &&
            !is_regular_file(join_path(paths.artifact_dir, "target_head_sliced.bin")) &&
            is_regular_file(join_path(paths.artifact_dir, "target_head.bin"))) {
        paths.dflash_full_head = true;
    }
    return true;
}

bool common_spec_sidecar_validate_artifacts(const common_spec_sidecar_profile & profile,
        const common_spec_sidecar_paths & paths, std::string & error) {
    return validate_profile_artifacts_impl(profile, paths, error);
}

bool common_spec_sidecar_probe(const common_spec_sidecar_profile & profile,
        const common_spec_sidecar_paths & paths, uint32_t n_seq, std::string & error) {
    if (!common_spec_sidecar_validate_artifacts(profile, paths, error)) {
        return false;
    }
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_MTP) {
        return common_spec_sidecar_mtp_probe(paths.library, paths.artifact_dir, paths.ids,
                profile.mtp_embedding_width, profile.mtp_head_rows, (int32_t) n_seq, error);
    }
    return common_spec_sidecar_dflash_probe(paths.library, paths.artifact_dir,
            profile.dflash_encoded_width, profile.dflash_block_size, (int32_t) n_seq, error);
}

bool common_spec_sidecar_probe(const common_spec_sidecar_profile & profile,
        uint32_t n_seq, std::string & error) {
    common_spec_sidecar_paths paths;
    if (!common_spec_sidecar_get_paths(profile, paths, error)) {
        return false;
    }
    return common_spec_sidecar_probe(profile, paths, n_seq, error);
}

using state_size_fn_t     = int (*)();
using state_get_fn_t      = int (*)(int32_t, void *, int);
using state_set_fn_t      = int (*)(int32_t, const void *, int);
using state_reset_fn_t    = int (*)(int32_t);
using state_truncate_fn_t = int (*)(int32_t, int32_t);
using mtp_state_commit_fn_t = int (*)(int32_t, int32_t, const float *);
using state_commit_fn_t      = int (*)(int32_t, int32_t);
using state_rebase_fn_t      = int (*)(int32_t, int32_t, int32_t, int32_t);
using attach_stream_fn_t     = int (*)(void *, int32_t);

using mtp_release_abi_fn = int (*)();
using mtp_check_fn      = int (*)(int32_t, int32_t, int32_t);
using mtp_init_context_fn = int (*)(const char *, const char *, int32_t, int32_t);
using mtp_init_device_context_fn = int (*)(const char *, const char *, int32_t, int32_t, int32_t);
using mtp_catchup_fn    = int (*)(int32_t, const int32_t *, const int32_t *, const float *, int);
using mtp_catchup_device_fn = int (*)(int32_t, const int32_t *, const int32_t *, const float *, int);
using mtp_draft_fn      = int (*)(int32_t, int32_t, int32_t, const float *, int, int32_t *);
using mtp_draft_device_fn = int (*)(int32_t, int32_t, int32_t, int, int32_t *);
using mtp_stochastic_top_k_fn = int (*)();
using mtp_draft_stochastic_fn = int (*)(int32_t, int32_t, int32_t, const float *, float, float, uint64_t, int, int32_t *, int32_t *, float *);
using mtp_draft_stochastic_device_fn = int (*)(int32_t, int32_t, int32_t, float, float, uint64_t, int, int32_t *, int32_t *, float *);

using dflash_release_abi_fn = int (*)();
using dflash_check_fn       = int (*)(int32_t, int32_t, int32_t);
using dflash_init_context_fn = int (*)(const char *, int32_t, int32_t);
using dflash_chunk_fn       = int (*)(int32_t, const int32_t *, const float *, int);
using dflash_chunk_device_fn = int (*)(int32_t, const int32_t *, const void * const *, int, int, int);
using dflash_draft_fn       = int (*)(int32_t, int32_t, int32_t, int32_t *);
using dflash_stochastic_top_k_fn = int (*)();
using dflash_draft_stochastic_fn = int (*)(int32_t, int32_t, int32_t, float, float, uint64_t, int, int32_t *, int32_t *, float *);

bool common_spec_sidecar_mtp_probe(const std::string & library_path,
        const std::string & weights_dir, const std::string & ids_path,
        int32_t embedding_width, int32_t head_rows, int32_t n_seq,
        std::string & error) {
    if (n_seq < 1 || n_seq > 8) {
        error = "MTP sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "MTP sidecar library path", error) ||
        !require_absolute(weights_dir, "MTP sidecar artifact directory", error) ||
        !require_absolute(ids_path, "MTP sidecar ID path", error) ||
        !require_file(join_path(weights_dir, "drafter_manifest.json"),
                "MTP sidecar manifest", error) ||
        !require_file(join_path(weights_dir, "drafter_weights.bin"),
                "MTP sidecar weights", error) ||
        !require_file(ids_path, "MTP sidecar ID table", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }
    mtp_release_abi_fn release = nullptr;
    mtp_check_fn check = nullptr;
    mtp_init_context_fn init_context = nullptr;
    mtp_init_device_context_fn init_device_context = nullptr;
    mtp_stochastic_top_k_fn top_k = nullptr;
    mtp_draft_stochastic_fn stochastic = nullptr;
    mtp_draft_stochastic_device_fn stochastic_device = nullptr;
    const bool symbols =
        resolve_symbol(handle, "spec_hip_release_abi", release, error) &&
        resolve_symbol(handle, "spec_hip_check", check, error) &&
        resolve_symbol(handle, "spec_hip_init_context", init_context, error) &&
        resolve_symbol(handle, "spec_hip_init_device_context", init_device_context, error) &&
        resolve_symbol(handle, "spec_hip_stochastic_top_k", top_k, error) &&
        resolve_symbol(handle, "spec_hip_draft_stochastic", stochastic, error) &&
        resolve_symbol(handle, "spec_hip_draft_stochastic_device", stochastic_device, error);
    const bool compatible = symbols && release() == SPEC_SIDECAR_MTP_RELEASE_ABI &&
            check(embedding_width, head_rows, n_seq) == 0 &&
            top_k() == SPEC_SIDECAR_MTP_DRAFT_TOP_K;
    if (!compatible && error.empty()) {
        error = "MTP sidecar stochastic ABI probe failed";
    }
    close_library(handle);
    return compatible;
}

bool common_spec_sidecar_dflash_probe(const std::string & library_path,
        const std::string & artifact_dir, int32_t encoded_width,
        int32_t block_size, int32_t n_seq, std::string & error) {
    if (n_seq < 1 || n_seq > 8) {
        error = "DFlash sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "DFlash sidecar library path", error) ||
        !require_absolute(artifact_dir, "DFlash sidecar artifact directory", error) ||
        !require_file(join_path(artifact_dir, "dflash_manifest.json"),
                "DFlash sidecar manifest", error) ||
        !require_file(join_path(artifact_dir, "dflash_weights.bin"),
                "DFlash sidecar weights", error) ||
        !require_file(join_path(artifact_dir,
                std::getenv("LLAMA_SPEC_HIP_FULL_HEAD") != nullptr
                    ? "target_head.bin" : "target_head_sliced.bin"),
                "DFlash target head", error) ||
        (std::getenv("LLAMA_SPEC_HIP_FULL_HEAD") == nullptr &&
         !require_file(join_path(artifact_dir, "draft_head_ids.bin"),
                "DFlash target-head ID table", error)) ||
        !require_file(join_path(artifact_dir, "drafter_manifest.json"),
                "DFlash target embedding manifest", error) ||
        !require_file(join_path(artifact_dir, "drafter_weights.bin"),
                "DFlash target embedding", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }
    dflash_release_abi_fn release = nullptr;
    dflash_check_fn check = nullptr;
    dflash_init_context_fn init_context = nullptr;
    dflash_stochastic_top_k_fn top_k = nullptr;
    dflash_draft_stochastic_fn stochastic = nullptr;
    const bool symbols =
        resolve_symbol(handle, "spec_dflash_release_abi", release, error) &&
        resolve_symbol(handle, "spec_dflash_check", check, error) &&
        resolve_symbol(handle, "spec_dflash_init_context", init_context, error) &&
        resolve_symbol(handle, "spec_dflash_stochastic_top_k", top_k, error) &&
        resolve_symbol(handle, "spec_dflash_draft_stochastic", stochastic, error);
    const bool compatible = symbols && release() == SPEC_SIDECAR_DFLASH_RELEASE_ABI &&
            check(encoded_width, block_size, n_seq) == 0 &&
            top_k() == SPEC_SIDECAR_DFLASH_DRAFT_TOP_K;
    if (!compatible && error.empty()) {
        error = "DFlash sidecar stochastic ABI probe failed";
    }
    close_library(handle);
    return compatible;
}

struct common_spec_sidecar_mtp::impl {
    void * handle = nullptr;
    bool active = false;
    state_size_fn_t state_size_fn = nullptr;
    state_get_fn_t state_get_fn = nullptr;
    state_set_fn_t state_set_fn = nullptr;
    state_reset_fn_t state_reset_fn = nullptr;
    state_truncate_fn_t state_truncate_fn = nullptr;
    mtp_state_commit_fn_t state_commit_fn = nullptr;
    state_rebase_fn_t state_rebase_fn = nullptr;
    attach_stream_fn_t attach_stream_fn = nullptr;
    mtp_catchup_fn catchup_fn = nullptr;
    mtp_catchup_device_fn catchup_device_fn = nullptr;
    mtp_draft_fn draft_fn = nullptr;
    mtp_draft_device_fn draft_device_fn = nullptr;
    mtp_stochastic_top_k_fn stochastic_top_k_fn = nullptr;
    mtp_draft_stochastic_fn draft_stochastic_fn = nullptr;
    mtp_draft_stochastic_device_fn draft_stochastic_device_fn = nullptr;
};

common_spec_sidecar_mtp::common_spec_sidecar_mtp() : pimpl(new impl) {}

common_spec_sidecar_mtp::~common_spec_sidecar_mtp() {
    // The release ABI currently has no shutdown function.  Keep a successfully
    // initialized library resident until process exit; unloading it would leave
    // its HIP allocations and static state without a supported teardown path.
}

bool common_spec_sidecar_mtp::load(const std::string & library_path,
        const std::string & weights_dir, const std::string & ids_path,
        int32_t embedding_width, int32_t head_rows, int32_t n_seq,
        int32_t max_context, std::string & error, int32_t device) {
    if (active()) {
        error = "MTP sidecar is already loaded";
        return false;
    }
    if (n_seq < 1 || n_seq > 8) {
        error = "MTP sidecar supports 1..8 sequences";
        return false;
    }
    if (max_context < 1) {
        error = "MTP sidecar target context must be positive";
        return false;
    }
    if (!require_absolute(library_path, "MTP sidecar library path", error) ||
        !require_absolute(weights_dir, "MTP sidecar artifact directory", error) ||
        !require_absolute(ids_path, "MTP sidecar ID path", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }

    mtp_release_abi_fn release_abi = nullptr;
    mtp_check_fn check = nullptr;
    mtp_init_context_fn init_context = nullptr;
    mtp_init_device_context_fn init_device_context = nullptr;
    if (!resolve_symbol(handle, "spec_hip_release_abi", release_abi, error) ||
        !resolve_symbol(handle, "spec_hip_check", check, error) ||
        !resolve_symbol(handle, "spec_hip_state_size", pimpl->state_size_fn, error) ||
        !resolve_symbol(handle, "spec_hip_get_state", pimpl->state_get_fn, error) ||
        !resolve_symbol(handle, "spec_hip_set_state", pimpl->state_set_fn, error) ||
        !resolve_symbol(handle, "spec_hip_reset_state", pimpl->state_reset_fn, error) ||
        !resolve_symbol(handle, "spec_hip_truncate_state", pimpl->state_truncate_fn, error) ||
        !resolve_symbol(handle, "spec_hip_commit_state", pimpl->state_commit_fn, error) ||
        !resolve_symbol(handle, "spec_hip_rebase_state", pimpl->state_rebase_fn, error) ||
        !resolve_symbol(handle, "spec_hip_attach_target_stream", pimpl->attach_stream_fn, error) ||
        !resolve_symbol(handle, "spec_hip_init_context", init_context, error) ||
        !resolve_symbol(handle, "spec_hip_init_device_context", init_device_context, error) ||
        !resolve_symbol(handle, "spec_hip_catchup", pimpl->catchup_fn, error) ||
        !resolve_symbol(handle, "spec_hip_catchup_device", pimpl->catchup_device_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft", pimpl->draft_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_device", pimpl->draft_device_fn, error) ||
        !resolve_symbol(handle, "spec_hip_stochastic_top_k", pimpl->stochastic_top_k_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_stochastic", pimpl->draft_stochastic_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_stochastic_device", pimpl->draft_stochastic_device_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != SPEC_SIDECAR_MTP_RELEASE_ABI) {
        error = "MTP sidecar ABI version mismatch (expected " +
                std::to_string(SPEC_SIDECAR_MTP_RELEASE_ABI) + ")";
        close_library(handle);
        return false;
    }
    if (check(embedding_width, head_rows, n_seq) != 0) {
        error = "MTP sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->stochastic_top_k_fn() != SPEC_SIDECAR_MTP_DRAFT_TOP_K) {
        error = "MTP sidecar stochastic top-k mismatch";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(spec_sidecar_state))) {
        error = "MTP sidecar state ABI size mismatch";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->catchup_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }
    const int init_rc = device >= 0
            ? init_device_context(weights_dir.c_str(), ids_path.c_str(), n_seq, device, max_context)
            : init_context(weights_dir.c_str(), ids_path.c_str(), n_seq, max_context);
    if (init_rc != 0) {
        error = "MTP sidecar initialization failed";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->catchup_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }

    pimpl->handle = handle;
    pimpl->active = true;
    return true;
}

bool common_spec_sidecar_mtp::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_spec_sidecar_mtp::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_spec_sidecar_mtp::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(spec_sidecar_state))) {
        data.clear();
        return false;
    }
    data.resize(static_cast<size_t>(size));
    if (pimpl->state_get_fn(seq_id, data.data(), size) != 0) {
        data.clear();
        return false;
    }
    return true;
}

bool common_spec_sidecar_mtp::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(spec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_spec_sidecar_mtp::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_spec_sidecar_mtp::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_mtp::commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max, hidden_device) == 0;
}

bool common_spec_sidecar_mtp::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_spec_sidecar_mtp::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_spec_sidecar_mtp::catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, int count) const {
    return active() && pimpl->catchup_fn != nullptr
        ? pimpl->catchup_fn(seq_id, tokens, positions, hidden_rows, count) : -1;
}

int common_spec_sidecar_mtp::catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows_device, int count) const {
    return active() && pimpl->catchup_device_fn != nullptr
        ? pimpl->catchup_device_fn(seq_id, tokens, positions, hidden_rows_device, count) : -1;
}

int common_spec_sidecar_mtp::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, hidden, max_draft, output_ids) : -1;
}

int common_spec_sidecar_mtp::draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_device_fn != nullptr
        ? pimpl->draft_device_fn(seq_id, last_token, past_tokens, max_draft, output_ids) : -1;
}

int common_spec_sidecar_mtp::draft_stochastic(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, const float * hidden, float temperature, float p_min,
        uint64_t rng_key, int max_draft, int32_t * output_ids,
        int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_fn != nullptr
        ? pimpl->draft_stochastic_fn(seq_id, last_token, past_tokens, hidden,
                temperature, p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}

int common_spec_sidecar_mtp::draft_stochastic_device(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, float temperature, float p_min, uint64_t rng_key,
        int max_draft, int32_t * output_ids, int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_device_fn != nullptr
        ? pimpl->draft_stochastic_device_fn(seq_id, last_token, past_tokens,
                temperature, p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}

struct common_spec_sidecar_dflash::impl {
    void * handle = nullptr;
    bool active = false;
    state_size_fn_t state_size_fn = nullptr;
    state_get_fn_t state_get_fn = nullptr;
    state_set_fn_t state_set_fn = nullptr;
    state_reset_fn_t state_reset_fn = nullptr;
    state_truncate_fn_t state_truncate_fn = nullptr;
    state_commit_fn_t state_commit_fn = nullptr;
    state_rebase_fn_t state_rebase_fn = nullptr;
    attach_stream_fn_t attach_stream_fn = nullptr;
    dflash_chunk_fn chunk_fn = nullptr;
    dflash_chunk_device_fn chunk_device_fn = nullptr;
    dflash_draft_fn draft_fn = nullptr;
    dflash_stochastic_top_k_fn stochastic_top_k_fn = nullptr;
    dflash_draft_stochastic_fn draft_stochastic_fn = nullptr;
};

common_spec_sidecar_dflash::common_spec_sidecar_dflash() : pimpl(new impl) {}

common_spec_sidecar_dflash::~common_spec_sidecar_dflash() {
    // See the MTP loader destructor: the current ABI intentionally remains
    // loaded until process exit because it cannot release HIP state.
}

bool common_spec_sidecar_dflash::load(const std::string & library_path,
        const std::string & artifact_dir, int32_t encoded_width, int32_t block_size,
        int32_t n_seq, int32_t max_context, std::string & error) {
    if (active()) {
        error = "DFlash sidecar is already loaded";
        return false;
    }
    if (n_seq < 1 || n_seq > 8) {
        error = "DFlash sidecar supports 1..8 sequences";
        return false;
    }
    if (max_context < 1) {
        error = "DFlash sidecar target context must be positive";
        return false;
    }
    if (!require_absolute(library_path, "DFlash sidecar library path", error) ||
        !require_absolute(artifact_dir, "DFlash sidecar artifact directory", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }

    dflash_release_abi_fn release_abi = nullptr;
    dflash_check_fn check = nullptr;
    dflash_init_context_fn init_context = nullptr;
    if (!resolve_symbol(handle, "spec_dflash_release_abi", release_abi, error) ||
        !resolve_symbol(handle, "spec_dflash_check", check, error) ||
        !resolve_symbol(handle, "spec_dflash_state_size", pimpl->state_size_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_get_state", pimpl->state_get_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_set_state", pimpl->state_set_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_reset_state", pimpl->state_reset_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_truncate_state", pimpl->state_truncate_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_commit_state", pimpl->state_commit_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_rebase_state", pimpl->state_rebase_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_attach_target_stream", pimpl->attach_stream_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_init_context", init_context, error) ||
        !resolve_symbol(handle, "spec_dflash_chunk", pimpl->chunk_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_chunk_device", pimpl->chunk_device_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_draft", pimpl->draft_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_stochastic_top_k", pimpl->stochastic_top_k_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_draft_stochastic", pimpl->draft_stochastic_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != SPEC_SIDECAR_DFLASH_RELEASE_ABI) {
        error = "DFlash sidecar ABI version mismatch (expected " +
                std::to_string(SPEC_SIDECAR_DFLASH_RELEASE_ABI) + ")";
        close_library(handle);
        return false;
    }
    if (check(encoded_width, block_size, n_seq) != 0) {
        error = "DFlash sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->stochastic_top_k_fn() != SPEC_SIDECAR_DFLASH_DRAFT_TOP_K) {
        error = "DFlash sidecar stochastic top-k mismatch";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(spec_sidecar_state))) {
        error = "DFlash sidecar state ABI size mismatch";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->chunk_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }
    if (init_context(artifact_dir.c_str(), n_seq, max_context) != 0) {
        error = "DFlash sidecar initialization failed";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->chunk_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }

    pimpl->handle = handle;
    pimpl->active = true;
    return true;
}

bool common_spec_sidecar_dflash::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_spec_sidecar_dflash::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_spec_sidecar_dflash::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(spec_sidecar_state))) {
        data.clear();
        return false;
    }
    data.resize(static_cast<size_t>(size));
    if (pimpl->state_get_fn(seq_id, data.data(), size) != 0) {
        data.clear();
        return false;
    }
    return true;
}

bool common_spec_sidecar_dflash::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(spec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_spec_sidecar_dflash::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_spec_sidecar_dflash::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_dflash::commit_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_dflash::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_spec_sidecar_dflash::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_spec_sidecar_dflash::chunk(int32_t seq_id, const int32_t * positions,
        const float * target_features, int count) const {
    return active() && pimpl->chunk_fn != nullptr
        ? pimpl->chunk_fn(seq_id, positions, target_features, count) : -1;
}

int common_spec_sidecar_dflash::chunk_device(int32_t seq_id, const int32_t * positions,
        const void * const * target_layer_features_device, int n_layers, int layer_width, int count) const {
    return active() && pimpl->chunk_device_fn != nullptr
        ? pimpl->chunk_device_fn(seq_id, positions, target_layer_features_device,
                                 n_layers, layer_width, count) : -1;
}

int common_spec_sidecar_dflash::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, output_ids) : -1;
}

int common_spec_sidecar_dflash::draft_stochastic(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, float temperature, float p_min, uint64_t rng_key,
        int max_draft, int32_t * output_ids, int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_fn != nullptr
        ? pimpl->draft_stochastic_fn(seq_id, last_token, past_tokens, temperature,
                p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}
