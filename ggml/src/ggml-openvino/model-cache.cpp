#include "model-cache.h"

#include "device/npu_config.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "openvino/rt_info/weightless_caching_attributes.hpp"

#include <cerrno>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <openvino/core/model.hpp>
#include <openvino/core/version.hpp>
#include <openvino/op/constant.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#else
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
inline uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    return fnv1a(h, &v, sizeof(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(const ggml_tensor * src) {
    if (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        return false;
    }
    return src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    const size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    return fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
void for_each_weight(const ggml_cgraph * cgraph, F && fn) {
    std::unordered_set<const ggml_tensor *> seen;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (!is_weight_src(src) || !seen.insert(src).second) {
                continue;
            }
            fn(src);
        }
    }
}

std::string ov_version_string() {
    const ov::Version v = ov::get_openvino_version();
    return std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(const std::string & path) {
#if defined(_WIN32)
    int rc = _mkdir(path.c_str());
#else
    int rc = ::mkdir(path.c_str(), 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        acc.push_back(c);
        const bool sep = (c == '/'
#if defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        if (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
#if defined(_WIN32)
            // A drive root such as "D:" cannot be mkdir'ed (EACCES, not EEXIST).
            if (component.size() == 2 && component[1] == ':') {
                continue;
            }
#endif
            if (!make_dir(component)) {
                return false;
            }
        }
    }
    return make_dir(path);
}

}  // namespace

bool ggml_openvino_model_cache_only() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

static const char * cache_settings[] = {
    "GGML_OPENVINO_REQUANT_KQUANT",
    "GGML_OPENVINO_NATIVE_SOFTPLUS",
    // GGML_OPENVINO_KV_SLICE only changes dynamic runtime bindings. Dynamic
    // attention graphs accept either shape and share the same compiled blob.
    "GGML_OPENVINO_GDN_NATIVE_STATE",
    "GGML_OPENVINO_GDN_DECOMPOSE",
    "GGML_OPENVINO_GDN_DECOMPOSE_CHUNK",
    "GGML_OPENVINO_MANUAL_GQA_ATTN",
    "GGML_OPENVINO_EXACT_ZERO_POINT",
    "GGML_OPENVINO_STATEFUL_EXECUTION",
    "GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT",
    "GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS",
    "GGML_OPENVINO_REDUCE_COMPILE_MEM",
    "GGML_OPENVINO_MEMORY_OPTIMIZE",
    "GGML_OPENVINO_PROFILING",
    "GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK",
};

void ggml_openvino_model_cache_init() {
    const bool cache_only = ggml_openvino_model_cache_only();
    const std::string dir = ggml_openvino_model_cache_dir();
    if (dir.empty()) {
        if (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        return;
    }
    if (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#if !defined(__linux__) && !defined(_WIN32)
    if (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    if (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

const ggml_openvino_source_mapping * ggml_openvino_find_source_mapping(const void * data, size_t size,
                                                                     std::vector<ggml_openvino_source_mapping> & mappings) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](const ggml_openvino_source_mapping & m) {
        return address >= m.begin && address < m.end && size <= m.end - address;
    };
    for (const auto & m : mappings) {
        if (contains(m)) {
            return &m;
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long begin, end, offset, inode;
        unsigned int dev_major, dev_minor;
        char permissions[5];
        int path_start = 0;
        if (sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            continue;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET, permissions[1] != 'w'};
        if (!contains(m)) {
            continue;
        }
        struct stat st;
        const std::string path = line.substr(path_start);
        if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            break;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        return &mappings.back();
    }
#elif defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    if (VirtualQuery(data, &memory, sizeof(memory)) == sizeof(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        while (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<void *>(data), name.data(), static_cast<DWORD>(name.size()));
            if (length == 0 || length < name.size() - 1) {
                break;
            }
            name.resize(name.size() * 2);
        }
        if (length > 0 && length < name.size() - 1) {
            name.resize(length);
            const std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file != INVALID_HANDLE_VALUE) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                const bool valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic));
                CloseHandle(file);
                if (valid) {
                    const uint64_t file_size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
                    const uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    if (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        // A view created read-only can never be made writable.
                        const bool read_only = (memory.AllocationProtect & (PAGE_READONLY | PAGE_EXECUTE_READ)) != 0;
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5), read_only};
                        if (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dwVolumeSerialNumber);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
                                                                 info.ftLastWriteTime.dwLowDateTime);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            return &mappings.back();
                        }
                    }
                }
            }
        }
    }
#endif
    return nullptr;
}

uint64_t ggml_openvino_source_fingerprint(const void * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    if (const auto * m = ggml_openvino_find_source_mapping(data, size, mappings)) {
        return fnv1a_u64(m->identity, m->offset + reinterpret_cast<uintptr_t>(data) - m->begin);
    }
    if (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    const size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    if (size > head) {
        const size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<const uint8_t *>(data) + size - tail, tail);
    }
    return h;
}

std::string ggml_openvino_model_cache_dir() {
    const char * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    if (!dir || strlen(dir) == 0) {
        return std::string();
    }
    std::string path(dir);
    if (ggml_openvino_model_cache_only()) {
        return path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    if (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        return std::string();
    }
    return path;
}

std::string ggml_openvino_model_cache_temp_path(const std::string & path) {
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    return path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg,
                                         const std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    // Version 3 includes dense Q8_0/Q5_1 requantization in q4_asym64_all.
    // Version 4 uses GatedDeltaNet output 1 for snapshots in snapshot mode; older
    // graphs use the incompatible three-output contract.
    h = fnv1a_u64(h, 4);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    for (const char * name : cache_settings) {
        const char * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    h = fnv1a_u64(h, ggml_openvino_gdn_decompose_enabled() ? ggml_openvino_gdn_decompose_chunk() : 0);
    if (const char * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", sizeof("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    if (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", sizeof("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    bool has_gdn = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
        has_gdn = has_gdn || node->op == GGML_OP_GATED_DELTA_NET;
    }
    if (has_gdn && ggml_openvino_is_npu()) {
        // The GDN q pre-scale factor is a constant in the graph (0: no pre-scale).
        const float q_scale = ggml_openvino_gdn_prescale_enabled() ? ggml_openvino_gdn_q_scale() : 0.0f;
        h = fnv1a(h, &q_scale, sizeof(q_scale));
        // GDN graphs switch to the PLUGIN compiler unless the environment sets the compiler type.
        h = fnv1a_u64(h, ggml_openvino_npu_gdn_uses_plugin_compiler() ? 1u : 0u);
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    if (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, sizeof(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    return h;
}

std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & inputs,
                                              const std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<int>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    for (const auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        for (const auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    return f.good();
}

bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    if (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        return false;
    }
    // header: ov_version
    if (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        return false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<int>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string line;
    std::getline(f, line);  // consume rest of ov_version line
    while (idx < expected.size() && std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        if (idx >= expected.size() || line != expected[idx]) {
            return false;
        }
        ++idx;
    }
    if (idx != expected.size()) {
        return false;
    }
    if (!std::getline(f, line)) {
        return true;
    }
    if (line != "ports") {
        return false;
    }
    for (auto * names : { &inputs, &outputs }) {
        size_t count;
        if (!(f >> count) || count > 100000) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            std::string name;
            if (!(f >> std::quoted(name))) {
                return false;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    return f.eof();
}

std::string ggml_openvino_model_cache_weights_path(const std::string & dir, uint64_t fp_a, uint64_t fp_b) {
    return dir + "/" + hex64(fp_a) + "-" + hex64(fp_b) + ".weights.bin";
}

static uint64_t weights_hash_mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h * 0x100000001b3ULL;
}

static uint64_t weights_hash_bytes(uint64_t h, const void * data, size_t size) {
    const auto * p = static_cast<const unsigned char *>(data);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x100000001b3ULL;
    }
    for (; i < size; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ULL;
    }
    return h;
}

// Content hash recorded in a weights file's size record, or false if it has none.
static bool read_weights_hash(const std::string & path, uint64_t & hash) {
    std::ifstream meta(path + ".meta");
    std::string key, value;
    while (meta >> key >> value) {
        if (key == "hash") {
            hash = std::stoull(value, nullptr, 16);
            return true;
        }
    }
    return false;
}

std::shared_ptr<ggml_openvino_weights_layout> ggml_openvino_model_cache_write_weights(
    const std::string & path,
    const std::vector<std::shared_ptr<ov::Model>> & models,
    bool keep_existing,
    ggml_openvino_weights_status * status) {
    struct region {
        const void * data;
        size_t size;
        size_t offset;
    };
    constexpr size_t alignment = 64;
    auto layout = std::make_shared<ggml_openvino_weights_layout>();
    layout->path = path;
    std::vector<region> regions;
    std::unordered_map<const void *, size_t> region_of;
    for (const auto & model : models) {
        if (!model) {
            continue;
        }
        for (const auto & node : model->get_ordered_ops()) {
            auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!cnst || cnst->get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) == 0 ||
                cnst->get_data_ptr() == nullptr || cnst->get_byte_size() == 0) {
                continue;
            }
            layout->constants.push_back(cnst);
            auto [it, inserted] = region_of.emplace(cnst->get_data_ptr(), regions.size());
            if (inserted) {
                regions.push_back({ cnst->get_data_ptr(), cnst->get_byte_size(), 0 });
            } else {
                regions[it->second].size = std::max(regions[it->second].size, cnst->get_byte_size());
            }
        }
    }

    size_t total = 0;
    for (auto & r : regions) {
        r.offset = total;
        total = (total + r.size + alignment - 1) / alignment * alignment;
        layout->regions[r.data] = { r.offset, r.size };
    }
    for (const auto & cnst : layout->constants) {
        cnst->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = ov::WeightlessCacheAttribute(
            cnst->get_byte_size(), layout->regions.at(cnst->get_data_ptr()).first, cnst->get_element_type());
    }

    uint64_t content_hash = weights_hash_mix(0xcbf29ce484222325ULL, total);
    for (const auto & r : regions) {
        content_hash = weights_hash_mix(content_hash, r.offset);
        content_hash = weights_hash_mix(content_hash, r.size);
        content_hash = weights_hash_bytes(content_hash, r.data, r.size);
    }

    if (keep_existing && ggml_openvino_model_cache_weights_valid(path)) {
        uint64_t existing_hash = 0;
        if (read_weights_hash(path, existing_hash) && existing_hash == content_hash) {
            if (status) {
                *status = ggml_openvino_weights_status::reused;
            }
            GGML_LOG_INFO("ggml-openvino: model cache REUSED %s (%zu regions, %.1f MiB)\n", path.c_str(),
                          regions.size(), total / 1024.0 / 1024.0);
            return layout;
        }
        if (status) {
            *status = ggml_openvino_weights_status::mismatch;
        }
        return nullptr;
    }
    if (status) {
        *status = ggml_openvino_weights_status::written;
    }

    // Publish the data file first and its size record last, so a valid record always describes a complete file.
    const std::string data_tmp = path + ".tmp";
    const std::string meta_path = path + ".meta";
    const std::string meta_tmp = meta_path + ".tmp";
    {
        std::ofstream f(data_tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            return nullptr;
        }
        static const char zeros[alignment] = {};
        size_t pos = 0;
        for (const auto & r : regions) {
            f.write(zeros, static_cast<std::streamsize>(r.offset - pos));
            f.write(static_cast<const char *>(r.data), static_cast<std::streamsize>(r.size));
            pos = r.offset + r.size;
        }
        f.write(zeros, static_cast<std::streamsize>(total - pos));
        f.close();
        if (!f.good()) {
            std::remove(data_tmp.c_str());
            return nullptr;
        }
    }
    {
        std::ofstream f(meta_tmp, std::ios::trunc);
        f << "size " << total << "\nregions " << regions.size() << "\nhash " << hex64(content_hash) << "\n";
        f.close();
        if (!f.good()) {
            std::remove(data_tmp.c_str());
            std::remove(meta_tmp.c_str());
            return nullptr;
        }
    }
    // std::rename does not replace an existing file on Windows.
    ggml_openvino_model_cache_remove_weights(path);
    if (std::rename(data_tmp.c_str(), path.c_str()) != 0 || std::rename(meta_tmp.c_str(), meta_path.c_str()) != 0) {
        std::remove(data_tmp.c_str());
        std::remove(meta_tmp.c_str());
        return nullptr;
    }
    GGML_LOG_INFO("ggml-openvino: model cache WROTE %s (%zu regions, %.1f MiB)\n", path.c_str(), regions.size(),
                  total / 1024.0 / 1024.0);
    return layout;
}

size_t ggml_openvino_model_cache_retag_weights(const std::shared_ptr<ov::Model> & model,
                                               const ggml_openvino_weights_layout & layout) {
    size_t resolved = 0;
    for (const auto & node : model->get_ordered_ops()) {
        auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!cnst) {
            continue;
        }
        auto & rt_info = cnst->get_rt_info();
        auto attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (attr == rt_info.end()) {
            continue;
        }
        auto it = layout.regions.find(cnst->get_data_ptr());
        if (it != layout.regions.end() && cnst->get_byte_size() <= it->second.second) {
            attr->second = ov::WeightlessCacheAttribute(cnst->get_byte_size(), it->second.first, cnst->get_element_type());
            ++resolved;
        } else {
            rt_info.erase(attr);
        }
    }
    return resolved;
}

bool ggml_openvino_model_cache_weights_valid(const std::string & path) {
    std::ifstream meta(path + ".meta");
    std::string key;
    uint64_t size = 0;
    if (!(meta >> key >> size) || key != "size") {
        return false;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.is_open() && static_cast<uint64_t>(f.tellg()) == size;
}

void ggml_openvino_model_cache_remove_weights(const std::string & path) {
    std::remove((path + ".meta").c_str());
    std::remove(path.c_str());
}
