// SPDX-License-Identifier: MIT
//

#include "llama-weight-cache.h"

#if defined(LLAMA_WEIGHT_CACHE)

#include "llama-model-loader.h"

#include "ggml.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sys/stat.h>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

static constexpr uint32_t LLAMA_WEIGHT_CACHE_MAGIC      = 0x31435747; // "GWC1"
static constexpr uint32_t LLAMA_WEIGHT_CACHE_VERSION    = 2;
static constexpr uint32_t LLAMA_WEIGHT_CACHE_ENDIANNESS = 0x01020304;

struct llama_weight_cache_header {
    uint32_t magic;
    uint32_t version;
    uint32_t endianness;
    uint32_t tensor_count;
    uint64_t source_size;
    int64_t  source_mtime_sec;
    int64_t  source_mtime_nsec;
    uint64_t source_hash;
    uint64_t layout_hash_a;
    uint64_t layout_hash_b;
    uint64_t payload_offset;
    uint64_t payload_size;
};

struct llama_weight_cache_source {
    std::string path;
    size_t size = 0;
    int64_t mtime_sec = 0;
    int64_t mtime_nsec = 0;
    uint64_t hash = 0;
    bool hash_valid = false;
};

struct llama_weight_cache_context {
    struct entry {
        ggml_tensor * tensor;
        const llama_model_loader::llama_tensor_weight * weight;
        ggml_backend_weight_cache_info info;
        size_t packed_offset;
    };

    std::vector<entry> tensors;
    const llama_weight_cache_source * source = nullptr;
    size_t payload_size = 0;
    uint64_t source_hash = 0;
    uint64_t layout_hash_a = 1469598103934665603ULL;
    uint64_t layout_hash_b = 1099511628211ULL;
};

struct llama_weight_cache::impl {
    enum mode {
        MODE_DISABLED,
        MODE_ENABLED,
        MODE_READONLY,
        MODE_REBUILD,
    };

    struct stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t bytes_written = 0;
        uint64_t mmap_hits = 0;
        uint64_t mmap_bytes = 0;
    };

    mode mode_value = MODE_DISABLED;
    std::string stats_path;
    stats counters;
    std::vector<llama_weight_cache_source> sources;
    llama_files files;
    llama_mmaps mappings;
    std::set<const ggml_tensor *> mapped_tensors;
};

static void * llama_weight_cache_get_cpu_proc(const char * name) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), name);
}

static const ggml_backend_weight_cache_i * llama_weight_cache_get_interface() {
    static auto get_interface = (ggml_backend_weight_cache_get_interface_t)
        llama_weight_cache_get_cpu_proc("ggml_backend_weight_cache_get_interface");
    static const ggml_backend_weight_cache_i * iface = get_interface ? get_interface() : nullptr;
    return iface;
}

static bool llama_weight_cache_supports_buft(ggml_backend_buffer_type_t buft) {
    const auto * iface = llama_weight_cache_get_interface();
    return iface && iface->supports_buft && iface->supports_buft(buft);
}

static bool llama_weight_cache_info_for_tensor(const ggml_tensor * tensor, ggml_backend_weight_cache_info & info) {
    const auto * iface = llama_weight_cache_get_interface();
    return iface && iface->get_info && iface->get_info(tensor, &info) == 0;
}

static bool llama_weight_cache_validate_tensor(const ggml_tensor * tensor, const void * data, size_t size) {
    const auto * iface = llama_weight_cache_get_interface();
    return iface && iface->validate_data && iface->validate_data(tensor, data, size) == 0;
}

static uint32_t llama_weight_cache_crc32(const void * data, size_t size) {
    static const std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> result;
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            result[i] = c;
        }
        return result;
    }();

    uint32_t c = 0xffffffffu;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

template <typename T>
static void llama_weight_cache_read(llama_file & file, T & value) {
    file.read_raw(&value, sizeof(value));
}

template <typename T>
static bool llama_weight_cache_write(FILE * out, const T & value) {
    return fwrite(&value, 1, sizeof(value), out) == sizeof(value);
}

static int64_t llama_weight_cache_mtime_nsec(const struct stat & st) {
#if defined(__APPLE__)
    return st.st_mtimespec.tv_nsec;
#elif defined(_WIN32)
    GGML_UNUSED(st);
    return 0;
#else
    return st.st_mtim.tv_nsec;
#endif
}

static std::string llama_weight_cache_basename(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string llama_weight_cache_path(const std::string & source_path) {
    const char * cache_dir = getenv("GGML_WEIGHT_CACHE_DIR");
    if (!cache_dir || cache_dir[0] == '\0') {
        return source_path + ".ggml-weight-cache-v2";
    }

    std::error_code ec;
    const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(
            std::filesystem::u8path(source_path), ec);
    const std::string identity_path = ec ? source_path : canonical_path.u8string();
    char hash[16];
    snprintf(hash, sizeof(hash), "%08x", llama_weight_cache_crc32(identity_path.data(), identity_path.size()));
    return (std::filesystem::path(cache_dir) /
            (llama_weight_cache_basename(identity_path) + "." + hash + ".ggml-weight-cache-v2")).string();
}

#if defined(_WIN32)
static std::string llama_weight_cache_tmp_path(const std::string & path) {
    static std::atomic<uint64_t> sequence = 0;
    const int process_id = _getpid();
    return path + ".tmp." + std::to_string(process_id) + "." +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}
#endif

static bool llama_weight_cache_replace_file(const std::string & source, const std::string & destination) {
#if defined(_WIN32)
    const std::filesystem::path source_path = std::filesystem::u8path(source);
    const std::filesystem::path destination_path = std::filesystem::u8path(destination);
    return MoveFileExW(source_path.c_str(), destination_path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(source.c_str(), destination.c_str()) == 0;
#endif
}

static size_t llama_weight_cache_pad(size_t value, size_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

static bool llama_weight_cache_read_header(llama_file & file, llama_weight_cache_header & header) {
    llama_weight_cache_read(file, header.magic);
    llama_weight_cache_read(file, header.version);
    llama_weight_cache_read(file, header.endianness);
    llama_weight_cache_read(file, header.tensor_count);
    llama_weight_cache_read(file, header.source_size);
    llama_weight_cache_read(file, header.source_mtime_sec);
    llama_weight_cache_read(file, header.source_mtime_nsec);
    llama_weight_cache_read(file, header.source_hash);
    llama_weight_cache_read(file, header.layout_hash_a);
    llama_weight_cache_read(file, header.layout_hash_b);
    llama_weight_cache_read(file, header.payload_offset);
    llama_weight_cache_read(file, header.payload_size);
    return header.magic == LLAMA_WEIGHT_CACHE_MAGIC &&
           header.version == LLAMA_WEIGHT_CACHE_VERSION &&
           header.endianness == LLAMA_WEIGHT_CACHE_ENDIANNESS;
}

static bool llama_weight_cache_write_header(FILE * out, const llama_weight_cache_header & header) {
    return llama_weight_cache_write(out, header.magic) &&
           llama_weight_cache_write(out, header.version) &&
           llama_weight_cache_write(out, header.endianness) &&
           llama_weight_cache_write(out, header.tensor_count) &&
           llama_weight_cache_write(out, header.source_size) &&
           llama_weight_cache_write(out, header.source_mtime_sec) &&
           llama_weight_cache_write(out, header.source_mtime_nsec) &&
           llama_weight_cache_write(out, header.source_hash) &&
           llama_weight_cache_write(out, header.layout_hash_a) &&
           llama_weight_cache_write(out, header.layout_hash_b) &&
           llama_weight_cache_write(out, header.payload_offset) &&
           llama_weight_cache_write(out, header.payload_size);
}

static uint64_t llama_weight_cache_hash(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i])*1099511628211ULL;
    }
    return hash;
}

template <typename T>
static void llama_weight_cache_hash_value(llama_weight_cache_context & cache_ctx, const T & value) {
    cache_ctx.layout_hash_a = llama_weight_cache_hash(cache_ctx.layout_hash_a, &value, sizeof(value));
    cache_ctx.layout_hash_b = llama_weight_cache_hash(cache_ctx.layout_hash_b, &value, sizeof(value));
}

static bool llama_weight_cache_collect(
        llama_weight_cache::impl & state,
        const llama_model_loader & loader,
        ggml_context * ctx,
        ggml_backend_buffer_type_t buft,
        llama_weight_cache_context & cache_ctx) {
    if (state.mode_value == llama_weight_cache::impl::MODE_DISABLED || !buft || !llama_weight_cache_supports_buft(buft)) {
        return false;
    }

    cache_ctx = {};
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    uint16_t source_idx = UINT16_MAX;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = loader.get_weight(ggml_get_name(tensor));
        if (!weight || weight->idx >= state.sources.size()) {
            return false;
        }
        if (source_idx == UINT16_MAX) {
            source_idx = weight->idx;
            cache_ctx.source = &state.sources[source_idx];
        } else if (source_idx != weight->idx) {
            return false;
        }
        ggml_backend_weight_cache_info info = {};
        if (!llama_weight_cache_info_for_tensor(tensor, info)) {
            return false;
        }
        const size_t packed_offset = cache_ctx.payload_size;
        cache_ctx.tensors.push_back({tensor, weight, info, packed_offset});

        const char * name = ggml_get_name(tensor);
        cache_ctx.layout_hash_a = llama_weight_cache_hash(cache_ctx.layout_hash_a, name, strlen(name));
        cache_ctx.layout_hash_b = llama_weight_cache_hash(cache_ctx.layout_hash_b, name, strlen(name));
        llama_weight_cache_hash_value(cache_ctx, tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            llama_weight_cache_hash_value(cache_ctx, tensor->ne[i]);
        }
        llama_weight_cache_hash_value(cache_ctx, weight->offs);
        const size_t original_size = ggml_nbytes(tensor);
        llama_weight_cache_hash_value(cache_ctx, original_size);
        llama_weight_cache_hash_value(cache_ctx, packed_offset);
        llama_weight_cache_hash_value(cache_ctx, info.pack_version);
        llama_weight_cache_hash_value(cache_ctx, info.cpu_features);
        llama_weight_cache_hash_value(cache_ctx, info.kernel_signature);
        llama_weight_cache_hash_value(cache_ctx, info.slot_count);
        llama_weight_cache_hash_value(cache_ctx, info.packed_size);

        cache_ctx.payload_size += llama_weight_cache_pad(info.packed_size, alignment);
    }
    if (cache_ctx.tensors.empty() || !cache_ctx.source || cache_ctx.source->path.empty() ||
            source_idx >= loader.mappings.size() || !loader.mappings[source_idx] ||
            loader.mappings[source_idx]->size() != cache_ctx.source->size) {
        return false;
    }
    llama_weight_cache_source & source = state.sources[source_idx];
    if (!source.hash_valid) {
        source.hash = XXH3_64bits(loader.mappings[source_idx]->addr(), loader.mappings[source_idx]->size());
        source.hash_valid = true;
    }
    cache_ctx.source_hash = source.hash;
    return true;
}

llama_weight_cache::llama_weight_cache(bool check_tensors, bool from_file_ptr) : pimpl(new impl()) {
    const char * stats = getenv("GGML_WEIGHT_CACHE_STATS");
    if (stats) {
        pimpl->stats_path = stats;
    }
    const char * env = getenv("GGML_WEIGHT_CACHE");
    if ((env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0 || strcmp(env, "off") == 0)) ||
            check_tensors || from_file_ptr) {
        return;
    }
    if (env && strcmp(env, "readonly") == 0) {
        pimpl->mode_value = impl::MODE_READONLY;
    } else if (env && strcmp(env, "rebuild") == 0) {
        pimpl->mode_value = impl::MODE_REBUILD;
    } else {
        pimpl->mode_value = impl::MODE_ENABLED;
    }
}

llama_weight_cache::~llama_weight_cache() {
    if (pimpl->stats_path.empty()) {
        return;
    }
    std::ofstream out(pimpl->stats_path, std::ios::app);
    if (out) {
        out << "{\"cache_hits\":" << pimpl->counters.hits
            << ",\"cache_misses\":" << pimpl->counters.misses
            << ",\"cache_bytes_read\":0"
            << ",\"cache_bytes_written\":" << pimpl->counters.bytes_written
            << ",\"cache_mmap_hits\":" << pimpl->counters.mmap_hits
            << ",\"cache_mmap_bytes\":" << pimpl->counters.mmap_bytes << "}\n";
    }
}

void llama_weight_cache::add_source(uint16_t idx, const std::string & path, const llama_file * file) {
    if (idx > 0 && pimpl->mode_value != impl::MODE_DISABLED) {
        static std::atomic_flag warning_emitted = ATOMIC_FLAG_INIT;
        if (!warning_emitted.test_and_set(std::memory_order_relaxed)) {
            LLAMA_LOG_INFO("backend weight cache: split GGUF models are not supported, cache disabled\n");
        }
        pimpl->mode_value = impl::MODE_DISABLED;
    }
    if (pimpl->sources.size() <= idx) {
        pimpl->sources.resize((size_t) idx + 1);
    }
    auto & source = pimpl->sources[idx];
    source.path = path;
    source.size = file->size();
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        source.size = (size_t) st.st_size;
        source.mtime_sec = (int64_t) st.st_mtime;
        source.mtime_nsec = llama_weight_cache_mtime_nsec(st);
    }
}

bool llama_weight_cache::contains(const ggml_tensor * tensor) const {
    return pimpl->mapped_tensors.count(tensor) != 0;
}

ggml_backend_buffer_t llama_weight_cache::load(
        const llama_model_loader & loader,
        ggml_context * ctx,
        ggml_backend_buffer_type_t buft,
        bool use_mlock,
        llama_mlocks * mlocks) {
    if (pimpl->mode_value == impl::MODE_DISABLED || pimpl->mode_value == impl::MODE_REBUILD || !llama_mmap::SUPPORTED) {
        return nullptr;
    }

    llama_weight_cache_context cache_ctx;
    if (!llama_weight_cache_collect(*pimpl, loader, ctx, buft, cache_ctx)) {
        return nullptr;
    }
    auto miss = [&]() {
        pimpl->counters.misses += cache_ctx.tensors.size();
        return (ggml_backend_buffer_t) nullptr;
    };

    const std::string path = llama_weight_cache_path(cache_ctx.source->path);
    llama_weight_cache_header header = {};
    const size_t expected_payload_offset = llama_weight_cache_pad(sizeof(llama_weight_cache_header),
            ggml_backend_buft_get_alignment(buft));

    bool header_validated = false;
    try {
        pimpl->files.emplace_back(new llama_file(path.c_str(), "rb", false));
        llama_file & file = *pimpl->files.back();
        if (!llama_weight_cache_read_header(file, header) ||
                header.source_size != cache_ctx.source->size ||
                header.source_mtime_sec != cache_ctx.source->mtime_sec ||
                header.source_mtime_nsec != cache_ctx.source->mtime_nsec ||
                header.source_hash != cache_ctx.source_hash ||
                header.tensor_count != cache_ctx.tensors.size() ||
                header.payload_size != cache_ctx.payload_size ||
                header.layout_hash_a != cache_ctx.layout_hash_a ||
                header.layout_hash_b != cache_ctx.layout_hash_b ||
                header.payload_offset != expected_payload_offset ||
                header.payload_offset > file.size() ||
                header.payload_size > file.size() - header.payload_offset) {
            pimpl->files.pop_back();
            return miss();
        }
        header_validated = true;
        pimpl->mappings.emplace_back(new llama_mmap(&file, 0, false));
    } catch (const std::exception & ex) {
        if (pimpl->files.size() > pimpl->mappings.size()) {
            pimpl->files.pop_back();
        }
        if (header_validated) {
            LLAMA_LOG_WARN("backend weight cache: failed to mmap '%s': %s\n", path.c_str(), ex.what());
        }
        return miss();
    }

    uint8_t * payload = (uint8_t *) pimpl->mappings.back()->addr() + header.payload_offset;
    for (const auto & entry : cache_ctx.tensors) {
        if (!llama_weight_cache_validate_tensor(entry.tensor,
                    payload + entry.packed_offset, entry.info.packed_size)) {
            pimpl->mappings.pop_back();
            pimpl->files.pop_back();
            return miss();
        }
    }

    const auto * iface = llama_weight_cache_get_interface();
    ggml_backend_buffer_t buffer = iface && iface->buffer_from_ptr ?
        iface->buffer_from_ptr(buft, payload, header.payload_size) : nullptr;
    if (!buffer) {
        pimpl->mappings.pop_back();
        pimpl->files.pop_back();
        return miss();
    }

    for (const auto & entry : cache_ctx.tensors) {
        ggml_tensor * tensor = entry.tensor;
        if (ggml_backend_tensor_alloc(buffer, tensor, payload + entry.packed_offset) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            pimpl->mappings.pop_back();
            pimpl->files.pop_back();
            return miss();
        }
        pimpl->mapped_tensors.insert(tensor);
    }

    if (use_mlock && mlocks) {
        std::unique_ptr<llama_mlock> lock(new llama_mlock());
        lock->init(payload);
        lock->grow_to(header.payload_size);
        mlocks->push_back(std::move(lock));
    }
    pimpl->counters.hits += cache_ctx.tensors.size();
    pimpl->counters.mmap_hits += cache_ctx.tensors.size();
    pimpl->counters.mmap_bytes += header.payload_size;
    return buffer;
}

void llama_weight_cache::save(
        const llama_model_loader & loader,
        ggml_context * ctx,
        ggml_backend_buffer_type_t buft) {
    if (pimpl->mode_value != impl::MODE_ENABLED && pimpl->mode_value != impl::MODE_REBUILD) {
        return;
    }
    llama_weight_cache_context cache_ctx;
    if (!llama_weight_cache_collect(*pimpl, loader, ctx, buft, cache_ctx) || contains(cache_ctx.tensors.front().tensor)) {
        return;
    }

    ggml_backend_buffer_t buffer = cache_ctx.tensors.front().tensor->buffer;
    uint8_t * base = buffer ? (uint8_t *) ggml_backend_buffer_get_base(buffer) : nullptr;
    const size_t buffer_size = buffer ? ggml_backend_buffer_get_size(buffer) : 0;
    if (!base || buffer_size != cache_ctx.payload_size) {
        return;
    }

    for (const auto & entry : cache_ctx.tensors) {
        if (entry.tensor->buffer != buffer || !entry.tensor->data) {
            return;
        }
        const size_t packed_offset = (uint8_t *) entry.tensor->data - base;
        if (packed_offset != entry.packed_offset ||
                packed_offset > buffer_size ||
                entry.info.packed_size > buffer_size - packed_offset) {
            return;
        }
    }

    llama_weight_cache_header header = {};
    header.magic = LLAMA_WEIGHT_CACHE_MAGIC;
    header.version = LLAMA_WEIGHT_CACHE_VERSION;
    header.endianness = LLAMA_WEIGHT_CACHE_ENDIANNESS;
    header.tensor_count = (uint32_t) cache_ctx.tensors.size();
    header.source_size = cache_ctx.source->size;
    header.source_mtime_sec = cache_ctx.source->mtime_sec;
    header.source_mtime_nsec = cache_ctx.source->mtime_nsec;
    header.source_hash = cache_ctx.source_hash;
    header.layout_hash_a = cache_ctx.layout_hash_a;
    header.layout_hash_b = cache_ctx.layout_hash_b;
    header.payload_offset = llama_weight_cache_pad(sizeof(llama_weight_cache_header),
            ggml_backend_buft_get_alignment(buft));
    header.payload_size = cache_ctx.payload_size;

    const std::string path = llama_weight_cache_path(cache_ctx.source->path);
    std::error_code ec;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty() && !std::filesystem::create_directories(parent, ec) && ec) {
        LLAMA_LOG_WARN("backend weight cache: failed to create directory for '%s'\n", path.c_str());
        return;
    }
#if defined(_WIN32)
    const std::string tmp_path = llama_weight_cache_tmp_path(path);
    const std::filesystem::path tmp_path_w = std::filesystem::u8path(tmp_path);
    HANDLE handle = CreateFileW(tmp_path_w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        LLAMA_LOG_WARN("backend weight cache: failed to create temporary cache for '%s'\n", path.c_str());
        return;
    }
    auto remove_tmp = [&]() {
        std::error_code remove_ec;
        std::filesystem::remove(tmp_path, remove_ec);
    };

    const int fd = _open_osfhandle((intptr_t) handle, _O_WRONLY | _O_BINARY);
    if (fd == -1) {
        CloseHandle(handle);
        remove_tmp();
        LLAMA_LOG_WARN("backend weight cache: failed to open '%s' for writing\n", tmp_path.c_str());
        return;
    }
    FILE * out = _fdopen(fd, "wb");
    if (!out) {
        _close(fd);
        remove_tmp();
        LLAMA_LOG_WARN("backend weight cache: failed to open '%s' for writing\n", tmp_path.c_str());
        return;
    }

    bool ok = llama_weight_cache_write_header(out, header);
    const __int64 pos = _ftelli64(out);
    if (pos < 0 || (uint64_t) pos > header.payload_offset) {
        ok = false;
    }
    if (ok) {
        std::vector<char> padding((size_t) (header.payload_offset - (uint64_t) pos), 0);
        ok = fwrite(padding.data(), 1, padding.size(), out) == padding.size() &&
             fwrite(base, 1, buffer_size, out) == buffer_size;
    }
    if (fclose(out) != 0) {
        ok = false;
    }
    if (!ok) {
        LLAMA_LOG_WARN("backend weight cache: failed to write '%s'\n", tmp_path.c_str());
        remove_tmp();
        return;
    }
#else
    const std::string pattern = path + ".tmp.XXXXXX";
    std::vector<char> tmp_name(pattern.begin(), pattern.end());
    tmp_name.push_back('\0');

    const int fd = mkstemp(tmp_name.data());
    if (fd == -1) {
        LLAMA_LOG_WARN("backend weight cache: failed to create temporary cache for '%s'\n", path.c_str());
        return;
    }

    const std::string tmp_path(tmp_name.data());
    auto remove_tmp = [&]() {
        unlink(tmp_path.c_str());
    };
    FILE * out = fdopen(fd, "wb");
    if (!out) {
        close(fd);
        remove_tmp();
        LLAMA_LOG_WARN("backend weight cache: failed to open '%s' for writing\n", tmp_path.c_str());
        return;
    }

    bool ok = llama_weight_cache_write_header(out, header);
    const off_t pos = ftello(out);
    if (pos < 0 || (uint64_t) pos > header.payload_offset) {
        ok = false;
    }
    if (ok) {
        std::vector<char> padding((size_t) (header.payload_offset - (uint64_t) pos), 0);
        ok = fwrite(padding.data(), 1, padding.size(), out) == padding.size() &&
             fwrite(base, 1, buffer_size, out) == buffer_size;
    }
    if (fclose(out) != 0) {
        ok = false;
    }
    if (!ok) {
        LLAMA_LOG_WARN("backend weight cache: failed to write '%s'\n", tmp_path.c_str());
        remove_tmp();
        return;
    }
#endif
    if (!llama_weight_cache_replace_file(tmp_path, path)) {
        LLAMA_LOG_WARN("backend weight cache: failed to rename '%s' to '%s'\n", tmp_path.c_str(), path.c_str());
        remove_tmp();
        return;
    }
    pimpl->counters.bytes_written += cache_ctx.payload_size;
}

#endif
