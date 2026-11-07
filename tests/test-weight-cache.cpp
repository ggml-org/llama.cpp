#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

static constexpr uint64_t CACHE_SOURCE_MTIME_SEC_OFFSET = 24;
static constexpr uint64_t CACHE_PAYLOAD_OFFSET_OFFSET = 64;
static constexpr uint64_t KLEIDIAI_FIRST_SLOT_SIZE_OFFSET = 24;

struct cache_stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_written = 0;
    uint64_t mmap_hits = 0;
};

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static cache_stats read_stats(const std::filesystem::path & path) {
    std::ifstream in(path);
    require((bool) in, "failed to open cache stats");

    cache_stats result;
    std::string line;
    while (std::getline(in, line)) {
        const auto entry = nlohmann::json::parse(line);
        result.hits += entry.at("cache_hits").get<uint64_t>();
        result.misses += entry.at("cache_misses").get<uint64_t>();
        result.bytes_written += entry.at("cache_bytes_written").get<uint64_t>();
        result.mmap_hits += entry.at("cache_mmap_hits").get<uint64_t>();
    }
    return result;
}

static void load_model(const std::string & model_path, const std::filesystem::path & cache_dir,
        const std::filesystem::path & stats_path, const char * mode,
        llama_load_mode load_mode = LLAMA_LOAD_MODE_MMAP) {
    std::filesystem::remove(stats_path);
    common_set_env("GGML_WEIGHT_CACHE", mode);
    common_set_env("GGML_WEIGHT_CACHE_DIR", cache_dir.string());
    common_set_env("GGML_WEIGHT_CACHE_STATS", stats_path.string());

    llama_model_params params = llama_model_default_params();
    params.load_mode = load_mode;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    require(model != nullptr, "failed to load model");
    llama_model_free(model);
}

static std::filesystem::path find_cache(const std::filesystem::path & cache_dir) {
    const std::string suffix = ".ggml-weight-cache-v2";
    std::filesystem::path result;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.size() >= suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            require(result.empty(), "found multiple cache files");
            result = entry.path();
        }
    }
    require(!result.empty(), "cache file was not created");
    return result;
}

static uint64_t read_u64(const std::filesystem::path & path, uint64_t offset) {
    std::ifstream in(path, std::ios::binary);
    require((bool) in, "failed to open cache file");
    in.seekg((std::streamoff) offset);
    uint64_t value;
    in.read((char *) &value, sizeof(value));
    require((bool) in, "failed to read cache file");
    return value;
}

static void write_u64(const std::filesystem::path & path, uint64_t offset, uint64_t value) {
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    require((bool) io, "failed to open cache file for update");
    io.seekp((std::streamoff) offset);
    io.write((const char *) &value, sizeof(value));
    require((bool) io, "failed to update cache file");
}

static std::string read_text(const std::filesystem::path & path) {
    std::ifstream in(path);
    require((bool) in, "failed to open sentinel file");
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static void mutate_file_preserve_mtime(const std::filesystem::path & path) {
    const auto mtime = std::filesystem::last_write_time(path);
    const uintmax_t size = std::filesystem::file_size(path);
    require(size > 0, "cannot mutate empty model file");

    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    require((bool) io, "failed to open model file for update");
    io.seekg((std::streamoff) size - 1);
    char value;
    io.read(&value, 1);
    require((bool) io, "failed to read model file");
    value ^= 1;
    io.seekp((std::streamoff) size - 1);
    io.write(&value, 1);
    io.close();
    require((bool) io, "failed to update model file");

    std::filesystem::last_write_time(path, mtime);
    require(std::filesystem::last_write_time(path) == mtime, "failed to restore model mtime");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::filesystem::path cache_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else {
            std::cerr << "usage: " << argv[0] << " -m model -c cache-dir\n";
            return 1;
        }
    }
    if (model_path.empty() || cache_dir.empty()) {
        std::cerr << "usage: " << argv[0] << " -m model -c cache-dir\n";
        return 1;
    }

    cache_dir /= "run-" + std::to_string(getpid()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    if (std::filesystem::exists(cache_dir)) {
        std::cerr << "cache directory already exists: " << cache_dir << '\n';
        return 1;
    }
    std::filesystem::create_directories(cache_dir);
    const std::filesystem::path test_model_path = cache_dir / "model.gguf";
    std::filesystem::copy_file(model_path, test_model_path);
    model_path = test_model_path.string();
    llama_backend_init();

    try {
        const auto populate_stats_path = cache_dir / "populate.jsonl";
        load_model(model_path, cache_dir, populate_stats_path, "1");
        const cache_stats populate = read_stats(populate_stats_path);
        require(populate.misses > 0, "populate did not report a cache miss");
        require(populate.bytes_written > 0, "populate did not write a cache");

        const std::filesystem::path cache_path = find_cache(cache_dir);
        const std::filesystem::path cache_backup = cache_dir / "cache.backup";
        std::filesystem::copy_file(cache_path, cache_backup, std::filesystem::copy_options::overwrite_existing);

        const auto hit_stats_path = cache_dir / "hit.jsonl";
        load_model(model_path, cache_dir, hit_stats_path, "readonly");
        const cache_stats hit = read_stats(hit_stats_path);
        require(hit.hits > 0 && hit.mmap_hits > 0 && hit.misses == 0, "valid cache was not reused");

        const auto no_mmap_stats_path = cache_dir / "no-mmap.jsonl";
        load_model(model_path, cache_dir, no_mmap_stats_path, "readonly", LLAMA_LOAD_MODE_NONE);
        const cache_stats no_mmap = read_stats(no_mmap_stats_path);
        require(no_mmap.hits == 0 && no_mmap.misses == 0 && no_mmap.mmap_hits == 0,
                "non-mmap load used the weight cache");

        const uint64_t payload_offset = read_u64(cache_path, CACHE_PAYLOAD_OFFSET_OFFSET);
        write_u64(cache_path, payload_offset + KLEIDIAI_FIRST_SLOT_SIZE_OFFSET, 0);
        const auto corrupt_stats_path = cache_dir / "corrupt.jsonl";
        load_model(model_path, cache_dir, corrupt_stats_path, "readonly");
        const cache_stats corrupt = read_stats(corrupt_stats_path);
        require(corrupt.misses > 0 && corrupt.mmap_hits == 0, "corrupt cache was not rejected");

        std::filesystem::copy_file(cache_backup, cache_path, std::filesystem::copy_options::overwrite_existing);
        write_u64(cache_path, CACHE_SOURCE_MTIME_SEC_OFFSET,
                read_u64(cache_path, CACHE_SOURCE_MTIME_SEC_OFFSET) + 1);
        const auto stale_stats_path = cache_dir / "stale.jsonl";
        load_model(model_path, cache_dir, stale_stats_path, "readonly");
        const cache_stats stale = read_stats(stale_stats_path);
        require(stale.misses > 0 && stale.mmap_hits == 0, "stale cache was not rejected");

        std::filesystem::copy_file(cache_backup, cache_path, std::filesystem::copy_options::overwrite_existing);
        mutate_file_preserve_mtime(model_path);
        const auto changed_model_stats_path = cache_dir / "changed-model.jsonl";
        load_model(model_path, cache_dir, changed_model_stats_path, "readonly");
        const cache_stats changed_model = read_stats(changed_model_stats_path);
        require(changed_model.misses > 0 && changed_model.mmap_hits == 0,
                "cache for different model contents was not rejected");

        const std::filesystem::path sentinel_path = cache_dir / "sentinel";
        const std::string sentinel = "weight cache sentinel";
        {
            std::ofstream out(sentinel_path);
            out << sentinel;
            require((bool) out, "failed to write sentinel file");
        }
        const std::filesystem::path predictable_tmp = cache_path.string() + ".tmp." + std::to_string(getpid()) + ".1";
        std::filesystem::create_symlink(sentinel_path, predictable_tmp);

        const auto rebuild_stats_path = cache_dir / "rebuild.jsonl";
        load_model(model_path, cache_dir, rebuild_stats_path, "rebuild");
        const cache_stats rebuild = read_stats(rebuild_stats_path);
        require(rebuild.bytes_written > 0, "rebuild did not replace the cache");
        require(read_text(sentinel_path) == sentinel, "cache rebuild followed a temporary-file symlink");

        const auto rebuilt_hit_stats_path = cache_dir / "rebuilt-hit.jsonl";
        load_model(model_path, cache_dir, rebuilt_hit_stats_path, "readonly");
        const cache_stats rebuilt_hit = read_stats(rebuilt_hit_stats_path);
        require(rebuilt_hit.hits > 0 && rebuilt_hit.mmap_hits > 0 && rebuilt_hit.misses == 0,
                "rebuilt cache was not reused");
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << '\n';
        common_set_env("GGML_WEIGHT_CACHE", "");
        common_set_env("GGML_WEIGHT_CACHE_DIR", "");
        common_set_env("GGML_WEIGHT_CACHE_STATS", "");
        llama_backend_free();
        std::filesystem::remove_all(cache_dir);
        return 1;
    }

    common_set_env("GGML_WEIGHT_CACHE", "");
    common_set_env("GGML_WEIGHT_CACHE_DIR", "");
    common_set_env("GGML_WEIGHT_CACHE_STATS", "");
    llama_backend_free();
    std::filesystem::remove_all(cache_dir);
    return 0;
}
