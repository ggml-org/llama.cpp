#include "router-scanner.h"

#include "log.h"
#include "router-config.h"

#include "common.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;

static std::string sanitize_repo_filename(const std::string & repo, const std::string & filename) {
    std::string name = repo + "_" + filename;
    std::replace(name.begin(), name.end(), '/', '_');
    return name;
}

static std::unordered_map<std::string, std::string> load_mmproj_map(const std::string & cache_dir) {
    std::unordered_map<std::string, std::string> mapping;

    std::error_code ec;
    for (std::filesystem::directory_iterator it(cache_dir, ec), end; it != end && !ec; ++it) {
        if (!it->is_regular_file()) {
            continue;
        }

        const std::string name = it->path().filename().string();
        constexpr std::string_view prefix = "manifest=";
        constexpr std::string_view suffix = ".json";
        if (name.rfind(prefix, 0) != 0 ||
            name.size() <= prefix.size() + suffix.size() ||
            name.substr(name.size() - suffix.size()) != suffix) {
            continue;
        }

        const std::string without_ext = it->path().stem().string();
        std::string       encoded     = without_ext.substr(prefix.size());

        std::vector<std::string> parts;
        size_t                   pos = 0;
        while (pos <= encoded.size()) {
            size_t next = encoded.find('=', pos);
            if (next == std::string::npos) {
                parts.push_back(encoded.substr(pos));
                break;
            }
            parts.push_back(encoded.substr(pos, next - pos));
            pos = next + 1;
        }

        if (parts.size() < 3) {
            continue;
        }

        const std::string repo = parts[0] + "/" + parts[1];

        json manifest;
        try {
            std::ifstream fin(it->path());
            if (!fin) {
                continue;
            }
            manifest = json::parse(fin);
        } catch (const std::exception &) {
            continue;
        }

        auto extract_rfilename = [](const json & obj, const char * key) -> std::string {
            if (obj.contains(key) && obj[key].contains("rfilename")) {
                return obj[key]["rfilename"].get<std::string>();
            }
            return {};
        };

        const std::string gguf_file   = extract_rfilename(manifest, "ggufFile");
        const std::string mmproj_file = extract_rfilename(manifest, "mmprojFile");

        if (gguf_file.empty() || mmproj_file.empty()) {
            continue;
        }

        const std::string gguf_path   = fs_get_cache_file(sanitize_repo_filename(repo, gguf_file));
        const std::string mmproj_path = fs_get_cache_file(sanitize_repo_filename(repo, mmproj_file));

        if (!std::filesystem::exists(gguf_path, ec) || !std::filesystem::exists(mmproj_path, ec)) {
            continue;
        }

        mapping.emplace(gguf_path, mmproj_path);
    }

    return mapping;
}

std::vector<ModelConfig> scan_default_models() {
    std::vector<ModelConfig> models;

    std::string cache_dir = fs_get_cache_directory();

    std::error_code ec;
    if (!std::filesystem::exists(cache_dir, ec) || ec) {
        return models;
    }

    const auto mmproj_map = load_mmproj_map(cache_dir);
    std::unordered_set<std::string> projector_paths;
    for (const auto & entry : mmproj_map) {
        projector_paths.insert(entry.second);
    }

    std::unordered_set<std::string> seen;

    for (std::filesystem::recursive_directory_iterator it(cache_dir, ec), end; it != end && !ec; ++it) {
        if (!it->is_regular_file()) {
            continue;
        }
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".gguf") {
            continue;
        }

        std::string full_path = it->path().string();
        if (seen.count(full_path)) {
            continue;
        }
        seen.insert(full_path);

        if (projector_paths.count(full_path)) {
            continue;
        }

        ModelConfig mc;
        mc.name  = it->path().filename().string();
        mc.path  = full_path;
        mc.state = "auto";
        if (auto it_mmproj = mmproj_map.find(full_path); it_mmproj != mmproj_map.end()) {
            mc.spawn = get_default_spawn();
            mc.spawn.push_back("--mmproj");
            mc.spawn.push_back(it_mmproj->second);
        }

        models.push_back(std::move(mc));
    }

    LOG_INF("Model scanner found %zu candidates in %s\n", models.size(), cache_dir.c_str());
    return models;
}
