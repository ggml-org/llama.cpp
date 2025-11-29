#include "router-scanner.h"

#include "router-config.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unordered_set>

std::vector<ModelConfig> scan_default_models() {
    std::vector<ModelConfig> models;

    std::string cache_dir = expand_user_path("~/.cache/llama.cpp");

    std::error_code ec;
    if (!std::filesystem::exists(cache_dir, ec) || ec) {
        return models;
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

        std::string filename = it->path().filename().string();
        if (seen.count(filename)) {
            continue;
        }
        seen.insert(filename);

        ModelConfig mc;
        mc.name  = filename;
        mc.path  = it->path().string();
        mc.state = "auto";
        models.push_back(std::move(mc));
    }

    return models;
}
