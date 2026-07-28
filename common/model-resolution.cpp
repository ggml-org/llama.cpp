#include "model-resolution.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>

namespace model_resolution {

gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
    std::smatch m;

    std::string prefix = path;
    if (!string_remove_suffix(prefix, ".gguf")) {
        return {};
    }

    int index = 1;
    int count = 1;

    if (std::regex_match(prefix, m, re_split)) {
        index = std::stoi(m[2].str());
        count = std::stoi(m[3].str());
        prefix = m[1].str();
    }

    std::string tag;
    if (std::regex_search(prefix, m, re_tag)) {
        tag = m[1].str();
        for (char & c : tag) {
            c = std::toupper((unsigned char)c);
        }
    }

    return {std::move(prefix), std::move(tag), index, count};
}

int extract_quant_bits(const std::string & filename) {
    auto split = get_gguf_split_info(filename);

    auto pos = split.tag.find_first_of("0123456789");
    if (pos == std::string::npos) {
        return 0;
    }

    return std::stoi(split.tag.substr(pos));
}

bool gguf_filename_is_model(const std::string & filepath) {
    if (!string_ends_with(filepath, ".gguf")) {
        return false;
    }

    std::string filename = filepath;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }

    return filename.find("mmproj")  == std::string::npos &&
           filename.find("imatrix") == std::string::npos &&
           filename.find("mtp-")    == std::string::npos &&
           filename.find("eagle3-") == std::string::npos &&
           filename.find("dflash-") == std::string::npos;
}

hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                   const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                  const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
}

hf_cache::hf_file find_best_sibling(const hf_cache::hf_files & files,
                                    const std::string        & model,
                                    const std::string        & keyword,
                                    const std::string        & tag) {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool best_exact = false;
    bool found = false;

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    int model_bits = 0;
    if (!tag_upper.empty()) {
        auto pos = tag_upper.find_first_of("0123456789");
        model_bits = pos == std::string::npos ? 0 : std::stoi(tag_upper.substr(pos));
    } else {
        model_bits = extract_quant_bits(model);
    }
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find(keyword) == std::string::npos) {
            continue;
        }

        auto sib_parts = string_split<std::string>(f.path, '/');
        auto sib_dir = sib_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      sib_parts.begin(), sib_dir);
        if (dir != sib_dir) {
            continue;
        }

        size_t depth = dir - sib_parts.begin();
        auto bits = extract_quant_bits(f.path);
        auto diff = std::abs(bits - model_bits);

        std::string path_upper = f.path;
        for (char & c : path_upper) {
            c = (char) std::toupper((unsigned char) c);
        }
        bool exact = !tag_upper.empty() && path_upper.find("-" + tag_upper + ".") != std::string::npos;

        if (!found || depth > best_depth ||
            (depth == best_depth && exact && !best_exact) ||
            (depth == best_depth && exact == best_exact && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    return best;
}


static hf_cache::hf_file find_best_mmproj(const hf_cache::hf_files & files,
                                          const std::string        & model) {
    return find_best_sibling(files, model, "mmproj");
}

static hf_cache::hf_file find_best_mtp(const hf_cache::hf_files & files,
                                       const std::string        & model,
                                       const std::string        & tag = "") {
    return find_best_sibling(files, model, "mtp-", tag);
}

static hf_cache::hf_file find_best_eagle3(const hf_cache::hf_files & files,
                                          const std::string        & model,
                                          const std::string        & tag = "") {
    return find_best_sibling(files, model, "eagle3-", tag);
}

static hf_cache::hf_file find_best_dflash(const hf_cache::hf_files & files,
                                          const std::string        & model,
                                          const std::string        & tag = "") {
    return find_best_sibling(files, model, "dflash-", tag);
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

common_download_hf_plan resolve(const hf_cache::hf_files & files,
                                const std::string        & repo,
                                const std::string        & tag,
                                const std::string        & hf_file,
                                const opts               & o) {
    common_download_hf_plan plan;

    // if preset.ini exists in the repo root, download only that file
    for (const auto & f : files) {
        if (f.path == "preset.ini") {
            plan.preset = f;
            return plan;
        }
    }

    hf_cache::hf_file primary;

    if (!hf_file.empty()) {
        for (const auto & f : files) {
            if (f.path == hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, hf_file.c_str());
            list_available_gguf_files(files);
            return plan;
        }
    } else {
        primary = find_best_model(files, tag);
        // a requested sidecar can resolve on its own, without a full model of the same tag
        if (primary.path.empty() && !o.mtp && !o.dflash && !o.eagle3) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(files);
            return plan;
        }
    }

    if (!primary.path.empty()) {
        plan.primary = primary;
        plan.model_files = get_split_files(files, primary);
    }

    if (o.mmproj && !primary.path.empty()) {
        plan.mmproj = find_best_mmproj(files, primary.path);
    }
    if (o.mtp) {
        plan.mtp = find_best_mtp(files, primary.path, tag);
    }
    if (o.dflash) {
        plan.dflash = find_best_dflash(files, primary.path, tag);
    }
    if (o.eagle3) {
        plan.eagle3 = find_best_eagle3(files, primary.path, tag);
    }

    if (primary.path.empty() &&
        plan.mtp.local_path.empty() && plan.dflash.local_path.empty() && plan.eagle3.local_path.empty()) {
        LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
        list_available_gguf_files(files);
    }

    return plan;
}

} // namespace model_resolution
