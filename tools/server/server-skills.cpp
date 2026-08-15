#include "server-skills.h"

#include "server-common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

namespace {

constexpr size_t SKILL_MAX_BYTES = 1024 * 1024;
constexpr size_t RESOURCE_MAX_BYTES = 10 * 1024 * 1024;
constexpr size_t RESOURCE_LIST_MAX = 256;
constexpr int RESOURCE_LIST_MAX_DEPTH = 4;
// Bounded path-free LRU catalog cache: entries are keyed by canonical
// effective CWD plus tokenizer generation, promoted on hit, and capped here.
constexpr size_t CATALOG_CACHE_MAX = 32;
// Client-supplied resource paths are capped before the missing-resource
// suggestion ranker runs: listing candidates are bounded by the depth-four
// enumeration (each component is filesystem-name limited, at most 255 bytes on
// POSIX), so this cap keeps the Damerau-Levenshtein matrix allocation bounded
// while staying far above any path the listing can produce.
constexpr size_t RESOURCE_PATH_MAX_BYTES = 4096;

struct skill_diagnostic {
    std::string severity;
    std::string code;
    std::string name;
    std::string scope;
    std::string provider;
    std::string message;
};

struct parsed_skill {
    std::string name;
    std::string description;
    std::string license;
    std::string compatibility;
    std::string allowed_tools;
    std::map<std::string, std::string> metadata;
    std::string body;
};

struct skill_entry {
    std::string id;
    std::string name;
    std::string scope;
    std::string provider;
    fs::path root;
    fs::path directory;
    fs::path skill_file;
    parsed_skill parsed;
    // instruction measurement for the catalog (exact or estimated)
    size_t tokens = 0;
    bool tokens_estimated = true;
};

struct skill_catalog {
    std::vector<skill_entry> skills;
    std::vector<skill_diagnostic> diagnostics;
};

struct resource_listing {
    std::vector<std::string> paths;
    bool truncated = false;
};

// One cached skill: safe parsed/serialized values, observable file state
// sufficient to detect an in-place mutation, and the measured instruction
// tokens. Deliberately path-free: no raw roots, resource paths, or
// authorization state.
struct cached_skill {
    std::string id;
    parsed_skill parsed;
    std::string file_state;
    size_t tokens = 0;
    bool tokens_estimated = true;
};

// One catalog cache entry, keyed by canonical effective CWD plus tokenizer
// generation. The key string is the canonical CWD (never a stored root or
// resource path).
struct catalog_cache_entry {
    std::string cwd;
    uint64_t generation = 0;
    std::vector<cached_skill> skills;
};

static const cached_skill * entry_find_skill(const catalog_cache_entry & entry, const std::string & id) {
    for (const auto & skill : entry.skills) {
        if (skill.id == id) {
            return &skill;
        }
    }
    return nullptr;
}

static void entry_upsert_skill(catalog_cache_entry & entry, cached_skill skill) {
    for (auto & existing : entry.skills) {
        if (existing.id == skill.id) {
            existing = std::move(skill);
            return;
        }
    }
    entry.skills.push_back(std::move(skill));
}

enum class read_result {
    ok,
    too_large,
    invalid_utf8,
    failed,
};

static fs::path path_from_utf8(const std::string & value) {
    return fs::u8path(value);
}

static std::string path_to_utf8(const fs::path & value) {
    return value.generic_u8string();
}

static bool is_valid_utf8(const std::string & value) {
    for (size_t i = 0; i < value.size();) {
        const unsigned char lead = static_cast<unsigned char>(value[i]);
        if (lead <= 0x7f) {
            ++i;
            continue;
        }
        size_t length = 0;
        unsigned int codepoint = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            length = 2;
            codepoint = lead & 0x1f;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            length = 3;
            codepoint = lead & 0x0f;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            length = 4;
            codepoint = lead & 0x07;
        } else {
            return false;
        }
        if (i + length > value.size()) {
            return false;
        }
        for (size_t j = 1; j < length; ++j) {
            const unsigned char byte = static_cast<unsigned char>(value[i + j]);
            if ((byte & 0xc0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (byte & 0x3f);
        }
        if ((length == 3 && codepoint < 0x800) ||
            (length == 4 && codepoint < 0x10000) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
            return false;
        }
        i += length;
    }
    return true;
}

static read_result read_utf8_file(const fs::path & path, size_t max_bytes, std::string & out) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        return read_result::failed;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return read_result::failed;
    }
    out.resize(max_bytes + 1);
    stream.read(&out[0], static_cast<std::streamsize>(out.size()));
    out.resize(static_cast<size_t>(stream.gcount()));
    if (out.size() > max_bytes) {
        return read_result::too_large;
    }
    if (!stream.eof() && stream.fail()) {
        return read_result::failed;
    }
    if (!is_valid_utf8(out)) {
        return read_result::invalid_utf8;
    }
    return read_result::ok;
}

static bool is_descendant(const fs::path & child, const fs::path & parent) {
    auto child_it = child.begin();
    auto parent_it = parent.begin();
    for (; parent_it != parent.end(); ++parent_it, ++child_it) {
        if (child_it == child.end() || *child_it != *parent_it) {
            return false;
        }
    }
    return true;
}

static bool canonical_below(const fs::path & path, const fs::path & root, fs::path & canonical_path) {
    std::error_code ec;
    canonical_path = fs::canonical(path, ec);
    return !ec && is_descendant(canonical_path, root);
}

static std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

static bool parse_scalar(const std::string & source, std::string & value) {
    value = trim(source);
    if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                              (value.front() == '"' && value.back() == '"'))) {
        value = value.substr(1, value.size() - 2);
    }
    return !value.empty() && value.front() != '[' && value.front() != '{' && value.front() != '&' && value.front() != '*';
}

static bool parse_skill(const std::string & source, parsed_skill & skill) {
    const size_t first_end = source.find('\n');
    const std::string first = source.substr(0, first_end == std::string::npos ? source.size() : first_end);
    if (trim(first) != "---") {
        return false;
    }

    size_t position = first_end == std::string::npos ? source.size() : first_end + 1;
    bool in_metadata = false;
    bool closed = false;
    while (position <= source.size()) {
        const size_t end = source.find('\n', position);
        std::string line = source.substr(position, end == std::string::npos ? std::string::npos : end - position);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (trim(line) == "---" || trim(line) == "...") {
            position = end == std::string::npos ? source.size() : end + 1;
            closed = true;
            break;
        }
        const bool indented = !line.empty() && (line.front() == ' ' || line.front() == '\t');
        if (in_metadata && indented) {
            const std::string item = trim(line);
            const size_t colon = item.find(':');
            std::string key;
            std::string value;
            if (colon != std::string::npos && !trim(item.substr(0, colon)).empty() &&
                parse_scalar(item.substr(colon + 1), value)) {
                key = trim(item.substr(0, colon));
                skill.metadata[key] = value;
            }
        } else if (!indented) {
            in_metadata = false;
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                const std::string key = trim(line.substr(0, colon));
                std::string value;
                if (key == "metadata" && trim(line.substr(colon + 1)).empty()) {
                    in_metadata = true;
                } else if (parse_scalar(line.substr(colon + 1), value)) {
                    if (key == "name") {
                        skill.name = value;
                    } else if (key == "description") {
                        skill.description = value;
                    } else if (key == "license") {
                        skill.license = value;
                    } else if (key == "compatibility") {
                        skill.compatibility = value;
                    } else if (key == "allowed-tools") {
                        skill.allowed_tools = value;
                    }
                }
            }
        }
        if (end == std::string::npos) {
            break;
        }
        position = end + 1;
    }
    if (!closed) {
        return false;
    }
    skill.body = source.substr(position);
    return true;
}

static std::string xml_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '\"': escaped += "&quot;"; break;
            case '\'': escaped += "&apos;"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

static std::string stable_id(const fs::path & origin, const std::string & scope, const std::string & provider) {
    const std::string input = path_to_utf8(origin) + "\n" + scope + "\n" + provider;
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char character : input) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    std::ostringstream value;
    value << "skill_" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return value.str();
}

static bool is_safe_name(const std::string & name) {
    return !name.empty() && is_valid_utf8(name) && name.find('/') == std::string::npos && name.find('\\') == std::string::npos &&
           name != "." && name != "..";
}

// Documented cosmetic public-name grammar: lowercase ASCII letters, digits,
// and hyphens (no leading or trailing hyphen), at most 64 characters. Enforced
// as a cosmetic defect (the safe public name is retained with a warning), not
// as a safety boundary.
static bool is_cosmetic_name(const std::string & name) {
    if (name.empty() || name.size() > 64) {
        return false;
    }
    if (name.front() == '-' || name.back() == '-') {
        return false;
    }
    for (const unsigned char character : name) {
        if ((character < 'a' || character > 'z') && (character < '0' || character > '9') && character != '-') {
            return false;
        }
    }
    return true;
}

static skill_diagnostic diagnostic(std::string severity, std::string code, std::string message,
                                   std::string name = "", std::string scope = "", std::string provider = "") {
    return {std::move(severity), std::move(code), std::move(name), std::move(scope), std::move(provider), std::move(message)};
}

static json diagnostics_json(const std::vector<skill_diagnostic> & diagnostics) {
    json out = json::array();
    for (const auto & item : diagnostics) {
        json value = {
            {"severity", item.severity},
            {"code", item.code},
            {"message", item.message},
        };
        if (!item.name.empty()) value["name"] = item.name;
        if (!item.scope.empty()) value["scope"] = item.scope;
        if (!item.provider.empty()) value["provider"] = item.provider;
        out.push_back(std::move(value));
    }
    return out;
}

static std::string modified_at(const fs::path & file) {
    std::error_code ec;
    const fs::file_time_type time = fs::last_write_time(file, ec);
    if (ec) {
        return "";
    }
    const auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            time - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    const std::time_t seconds = std::chrono::system_clock::to_time_t(system_time);
    std::tm timestamp {};
#if defined(_WIN32)
    gmtime_s(&timestamp, &seconds);
#else
    gmtime_r(&seconds, &timestamp);
#endif
    std::ostringstream out;
    out << std::put_time(&timestamp, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

// Observable file state (modification time plus size) sufficient to detect an
// in-place mutation of a skill file. The catalog cache stores this instead of
// raw file contents so it can decide whether a cached parse is still current.
// Returns an empty string when the state cannot be observed, which forces a
// re-read instead of risking a stale cached parse.
static std::string observable_file_state(const fs::path & file) {
    std::error_code ec;
    const fs::file_time_type time = fs::last_write_time(file, ec);
    if (ec) {
        return "";
    }
    const uintmax_t size = fs::file_size(file, ec);
    if (ec) {
        return "";
    }
    std::ostringstream out;
    out << time.time_since_epoch().count() << ":" << size;
    return out.str();
}

static bool safe_relative_path(const std::string & source, std::string & normalized) {
    if (source.empty() || source.find('\\') != std::string::npos || !is_valid_utf8(source)) {
        return false;
    }
    const fs::path path = path_from_utf8(source);
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto & part : path) {
        if (part == "." || part == ".." || part.empty()) {
            return false;
        }
    }
    normalized = path_to_utf8(path.lexically_normal());
    return !normalized.empty() && normalized != ".";
}

static resource_listing list_resources(const skill_entry & skill) {
    std::set<std::string> paths;
    bool truncated = false;
    std::error_code ec;
    fs::recursive_directory_iterator it(skill.directory, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end) {
        const fs::directory_entry entry = *it;
        const int depth = it.depth() + 1;
        if (entry.is_directory(ec)) {
            if (ec || depth >= RESOURCE_LIST_MAX_DEPTH) {
                it.disable_recursion_pending();
                ec.clear();
            }
        } else if (entry.is_regular_file(ec) && !ec && entry.path().filename() != "SKILL.md" && depth <= RESOURCE_LIST_MAX_DEPTH) {
            fs::path canonical_file;
            if (canonical_below(entry.path(), skill.directory, canonical_file)) {
                std::error_code relative_ec;
                const fs::path relative = fs::relative(canonical_file, skill.directory, relative_ec);
                const std::string value = relative_ec ? "" : path_to_utf8(relative);
                std::string normalized;
                if (!value.empty() && safe_relative_path(value, normalized)) {
                    paths.insert(normalized);
                    if (paths.size() > RESOURCE_LIST_MAX) {
                        paths.erase(std::prev(paths.end()));
                        truncated = true;
                    }
                }
            }
        }
        ec.clear();
        it.increment(ec);
    }
    return {{paths.begin(), paths.end()}, truncated};
}

static size_t line_count(const std::string & text) {
    if (text.empty()) {
        return 0;
    }
    return static_cast<size_t>(std::count(text.begin(), text.end(), '\n')) + (text.back() == '\n' ? 0 : 1);
}

static std::string header_value(const server_http_req & request, const std::string & wanted) {
    std::string lower_wanted = wanted;
    std::transform(lower_wanted.begin(), lower_wanted.end(), lower_wanted.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const auto & header : request.headers) {
        std::string key = header.first;
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
        if (key == lower_wanted) {
            return header.second;
        }
    }
    return "";
}

static bool header_present(const server_http_req & request, const std::string & wanted) {
    std::string lower_wanted = wanted;
    std::transform(lower_wanted.begin(), lower_wanted.end(), lower_wanted.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const auto & header : request.headers) {
        std::string key = header.first;
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
        if (key == lower_wanted) {
            return true;
        }
    }
    return false;
}

static server_http_res_ptr error_response(int status, const std::string & message, error_type type,
                                          const std::vector<std::string> & suggestions = {}) {
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    json data = {{"error", format_error_response(message, type)}};
    if (!suggestions.empty()) {
        data["suggestions"] = suggestions;
    }
    response->data = safe_json_to_str(data);
    return response;
}

static int damerau_levenshtein(const std::string & left, const std::string & right) {
    const int rows = static_cast<int>(left.size());
    const int columns = static_cast<int>(right.size());
    const int maximum = rows + columns;
    std::vector<std::vector<int>> distance(rows + 2, std::vector<int>(columns + 2));
    std::array<int, 256> last {};
    distance[0][0] = maximum;
    for (int i = 0; i <= rows; ++i) {
        distance[i + 1][0] = maximum;
        distance[i + 1][1] = i;
    }
    for (int j = 0; j <= columns; ++j) {
        distance[0][j + 1] = maximum;
        distance[1][j + 1] = j;
    }
    for (int i = 1; i <= rows; ++i) {
        int last_match_column = 0;
        for (int j = 1; j <= columns; ++j) {
            const int matching_row = last[static_cast<unsigned char>(right[j - 1])];
            const int matching_column = last_match_column;
            const int cost = left[i - 1] == right[j - 1] ? 0 : 1;
            if (cost == 0) {
                last_match_column = j;
            }
            distance[i + 1][j + 1] = std::min({
                distance[i][j] + cost,
                distance[i + 1][j] + 1,
                distance[i][j + 1] + 1,
                distance[matching_row][matching_column] + (i - matching_row - 1) + 1 + (j - matching_column - 1),
            });
        }
        last[static_cast<unsigned char>(left[i - 1])] = i;
    }
    return distance[rows + 1][columns + 1];
}

struct ranked_path {
    double score;
    std::string path;
};

// Bounded top-three suggestion selection. Preserves the ordering of the full
// (normalized Damerau-Levenshtein, then lexical) sort while retaining only the
// three best candidates, so the missing-resource path never ranks or stores
// every entry of the (already capped) listing.
static void keep_best_three(std::vector<ranked_path> & best, ranked_path candidate) {
    auto position = best.end();
    for (auto it = best.begin(); it != best.end(); ++it) {
        if (candidate.score < it->score || (candidate.score == it->score && candidate.path < it->path)) {
            position = it;
            break;
        }
    }
    if (position == best.end() && best.size() == 3) {
        return; // worse than every retained candidate
    }
    best.insert(position, std::move(candidate));
    if (best.size() > 3) {
        best.pop_back();
    }
}

// Shared discovery pass over one configured root: enumerate immediate child
// directories in lexical order, validate each candidate (canonical containment
// under the root, exact SKILL.md, bounded UTF-8 frontmatter), warn on cosmetic
// public-name grammar defects while retaining the safe frontmatter name, reject
// only unsafe names, and retain only the first (highest-precedence) entry per
// name. `base` is the canonical HOME or effective CWD the configured root must
// stay beneath: a root whose canonical target escapes that base is rejected
// with skill_root_invalid.
//
// When `cache_entry` is non-null, discovery consults the path-free catalog
// cache entry: an unchanged observable file state reuses the cached parse and
// instruction measurement without re-reading the file; otherwise the file is
// re-read and re-measured. `count_tokens` (when non-null) supplies exact
// direct-tokenizer counts; on absence or an unavailable tokenizer the
// instruction is estimated as ceil(bytes / 4).
static void add_root_skills(const fs::path & candidate_root, const fs::path & base, const std::string & scope,
                            const std::string & provider, skill_catalog & catalog,
                            catalog_cache_entry * cache_entry, const token_count_callback * count_tokens) {
    std::error_code ec;
    if (!fs::exists(candidate_root, ec) || ec) {
        return;
    }
    const fs::path root = fs::canonical(candidate_root, ec);
    if (ec || !fs::is_directory(root, ec) || ec || !is_descendant(root, base)) {
        catalog.diagnostics.push_back(diagnostic("warning", "skill_root_invalid", "Skill root could not be used", "", scope, provider));
        return;
    }
    std::vector<fs::path> children;
    for (fs::directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec) && !ec) {
            children.push_back(it->path());
        }
        ec.clear();
    }
    std::sort(children.begin(), children.end(), [](const fs::path & left, const fs::path & right) {
        return path_to_utf8(left.filename()) < path_to_utf8(right.filename());
    });
    for (const fs::path & child : children) {
        // Diagnostic identity for rejected candidates comes from the candidate
        // directory basename, derived before any file parsing. Unsafe basenames
        // are not echoed: the name stays empty.
        const std::string candidate_name = path_to_utf8(child.filename());
        const std::string candidate_diagnostic_name = is_safe_name(candidate_name) ? candidate_name : "";
        fs::path directory;
        fs::path skill_file;
        if (!canonical_below(child, root, directory) ||
            !canonical_below(child / "SKILL.md", root, skill_file) ||
            !is_descendant(skill_file, directory)) {
            catalog.diagnostics.push_back(diagnostic("warning", "skill_unsafe_path", "Skill could not be used", candidate_diagnostic_name, scope, provider));
            continue;
        }
        const std::string id = stable_id(directory, scope, provider);
        const std::string state = observable_file_state(skill_file);
        const cached_skill * cached = cache_entry != nullptr ? entry_find_skill(*cache_entry, id) : nullptr;
        const bool reuse = cached != nullptr && !state.empty() && cached->file_state == state;

        parsed_skill parsed;
        if (reuse) {
            parsed = cached->parsed;
        } else {
            std::string source;
            const read_result read_state = read_utf8_file(skill_file, SKILL_MAX_BYTES, source);
            if (read_state != read_result::ok) {
                const char * code = read_state == read_result::too_large ? "skill_too_large" :
                                   read_state == read_result::invalid_utf8 ? "skill_invalid_utf8" : "skill_unreadable";
                catalog.diagnostics.push_back(diagnostic("warning", code, "Skill could not be read", candidate_diagnostic_name, scope, provider));
                continue;
            }
            if (!parse_skill(source, parsed)) {
                catalog.diagnostics.push_back(diagnostic("warning", "skill_invalid_frontmatter", "Skill frontmatter is invalid", candidate_diagnostic_name, scope, provider));
                continue;
            }
        }
        if (parsed.description.empty()) {
            catalog.diagnostics.push_back(diagnostic("warning", "skill_missing_description", "Skill has no description", candidate_diagnostic_name, scope, provider));
            continue;
        }
        std::string name = parsed.name;
        if (!is_safe_name(name)) {
            // Only unsafe names (empty, separator- or traversal-bearing) are
            // rejected. The diagnostic carries no name because echoing an
            // unsafe name could reflect unvalidated input.
            catalog.diagnostics.push_back(diagnostic("warning", "skill_name_invalid", "Skill name is invalid", "", scope, provider));
            continue;
        }
        if (!is_cosmetic_name(name)) {
            // Cosmetic grammar defects warn but load: the safe public
            // frontmatter name is retained, never replaced by the directory
            // name, so selection identity is preserved.
            catalog.diagnostics.push_back(diagnostic("warning", "skill_name_invalid", "Skill name is invalid", name, scope, provider));
        }
        const auto existing = std::find_if(catalog.skills.begin(), catalog.skills.end(), [&](const skill_entry & entry) {
            return entry.name == name;
        });
        if (existing != catalog.skills.end()) {
            catalog.diagnostics.push_back(diagnostic("warning", "skill_shadowed", "Skill is shadowed by a higher-precedence entry", name, scope, provider));
            continue;
        }

        // Instruction measurement: reuse cached counts on an unchanged file,
        // otherwise measure exactly (direct tokenizer available) or estimate
        // the integer ceiling of bytes / 4.
        size_t tokens = (parsed.body.size() + 3) / 4;
        bool tokens_estimated = true;
        if (reuse) {
            tokens = cached->tokens;
            tokens_estimated = cached->tokens_estimated;
        } else if (count_tokens != nullptr) {
            try {
                if (auto snapshot = (*count_tokens)(parsed.body)) {
                    tokens = snapshot->count;
                    tokens_estimated = false;
                }
            } catch (...) {
            }
        }

        if (cache_entry != nullptr) {
            cached_skill cached_value;
            cached_value.id = id;
            cached_value.parsed = parsed;
            cached_value.file_state = state;
            cached_value.tokens = tokens;
            cached_value.tokens_estimated = tokens_estimated;
            entry_upsert_skill(*cache_entry, std::move(cached_value));
        }
        catalog.skills.push_back({std::move(id), std::move(name), scope, provider, root, directory, skill_file,
                                  std::move(parsed), tokens, tokens_estimated});
    }
}

} // namespace

// Path-free bounded LRU catalog cache (declared in server-skills.h). Entries
// are keyed by canonical effective CWD plus tokenizer generation, promoted on
// hit, and capped at CATALOG_CACHE_MAX. Entries hold only safe
// parsed/serialized values, observable file state, and measured instruction
// tokens -- never raw roots, resource paths, or authorization state.
//
// The cache is shared by concurrent HTTP worker threads, so every access is
// serialized by `mutex` and no raw pointer into the internal lists ever
// escapes: `lookup` returns a copy of the entry (promoting on hit), the
// caller mutates and measures the copy while doing file I/O, and `store`
// writes it back under the lock. A concurrent store/eviction can therefore
// never invalidate an entry a handler is still using.
struct skill_catalog_cache {
    std::mutex mutex;
    std::list<catalog_cache_entry> entries; // front = most recently used
    std::map<std::pair<std::string, uint64_t>, std::list<catalog_cache_entry>::iterator> index;

    // Find the entry for (cwd, generation) under the cache lock, promote it to
    // most-recently-used on a hit, and return a copy of it; returns nullopt on
    // a miss. The copy keeps the caller free of any pointer into the cache.
    std::optional<catalog_cache_entry> lookup(const std::string & cwd, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto key = std::make_pair(cwd, generation);
        const auto found = index.find(key);
        if (found == index.end()) {
            return std::nullopt;
        }
        entries.splice(entries.begin(), entries, found->second); // promote on hit
        return *found->second; // copy
    }

    // Insert or replace the entry for its (cwd, generation) key under the
    // cache lock, evicting the least-recently-used entry when the cap is
    // exceeded. Replacing an existing entry keeps the size unchanged, so
    // eviction only ever happens on a genuine insert.
    void store(catalog_cache_entry entry) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto key = std::make_pair(entry.cwd, entry.generation);
        const auto found = index.find(key);
        if (found != index.end()) {
            entries.erase(found->second);
            index.erase(found);
        }
        entries.push_front(std::move(entry));
        index[key] = entries.begin();
        while (entries.size() > CATALOG_CACHE_MAX) {
            index.erase(std::make_pair(entries.back().cwd, entries.back().generation));
            entries.pop_back();
        }
    }
};

server_skills::server_skills(server_skills_config config, token_count_callback count_tokens)
    : config(std::move(config)),
      count_tokens(std::move(count_tokens)),
      process_cwd(std::filesystem::canonical(std::filesystem::current_path())),
      catalog_cache(std::make_unique<skill_catalog_cache>()) {
    // Follow the server-tools.cpp home_dir() convention: wide environment
    // lookup on Windows (the narrow getenv returns the profile path in the
    // active code page), plain narrow lookup on POSIX.
    std::string home;
#if defined(_WIN32)
    const wchar_t * wide_home = _wgetenv(L"HOME");
    if (wide_home == nullptr || *wide_home == L'\0') {
        wide_home = _wgetenv(L"USERPROFILE");
    }
    if (wide_home != nullptr && *wide_home != L'\0') {
        home = path_to_utf8(fs::path(wide_home));
    }
#else
    const char * posix_home = std::getenv("HOME");
    if (posix_home != nullptr && *posix_home != '\0') {
        home = posix_home;
    }
#endif
    if (!home.empty()) {
        std::error_code ec;
        const fs::path value = fs::canonical(path_from_utf8(home), ec);
        if (!ec && fs::is_directory(value, ec) && !ec) {
            process_home = value;
        }
    }

    handle_get = [this](const server_http_req & request) {
        try {
            const bool explicit_cwd = header_present(request, "X-Skill-Cwd");
            fs::path cwd = process_cwd;
            if (explicit_cwd) {
                const std::string value = header_value(request, "X-Skill-Cwd");
                std::error_code ec;
                cwd = value.empty() ? fs::path{} : fs::canonical(path_from_utf8(value), ec);
                if (value.empty() || ec || !fs::is_directory(cwd, ec) || ec) {
                    return error_response(400, "invalid skill working directory", ERROR_TYPE_INVALID_REQUEST);
                }
            }

            // Tokenizer generation for cache keying: an empty-text probe
            // returns the current generation, or nullopt when the direct
            // tokenizer is unavailable (router/unloaded/sleeping), in which
            // case catalogs are measured by estimation and keyed at
            // generation 0.
            uint64_t generation = 0;
            if (this->count_tokens) {
                try {
                    if (auto snapshot = this->count_tokens("")) {
                        generation = snapshot->generation;
                    }
                } catch (...) {
                }
            }
            const std::string cwd_key = path_to_utf8(cwd);
            // lookup returns a copy of the entry (or nullopt on a miss); the
            // working entry is a request-local value, so a concurrent store or
            // eviction can never invalidate it while file I/O runs below.
            std::optional<catalog_cache_entry> cached_entry = this->catalog_cache->lookup(cwd_key, generation);
            catalog_cache_entry entry;
            if (cached_entry.has_value()) {
                entry = std::move(*cached_entry);
            } else {
                entry.cwd = cwd_key;
                entry.generation = generation;
            }
            catalog_cache_entry * cache = &entry;

            skill_catalog catalog;
            const token_count_callback * count_tokens = this->count_tokens ? &this->count_tokens : nullptr;
            if (this->config.trust_project_skills) {
                add_root_skills(cwd / ".agents" / "skills", cwd, "project", "agents", catalog, cache, count_tokens);
                for (const std::string & provider : this->config.providers) {
                    if (provider == "agents") {
                        continue; // built-in agents root already scanned above
                    }
                    add_root_skills(cwd / ("." + provider) / "skills", cwd, "project", provider, catalog, cache, count_tokens);
                }
            }
            if (!process_home.empty()) {
                add_root_skills(process_home / ".agents" / "skills", process_home, "global", "agents", catalog, cache, count_tokens);
                for (const std::string & provider : this->config.providers) {
                    if (provider == "agents") {
                        continue; // built-in agents root already scanned above
                    }
                    add_root_skills(process_home / ("." + provider) / "skills", process_home, "global", provider, catalog, cache, count_tokens);
                }
            }
            // write the working entry back under the cache lock (replace on a
            // hit, insert with LRU eviction on a miss)
            this->catalog_cache->store(std::move(entry));

            json output = {
                {"skills", json::array()},
                {"catalog_instruction_xml", catalog.skills.empty() ? "" : "<available_skills>Call read_skill(name) when a task matches a skill description.</available_skills>"},
                {"diagnostics", diagnostics_json(catalog.diagnostics)},
            };
            for (const skill_entry & skill : catalog.skills) {
                // resource lists are re-enumerated for every catalog read and
                // never stored in cache entries
                const resource_listing resources = list_resources(skill);
                const size_t bytes = skill.parsed.body.size();
                const std::string last_modified = modified_at(skill.skill_file);
                json instruction = {
                    {"bytes", bytes},
                    {"lines", line_count(skill.parsed.body)},
                    {"tokens", skill.tokens},
                    {"tokens_estimated", skill.tokens_estimated},
                    {"modified_at", last_modified.empty() ? json(nullptr) : json(last_modified)},
                };
                output["skills"].push_back({
                    {"id", skill.id},
                    {"name", skill.name},
                    {"description", skill.parsed.description},
                    {"scope", skill.scope},
                    {"provider", skill.provider},
                    {"instruction", std::move(instruction)},
                    {"resources", {{"count", resources.paths.size()}, {"truncated", resources.truncated}}},
                    {"catalog_xml", "<skill><name>" + xml_escape(skill.name) + "</name><description>" + xml_escape(skill.parsed.description) + "</description></skill>"},
                });
            }
            auto response = std::make_unique<server_http_res>();
            response->data = safe_json_to_str(output);
            return response;
        } catch (...) {
            return error_response(500, "unable to read skills", ERROR_TYPE_SERVER);
        }
    };

    handle_post = [this](const server_http_req & request) {
        try {
            json request_body;
            try {
                request_body = json::parse(request.body);
            } catch (...) {
                return error_response(400, "request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
            }
            if (!request_body.is_object() || !request_body.contains("name") || !request_body.at("name").is_string() ||
                (request_body.size() != 1 && request_body.size() != 2) ||
                (request_body.size() == 2 && (!request_body.contains("path") || !request_body.at("path").is_string()))) {
                return error_response(400, "invalid skill request", ERROR_TYPE_INVALID_REQUEST);
            }
            const std::string wanted_name = request_body.at("name").get<std::string>();
            if (!is_safe_name(wanted_name)) {
                return error_response(400, "invalid skill name", ERROR_TYPE_INVALID_REQUEST);
            }
            std::string requested_path;
            const bool is_resource = request_body.contains("path");
            if (is_resource) {
                const std::string raw_path = request_body.at("path").get<std::string>();
                if (raw_path.size() > RESOURCE_PATH_MAX_BYTES || !safe_relative_path(raw_path, requested_path)) {
                    return error_response(400, "invalid skill resource path", ERROR_TYPE_INVALID_REQUEST);
                }
            }

            const bool explicit_cwd = header_present(request, "X-Skill-Cwd");
            fs::path cwd = process_cwd;
            if (explicit_cwd) {
                const std::string value = header_value(request, "X-Skill-Cwd");
                std::error_code ec;
                cwd = value.empty() ? fs::path{} : fs::canonical(path_from_utf8(value), ec);
                if (value.empty() || ec || !fs::is_directory(cwd, ec) || ec) {
                    return error_response(400, "invalid skill working directory", ERROR_TYPE_INVALID_REQUEST);
                }
            }

            skill_catalog catalog;
            if (this->config.trust_project_skills) {
                add_root_skills(cwd / ".agents" / "skills", cwd, "project", "agents", catalog, nullptr, nullptr);
                for (const std::string & provider : this->config.providers) {
                    if (provider == "agents") {
                        continue; // built-in agents root already scanned above
                    }
                    add_root_skills(cwd / ("." + provider) / "skills", cwd, "project", provider, catalog, nullptr, nullptr);
                }
            }
            if (!process_home.empty()) {
                add_root_skills(process_home / ".agents" / "skills", process_home, "global", "agents", catalog, nullptr, nullptr);
                for (const std::string & provider : this->config.providers) {
                    if (provider == "agents") {
                        continue; // built-in agents root already scanned above
                    }
                    add_root_skills(process_home / ("." + provider) / "skills", process_home, "global", provider, catalog, nullptr, nullptr);
                }
            }
            const auto found = std::find_if(catalog.skills.begin(), catalog.skills.end(), [&](const skill_entry & skill) {
                return skill.name == wanted_name;
            });
            if (found == catalog.skills.end()) {
                return error_response(404, "skill not found", ERROR_TYPE_NOT_FOUND);
            }
            const skill_entry & skill = *found;

            if (!is_resource) {
                std::string source;
                const read_result state = read_utf8_file(skill.skill_file, SKILL_MAX_BYTES, source);
                if (state != read_result::ok) {
                    return error_response(400, "skill content is unavailable", ERROR_TYPE_INVALID_REQUEST);
                }
                parsed_skill current;
                if (!parse_skill(source, current) || current.description.empty()) {
                    return error_response(400, "skill content is unavailable", ERROR_TYPE_INVALID_REQUEST);
                }
                const resource_listing resources = list_resources(skill);
                json metadata = {
                    {"name", skill.name},
                    {"description", current.description},
                };
                if (!current.license.empty()) metadata["license"] = current.license;
                if (!current.compatibility.empty()) metadata["compatibility"] = current.compatibility;
                if (!current.allowed_tools.empty()) metadata["allowed_tools"] = current.allowed_tools;
                if (!current.metadata.empty()) metadata["metadata"] = current.metadata;
                std::string content_xml = "<skill_content name=\"" + xml_escape(skill.name) + "\">" + xml_escape(current.body) + "<skill_resources>";
                for (const std::string & path : resources.paths) {
                    content_xml += "<file>" + xml_escape(path) + "</file>";
                }
                content_xml += "</skill_resources></skill_content>";
                auto response = std::make_unique<server_http_res>();
                response->data = safe_json_to_str({
                    {"kind", "skill"},
                    {"skill", {{"id", skill.id}, {"name", skill.name}, {"scope", skill.scope}, {"provider", skill.provider}, {"metadata", std::move(metadata)}}},
                    {"resources", {{"paths", resources.paths}, {"truncated", resources.truncated}}},
                    {"source", source},
                    {"body_markdown", current.body},
                    {"content_xml", content_xml},
                    {"diagnostics", diagnostics_json(catalog.diagnostics)},
                });
                return response;
            }

            const fs::path resource_path = skill.directory / path_from_utf8(requested_path);
            std::error_code ec;
            if (!fs::exists(resource_path, ec) || ec) {
                const resource_listing listing = list_resources(skill);
                std::vector<ranked_path> best;
                for (const std::string & path : listing.paths) {
                    const size_t denominator = std::max(path.size(), requested_path.size());
                    keep_best_three(best, {denominator == 0 ? 0.0 : static_cast<double>(damerau_levenshtein(requested_path, path)) / denominator, path});
                }
                std::vector<std::string> suggestions;
                suggestions.reserve(best.size());
                for (const ranked_path & item : best) {
                    suggestions.push_back(item.path);
                }
                return error_response(404, "skill resource not found", ERROR_TYPE_NOT_FOUND, suggestions);
            }
            fs::path canonical_resource;
            if (!canonical_below(resource_path, skill.directory, canonical_resource)) {
                return error_response(400, "invalid skill resource path", ERROR_TYPE_INVALID_REQUEST);
            }
            if (!fs::is_regular_file(canonical_resource, ec) || ec) {
                return error_response(400, "invalid skill resource", ERROR_TYPE_INVALID_REQUEST);
            }
            std::string content;
            const read_result state = read_utf8_file(canonical_resource, RESOURCE_MAX_BYTES, content);
            if (state != read_result::ok) {
                return error_response(400, "skill resource is unavailable", ERROR_TYPE_INVALID_REQUEST);
            }
            auto response = std::make_unique<server_http_res>();
            response->data = safe_json_to_str({
                {"kind", "resource"},
                {"skill", {{"id", skill.id}, {"name", skill.name}, {"scope", skill.scope}, {"provider", skill.provider}}},
                {"resource", {{"path", requested_path}}},
                {"content_xml", "<skill_resource name=\"" + xml_escape(skill.name) + "\" path=\"" + xml_escape(requested_path) + "\">" + xml_escape(content) + "</skill_resource>"},
                {"diagnostics", diagnostics_json(catalog.diagnostics)},
            });
            return response;
        } catch (...) {
            return error_response(500, "unable to read skill", ERROR_TYPE_SERVER);
        }
    };
}

server_skills::server_skills(server_skills &&) noexcept = default;
server_skills & server_skills::operator=(server_skills &&) noexcept = default;
server_skills::~server_skills() = default;
