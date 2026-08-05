#include "server-tools.h"

#include "subproc.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include <functional>
#include <memory>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#endif

namespace fs = std::filesystem;

//
// internal helpers
//

#if defined(_WIN32)
// A chunk can end in the middle of a multi-byte sequence, so the incomplete
// tail is dropped before validating what precedes it.
static bool is_utf8_text(const std::string & text) {
    return is_valid_utf8(text.substr(0, validate_utf8(text)));
}

// A child process writes its output in the OEM code page, which is not UTF-8
// on a western Windows install, so accented text reaches the JSON layer as
// invalid bytes and is replaced there. Text that already decodes as UTF-8 is
// returned untouched, so a child that emits UTF-8 is never decoded twice.
// run() spawns without a console, so the console code page does not apply.
static std::string console_output_to_utf8(const std::string & text) {
    if (text.empty() || is_utf8_text(text)) {
        return text;
    }

    const UINT cp = GetOEMCP();

    // fail rather than emit replacement characters when the code page is wrong
    const int wide_len = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, text.data(), (int) text.size(), nullptr, 0);
    if (wide_len <= 0) {
        return text;
    }
    std::wstring wide(wide_len, L'\0');
    MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, text.data(), (int) text.size(), wide.data(), wide_len);

    const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 0) {
        return text;
    }
    std::string utf8(utf8_len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, utf8.data(), utf8_len, nullptr, nullptr);
    return utf8;
}
#else
static std::string console_output_to_utf8(const std::string & text) {
    return text;
}
#endif

json server_tool::to_json() const {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", type()},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

static constexpr size_t SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT = 8 * 1024 * 1024; // 8 MB
// budget for one listing call, shared by the git and walker paths
static constexpr int SERVER_TOOL_LIST_ENTRIES_TIMEOUT = 15; // seconds

// entry kinds a directory listing may return
enum class list_kind {
    files, // regular files only
    dirs,  // directories only
    all,   // both
};

// home directory, read once at first use (getenv is not thread safe against setenv)
static const std::string & home_dir() {
    static const std::string home = [] {
        const char * h = getenv("HOME");
#ifdef _WIN32
        if (h == nullptr) h = getenv("USERPROFILE");
#endif
        return h ? std::string(h) : std::string();
    }();
    return home;
}

static std::string expand_home(const std::string & path) {
    if (path.empty() || path[0] != '~') return path;
    if (path.size() > 1 && path[1] != '/' && path[1] != '\\') return path;
    const std::string & home = home_dir();
    if (home.empty()) return path;
    return home + path.substr(1);
}

// depth of a '/'-separated relative path: "a/b/c" is 3
static int entry_depth(const std::string & rel) {
    return 1 + (int) std::count(rel.begin(), rel.end(), '/');
}

class tools_io {
public:
    struct exec_result {
        std::string output;
        int  exit_code = -1;
        bool timed_out = false;
    };

    virtual ~tools_io() = default;

    virtual bool is_directory(const std::string & path) const = 0;
    virtual bool is_regular_file(const std::string & path) const = 0;
    virtual bool file_size(const std::string & path, uintmax_t & out_size) const = 0;
    virtual bool read_file(const std::string & path, std::string & out) const = 0;
    virtual bool write_file(const std::string & path, const std::string & content) const = 0;
    // resolve `path` against the IO's working directory; absolute paths are returned unchanged
    virtual std::string resolve(const std::string & path) const = 0;
    struct list_entry {
        std::string rel; // '/'-separated, relative to `base`
        bool is_dir = false;
    };
    // entries relative to `base`; sets `err` if `base` isn't a directory
    // max_depth == 0 means unlimited, 1 means direct children of `base` only
    // `base` must already be resolved (absolute); `caller_path` is the path the
    // caller passed, used only for error messages
    virtual std::vector<list_entry> list_entries(const std::string & base, const std::string & caller_path, int max_depth, list_kind kind, std::string & err, bool & truncated) const = 0;
    // on_chunk, if set, is called with each chunk of output as it is read (before truncation cuts in);
    // returning false terminates the process early (e.g. the client disconnected)
    virtual exec_result run(
            const std::vector<std::string> & args,
            size_t max_output,
            int timeout_secs,
            const std::function<bool(const std::string &)> & on_chunk = nullptr) const = 0;
};

class tools_io_basic : public tools_io {
public:
    // cwd, if non-empty, is used to resolve relative paths and as the working directory for run()
    explicit tools_io_basic(std::string cwd = "") : cwd(std::move(cwd)) {}

    // expands a leading `~`, then resolves `path` against `cwd` (or the server
    // working directory when `cwd` is unset); the result is always absolute
    std::string resolve(const std::string & path) const override {
        std::string p = expand_home(path);
        if (fs::path(p).is_absolute()) {
            return p;
        }
        if (cwd.empty()) {
            std::error_code ec;
            fs::path cur = fs::current_path(ec);
            if (ec) return p;
            return (cur / p).string();
        }
        return (fs::path(cwd) / p).string();
    }

    bool is_directory(const std::string & path) const override {
        std::error_code ec;
        return fs::is_directory(resolve(path), ec) && !ec;
    }

    bool is_regular_file(const std::string & path) const override {
        std::error_code ec;
        return fs::is_regular_file(resolve(path), ec) && !ec;
    }

    bool file_size(const std::string & path, uintmax_t & out_size) const override {
        std::error_code ec;
        out_size = fs::file_size(resolve(path), ec);
        return !ec;
    }

    bool read_file(const std::string & path, std::string & out) const override {
        std::ifstream f(resolve(path), std::ios::binary);
        if (!f) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return true;
    }

    bool write_file(const std::string & path, const std::string & content) const override {
        std::error_code ec;
        fs::path fpath(resolve(path));
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) return false;
        }
        std::ofstream f(fpath, std::ios::binary);
        if (!f) return false;
        f << content;
        return (bool) f;
    }

    std::vector<list_entry> list_entries(const std::string & base, const std::string & caller_path, int max_depth, list_kind kind, std::string & err, bool & truncated) const override {
        err.clear();
        truncated = false;
        std::error_code ec;
        if (!fs::is_directory(base, ec) || ec) {
            err = "path does not exist or is not a directory: " + caller_path;
            return {};
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(SERVER_TOOL_LIST_ENTRIES_TIMEOUT);

        // git ls-files cannot list directories; use the walker when they are requested
        if (kind == list_kind::files) {
            auto res = run(
                {"git", "-C", base, "ls-files", "--cached", "--others", "--exclude-standard"},
                SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT, SERVER_TOOL_LIST_ENTRIES_TIMEOUT);

            if (res.exit_code == 0 && !res.timed_out) {
                std::vector<list_entry> result;
                std::istringstream iss(res.output);
                std::string line;
                while (std::getline(iss, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    std::replace(line.begin(), line.end(), '\\', '/');
                    if (max_depth > 0 && entry_depth(line) > max_depth) continue;
                    if (is_regular_file((fs::path(base) / line).string())) {
                        result.push_back({line, false});
                    }
                }
                return result;
            }
        }

        return list_entries_fallback(base, max_depth, kind, deadline, truncated);
    }

    exec_result run(
            const std::vector<std::string> & args,
            size_t max_output,
            int timeout_secs,
            const std::function<bool(const std::string &)> & on_chunk = nullptr) const override {
        exec_result res;

        common_subproc proc;

        int options = subprocess_option_no_window
                    | subprocess_option_combined_stdout_stderr
                    | subprocess_option_inherit_environment
                    | subprocess_option_search_user_path;

        if (!proc.create(args, options, {}, cwd.empty() ? nullptr : cwd.c_str())) {
            res.output = "failed to spawn process";
            return res;
        }

        std::atomic<bool> done{false};
        std::atomic<bool> timed_out{false};

        std::thread timeout_thread([&]() {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
            while (!done.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    timed_out.store(true);
                    proc.terminate();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        FILE * f = proc.stdout_file();
        std::string output;
        bool truncated = false;
        if (f) {
            char buf[4096];
            while (fgets(buf, sizeof(buf), f) != nullptr) {
                if (!truncated) {
                    size_t len = strlen(buf);
                    if (output.size() + len <= max_output) {
                        output.append(buf, len);
                        if (on_chunk && !on_chunk(console_output_to_utf8(std::string(buf, len)))) {
                            proc.terminate();
                            break;
                        }
                    } else {
                        size_t remaining = max_output - output.size();
                        output.append(buf, remaining);
                        if (on_chunk && remaining > 0) on_chunk(console_output_to_utf8(std::string(buf, remaining)));
                        truncated = true;
                    }
                }
            }
        }

        done.store(true);
        if (timeout_thread.joinable()) {
            timeout_thread.join();
        }

        res.exit_code = proc.join();

        res.output    = console_output_to_utf8(output);
        res.timed_out = timed_out.load();
        if (truncated) {
            res.output += "\n[output truncated]";
        }
        return res;
    }

private:
    std::string cwd;

    static const std::unordered_set<std::string> & junk_dir_names() {
        static const std::unordered_set<std::string> names = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__",
            ".venv", "venv", "dist", "build", "target", ".cache", ".idea", ".vscode",
        };
        return names;
    }

    std::vector<list_entry> list_entries_fallback(const std::string & base, int max_depth, list_kind kind,
                                                  std::chrono::steady_clock::time_point deadline, bool & truncated) const {
        std::vector<list_entry> result;
        std::error_code ec;

        std::vector<std::tuple<fs::path, fs::path, int>> stack;
        stack.emplace_back(fs::path(base), fs::path(), 0);

        while (!stack.empty()) {
            auto [dir, rel_dir, depth] = stack.back();
            stack.pop_back();

            // the throwing increment would escape the tool on a directory that
            // goes away mid walk, so step the iterator explicitly
            fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
            for (const fs::directory_iterator end; it != end; it.increment(ec)) {
                if (ec) break;
                if (std::chrono::steady_clock::now() >= deadline) {
                    truncated = true;
                    return result;
                }
                const fs::directory_entry & entry = *it;
                std::string fname = entry.path().filename().string();
                std::error_code tec;
                if (entry.is_directory(tec)) {
                    std::string rel = (rel_dir / fname).string();
                    std::replace(rel.begin(), rel.end(), '\\', '/');
                    if (kind == list_kind::dirs || kind == list_kind::all) {
                        result.push_back({rel, true});
                    }
                    // junk directories stay selectable but are never walked: they
                    // hold nothing worth searching and can be enormous
                    if (junk_dir_names().count(fname) > 0) continue;
                    // do not descend into symlinks: a link can point back to an
                    // ancestor and loop forever
                    if (!entry.is_symlink(tec) && (max_depth == 0 || depth + 1 < max_depth)) {
                        stack.emplace_back(entry.path(), rel_dir / fname, depth + 1);
                    }
                } else if (entry.is_regular_file(tec)) {
                    std::string rel = (rel_dir / fname).string();
                    std::replace(rel.begin(), rel.end(), '\\', '/');
                    if (kind == list_kind::files || kind == list_kind::all) {
                        result.push_back({rel, false});
                    }
                }
            }
        }

        return result;
    }
};

static std::unique_ptr<tools_io> make_tools_io(const json & params) {
    std::string cwd = json_value(params, "cwd", std::string());
    return std::make_unique<tools_io_basic>(cwd);
}

// no '/' in pattern -> match basename at any depth; else match full relative path
static bool path_glob_match(const std::string & pattern, const std::string & rel_path) {
    if (pattern.find('/') == std::string::npos) {
        return glob_match(pattern, fs::path(rel_path).filename().string());
    }
    if (pattern == "**" || pattern.rfind("**/", 0) == 0 || pattern.rfind('/', 0) == 0) {
        return glob_match(pattern, rel_path);
    }
    return glob_match("**/" + pattern, rel_path);
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        auto io = make_tools_io(params);

        uintmax_t file_size = 0;
        if (!io->file_size(path, file_size)) {
            return {{"error", "cannot stat file: " + path}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::string content;
        if (!io->read_file(path, content)) {
            return {{"error", "failed to open file: " + path}};
        }

        std::istringstream f(content);
        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192" + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;
static constexpr const char * SERVER_TOOL_FILE_SEARCH_TYPE_FILE = "file";
static constexpr const char * SERVER_TOOL_FILE_SEARCH_TYPE_DIR  = "dir";
static constexpr const char * SERVER_TOOL_FILE_SEARCH_TYPE_ALL  = "all";

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Recursively search for files matching a glob pattern under a directory. "
                    "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
                    "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
                    "A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. "
                    "A pattern containing '/' matches the full relative path; unless already anchored with "
                    "\"**/\" or a leading '/', it is automatically prefixed with \"**/\". "
                    "Use type=\"dir\" or \"all\" to also list directories; directory entries are suffixed with '/' in the output. "
                    "Note: directory listings do not apply .gitignore filtering."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",      {{"type", "string"},  {"description", "Base directory to search in"}}},
                        {"include",   {{"type", "string"},  {"description", "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"}}},
                        {"exclude",   {{"type", "string"},  {"description", "Glob pattern for files to exclude"}}},
                        {"type",      {{"type", "string"},  {"description", "Entry type to return: \"file\" (default), \"dir\" or \"all\""}}},
                        {"max_depth", {{"type", "integer"}, {"description", "Maximum depth to descend into subdirectories (default: 0 = unlimited; 1 = direct children only)"}}},
                        {"limit",     {{"type", "integer"}, {"description", string_format("Maximum number of results to return (default %zu; values below 1 fall back to the default)", SERVER_TOOL_FILE_SEARCH_MAX_RESULTS)}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        auto io = make_tools_io(params);

        std::string base      = io->resolve(params.at("path").get<std::string>());
        // normalize to forward slashes so the web UI (which assumes '/') can
        // join the relative entries into absolute paths on Windows too
        std::replace(base.begin(), base.end(), '\\', '/');
        std::string include   = json_value(params, "include", std::string("**"));
        std::string exclude   = json_value(params, "exclude", std::string(""));
        std::string type      = json_value(params, "type",    std::string("file"));
        int         max_depth = std::max(0, json_value(params, "max_depth", 0));
        int         limit     = json_value(params, "limit", (int) SERVER_TOOL_FILE_SEARCH_MAX_RESULTS);
        if (limit < 1) limit = SERVER_TOOL_FILE_SEARCH_MAX_RESULTS;
        limit = std::min(limit, (int) SERVER_TOOL_FILE_SEARCH_MAX_RESULTS);

        list_kind kind;
        if (type == SERVER_TOOL_FILE_SEARCH_TYPE_FILE) {
            kind = list_kind::files;
        } else if (type == SERVER_TOOL_FILE_SEARCH_TYPE_DIR) {
            kind = list_kind::dirs;
        } else if (type == SERVER_TOOL_FILE_SEARCH_TYPE_ALL) {
            kind = list_kind::all;
        } else {
            return {{"error", "invalid type: " + type + " (expected \"file\", \"dir\" or \"all\")"}};
        }

        std::string err;
        bool truncated = false;
        auto entries = io->list_entries(base, params.at("path").get<std::string>(), max_depth, kind, err, truncated);
        if (!err.empty()) {
            return {{"error", err}};
        }

        std::vector<tools_io::list_entry> matches;
        for (const auto & entry : entries) {
            if (!path_glob_match(include, entry.rel)) continue;
            if (!exclude.empty() && path_glob_match(exclude, entry.rel)) continue;
            matches.push_back(entry);
        }

        size_t total = matches.size();
        size_t shown = std::min(total, (size_t) limit);

        std::ostringstream output_text;
        json entries_json = json::array();
        for (size_t i = 0; i < shown; i++) {
            output_text << matches[i].rel << (matches[i].is_dir ? "/" : "") << "\n";
            entries_json.push_back({
                {"path", matches[i].rel},
                {"type", matches[i].is_dir ? "dir" : "file"},
            });
        }

        output_text << "\n---\nTotal matches: " << total << "\n";
        if (total > shown) {
            output_text << string_format(
                "[%zu results limit reached (%zu total matches). Refine the glob pattern to narrow the search.]\n",
                shown, total);
        }
        if (truncated) {
            output_text << "[search timed out, results truncated]\n";
        }

        // `base` is always absolute (resolve falls back to the server cwd), so
        // API clients (e.g. the web UI picker) can join the relative entries
        // into absolute paths. `plain_text_response` is what the model sees;
        // `entries` is the same data as structured JSON for the UI picker,
        // which reads `entries`/`base` instead of re-parsing the text.
        return {{"plain_text_response", output_text.str()}, {"entries", entries_json}, {"base", base}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Search for a pattern in files under a path. Returns matching lines with file paths "
                    "(and, unless searching a single file, paths relative to the given directory). "
                    "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
                    "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
                    "include/exclude: a pattern with no '/' matches the basename at any depth; a pattern "
                    "containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Pattern to search for (regular expression unless literal is true)"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                        {"literal",             {{"type", "boolean"}, {"description", "Treat pattern as a literal string instead of a regular expression (default: false)"}}},
                        {"ignore_case",         {{"type", "boolean"}, {"description", "Case-insensitive search (default: false)"}}},
                        {"context_lines",       {{"type", "integer"}, {"description", "Number of lines of context to show before and after each match (default: 0)"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string path        = params.at("path").get<std::string>();
        std::string pat_str     = params.at("pattern").get<std::string>();
        std::string include     = json_value(params, "include", std::string("**"));
        std::string exclude     = json_value(params, "exclude", std::string(""));
        bool        show_lineno = json_value(params, "return_line_numbers", false);
        bool        literal     = json_value(params, "literal", false);
        bool        ignore_case = json_value(params, "ignore_case", false);
        int         ctx_lines   = std::max(0, json_value(params, "context_lines", 0));

        std::string pattern_src = pat_str;
        if (literal) {
            static const std::string specials = "\\^$.|?*+()[]{}";
            std::string escaped;
            escaped.reserve(pat_str.size() * 2);
            for (char c : pat_str) {
                if (specials.find(c) != std::string::npos) escaped += '\\';
                escaped += c;
            }
            pattern_src = escaped;
        }

        std::regex pattern;
        try {
            auto flags = std::regex::ECMAScript;
            if (ignore_case) flags |= std::regex::icase;
            pattern = std::regex(pattern_src, flags);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        auto io = make_tools_io(params);

        // collect (absolute_path, display_path) pairs to search
        std::vector<std::pair<std::string, std::string>> files;

        const std::string abs_path = io->resolve(path);
        if (io->is_regular_file(abs_path)) {
            files.emplace_back(abs_path, path);
        } else if (io->is_directory(abs_path)) {
            std::string err;
            bool truncated = false;
            auto candidates = io->list_entries(abs_path, path, 0, list_kind::files, err, truncated);
            if (!err.empty()) {
                return {{"error", err}};
            }
            for (const auto & entry : candidates) {
                if (!path_glob_match(include, entry.rel)) continue;
                if (!exclude.empty() && path_glob_match(exclude, entry.rel)) continue;
                files.emplace_back((fs::path(abs_path) / entry.rel).string(), entry.rel);
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        std::ostringstream output_text;
        size_t total = 0;
        bool limit_reached = false;
        bool show_num = show_lineno || ctx_lines > 0;

        for (const auto & file_entry : files) {
            if (limit_reached) break;
            const std::string & fpath        = file_entry.first;
            const std::string & display_path = file_entry.second;

            std::string content;
            if (!io->read_file(fpath, content)) continue;
            std::vector<std::string> lines;
            {
                std::istringstream f(content);
                std::string line;
                while (std::getline(f, line)) lines.push_back(line);
            }

            for (size_t i = 0; i < lines.size(); i++) {
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                    limit_reached = true;
                    break;
                }
                if (!std::regex_search(lines[i], pattern)) continue;

                long ctx_start = ctx_lines > 0 ? std::max<long>(0, (long) i - ctx_lines) : (long) i;
                long ctx_end   = ctx_lines > 0 ? std::min<long>((long) lines.size() - 1, (long) i + ctx_lines) : (long) i;

                for (long j = ctx_start; j <= ctx_end; j++) {
                    bool is_match = (j == (long) i);
                    output_text << display_path << (is_match ? ':' : '-');
                    if (show_num) {
                        output_text << (j + 1) << (is_match ? ':' : '-');
                    }
                    output_text << lines[j] << "\n";
                }
                if (ctx_lines > 0) {
                    output_text << "--\n";
                }
                total++;
            }
        }

        output_text << "\n---\nTotal matches: " << total << "\n";
        if (limit_reached) {
            output_text << string_format(
                "[%zu matches limit reached. Narrow the path/pattern/include to see more.]\n",
                SERVER_TOOL_GREP_SEARCH_MAX_RESULTS);
        }

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
        support_stream = true;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream * st) const override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto io = make_tools_io(params);

        if (st) {
            auto res = io->run(args, max_output, timeout, [st](const std::string & chunk) {
                st->push(chunk);
                return !st->alive || st->alive();
            });
            if (st->alive && !st->alive()) {
                return json();
            }
            std::string tail = string_format("\n[exit code: %d]", res.exit_code);
            if (res.timed_out) {
                tail += " [exit due to timed out]";
            }
            st->push(tail);
            return json();
        }

        auto res = io->run(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        auto io = make_tools_io(params);
        if (!io->write_file(path, content)) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: exact text replacement, one or more edits per call
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file "
                    "and is matched against the original content, not incrementally. Merge nearby changes into "
                    "one edit instead of overlapping edits. Use write_file to replace the whole file."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",  {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"edits", {
                            {"type", "array"},
                            {"description", "One or more exact text replacements to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"old_text", {{"type", "string"}, {"description", "Exact text to find; must be unique in the file and must not overlap with other edits"}}},
                                    {"new_text", {{"type", "string"}, {"description", "Text to replace old_text with"}}},
                                }},
                                {"required", json::array({"old_text", "new_text"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "edits"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string path = params.at("path").get<std::string>();
        const json & edits_json = params.at("edits");

        if (!edits_json.is_array() || edits_json.empty()) {
            return {{"error", "\"edits\" must be a non-empty array"}};
        }

        struct edit_req {
            std::string old_text;
            std::string new_text;
        };
        std::vector<edit_req> edits;
        edits.reserve(edits_json.size());
        for (const auto & e : edits_json) {
            edit_req er;
            er.old_text = e.at("old_text").get<std::string>();
            er.new_text = e.at("new_text").get<std::string>();
            if (er.old_text.empty()) {
                return {{"error", string_format("edits[%zu].old_text must not be empty", edits.size())}};
            }
            edits.push_back(std::move(er));
        }

        auto io = make_tools_io(params);
        std::string original_content;
        if (!io->read_file(path, original_content)) {
            return {{"error", "failed to open file: " + path}};
        }

        // does any old_text need fuzzy matching (no exact match found)?
        bool any_fuzzy = false;
        for (size_t i = 0; i < edits.size(); i++) {
            if (original_content.find(edits[i].old_text) != std::string::npos) continue;
            std::string fuzzy_content = normalize_for_fuzzy_match(original_content);
            std::string fuzzy_old     = normalize_for_fuzzy_match(edits[i].old_text);
            if (fuzzy_content.find(fuzzy_old) == std::string::npos) {
                return {{"error", string_format(
                    "could not find edits[%zu].old_text in %s, it must match the file's current content exactly",
                    i, path.c_str())}};
            }
            any_fuzzy = true;
        }

        std::string base_content = any_fuzzy ? normalize_for_fuzzy_match(original_content) : original_content;

        // uniqueness check always uses fuzzy-normalized text, so a whitespace-only duplicate still counts
        std::vector<matched_edit> matched;
        matched.reserve(edits.size());
        for (size_t i = 0; i < edits.size(); i++) {
            std::string needle = any_fuzzy ? normalize_for_fuzzy_match(edits[i].old_text) : edits[i].old_text;
            size_t occurrences = count_occurrences(
                normalize_for_fuzzy_match(original_content),
                normalize_for_fuzzy_match(edits[i].old_text));
            if (occurrences > 1) {
                return {{"error", string_format(
                    "found %zu occurrences of edits[%zu].old_text in %s, it must be unique",
                    occurrences, i, path.c_str())}};
            }
            size_t idx = base_content.find(needle);
            matched.push_back({i, idx, needle.size(), edits[i].new_text});
        }

        std::sort(matched.begin(), matched.end(), [](const matched_edit & a, const matched_edit & b) {
            return a.match_index < b.match_index;
        });
        for (size_t i = 1; i < matched.size(); i++) {
            if (matched[i - 1].match_index + matched[i - 1].match_length > matched[i].match_index) {
                return {{"error", string_format(
                    "edits[%zu] and edits[%zu] overlap in %s; merge them into one edit or target disjoint regions",
                    matched[i - 1].edit_index, matched[i].edit_index, path.c_str())}};
            }
        }

        std::string new_content = any_fuzzy
            ? apply_replacements_preserving_unchanged_lines(original_content, base_content, matched)
            : apply_replacements(base_content, matched, 0);

        if (new_content == original_content) {
            return {{"error", "no changes made: the replacement(s) produced identical content"}};
        }

        if (!io->write_file(path, new_content)) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"edits_applied", (int) matched.size()}};
    }

private:
    // strip trailing whitespace, normalize smart quotes/dashes/spaces to ASCII
    static std::string normalize_line_for_fuzzy_match(const std::string & line) {
        size_t end = line.size();
        while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\t' || line[end - 1] == '\r')) {
            end--;
        }
        std::string s = line.substr(0, end);

        auto replace_all = [](std::string & str, const std::string & from, const std::string & to) {
            if (from.empty()) return;
            size_t pos = 0;
            while ((pos = str.find(from, pos)) != std::string::npos) {
                str.replace(pos, from.size(), to);
                pos += to.size();
            }
        };

        // smart single quotes -> '
        for (unsigned char b : {0x98, 0x99, 0x9A, 0x9B}) {
            replace_all(s, std::string("\xE2\x80") + (char) b, "'");
        }
        // smart double quotes -> "
        for (unsigned char b : {0x9C, 0x9D, 0x9E, 0x9F}) {
            replace_all(s, std::string("\xE2\x80") + (char) b, "\"");
        }
        // various dashes -> -
        for (unsigned char b = 0x90; b <= 0x95; b++) {
            replace_all(s, std::string("\xE2\x80") + (char) b, "-");
        }
        replace_all(s, "\xE2\x88\x92", "-"); // minus sign
        // special spaces -> ' '
        replace_all(s, "\xC2\xA0", " "); // no-break space
        for (unsigned char b = 0x82; b <= 0x8A; b++) {
            replace_all(s, std::string("\xE2\x80") + (char) b, " ");
        }
        replace_all(s, "\xE2\x80\xAF", " "); // narrow no-break space
        replace_all(s, "\xE2\x81\x9F", " "); // medium mathematical space
        replace_all(s, "\xE3\x80\x80", " "); // ideographic space

        return s;
    }

    // applies the per-line transform above to every line; preserves line count/positions
    static std::string normalize_for_fuzzy_match(const std::string & content) {
        std::string result;
        result.reserve(content.size());
        size_t start = 0;
        while (true) {
            size_t nl = content.find('\n', start);
            bool   is_last = nl == std::string::npos;
            std::string line = is_last ? content.substr(start) : content.substr(start, nl - start);
            result += normalize_line_for_fuzzy_match(line);
            if (is_last) break;
            result += '\n';
            start = nl + 1;
        }
        return result;
    }

    // lines with trailing '\n' kept, so untouched ones can be reconstructed verbatim
    static std::vector<std::string> split_lines_with_endings(const std::string & content) {
        std::vector<std::string> lines;
        size_t start = 0;
        while (start < content.size()) {
            size_t nl = content.find('\n', start);
            if (nl == std::string::npos) {
                lines.push_back(content.substr(start));
                break;
            }
            lines.push_back(content.substr(start, nl - start + 1));
            start = nl + 1;
        }
        return lines;
    }

    struct line_span {
        size_t start;
        size_t end;
    };

    static std::vector<line_span> get_line_spans(const std::string & content) {
        std::vector<line_span> spans;
        size_t offset = 0;
        for (const auto & line : split_lines_with_endings(content)) {
            spans.push_back({offset, offset + line.size()});
            offset += line.size();
        }
        return spans;
    }

    // count non-overlapping occurrences of `needle` in `content`
    static size_t count_occurrences(const std::string & content, const std::string & needle) {
        if (needle.empty()) return 0;
        size_t count = 0, pos = 0;
        while ((pos = content.find(needle, pos)) != std::string::npos) {
            count++;
            pos += needle.size();
        }
        return count;
    }

    struct matched_edit {
        size_t      edit_index;
        size_t      match_index;  // offset into the "base content" (see below)
        size_t      match_length;
        std::string new_text;
    };

    // replacements must be sorted ascending by match_index and non-overlapping
    static std::string apply_replacements(
            const std::string & content,
            const std::vector<matched_edit> & replacements,
            size_t offset) {
        std::string result = content;
        for (auto it = replacements.rbegin(); it != replacements.rend(); ++it) {
            size_t local_index = it->match_index - offset;
            result = result.substr(0, local_index) + it->new_text + result.substr(local_index + it->match_length);
        }
        return result;
    }

    // widen a replacement's byte range to the line(s) of `lines` it touches
    static bool get_replacement_line_range(
            const std::vector<line_span> & lines,
            size_t match_index, size_t match_length,
            size_t & out_start_line, size_t & out_end_line /* exclusive */) {
        size_t replacement_start = match_index;
        size_t replacement_end   = match_index + match_length;

        size_t start_line = (size_t) -1;
        for (size_t i = 0; i < lines.size(); i++) {
            if (replacement_start >= lines[i].start && replacement_start < lines[i].end) {
                start_line = i;
                break;
            }
        }
        if (start_line == (size_t) -1) return false;

        size_t end_line = start_line;
        while (end_line < lines.size() && lines[end_line].end < replacement_end) {
            end_line++;
        }
        if (end_line >= lines.size()) return false;

        out_start_line = start_line;
        out_end_line   = end_line + 1;
        return true;
    }

    // like apply_replacements, but untouched lines come from `original_content`
    static std::string apply_replacements_preserving_unchanged_lines(
            const std::string & original_content,
            const std::string & base_content,
            const std::vector<matched_edit> & replacements /* ascending, non-overlapping */) {
        auto original_lines = split_lines_with_endings(original_content);
        auto base_lines      = get_line_spans(base_content);

        struct group {
            size_t start_line;
            size_t end_line; // exclusive
            std::vector<matched_edit> reps;
        };
        std::vector<group> groups;

        for (const auto & rep : replacements) {
            size_t start_line = 0, end_line = 0;
            get_replacement_line_range(base_lines, rep.match_index, rep.match_length, start_line, end_line);
            if (!groups.empty() && start_line < groups.back().end_line) {
                groups.back().end_line = std::max(groups.back().end_line, end_line);
                groups.back().reps.push_back(rep);
            } else {
                groups.push_back({start_line, end_line, {rep}});
            }
        }

        size_t original_line_index = 0;
        std::string result;
        for (auto & g : groups) {
            for (size_t i = original_line_index; i < g.start_line; i++) {
                result += original_lines[i];
            }

            size_t group_start_offset = base_lines[g.start_line].start;
            size_t group_end_offset   = base_lines[g.end_line - 1].end;
            std::string slice = base_content.substr(group_start_offset, group_end_offset - group_start_offset);
            result += apply_replacements(slice, g.reps, group_start_offset);

            original_line_index = g.end_line;
        }
        for (size_t i = original_line_index; i < original_lines.size(); i++) {
            result += original_lines[i];
        }

        return result;
    }
};

//
// get_datetime: returns the current date and time
//

struct server_tool_get_datetime : server_tool {
    server_tool_get_datetime() {
        name = "get_datetime";
        display_name = "Get Date & Time";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns the current date and time in UTC"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"format", {
                            {"type", "string"},
                            {"description",
                                "strftime()-style format string for the output (default: \"%Y-%m-%dT%H:%M:%SZ\", "
                                "e.g. ISO 8601). Choose your own format if you need something else, "
                                "e.g. \"%A, %B %d %Y\" for a human-readable date."},
                        }},
                    }},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string format = json_value(params, "format", std::string("%Y-%m-%dT%H:%M:%SZ"));

        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_utc;
#ifdef _WIN32
        gmtime_s(&tm_utc, &time);
#else
        gmtime_r(&time, &tm_utc);
#endif

        char buf[256];
        size_t len = std::strftime(buf, sizeof(buf), format.c_str(), &tm_utc);
        if (len == 0) {
            return {{"error", "invalid format string"}};
        }

        return {{"result", std::string(buf, len)}};
    }
};

//
// get_info: returns runtime info (OS name/version and cwd)
//

static constexpr size_t SERVER_TOOL_GET_INFO_MAX_OUTPUT = 4096;
static constexpr int    SERVER_TOOL_GET_INFO_TIMEOUT    = 5; // seconds

struct server_tool_get_info : server_tool {
    server_tool_get_info() {
        name = "get_info";
        display_name = "Get Runtime Info";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns runtime info: the OS name/version and the current working directory"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", json::object()},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        auto io = make_tools_io(params);

#ifdef _WIN32
        auto res = io->run({"cmd", "/c", "ver"}, SERVER_TOOL_GET_INFO_MAX_OUTPUT, SERVER_TOOL_GET_INFO_TIMEOUT);
#else
        auto res = io->run({"uname", "-a"}, SERVER_TOOL_GET_INFO_MAX_OUTPUT, SERVER_TOOL_GET_INFO_TIMEOUT);
#endif
        // "ver" prints a blank line before the version, so the output is stripped on both ends;
        // a failed spawn or a timeout leaves a diagnostic in res.output, which is not an OS name
        std::string os_info = res.exit_code == 0 && !res.timed_out ? string_strip(res.output) : "unknown";

        std::string cwd = json_value(params, "cwd", std::string());
        if (cwd.empty()) {
            std::error_code ec;
            cwd = fs::current_path(ec).string();
        }

        return {
            {"os",  os_info},
            {"cwd", cwd},
        };
    }
};

struct server_tool_stream_result : server_task_result {
    std::string chunk;
    bool done = false;
    std::string error_msg;

    json to_json() override {
        if (!done) {
            return {{"chunk", chunk}};
        } else {
            json result = {{"done", true}};
            if (!error_msg.empty()) {
                result["error"] = error_msg;
            }
            return result;
        }
    }
};

void server_tool::stream::push(const std::string & chunk) {
    if (chunk.empty()) return;
    auto r = std::make_unique<server_tool_stream_result>();
    r->id    = id;
    r->chunk = chunk;
    qr.send(std::move(r));
}

struct server_tools_res : server_http_res {
    std::thread worker;
    server_response * qr = nullptr; // set only for streaming responses
    int id = -1;

    ~server_tools_res() override {
        if (worker.joinable()) {
            worker.join();
        }
        if (qr) {
            qr->remove_waiting_task_id(id);
        }
    }
};

//
// server_mcp_tool: exposes one tool from a running MCP server as a server_tool.
//
struct server_mcp_tool : server_tool {
    std::string server_name;
    std::string tool_name;
    server_mcp_tool_def def;
    server_mcp & mcp_mgr;

    server_mcp_tool(server_mcp_tool_def d, server_mcp & mgr)
        : server_name(d.server_name)
        , tool_name(d.name)
        , def(std::move(d))
        , mcp_mgr(mgr)
    {
        name = server_name + "_" + tool_name;
        display_name = name;
        permission_write = false;
        support_stream = false;
    }

    std::string type() const override { return "mcp"; }

    json get_definition() const override {
        json schema = def.input_schema;
        if (schema.is_null() || !schema.is_object()) {
            schema = json::object();
        }
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", def.description},
                {"parameters", schema},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        return mcp_mgr.call_tool(server_name, tool_name, params);
    }
};

static server_tool & find_tool(std::vector<std::unique_ptr<server_tool>> & tools, const std::string & name, bool require_stream) {
    for (auto & t : tools) {
        if (t->name == name) {
            if (require_stream && !t->support_stream) {
                throw std::invalid_argument(string_format("tool \"%s\" does not support stream = true", name.c_str()));
            }
            return *t;
        }
    }
    throw std::invalid_argument(string_format("unknown tool \"%s\"", name.c_str()));
}

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools() {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_get_datetime>());
    tools.push_back(std::make_unique<server_tool_get_info>());
    return tools;
}

static std::string str_to_lower(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static std::string get_header(const std::map<std::string, std::string> & headers, const std::string & key, std::string default_value = "") {
    const auto lowered_key = str_to_lower(key);
    for (const auto & h : headers) {
        if (str_to_lower(h.first) == lowered_key) {
            return h.second;
        }
    }
    return default_value;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools,
                         server_mcp & mcp_mgr) {
    if (!enabled_tools.empty()) {
        if (!common_subproc::is_supported()) {
            throw std::runtime_error("subprocess is not enabled on this build");
        }

        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools();

        // collect all known tool names for validation
        std::vector<std::string> known_names;
        known_names.reserve(all_tools.size());
        for (const auto & t : all_tools) {
            known_names.push_back(t->name);
        }

        // validate that every requested tool is known
        for (const auto & name : enabled_tools) {
            if (name == "all") continue;
            if (std::find(known_names.begin(), known_names.end(), name) == known_names.end()) {
                throw std::runtime_error(string_format(
                    "unknown tool \"%s\". available tools: %s",
                    name.c_str(),
                    string_join(known_names, ", ").c_str()));
            }
        }

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    // append MCP tools, skipping any that collide with a built-in or another MCP tool of the same "<server>_<tool>" name
    if (!mcp_mgr.empty()) {
        std::unordered_set<std::string> seen_names;
        for (auto & t : tools) {
            seen_names.insert(t->name);
        }
        size_t n_added = 0;
        for (const auto & def : mcp_mgr.list_tools()) {
            std::string mcp_name = def.server_name + "_" + def.name;
            if (seen_names.count(mcp_name)) {
                SRV_WRN("MCP tool \"%s\" from server \"%s\" collides with an existing tool, skipping\n",
                    mcp_name.c_str(), def.server_name.c_str());
                continue;
            }
            seen_names.insert(mcp_name);
            tools.push_back(std::make_unique<server_mcp_tool>(def, mcp_mgr));
            n_added++;
        }
        if (n_added > 0) {
            SRV_INF("Added %zu MCP tools\n", n_added);
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_tools_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            bool stream = body.value("stream", false);

            // accept x-tool-cwd header to override of the process
            auto cwd = get_header(req.headers, "x-tool-cwd");
            if (!cwd.empty()) {
                params["cwd"] = cwd;
            }

            server_tool & tool = find_tool(tools, tool_name, stream);

            if (stream) {
                int id = res_id.fetch_add(1);
                queue_res.add_waiting_task_id(id);
                res->qr = &queue_res;
                res->id = id;

                res->worker = std::thread([this, id, &req, &tool, params]() mutable {
                    server_tool::stream st{queue_res, id, [&req]() {
                        return !req.should_stop();
                    }};

                    auto done = std::make_unique<server_tool_stream_result>();
                    try {
                        tool.invoke(params, &st);
                    } catch (const std::exception & e) {
                        done->error_msg = e.what();
                    } catch (...) {
                        done->error_msg = "An unknown error occurred";
                    }
                    done->id    = st.id;
                    done->done  = true;
                    st.qr.send(std::move(done));
                });

                res->content_type = "text/event-stream";
                res->status = 200;
                res->next   = [this, id](std::string & output) -> bool {
                    auto result = queue_res.recv(id);
                    auto * r = dynamic_cast<server_tool_stream_result *>(result.get());
                    GGML_ASSERT(r != nullptr);
                    output = "data: " + safe_json_to_str(r->to_json()) + "\n\n";
                    if (r->done) {
                        queue_res.remove_waiting_task_id(id);
                        return false;
                    }
                    return true;
                };
            } else {
                json result = tool.invoke(params, nullptr);
                res->status = 200;
                res->data   = safe_json_to_str(result);
            }
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::invalid_argument & e) {
            res->status = 404;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}
