// llama-ui-embed: generate ui.cpp / ui.h that embed UI assets as C arrays.
//
// Usage:
//   llama-ui-embed <out_cpp> <out_h> <asset_dir>
//
// Recursively embeds every regular file under <asset_dir>.
// Asset names are relative paths from <asset_dir> (e.g. "_app/immutable/bundle.HASH.js").
// Without <asset_dir>, emits an empty asset table.

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>


static const char * mime_from_ext(const std::string & name) {
    auto ext = name.rfind('.');
    if (ext == std::string::npos) return "application/octet-stream";
    std::string e = name.substr(ext + 1);
    if (e == "html")        return "text/html; charset=utf-8";
    if (e == "css")         return "text/css";
    if (e == "js")          return "application/javascript";
    if (e == "json")        return "application/json";
    if (e == "webmanifest") return "application/manifest+json";
    if (e == "svg")         return "image/svg+xml";
    if (e == "png")         return "image/png";
    if (e == "jpg" ||
        e == "jpeg")        return "image/jpeg";
    if (e == "ico")         return "image/x-icon";
    if (e == "woff")        return "font/woff";
    if (e == "woff2")       return "font/woff2";
    return "application/octet-stream";
}

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool read_file(const std::filesystem::path & path, std::vector<unsigned char> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "embed: cannot open %s\n", path.string().c_str());
        return false;
    }
    const auto sz = f.tellg();
    if (sz < 0) {
        return false;
    }
    f.seekg(0);
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && !f.read(reinterpret_cast<char *>(out.data()), sz)) {
        return false;
    }
    return true;
}

static void append_bytes_hex(std::string & out, const std::vector<unsigned char> & bytes) {
    static const char hex[] = "0123456789abcdef";
    out.reserve(out.size() + bytes.size() * 5);
    for (unsigned char b : bytes) {
        out += '0';
        out += 'x';
        out += hex[b >> 4];
        out += hex[b & 0xf];
        out += ',';
    }
}

static bool write_if_different(const std::string & path, const std::string & content) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (f) {
        const auto sz = f.tellg();
        if (sz >= 0 && static_cast<size_t>(sz) == content.size()) {
            std::string existing(static_cast<size_t>(sz), '\0');
            f.seekg(0);
            if (sz == 0 || f.read(existing.data(), sz)) {
                if (existing == content) {
                    return true;
                }
            }
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "embed: cannot write %s\n", path.c_str());
        return false;
    }
    if (!content.empty()) {
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    bool ok = out.good();
    if (ok) {
        printf("embed: write output file %s\n", path.c_str());
    }
    return ok;
}

static std::string path_basename(const std::string & name) {
    const size_t p = name.rfind('/');
    return p == std::string::npos ? name : name.substr(p + 1);
}
static bool str_starts_with(const std::string & s, const char * prefix) {
    const size_t n = strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}
static bool str_ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static std::string fmt(const char * pattern, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, pattern);
    const int n = vsnprintf(tmp, sizeof(tmp), pattern, ap);
    va_end(ap);
    return (n > 0) ? std::string(tmp, static_cast<size_t>(n)) : std::string();
}

struct asset_entry {
    std::string           name;
    std::filesystem::path path;
};

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <out_cpp> <out_h> [<asset_dir>]\n", argv[0]);
        return 1;
    }

    const std::string out_cpp   = argv[1];
    const std::string out_h     = argv[2];
    const std::string asset_dir = argv[3];

    const bool        use_gzip = std::filesystem::exists(asset_dir + "/_gzip");
    const std::string in_dir   = use_gzip ? (asset_dir + "/_gzip") : asset_dir;

    std::vector<asset_entry> assets;
    if (!in_dir.empty()) {
        const std::filesystem::path dir = in_dir;

        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(dir, ec);
        if (ec) {
            fprintf(stderr, "embed: cannot iterate %s: %s\n", argv[3], ec.message().c_str());
            return 1;
        }
        for (const auto & entry : it) {
            if (!entry.is_regular_file()) {
                continue;
            }
            // name is the relative path from dir, with forward slashes
            const std::string name = entry.path().lexically_relative(dir).generic_string();
            assets.push_back({ name, entry.path() });
        }

        // directory iteration order is unspecified; sort for reproducible output
        std::sort(assets.begin(), assets.end(),
                  [](const asset_entry & a, const asset_entry & b) { return a.name < b.name; });
    }

    const int n_assets = static_cast<int>(assets.size());

    if (n_assets > 0) {
        using match_fn = std::function<bool(const std::string &)>;
        auto exact = [](const char * name) -> match_fn {
            return [name](const std::string & base) { return base == name; };
        };

        struct required_check { const char * label; match_fn match; bool found; };
        required_check checks[] = {
            { "index.html",           exact("index.html"),           false },
            { "loading.html",         exact("loading.html"),         false },
            { "manifest.webmanifest", exact("manifest.webmanifest"), false },
            { "sw.js",                exact("sw.js"),                false },
            { "build.json",           exact("build.json"),           false },
            { "version.json",         exact("version.json"),         false },
            { "bundle[hash].js",      [](const std::string & b) {
                return str_starts_with(b, "bundle")  && str_ends_with(b, ".js");
            }, false },
            { "bundle[hash].css",     [](const std::string & b) {
                return str_starts_with(b, "bundle")  && str_ends_with(b, ".css");
            }, false },
            { "workbox[hash].js",     [](const std::string & b) {
                return str_starts_with(b, "workbox") && str_ends_with(b, ".js");
            }, false },
        };

        for (const auto & a : assets) {
            const std::string base = path_basename(a.name);
            for (auto & c : checks) {
                if (!c.found) { c.found = c.match(base); }
            }
        }

        std::vector<const char *> missing;
        for (const auto & c : checks) {
            if (!c.found) { missing.push_back(c.label); }
        }
        if (!missing.empty()) {
            fprintf(stderr, "\ncurrent asset files:\n");
            for (const auto & a : assets) {
                fprintf(stderr, "    %s\n", a.name.c_str());
            }
            fprintf(stderr, "missing required asset(s):\n");
            for (const char * m : missing) {
                fprintf(stderr, "    %s\n", m);
            }
            fprintf(stderr, "hint: try cleaning your build directory: %s\n", in_dir.c_str());
            return 1;
        }
    }

    std::string h;
    h += "#pragma once\n\n#include <array>\n#include <string>\n\n";
    if (n_assets > 0) {
        h += "#define LLAMA_UI_HAS_ASSETS 1\n\n";
    }
    h +=
        "struct llama_ui_asset {\n"
        "    std::string           name;\n"
        "    const unsigned char * data;\n"
        "    std::size_t           size;\n"
        "    std::string           etag;\n"
        "    std::string           type;\n"
        "};\n\n"
        "const llama_ui_asset * llama_ui_find_asset(const std::string & name);\n"
        "bool llama_ui_use_gzip();\n";
    h += fmt("const std::array<llama_ui_asset, %d> & llama_ui_get_assets();\n", n_assets);

    std::string cpp;
    cpp += "#include \"ui.h\"\n\n";

    if (n_assets > 0) {
        for (int i = 0; i < n_assets; i++) {
            std::vector<unsigned char> bytes;
            if (!read_file(assets[i].path, bytes)) {
                return 1;
            }
            if (bytes.empty()) {
                fprintf(stderr, "embed: empty file: %s\n", assets[i].path.generic_string().c_str());
                return 1;
            }
            cpp += fmt("static const unsigned char asset_%d_data[] = {", i);
            append_bytes_hex(cpp, bytes);
            const auto hash = fnv_hash(bytes.data(), bytes.size());

            cpp += fmt("};\nstatic const std::size_t   asset_%d_size = %zu;\n",
                       i, bytes.size());
            cpp += fmt("static const char          asset_%d_etag[] = \"\\\"0x%016" PRIx64 "\\\"\";\n\n",
                       i, hash);
        }

        cpp += fmt("static const std::array<llama_ui_asset, %d> g_assets = {{\n", n_assets);
        for (int i = 0; i < n_assets; i++) {
            const std::string & name = assets[i].name;
            cpp += fmt("    { \"%s\", asset_%d_data, asset_%d_size, asset_%d_etag, \"%s\" },\n",
                       name.c_str(), i, i, i, mime_from_ext(name));
        }
        cpp += "}};\n\n";

        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const std::string & name) {\n"
            "    for (const auto & a : g_assets) {\n"
            "        if (a.name == name) {\n"
            "            return &a;\n"
            "        }\n"
            "    }\n"
            "    return nullptr;\n"
            "}\n";
        cpp += fmt("const std::array<llama_ui_asset, %d> & llama_ui_get_assets() {\n", n_assets);
        cpp +=     "    return g_assets;\n"
                   "}\n";
    } else {
        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const std::string &) {\n"
            "    return nullptr;\n"
            "}\n"
            "const std::array<llama_ui_asset, 0> & llama_ui_get_assets() {\n"
            "    static const std::array<llama_ui_asset, 0> empty{};\n"
            "    return empty;\n"
            "}\n";
    }
    cpp += fmt("bool llama_ui_use_gzip() { return %s; }\n", use_gzip ? "true" : "false");

    bool ok = true;
    ok = write_if_different(out_h,   h)   && ok;
    ok = write_if_different(out_cpp, cpp) && ok;
    return ok ? 0 : 1;
}
