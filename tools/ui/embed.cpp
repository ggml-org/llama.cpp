// llama-ui-embed: generate ui.cpp / ui.h that embed UI assets as C arrays.
//
// Usage:
//   llama-ui-embed <out_cpp> <out_h> [<asset_dir>]
//
// Embeds every regular file under <asset_dir>. Asset names are the
// dir-relative paths with forward slashes. Without <asset_dir>, emits an
// empty asset table.

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
    return out.good();
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

    const std::string out_cpp = argv[1];
    const std::string out_h   = argv[2];

    std::vector<asset_entry> assets;
    if (argc == 4) {
        const std::filesystem::path dir = argv[3];

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
            assets.push_back({ entry.path().lexically_relative(dir).generic_string(), entry.path() });
        }

        // directory iteration order is unspecified; sort for reproducible output
        std::sort(assets.begin(), assets.end(),
                  [](const asset_entry & a, const asset_entry & b) { return a.name < b.name; });
    }

    const int n_assets = static_cast<int>(assets.size());

    std::string h;
    h += "#pragma once\n\n#include <stddef.h>\n\n";
    if (n_assets > 0) {
        h += "#define LLAMA_UI_HAS_ASSETS 1\n\n";
    }
    h +=
        "struct llama_ui_asset {\n"
        "    const char *          name;\n"
        "    const unsigned char * data;\n"
        "    size_t                size;\n"
        "    const char *          etag;\n"
        "};\n\n"
        "const llama_ui_asset * llama_ui_find_asset(const char * name);\n";

    std::string cpp;
    cpp += "#include \"ui.h\"\n\n#include <string.h>\n\n";

    if (n_assets > 0) {
        for (int i = 0; i < n_assets; i++) {
            std::vector<unsigned char> bytes;
            if (!read_file(assets[i].path, bytes)) {
                return 1;
            }
            cpp += fmt("static const unsigned char asset_%d_data[] = {", i);
            append_bytes_hex(cpp, bytes);
            const auto hash = fnv_hash(bytes.data(), bytes.size());

            cpp += fmt("};\nstatic const size_t        asset_%d_size = %zu;\n",
                       i, bytes.size());
            cpp += fmt("static const char        asset_%d_etag[] = \"\\\"0x%016" PRIx64 "\\\"\";\n\n",
                       i, hash);
        }

        cpp += "static const llama_ui_asset g_assets[] = {\n";
        for (int i = 0; i < n_assets; i++) {
            cpp += fmt("    { \"%s\", asset_%d_data, asset_%d_size, asset_%d_etag },\n",
                       assets[i].name.c_str(), i, i, i);
        }
        cpp += "};\n\n";

        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const char * name) {\n"
            "    for (const auto & a : g_assets) {\n"
            "        if (strcmp(a.name, name) == 0) {\n"
            "            return &a;\n"
            "        }\n"
            "    }\n"
            "    return nullptr;\n"
            "}\n";
    } else {
        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const char *) {\n"
            "    return nullptr;\n"
            "}\n";
    }

    bool ok = true;
    ok = write_if_different(out_h,   h)   && ok;
    ok = write_if_different(out_cpp, cpp) && ok;
    return ok ? 0 : 1;
}
