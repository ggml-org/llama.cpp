#include "clipboard-image.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr size_t MAX_CLIPBOARD_BYTES = 50 * 1024 * 1024;  // 50 MB

// Read all output from a popen command into a byte vector.
// Returns empty vector on failure.
static std::vector<uint8_t> popen_read(const char * cmd, size_t max_bytes = MAX_CLIPBOARD_BYTES) {
    std::vector<uint8_t> result;
    FILE * fp = popen(cmd, "r");
    if (!fp) {
        return result;
    }

    char buf[8192];
    while (!feof(fp)) {
        size_t n = fread(buf, 1, sizeof(buf), fp);
        if (n > 0) {
            result.insert(result.end(), buf, buf + n);
            if (result.size() > max_bytes) {
                pclose(fp);
                return {};
            }
        }
    }
    pclose(fp);
    return result;
}

#if defined(__APPLE__)

// macOS: use JavaScript for Automation (JXA) via osascript to read clipboard image.
// JXA can access NSPasteboard through the ObjC bridge and output base64 to stdout.
// Output format: "<mime>:<base64>" or "NONE" if no image.
std::optional<clipboard_image> clipboard_read_image() {
    // Pass script via heredoc to avoid shell quoting issues with single quotes
    static const char * jxa_script =
        "ObjC.import('AppKit');\n"
        "ObjC.import('Foundation');\n"
        "var pb = $.NSPasteboard.generalPasteboard;\n"
        "var types = pb.types;\n"
        "var arr = [];\n"
        "for (var i = 0; i < types.count; i++) {\n"
        "  arr.push(ObjC.unwrap(types.objectAtIndex(i)));\n"
        "}\n"
        "var preferred = [\n"
        "  ['public.png', 'image/png'],\n"
        "  ['public.jpeg', 'image/jpeg'],\n"
        "  ['com.compuserve.gif', 'image/gif'],\n"
        "  ['org.webmproject.webp', 'image/webp']\n"
        "];\n"
        "var found = null;\n"
        "for (var j = 0; j < preferred.length; j++) {\n"
        "  if (arr.indexOf(preferred[j][0]) >= 0) { found = preferred[j]; break; }\n"
        "}\n"
        "if (!found) { 'NONE'; } else {\n"
        "  var data = pb.dataForType(found[0]);\n"
        "  if (!data || data.isNil()) { 'NONE'; } else {\n"
        "    found[1] + ':' + ObjC.unwrap(data.base64EncodedStringWithOptions(0));\n"
        "  }\n"
        "}\n";

    std::string cmd = "osascript -l JavaScript 2>/dev/null <<'LLAMA_JXA_EOF'\n";
    cmd += jxa_script;
    cmd += "LLAMA_JXA_EOF";

    auto output = popen_read(cmd.c_str());
    if (output.empty()) {
        return std::nullopt;
    }

    // Trim trailing whitespace
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
        output.pop_back();
    }

    std::string str(output.begin(), output.end());
    if (str == "NONE" || str.empty()) {
        return std::nullopt;
    }

    // Parse "<mime>:<base64>"
    size_t colon = str.find(':');
    if (colon == std::string::npos || colon == 0) {
        return std::nullopt;
    }

    std::string mime = str.substr(0, colon);
    std::string b64  = str.substr(colon + 1);
    if (b64.empty()) {
        return std::nullopt;
    }

    std::string decoded;
    try {
        decoded = base64::decode(b64);
    } catch (...) {
        return std::nullopt;
    }
    if (decoded.empty()) {
        return std::nullopt;
    }

    clipboard_image img;
    img.bytes.assign(decoded.begin(), decoded.end());
    img.mime_type = std::move(mime);
    return img;
}

#elif defined(__linux__)

static bool is_wayland() {
    const char * wayland = std::getenv("WAYLAND_DISPLAY");
    if (wayland && wayland[0] != '\0') {
        return true;
    }
    const char * session = std::getenv("XDG_SESSION_TYPE");
    return session && std::strcmp(session, "wayland") == 0;
}

static const char * select_image_type(const std::vector<uint8_t> & types_output) {
    std::string types(types_output.begin(), types_output.end());
    // Check in preference order
    static const char * preferred[] = {"image/png", "image/jpeg", "image/webp", "image/gif"};
    for (const char * mime : preferred) {
        if (types.find(mime) != std::string::npos) {
            return mime;
        }
    }
    return nullptr;
}

static std::optional<clipboard_image> read_via_wl_paste() {
    auto types = popen_read("wl-paste --list-types 2>/dev/null");
    if (types.empty()) {
        return std::nullopt;
    }

    const char * mime = select_image_type(types);
    if (!mime) {
        return std::nullopt;
    }

    std::string cmd = "wl-paste --type ";
    cmd += mime;
    cmd += " --no-newline 2>/dev/null";

    auto bytes = popen_read(cmd.c_str());
    if (bytes.empty()) {
        return std::nullopt;
    }

    return clipboard_image{std::move(bytes), mime};
}

static std::optional<clipboard_image> read_via_xclip() {
    auto types = popen_read("xclip -selection clipboard -t TARGETS -o 2>/dev/null");

    const char * mime = nullptr;
    if (!types.empty()) {
        mime = select_image_type(types);
    }

    if (mime) {
        // TARGETS told us which type to use
        std::string cmd = "xclip -selection clipboard -t ";
        cmd += mime;
        cmd += " -o 2>/dev/null";
        auto bytes = popen_read(cmd.c_str());
        if (!bytes.empty()) {
            return clipboard_image{std::move(bytes), mime};
        }
    } else {
        // TARGETS unavailable — try common types directly
        static const char * try_types[] = {"image/png", "image/jpeg", "image/webp", "image/gif"};
        for (const char * t : try_types) {
            std::string cmd = "xclip -selection clipboard -t ";
            cmd += t;
            cmd += " -o 2>/dev/null";
            auto bytes = popen_read(cmd.c_str());
            if (!bytes.empty()) {
                return clipboard_image{std::move(bytes), t};
            }
        }
    }

    return std::nullopt;
}

std::optional<clipboard_image> clipboard_read_image() {
    if (is_wayland()) {
        auto img = read_via_wl_paste();
        if (img) return img;
        // Fall back to xclip (some Wayland compositors support both)
        return read_via_xclip();
    }
    return read_via_xclip();
}

#else

// Windows / other: not supported yet
std::optional<clipboard_image> clipboard_read_image() {
    return std::nullopt;
}

#endif
