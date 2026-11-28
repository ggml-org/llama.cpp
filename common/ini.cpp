#include "ini.h"
#include "common.h"
#include "log.h"
#include "peg-parser.h"
#include "preset.h"

#include <fstream>
#include <string_view>

static std::vector<std::string_view> ini_lines(const std::string & contents) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    while (start < contents.size()) {
        const size_t end = contents.find_first_of("\r\n", start);
        if (end == std::string::npos) {
            lines.emplace_back(contents.data() + start, contents.size() - start);
            break;
        }
        lines.emplace_back(contents.data() + start, end - start);
        start = end + (contents[end] == '\r' && end + 1 < contents.size() && contents[end + 1] == '\n' ? 2 : 1);
    }
    return lines;
}

size_t ini_line_number(const std::string & contents, size_t offset) {
    size_t line = 1;
    for (size_t i = 0; i < offset; ++i) {
        if (contents[i] == '\r' || (contents[i] == '\n' && (i == 0 || contents[i - 1] != '\r'))) {
            ++line;
        }
    }
    return line;
}

std::string read_ini_file(const std::filesystem::path & path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("preset file does not exist: " + path.string());
    }
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server preset file: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

struct ini_version {
    std::string name = "legacy";
    size_t line = 0;
};

static ini_version get_ini_version(const std::string & contents, const std::filesystem::path & path) {
    ini_version version;
    size_t line_number = 0;
    for (const auto line : ini_lines(contents)) {
        ++line_number;
        if (!line.empty() && line.front() == '[') {
            break;
        }
        if (line.empty() || line.front() == ' ' || line.front() == '\t') {
            continue;
        }
        const size_t equals = line.find('=');
        if (equals != std::string_view::npos && string_strip(std::string(line.substr(0, equals))) == "version") {
            if (version.line != 0) {
                throw std::runtime_error("version may only be declared once at line " + std::to_string(line_number) + ": " + path.string());
            }
            const auto value = line.substr(equals + 1);
            // Version detection accepts legacy inline comments to allow for parsing legacy format.
            version.name = string_strip(std::string(value.substr(0, value.find_first_of(";#"))));
            version.line = line_number;
        }
    }
    return version;
}

std::string ini_version_from_file(const std::filesystem::path & path) {
    const auto version = get_ini_version(read_ini_file(path), path);
    if (version.name != "legacy" && version.name != "1") {
        throw std::runtime_error("unsupported INI version: " + version.name + " at line " + std::to_string(version.line) + ": " + path.string());
    }
    return version.name;
}

void warn_legacy_ini(const std::string & contents, const std::filesystem::path & path) {
    if (get_ini_version(contents, path).line != 0) {
        return;
    }
    const auto lines = ini_lines(contents);
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto line = lines[i];
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string_view::npos || line[first] == ';' || line[first] == '#') {
            continue;
        }
        const size_t value_start = line.find(line[first] == '[' ? ']' : '=', first);
        if (value_start != std::string_view::npos && line.find_first_of(";#", value_start + 1) != std::string_view::npos) {
            LOG_WRN("%s:%zu: inline comments use the legacy INI parser; the default parser will change in the future. Move comments to separate lines and add 'version = 1' before any section, or add 'version = legacy' to keep the old behavior.\n", path.string().c_str(), i + 1);
        }
    }
}

std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file_v1(const std::filesystem::path & path) {
    const std::string contents = read_ini_file(path);
    std::map<std::string, std::map<std::string, std::string>> parsed;

    static const auto parser = build_peg_parser([](auto & p) {
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));
        auto eol = p.rule("eol", ws + (newline | p.end()));
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol) + p.any()));
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);
        auto indent = p.rule("indent", p.chars("[ ]", 1, -1));
        auto kv_line = p.rule("kv-line", p.optional(indent) + p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);
        auto continuation_line = p.rule("continuation-line", p.tag("continuation", indent + value + eol));
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));
        auto entry = p.tag("entry", kv_line | continuation_line);
        auto line = p.rule("line", header_line | comment_line | blank_line | entry);
        return p.rule("ini", p.zero_or_more(line));
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success() || result.end != contents.size()) {
        throw std::runtime_error("failed to parse server config file at line " + std::to_string(ini_line_number(contents, result.end)) + ": " + path.string());
    }

    std::string current_section = COMMON_PRESET_DEFAULT_NAME;
    std::string current_key;
    size_t current_indent = 0;
    bool skip_entry = false;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            current_section = std::string(node.text);
            parsed[current_section] = {};
            current_key.clear();
        } else if (node.tag == "entry") {
            const size_t indentation = node.text.find_first_not_of(' ');
            skip_entry = !current_key.empty() && indentation > current_indent;
            if (skip_entry) {
                const size_t start = current_indent + 1;
                const size_t end = node.text.find_last_not_of(" \t\r\n");
                parsed[current_section][current_key] += "\n" + std::string(node.text.substr(start, end - start + 1));
            } else {
                current_key.clear();
                current_indent = indentation;
            }
        } else if (node.tag == "continuation" && !skip_entry) {
            throw std::runtime_error("continuation line must be indented deeper than its key at line " + std::to_string(ini_line_number(contents, node.start)) + ": " + path.string());
        } else if (node.tag == "key" && !skip_entry) {
            current_key = std::string(node.text);
        } else if (node.tag == "value" && !skip_entry && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
        }
    });
    return parsed;
}
