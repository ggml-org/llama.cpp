#pragma once

#include "json-schema-to-grammar.h"

#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void write_file(const std::string & path, const std::string & text) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::binary);
    out << text;
    require(out.good(), "cannot write " + path);
}

static common_json make_schema(const common_json & tools) {
    common_json variants = common_json::array();
    std::set<std::string> names;
    for (const auto & tool : tools) {
        const auto & fn = tool.at("function");
        require(names.insert(fn.at("name").get<std::string>()).second, "duplicate tool name");
        variants.push_back({
            {"type", "object"},
            {"description", fn.at("description")},
            {"properties", {
                {"name", {{"const", fn.at("name")}}},
                {"arguments", fn.at("parameters")},
            }},
            {"required", common_json::array({"name", "arguments"})},
            {"additionalProperties", false},
        });
    }
    require(!variants.empty(), "tool list is empty");
    return {{"type", "array"}, {"minItems", 1}, {"items", {{"oneOf", variants}}}};
}

static void check_coverage(const common_json & fixture) {
    std::set<std::string> tools, examples;
    for (const auto & tool : fixture.at("tools")) {
        tools.insert(tool.at("function").at("name").get<std::string>());
    }
    for (const auto & call : fixture.at("examples")) {
        examples.insert(call.at("name").get<std::string>());
    }
    require(tools == examples, "examples must cover every tool and contain no unknown tools");
}

static common_json load_fixture(const std::string & path) {
    std::ifstream input(path);
    require(input.good(), "cannot open " + path + " (use --tools from another directory)");
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    common_json fixture = common_json::parse(text);
    check_coverage(fixture);
    return fixture;
}

static int positive_int(const std::string & value) {
    size_t end;
    const int result = std::stoi(value, &end);
    require(end == value.size() && result > 0, "expected a positive integer: " + value);
    return result;
}
