#pragma once
#include <json.hpp>

struct common_healing_marker {
    std::string marker;
    std::string json_dump_marker;
};

struct common_json {
    nlohmann::ordered_json json;
    common_healing_marker healing_marker;
};

bool common_json_parse(
    const std::string & input,
    const std::string & healing_marker,
    common_json & out);

bool common_json_parse(
    std::string::const_iterator & it,
    const std::string::const_iterator & end,
    const std::string & healing_marker,
    common_json & out);
