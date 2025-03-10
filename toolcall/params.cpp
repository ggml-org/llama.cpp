
#include "toolcall-params.h"
#include <stdexcept>
#include <json.hpp>

using json = nlohmann::json;

static bool starts_with(const std::string & str, const std::string & prefix) {
    return str.size() >= prefix.size()
        && str.compare(0, prefix.size(), prefix) == 0;
}

toolcall::params::params(std::string tools, std::string choice) {
    this->tools(tools);
    this->choice(choice);
}

void toolcall::params::tools(std::string tools) {
    try {
        if (! tools.empty()) {
            if (starts_with(tools, "http")) {
#ifndef LLAMA_USE_CURL
                throw std::invalid_argument(
                    "Model Context Protocol (MCP) only works when llama.cpp is compiled with libcurl");
#endif
                has_uri_ = true;

            } else {
                json j = json::parse(tools); // Just for early validation
                if (! j.is_array()) {
                    throw std::invalid_argument(
                        "tools must be a valid URL or a JSON array containing tool definitions");
                }
                has_uri_ = false;
            }
        }
        tools_ = std::move(tools);

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

void toolcall::params::choice(std::string choice) {
    try {
        if (choice == "auto" || choice == "required" || choice == "none") {
            tool_choice_ = std::move(choice);

        } else {
            throw std::invalid_argument(
                "tool choice must be set to \"auto\", \"required\", or \"none\"");
        }

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

toolcall::params::operator bool() const  {
    return ! tools_.empty();
}
