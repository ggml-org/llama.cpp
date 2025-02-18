
#include "../json.hpp" // Must come before params due to forward decl.
#include "params.hpp"
#include <stdexcept>

using json = nlohmann::ordered_json;

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
        if (tools.empty()) {
            tools_ = std::move(tools);

        } else if (starts_with(tools, "mcp+http")) {
#ifdef LLAMA_USE_CURL
            tools_ = std::move(tools);
#else
            throw std::invalid_argument(
                "Model Context Protocol (MCP) only works when llama.cpp is compiled with libcurl");
#endif
        } else {
            tools_ = std::make_shared<json>(json::parse(tools));
            auto tools_ptr = std::get<std::shared_ptr<json>>(tools_);
            if (! tools_ptr->is_array()) {
                throw std::invalid_argument(
                    "tools must be a URL of the form \"mcp+http(s)://hostname[:port]/\""
                    ", or a valid JSON array containing tool definitions");
            }
        }

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

void toolcall::params::choice(std::string choice) {
    try {
        if (choice == "auto" || choice == "required" || choice == "none") {
            tool_choice_ = std::move(choice);

        } else {
            auto choice_ptr = std::make_shared<json>(json::parse(choice));
            tool_choice_ = choice_ptr;
            if (! choice_ptr->is_object()) {
                throw std::invalid_argument(
                    "tool choice must be a valid JSON object, \"auto\", \"required\", or \"none\"");
            }
        }

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

toolcall::params::operator bool() const  {
    if (std::holds_alternative<std::string>(tools_)) {
        return ! std::get<std::string>(tools_).empty();

    } else {
        return std::get<json_ptr>(tools_) != nullptr;
    }
}
