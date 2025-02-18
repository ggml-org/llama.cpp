#pragma once

#include <string>
#include <variant>
#include <memory>

#include "../json.hpp" // TODO: switch to foreward decl.
// namespace nlohmann { class ordered_json; }

namespace toolcall
{
    class params {
    public:
        using json_ptr = std::shared_ptr<nlohmann::ordered_json>;
        using tools_t = std::variant<std::string, json_ptr>;
        using tool_choice_t = std::variant<std::string, json_ptr>;

        params(std::string tools = "", std::string choice = "auto");

        params(const params & other) = default;
        params(params && other) noexcept = default;
        params & operator=(const params & other) = default;
        params & operator=(params && other) noexcept = default;

        operator bool() const;

        void tools(std::string tools);
        const tools_t tools() const { return tools_; }

        void choice(std::string choice);
        const tool_choice_t & choice() const { return tool_choice_; }

    private:
        tools_t tools_;
        tool_choice_t tool_choice_;
    };
}
