#pragma once

#include <string>

namespace toolcall
{
    class params {
    public:
        params(std::string tools = "", std::string choice = "auto");

        params(const params & other) = default;
        params(params && other) noexcept = default;
        params & operator=(const params & other) = default;
        params & operator=(params && other) noexcept = default;

        operator bool() const;

        void tools(std::string tools);
        const std::string & tools() const { return tools_; }

        void choice(std::string choice);
        const std::string & choice() const { return tool_choice_; }

        bool has_uri() const { return has_uri_; }

    private:
        std::string tools_;
        std::string tool_choice_;
        bool has_uri_;
    };
}
