#pragma once

#include "mcp_transport.hpp"

#include <string>
#include <vector>

namespace toolcall
{
    class mcp_stdio_transport : public mcp_transport {
    public:
        mcp_stdio_transport(std::vector<std::string> argv);

        [[noreturn]] virtual void start() override;
        [[noreturn]] virtual void stop()  override;
        [[noreturn]] virtual bool send(const mcp::message_variant & request) override;

    private:
        std::vector<std::string> argv_;
    };
}
