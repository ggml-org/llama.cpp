#pragma once

#include "mcp_transport.h"

#include <string>
#include <vector>

namespace toolcall
{
    class mcp_stdio_transport : public mcp_transport {
    public:
        mcp_stdio_transport(std::vector<std::string> argv);

        [[noreturn]] virtual void start() override;
        [[noreturn]] virtual void stop()  override;
        [[noreturn]] virtual bool send(const std::string & request_json) override;

    private:
        std::vector<std::string> argv_;
    };
}
