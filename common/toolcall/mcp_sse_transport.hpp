#pragma once

#include "mcp_transport.hpp"

namespace toolcall
{
    class mcp_sse_transport : public mcp_transport {
    public:
        mcp_sse_transport(std::string server_uri);

        virtual void start() override;
        virtual void stop()  override;
        virtual bool send(const mcp::message_variant & request) override;

    private:
        std::string server_uri_;
    };
}
