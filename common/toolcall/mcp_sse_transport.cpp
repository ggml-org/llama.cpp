
#include "mcp_sse_transport.hpp"

toolcall::mcp_sse_transport::mcp_sse_transport(std::string server_uri)
    : server_uri_(std::move(server_uri))
{
}

void toolcall::mcp_sse_transport::start() {
}

void toolcall::mcp_sse_transport::stop() {
}

bool toolcall::mcp_sse_transport::send(const mcp::message_variant & /*request*/) {
    return false;
}
