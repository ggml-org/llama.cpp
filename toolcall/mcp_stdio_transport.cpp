
#include "mcp_stdio_transport.h"

#include <stdexcept>

toolcall::mcp_stdio_transport::mcp_stdio_transport(std::vector<std::string> argv)
    : argv_(std::move(argv))
{
}

[[noreturn]] void toolcall::mcp_stdio_transport::start() {
    throw std::logic_error(std::string("Function not implemented: ") + __func__);
}

[[noreturn]] void toolcall::mcp_stdio_transport::stop() {
    throw std::logic_error(std::string("Function not implemented: ") + __func__);
}

[[noreturn]] bool toolcall::mcp_stdio_transport::send(const std::string & /*request_json*/) {
    throw std::logic_error(std::string("Function not implemented: ") + __func__);
}
