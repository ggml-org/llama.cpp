
#include "../json.hpp"
#include "handler.hpp"
#include "params.hpp"

#ifdef LLAMA_USE_CURL
#    include "mcp_sse_transport.hpp"
#endif

#include "mcp_stdio_transport.hpp"

using json = toolcall::json;

std::shared_ptr<toolcall::handler> toolcall::create_handler(const toolcall::params & params) {
    std::shared_ptr<toolcall::handler> result;

    auto tools = params.tools();
    auto choice = params.choice();
    bool has_uri = std::holds_alternative<std::string>(tools);
    if (has_uri) {
#ifdef LLAMA_USE_CURL
        auto tools_str = std::get<std::string>(tools);
        if (! tools_str.empty()) {
            result.reset(new toolcall::handler(std::make_unique<toolcall::mcp_impl>(tools_str, choice)));
        }
#endif
    } else {
        auto tools_ptr = std::get<toolcall::json_ptr>(tools);
        if (tools_ptr != nullptr) {
            result.reset(new toolcall::handler(std::make_unique<toolcall::loopback_impl>(*tools_ptr, choice)));
        }
    }

    return result;
}

json toolcall::handler::tool_list() {
    return impl_->tool_list();
}

toolcall::action toolcall::handler::call(const json & request, json & response) {
    last_action_ = impl_->call(request, response);
    return last_action_;
}

const toolcall::tool_choice_t & toolcall::handler::tool_choice() const {
    return impl_->tool_choice();
}
toolcall::action toolcall::handler::last_action() const {
    return last_action_;
}

#ifdef LLAMA_USE_CURL
toolcall::mcp_impl::mcp_impl(std::string server_uri, tool_choice_t tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_sse_transport(server_uri))
{
    transport_->start();
}
#else
toolcall::mcp_impl::mcp_impl(std::string /*server_uri*/, tool_choice_t tool_choice)
    : handler_impl(tool_choice)
{
}
#endif

toolcall::mcp_impl::mcp_impl(std::vector<std::string> argv, tool_choice_t tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_stdio_transport(argv))
{
    transport_->start();
}

json toolcall::mcp_impl::tool_list() {
    // Construct tools/list call and send to transport
    return json{};// TODO
}

toolcall::action toolcall::mcp_impl::call(const json & /*request*/, json & /*response*/) {
    // Construct tool call and send to transport
    return toolcall::ACCEPT; // TODO
}
