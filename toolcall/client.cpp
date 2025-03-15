
#include <json.hpp>
#include "toolcall-client.h"
#include <chrono>
#include <stdexcept>

#ifdef LLAMA_USE_CURL
#    include "mcp_sse_transport.h"
#endif

#include "mcp_stdio_transport.h"

using json = nlohmann::json;

std::shared_ptr<toolcall::client> toolcall::create_client(const toolcall::params & params) {
    std::shared_ptr<toolcall::client> client;

    auto tools = params.tools();
    auto choice = params.choice();
    if (params) {
        if (params.has_uri()) {
#ifdef LLAMA_USE_CURL
            client.reset(new toolcall::client(
                              std::make_unique<toolcall::mcp_impl>(tools, choice)));
#endif
        } else {
            client.reset(new toolcall::client(
                              std::make_unique<toolcall::loopback_impl>(tools, choice)));
        }
    }
    return client;
}

std::string toolcall::client::tool_list() {
    return impl_->tool_list();
}

bool toolcall::client::tool_list_dirty() const {
    return impl_->tool_list_dirty();
}

toolcall::result_set toolcall::client::call(const std::string & name,
                                            const std::string & arguments,
                                            const std::string & id) {
    return impl_->call(name, arguments, id);
}

const std::string & toolcall::client::tool_choice() const {
    return impl_->tool_choice();
}

void toolcall::client::initialize() {
    impl_->initialize();
}

#ifdef LLAMA_USE_CURL
toolcall::mcp_impl::mcp_impl(std::string server_uri, std::string tool_choice)
    : client_impl(tool_choice),
      transport_(new mcp_sse_transport(server_uri)),
      tools_("[]"),
      tools_mutex_(),
      tools_populating_(),
      next_id_(1)
{
}
#else
toolcall::mcp_impl::mcp_impl(std::string /*server_uri*/, std::string tool_choice)
    : client_impl(tool_choice),
      transport_(nullptr),
      tools_("[]"),
      tools_mutex_(),
      tools_populating_(),
      next_id_(1)
{
}
#endif

toolcall::mcp_impl::mcp_impl(std::vector<std::string> argv, std::string tool_choice)
    : client_impl(tool_choice),
      transport_(new mcp_stdio_transport(argv))
{
}

void toolcall::mcp_impl::initialize() {
    using on_response = toolcall::callback<mcp::initialize_response>;
    using on_list_changed = toolcall::callback<mcp::tools_list_changed_notification>;

    if (transport_ == nullptr) return;
    std::unique_lock<std::mutex> lock(tools_mutex_);

    transport_->start();

    bool caps_received = false;
    mcp::capabilities caps;
    on_response set_caps = [this, &caps, &caps_received] (const mcp::initialize_response & resp) {
        std::unique_lock<std::mutex> lock(tools_mutex_);
        caps = resp.capabilities();
        caps_received = true;
        tools_populating_.notify_one();
    };

    transport_->send(mcp::initialize_request(next_id_++), set_caps);
    tools_populating_.wait_for(lock, std::chrono::seconds(15), [&caps_received] { return caps_received; });

    on_list_changed update_dirty = [&update_dirty, this] (const mcp::tools_list_changed_notification &) {
        tool_list_dirty_ = true;
        transport_->subscribe(update_dirty);
    };

    bool has_tools = false;
    for (const auto & cap : caps) {
        if (cap.name == "tools") {
            has_tools = true;
            if (cap.listChanged) {
                transport_->subscribe(update_dirty);
            }
            break;
        }
    }
    if (! has_tools) {
        throw std::runtime_error("MCP server does not support toolcalls!");
    }

    transport_->send(mcp::initialized_notification());
}

static std::string tools_list_to_oai_json(const mcp::tools_list & tools) {
    json tool_list = json::array();
    for (const auto & tool : tools) {
        json t = json::object();

        t["type"] = "function";
        t["function"]["name"] = tool.tool_name;
        t["function"]["description"] = tool.tool_description;

        json props = json::object();
        for (const auto & param : tool.params) {
            props[param.name]["type"] = param.type;
            props[param.name]["description"] = param.description;
        }
        t["function"]["parameters"]["type"] = "object";
        t["function"]["parameters"]["properties"] = props;

        json required = json::array();
        for (const auto & name : tool.required_params) {
            required.push_back(name);
        }
        t["function"]["required"] = required;

        tool_list.push_back(t);
    }

    return tool_list.dump(-1);
}

std::string toolcall::mcp_impl::tool_list() {
    using on_response = toolcall::callback<mcp::tools_list_response>;

    if (tool_list_dirty_) {
        std::unique_lock<std::mutex> lock(tools_mutex_);

        mcp::tools_list tools;
        on_response set_tools = [this, &tools, &set_tools] (const mcp::tools_list_response & resp) {
            std::unique_lock<std::mutex> lock(tools_mutex_);

            tools.insert(tools.end(), resp.tools().begin(), resp.tools().end());
            auto cursor = resp.next_cursor();
            if (! cursor.empty()) {
                transport_->send(mcp::tools_list_request(next_id_++, cursor), set_tools);
                return;
            }
            tool_list_dirty_ = false;
            lock.unlock();
            tools_populating_.notify_one();
        };

        transport_->send(mcp::tools_list_request(next_id_++), set_tools);
        tools_populating_.wait_for(lock, std::chrono::seconds(15), [this] { return ! tool_list_dirty_; });

        tools_ = tools_list_to_oai_json(tools);
    }
    return tools_;
}

static toolcall::result_set tools_call_response_to_result(const mcp::tools_call_response & resp) {
    toolcall::result_set result;
    for (const auto & res : resp.tool_result()) {
        result.push_back(toolcall::result{
                res.type, res.value, res.mime_type.value_or("text/plain"), res.uri, resp.tool_error()
            });
    }
    return std::move(result);
}

toolcall::result_set toolcall::mcp_impl::call(const std::string & name,
                                              const std::string & arguments,
                                              const std::string & id)
{
    using on_response = toolcall::callback<mcp::tools_call_response>;

    if (transport_ == nullptr) {
        return toolcall::result_set();
    }
    std::unique_lock<std::mutex> lock(tools_mutex_);

    toolcall::result_set response;
    on_response set_response = [this, &response] (const mcp::tools_call_response & resp) {
        std::unique_lock<std::mutex> lock(tools_mutex_);
        response = tools_call_response_to_result(resp);
        tools_populating_.notify_one();
    };
    std::string req_id = id.empty() ? std::to_string(next_id_++) : id;
    mcp::tool_arg_list req_args;
    auto json_args = json::parse(arguments); // TODO check errors
    for (const auto & [key, val] : json_args.items()) {
        req_args.push_back({key, val});
    }

    transport_->send(mcp::tools_call_request(req_id, name, req_args), set_response);
    tools_populating_.wait_for(lock, std::chrono::seconds(15), [&response] { return ! response.empty(); });

    return response;
}
