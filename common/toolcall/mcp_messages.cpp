#include "mcp_messages.hpp"
#include <iostream>

using json = nlohmann::json;

const std::string mcp::JsonRpcVersion = "2.0";
const std::string mcp::McpVersion     = "2024-11-05";
const std::string mcp::ClientVersion  = "1.0.0";
const std::string mcp::ClientName     = "llama.cpp";

json mcp::request::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    if (id()) {
        j["id"] = id().value();
    }
    j["method"] = method();
    if (params()) {
        j["params"] = params().value();
    }
    return j;
}

json mcp::response::error::toJson() const {
    json j;
    j["code"] = code;
    j["message"] = message;
    if (data) {
        j["data"] = data.value();
    }
    return j;
}

json mcp::response::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    if (id()) {
        j["id"] = id().value();
    }
    if (result()) {
        j["result"] = result().value();
    } else if (getError()) {
        j["error"] = getError()->toJson();
    }
    return j;
}

json mcp::notification::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    j["method"] = method();
    if (params()) {
        j["params"] = params().value();
    }
    return j;
}

mcp::initialize_request::initialize_request(nlohmann::json id, mcp::capabilities caps)
    : request(id, "initialize"), caps_(std::move(caps))
{
     refreshParams();
}

void mcp::initialize_request::refreshParams() {
    json params;
    params["protocolVersion"] = protoVersion();
    params["clientInfo"]["name"] = name();
    params["clientInfo"]["version"] = version();
    params["capabilities"] = {};

    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;

        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }

        params["capabilities"][cap->name] = cap_json;
    }

    this->params(std::move(params));
}

void mcp::initialize_request::capabilities(mcp::capabilities caps) {
    caps_ = std::move(caps);
    refreshParams();
}

const mcp::capabilities & mcp::initialize_request::capabilities() const {
    return caps_;
}

mcp::initialize_response::initialize_response(
    nlohmann::json id, std::string name, std::string version, std::string protoVersion,
    mcp::capabilities caps)
    : response(id), name_(std::move(name)), version_(std::move(version)),
      protoVersion_(std::move(protoVersion)), caps_(std::move(caps))
{
    refreshResult();
}

void mcp::initialize_response::refreshResult() {
    json result;
    result["protocolVersion"] = protoVersion();
    result["serverInfo"]["name"] = name();
    result["serverInfo"]["version"] = version();
    result["capabilities"] = {};

    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;

        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }

        result["capabilities"][cap->name] = cap_json;
    }

    this->result(std::move(result));
}

void mcp::initialize_response::name(std::string name) {
    name_ = std::move(name);
    refreshResult();
}

const std::string & mcp::initialize_response::name() const {
    return name_;
}

void mcp::initialize_response::version(std::string version) {
    version_ = std::move(version);
    refreshResult();
}

const std::string & mcp::initialize_response::version() const {
    return version_;
}

void mcp::initialize_response::protoVersion(std::string protoVersion) {
    protoVersion_ = std::move(protoVersion);
    refreshResult();
}

const std::string & mcp::initialize_response::protoVersion() const {
    return protoVersion_;
}

void mcp::initialize_response::capabilities(mcp::capabilities caps) {
    caps_ = std::move(caps);
    refreshResult();
}

const mcp::capabilities & mcp::initialize_response::capabilities() const {
    return caps_;
}

mcp::initialize_response mcp::initialize_response::fromJson(const nlohmann::json& j) {
    std::string name = j["result"]["serverInfo"]["name"];
    std::string version = j["result"]["serverInfo"]["version"];
    std::string protoVersion = j["result"]["protocolVersion"];

    mcp::capabilities caps;
    if (j["result"].contains("capabilities")) {
        for (const auto& [key, value] : j["result"]["capabilities"].items()) {
            capability cap;
            cap.name = key;
            cap.subscribe = value.value("subscribe", false);
            cap.listChanged = value.value("listChanged", false);
            caps.push_back(cap);
        }
    }

    return initialize_response(j["id"], name, version, protoVersion, caps);
}

mcp::tools_list_request::tools_list_request(std::optional<nlohmann::json> id, std::string cursor)
    : request(id, "tools/list"),
      cursor_(std::move(cursor))
{
    refreshParams();
}

void mcp::tools_list_request::cursor(std::string cursor) {
    cursor_ = std::move(cursor);
    refreshParams();
}

void mcp::tools_list_request::refreshParams() {
    if (! cursor_.empty()) {
        json params;
        params["cursor"] = cursor_;
        this->params(params);
    }
}

mcp::tools_list_response::tools_list_response(nlohmann::json id,
                                              mcp::tools_list tools,
                                              std::string next_cursor)
    : response(id),
      tools_(std::move(tools)),
      next_cursor_(std::move(next_cursor))
{
    refreshResult();
}

void mcp::tools_list_response::tools(mcp::tools_list tools) {
    tools_ = std::move(tools);
    refreshResult();
}

void mcp::tools_list_response::next_cursor(std::string next_cursor) {
    next_cursor_ = std::move(next_cursor);
    refreshResult();
}

void mcp::tools_list_response::refreshResult() {
    json result;

    json tools = json::array();
    for (const auto & tool : tools_) {
        json t;

        t["name"] = tool.tool_name;
        t["description"] = tool.tool_description;
        t["inputSchema"]["type"] = "object";

        json props;
        for (const auto & param : tool.params) {
            props[param.name] = {
                {"type"}, {param.type},
                {"description"}, {param.description}
            };
        }
        t["inputSchema"]["properties"] = props;

        json required = json::array();
        for (const auto & req_param : tool.required_params) {
            required.push_back(req_param);
        }
        t["inputSchema"]["required"] = required;

        tools.push_back(t);
    }
    result["tools"] = tools;

    if (! next_cursor_.empty()) {
        result["nextCursor"] = next_cursor_;
    }

    this->result(result);
}
