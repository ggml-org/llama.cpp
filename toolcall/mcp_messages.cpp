#include "mcp_messages.h"
#include <iostream>
#include <log.h>
#include <stdexcept>

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

    json capabilities = json::object();
    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;
        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }
        capabilities[cap->name] = cap_json;
    }
    params["capabilities"] = capabilities;

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

    json capabilities = json::object();
    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;
        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }
        capabilities[cap->name] = cap_json;
    }
    result["capabilities"] = capabilities;

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
    : request(id, Method),
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

mcp::tools_list_response mcp::tools_list_response::fromJson(const nlohmann::json & j) {
    mcp::tools_list tools;
    for (const auto & t : j["result"]["tools"]) {
        mcp::tool tool;
        tool.tool_name = t["name"];
        tool.tool_description = t["description"];
        for (const auto & [key, value] : t["inputSchema"]["properties"].items()) {
            mcp::tool::param param;
            param.name = key;
            param.type = value["type"];
            param.description = value["description"];
            tool.params.push_back(param);
        }
        if (t["inputSchema"].contains("required") && t["inputSchema"]["required"].is_array()) {
            for (const auto & required : t["inputSchema"]["required"]) {
                tool.required_params.push_back(required);
            }
        }
        tools.push_back(std::move(tool));
    }
    std::string next_cursor = j["result"].value("nextCursor", "");
    return tools_list_response(j["id"], std::move(tools), next_cursor);
}

mcp::tools_list_changed_notification mcp::tools_list_changed_notification::fromJson(const nlohmann::json & j) {
    if (! (j.is_object() && j.contains("method") && j["method"] == Method)) {
        throw std::invalid_argument("Invalid tools_list_changed message");
    }
    return tools_list_changed_notification();
}

mcp::tools_call_request::tools_call_request(nlohmann::json id, std::string name, tool_arg_list args)
    : request(id, Method), name_(std::move(name)), args_(std::move(args))
{
    refreshParams();
}

void mcp::tools_call_request::name(std::string name) {
    name_ = std::move(name);
    refreshParams();
}

void mcp::tools_call_request::args(mcp::tool_arg_list args) {
    args_ = std::move(args);
    refreshParams();
}

void mcp::tools_call_request::refreshParams() {
    json params = json::object();
    params["name"] = name_;
    if (! args_.empty()) {
        json args = json::object();
        for (const auto & arg : args_) {
            args[arg.name] = arg.value;
        }
        params["arguments"] = args;
    }
    this->params(params);
}

mcp::tools_call_response::tools_call_response(nlohmann::json id, tool_result_list result, bool error)
    : response(id), tool_result_(std::move(result)), error_(error)
{
    refreshResult();
}

void mcp::tools_call_response::tool_result(mcp::tool_result_list result) {
    tool_result_ = std::move(result);
    refreshResult();
}

void mcp::tools_call_response::tool_error(bool error) {
    error_ = error;
    refreshResult();
}

void mcp::tools_call_response::refreshResult() {
    json result = json::object();
    result["isError"] = error_;
    json content = json::array();
    for (const auto & res : tool_result_) {
        json r;
        r["type"] = res.type;
        if (res.type == "text") {
            r["text"] = res.value;

        } else if (res.type == "image" || res.type == "audio") {
            r["data"] = res.value;
            r["mimeType"] = res.mime_type.value(); // throws

        } else if (res.type == "resource") {
            json rr;
            rr["uri"] = res.uri.value(); // throws
            rr["mimeType"] = res.mime_type.value(); //throws
            rr["text"] = res.value;

            r["resource"] = rr;

        } else {
            // throw
        }
        content.push_back(r);
    }
    result["content"] = content;
    this->result(std::move(result));
}

mcp::tools_call_response mcp::tools_call_response::fromJson(const nlohmann::json & j) {
    mcp::tool_result_list result_list;
    for (const auto & content : j["result"]["content"]) {
        mcp::tool_result result;

        result.type = content["type"];
        if (content["type"] == "text") {
            result.value = content["text"];

        } else if (content["type"] == "image" || content["type"] == "audio") {
            result.value = content["data"];
            result.mime_type = content["mimeType"];

        } else if (content["type"] == "resource") {
            result.value = content["resource"]["text"];
            result.mime_type = content["resource"]["mimeType"];
            result.uri = content["resource"]["uri"];
        }

        result_list.push_back(std::move(result));
    }

    bool error = j["result"].value("isError", false);

    return mcp::tools_call_response(j["id"], std::move(result_list), error);
}
