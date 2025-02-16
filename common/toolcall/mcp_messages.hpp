#include <string>
#include <optional>
#include <vector>
#include <variant>
#include "../json.hpp"

namespace mcp
{
    extern const std::string JsonRpcVersion;
    extern const std::string McpVersion;
    extern const std::string ClientVersion;
    extern const std::string ClientName;

    template <typename Derived>
    class message {
    public:
        message(std::optional<nlohmann::json> id = std::nullopt)
            : id_(std::move(id)) {}

        nlohmann::json toJson() const {
            return static_cast<Derived*>(this)->toJson();
        }

        void id(std::optional<nlohmann::json> id) {
            id_ = std::move(id);
        }

        const std::optional<nlohmann::json> & id() const {
            return id_;
        }

    private:
        std::optional<nlohmann::json> id_;
    };

    class request : public message<request> {
    public:
        request(std::optional<nlohmann::json> id,
                std::string method,
                std::optional<nlohmann::json> params = std::nullopt)

            : message(id),
              method_(std::move(method)),
              params_(std::move(params)) {}

        void method(std::string method) { method_ = std::move(method); }
        const std::string & method() const { return method_; }

        void params(std::optional<nlohmann::json> params) { params_ = std::move(params); }
        const std::optional<nlohmann::json> & params() const { return params_; }

        nlohmann::json toJson() const;

    private:
        std::string method_;
        std::optional<nlohmann::json> params_;
    };

    class response : public message<response> {
    public:
        struct error {
            int code;
            std::string message;
            std::optional<nlohmann::json> data;
            nlohmann::json toJson() const;
        };

        response(std::optional<nlohmann::json> id,
                 std::optional<nlohmann::json> result = std::nullopt,
                 std::optional<error> error = std::nullopt)

            : message(id),
              result_(std::move(result)),
              error_(std::move(error)) {}

        void result(std::optional<nlohmann::json> result) { result_ = std::move(result); }
        const std::optional<nlohmann::json> & result() const { return result_; }

        void setError(std::optional<error> error) { error_ = std::move(error); }
        const std::optional<error> & getError() const { return error_; }

        nlohmann::json toJson() const;

    private:
        std::optional<nlohmann::json> result_;
        std::optional<error> error_;
    };

    class notification : public message<notification> {
    public:
        notification(std::string method,
                     std::optional<nlohmann::json> params = std::nullopt)

            : message(),
              method_(method),
              params_(params) {}

        void method(std::string method) { method_ = std::move(method); }
        const std::string & method() const { return method_; }

        void params(std::optional<nlohmann::json> params) { params_ = std::move(params); }
        const std::optional<nlohmann::json> & params() const { return params_; }

        nlohmann::json toJson() const;

    private:
        std::string method_;
        std::optional<nlohmann::json> params_;
    };

    struct capability {
        std::string name;
        bool subscribe   = false;
        bool listChanged = false;
    };

    using capabilities = std::vector<capability>;

    class initialize_request : public request {
    public:
        initialize_request(nlohmann::json id, mcp::capabilities caps);

        const std::string & name()    const { return ClientName; }
        const std::string & version() const { return ClientVersion; }
        const std::string & protoVersion() const { return McpVersion; }

        void capabilities(mcp::capabilities capabilities);
        const mcp::capabilities & capabilities() const;

    private:
        void refreshParams();

        mcp::capabilities caps_;
    };

    class initialize_response : public response {
    public:
        initialize_response(nlohmann::json id,
                            std::string name,
                            std::string version,
                            std::string protoVersion,
                            mcp::capabilities caps);

        void name(std::string name);
        const std::string & name() const;

        void version(std::string version);
        const std::string & version() const;

        void protoVersion(std::string protoVersion);
        const std::string & protoVersion() const;

        void capabilities(mcp::capabilities capabilities);
        const mcp::capabilities & capabilities() const;

        static initialize_response fromJson(const nlohmann::json& j);

    private:
        void refreshResult();

        std::string name_;
        std::string version_;
        std::string protoVersion_;
        mcp::capabilities caps_;
    };

    class initialized_notification : public notification {
    public:
        initialized_notification()
            : notification("notifications/initialized") {}
    };

    class tools_list_request : public request {
    public:
        tools_list_request(std::optional<nlohmann::json> id, std::string cursor = "");

        void cursor(std::string cursor);
        const std::string & cursor() { return cursor_; }

    private:
        void refreshParams();
        std::string cursor_;
    };

    struct tool {
        struct param {
            std::string name;
            std::string type;
            std::string description;
        };
        std::string tool_name;
        std::string tool_description;
        std::vector<param> params;
        std::vector<std::string> required_params;
    };

    using tools_list = std::vector<tool>;

    class tools_list_response : public response {
    public:
        tools_list_response(nlohmann::json id,
                            tools_list tools = tools_list(),
                            std::string next_cursor = "");

        void tools(tools_list tools);
        const tools_list & tools() const { return tools_; }

        void next_cursor(std::string next_cursor);
        const std::string & next_cursor() { return next_cursor_; }

    private:
        void refreshResult();
        tools_list tools_;
        std::string next_cursor_;
    };

    using message_variant = std::variant<
        initialize_request, initialize_response, initialized_notification,
        tools_list_request, tools_list_response>;
}
