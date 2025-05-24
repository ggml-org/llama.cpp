#pragma once

#include "toolcall-params.h"
#include <string>
#include <optional>
#include <memory>
#include <vector>
#include <condition_variable>
#include <mutex>

namespace toolcall
{
    struct result {
        std::string type;
        std::string data;
        std::string mime_type;
        std::optional<std::string> uri;
        bool error;
    };

    using result_set = std::vector<result>;

    class client_impl;
    class client {
    public:
        using ptr = std::shared_ptr<client>;

        client(std::unique_ptr<client_impl> impl) : impl_(std::move(impl)) {}

        result_set call(const std::string & name,
                        const std::string & arguments,
                        const std::string & id = "");

        std::string tool_list();
        bool tool_list_dirty() const;

        const std::string & tool_choice() const;

        void initialize();

    private:
        std::unique_ptr<client_impl> impl_;
    };

    std::shared_ptr<toolcall::client> create_client(const toolcall::params & params);

    class client_impl {
    public:
        client_impl(std::string tool_choice)
            : tool_choice_(std::move(tool_choice)), tool_list_dirty_(true) {}

        virtual ~client_impl() = default;

        virtual std::string tool_list() = 0;

        virtual bool tool_list_dirty() const {
            return tool_list_dirty_;
        }

        virtual result_set call(const std::string & name,
                                const std::string & arguments,
                                const std::string & id = "") = 0;

        const std::string & tool_choice() const { return tool_choice_; }

        virtual void initialize() {}

    protected:
        std::string tool_choice_;
        bool tool_list_dirty_;
    };

    class loopback_impl : public client_impl {
    public:
        loopback_impl(std::string tools, std::string tool_choice)
            : client_impl(tool_choice), tools_(std::move(tools)) {}

        virtual std::string tool_list() override {
            tool_list_dirty_ = false;
            return tools_;
        }

        virtual result_set call(const std::string & /* name */,
                                const std::string & /* arguments */,
                                const std::string & /* id = "" */) override {
            return result_set {
                {"text", "", "text/plain", std::nullopt, false}
            };
        }

    private:
        std::string tools_;
    };

    class mcp_transport;
    class mcp_impl : public client_impl {
    public:
        mcp_impl(std::string server_uri, std::string tool_choice);
        mcp_impl(std::vector<std::string> argv, std::string tool_choice);

        virtual std::string tool_list() override;

        virtual result_set call(const std::string & name,
                                const std::string & arguments,
                                const std::string & id = "") override;

        virtual void initialize() override;

    private:
        std::unique_ptr<mcp_transport> transport_;
        std::string tools_;
        std::mutex tools_mutex_;
        std::condition_variable tools_populating_;
        int next_id_;
    };
}
