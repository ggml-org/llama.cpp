#pragma once

#include "mcp_messages.hpp"

namespace toolcall
{
    class mcp_transport {
    public:
        using on_message_callback = std::function<void(const mcp::message_variant &)>;

        virtual ~mcp_transport() = default;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool send(const mcp::message_variant & request) = 0;

        void on_received(on_message_callback callback) { callback_ = std::move(callback); }
        const on_message_callback & on_received() const { return callback_; }

    protected:
        on_message_callback callback_;
    };
}
