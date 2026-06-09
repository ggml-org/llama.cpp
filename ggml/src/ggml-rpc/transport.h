#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    void set_skip_tensor_hash(bool value);
    bool skip_tensor_hash() const;

    void set_supports_device_type(bool value);
    bool supports_device_type() const;

    void set_supports_set_tensor_zlib(bool value);
    bool supports_set_tensor_zlib() const;

    void set_supports_copy_tensor_async(bool value);
    bool supports_copy_tensor_async() const;

    void set_label(const char * label);
    const std::string & label() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
