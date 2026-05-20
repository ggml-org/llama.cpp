#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // Wait up to timeout_seconds for the socket to become readable.
    // Returns true if data is ready (recv_data will not block on the
    // initial byte), false on timeout or socket error.
    bool wait_readable(int timeout_seconds);

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // Post-HELLO capabilities negotiation (protocol minor >= 1). See
    // GGML_RPC_FEATURE_* bits in ggml-rpc.cpp.
    void     set_negotiated_features(uint64_t features);
    uint64_t get_negotiated_features() const;

    // SHM transport segment, opaque to transport.h to keep the dependency
    // direction clean. Stored as shared_ptr<void> in the pimpl; the actual
    // type is rpc_shm_segment (defined in ggml-rpc.cpp). 3b only owns
    // lifecycle; 3c will plumb the rings through send_data/recv_data.
    void  set_shm_segment(std::shared_ptr<void> seg);
    void* get_shm_segment_raw() const;

    // True when the remote peer is on this machine (loopback). Used by the
    // CAPS exchange to gate same-host-only features such as SHM transport.
    // Cheap: a single getpeername() + IPv4/IPv6 prefix check.
    bool is_same_host() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
