#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdio>

// Regression test: an unreachable or malformed RPC endpoint must be reported as an
// error instead of aborting the calling process.
//
// Before the fix, ggml_backend_rpc_add_server() called GGML_ABORT (SIGABRT) when the
// endpoint could not be parsed, the transport failed to initialize, the socket could
// not connect, or the HELLO handshake failed - so a typo in an endpoint or a node
// that was still booting killed the whole application, and this test would die
// instead of failing cleanly.
//
// After the fix it returns nullptr, and the same condition is visible through
// ggml_backend_rpc_get_device_count() == 0.
//
// ref: https://github.com/ggml-org/llama.cpp/pull/30415

int main(void) {
    ggml_backend_load_all();

    // 1) nothing listens on this port: the connect must fail, not abort
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server("127.0.0.1:1");
    GGML_ASSERT(reg == nullptr);

    // 2) a malformed endpoint must take the parse-failure path and also return nullptr
    reg = ggml_backend_rpc_add_server("this-is-not-an-endpoint");
    GGML_ASSERT(reg == nullptr);

    // 3) the device-count helper must report "no devices" instead of crashing
    GGML_ASSERT(ggml_backend_rpc_get_device_count("127.0.0.1:1") == 0);

    printf("ok\n");
    return 0;
}
