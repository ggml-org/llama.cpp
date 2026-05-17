message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                         transport.cpp
                         rpc-trace.cpp
                        )

# Expose rpc-trace.h so tools/rpc/rpc-server.cpp and any other consumer that
# links against ggml-rpc can include it without copying.
target_include_directories(ggml-rpc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()

# RDMA auto-detection (Linux only, requires libibverbs)
if (NOT WIN32 AND NOT APPLE)
    find_library(IBVERBS_LIB ibverbs)
    if (IBVERBS_LIB)
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" ON)
    else()
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" OFF)
    endif()
else()
    set(GGML_RPC_RDMA OFF CACHE BOOL "RDMA not available on this platform" FORCE)
endif()

if (GGML_RPC_RDMA)
    if (NOT IBVERBS_LIB)
        find_library(IBVERBS_LIB ibverbs REQUIRED)
    endif()
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
    target_link_libraries(ggml-rpc PRIVATE ${IBVERBS_LIB})
    message(STATUS "  RDMA transport enabled (auto-detected)")
else()
    message(STATUS "  RDMA transport disabled")
endif()

# OpenTelemetry tracing (off by default — opt-in for diagnostics).
# When enabled, every RPC send/recv emits a span exported via OTLP/gRPC. See
# docs/RPC_TRACING.md for usage. Requires opentelemetry-cpp (find_package).
option(GGML_RPC_OPENTELEMETRY "ggml: emit OpenTelemetry spans from rpc-server" OFF)

if (GGML_RPC_OPENTELEMETRY)
    find_package(opentelemetry-cpp CONFIG REQUIRED)
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_OPENTELEMETRY)
    target_link_libraries(ggml-rpc PRIVATE
        opentelemetry-cpp::otlp_grpc_exporter
        opentelemetry-cpp::trace
        opentelemetry-cpp::resources
    )
    message(STATUS "  OpenTelemetry tracing enabled — spans -> $ENV{GGML_RPC_OTLP_ENDPOINT} (default http://localhost:4317)")
else()
    message(STATUS "  OpenTelemetry tracing disabled (set -DGGML_RPC_OPENTELEMETRY=ON to enable)")
endif()
