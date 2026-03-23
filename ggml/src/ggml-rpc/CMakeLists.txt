message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                        )

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()

option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC (requires libibverbs)" OFF)
if (GGML_RPC_RDMA)
    find_library(IBVERBS_LIB ibverbs REQUIRED)
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
    target_link_libraries(ggml-rpc PRIVATE ${IBVERBS_LIB})
    message(STATUS "  RDMA transport enabled")
endif()
