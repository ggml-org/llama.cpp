set(TARGET cpp-httplib)

find_package(Threads REQUIRED)

add_library(${TARGET} STATIC httplib.cpp httplib.h)
if (NOT MSVC)
    # disable warnings in 3rd party code
    target_compile_options(${TARGET} PRIVATE -w)
endif()

target_link_libraries  (${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET} PRIVATE
    # increase max payload length to allow use of larger context size
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    # increase backlog size to avoid connection resets for >> 1 slots
    CPPHTTPLIB_LISTEN_BACKLOG=512
    # increase max URI length to handle longer prompts in query string
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    # disable Nagle's algorithm
    CPPHTTPLIB_TCP_NODELAY=1
)

if (${CMAKE_SYSTEM_NAME} MATCHES "visionOS")
    # quick fix for https://github.com/ggml-org/llama.cpp/actions/runs/19247291428/job/55024294176?pr=17150
    target_compile_definitions(${TARGET} PRIVATE NI_MAXHOST=1025)
endif()
