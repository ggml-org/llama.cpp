get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
