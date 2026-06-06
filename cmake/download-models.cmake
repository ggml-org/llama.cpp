get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

set(DOWNLOAD_HEADERS "")
if(DEFINED ENV{HF_TOKEN} AND NOT "$ENV{HF_TOKEN}" STREQUAL "")
    set(DOWNLOAD_HEADERS HTTPHEADER "Authorization: Bearer $ENV{HF_TOKEN}")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    ${DOWNLOAD_HEADERS}
    STATUS status
)

list(GET status 0 code)

if(code EQUAL 0)
    string(REPLACE "=" ";" hash_parts "${HASH}")
    list(LENGTH hash_parts hash_parts_len)
    if(NOT hash_parts_len EQUAL 2)
        file(REMOVE "${DEST}")
        message(FATAL_ERROR "Invalid hash spec '${HASH}'")
    endif()

    list(GET hash_parts 0 hash_algorithm)
    list(GET hash_parts 1 expected_hash)
    file("${hash_algorithm}" "${DEST}" actual_hash)

    if(NOT actual_hash STREQUAL expected_hash)
        file(REMOVE "${DEST}")
        message(FATAL_ERROR
            "Downloaded ${NAME} but hash check failed for ${DEST}: expected ${expected_hash}, got ${actual_hash}")
    endif()
else()
    list(GET status 1 msg)
    file(REMOVE "${DEST}")
    message(WARNING "Failed to download ${NAME}: ${msg} (test will be skipped)")
endif()
