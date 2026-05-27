if (NOT MACA_PATH)
    if (NOT EXISTS $ENV{MACA_PATH})
        if (EXISTS /opt/maca)
            set(MACA_PATH /opt/maca)
        else()
            set(MACA_PATH /usr/local/cuda)
        endif()
    else()
        set(MACA_PATH $ENV{MACA_PATH})
    endif()
endif()

if (NOT MACA_PATH)
    set(MACA_PATH /usr/local/cuda)
endif()

if (DEFINED ENV{MACA_PATH} AND NOT MACA_PATH)
    set(MACA_PATH $ENV{MACA_PATH})
endif()

if (NOT EXISTS ${MACA_PATH})
    message(FATAL_ERROR "MetaX backend: MACA_PATH does not exist: ${MACA_PATH}")
endif()

if (NOT EXISTS "${MACA_PATH}/bin/nvcc")
    message(FATAL_ERROR "MetaX backend: expected CUDA-compat compiler at ${MACA_PATH}/bin/nvcc")
endif()

message(STATUS "MetaX backend: using MACA path ${MACA_PATH}")

set(METAX_NVCC "${MACA_PATH}/bin/nvcc")
if (EXISTS "/opt/maca/include")
    set(METAX_RUNTIME_PATH "/opt/maca")
else()
    set(METAX_RUNTIME_PATH "${MACA_PATH}")
endif()
if (EXISTS "/opt/maca/tools/cu-bridge/bin/gomxccbin")
    set(METAX_CUCC_PATH "/opt/maca/tools/cu-bridge")
elseif (EXISTS "${MACA_PATH}/tools/cu-bridge/bin/gomxccbin")
    set(METAX_CUCC_PATH "${MACA_PATH}/tools/cu-bridge")
else()
    message(FATAL_ERROR "MetaX backend: cannot locate cu-bridge (gomxccbin)")
endif()
set(METAX_CUBRIDGE_INCLUDE "${METAX_CUCC_PATH}/include")

file(GLOB GGML_SOURCES_METAX "../ggml-cuda/*.cu")

file(GLOB SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_METAX ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_METAX ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_METAX ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_METAX ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_METAX ${SRCS})
else()
    list(APPEND GGML_SOURCES_METAX
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

set(METAX_NVCC_FLAGS
    -x cu
    --maca-path=${METAX_RUNTIME_PATH}
    -Xcompiler=-fPIC
    -std=c++17
    -use_fast_math
    -extended-lambda
    -imacros
    ${METAX_CUBRIDGE_INCLUDE}/__macro_mxcc.h
    -I${METAX_CUBRIDGE_INCLUDE}
    -I${MACA_PATH}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/..
    -I${CMAKE_CURRENT_SOURCE_DIR}/../..
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set(METAX_DEFS
    -DGGML_USE_METAX
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
    -DGGML_CUDA_FORCE_CUBLAS
    -DGGML_CUDA_NO_FA
)
if (GGML_CUDA_GRAPHS)
    list(APPEND METAX_DEFS -DGGML_CUDA_USE_GRAPHS)
endif()
if (GGML_CUDA_FORCE_MMQ)
    list(APPEND METAX_DEFS -DGGML_CUDA_FORCE_MMQ)
endif()
if (GGML_CUDA_NO_VMM)
    list(APPEND METAX_DEFS -DGGML_CUDA_NO_VMM)
endif()
if (GGML_CUDA_NO_PEER_COPY)
    list(APPEND METAX_DEFS -DGGML_CUDA_NO_PEER_COPY)
endif()

set(METAX_OBJECTS)
foreach(SOURCE ${GGML_SOURCES_METAX})
    get_filename_component(SOURCE_ABS "${SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(RELATIVE_PATH SOURCE_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SOURCE_ABS}")
    string(REPLACE "/" "_" OBJ_BASENAME "${SOURCE_REL}")
    string(REPLACE "." "_" OBJ_BASENAME "${OBJ_BASENAME}")

    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/metax_objs/${OBJ_BASENAME}.o")

    add_custom_command(
        OUTPUT "${OBJ_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metax_objs"
        COMMAND ${CMAKE_COMMAND} -E env
            "MACA_PATH=${MACA_PATH}"
            "CUCC_PATH=${METAX_CUCC_PATH}"
            "C_INCLUDE_PATH=${METAX_CUBRIDGE_INCLUDE}:${MACA_PATH}/include"
            "CPLUS_INCLUDE_PATH=${METAX_CUBRIDGE_INCLUDE}:${MACA_PATH}/include"
            "${METAX_NVCC}" ${METAX_NVCC_FLAGS} ${METAX_DEFS} -c "${SOURCE_ABS}" -o "${OBJ_FILE}"
        DEPENDS "${SOURCE_ABS}"
        VERBATIM
    )
    list(APPEND METAX_OBJECTS "${OBJ_FILE}")
endforeach()

ggml_add_backend_library(ggml-metax
                         ${METAX_OBJECTS}
                        )

if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()
target_include_directories(ggml-metax PRIVATE "${MACA_PATH}/include")

target_link_directories(ggml-metax PRIVATE "${MACA_PATH}/lib64")
if (GGML_STATIC)
    target_link_libraries(ggml-metax PRIVATE cudart_static mcblas_static mcblasLt_static)
else()
    target_link_libraries(ggml-metax PRIVATE cudart mcblas mcblasLt)
endif()

if (NOT GGML_CUDA_NO_VMM)
    target_link_libraries(ggml-metax PRIVATE mxc-runtime64)
endif()

set_target_properties(ggml-metax PROPERTIES LINKER_LANGUAGE CXX)
