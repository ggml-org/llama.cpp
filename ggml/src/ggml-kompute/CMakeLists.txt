
find_package(Vulkan COMPONENTS glslc REQUIRED)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

if (NOT glslc_executable)
    message(FATAL_ERROR "glslc not found")
endif()

ggml_add_backend_library(ggml-kompute
                         ggml-kompute.cpp
                         ../../include/ggml-kompute.h
                        )

target_link_libraries(ggml-kompute PRIVATE ggml-base kompute)
target_include_directories(ggml-kompute PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

function(compile_shader)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(compile_shader "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    foreach(source ${compile_shader_SOURCES})
        get_filename_component(filename ${source} NAME)
        set(spv_file ${filename}.spv)
        add_custom_command(
            OUTPUT ${spv_file}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/common.comp
            ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_getrows.comp
            ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n_pre.comp
            ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n.comp
            COMMAND ${glslc_executable} --target-env=vulkan1.2 -o ${spv_file} ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            COMMENT "Compiling ${source} to ${spv_file}"
            )

        get_filename_component(RAW_FILE_NAME ${spv_file} NAME)
        set(FILE_NAME "shader${RAW_FILE_NAME}")
        string(REPLACE ".comp.spv" ".h" HEADER_FILE ${FILE_NAME})
        string(TOUPPER ${HEADER_FILE} HEADER_FILE_DEFINE)
        string(REPLACE "." "_" HEADER_FILE_DEFINE "${HEADER_FILE_DEFINE}")
        set(OUTPUT_HEADER_FILE "${HEADER_FILE}")
        message(STATUS "${HEADER_FILE} generating ${HEADER_FILE_DEFINE}")
        if(CMAKE_GENERATOR MATCHES "Visual Studio")
            add_custom_command(
                OUTPUT ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                DEPENDS ${spv_file} xxd
                COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd"
                )
        else()
            add_custom_command(
                OUTPUT ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_BINARY_DIR}/bin/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                DEPENDS ${spv_file} xxd
                COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/xxd"
                )
        endif()
    endforeach()
endfunction()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kompute/CMakeLists.txt")
    message(STATUS "Kompute found")
    set(KOMPUTE_OPT_LOG_LEVEL Error CACHE STRING "Kompute log level")
    add_subdirectory(kompute)

    # Compile our shaders
    compile_shader(SOURCES
        kompute-shaders/op_scale.comp
        kompute-shaders/op_scale_8.comp
        kompute-shaders/op_add.comp
        kompute-shaders/op_addrow.comp
        kompute-shaders/op_mul.comp
        kompute-shaders/op_silu.comp
        kompute-shaders/op_relu.comp
        kompute-shaders/op_gelu.comp
        kompute-shaders/op_softmax.comp
        kompute-shaders/op_norm.comp
        kompute-shaders/op_rmsnorm.comp
        kompute-shaders/op_diagmask.comp
        kompute-shaders/op_mul_mat_mat_f32.comp
        kompute-shaders/op_mul_mat_f16.comp
        kompute-shaders/op_mul_mat_q8_0.comp
        kompute-shaders/op_mul_mat_q4_0.comp
        kompute-shaders/op_mul_mat_q4_1.comp
        kompute-shaders/op_mul_mat_q4_k.comp
        kompute-shaders/op_mul_mat_q6_k.comp
        kompute-shaders/op_getrows_f32.comp
        kompute-shaders/op_getrows_f16.comp
        kompute-shaders/op_getrows_q4_0.comp
        kompute-shaders/op_getrows_q4_1.comp
        kompute-shaders/op_getrows_q6_k.comp
        kompute-shaders/op_rope_norm_f16.comp
        kompute-shaders/op_rope_norm_f32.comp
        kompute-shaders/op_rope_neox_f16.comp
        kompute-shaders/op_rope_neox_f32.comp
        kompute-shaders/op_cpy_f16_f16.comp
        kompute-shaders/op_cpy_f16_f32.comp
        kompute-shaders/op_cpy_f32_f16.comp
        kompute-shaders/op_cpy_f32_f32.comp
    )

    # Create a custom target for our generated shaders
    add_custom_target(generated_shaders DEPENDS
        shaderop_scale.h
        shaderop_scale_8.h
        shaderop_add.h
        shaderop_addrow.h
        shaderop_mul.h
        shaderop_silu.h
        shaderop_relu.h
        shaderop_gelu.h
        shaderop_softmax.h
        shaderop_norm.h
        shaderop_rmsnorm.h
        shaderop_diagmask.h
        shaderop_mul_mat_mat_f32.h
        shaderop_mul_mat_f16.h
        shaderop_mul_mat_q8_0.h
        shaderop_mul_mat_q4_0.h
        shaderop_mul_mat_q4_1.h
        shaderop_mul_mat_q4_k.h
        shaderop_mul_mat_q6_k.h
        shaderop_getrows_f32.h
        shaderop_getrows_f16.h
        shaderop_getrows_q4_0.h
        shaderop_getrows_q4_1.h
        shaderop_getrows_q6_k.h
        shaderop_rope_norm_f16.h
        shaderop_rope_norm_f32.h
        shaderop_rope_neox_f16.h
        shaderop_rope_neox_f32.h
        shaderop_cpy_f16_f16.h
        shaderop_cpy_f16_f32.h
        shaderop_cpy_f32_f16.h
        shaderop_cpy_f32_f32.h
    )

    # Create a custom command that depends on the generated_shaders
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
        DEPENDS generated_shaders
        COMMENT "Ensuring shaders are generated before compiling ggml-kompute.cpp"
    )

    # Add the stamp to the main sources to ensure dependency tracking
    target_sources(ggml-kompute PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp)
else()
    message(WARNING "Kompute not found")
endif()
