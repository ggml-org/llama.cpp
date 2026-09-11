# mtmd

set(MTMD_VIDEO_HELP "enable video support in mtmd (requires ffmpeg binary in PATH)")

set(MTMD_VIDEO ON CACHE BOOL "${MTMD_VIDEO_HELP}")
# TODO: add MTMD_VIDEO_METHOD in the future to select between ffmpeg and other backends

if (MTMD_VIDEO AND NOT LLAMA_SUBPROCESS)
    message(STATUS "Disabling MTMD_VIDEO because LLAMA_SUBPROCESS is OFF")
    set(MTMD_VIDEO OFF CACHE BOOL "${MTMD_VIDEO_HELP}" FORCE)
endif()

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd-image.cpp
            mtmd.h
            mtmd-internal.h
            mtmd-helper.cpp
            mtmd-helper-gen.cpp
            mtmd-helper-common.h
            mtmd-helper.h
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            models/models.h
            models/cogvlm.cpp
            models/conformer.cpp
            models/deepseek4v.cpp
            models/dots3note.cpp
            models/dotsocr.cpp
            models/exaone4_5.cpp
            models/gemma4a.cpp
            models/gemma4v.cpp
            models/gemma4ua.cpp
            models/gemma4uv.cpp
            models/glm4v.cpp
            models/glm5next-vision.cpp
            models/granite-speech.cpp
            models/granite4-vision.cpp
            models/hunyuanvl.cpp
            models/internvl.cpp
            models/kimivl.cpp
            models/kimik25.cpp
            models/nemotron-v2-vl.cpp
            models/muse-glimmer.cpp
            models/llama4.cpp
            models/llava.cpp
            models/minicpmv.cpp
            models/paddleocr.cpp
            models/pixtral.cpp
            models/qwen2vl.cpp
            models/minimax-m3.cpp
            models/qwen3vl.cpp
            models/mimovl.cpp
            models/qwen3a.cpp
            models/mimo-audio.cpp
            models/qwen3tts-spkenc.cpp
            models/qwen3tts-gen.cpp
            models/pockettts-seanet.cpp
            models/pockettts-spkenc.cpp
            models/pockettts-gen.cpp
            models/step3vl.cpp
            models/siglip.cpp
            models/whisper-enc.cpp
            models/deepseekocr.cpp
            models/deepseekocr2.cpp
            models/mobilenetv5.cpp
            models/youtuvl.cpp
            models/yasa2.cpp
            models/parakeet.cpp
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
target_include_directories(mtmd PUBLIC  .)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (MTMD_VIDEO)
    target_compile_definitions(mtmd PRIVATE MTMD_VIDEO)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)

    # export all symbols so that internal components can be tested by test-mtmd-impl
    set_target_properties     (mtmd PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

set_target_properties(mtmd
    PROPERTIES
    PRIVATE_HEADER debug/mtmd-debug.h)

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
    # miniaudio.h defines ma_android_sdk_version() without a prior prototype
    target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

# if mtmd is linked against llama-common, we throw an error
if (TARGET mtmd)
    get_target_property(libs mtmd LINK_LIBRARIES)
    if (libs AND "llama-common" IN_LIST libs)
        message(FATAL_ERROR "mtmd is designed to be a public library.\n"
                            "It must not link against llama-common")
    endif()
endif()

# Gate CLI binaries on LLAMA_BUILD_TOOLS so that standalone library-only
# builds (LLAMA_BUILD_MTMD=ON with LLAMA_BUILD_TOOLS=OFF — e.g. Apple
# XCFramework packaging) skip the executables entirely. LLAMA_BUILD_COMMON
# defaults to ON in standalone builds, so we cannot rely on it for gating.
if (LLAMA_BUILD_TOOLS)
    add_executable(llama-llava-cli    deprecation-warning.cpp)
    add_executable(llama-gemma3-cli   deprecation-warning.cpp)
    add_executable(llama-minicpmv-cli deprecation-warning.cpp)
    add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

    set(TARGET llama-mtmd-cli)
    add_executable         (${TARGET} mtmd-cli.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE llama-common mtmd Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # mtmd-debug tool
    add_executable(llama-mtmd-debug debug/mtmd-debug.cpp)
    set_target_properties(llama-mtmd-debug PROPERTIES OUTPUT_NAME llama-mtmd-debug)
    target_link_libraries(llama-mtmd-debug PRIVATE llama-common mtmd Threads::Threads)
    target_compile_features(llama-mtmd-debug PRIVATE cxx_std_17)
endif()
