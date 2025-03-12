set(TARGET llama-cogagent-cli)
add_executable(${TARGET} cogagent-cli.cpp)
add_library(cogagent OBJECT
            vision_encoder.cpp
            vision_encoder.h
            cross_vision.cpp
            cross_vision.h
            cogagent_util.cpp
            cogagent_util.h
            image_util.cpp
            image_util.h)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-cogagent-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common cogagent ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(cogagent PUBLIC ../../ggml/include)
target_include_directories(cogagent PUBLIC ../../include)
target_include_directories(cogagent PUBLIC ../../common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)