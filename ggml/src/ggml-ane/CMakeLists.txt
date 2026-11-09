find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(IOSURFACE_LIBRARY  IOSurface  REQUIRED)

ggml_add_backend_library(ggml-ane
                         ggml-ane.cpp
                         ggml-ane-mil.cpp
                         ggml-ane-runtime.m
                        )

set_source_files_properties(ggml-ane-runtime.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_link_libraries(ggml-ane PRIVATE
                      ${FOUNDATION_LIBRARY}
                      ${IOSURFACE_LIBRARY}
                      )
