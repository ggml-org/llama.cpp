message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
add_compile_options(--std=c++20)
ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                        )

target_link_libraries(ggml-tsavorite PRIVATE ${TLIBS} dl rt)
