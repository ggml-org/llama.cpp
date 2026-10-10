# Plugin skeleton implementation target
add_library(deterministic_draft_spec_plugin SHARED
    "deterministic_regex_plugin.c"
)

target_include_directories(deterministic_draft_spec_plugin PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

set_target_properties(deterministic_draft_spec_plugin PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
