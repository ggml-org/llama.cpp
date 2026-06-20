set(TARGET omtd)

add_library(${TARGET} omtd.cpp)

install(TARGETS ${TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

target_include_directories(${TARGET} PUBLIC .)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PUBLIC cxx_std_17)

set_target_properties(${TARGET} PROPERTIES PUBLIC_HEADER omtd.h)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
