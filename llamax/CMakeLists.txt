#
# Define version
#

set(LLAMAX_MAJOR_VERSION 2)
set(LLAMAX_MINOR_VERSION 1)
set(LLAMAX_PATCH_VERSION 0)
set(LLAMAX_VERSION ${LLAMAX_MAJOR_VERSION}.${LLAMAX_MINOR_VERSION}.${LLAMAX_PATCH_VERSION})

#
# Build llamax
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LLAMAX_SRCS src/llamax.cpp)
add_library(llamax SHARED ${LLAMAX_SRCS})
target_link_libraries(llamax PRIVATE llama)

set(LLAMAX_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llamax.h)

set_target_properties(llamax
    PROPERTIES
        PUBLIC_HEADER "${LLAMAX_PUBLIC_HEADERS}")

add_subdirectory(examples)

#
# install
#

install(TARGETS llamax LIBRARY PUBLIC_HEADER)
#
# Config files
#

include(CMakePackageConfigHelpers)
configure_package_config_file(llamaxConfig.cmake.in "${PROJECT_BINARY_DIR}/llamaxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
    )
write_basic_package_version_file("${PROJECT_BINARY_DIR}/llamaxConfigVersion.cmake" VERSION ${LLAMAX_VERSION} COMPATIBILITY SameMajorVersion)

# Install the llamaxConfig.cmake and llamaxConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}/llamaxConfig.cmake"
  "${PROJECT_BINARY_DIR}/llamaxConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/llama" COMPONENT dev)
