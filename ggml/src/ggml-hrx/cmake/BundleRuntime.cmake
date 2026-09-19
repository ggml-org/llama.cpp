# Capture the module directory while it is the active list file. On CMake 3.14,
# CMAKE_CURRENT_LIST_DIR inside a function refers to the function's call site.
set(_GGML_HRX_BUNDLE_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Find every entry matching PATTERN in exactly one of SEARCH_DIRS.
#
# OUT_PATHS receives the sorted matching paths. Missing families are fatal.
function(_ggml_hrx_find_bundle_entries OUT_PATHS LABEL PATTERN SEARCH_DIRS)
    set(GGML_HRX_SELECTED_PATHS)
    set(GGML_HRX_SELECTED_DIR "")
    foreach(GGML_HRX_SEARCH_DIR IN LISTS SEARCH_DIRS)
        file(GLOB GGML_HRX_DIR_MATCHES
            CONFIGURE_DEPENDS
            LIST_DIRECTORIES FALSE
            "${GGML_HRX_SEARCH_DIR}/${PATTERN}")
        if (GGML_HRX_DIR_MATCHES)
            if (NOT GGML_HRX_SELECTED_DIR STREQUAL "")
                message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS found ${LABEL} in multiple source directories: ${GGML_HRX_SELECTED_DIR};${GGML_HRX_SEARCH_DIR}")
            endif()
            set(GGML_HRX_SELECTED_DIR "${GGML_HRX_SEARCH_DIR}")
            set(GGML_HRX_SELECTED_PATHS ${GGML_HRX_DIR_MATCHES})
        endif()
    endforeach()

    if (NOT GGML_HRX_SELECTED_PATHS)
        message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS could not find required ${LABEL} matching ${PATTERN}. Searched: ${SEARCH_DIRS}")
    endif()

    foreach(GGML_HRX_SELECTED_PATH IN LISTS GGML_HRX_SELECTED_PATHS)
        if (IS_SYMLINK "${GGML_HRX_SELECTED_PATH}")
            if (NOT EXISTS "${GGML_HRX_SELECTED_PATH}")
                message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS matched a broken symlink: ${GGML_HRX_SELECTED_PATH}")
            endif()
        elseif(NOT EXISTS "${GGML_HRX_SELECTED_PATH}")
            message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS matched a nonexistent file: ${GGML_HRX_SELECTED_PATH}")
        endif()
    endforeach()
    list(LENGTH GGML_HRX_SELECTED_PATHS GGML_HRX_SELECTED_COUNT)
    message(STATUS "  ${LABEL}: ${GGML_HRX_SELECTED_DIR} (${GGML_HRX_SELECTED_COUNT} entries)")

    set(${OUT_PATHS} "${GGML_HRX_SELECTED_PATHS}" PARENT_SCOPE)
endfunction()

# Keep the target's existing relative build and install RPATH entries and drop
# absolute entries. Append the paths needed by the adjacent runtime bundle. The
# resulting list is returned through OUT_VAR.
function(_ggml_hrx_collect_portable_rpath TARGET_NAME OUT_VAR)
    set(GGML_HRX_PORTABLE_RPATH)
    foreach(GGML_HRX_RPATH_PROPERTY BUILD_RPATH INSTALL_RPATH)
        get_target_property(GGML_HRX_RPATH_ENTRIES "${TARGET_NAME}" "${GGML_HRX_RPATH_PROPERTY}")
        if (NOT GGML_HRX_RPATH_ENTRIES)
            continue()
        endif()
        foreach(GGML_HRX_RPATH_ENTRY IN LISTS GGML_HRX_RPATH_ENTRIES)
            if (GGML_HRX_RPATH_ENTRY STREQUAL "")
                continue()
            endif()
            if (IS_ABSOLUTE "${GGML_HRX_RPATH_ENTRY}")
                continue()
            endif()
            list(APPEND GGML_HRX_PORTABLE_RPATH "${GGML_HRX_RPATH_ENTRY}")
        endforeach()
    endforeach()
    list(APPEND GGML_HRX_PORTABLE_RPATH
        "$ORIGIN"
        "$ORIGIN/rocm_sysdeps/lib")
    list(REMOVE_DUPLICATES GGML_HRX_PORTABLE_RPATH)
    set(${OUT_VAR} "${GGML_HRX_PORTABLE_RPATH}" PARENT_SCOPE)
endfunction()

# Add matching build-tree copy and install rules for SOURCES.
# RELATIVE_DESTINATION is appended below both backend destinations.
function(_ggml_hrx_add_bundle_rules TARGET_NAME SOURCES RELATIVE_DESTINATION INSTALL_BASE COPY_SCRIPT)
    set(GGML_HRX_BUILD_DESTINATION "$<TARGET_FILE_DIR:${TARGET_NAME}>")
    set(GGML_HRX_INSTALL_DESTINATION "${INSTALL_BASE}")
    if (NOT RELATIVE_DESTINATION STREQUAL "")
        string(APPEND GGML_HRX_BUILD_DESTINATION "/${RELATIVE_DESTINATION}")
        string(APPEND GGML_HRX_INSTALL_DESTINATION "/${RELATIVE_DESTINATION}")
    endif()

    if (NOT RELATIVE_DESTINATION STREQUAL "")
        add_custom_command(TARGET "${TARGET_NAME}" POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${GGML_HRX_BUILD_DESTINATION}"
            VERBATIM)
    endif()
    add_custom_command(TARGET "${TARGET_NAME}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
            "-DGGML_HRX_BUNDLE_SOURCES=${SOURCES}"
            "-DGGML_HRX_BUNDLE_DESTINATION=${GGML_HRX_BUILD_DESTINATION}"
            -P "${COPY_SCRIPT}"
        VERBATIM)
    install(FILES ${SOURCES}
        DESTINATION "${GGML_HRX_INSTALL_DESTINATION}")
endfunction()

# Discover, copy, and install the HRX runtime dependencies for TARGET_NAME.
function(ggml_hrx_bundle_runtime TARGET_NAME)
    set(GGML_HRX_COPY_SCRIPT "${_GGML_HRX_BUNDLE_RUNTIME_DIR}/copy_bundle_entry.cmake")
    if (NOT TARGET "${TARGET_NAME}")
        message(FATAL_ERROR "ggml_hrx_bundle_runtime target does not exist: ${TARGET_NAME}")
    endif()
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS is currently implemented for Linux only")
    endif()
    if (NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR "GGML_HRX_BUNDLE_RUNTIME_LIBS requires BUILD_SHARED_LIBS=ON")
    endif()

    set(GGML_HRX_BUNDLE_SEARCH_DIRS)
    foreach(GGML_HRX_BUNDLE_SEARCH_DIR IN LISTS GGML_HRX_BUNDLE_LIBRARY_DIRS)
        if (GGML_HRX_BUNDLE_SEARCH_DIR STREQUAL "")
            message(FATAL_ERROR "GGML_HRX_BUNDLE_LIBRARY_DIRS contains an empty directory entry")
        endif()
        get_filename_component(GGML_HRX_BUNDLE_SEARCH_DIR_ABSOLUTE "${GGML_HRX_BUNDLE_SEARCH_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if (NOT IS_DIRECTORY "${GGML_HRX_BUNDLE_SEARCH_DIR_ABSOLUTE}")
            message(FATAL_ERROR "GGML_HRX_BUNDLE_LIBRARY_DIRS contains a nonexistent directory: ${GGML_HRX_BUNDLE_SEARCH_DIR}")
        endif()
        get_filename_component(GGML_HRX_BUNDLE_SEARCH_DIR_CANONICAL "${GGML_HRX_BUNDLE_SEARCH_DIR_ABSOLUTE}" REALPATH)
        list(APPEND GGML_HRX_BUNDLE_SEARCH_DIRS "${GGML_HRX_BUNDLE_SEARCH_DIR_CANONICAL}")
    endforeach()
    list(REMOVE_DUPLICATES GGML_HRX_BUNDLE_SEARCH_DIRS)
    if (NOT GGML_HRX_BUNDLE_SEARCH_DIRS)
        message(FATAL_ERROR "GGML_HRX_BUNDLE_LIBRARY_DIRS must list at least one directory when GGML_HRX_BUNDLE_RUNTIME_LIBS=ON")
    endif()

    message(STATUS "HRX runtime bundling search directories: ${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    message(STATUS "HRX runtime bundle selection:")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_HRX_LIBS
        "libhrx" "libhrx.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_LOOMC_LIBS
        "libloomc" "libloomc.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_HSA_RUNTIME_LIBS
        "libhsa-runtime64" "libhsa-runtime64.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_HSA_AQLPROFILE_LIBS
        "libhsa-amd-aqlprofile64" "libhsa-amd-aqlprofile64.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_ROCPROFILER_REGISTER_LIBS
        "librocprofiler-register" "librocprofiler-register.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_OMP_LIBS
        "libomp" "libomp.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")
    # HRX runtime bundles require the ROCm sysdeps overlay.
    _ggml_hrx_find_bundle_entries(
        GGML_HRX_BUNDLE_SYSDEP_LIBS
        "rocm_sysdeps/lib overlay" "rocm_sysdeps/lib/*.so*" "${GGML_HRX_BUNDLE_SEARCH_DIRS}")

    set(GGML_HRX_BUNDLE_MAIN_LIBS
        ${GGML_HRX_BUNDLE_HRX_LIBS}
        ${GGML_HRX_BUNDLE_LOOMC_LIBS}
        ${GGML_HRX_BUNDLE_HSA_RUNTIME_LIBS}
        ${GGML_HRX_BUNDLE_HSA_AQLPROFILE_LIBS}
        ${GGML_HRX_BUNDLE_ROCPROFILER_REGISTER_LIBS}
        ${GGML_HRX_BUNDLE_OMP_LIBS}
    )
    set_property(TARGET "${TARGET_NAME}" APPEND PROPERTY LINK_DEPENDS
        ${GGML_HRX_BUNDLE_MAIN_LIBS}
        ${GGML_HRX_BUNDLE_SYSDEP_LIBS}
        "${GGML_HRX_COPY_SCRIPT}")

    # GGML_BACKEND_DL builds backends as runtime-loaded modules instead of
    # normally linked libraries. Install dependencies next to the backend:
    # modules use GGML_BACKEND_DIR or bin; linked backends use the standard
    # library directory.
    if (GGML_BACKEND_DL)
        if (GGML_BACKEND_DIR)
            set(GGML_HRX_BUNDLE_INSTALL_DIR "${GGML_BACKEND_DIR}")
        else()
            set(GGML_HRX_BUNDLE_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
        endif()
    else()
        set(GGML_HRX_BUNDLE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()
    message(STATUS "HRX runtime bundle install directory: ${GGML_HRX_BUNDLE_INSTALL_DIR}")

    # Give build and install artifacts the same portable RUNPATH. Keep relative
    # entries, add adjacent bundle directories, and prevent absolute HRX/ROCm
    # link directories.
    _ggml_hrx_collect_portable_rpath("${TARGET_NAME}" GGML_HRX_PORTABLE_RPATH)
    set_target_properties("${TARGET_NAME}" PROPERTIES
        BUILD_RPATH "${GGML_HRX_PORTABLE_RPATH}"
        INSTALL_RPATH "${GGML_HRX_PORTABLE_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
    message(STATUS "HRX backend RUNPATH: ${GGML_HRX_PORTABLE_RPATH}")

    _ggml_hrx_add_bundle_rules(
        "${TARGET_NAME}"
        "${GGML_HRX_BUNDLE_MAIN_LIBS}"
        ""
        "${GGML_HRX_BUNDLE_INSTALL_DIR}"
        "${GGML_HRX_COPY_SCRIPT}")
    _ggml_hrx_add_bundle_rules(
        "${TARGET_NAME}"
        "${GGML_HRX_BUNDLE_SYSDEP_LIBS}"
        "rocm_sysdeps/lib"
        "${GGML_HRX_BUNDLE_INSTALL_DIR}"
        "${GGML_HRX_COPY_SCRIPT}")
endfunction()
