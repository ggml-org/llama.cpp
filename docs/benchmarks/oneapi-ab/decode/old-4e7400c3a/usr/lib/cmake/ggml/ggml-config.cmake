
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ggml-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# Use original install prefix when loaded through a "/usr move"
# cross-prefix symbolic link such as /lib -> /usr/lib.
get_filename_component(_realCurr "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_realOrig "/usr/lib/cmake/ggml" REALPATH)
if(_realCurr STREQUAL _realOrig)
  set(PACKAGE_PREFIX_DIR "/usr")
endif()
unset(_realOrig)
unset(_realCurr)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################


####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

set(GGML_ACCELERATE "ON")
set(GGML_ALL_WARNINGS "ON")
set(GGML_ALL_WARNINGS_3RD_PARTY "OFF")
set(GGML_AMX_BF16 "OFF")
set(GGML_AMX_INT8 "OFF")
set(GGML_AMX_TILE "OFF")
set(GGML_AVAILABLE_BACKENDS "ggml-cpu;ggml-sycl")
set(GGML_AVX "OFF")
set(GGML_AVX2 "OFF")
set(GGML_AVX512 "OFF")
set(GGML_AVX512_BF16 "OFF")
set(GGML_AVX512_VBMI "OFF")
set(GGML_AVX512_VNNI "OFF")
set(GGML_AVX_VNNI "OFF")
set(GGML_BACKEND_DIR "")
set(GGML_BACKEND_DL "OFF")
set(GGML_BIN_INSTALL_DIR "bin")
set(GGML_BLAS "OFF")
set(GGML_BLAS_DEFAULT "OFF")
set(GGML_BLAS_VENDOR "Generic")
set(GGML_BLAS_VENDOR_DEFAULT "Generic")
set(GGML_BMI2 "OFF")
set(GGML_BUILD_COMMIT "4e7400c3a-dirty")
set(GGML_BUILD_EXAMPLES "OFF")
set(GGML_BUILD_NUMBER "12305")
set(GGML_BUILD_TESTS "OFF")
set(GGML_CCACHE "ON")
set(GGML_CCACHE_FOUND "/usr/bin/ccache")
set(GGML_CPU "ON")
set(GGML_CPU_ALL_VARIANTS "OFF")
set(GGML_CPU_ARM_ARCH "")
set(GGML_CPU_HBM "OFF")
set(GGML_CPU_KLEIDIAI "OFF")
set(GGML_CPU_POWERPC_CPUTYPE "")
set(GGML_CPU_REPACK "ON")
set(GGML_F16C "OFF")
set(GGML_FATAL_WARNINGS "OFF")
set(GGML_FMA "OFF")
set(GGML_GIT_DIRTY "1")
set(GGML_GPROF "OFF")
set(GGML_INCLUDE_INSTALL_DIR "include")
set(GGML_LASX "ON")
set(GGML_LIB_INSTALL_DIR "lib")
set(GGML_LLAMAFILE "ON")
set(GGML_LLAMAFILE_DEFAULT "OFF")
set(GGML_LSX "ON")
set(GGML_LTO "OFF")
set(GGML_NATIVE "ON")
set(GGML_NATIVE_DEFAULT "OFF")
set(GGML_OPENMP "ON")
set(GGML_OPENMP_ENABLED "ON")
set(GGML_OPENMP_FETCH "OFF")
set(GGML_OPENVINO "OFF")
set(GGML_PUBLIC_HEADERS "include/ggml.h;include/ggml-cpu.h;include/ggml-alloc.h;include/ggml-backend.h;include/ggml-blas.h;include/ggml-cpp.h;include/ggml-opt.h;include/ggml-sycl.h;include/ggml-vulkan.h;include/ggml-openvino.h;include/gguf.h")
set(GGML_RVV "ON")
set(GGML_RV_ZFH "ON")
set(GGML_RV_ZICBOP "ON")
set(GGML_RV_ZIHINTPAUSE "ON")
set(GGML_RV_ZVFBFWMA "OFF")
set(GGML_RV_ZVFH "ON")
set(GGML_SANITIZE_ADDRESS "OFF")
set(GGML_SANITIZE_THREAD "OFF")
set(GGML_SANITIZE_UNDEFINED "OFF")
set(GGML_SCCACHE_FOUND "/usr/bin/sccache")
set(GGML_SCHED_MAX_COPIES "4")
set(GGML_SCHED_NO_REALLOC "OFF")
set(GGML_SHARED_LIB "ON")
set(GGML_SSE42 "OFF")
set(GGML_STANDALONE "OFF")
set(GGML_STATIC "OFF")
set(GGML_SYCL "ON")
set(GGML_SYCL_DEVICE_ARCH "acm-g10")
set(GGML_SYCL_DEVICE_CODE_SPLIT "ON")
set(GGML_SYCL_DNN "OFF")
set(GGML_SYCL_F16 "ON")
set(GGML_SYCL_GRAPH "ON")
set(GGML_SYCL_HOST_MEM_FALLBACK "ON")
set(GGML_SYCL_MAX_PARALLEL_LINK_JOBS "24")
set(GGML_SYCL_SCALAR_MMQ "OFF")
set(GGML_SYCL_SEPARATE_BUILD "OFF")
set(GGML_SYCL_SEPARATE_BUILD_ARGS "")
set(GGML_SYCL_SEPARATE_BUILD_CXX "icpx")
set(GGML_SYCL_SEPARATE_BUILD_CXX_DEFAULT "icpx")
set(GGML_SYCL_SUPPORT_LEVEL_ZERO_API "ON")
set(GGML_SYCL_TARGET "INTEL")
set(GGML_SYCL_TURBO_QUANT "ON")
set(GGML_VERSION "0.25.3")
set(GGML_VERSION_BASE "0.25.3")
set(GGML_VERSION_MAJOR "0")
set(GGML_VERSION_MINOR "25")
set(GGML_VERSION_PATCH "3")
set(GGML_VULKAN "OFF")
set(GGML_VULKAN_CHECK_RESULTS "OFF")
set(GGML_VULKAN_DEBUG "OFF")
set(GGML_VULKAN_MEMORY_DEBUG "OFF")
set(GGML_VULKAN_RUN_TESTS "OFF")
set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "")
set(GGML_VULKAN_SHADER_DEBUG_INFO "OFF")
set(GGML_VULKAN_VALIDATE "OFF")
set(GGML_VXE "ON")
set(GGML_XTHEADVECTOR "OFF")


# Find all dependencies before creating any target.
include(CMakeFindDependencyMacro)
find_dependency(Threads)
if (NOT GGML_SHARED_LIB)
    set(GGML_BASE_INTERFACE_LINK_LIBRARIES "")
    set(GGML_CPU_INTERFACE_LINK_LIBRARIES "")
    set(GGML_CPU_INTERFACE_LINK_OPTIONS   "")

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(NOT ACCELERATE_FRAMEWORK)
            set(${CMAKE_FIND_PACKAGE_NAME}_FOUND 0)
            return()
        endif()
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES ${ACCELERATE_FRAMEWORK})
    endif()

    if (GGML_OPENMP_ENABLED)
        find_dependency(OpenMP)
        set(GGML_OPENMP_INTERFACE_LINK_LIBRARIES "")
        if (TARGET OpenMP::OpenMP_C)
            list(APPEND GGML_OPENMP_INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_C)
        endif()
        if (TARGET OpenMP::OpenMP_CXX)
            list(APPEND GGML_OPENMP_INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
        endif()
        list(APPEND GGML_BASE_INTERFACE_LINK_LIBRARIES ${GGML_OPENMP_INTERFACE_LINK_LIBRARIES})
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES ${GGML_OPENMP_INTERFACE_LINK_LIBRARIES})
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind)
        if(NOT memkind)
            set(${CMAKE_FIND_PACKAGE_NAME}_FOUND 0)
            return()
        endif()
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES memkind)
    endif()

    if (GGML_BLAS)
        find_dependency(BLAS)
        list(APPEND GGML_BLAS_INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})
        list(APPEND GGML_BLAS_INTERFACE_LINK_OPTIONS   ${BLAS_LINKER_FLAGS})
    endif()




    if (GGML_VULKAN)
        find_dependency(Vulkan)
        set(GGML_VULKAN_INTERFACE_LINK_LIBRARIES $<LINK_ONLY:Vulkan::Vulkan>)
    endif()


    if (GGML_SYCL)
        set(GGML_SYCL_INTERFACE_LINK_LIBRARIES "")
        find_package(DNNL)
        if (${DNNL_FOUND} AND GGML_SYCL_TARGET STREQUAL "INTEL")
            list(APPEND GGML_SYCL_INTERFACE_LINK_LIBRARIES DNNL::dnnl)
        endif()
        if (WIN32)
            find_dependency(IntelSYCL)
            find_dependency(MKL)
            list(APPEND GGML_SYCL_INTERFACE_LINK_LIBRARIES IntelSYCL::SYCL_CXX MKL::MKL MKL::MKL_SYCL)
        endif()
    endif()
endif()

set_and_check(GGML_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set_and_check(GGML_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")
#set_and_check(GGML_BIN_DIR "${PACKAGE_PREFIX_DIR}/bin")

if (NOT GGML_SHARED_LIB AND GGML_CPU_KLEIDIAI)
    unset(KLEIDIAI_LIBRARY CACHE)
    unset(KLEIDIAI_LIBRARY)
    find_library(KLEIDIAI_LIBRARY kleidiai
        REQUIRED
        HINTS ${GGML_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)
    list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES ${KLEIDIAI_LIBRARY})
endif()

if(NOT TARGET ggml::ggml)
    find_package(Threads REQUIRED)

    unset(GGML_LIBRARY CACHE)
    find_library(GGML_LIBRARY ggml
        REQUIRED
        HINTS ${GGML_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)

    add_library(ggml::ggml UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml
        PROPERTIES
            IMPORTED_LOCATION "${GGML_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE_DIR}")

    unset(GGML_BASE_LIBRARY CACHE)
    find_library(GGML_BASE_LIBRARY ggml-base
        REQUIRED
        HINTS ${GGML_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)

    add_library(ggml::ggml-base UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml-base
        PROPERTIES
            IMPORTED_LOCATION "${GGML_BASE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${GGML_BASE_INTERFACE_LINK_LIBRARIES}")

    set(_ggml_all_targets "")
    if (NOT GGML_BACKEND_DL)
        foreach(_ggml_backend ${GGML_AVAILABLE_BACKENDS})
            string(REPLACE "-" "_" _ggml_backend_pfx "${_ggml_backend}")
            string(TOUPPER "${_ggml_backend_pfx}" _ggml_backend_pfx)

            unset(${_ggml_backend_pfx}_LIBRARY CACHE)
            find_library(${_ggml_backend_pfx}_LIBRARY ${_ggml_backend}
                REQUIRED
                HINTS ${GGML_LIB_DIR}
                NO_CMAKE_FIND_ROOT_PATH)

            message(STATUS "Found ${${_ggml_backend_pfx}_LIBRARY}")

            add_library(ggml::${_ggml_backend} UNKNOWN IMPORTED)
            set_target_properties(ggml::${_ggml_backend}
                PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE_DIR}"
                    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                    IMPORTED_LOCATION "${${_ggml_backend_pfx}_LIBRARY}"
                    INTERFACE_COMPILE_FEATURES c_std_90
                    POSITION_INDEPENDENT_CODE ON)

            string(REGEX MATCH "^ggml-cpu" is_cpu_variant "${_ggml_backend}")
            if(is_cpu_variant)
                list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES "ggml::ggml-base")
                set_target_properties(ggml::${_ggml_backend}
                PROPERTIES
                    INTERFACE_LINK_LIBRARIES "${GGML_CPU_INTERFACE_LINK_LIBRARIES}")

                if(GGML_CPU_INTERFACE_LINK_OPTIONS)
                    set_target_properties(ggml::${_ggml_backend}
                        PROPERTIES
                            INTERFACE_LINK_OPTIONS "${GGML_CPU_INTERFACE_LINK_OPTIONS}")
                endif()

            else()
                list(APPEND ${_ggml_backend_pfx}_INTERFACE_LINK_LIBRARIES "ggml::ggml-base")
                set_target_properties(ggml::${_ggml_backend}
                    PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${${_ggml_backend_pfx}_INTERFACE_LINK_LIBRARIES}")

                if(${_ggml_backend_pfx}_INTERFACE_LINK_OPTIONS)
                    set_target_properties(ggml::${_ggml_backend}
                        PROPERTIES
                            INTERFACE_LINK_OPTIONS "${${_ggml_backend_pfx}_INTERFACE_LINK_OPTIONS}")
                endif()
            endif()

            list(APPEND _ggml_all_targets ggml::${_ggml_backend})
        endforeach()
    endif()

    list(APPEND GGML_INTERFACE_LINK_LIBRARIES ggml::ggml-base "${_ggml_all_targets}")
    set_target_properties(ggml::ggml
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "${GGML_INTERFACE_LINK_LIBRARIES}")

    add_library(ggml::all INTERFACE IMPORTED)
    set_target_properties(ggml::all
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_ggml_all_targets}")

endif()

check_required_components(ggml)
