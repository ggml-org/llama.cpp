set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

set( target arm64-pc-windows-msvc )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

if (NOT DEFINED GGML_OPENMP OR GGML_OPENMP)
    file(TO_CMAKE_PATH
            "$ENV{VCINSTALLDIR}/Tools/Llvm/ARM64"
            llvm_arm64_root
    )

    set( LLVM_ARM64_ROOT
            "${llvm_arm64_root}"
            CACHE PATH
            "ARM64 LLVM installation containing the target OpenMP runtime"
    )

    if (EXISTS "${LLVM_ARM64_ROOT}/lib/libomp.lib")
        # Prefer the target LLVM OpenMP library over the MSVC import library.
        set( OpenMP_libomp_LIBRARY
                "${LLVM_ARM64_ROOT}/lib/libomp.lib"
                CACHE FILEPATH
                "ARM64 LLVM OpenMP import library"
                FORCE
        )
    else()
        message(WARNING "LLVM ARM64 OpenMP library not found: ${LLVM_ARM64_ROOT}/lib/libomp.lib")
    endif()
endif()

set( arch_c_flags "-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only" )
set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function -Wno-gnu-zero-variadic-macro-arguments" )

set( CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}" )
