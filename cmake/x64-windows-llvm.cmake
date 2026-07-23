set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

if (NOT DEFINED GGML_OPENMP OR GGML_OPENMP)
    file(TO_CMAKE_PATH
            "$ENV{VCINSTALLDIR}/Tools/Llvm/x64"
            llvm_X64_root
    )

    set( LLVM_X64_ROOT
            "${llvm_X64_root}"
            CACHE PATH
            "x64 LLVM installation containing the target OpenMP runtime"
    )

    if (EXISTS "${LLVM_X64_ROOT}/lib/libomp.lib")
        # Prefer the target LLVM OpenMP library over the MSVC import library.
        set( OpenMP_libomp_LIBRARY
                "${LLVM_X64_ROOT}/lib/libomp.lib"
                CACHE FILEPATH
                "x64 LLVM OpenMP import library"
                FORCE
        )
    else()
        message(WARNING "LLVM x64 OpenMP library not found: ${LLVM_X64_ROOT}/lib/libomp.lib")
    endif()
endif()
