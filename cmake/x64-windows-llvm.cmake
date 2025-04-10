set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

set( arch_c_flags "-march=native" )

set( CMAKE_C_FLAGS_INIT   "${arch_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags}" )

