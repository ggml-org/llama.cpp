set( CMAKE_SYSTEM_NAME Darwin )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

set( target arm64-apple-darwin-macho )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

set( arch_c_flags "-march=armv8.4-a -fvectorize -ffp-model=fast -fno-finite-math-only" )
set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" )

set( CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}" )
