#TODO
#not work on Linux
set( CMAKE_SYSTEM_NAME mingw )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

set( target aarch64-w64-mingw32 )

set( CMAKE_C_COMPILER    aarch64-w64-mingw32-gcc )
set( CMAKE_CXX_COMPILER  aarch64-w64-mingw32-g++ )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

#set( arch_c_flags "-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only" )
#set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function -Wno-gnu-zero-variadic-macro-arguments" )

set( CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}" )
