
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

IF /I "%1"=="--help" (
    echo Usage: win-build-sycl.bat [fp32^|fp16] [--help]
    echo.
    echo Options:
    echo   fp32    Build with FP32 precision ^(default^)
    echo   fp16    Build with FP16 precision ^(faster for long-prompt inference^)
    echo   --help  Print this help message
    exit /B 0
)

SET PRECISION=%1
IF "%PRECISION%"=="" SET PRECISION=fp32
IF /I NOT "%PRECISION%"=="fp32" IF /I NOT "%PRECISION%"=="fp16" (
    echo Error: invalid value '%PRECISION%'. Use 'fp32' or 'fp16'.
    echo Usage: win-build-sycl.bat [fp32^|fp16] [--help]
    exit /B 1
)

IF not exist build (mkdir build)
cd build
if %errorlevel% neq 0 goto ERROR

@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
if %errorlevel% neq 0 goto ERROR

IF /I "%PRECISION%"=="fp16" (
    ::  for FP16
    ::  faster for long-prompt inference
    cmake -G "MinGW Makefiles" .. -DLLAMA_OPENSSL=OFF -DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON
) ELSE (
    ::  for FP32
    cmake -G "Ninja" .. -DLLAMA_OPENSSL=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
)
if %errorlevel% neq 0 goto ERROR

::  build all binary
cmake --build . -j
if %errorlevel% neq 0 goto ERROR

cd ..
exit /B 0

:ERROR
echo comomand error: %errorlevel%
exit /B %errorlevel%
