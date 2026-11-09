
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

SET "PRECISION=fp32"
SET "SSL=ON"
SET "OPENSSL_OPTION="

:PARSE_ARGS
IF "%~1"=="" GOTO ARGS_DONE
IF /I "%~1"=="--help" GOTO HELP
IF /I "%~1"=="fp32" GOTO ARG_FP32
IF /I "%~1"=="fp16" GOTO ARG_FP16
IF /I "%~1"=="ssl" GOTO ARG_SSL
IF /I "%~1"=="nossl" GOTO ARG_NOSSL
GOTO INVALID_ARG

:ARG_FP32
SET "PRECISION=fp32"
GOTO NEXT_ARG

:ARG_FP16
SET "PRECISION=fp16"
GOTO NEXT_ARG

:ARG_SSL
SET "SSL=ON"
SET "OPENSSL_OPTION="
GOTO NEXT_ARG

:ARG_NOSSL
SET "SSL=OFF"
SET "OPENSSL_OPTION=-DLLAMA_OPENSSL=OFF"
GOTO NEXT_ARG

:NEXT_ARG
SHIFT
GOTO PARSE_ARGS

:HELP
SET "ARG_ERROR=0"
GOTO PRINT_USAGE

:INVALID_ARG
SET "ARG_ERROR=1"
echo Error: unknown option '%~1'

:PRINT_USAGE
echo Usage: win-build-sycl.bat [fp32^|fp16] [ssl^|nossl] [--help]
echo.
echo Options:
echo   fp32      Build with FP32 precision ^(default^)
echo   fp16      Build with FP16 precision ^(faster for long-prompt inference^)
echo   ssl       Build with BoringSSL ^(default^)
echo   nossl     Build without BoringSSL
echo   --help    Print this help message
IF "%ARG_ERROR%"=="1" exit /B 1
exit /B 0

:ARGS_DONE

IF not exist build (mkdir build)
cd build
if %errorlevel% neq 0 goto ERROR

@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
if %errorlevel% neq 0 goto ERROR

IF /I "%PRECISION%"=="fp16" (
    ::  for FP16
    ::  faster for long-prompt inference
    cmake -G "MinGW Makefiles" .. -DLLAMA_BUILD_BORINGSSL=%SSL% %OPENSSL_OPTION% -DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON
) ELSE (
    ::  for FP32
    cmake -G "Ninja" .. -DLLAMA_BUILD_BORINGSSL=%SSL% %OPENSSL_OPTION% -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
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
