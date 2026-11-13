
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

SET "PRECISION=fp32"
SET "SSL=ON"
SET "OPENSSL_OPTION="
SET "AOT_OPTION="
SET "XMX_OPTION="

:PARSE_ARGS
IF "%~1"=="" GOTO ARGS_DONE
IF /I "%~1"=="--help" GOTO HELP
IF /I "%~1"=="fp32" GOTO ARG_FP32
IF /I "%~1"=="fp16" GOTO ARG_FP16
IF /I "%~1"=="ssl" GOTO ARG_SSL
IF /I "%~1"=="no-ssl" GOTO ARG_NO_SSL
IF /I "%~1"=="no-xmx" GOTO ARG_NO_XMX
IF /I "%~1"=="--aot" GOTO ARG_AOT
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

:ARG_NO_SSL
SET "SSL=OFF"
SET "OPENSSL_OPTION=-DLLAMA_OPENSSL=OFF"
GOTO NEXT_ARG

:ARG_NO_XMX
SET "XMX_OPTION=-DGGML_SYCL_SUPPORT_XMX=OFF"
GOTO NEXT_ARG

:ARG_AOT
IF "%~2"=="" GOTO MISSING_AOT
SET "AOT_VALUE=%~2"
IF "%AOT_VALUE:~0,2%"=="--" GOTO MISSING_AOT
SET "AOT_OPTION=-DGGML_SYCL_DEVICE_ARCH=%~2"
SHIFT
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
GOTO PRINT_USAGE

:MISSING_AOT
SET "ARG_ERROR=1"
echo Error: --aot requires aot_flags

:PRINT_USAGE
echo Usage: win-build-sycl.bat [fp32^|fp16] [ssl^|no-ssl] [no-xmx] [--aot aot_flags] [--help]
echo.
echo Options:
echo   fp32      Build with FP32 precision ^(default^)
echo   fp16      Build with FP16 precision ^(faster for long-prompt inference^)
echo   ssl       Build with BoringSSL ^(default^)
echo   no-ssl    Build without BoringSSL
echo   no-xmx    Disable XMX ^(DPAS^) support
echo   --aot     Set SYCL AOT device architecture, e.g. arl-h,acm-g10,bmg-g21 or bmg-g21
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
    cmake -G "MinGW Makefiles" .. -DLLAMA_BUILD_BORINGSSL=%SSL% %OPENSSL_OPTION% -DGGML_SYCL=ON %XMX_OPTION% %AOT_OPTION% -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON
) ELSE (
    ::  for FP32
    cmake -G "Ninja" .. -DLLAMA_BUILD_BORINGSSL=%SSL% %OPENSSL_OPTION% -DGGML_SYCL=ON %XMX_OPTION% %AOT_OPTION% -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
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
