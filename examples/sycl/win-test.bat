::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT


@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM MIT license
REM Copyright (C) 2024 Intel Corporation
REM SPDX-License-Identifier: MIT

set "BIN_FILE=.\build\bin\llama-completion.exe"
set "SEED=0"
set "GPUS_SETTING="

set "INPUT_PROMPT=Building a website can be done in 10 simple steps:^nStep 1:"
set "MODEL_FILE=..\models\llama-2-7b.Q4_0.gguf"
set "NGL=99"
set "CONTEXT=4096"
set "GGML_SYCL_DEVICE=-1"
set "SPLIT_MODE=layer"
set "LOG_VERBOSE=3"

if "%~1"=="" goto after_args

:parse_args
if "%~1"=="" goto after_args

if /I "%~1"=="-c" (
  if "%~2"=="" goto missing_value
  set "CONTEXT=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--context" (
  if "%~2"=="" goto missing_value
  set "CONTEXT=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-p" (
  if "%~2"=="" goto missing_value
  set "INPUT_PROMPT=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--promote" (
  if "%~2"=="" goto missing_value
  set "INPUT_PROMPT=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-m" (
  if "%~2"=="" goto missing_value
  set "MODEL_FILE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--model" (
  if "%~2"=="" goto missing_value
  set "MODEL_FILE=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-mg" (
  if "%~2"=="" goto missing_value
  set "GGML_SYCL_DEVICE=%~2"
  set "SPLIT_MODE=none"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--main-gpu" (
  if "%~2"=="" goto missing_value
  set "GGML_SYCL_DEVICE=%~2"
  set "SPLIT_MODE=none"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-sm" (
  if "%~2"=="" goto missing_value
  set "SPLIT_MODE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--split-mode" (
  if "%~2"=="" goto missing_value
  set "SPLIT_MODE=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-ngl" (
  if "%~2"=="" goto missing_value
  set "NGL=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--n-gpu-layers" (
  if "%~2"=="" goto missing_value
  set "NGL=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-lv" (
  if "%~2"=="" goto missing_value
  set "LOG_VERBOSE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--log-verbosity" (
  if "%~2"=="" goto missing_value
  set "LOG_VERBOSE=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help

echo Invalid option: %~1
exit /b 1

:missing_value
echo Missing value for option: %~1
exit /b 1

:help
echo Usage: %~n0 [OPTIONS]
echo.
echo This script processes files with specified options.
echo.
echo Options:
echo   -h, --help    Display this help message and exit.
echo   -c, --context ^<value^>    Set context length. Bigger need more memory.
echo   -p, --promote ^<value^>    Prompt to start generation with.
echo   -m, --model   ^<value^>    Full model file path.
echo   -mg,--main-gpu ^<value^>   Set main GPU ID (0 - n) for single GPU mode.
echo   -sm,--split-mode ^<value^> How to split the model across multiple GPUs, one of:
echo                             - none: use one GPU only
echo                             - layer (default): split layers and KV across GPUs
echo                             - row: split rows across GPUs
echo   -ngl,--n-gpu-layers ^<value^>  Max. number of layers to store in VRAM (default: -1)
echo   -lv,--log-verbosity ^<value^>  Set the verbosity threshold. Messages with a higher verbosity will be
echo                                ignored. Values:
echo                                 - 0: generic output
echo                                 - 1: error
echo                                 - 2: warning
echo                                 - 3: info
echo                                 - 4: debug
exit /b 0

:after_args

REM In Windows CMD, source is not available; call oneAPI setvars if present.
if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
  call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul
) else (
  echo Warning: oneAPI setvars.bat not found. Continuing without environment setup.
)

REM Support malloc device memory more than 4GB.
set "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1"
echo UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=%UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS%

if not "%GGML_SYCL_DEVICE%"=="-1" (
  echo Use %GGML_SYCL_DEVICE% as main GPU
  REM Use single GPU only.
  set "GPUS_SETTING=-mg %GGML_SYCL_DEVICE% -sm %SPLIT_MODE%"
  set "ONEAPI_DEVICE_SELECTOR=level_zero:%GGML_SYCL_DEVICE%"
  echo ONEAPI_DEVICE_SELECTOR=%ONEAPI_DEVICE_SELECTOR%
) else (
  echo Use all Intel GPUs, including iGPU ^& dGPU
  set "GPUS_SETTING=-sm %SPLIT_MODE%"
)

echo run cmd: ZES_ENABLE_SYSMAN=1 %BIN_FILE% -m %MODEL_FILE% -no-cnv -p "%INPUT_PROMPT%" -n 200 -e -ngl %NGL% -s %SEED% -c %CONTEXT% %GPUS_SETTING% -lv %LOG_VERBOSE% --mmap
set "ZES_ENABLE_SYSMAN=1"
%BIN_FILE% -m "%MODEL_FILE%" -no-cnv -p "%INPUT_PROMPT%" -n 200 -e -ngl %NGL% -s %SEED% -c %CONTEXT% %GPUS_SETTING% -lv %LOG_VERBOSE% --mmap

endlocal

