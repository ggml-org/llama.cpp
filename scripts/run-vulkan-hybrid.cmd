@echo off
setlocal

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022 > NUL
if errorlevel 1 exit /b 1

set "REPO_ROOT=%~dp0.."
set "CHECK_EXE=%REPO_ROOT%\build-sycl-vulkan\bin\hybrid-inference-check.exe"
set "MODEL=%~2"
set "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
set "GGML_VULKAN_HYBRID_FORCE_FAIL="

if /I "%~1"=="off" (
    set "GGML_VULKAN_HYBRID=0"
    set "ONEDNN_VERBOSE="
) else if /I "%~1"=="on" (
    set "GGML_VULKAN_HYBRID=1"
    set "GGML_VULKAN_HYBRID_FORCE_FAIL="
    set "ONEDNN_VERBOSE=2"
) else if /I "%~1"=="fail" (
    set "GGML_VULKAN_HYBRID=1"
    set "GGML_VULKAN_HYBRID_FORCE_FAIL=1"
    set "ONEDNN_VERBOSE=2"
) else (
    echo Usage: %~nx0 off^|on^|fail [model.gguf]
    exit /b 2
)

set "PATH=%REPO_ROOT%\build-sycl-vulkan\bin;C:\VulkanSDK\1.4.357.0\Bin;C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin;C:\Program Files (x86)\Intel\oneAPI\dnnl\2026.0\bin;C:\Program Files (x86)\Intel\oneAPI\mkl\2026.1\bin;C:\Program Files (x86)\Intel\oneAPI\tbb\2023.1\bin;%PATH%"

if "%MODEL%"=="" (
    "%CHECK_EXE%"
) else (
    "%CHECK_EXE%" "%MODEL%"
)
exit /b %ERRORLEVEL%
