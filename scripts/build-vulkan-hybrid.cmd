@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "BUILD_DIR=%REPO_ROOT%\build-sycl-vulkan"
set "BRIDGE_SRC=%REPO_ROOT%\ggml\src\ggml-vulkan\ggml-vulkan-onednn.cpp"
set "BRIDGE_OUT=%BUILD_DIR%\bin\ggml-vulkan-onednn.dll"
set "CHECK_SRC=%REPO_ROOT%\tools\vulkan-sdpa-bench\hybrid-inference-check.cpp"
set "CHECK_OUT=%BUILD_DIR%\bin\hybrid-inference-check.exe"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
if errorlevel 1 exit /b 1

if "%DNNLROOT%"=="" (
    echo ERROR: DNNLROOT is not defined.
    exit /b 2
)

cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -DGGML_VULKAN_HYBRID=ON
if errorlevel 1 exit /b 3

cmake --build "%BUILD_DIR%" --target llama-cli llama-debug -j 4
if errorlevel 1 exit /b 4

if not exist "%BUILD_DIR%\bin" mkdir "%BUILD_DIR%\bin"

icx -fsycl /O2 /EHsc /std:c++17 /LD ^
    /I"%DNNLROOT%\include" ^
    "%BRIDGE_SRC%" ^
    "%DNNLROOT%\lib\dnnl.lib" ^
    /Fe"%BRIDGE_OUT%"
if errorlevel 1 exit /b 5

cl /nologo /EHsc /std:c++17 /utf-8 ^
    /I"%REPO_ROOT%\include" /I"%REPO_ROOT%\ggml\include" ^
    "%CHECK_SRC%" ^
    "%BUILD_DIR%\src\llama.lib" ^
    "%BUILD_DIR%\ggml\src\ggml.lib" ^
    "%BUILD_DIR%\ggml\src\ggml-base.lib" ^
    /Fo"%BUILD_DIR%\bin\hybrid-inference-check.obj" /Fe"%CHECK_OUT%"
if errorlevel 1 exit /b 6

cl /nologo /EHsc /I"C:\VulkanSDK\1.4.357.0\Include" /I"%LEVEL_ZERO_V1_SDK_PATH%include" ^
    "%REPO_ROOT%\tools\vulkan-sdpa-bench\external-memory-smoke.cpp" ^
    /link /LIBPATH:"C:\VulkanSDK\1.4.357.0\Lib" /LIBPATH:"%LEVEL_ZERO_V1_SDK_PATH%lib" vulkan-1.lib ze_loader.lib ^
    /Fo"%BUILD_DIR%\bin\external-memory-smoke.obj" /Fe"%BUILD_DIR%\bin\external-memory-smoke.exe"
if errorlevel 1 exit /b 7

echo Hybrid Vulkan build OK
echo Bridge: %BRIDGE_OUT%
echo Check: %CHECK_OUT%
endlocal
