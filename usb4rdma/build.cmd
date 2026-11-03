@echo off
rem Build llama-cli.exe and its DLLs from this source tree, with the NetworkDirect (RDMA) transport.
rem
rem   usb4rdma\build.cmd [WORKDIR]
rem
rem WORKDIR defaults to build-usb4rdma at the root of this repository (git ignores it). It receives:
rem   NetworkDirect\   Microsoft's NetworkDirect SDK headers (+ the generated ndstatus.h)
rem   build\bin\       llama-cli.exe and its DLLs: copy this folder to every node
rem
rem Needs git, the Vulkan SDK (VULKAN_SDK set by its installer), and Visual Studio 2022 or its
rem Build Tools with the "Desktop development with C++" workload (cl, cmake, ninja, mc). Run it
rem from a normal prompt or a VS developer prompt; vcvars64 is found with vswhere if cl is not on
rem PATH. Behind a proxy, set HTTPS_PROXY before running it; git honors it.
setlocal EnableExtensions

set "ND_REPO=https://github.com/microsoft/NetworkDirect"
set "ND_COMMIT=043f1bf05e41343cc88dda619ddd86aee60a02fe"

rem llama-server embeds a prebuilt web UI that CMake downloads by build number (b<commit count>).
rem Every commit on top of the base changes that number, and a newer UI does not fit this tree, so
rem the UI is pinned to the one this tree was checked with. Set HF_UI_VERSION first to override.
if not defined HF_UI_VERSION set "HF_UI_VERSION=b9874"

rem The source is the repository this script is in: the folder above usb4rdma\.
for %%i in ("%~dp0..") do set "SRC=%%~fi"
if not exist "%SRC%\CMakeLists.txt" (echo ERROR: %SRC% is not the root of this repository& goto :fail)
set "W=%~1"
if "%W%"=="" set "W=%SRC%\build-usb4rdma"
for %%i in ("%W%") do set "W=%%~fi"
if not exist "%W%" mkdir "%W%" || goto :fail

where git >nul 2>&1 || (echo ERROR: git is not on PATH& goto :fail)
if not defined VULKAN_SDK (echo ERROR: VULKAN_SDK is not set; install the Vulkan SDK and open a new prompt& goto :fail)

where cl >nul 2>&1 && goto :have_vs
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo ERROR: Visual Studio 2022 or its Build Tools not found& goto :fail)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo ERROR: no Visual Studio install with the x64 C++ tools& goto :fail)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || goto :fail
:have_vs
for %%t in (cl cmake ninja mc) do where %%t >nul 2>&1 || (echo ERROR: %%t is not on PATH after vcvars64& goto :fail)

echo === 1/3 NetworkDirect SDK headers
if not exist "%W%\NetworkDirect\.git" git clone %ND_REPO% "%W%\NetworkDirect" || goto :fail
git -C "%W%\NetworkDirect" checkout -q %ND_COMMIT% || goto :fail
rem ndstatus.h is generated from ndstatus.mc; CMake looks for it in out\Release-x64\include.
set "NDINC=%W%\NetworkDirect\out\Release-x64\include"
if not exist "%NDINC%" mkdir "%NDINC%" || goto :fail
mc -h "%NDINC%" -r "%NDINC%" "%W%\NetworkDirect\src\ndutil\ndstatus.mc" || goto :fail

echo === 2/3 configure %SRC%
set "COMMIT="
for /f %%c in ('git -C "%SRC%" rev-parse --short HEAD 2^>nul') do set "COMMIT=%%c"
if defined COMMIT echo     commit %COMMIT%
rem Forward slashes: CMake can read a backslash in a -D value as an escape.
set "NDSDK=%W:\=/%/NetworkDirect"
cmake -S "%SRC%" -B "%W%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_DIST_TP_ND=ON "-DLLAMA_DIST_TP_ND_SDK=%NDSDK%" || goto :fail
findstr /b /c:"LLAMA_DIST_TP_ND:BOOL=ON" "%W%\build\CMakeCache.txt" >nul || (echo ERROR: the NetworkDirect transport is not enabled in this configuration& goto :fail)

echo === 3/3 build
cmake --build "%W%\build" || goto :fail

echo.
echo Done: %W%\build\bin
exit /b 0

:fail
echo.
echo BUILD FAILED
exit /b 1
