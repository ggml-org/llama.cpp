@echo off
rem Build llama.cpp dist-tp with the NetworkDirect (RDMA) transport, from scratch.
rem
rem   build.cmd [WORKDIR]
rem
rem WORKDIR defaults to .\work next to this script. It receives:
rem   NetworkDirect\     Microsoft's NetworkDirect SDK headers (+ the generated ndstatus.h)
rem   llama.cpp-disttp\  the dist-tp source with patches\*.patch applied
rem   build\bin\         llama-cli.exe and its DLLs: copy this folder to every node
rem
rem Needs git, the Vulkan SDK (VULKAN_SDK set by its installer), and Visual Studio 2022 or its
rem Build Tools with the "Desktop development with C++" workload (cl, cmake, ninja, mc). Run it
rem from a normal prompt or a VS developer prompt; vcvars64 is found with vswhere if cl is not on
rem PATH. Behind a proxy, set HTTPS_PROXY before running it; git honors it.
setlocal EnableExtensions

set "ND_REPO=https://github.com/microsoft/NetworkDirect"
set "ND_COMMIT=043f1bf05e41343cc88dda619ddd86aee60a02fe"
set "SRC_REPO=https://github.com/hli25/llama.cpp"
set "SRC_COMMIT=67f2cab9652be2103876b4dcb85daa07943a7341"
rem The source tree after every patch is applied. A different value means a patch went in wrong.
set "EXPECT_TREE=0a932f89c0d705f044554e91111c8460df89fcf2"

set "HERE=%~dp0"
set "PATCHES=%HERE%patches"
set "W=%~1"
if "%W%"=="" set "W=%HERE%work"
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

echo === 1/4 NetworkDirect SDK headers
if not exist "%W%\NetworkDirect\.git" git clone %ND_REPO% "%W%\NetworkDirect" || goto :fail
git -C "%W%\NetworkDirect" checkout -q %ND_COMMIT% || goto :fail
rem ndstatus.h is generated from ndstatus.mc; CMake looks for it in out\Release-x64\include.
set "NDINC=%W%\NetworkDirect\out\Release-x64\include"
if not exist "%NDINC%" mkdir "%NDINC%" || goto :fail
mc -h "%NDINC%" -r "%NDINC%" "%W%\NetworkDirect\src\ndutil\ndstatus.mc" || goto :fail

echo === 2/4 llama.cpp dist-tp source and patches
set "SRC=%W%\llama.cpp-disttp"
if exist "%SRC%" (echo ERROR: %SRC% already exists; remove it or pass another WORKDIR& goto :fail)
rem core.longpaths: some paths in the tree exceed MAX_PATH and the checkout fails without it.
git clone -c core.longpaths=true %SRC_REPO% "%SRC%" || goto :fail
git -C "%SRC%" checkout -q -b usb4rdma %SRC_COMMIT% || goto :fail
rem One patch per call: cmd does not expand wildcards for git. A plain for loop visits the files in
rem name order on NTFS, which is the patch order.
for %%p in ("%PATCHES%\*.patch") do (
  echo     %%~nxp
  git -C "%SRC%" -c user.name=usb4rdma -c user.email=usb4rdma@localhost am -q "%%~fp" || goto :fail
)
set "TREE="
rem "HEAD:" names the root tree of HEAD, with no caret for cmd to eat.
for /f %%t in ('git -C "%SRC%" rev-parse HEAD:') do set "TREE=%%t"
if /i not "%TREE%"=="%EXPECT_TREE%" (echo ERROR: patched tree is %TREE%, expected %EXPECT_TREE%& goto :fail)
echo     source tree %TREE% OK

echo === 3/4 configure
rem Forward slashes: CMake can read a backslash in a -D value as an escape.
set "NDSDK=%W:\=/%/NetworkDirect"
cmake -S "%SRC%" -B "%W%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_DIST_TP_ND=ON "-DLLAMA_DIST_TP_ND_SDK=%NDSDK%" || goto :fail
findstr /b /c:"LLAMA_DIST_TP_ND:BOOL=ON" "%W%\build\CMakeCache.txt" >nul || (echo ERROR: the NetworkDirect transport is not enabled in this configuration& goto :fail)

echo === 4/4 build
cmake --build "%W%\build" || goto :fail

echo.
echo Done: %W%\build\bin
exit /b 0

:fail
echo.
echo BUILD FAILED
exit /b 1
