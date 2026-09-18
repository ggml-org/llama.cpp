@echo off
setlocal
where cl.exe >nul 2>nul
if errorlevel 1 goto setup
where icpx.exe >nul 2>nul
if not errorlevel 1 goto run
where icx.exe >nul 2>nul
if not errorlevel 1 goto run
:setup
if defined ONEAPI_ROOT if exist "%ONEAPI_ROOT%\setvars.bat" (
    call "%ONEAPI_ROOT%\setvars.bat"
    goto run
)
if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    goto run
)
echo ERROR: use an initialized oneAPI + MSVC developer shell.
exit /b 1
:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_and_test.ps1" %*
exit /b %errorlevel%
