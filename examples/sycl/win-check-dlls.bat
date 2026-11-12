::  MIT license
::  SPDX-License-Identifier: MIT

@echo off
setlocal

set "BINDIR=%~1"
if "%BINDIR%"=="" set "BINDIR=build\bin"

set "PWSH=powershell"
where pwsh >nul 2>nul && set "PWSH=pwsh"

"%PWSH%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0win-check-dlls.ps1" -BinDir "%BINDIR%"
exit /b %ERRORLEVEL%
