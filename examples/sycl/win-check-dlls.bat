::  MIT license
::  SPDX-License-Identifier: MIT

@echo off
setlocal

set "PWSH=powershell"
where pwsh >nul 2>nul && set "PWSH=pwsh"

if /i "%~1"=="/?" goto :help

"%PWSH%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0win-check-dlls.ps1" %*
exit /b %ERRORLEVEL%

:help
"%PWSH%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0win-check-dlls.ps1" -Help
exit /b 0
