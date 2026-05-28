#requires -Version 5
<#
.SYNOPSIS
  Build llama.cpp for Snapdragon Windows ARM and verify the HTP signature.

.DESCRIPTION
  Assumes setup_build_env.ps1 has already exported the toolchain env vars
  (OPENCL_SDK_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT, WINDOWS_SDK_BIN,
  VCVARS_BAT, VCVARS_ARGS, LLVM_BIN). Sources vcvars into the current
  PowerShell process so cmake's child processes inherit PATH/INCLUDE/LIB.

  Drives the canonical sequence from
  llama.cpp/docs/backend/snapdragon/windows.md:

    cmake --preset arm64-windows-snapdragon-release -B build-wos
    cmake --build build-wos
    cmake --install build-wos --prefix pkg-snapdragon
    signtool.exe verify /v /pa pkg-snapdragon\lib\libggml-htp.cat

  Returns the signtool result via -SigntoolResult variable in the caller's
  scope ("ok" / "fail") so run_build.ps1 can record it in build_meta.json.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$SourceDir,
    [Parameter(Mandatory)] [string]$PfxPath,
    [string]$BuildDir = 'build-wos',
    [string]$InstallPrefix = 'pkg-snapdragon'
)

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "=== [build_llamacpp] $msg ===" }

if (-not (Test-Path -LiteralPath $SourceDir)) { throw "SourceDir not found: $SourceDir" }
if (-not (Test-Path -LiteralPath $PfxPath))   { throw "PfxPath not found: $PfxPath" }

$resolvedSource = (Resolve-Path -LiteralPath $SourceDir).Path
$resolvedPfx    = (Resolve-Path -LiteralPath $PfxPath).Path

$env:HEXAGON_HTP_CERT = $resolvedPfx
Write-Host "HEXAGON_HTP_CERT = $env:HEXAGON_HTP_CERT"

foreach ($var in 'OPENCL_SDK_ROOT','HEXAGON_SDK_ROOT','HEXAGON_TOOLS_ROOT','WINDOWS_SDK_BIN','VCVARS_BAT','LLVM_BIN') {
    $val = [Environment]::GetEnvironmentVariable($var, 'Process')
    if (-not $val) { throw "Required env var $var not set; run setup_build_env.ps1 first" }
    Write-Host "$var = $val"
}

Write-Section 'sourcing vcvars into current process'
# Write a small wrapper .bat so cmd doesn't have to deal with PS-quoted paths
# that contain spaces. Then cmd /c "wrapper.bat && set" gives us the env.
$wrapper = Join-Path $env:TEMP "vcvars-wrap-$(Get-Random).bat"
$line1 = '@echo off'
$line2 = 'call "' + $env:VCVARS_BAT + '" ' + $env:VCVARS_ARGS
$line3 = 'set'
Set-Content -LiteralPath $wrapper -Value @($line1, $line2, $line3) -Encoding ascii
try {
    $envDump = cmd /c "`"$wrapper`""
    foreach ($line in $envDump) {
        if ($line -match '^(.+?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
} finally {
    Remove-Item -LiteralPath $wrapper -Force -ErrorAction SilentlyContinue
}
$env:PATH = "$env:LLVM_BIN;$env:PATH"

function Invoke-Cmake {
    [CmdletBinding()]
    param(
        [string[]]$Args,
        [string]$LogTag
    )
    $logDir = 'C:\Temp\QDC_Logs'
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $combined = Join-Path $logDir "cmake-$LogTag.log"
    Write-Host "$ cmake $($Args -join ' ')"
    # & with cmd-style redirection to merge stderr into stdout in-process
    # (Start-Process strips inherited env). PS 2>&1 pipe sometimes inserts
    # ErrorRecord objects; use cmd /c for a clean byte-stream capture.
    $argLine = ($Args | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    cmd /c "cmake $argLine 1>`"$combined`" 2>&1"
    $rc = $LASTEXITCODE
    Write-Host "--- cmake $LogTag tail (last 30 lines, rc=$rc) ---"
    Get-Content $combined -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    if ($rc -ne 0) {
        throw "cmake $LogTag failed: $rc"
    }
}

Push-Location $resolvedSource
try {
    Write-Section 'staging CMakeUserPresets.json'
    $presetSrc = Join-Path $resolvedSource 'docs\backend\snapdragon\CMakeUserPresets.json'
    $presetDst = Join-Path $resolvedSource 'CMakeUserPresets.json'
    if (Test-Path -LiteralPath $presetSrc) {
        Copy-Item -LiteralPath $presetSrc -Destination $presetDst -Force
    } else {
        Write-Host "WARN: $presetSrc not found; assuming preset already at repo root"
    }

    # llama.cpp's `arm64-windows-snapdragon` preset omits -fno-finite-math-only
    # in CMAKE_C_FLAGS / CMAKE_CXX_FLAGS, which makes ggml-cpu/vec.h fail to
    # compile (it explicitly requires non-finite math). Patch every flag line
    # that has `-ffp-model=fast -flto` but not `-fno-finite-math-only`.
    if (Test-Path -LiteralPath $presetDst) {
        $text = Get-Content -LiteralPath $presetDst -Raw
        # Match any flag string containing `-ffp-model=fast` followed eventually
        # by `-flto`, but only when -fno-finite-math-only is missing. Insert it
        # just before -flto. Repeat until no more replacements happen.
        $patched = $text
        while ($patched -match '("[^"]*-ffp-model=fast(?![^"]*-fno-finite-math-only)[^"]*?) -flto') {
            $patched = $patched -replace '("[^"]*-ffp-model=fast(?![^"]*-fno-finite-math-only)[^"]*?) -flto', '$1 -fno-finite-math-only -flto'
        }
        if ($patched -ne $text) {
            Set-Content -LiteralPath $presetDst -Value $patched -Encoding utf8
            Write-Host 'patched preset: added -fno-finite-math-only to all -ffp-model=fast flag lines'
        }
    }

    Write-Section "cmake configure (preset arm64-windows-snapdragon-release)"
    # Pin HOST_CXX_COMPILER to the ARM64-native clang++ explicitly. cmake's
    # find_program(NAMES g++ clang++) inside tools/ui/CMakeLists.txt
    # otherwise picks the i686 clang from VC\Tools\Llvm\bin (whose default
    # target is i686-pc-windows-msvc and which therefore can't link against
    # the arm64 vcvars-provided libs, producing LNK2019 unresolved symbols).
    $hostClangPp = Join-Path $env:LLVM_BIN 'clang++.exe'
    Invoke-Cmake -Args @(
        '--preset','arm64-windows-snapdragon-release','-B',$BuildDir,
        "-DHOST_CXX_COMPILER=$hostClangPp"
    ) -LogTag 'configure'

    Write-Section "cmake build"
    Invoke-Cmake -Args @('--build',$BuildDir) -LogTag 'build'

    Write-Section "cmake install --prefix $InstallPrefix"
    Invoke-Cmake -Args @('--install',$BuildDir,'--prefix',$InstallPrefix) -LogTag 'install'

    Write-Section 'verifying HTP cat signature'
    $cat = Join-Path $InstallPrefix 'lib\libggml-htp.cat'
    if (-not (Test-Path -LiteralPath $cat)) {
        throw "libggml-htp.cat missing at $cat after install"
    }
    $signtool = Join-Path $env:WINDOWS_SDK_BIN 'signtool.exe'
    if (-not (Test-Path -LiteralPath $signtool)) {
        throw "signtool.exe not found at $signtool"
    }
    $verifyOut = & $signtool verify /v /pa $cat 2>&1 | Out-String
    Write-Host $verifyOut
    if ($verifyOut -match 'Successfully verified') {
        $script:SigntoolResult = 'ok'
    } else {
        $script:SigntoolResult = 'fail'
        throw "signtool verify did not report success"
    }

    Write-Section 'install tree summary'
    Get-ChildItem -LiteralPath (Join-Path $InstallPrefix 'lib') | Format-Table -AutoSize | Out-String | Write-Host
    Get-ChildItem -LiteralPath (Join-Path $InstallPrefix 'bin') | Select-Object -First 10 | Format-Table -AutoSize | Out-String | Write-Host
}
finally {
    Pop-Location
}
