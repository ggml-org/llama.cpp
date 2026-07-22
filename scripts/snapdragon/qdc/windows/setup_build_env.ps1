#requires -Version 5
<#
.SYNOPSIS
  Provision the Snapdragon Windows-ARM build toolchain for llama.cpp on a
  one-shot QDC device.

.DESCRIPTION
  Installs Adreno OpenCL SDK, Hexagon SDK, and the Windows Driver Kit (WDK),
  then discovers Visual Studio + Windows SDK paths. Exports the env vars
  required by the `arm64-windows-snapdragon-release` cmake preset into the
  current process:

    OPENCL_SDK_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT,
    WINDOWS_SDK_BIN, VCVARS_BAT, VCVARS_ARGS, LLVM_BIN

  When -ProbeOnly is given, prints the current state of every dependency
  and the testsigning flag, then exits without changing anything.

  Designed to be dot-sourced from run_build.ps1 so env vars persist:
    . .\setup_build_env.ps1
    . .\build_llamacpp.ps1
#>
[CmdletBinding()]
param(
    [switch]$ProbeOnly,
    [string]$OpenCLVersion       = '2.3.2',
    [string]$HexagonVersion      = '6.4.0.2',
    [string]$HexagonToolsVersion = '19.0.04'
)

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "=== [setup_build_env] $msg ===" }

# --- Probe ------------------------------------------------------------------

function Get-WinSdkArm64Bin {
    Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.*\arm64' `
        -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

function Get-Vcvars($arch = 'arm64') {
    $patternsArch = @(
        "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars$arch.bat",
        "C:\BuildTools\VC\Auxiliary\Build\vcvars$arch.bat"
    )
    $vcvars = $null
    foreach ($pat in $patternsArch) {
        $vcvars = Get-ChildItem $pat -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
        if ($vcvars) { break }
    }
    $vcvarsArgs = ''
    if (-not $vcvars) {
        $patternsAll = @(
            'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat',
            'C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
        )
        foreach ($pat in $patternsAll) {
            $vcvars = Get-ChildItem $pat -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
            if ($vcvars) { break }
        }
        $vcvarsArgs = $arch
    }
    [pscustomobject]@{ Bat = $vcvars; Args = $vcvarsArgs }
}

function Get-LlvmBin {
    # Resolve the bin DIRECTORY itself, not its child files. Glob with a
    # trailing wildcard then take the parent directory.
    # Prefer the ARM64-native clang (under VC\Tools\Llvm\ARM64\bin) on
    # ARM64 Windows so host-side `clang++` defaults to arm64-pc-windows-msvc
    # (the x86 clang at VC\Tools\Llvm\bin\ defaults to i686-pc-windows-msvc
    # and produces unresolved-symbol link failures with the arm64 vcvars).
    $patterns = @(
        'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\Llvm\ARM64\bin\clang.exe',
        'C:\BuildTools\VC\Tools\Llvm\ARM64\bin\clang.exe',
        'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\Llvm\bin\clang.exe',
        'C:\BuildTools\VC\Tools\Llvm\bin\clang.exe'
    )
    foreach ($pat in $patterns) {
        $hit = Get-ChildItem $pat -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.DirectoryName }
    }
    return $null
}

function Get-TestsigningState {
    $output = & bcdedit /enum '{current}' 2>&1 | Out-String
    if ($output -match '(?im)^\s*testsigning\s+(\w+)') {
        return $matches[1]
    }
    return 'unknown'
}

function Probe {
    $openclRoot       = "C:\Qualcomm\OpenCL_SDK\$OpenCLVersion"
    $hexagonRoot      = "C:\Qualcomm\Hexagon_SDK\$HexagonVersion"
    $hexagonToolsRoot = Join-Path $hexagonRoot "tools\HEXAGON_Tools\$HexagonToolsVersion"
    $winsdkArm64      = Get-WinSdkArm64Bin
    $vc               = Get-Vcvars
    $llvmBin          = Get-LlvmBin
    $wdkCatalogs      = 'C:\Program Files (x86)\Windows Kits\10\Catalogs'
    $testsigning      = Get-TestsigningState

    function MarkPath($p) { if ($p -and (Test-Path -LiteralPath $p)) { 'present' } else { 'MISSING' } }

    [pscustomobject]@{
        OpenCLSdk        = MarkPath $openclRoot
        HexagonSdk       = MarkPath $hexagonRoot
        HexagonTools     = MarkPath $hexagonToolsRoot
        WinSdkArm64Bin   = if ($winsdkArm64)   { $winsdkArm64 }   else { 'MISSING' }
        VcvarsBat        = if ($vc.Bat)        { "$($vc.Bat) $($vc.Args)".Trim() } else { 'MISSING' }
        LlvmBin          = if ($llvmBin)       { $llvmBin }       else { 'MISSING' }
        WdkCatalogs      = MarkPath $wdkCatalogs
        TestSigning      = $testsigning
        Cmake            = if (Get-Command cmake -ErrorAction SilentlyContinue)    { (Get-Command cmake).Source }    else { 'MISSING' }
        Git              = if (Get-Command git -ErrorAction SilentlyContinue)      { (Get-Command git).Source }      else { 'MISSING' }
        Python           = if (Get-Command python -ErrorAction SilentlyContinue)   { (Get-Command python).Source }   else { 'MISSING' }
        Signtool         = if ($winsdkArm64) {
                               $st = Join-Path $winsdkArm64 'signtool.exe'
                               if (Test-Path -LiteralPath $st) { $st } else { 'MISSING' }
                           } else { 'MISSING' }
    }
}

# --- Install ----------------------------------------------------------------

function Install-Tarball($url, $target) {
    if (Test-Path -LiteralPath $target) {
        Write-Host "Already installed: $target"
        return
    }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $archive = Join-Path $env:TEMP ([System.IO.Path]::GetFileName($url))
    Write-Host "Downloading $url"
    & curl.exe -L --fail --retry 3 -o $archive $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $url" }
    Write-Host "Extracting to $target"
    & tar.exe -xf $archive -C $target --strip-components=1
    if ($LASTEXITCODE -ne 0) { throw "tar -xf failed for $archive" }
    Remove-Item $archive -Force
}

function Install-VsBuildTools {
    # Skip if VS is already discoverable.
    if ((Get-Vcvars).Bat) {
        Write-Host 'Visual Studio already installed; skipping VS Build Tools install'
        return
    }
    Write-Section 'installing Visual Studio 2022 Build Tools (offline-friendly bootstrapper)'
    $url    = 'https://aka.ms/vs/17/release/vs_BuildTools.exe'
    $setup  = Join-Path $env:TEMP 'vs_BuildTools.exe'
    $log    = Join-Path $env:TEMP 'vs_buildtools.log'
    & curl.exe -L --fail --retry 3 -o $setup $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $url" }
    $size = (Get-Item $setup).Length
    Write-Host "vs_BuildTools.exe size: $size bytes"
    if ($size -lt 1MB) { throw "vs_BuildTools.exe download too small ($size bytes)" }

    # VCTools workload + ARM64 build tools + bundled clang + matching Win11 SDK + cmake + git.
    # cmake/git come from the workload's "VC.CMake.Project" and Git for Windows components.
    $args = @(
        '--quiet','--wait','--norestart','--nocache',
        '--installPath','C:\BuildTools',
        '--add','Microsoft.VisualStudio.Workload.VCTools',
        '--add','Microsoft.VisualStudio.Component.VC.Tools.ARM64',
        '--add','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '--add','Microsoft.VisualStudio.Component.VC.Llvm.Clang',
        '--add','Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset',
        '--add','Microsoft.VisualStudio.Component.Windows11SDK.26100',
        '--add','Microsoft.VisualStudio.Component.VC.CMake.Project',
        '--add','Microsoft.VisualStudio.Component.Git',
        '--includeRecommended'
    )
    Write-Host "vs_BuildTools.exe $($args -join ' ')"
    $p = Start-Process -FilePath $setup -ArgumentList $args -Wait -PassThru
    Write-Host "vs_BuildTools.exe exit code: $($p.ExitCode)"
    # 0 = success, 3010 = success+reboot pending. Anything else fails.
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
        throw "vs_BuildTools.exe failed with exit code $($p.ExitCode)"
    }
    # Refresh PATH in this process so cmake/git become resolvable without
    # re-launching PS. VS adds %ProgramData%\chocolatey-style shims under
    # C:\BuildTools, but we'll let the cmake preset / vcvars handle PATH;
    # here we just need cmake/git for direct invocation.
    $vsPaths = @(
        'C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin',
        'C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja',
        'C:\BuildTools\VC\Tools\Llvm\bin',
        "$env:ProgramFiles\Git\cmd"
    ) | Where-Object { Test-Path $_ }
    foreach ($p2 in $vsPaths) {
        if ($env:PATH -notlike "*$p2*") { $env:PATH = "$p2;$env:PATH" }
    }
}

function Install-Wdk {
    $marker = Join-Path $env:TEMP 'llama-build-wdk.installed'
    if (Test-Path -LiteralPath $marker) {
        Write-Host 'WDK install already attempted in this session; skipping'
        return
    }
    $wdkUrl = 'https://download.microsoft.com/download/41fb59c2-1723-45f9-a270-96b73ad58233/KIT_BUNDLE_WDK_MEDIACREATION/wdksetup.exe'
    $setup  = Join-Path $env:TEMP 'wdksetup.exe'
    $log    = Join-Path $env:TEMP 'wdksetup.log'
    Write-Host "Downloading WDK installer"
    & curl.exe -L --fail --retry 3 -o $setup $wdkUrl
    if ($LASTEXITCODE -ne 0) { throw "curl failed for WDK installer" }
    $size = (Get-Item $setup).Length
    Write-Host "wdksetup.exe size: $size bytes"
    if ($size -lt 1MB) { throw "wdksetup.exe download too small ($size bytes); URL likely returned HTML" }
    $p = Start-Process -FilePath $setup `
        -ArgumentList '/features','+','/quiet','/norestart','/log',$log `
        -Wait -PassThru
    Write-Host "wdksetup exit code: $($p.ExitCode)"
    # Exit code 3010 = "reboot required" - treat as success (we won't reboot;
    # Inf2Cat works without reboot in geniex's experience).
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
        if (Test-Path $log) {
            Write-Host '--- wdksetup log (tail) ---'
            Get-Content $log -Tail 80
        }
        throw "wdksetup.exe failed with exit code $($p.ExitCode)"
    }
    New-Item -ItemType File -Path $marker -Force | Out-Null
}

function Install-Python {
    # cmake's find_package(Python3) needs a real interpreter (the WindowsApps
    # stub at AppData\Local\Microsoft\WindowsApps\python.exe doesn't qualify
    # - it just opens the Store). Drop a real arm64 embedded distribution
    # under C:\Python311 and prepend it to PATH.
    $marker = 'C:\Python311\python.exe'
    if (Test-Path -LiteralPath $marker) {
        Write-Host "Python already present at $marker"
        return $marker
    }
    Write-Section 'installing Python 3.11.9 (embedded arm64) for cmake find_package(Python3)'
    $url    = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-arm64.zip'
    $zip    = Join-Path $env:TEMP 'python-embed-arm64.zip'
    & curl.exe -L --fail --retry 3 -o $zip $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $url" }
    if (Test-Path -LiteralPath 'C:\Python311') { Remove-Item -Recurse -Force 'C:\Python311' }
    New-Item -ItemType Directory -Path 'C:\Python311' -Force | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath 'C:\Python311' -Force
    Remove-Item $zip -Force
    # Make site-packages discoverable so pip works (matches run_tests.ps1's pattern).
    $pth = 'C:\Python311\python311._pth'
    if (Test-Path -LiteralPath $pth) {
        $content = (Get-Content -LiteralPath $pth -Raw) -replace '#\s*import site', 'import site'
        Set-Content -LiteralPath $pth -Value $content -Encoding ascii
    }
    return $marker
}

function Install-Toolchain {
    $openclRoot  = "C:\Qualcomm\OpenCL_SDK\$OpenCLVersion"
    $hexagonRoot = "C:\Qualcomm\Hexagon_SDK\$HexagonVersion"

    Install-VsBuildTools

    Write-Section 'installing OpenCL + Hexagon SDKs'
    Install-Tarball "https://github.com/snapdragon-toolchain/opencl-sdk/releases/download/v$OpenCLVersion/adreno-opencl-sdk-v$OpenCLVersion-arm64-wos.tar.xz"  $openclRoot
    Install-Tarball "https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v$HexagonVersion/hexagon-sdk-v$HexagonVersion-arm64-wos.tar.xz" $hexagonRoot

    Write-Section 'installing Windows Driver Kit'
    Install-Wdk

    Install-Python
}

# --- Export -----------------------------------------------------------------

function Export-Env {
    $openclRoot       = "C:\Qualcomm\OpenCL_SDK\$OpenCLVersion"
    $hexagonRoot      = "C:\Qualcomm\Hexagon_SDK\$HexagonVersion"
    $hexagonToolsRoot = Join-Path $hexagonRoot "tools\HEXAGON_Tools\$HexagonToolsVersion"

    if (-not (Test-Path -LiteralPath $hexagonToolsRoot)) {
        throw "HEXAGON_TOOLS_ROOT not found: $hexagonToolsRoot"
    }
    $winsdkArm64 = Get-WinSdkArm64Bin
    if (-not $winsdkArm64) {
        throw 'Windows SDK arm64 bin directory not found under C:\Program Files (x86)\Windows Kits\10\bin\'
    }
    $vc = Get-Vcvars
    if (-not $vc.Bat) { throw 'vcvars not found under any Visual Studio install' }
    $llvmBin = Get-LlvmBin
    if (-not $llvmBin) { throw 'VS-bundled Llvm\bin not found' }

    # Process-level env vars so child processes (cmake, ninja, clang) inherit.
    $env:OPENCL_SDK_ROOT    = $openclRoot
    $env:HEXAGON_SDK_ROOT   = $hexagonRoot
    $env:HEXAGON_TOOLS_ROOT = $hexagonToolsRoot
    $env:WINDOWS_SDK_BIN    = $winsdkArm64
    $env:VCVARS_BAT         = $vc.Bat
    $env:VCVARS_ARGS        = $vc.Args
    $env:LLVM_BIN           = $llvmBin

    # vcvars itself doesn't add VS-bundled cmake/ninja; do it ourselves so
    # `cmake` / `ninja` resolve unqualified after Export-Env returns.
    # C:\Python311 is added too so cmake's find_package(Python3) hits a real
    # interpreter, not the WindowsApps Store-stub.
    $extraPath = @(
        'C:\Python311',
        (Join-Path (Split-Path -Parent $vc.Bat) '..\..\..\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'),
        (Join-Path (Split-Path -Parent $vc.Bat) '..\..\..\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'),
        $llvmBin
    ) | ForEach-Object {
        if ($_ -and (Test-Path -LiteralPath $_)) {
            (Resolve-Path -LiteralPath $_).Path
        }
    } | Where-Object { $_ }
    foreach ($p in $extraPath) {
        if (";$env:PATH;" -notlike "*;$p;*") { $env:PATH = "$p;$env:PATH" }
    }

    Write-Host "OPENCL_SDK_ROOT    = $env:OPENCL_SDK_ROOT"
    Write-Host "HEXAGON_SDK_ROOT   = $env:HEXAGON_SDK_ROOT"
    Write-Host "HEXAGON_TOOLS_ROOT = $env:HEXAGON_TOOLS_ROOT"
    Write-Host "WINDOWS_SDK_BIN    = $env:WINDOWS_SDK_BIN"
    Write-Host "VCVARS_BAT         = $env:VCVARS_BAT $env:VCVARS_ARGS"
    Write-Host "LLVM_BIN           = $env:LLVM_BIN"
    Write-Host "PATH (head)        = $($env:PATH.Substring(0, [Math]::Min(300, $env:PATH.Length)))"
}

# --- Main -------------------------------------------------------------------

Write-Section "ProbeOnly=$ProbeOnly"
$probe = Probe
$probe | Format-List | Out-String | Write-Host

if ($ProbeOnly) {
    Write-Host ("INFO: testsigning state = '" + $probe.TestSigning + "' (build-only path doesn't require ON; runtime test images do)")
    return
}

Install-Toolchain
Export-Env

Write-Host ("INFO: testsigning state = '" + (Get-TestsigningState) + "' (build-only path does not require ON)")
