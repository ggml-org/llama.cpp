# Compile-time profiling using clang -ftime-trace + ClangBuildAnalyzer.
#
# Usage:
#   .\scripts\build-profile.ps1 [-Full] [-Jobs N]
#
# -Full : include Server, Tools, and Tests (default: minimal build)
# -Jobs :  number of parallel jobs (default: all cores)
#
# Requires ClangBuildAnalyzer:
#   https://github.com/aras-p/ClangBuildAnalyzer

param(
    [switch]$Full,
    [int]$Jobs = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

if ($Full) {
    $BuildDir = Join-Path $RootDir "build-profile-full"
    $Report = Join-Path $BuildDir "profile-report-full.txt"
} else {
    $BuildDir = Join-Path $RootDir "build-profile-baseline"
    $Report = Join-Path $BuildDir "profile-report.txt"
}

$OutputBin = Join-Path $BuildDir "clang_analysis.bin"

if (-not (Get-Command clang++ -ErrorAction SilentlyContinue)) {
    Write-Error "clang++ not found"
    exit 1
}

if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "ninja not found (required so cmake does not fall back to the Visual Studio/MSVC generator)"
    exit 1
}

if (-not (Get-Command ClangBuildAnalyzer -ErrorAction SilentlyContinue)) {
    Write-Error "ClangBuildAnalyzer not found`n  https://github.com/aras-p/ClangBuildAnalyzer/releases"
    exit 1
}

$ClangVer = (clang++ --version | Select-Object -First 1)
Write-Host "compiler : $ClangVer"
Write-Host "build dir: $BuildDir"
Write-Host "output   : $OutputBin"
Write-Host "jobs     : $Jobs"
Write-Host ""

if (Get-Command ccache -ErrorAction SilentlyContinue) {
    Write-Host "clearing ccache..."
    ccache -C -z
}

$env:CCACHE_DISABLE = "1"

$TestsFlag = if ($Full) { "ON" } else { "OFF" }
$ToolsFlag = if ($Full) { "ON" } else { "OFF" }
$ServerFlag = if ($Full) { "ON" } else { "OFF" }

cmake --fresh `
    -S $RootDir `
    -B $BuildDir `
    -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_C_FLAGS="-ftime-trace" `
    -DCMAKE_CXX_FLAGS="-ftime-trace" `
    -DGGML_CCACHE=OFF `
    -DGGML_OPENMP=ON `
    -DGGML_NATIVE=OFF `
    "-DLLAMA_BUILD_TESTS=$TestsFlag" `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    "-DLLAMA_BUILD_TOOLS=$ToolsFlag" `
    "-DLLAMA_BUILD_SERVER=$ServerFlag" `
    -DLLAMA_BUILD_APP=OFF

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$StrayTrace = Join-Path $RootDir "-.json"
if (Test-Path $StrayTrace) {
    Remove-Item $StrayTrace -Force
}

Write-Host ""
Write-Host "Initializing ClangBuildAnalyzer..."
ClangBuildAnalyzer --start $BuildDir
Write-Host ""

Write-Host "building..."
Write-Host ""

$StartTime = Get-Date

cmake --build $BuildDir --clean-first -j $Jobs

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Elapsed = (Get-Date) - $StartTime

Write-Host ""
Write-Host ("build time: {0}s ({1}m {2}s)" -f [int]$Elapsed.TotalSeconds, [int]$Elapsed.TotalMinutes, $Elapsed.Seconds)
Write-Host ""

Write-Host "Aggregating profile metrics..."
ClangBuildAnalyzer --stop $BuildDir $OutputBin | Out-Null

Write-Host ""
Write-Host ("=" * 80)

$TUs = "?"
if (Test-Path $Report) {
    $Match = Select-String -Path $Report -Pattern "Compilation \((\d+)" | Select-Object -First 1
    if ($Match) { $TUs = $Match.Matches[0].Groups[1].Value }
}

ClangBuildAnalyzer --analyze $OutputBin | Tee-Object -FilePath $Report

Write-Host ""
Write-Host "translation units: $TUs"
Write-Host ""
Write-Host "largest trace files (top 20 by size):"

Get-ChildItem -Path $BuildDir -Recurse -Filter "*.json" |
    Where-Object { $_.Name -ne "compile_commands.json" } |
    Sort-Object Length -Descending |
    Select-Object -First 20 |
    ForEach-Object { "{0,8:F1} KB  {1}" -f ($_.Length / 1024), $_.FullName }

Write-Host ""
Write-Host "ClangBuildAnalyzer report was generated: $Report"
