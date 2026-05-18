#!/usr/bin/env pwsh
# Basedir on device
$basedir=".\pkg-snapdragon"
$cli_opts=$args
$model="Llama-3.2-3B-Instruct-Q4_0.gguf"
if ($null -ne $env:M) {
    $model=$env:M
}
$device="HTP0"
if ($null -ne $env:D) {
    $device=$env:D
}
if ($null -ne $env:V) {
    $env:GGML_HEXAGON_VERBOSE=$env:V
}
if ($null -ne $env:PROF) {
    $env:GGML_HEXAGON_PROFILE=$env:PROF
}
if ($null -ne $env:OPSTAGE) {
    $env:GGML_HEXAGON_OPSTAGE=$env:OPSTAGE
}
if ($null -ne $env:NHVX) {
    $env:GGML_HEXAGON_NHVX=$env:NHVX
}
if ($null -ne $env:NDEV) {
    $env:GGML_HEXAGON_NDEV=$env:NDEV
}
if ($null -ne $env:HB) {
    $env:GGML_HEXAGON_HOSTBUF=$env:HB
}

$binary    = "$basedir\bin\llama-bench.exe"
$modelPath = "$basedir\..\..\gguf\$model"
$libPath   = "$basedir\lib"

if (-not (Test-Path $basedir)) {
    Write-Error "Base directory not found: $basedir"
    exit 1
}
if (-not (Test-Path $binary)) {
    Write-Error "Binary not found: $binary"
    exit 1
}
if (-not (Test-Path $modelPath)) {
    Write-Error "Model file not found: $modelPath"
    exit 1
}
if (-not (Test-Path $libPath)) {
    Write-Error "Lib directory not found: $libPath"
    exit 1
}

$env:ADSP_LIBRARY_PATH="$basedir\lib"

& "$binary" `
    --mmap 0 -m "$modelPath" `
    --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 `
    --batch-size 128 -ngl 99 --device $device $cli_opts

if ($LASTEXITCODE -ne 0) {
    Write-Error "llama-bench.exe failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
