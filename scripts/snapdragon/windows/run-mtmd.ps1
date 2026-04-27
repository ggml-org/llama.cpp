#!/usr/bin/env pwsh

# Basedir on device
$basedir=".\pkg-snapdragon"

$cli_opts=$args

$model="gemma-3-4b-it-Q4_0.gguf"
if ($null -ne $env:M) {
    $model=$env:M
}

$mmproj="mmproj-F16.gguf"
if ($null -ne $env:MMPROJ) {
    $mmproj=$env:MMPROJ
}

$image=""
if ($null -ne $env:IMG) {
    $image=$env:IMG
}

$device="HTP0"
if ($null -ne $env:D) {
    $device=$env:D
}

if ($null -ne $env:V) {
    $env:GGML_HEXAGON_VERBOSE=$env:V
}

if ($null -ne $env:SCHED) {
    $env:GGML_SCHED_DEBUG=$env:SCHED; $cli_opts="$cli_opts -v"
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

if ($null -ne $env:MTMD_DEVICE) {
    $env:MTMD_BACKEND_DEVICE=$env:MTMD_DEVICE
}

$env:ADSP_LIBRARY_PATH="$basedir\lib"

& "$basedir\bin\llama-mtmd-cli.exe" `
    --no-mmap -m $basedir\..\..\gguf\$model `
    --mmproj $basedir\..\..\gguf\$mmproj `
    --image $basedir\..\..\gguf\$image `
    --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 `
    --ctx-size 8192 --ubatch-size 256 -fa on `
    -ngl 99 --device $device -v $cli_opts
