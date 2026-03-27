
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

if ($null -ne $env:E) {
    $env:GGML_HEXAGON_EXPERIMENTAL=$env:E
}

if ($null -ne $env:PROF) {
    $env:GGML_HEXAGON_PROFILE=$env:PROF; $env:GGML_HEXAGON_OPSYNC=1
}

if ($null -ne $env:OPMASK) {
    $env:GGML_HEXAGON_OPMASK=$env:OPMASK
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

$env:ADSP_LIBRARY_PATH="$basedir\lib"

& "$basedir\bin\llama-bench.exe" `
    --mmap 0 -m $basedir\..\..\gguf\$model `
    --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 `
    --batch-size 128 -ngl 99 --device $device $cli_opts
