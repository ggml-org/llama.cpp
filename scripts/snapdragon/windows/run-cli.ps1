
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

if ($null -ne $env:SCHED) {
    $env:GGML_SCHED_DEBUG=$env:SCHED; $cli_opts="$cli_opts -v"
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

$env:ADSP_LIBRARY_PATH="$basedir\lib"

& "$basedir\bin\llama-completion.exe" `
    --no-mmap -no-cnv -m $basedir\..\..\gguf\$model `
    --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 `
    --ctx-size 8192 --ubatch-size 128 -fa on `
    -ngl 99 --device $device $cli_opts
