
#!/usr/bin/env pwsh

# Basedir on device
$basedir=".\pkg-snapdragon"

if ($args.Count -eq 0) {
    Write-Host "No arguments provided.Expected the tool and argument to run."
    exit -1
}

$tool=$args[0]
$cli_opts=@()

if ($args.Count -gt 1) {
    $cli_opts=$args[1..($args.Count - 1)]
    $remainingArgs = $args[1..($args.Count - 1)]
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

& "$basedir\bin\$tool" `
    $cli_opts
