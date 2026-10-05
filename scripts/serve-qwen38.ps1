<#
.SYNOPSIS
    Launch the specialized text-only Qwen3.8-27B (IQ3_S) llama-server on RTX 3090.
.DESCRIPTION
    Serves the Unsloth Qwen3.8-27B-UD-IQ3_S GGUF with the tuned, validated profile:
    262144 context, Q8_0 K/V cache, Flash Attention, full GPU offload.
    Binds to 127.0.0.1 only. This is a text-only build; --mmproj is refused.
#>
[CmdletBinding()]
param(
    [string]$ModelPath = "C:\Users\19082\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-IQ3_S.gguf",
    [string]$ServerExe = "$PSScriptRoot\..\build-qwen38-prod\bin\llama-server.exe",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8080,
    [int]$CtxSize = 262144,
    [int]$UBatch = 512,
    [int]$Threads = 8,
    [int]$NGpuLayers = 99
)

$ErrorActionPreference = "Stop"

# fall back to the baseline build if the production binary is not present
if (-not (Test-Path -LiteralPath $ServerExe)) {
    $fallback = Join-Path $PSScriptRoot "..\build-qwen38-baseline\bin\llama-server.exe"
    if (Test-Path -LiteralPath $fallback) {
        $ServerExe = $fallback
    } else {
        throw "llama-server.exe not found. Build it first (preset qwen38-prod or qwen38-baseline). Looked for: $ServerExe"
    }
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model not found: $ModelPath"
}

# CUDA 12.5 runtime DLLs (cudart64_12, cublas64_12, cublasLt64_12) must be reachable
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if (Test-Path -LiteralPath $cudaBin) {
    $env:PATH = "$cudaBin;$env:PATH"
}

$serverArgs = @(
    "-m", $ModelPath,
    "--host", $ListenHost,
    "--port", $Port,
    "--ctx-size", $CtxSize,
    "--parallel", 1,
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--n-gpu-layers", $NGpuLayers,
    "--fit", "off",
    "--batch-size", 2048,
    "--ubatch-size", $UBatch,
    "--threads", $Threads,
    "--no-context-shift",
    "--jinja",
    "--no-webui"
)

Write-Host "Starting Qwen3.8-27B text-only server on http://$ListenHost`:$Port" -ForegroundColor Cyan
Write-Host "  model : $ModelPath"
Write-Host "  exe   : $ServerExe"
Write-Host "  ctx   : $CtxSize  kv: q8_0  fa: on  ngl: $NGpuLayers  ub: $UBatch  threads: $Threads"

# llama-server logs to stderr; do not let PowerShell treat that as a terminating error
$ErrorActionPreference = "Continue"
& $ServerExe @serverArgs
exit $LASTEXITCODE
