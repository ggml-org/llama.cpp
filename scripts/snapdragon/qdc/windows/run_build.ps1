#requires -Version 5
<#
.SYNOPSIS
  Entry script for the QDC POWERSHELL automated build job.

.DESCRIPTION
  QDC unzips the artifact under C:\Temp\TestContent\ and runs this script
  in its own PowerShell host. Layout expected:
    .\llamacpp-src.zip
    .\CMakeUserPresets.json
    .\ggml-htp-v1.cer
    .\ggml-htp-v1.pfx
    .\setup_build_env.ps1
    .\build_llamacpp.ps1
    .\run_build.ps1   (this file)

  Output:
    C:\Temp\QDC_Logs\build.transcript.log   stdout/stderr
    C:\Temp\QDC_Logs\build_meta.json        machine-readable result
    C:\Temp\QDC_Logs\pkg-snapdragon.zip     compiled product (on success)
    C:\Temp\QDCTestDone.txt                 QDC completion sentinel
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

$QdcLogs = 'C:\Temp\QDC_Logs'
if (-not (Test-Path -LiteralPath $QdcLogs)) {
    New-Item -ItemType Directory -Path $QdcLogs -Force | Out-Null
}

Start-Transcript -Path (Join-Path $QdcLogs 'build.transcript.log') -Force | Out-Null

$startTime    = Get-Date
$signtoolRes  = 'unknown'
$stepFailed   = $null
$pkgZipPath   = $null
$pkgZipSize   = 0
$commitSha    = ''

function Write-Section($msg) { Write-Host "=== [run_build] $msg ===" }

try {
    Write-Section 'environment'
    Write-Host ("date: {0:yyyy-MM-ddTHH:mm:ssZ}" -f $startTime.ToUniversalTime())
    Write-Host "hostname: $env:COMPUTERNAME"
    Write-Host "whoami:   $env:USERNAME"
    Write-Host "cwd:      $ScriptDir"
    Get-ChildItem -Force | Format-Table -AutoSize | Out-String | Write-Host

    Write-Section 'importing HTP signing cert'
    $stepFailed = 'cert-import'
    $cer = Join-Path $ScriptDir 'ggml-htp-v1.cer'
    if (-not (Test-Path -LiteralPath $cer)) { throw "ggml-htp-v1.cer not found in artifact" }
    & certutil.exe -addstore -f Root             $cer | Out-String | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "certutil -addstore Root failed: $LASTEXITCODE" }
    & certutil.exe -addstore -f TrustedPublisher $cer | Out-String | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "certutil -addstore TrustedPublisher failed: $LASTEXITCODE" }

    Write-Section 'extracting llama.cpp source'
    $stepFailed = 'extract-source'
    $srcZip = Join-Path $ScriptDir 'llamacpp-src.zip'
    if (-not (Test-Path -LiteralPath $srcZip)) { throw "llamacpp-src.zip not found" }
    $srcDir = Join-Path $ScriptDir 'llama_cpp'
    Expand-Archive -LiteralPath $srcZip -DestinationPath $srcDir -Force

    # Expand-Archive treats the zip-stored timestamp as local time, which on a
    # device whose timezone is UTC-N puts every extracted file N hours into the
    # future relative to wall-clock. Ninja then loops "manifest still dirty
    # after 100 tries, perhaps system time is not set" and fails. Stamp every
    # extracted file to current time so CMakeLists.txt mtime <= build.ninja
    # mtime by construction.
    $now = Get-Date
    Get-ChildItem -LiteralPath $srcDir -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_.LastWriteTime = $now } catch {}
    }

    # Read commit_sha hint dropped by orchestrator (optional).
    $shaFile = Join-Path $ScriptDir 'commit_sha.txt'
    if (Test-Path -LiteralPath $shaFile) {
        $commitSha = (Get-Content -LiteralPath $shaFile -Raw).Trim()
    }

    # Stage CMakeUserPresets.json at the repo root (build-android/build-linux pattern).
    $presetArtifact = Join-Path $ScriptDir 'CMakeUserPresets.json'
    if (Test-Path -LiteralPath $presetArtifact) {
        Copy-Item -LiteralPath $presetArtifact -Destination (Join-Path $srcDir 'CMakeUserPresets.json') -Force
    }

    Write-Section 'sourcing setup_build_env.ps1'
    $stepFailed = 'setup-env'
    . (Join-Path $ScriptDir 'setup_build_env.ps1')

    Write-Section 'sourcing build_llamacpp.ps1'
    $stepFailed = 'build'
    $pfx = Join-Path $ScriptDir 'ggml-htp-v1.pfx'
    if (-not (Test-Path -LiteralPath $pfx)) { throw "ggml-htp-v1.pfx not found" }
    . (Join-Path $ScriptDir 'build_llamacpp.ps1') -SourceDir $srcDir -PfxPath $pfx
    $signtoolRes = $script:SigntoolResult

    Write-Section 'zipping pkg-snapdragon'
    $stepFailed = 'zip-pkg'
    $pkgDir = Join-Path $srcDir 'pkg-snapdragon'
    if (-not (Test-Path -LiteralPath $pkgDir)) { throw "pkg-snapdragon dir missing after install" }
    $pkgZipPath = Join-Path $QdcLogs 'pkg-snapdragon.zip'
    if (Test-Path -LiteralPath $pkgZipPath) { Remove-Item -LiteralPath $pkgZipPath -Force }
    Compress-Archive -Path (Join-Path $pkgDir '*') -DestinationPath $pkgZipPath -CompressionLevel Optimal
    $pkgZipSize = (Get-Item -LiteralPath $pkgZipPath).Length
    Write-Host "pkg-snapdragon.zip: $pkgZipSize bytes"

    $stepFailed = $null
    $exitCode = 0
}
catch {
    Write-Host "FATAL ($stepFailed): $_"
    Write-Host $_.ScriptStackTrace
    $exitCode = 1
}
finally {
    $endTime = Get-Date
    $meta = [ordered]@{
        signtool_verify = $signtoolRes
        step_failed     = $stepFailed
        duration_s      = [int]($endTime - $startTime).TotalSeconds
        commit_sha      = $commitSha
        host            = $env:COMPUTERNAME
        pkg_zip_size    = $pkgZipSize
    }
    $meta | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $QdcLogs 'build_meta.json') -Encoding ascii

    Stop-Transcript | Out-Null

    # QDC completion sentinel - must be created regardless of success/failure
    # so the job transitions to a terminal state instead of timing out.
    New-Item -ItemType File -Path 'C:\Temp\QDCTestDone.txt' -Force | Out-Null
    exit $exitCode
}
