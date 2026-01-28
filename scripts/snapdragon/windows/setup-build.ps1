# Requires Run as Administrator is NOT strictly necessary for User-scope env vars,
# but recommended for creating directories in C:\ root if permissions are restricted.

$ErrorActionPreference = "Stop"

# --- Configuration ---
$BaseDir = "C:\Qualcomm"

# SDK 1: Hexagon
$HexagonUrl     = "https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.4.0.2/hexagon-sdk-v6.4.0.2-arm64-wos.tar.xz"
$HexagonParent  = Join-Path $BaseDir "Hexagon_SDK"
$HexagonSdkVersion   = "6.4.0.2"
$HexagonToolsVersion = "19.0.04"
$HexagonSdkTarget    = Join-Path $HexagonParent $HexagonSdkVersion
$HexagonToolsTarget  = Join-Path $HexagonSdkTarget "\tools\HEXAGON_Tools\$HexagonToolsVersion"

# SDK 2: OpenCL
$OpenCLUrl      = "https://github.com/snapdragon-toolchain/opencl-sdk/releases/download/v2.3.2/adreno-opencl-sdk-v2.3.2-arm64-wos.tar.xz"
$OpenCLParent   = Join-Path $BaseDir "OpenCL_SDK"
$OpenCLVersion  = "2.3.2"
$OpenCLTarget   = Join-Path $OpenCLParent $OpenCLVersion

# --- Helper Function ---
function Install-QualcommSDK {
    param (
        [string]$Url,
        [string]$ParentDir,
        [string]$TargetDir,
        [string]$Name
    )

    # 1. Create Parent Directory
    if (-not (Test-Path -Path $ParentDir)) {
        Write-Host "Creating directory: $ParentDir" -ForegroundColor Cyan
        New-Item -Path $ParentDir -ItemType Directory -Force | Out-Null
    }

    # 2. Check for Specific Version Directory
    if (Test-Path -Path $TargetDir) {
        Write-Host "$Name ($TargetDir) already exists. Skipping download." -ForegroundColor Green
    }
    else {
        Write-Host "$Name not found. preparing to download..." -ForegroundColor Yellow

        # Create the target directory to extract into
        New-Item -Path $TargetDir -ItemType Directory -Force | Out-Null

        # Define temporary archive path
        $TempFile = Join-Path $ParentDir "temp_sdk.tar.xz"

        try {
            # Download
            Write-Host "Downloading from: $Url"
            Invoke-WebRequest -Uri $Url -OutFile $TempFile

            # Untar
            # Note: We assume Windows includes tar.exe (Win 10 build 17063+)
            Write-Host "Extracting archive to $TargetDir..."

            # We use -C to extract contents INTO the target directory created above
            tar -xJvf $TempFile -C $TargetDir\..

            Write-Host "Extraction complete." -ForegroundColor Green
        }
        catch {
            Write-Error "Failed to download or extract $Name. Error: $_"
            # Cleanup target dir if failed so script tries again next time
            Remove-Item -Path $TargetDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        finally {
            # Cleanup Archive
            if (Test-Path $TempFile) { Remove-Item $TempFile -Force }
        }
    }
}

# --- Execution ---

# 1. Ensure Base C:\Qualcomm exists
if (-not (Test-Path $BaseDir)) {
    New-Item -Path $BaseDir -ItemType Directory -Force | Out-Null
}

# 2. Run Install Logic
Install-QualcommSDK -Url $HexagonUrl -ParentDir $HexagonParent -TargetDir $HexagonSdkTarget -Name "Hexagon SDK"
Install-QualcommSDK -Url $OpenCLUrl -ParentDir $OpenCLParent -TargetDir $OpenCLTarget -Name "OpenCL SDK"

# --- Environment Variables ---

Write-Host "`nSetting Environment Variables..." -ForegroundColor Cyan

# Set OPENCL_SDK_ROOT
[System.Environment]::SetEnvironmentVariable('OPENCL_SDK_ROOT', $OpenCLTarget, [System.EnvironmentVariableTarget]::User)
$env:OPENCL_SDK_ROOT = $OpenCLTarget # Set for current session as well
Write-Host "OPENCL_SDK_ROOT set to:  $OpenCLTarget"

# Set HEXAGON_SDK_ROOT
[System.Environment]::SetEnvironmentVariable('HEXAGON_SDK_ROOT', $HexagonSdkTarget, [System.EnvironmentVariableTarget]::User)
$env:HEXAGON_SDK_ROOT = $HexagonSdkTarget # Set for current session as well
Write-Host "HEXAGON_SDK_ROOT set to: $HexagonSdkTarget"

# Set HEXAGON_SDK_ROOT
[System.Environment]::SetEnvironmentVariable('HEXAGON_TOOLS_ROOT', $HexagonToolsTarget, [System.EnvironmentVariableTarget]::User)
$env:HEXAGON_TOOLS_ROOT = $HexagonToolsTarget # Set for current session as well
Write-Host "HEXAGON_TOOLS_ROOT set to: $HexagonToolsTarget"
