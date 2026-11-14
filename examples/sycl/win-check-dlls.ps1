# Check that all runtime dependencies of the DLLs in a directory are present.
# Use after a oneAPI upgrade on build\bin or on an extracted release package.
# Usage: win-check-dlls.ps1 [-BinDir <dir>] [-Dumpbin <path>]
param(
    [string]$BinDir = "build\bin",
    [string]$Dumpbin = ""
)

if (-not (Test-Path $BinDir)) {
    Write-Host "ERROR: directory not found: $BinDir"
    exit 1
}

if (-not $Dumpbin) {
    $cmd = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $Dumpbin = $cmd.Source
    } else {
        $Dumpbin = Get-Item "C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    }
}
if (-not $Dumpbin) {
    Write-Host "ERROR: dumpbin.exe not found, install Visual Studio or run from a developer prompt"
    exit 1
}

$dlls = Get-ChildItem $BinDir -Filter *.dll
if (-not $dlls) {
    Write-Host "ERROR: no DLLs found in: $BinDir"
    exit 1
}

$shipped = $dlls | ForEach-Object { $_.Name.ToLower() }
$driverProvided = @("ze_loader.dll") # provided by the Intel GPU driver on the target machine
$problems = @{}

foreach ($dll in $dlls) {
    foreach ($line in (& $Dumpbin /dependents $dll.FullName 2>$null)) {
        if ($line -match "^\s+(\S+\.dll)\s*$") {
            $dep = $Matches[1].ToLower()
            if ($dep -like "api-ms-*") { continue } # OS-guaranteed API set
            if ($driverProvided -contains $dep) { continue }
            if ($shipped -contains $dep) { continue }
            if (Test-Path (Join-Path $env:SystemRoot "System32\$dep")) { continue }
            if (-not $problems.ContainsKey($dep)) { $problems[$dep] = @() }
            $problems[$dep] += $dll.Name
        }
    }
}

if ($problems.Count -gt 0) {
    Write-Host "DLL dependencies missing from $BinDir :"
    foreach ($dep in $problems.Keys) {
        $onPath = Get-Command $dep -ErrorAction SilentlyContinue
        $hint = ""
        if ($onPath) { $hint = " (on PATH: $($onPath.Source))" }
        Write-Host "  $dep  <- required by $($problems[$dep] -join ', ')$hint"
    }
    Write-Host ""
    Write-Host "The directory is not self-contained. Add the missing DLLs to the release package."
    exit 1
}

Write-Host "All DLL dependencies are present in $BinDir"
exit 0
