[CmdletBinding()]
param(
    [string]$Compiler='', [string]$Builtins='', [string]$Device='A750',
    [ValidateSet('smoke','quick','full')][string]$Suite='smoke',
    [string]$Out='', [string]$Gate='', [string]$Up='', [string]$Down='',
    [int]$Repeats=28, [switch]$DumpContract, [switch]$BuildOnly
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_windows.ps1') -Compiler $Compiler -Builtins $Builtins -GroupingOnly
if($BuildOnly){return}
if(-not $Out){$Out=Join-Path $Root ('build/architecture-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))}
$PythonArgs=@((Join-Path $Root 'tools/run_architecture_suite.py'),'--exe',(Join-Path $Root 'build/maple-architecture-compare.exe'),
    '--device',$Device,'--suite',$Suite,'--out',$Out,'--repeats',"$Repeats")
if($Gate -or $Up -or $Down){
    if(-not($Gate -and $Up -and $Down)){throw 'Supply Gate, Up and Down together.'}
    $PythonArgs+=@('--gate',$Gate,'--up',$Up,'--down',$Down)
}
if($DumpContract){$PythonArgs+=@('--dump-contract')}
& python @PythonArgs
if($LASTEXITCODE -ne 0){throw "Architecture test failed. Preserve $Out; no fallback or model-level pass implied."}
& python (Join-Path $Root 'tools/analyze_architecture_results.py') $Out
if($LASTEXITCODE -ne 0){throw 'Analysis failed; raw evidence has been preserved.'}
