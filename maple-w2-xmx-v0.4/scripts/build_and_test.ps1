[CmdletBinding()]
param(
 [string]$Compiler='', [string]$Builtins='', [string]$Python='python', [string]$Device='A750',
 [ValidateSet('smoke','quick','full')][string]$Suite='quick', [string]$Model='', [string]$CapsuleDir='',
 [int]$Layer=0, [ValidateRange(1,10000)][int]$Repeats=28,
 [ValidateRange(0,128)][int]$ScrubMiB=0, [switch]$NoBuild, [switch]$MoeOnly
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$Out=Join-Path $Root "build/results/$Stamp-$Suite"
New-Item -ItemType Directory -Force $Out | Out-Null
$Transcript=$false
try{
 Start-Transcript -Path (Join-Path $Out 'run.log') | Out-Null; $Transcript=$true
 if($Model -and $CapsuleDir){throw 'Choose Model or CapsuleDir, not both.'}
 if($CapsuleDir -and -not $MoeOnly){throw 'CapsuleDir requires -MoeOnly; the legacy tensor suite uses -Model.'}
 if(-not $NoBuild){& (Join-Path $PSScriptRoot 'build_windows.ps1') -Compiler $Compiler -Builtins $Builtins}
 $Py=Get-Command $Python -ErrorAction SilentlyContinue
 if(-not $Py){throw 'Python 3 required; standard library only.'}
 & $Py.Source -m unittest discover -s (Join-Path $Root 'tests') -p 'test_*.py'
 if($LASTEXITCODE -ne 0){throw 'Python unit tests failed.'}
 if(-not $MoeOnly){
  $Legacy=Join-Path $Out 'tensor'
  $ArgsLegacy=@((Join-Path $Root 'tools/run_suite.py'),'--exe',(Join-Path $Root 'build/maple-w2-compare.exe'),
    '--suite',$Suite,'--device',$Device,'--out',$Legacy,'--repeats',"$Repeats",'--scrub-mib',"$ScrubMiB")
  if($Model){$ArgsLegacy+=@('--model',$Model,'--layer',"$Layer")}
  & $Py.Source @ArgsLegacy
  if($LASTEXITCODE -ne 0){throw "Tensor suite failed; preserve $Out."}
  if($Model -and $Suite -ne 'smoke'){$CapsuleDir=Join-Path $Legacy 'weights'}
 }
 $Moe=Join-Path $Out 'moe'
 $ArgsMoe=@((Join-Path $Root 'tools/run_moe_suite.py'),'--exe',(Join-Path $Root 'build/maple-moe-compare.exe'),
   '--suite',$Suite,'--device',$Device,'--out',$Moe,'--repeats',"$Repeats",'--scrub-mib',"$ScrubMiB")
 if($CapsuleDir){$ArgsMoe+=@('--capsule-dir',$CapsuleDir)}elseif($Model){$ArgsMoe+=@('--model',$Model,'--layer',"$Layer")}
 & $Py.Source @ArgsMoe
 if($LASTEXITCODE -ne 0){throw "MoE suite failed; preserve $Out."}
 Write-Host "Completed. Results: $Out. No llama DLL/settings changed. This is NOT server TG."
}finally{if($Transcript){Stop-Transcript | Out-Null}}
