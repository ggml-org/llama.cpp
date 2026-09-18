[CmdletBinding()]
param(
 [string]$Compiler='', [string]$Builtins='', [string]$Python='python', [string]$Device='A750',
 [ValidateSet('smoke','quick','full')][string]$Suite='full', [string]$Model='', [string]$CapsuleDir='',
 [ValidateRange(0,1000)][int]$Layer=0, [ValidateRange(2,10000)][int]$Repeats=28,
 [ValidateRange(0,128)][int]$ScrubMiB=0, [switch]$NoBuild, [switch]$Synthetic, [switch]$MoeOnly
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$Out=Join-Path $Root "build/results/$Stamp-v04-grouping-$Suite"
$Sources=0;if($Model){$Sources++};if($CapsuleDir){$Sources++};if($Synthetic){$Sources++}
if($Sources -gt 1){throw 'Choose exactly one of Model, CapsuleDir, Synthetic.'}
if($Suite -ne 'smoke' -and $Sources -eq 0){throw 'Provide -Model or -CapsuleDir. Use -Synthetic explicitly for synthetic-only tests.'}
New-Item -ItemType Directory -Force $Out | Out-Null
$Transcript=$false
try{
 Start-Transcript -Path (Join-Path $Out 'run.log') | Out-Null;$Transcript=$true
 # v0.4 always runs ONLY the grouping experiment. MoeOnly is accepted for old launch commands.
 if(-not $NoBuild){& (Join-Path $PSScriptRoot 'build_windows.ps1') -Compiler $Compiler -Builtins $Builtins -GroupingOnly}
 $Py=Get-Command $Python -ErrorAction SilentlyContinue
 if(-not $Py){throw 'Python 3.10+ required (standard library only).'}
 & $Py.Source -m unittest discover -s (Join-Path $Root 'tests') -p 'test_*.py'
 if($LASTEXITCODE -ne 0){throw 'Python tests failed.'}
 $RunArgs=@((Join-Path $Root 'tools/run_grouping_suite.py'),'--exe',(Join-Path $Root 'build/maple-grouping-compare.exe'),
  '--suite',$Suite,'--device',$Device,'--out',(Join-Path $Out 'grouping'),'--repeats',"$Repeats",'--scrub-mib',"$ScrubMiB",'--layer',"$Layer")
 if($Model){$RunArgs+=@('--model',$Model)}
 if($CapsuleDir){$RunArgs+=@('--capsule-dir',$CapsuleDir)}
 if($Synthetic){$RunArgs+=@('--synthetic')}
 & $Py.Source @RunArgs
 if($LASTEXITCODE -ne 0){throw "Grouping suite failed. Preserve $Out."}
 Write-Host "Completed: $Out/grouping/q_speedup.csv"
 Write-Host 'Only per-expert job grouping changed. Same-build paired A/B; grouping cost included. No server DLL/settings changed.'
}finally{if($Transcript){Stop-Transcript | Out-Null}}
