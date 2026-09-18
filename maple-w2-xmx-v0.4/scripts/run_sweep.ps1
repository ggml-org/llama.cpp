[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Weights,
    [string]$Weights2='',
    [ValidateSet('f32','f16')][string]$InputType='f32',
    [ValidateSet(0,1)][int]$PerSelection=0,
    [int]$Tokens=1,[int]$TopK=8,[int]$Repeats=30,[string]$Device='A750',
    [string]$XFile='', [string]$IdsFile=''
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Exe=Join-Path $Root 'build/maple-w2-bench.exe'
if(-not(Test-Path $Exe)){throw 'Run build_windows.ps1 first.'}
$Weights=(Resolve-Path $Weights).Path
if($Weights2){$Weights2=(Resolve-Path $Weights2).Path}
if($XFile){$XFile=(Resolve-Path $XFile).Path}
if($IdsFile){$IdsFile=(Resolve-Path $IdsFile).Path}
$Logs=Join-Path $Root ('build/sweep-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force $Logs | Out-Null
# Read only the K field from the capsule header to choose legal split factors.
$Stream=[IO.File]::OpenRead($Weights);$Reader=[IO.BinaryReader]::new($Stream)
try{$null=$Stream.Seek(12,[IO.SeekOrigin]::Begin);$K=$Reader.ReadUInt32()}finally{$Reader.Dispose()}
$Cases=@(@{Mode='fma';Layout='native';Split=1})
foreach($Layout in @('native','tile8')) {foreach($S in @(1,2,4,8)) {
    if((($K/256)%$S)-eq 0){$Cases+=@{Mode='xmx';Layout=$Layout;Split=$S}}
}}
foreach($Case in $Cases){
    $Name="$($Case.Mode)-$($Case.Layout)-s$($Case.Split)-q$Tokens"
    $Args=@('--weights',$Weights,'--mode',$Case.Mode,'--layout',$Case.Layout,
      '--input',$InputType,'--split',"$($Case.Split)",'--tokens',"$Tokens",'--topk',"$TopK",
      '--per-selection',"$PerSelection",'--device',$Device,'--warmup','3','--repeats',"$Repeats",
      '--id-mode','rotate','--csv',(Join-Path $Logs "$Name.csv"))
    if($Weights2){$Args+=@('--weights2',$Weights2)}
    if($XFile){$Args+=@('--x-file',$XFile)}
    if($IdsFile){$Args+=@('--ids-file',$IdsFile)}
    & $Exe @Args 2>&1 | Tee-Object -FilePath (Join-Path $Logs "$Name.log")
    if($LASTEXITCODE -ne 0){throw "Failed: $Name. Do not use its timing as a valid result."}
}
Write-Host "Logs: $Logs"
