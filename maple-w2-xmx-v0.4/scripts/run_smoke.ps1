[CmdletBinding()]
param([string]$Device='A750')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Exe=Join-Path $Root 'build/maple-w2-bench.exe'
if (-not (Test-Path $Exe)) {throw 'Run build_windows.ps1 first.'}
$Logs=Join-Path $Root 'build/smoke'
New-Item -ItemType Directory -Force $Logs | Out-Null
# Nonzero and boundary expert IDs; weights/activations are synthetic and labeled so.
$Ids=Join-Path $Logs 'ids.i32'
$Memory=[System.IO.MemoryStream]::new()
$Writer=[System.IO.BinaryWriter]::new($Memory)
foreach($id in @(255,0,1,2,3,4,5,6)) {$Writer.Write([int]$id)}
$Writer.Flush(); [System.IO.File]::WriteAllBytes($Ids,$Memory.ToArray()); $Writer.Dispose();$Memory.Dispose()
foreach($Layout in @('native','tile8')) {
  foreach($InputType in @('f32','f16')) {
    foreach($Split in @(1,2)) {
      $Name="$Layout-$InputType-s$Split"
      $Args=@('--mode','xmx','--layout',$Layout,'--input',$InputType,
        '--k','512','--m','64','--experts','256','--tokens','1','--topk','8',
        '--split',"$Split",'--device',$Device,'--synthetic-pair','1',
        '--ids-file',$Ids,'--warmup','2','--repeats','5','--csv',(Join-Path $Logs "$Name.csv"))
      & $Exe @Args 2>&1 | Tee-Object -FilePath (Join-Path $Logs "$Name.log")
      if($LASTEXITCODE -ne 0){throw "GPU smoke failed: $Name. Preserve logs and keep the Vulkan path."}
    }
  }
}
# Full K=2048 and the largest supported split, then expert-specific down inputs.
foreach($Case in @(@{K=2048;M=512;S=8;Per=0},@{K=512;M=2048;S=2;Per=1})) {
    $Name="maple-k$($Case.K)-m$($Case.M)-per$($Case.Per)"
    $Args=@('--mode','xmx','--layout','tile8','--k',"$($Case.K)",'--m',"$($Case.M)",
      '--experts','256','--tokens','1','--topk','8','--split',"$($Case.S)",
      '--per-selection',"$($Case.Per)",'--device',$Device,'--ids-file',$Ids,
      '--warmup','2','--repeats','5','--csv',(Join-Path $Logs "$Name.csv"))
    & $Exe @Args 2>&1 | Tee-Object -FilePath (Join-Path $Logs "$Name.log")
    if($LASTEXITCODE -ne 0){throw "GPU smoke failed: $Name."}
}
Write-Host 'GPU synthetic checks passed on this machine. This is not end-to-end Maple validation.'
