# Are all ranks running the SAME dist-tp build, the same model and prompt, and the expected RDMA
# driver? There is no version handshake between ranks: a rank on a different build can still
# complete a run and print a plausible t/s. Run this before any multi-node run.
#
# The first node listed is the reference: every other node must match it.
#
# BINARIES. Compared: llama-cli.exe and EVERY .dll in each node's run directory (RunDir, or that
# node's RunDirByNode entry), because llama-cli can load any of them. Not compared: other .exe
# tools, and the Microsoft C++ runtime DLLs a node may carry next to llama-cli (a node without them
# loads its System32 copy); those are listed separately.
#
# DRIVERS. The image each driver SERVICE actually loads is hashed on every node. If
# $CLUSTER_CFG.DriverSha256 lists known-good hashes, anything else fails: an OS update can replace
# these silently, and every RDMA run then goes out over TCP.
param(
  # Empty = every node in nodes.ps1; Model and Prompt default from nodes.ps1 too.
  [string]$RankNodes = '',
  [string]$Model  = '',
  [string]$Prompt = ''
)
. "$PSScriptRoot\nodes.ps1"
if(-not $RankNodes){ $RankNodes = @($CLUSTER_NODES.Keys) -join ',' }
if(-not $Model)    { $Model     = $CLUSTER_CFG.Model }
if(-not $Prompt)   { $Prompt    = $CLUSTER_CFG.Prompt }
$nodeList = @(Get-NodeList $RankNodes)
# The same rule as run_disttp.ps1: a node's RunDirByNode entry overrides RunDir.
function RunDirFor([string]$n){ if($CLUSTER_CFG.RunDirByNode -and $CLUSTER_CFG.RunDirByNode[$n]){ $CLUSTER_CFG.RunDirByNode[$n] } else { $CLUSTER_CFG.RunDir } }
# Microsoft C++ runtime DLLs, app-local on some nodes and loaded from System32 on the others.
$runtimeDll = '^(concrt140|msvcp140(_\w+)?|vcruntime140(_\w+)?|vccorlib140)\.dll$'

# Every shard of a split GGUF, derived from the first shard's name.
$files = @($Model)
if($Model -match '^(.*-)00001-of-(\d{5})\.gguf$'){
  $stem = $Matches[1]; $cnt = $Matches[2]
  $files = @(1..[int]$cnt | ForEach-Object { '{0}{1:D5}-of-{2}.gguf' -f $stem, $_, $cnt })
}
$fileList = (@($files) + $Prompt | ForEach-Object { "'$_'" }) -join ','
$drvList  = (@($CLUSTER_CFG.DriverFiles) | ForEach-Object { "'$_'" }) -join ','

# Models are compared by exact byte length (hashing tens of GiB per node takes many minutes); that
# catches a truncated or partial copy. Files under 64 MB, like the prompt, are hashed.
$probe = @"
foreach(`$f in @(Get-ChildItem '__RUNDIR__' -File -EA SilentlyContinue | Where-Object { `$_.Extension -in '.exe','.dll' })){
  if(`$f.Extension -eq '.dll' -or `$f.Name -ieq 'llama-cli.exe'){ `$kind = 'BIN' } else { `$kind = 'TOOL' }
  `$kind + '|' + `$f.Name + '|' + (Get-FileHash `$f.FullName -Algorithm SHA256).Hash.Substring(0,16) + '|' + `$f.Length
}
foreach(`$m in @($fileList)){
  if(-not (Test-Path `$m)){ 'FILE|' + (Split-Path `$m -Leaf) + '|MISSING'; continue }
  `$len = (Get-Item `$m).Length
  if(`$len -lt 64MB){ 'FILE|' + (Split-Path `$m -Leaf) + '|' + (Get-FileHash `$m -Algorithm SHA256).Hash.Substring(0,16) + ' (' + `$len + ' B)' }
  else { 'FILE|' + (Split-Path `$m -Leaf) + '|' + `$len + ' B' }
}
foreach(`$d in @($drvList)){
  `$sd = @(Get-CimInstance Win32_SystemDriver -EA SilentlyContinue | Where-Object { `$_.PathName -and ((Split-Path `$_.PathName -Leaf) -ieq `$d) })
  if(-not `$sd.Count){ 'DRV|' + `$d + '|NO SERVICE'; continue }
  `$p = `$sd[0].PathName -replace '^\\\?\?\\',''
  if(`$p -match '^\\SystemRoot\\'){ `$p = Join-Path `$env:SystemRoot `$p.Substring(12) }
  elseif(`$p -notmatch '^[A-Za-z]:'){ `$p = Join-Path `$env:SystemRoot `$p }
  if(-not (Test-Path `$p)){ 'DRV|' + `$d + '|IMAGE MISSING ' + `$p; continue }
  'DRV|' + `$d + '|' + (Get-FileHash `$p -Algorithm SHA256).Hash + '|' + `$sd[0].State
}
"@

$data = @{}; $tools = @{}; $drv = @{}
foreach($n in $nodeList){
  $out = "$((Invoke-Node -N $n -Ps -TimeoutSec 400 -Cmd $probe.Replace('__RUNDIR__', (RunDirFor $n))).StdOut)"
  $d = @{}; $t = @{}; $v = @{}
  foreach($ln in ($out -split "`r?`n")){
    $p = $ln.Trim() -split '\|'
    if($p[0] -eq 'BIN'  -and $p.Count -ge 4){ $d["bin:$($p[1])"]  = "$($p[2]) ($($p[3]) B)" }
    if($p[0] -eq 'TOOL' -and $p.Count -ge 4){ $t[$p[1]] = $p[2] }
    if($p[0] -eq 'FILE' -and $p.Count -ge 3){ $d["file:$($p[1])"] = $p[2] }
    if($p[0] -eq 'DRV'  -and $p.Count -ge 3){ $v[$p[1]] = @{ Hash=$p[2]; State=$(if($p.Count -ge 4){$p[3]}else{''}) } }
  }
  $data[$n] = $d; $tools[$n] = $t; $drv[$n] = $v
}

$ref = $nodeList[0]
if(-not $data[$ref].Count){ throw "no probe output from $ref" }
if(-not $data[$ref].ContainsKey('bin:llama-cli.exe')){ throw "no llama-cli.exe in $(RunDirFor $ref) on $ref" }
# Union over all nodes, so a file present only on a non-reference node is flagged too.
$allKeys = @($data.Values | ForEach-Object { $_.Keys } | Sort-Object -Unique)
$keys = @($allKeys | Where-Object { $_ -notmatch ('^bin:' + $runtimeDll.TrimStart('^')) })
$rtKeys = @($allKeys | Where-Object { $_ -match ('^bin:' + $runtimeDll.TrimStart('^')) })

Write-Output ("run directories: " + (($nodeList | ForEach-Object { "$_=$(RunDirFor $_)" }) -join '  '))
Write-Output ""

$bad = @()
Write-Output ("{0,-46} {1}" -f 'item', (($nodeList | ForEach-Object { '{0,-30}' -f $_ }) -join ''))
Write-Output ('-' * (46 + 30 * $nodeList.Count))
foreach($k in $keys){
  $vals = @($nodeList | ForEach-Object { if($data[$_].ContainsKey($k)){ $data[$_][$k] } else { 'ABSENT' } })
  $same = (@($vals | Select-Object -Unique).Count -eq 1)
  Write-Output ("{0,-46} {1}  {2}" -f $k, (($vals | ForEach-Object { '{0,-30}' -f $_ }) -join ''), $(if($same){''}else{'<-- MISMATCH'}))
  if(-not $same -or ($vals -join ' ') -match 'MISSING|ABSENT'){ $bad += $k }
}
if($rtKeys.Count){
  Write-Output ""
  Write-Output "C++ runtime DLLs next to llama-cli (not compared; a node without them uses its System32 runtime):"
  foreach($k in $rtKeys){
    Write-Output ("  {0,-44} {1}" -f $k, (($nodeList | ForEach-Object { '{0,-30}' -f $(if($data[$_].ContainsKey($k)){ $data[$_][$k] } else { 'System32' }) }) -join ''))
  }
}
$toolNames = @($tools.Values | ForEach-Object { $_.Keys } | Sort-Object -Unique)
if($toolNames.Count){
  Write-Output ""
  Write-Output ("Not compared ({0} other .exe tools in RunDir; the scripts run only llama-cli.exe): {1}" -f $toolNames.Count, (($toolNames | Select-Object -First 6) -join ', ') + $(if($toolNames.Count -gt 6){', ...'}else{''}))
}

Write-Output ""
$known = $CLUSTER_CFG.DriverSha256
$haveKnown = ($known -and $known.Count)
Write-Output ("USB4/RDMA driver images loaded per node ({0}):" -f $(if($haveKnown){'checked against DriverSha256 in nodes.ps1'}else{'DriverSha256 is empty in nodes.ps1, reporting only'}))
$drvBad = @(); $drvWarn = @()
foreach($f in @($CLUSTER_CFG.DriverFiles)){
  $hs = @()
  foreach($n in $nodeList){
    $e = $drv[$n][$f]
    $h = if($e){ $e.Hash } else { 'no answer' }
    $tag = ''
    if($haveKnown -and $known.ContainsKey($f)){
      if($h -eq $known[$f]){ $tag = 'known-good' } else { $tag = 'NOT THE KNOWN-GOOD IMAGE'; $drvBad += "$n/$f" }
    }
    Write-Output ("  [{0}] {1,-24} {2,-20} {3,-10} {4}" -f $n, $f, $(if($h.Length -ge 16){$h.Substring(0,16)}else{$h}), $(if($e){$e.State}else{''}), $tag)
    $hs += $h
  }
  if(@($hs | Select-Object -Unique).Count -gt 1){ $drvWarn += $f }
}

Write-Output ""
if($bad.Count){
  Write-Output ("MISMATCHED OR MISSING: " + ($bad -join ', '))
  throw "the ranks are not running identical inputs; fix this before running"
}
if($drvBad.Count){
  Write-Output ("DRIVER NOT KNOWN-GOOD: " + ($drvBad -join ', '))
  throw "a node is not running the known-good USB4/RDMA driver image (an OS update can replace it silently); RDMA runs could go over TCP. Restore it, or if the new image has been verified with an all-ND run, update DriverSha256."
}
if($drvWarn.Count){ Write-Output ("WARNING: driver images differ between nodes: " + ($drvWarn -join ', ') + ". Record known-good hashes in DriverSha256 once an RDMA run is verified.") }
Write-Output "OK: all ranks identical on llama-cli, DLLs, model shards and prompt$(if($haveKnown){'; known-good RDMA driver on every node'}else{''})"
