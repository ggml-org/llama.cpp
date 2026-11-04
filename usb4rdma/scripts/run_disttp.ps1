# Run llama-cli with distributed tensor parallelism on 1..N nodes, over RDMA or TCP, and check what
# actually happened: which transport every ring edge used, whether any rank failed, and the
# generated text.
#
# Node names, cable addresses, binary/model/prompt paths, device, threads and the rendezvous port
# all come from nodes.ps1. Rank 0 is the first node in -RankNodes. The binaries find the cable
# between each pair of ranks themselves; the address each rank should dial for its ring-next peer is
# derived from the rank order against $CLUSTER_P2P and checked against the one it logged.
#
# Ring order matters for RDMA: on each edge the lower rank accepts the ND connection, and from 3
# ranks up each edge carries data one way, rank r -> r+1. Use the order recorded in $CLUSTER_RINGS.
#
# The master socket is for the rendezvous only and carries no tensor traffic. It is rank 0's
# RendezvousIp (default: its HostIp), which every rank can reach.
#
# Examples:
#   .\run_disttp.ps1 -Transport rdma                       # $CLUSTER_RINGS[2]
#   .\run_disttp.ps1 -Transport rdma -RankNodes A,B,C
#   .\run_disttp.ps1 -Transport rdma -RankNodes A,B,C -ExtraArgs '-fa on -ctk q8_0 -ctv q8_0'
#   .\run_disttp.ps1 -Transport tcp  -RankNodes A,B
#   .\run_disttp.ps1 -Transport none -RankNodes A          # single node baseline
param(
  [Parameter(Mandatory)][ValidateSet('rdma','tcp','none')][string]$Transport,
  # Empty = $CLUSTER_RINGS[2]. Model, Prompt, Threads and Device likewise default from nodes.ps1;
  # param defaults are evaluated before nodes.ps1 is loaded, hence the empty/0 sentinels.
  [string]$RankNodes = '',
  [string]$Model  = '',
  [string]$Prompt = '',
  [int]$NPred = 64,
  [int]$Ctx = 2048,
  [int]$Threads = 0,
  [int]$Ngl = 99,
  # Appended verbatim to llama-cli on every rank, e.g. '-fa on -ctk q8_0 -ctv q8_0' for long
  # context.
  [string]$ExtraArgs = '',
  [string]$Device = '',
  # Leave empty to derive it. The master MUST be rank 0's own address: rank 0 binds and listens on
  # it, so any other address means the rendezvous never forms and every rank waits forever at
  # "split_mode=3" with no transport line and no error.
  [string]$Master = '',
  # Give up if no transport line has appeared within this long. The rendezvous either completes in
  # well under a minute or never completes.
  [int]$RendezvousTimeoutSec = 900,
  # LLAMA_DIST_TP_RDMA_DEBUG=1 on every rank: the ND layer logs each edge's local address, listener,
  # connection request, connect and teardown. For diagnosing an edge, not for timing.
  [switch]$RdmaDebug,
  # More environment for every rank, 'NAME=value;NAME=value', set after the variables above so it
  # can override them: the Vulkan switches in the README. A value must not contain & | < > ^ " or %.
  [string]$ExtraEnv = '',
  # Start ranks 1..N-1 detached, then STOP and print the exact command to run by hand in a console
  # on rank 0. llama.cpp block-buffers stdout when it is not attached to a console, so this is the
  # way to watch tokens stream.
  [switch]$Rank0Manual,
  [int]$RunTimeoutSec = 5400,
  # End the run if rank 0's log has not grown for this long once the transport line is up. A stalled
  # transfer blocks every rank with sockets still established and no error.
  [int]$StallSec = 1800,
  [int]$PollSec = 30,
  [string]$Tag,
  [string]$ResultsFile
)
. "$PSScriptRoot\nodes.ps1"
. "$PSScriptRoot\distprof.ps1"

if(-not $RankNodes){ $RankNodes = $CLUSTER_RINGS[2] }
if(-not $Model)    { $Model     = $CLUSTER_CFG.Model }
if(-not $Prompt)   { $Prompt    = $CLUSTER_CFG.Prompt }
if(-not $Threads)  { $Threads   = $CLUSTER_CFG.Threads }
if(-not $Device)   { $Device    = $CLUSTER_CFG.Device }

$nodeList = @(Get-NodeList $RankNodes)
$world = $nodeList.Count
if(-not $Tag){ $Tag = "run_{0}n_{1}" -f $world, $Transport }
if(-not $ResultsFile){ $ResultsFile = Join-Path $PSScriptRoot 'logs\results.jsonl' }
if($world -eq 1 -and $Transport -ne 'none'){ throw "a single-rank run has no transport; pass -Transport none" }
if($world -gt 1 -and $Transport -eq 'none'){ throw "a multi-rank run needs -Transport rdma or tcp" }

$RUN = $CLUSTER_CFG.RunDir
function RunDirFor([string]$n){ if($CLUSTER_CFG.RunDirByNode -and $CLUSTER_CFG.RunDirByNode[$n]){ $CLUSTER_CFG.RunDirByNode[$n] } else { $RUN } }
# At 3 ranks the binaries always run the full mesh: each small collective goes to both peers at once.
$Mesh = $world -eq 3
# Not $extraEnv: PowerShell variable names ignore case, so that would be the parameter itself.
$envExtra = @($ExtraEnv -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach($kv in $envExtra){
  if($kv -notmatch '^[A-Za-z_][A-Za-z0-9_]*=[^&|<>^"%]*$'){ throw "-ExtraEnv entry '$kv' is not NAME=value, or its value has a character cmd would interpret" }
}
$LOGDIR = Join-Path $PSScriptRoot 'logs'
if(-not (Test-Path $LOGDIR)){ New-Item -ItemType Directory $LOGDIR | Out-Null }

# The ring-next address each rank should dial, derived from the rank ORDER against $CLUSTER_P2P.
$next = @()
if($world -gt 1){
  for($i = 0; $i -lt $world; $i++){ $next += (Get-P2PAddr $nodeList[$i] $nodeList[($i + 1) % $world]) }
} else { $next = @($null) }

if(-not $Master -and $world -gt 1){ $Master = (Node-RendezvousIp $nodeList[0]) + ':' + $CLUSTER_CFG.MasterPort }

Write-Host "==== $Tag :: world=$world nodes=$($nodeList -join ',') transport=$Transport ===="
Write-Host "  model  : $(Split-Path $Model -Leaf)"
Write-Host "  fixed  : -n $NPred -c $Ctx -ngl $Ngl -t $Threads --device $Device --temp 0 --seed 1234 $ExtraArgs"
if($envExtra.Count){ Write-Host "  env    : $($envExtra -join ' ')" }
if($world -gt 1){
  for($i=0; $i -lt $world; $i++){ Write-Host ("  rank{0} {1} -> {2}" -f $i, $nodeList[$i], $next[$i]) }
  Write-Host "  master : $Master (rendezvous only, no tensor traffic)"
}

$clean = '$ProgressPreference="SilentlyContinue";Get-Process llama-cli -EA SilentlyContinue|Stop-Process -Force -EA SilentlyContinue;"ok"'
foreach($n in $nodeList){ Invoke-Node -N $n -Ps -Cmd $clean -TimeoutSec 120 | Out-Null }
Start-Sleep 3

$common = "-m `"$Model`" -f `"$Prompt`" -n $NPred -c $Ctx -ngl $Ngl -t $Threads --temp 0 --seed 1234 --device $Device"
if($ExtraArgs){ $common += " $ExtraArgs" }
if($world -gt 1){ $common += " --split-mode tensor --world-size $world --dist-master $Master" }

# "NAME=value", or "NAME=" to clear it, for every dist-tp variable of a rank. Every one is set OR
# cleared explicitly: a node with one of them set system-wide would otherwise turn a TCP run into an
# RDMA one. Shared by LaunchRank and -Rank0Manual.
function RankEnv([int]$rank){
  if($world -le 1){ return @() }
  @(
    "LLAMA_DIST_TP_RDMA=$(if($Transport -ne 'rdma'){0})"
    "LLAMA_DIST_TP_RDMA_DEBUG=$(if($RdmaDebug){1})"
  )
}
# A rank's whole environment: RankEnv without the names -ExtraEnv sets, then -ExtraEnv (also at
# one rank).
function AllEnv([int]$rank){
  $names = @($envExtra | ForEach-Object { ($_ -split '=', 2)[0] })
  @(@(RankEnv $rank) | Where-Object { $names -notcontains ($_ -split '=', 2)[0] }) + $envExtra
}

$logs = @{}
function LaunchRank([int]$rank){
  $node = $nodeList[$rank]
  $rd   = RunDirFor $node
  $log  = "$rd\$Tag.rank$rank.log"
  $logs[$rank] = $log
  # `set X=` clears X, and cmd carries on down the && chain whether or not X existed.
  $envPfx = (@(AllEnv $rank) | ForEach-Object { "set $_&&" }) -join ''
  $rankArg = if($world -gt 1){ " --world-rank $rank" } else { '' }
  # No leading quote in the cmd string: when a `cmd /c` line STARTS with a quote, cmd strips the
  # outer pair and the redirect silently never happens. Quotes further in are fine and let RunDir
  # contain spaces. RunDir, Model and Prompt must not contain a single quote.
  $cl = "cmd /c ${envPfx}set LLAMA_DISTPROF=1&&cd /d `"$rd`"&&llama-cli.exe $common$rankArg > `"$log`" 2>&1"
  # Win32_Process Create, not Start-Process: a process started with Start-Process dies with the ssh
  # session. Wait up to 90 s for llama-cli to appear: endpoint security can make process start slow.
  $ps = @"
`$ProgressPreference='SilentlyContinue'
Remove-Item '$log' -EA SilentlyContinue
`$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='$cl'}
`$n=0; `$sw=[Diagnostics.Stopwatch]::StartNew()
while(`$sw.Elapsed.TotalSeconds -lt 90){
  `$procs = @(Get-Process -Name llama-cli -EA SilentlyContinue)
  if(`$procs.Count){
    # Windows EcoQoS can throttle detached background processes on hybrid CPUs; High priority avoids it.
    foreach(`$x in `$procs){ try{ `$x.PriorityClass='High'; `$n++ }catch{} }
    break
  }
  Start-Sleep 2
}
'rank$rank PID=' + `$p.ProcessId + ' raised=' + `$n
"@
  $out = (Invoke-Node -N $node -Ps -TimeoutSec 240 -Cmd $ps).StdOut.Trim()
  Write-Host ("[$node] " + $out)
  if($out -notmatch 'raised=[1-9]'){
    foreach($n in $nodeList){ Invoke-Node -N $n -Ps -Cmd $clean -TimeoutSec 120 | Out-Null }
    throw "rank $rank on $node did not start llama-cli within 90 s ($out). Check $rd on $node, and application control policy: a blocked binary never starts."
  }
}

# Highest rank first: the non-master ranks retry the rendezvous, but rank 0 must not be left
# waiting on peers that have not started.
for($r = $world - 1; $r -ge 1; $r--){ LaunchRank $r; Start-Sleep 10 }

if($Rank0Manual){
  $envLines = @(@(AllEnv 0) | ForEach-Object { "set $_" }) + 'set LLAMA_DISTPROF=1'
  $rank0Arg = if($world -gt 1){ ' --world-rank 0' } else { '' }

  Write-Host ""
  Write-Host "================================================================"
  if($world -gt 1){ Write-Host (" Ranks 1..{0} are running and waiting at the rendezvous." -f ($world - 1)) }
  Write-Host " Now open cmd.exe ON $($nodeList[0]) (NOT PowerShell) and paste this."
  Write-Host " In PowerShell, 'set X=1' creates a shell variable and sets no environment variable,"
  Write-Host " so the run would silently go out on TCP."
  Write-Host "================================================================"
  Write-Host ""
  foreach($l in $envLines){ Write-Host "  $l" }
  Write-Host "  cd /d $(RunDirFor $nodeList[0])"
  Write-Host "  llama-cli.exe $common$rank0Arg"
  Write-Host ""
  Write-Host " Check the 'dist-tp transport:' line before trusting the run."
  if($world -gt 2){
    Write-Host " At $world ranks it reads 'RDMA (ND)' on any rank with even ONE edge on ND. Every"
    Write-Host " entry of the list after it, e.g. (->1 ND, 2-> ND), must say ND on every rank."
  }
  if($world -gt 1){
    Write-Host ""
    Write-Host "Peers are left running. To stop them afterwards:"
    foreach($n in $nodeList[1..($world-1)]){ Write-Host "  [$n]  taskkill /F /IM llama-cli.exe" }
  }
  return
}

LaunchRank 0

# ---- wait for the run to finish, or for ANY rank to fail ----------------------------------------
# llama-cli does NOT exit when the model fails to load: it drops to the interactive prompt, so the
# logs are checked for fatal patterns. Peers are watched too: a peer that dies leaves rank 0 blocked
# in the first collective, which otherwise looks like a slow run.
$failPat = 'Failed to load the model|failed to load model|unable to allocate|GGML_ASSERT|llama_decode\(\) failed|NetworkDirect recv failed|meta exchange failed'
function ProbeRank([int]$r){
  $q = '$n=0;$perf=$false;$fail=$false;$tr=$false;if(Test-Path "' + $logs[$r] + '"){$t=Get-Content "' + $logs[$r] + '" -Raw -EA SilentlyContinue;$n=($t -split "`n").Count;if($t -match "\[\s*Prompt:.*Generation:"){$perf=$true};if($t -match "' + $failPat + '"){$fail=$true};$tr=($t -match "dist-tp transport:")};"alive=" + [bool](Get-Process llama-cli -EA SilentlyContinue) + " perf=$perf fail=$fail transport=$tr lines=$n"'
  (Invoke-Node -N $nodeList[$r] -Ps -TimeoutSec 150 -Cmd $q).StdOut.Trim()
}
$sw = [Diagnostics.Stopwatch]::StartNew(); $endReason = 'timeout'
$lastLines = -1; $lastGrowth = 0
while($sw.Elapsed.TotalSeconds -lt $RunTimeoutSec){
  Start-Sleep $PollSec
  $a = ProbeRank 0
  Write-Host ("  [{0,5}s] r0 {1}" -f [int]$sw.Elapsed.TotalSeconds, $a)
  $lines = if($a -match 'lines=(\d+)'){ [int]$Matches[1] } else { -1 }
  if($lines -ne $lastLines){ $lastLines = $lines; $lastGrowth = $sw.Elapsed.TotalSeconds }
  elseif($a -match 'transport=True' -and $a -notmatch 'perf=True' -and ($sw.Elapsed.TotalSeconds - $lastGrowth) -gt $StallSec){
    $endReason = 'stall'
    Write-Host ("  rank 0's log has not grown in {0}s; the run is stalled (check the cables with link_throughput.ps1)" -f [int]($sw.Elapsed.TotalSeconds - $lastGrowth))
    break
  }
  if($a -match 'perf=True'){ $endReason = 'perf'; Write-Host "  perf line seen, stopping"; break }
  if($a -match 'fail=True'){ $endReason = 'fail_rank0'; Write-Host "  fatal error in rank 0's log"; break }
  if($a -match 'alive=False'){ $endReason = 'rank0_exited'; Write-Host "  rank 0 exited"; break }
  $peerDown = $null
  for($r = 1; $r -lt $world; $r++){
    $p = ProbeRank $r
    if($p -match 'fail=True'){ $peerDown = "fail_rank$r"; Write-Host "  fatal error in rank $r's log ($($nodeList[$r])): $p"; break }
    if($p -match 'alive=False'){ $peerDown = "rank${r}_exited"; Write-Host "  rank $r ($($nodeList[$r])) is no longer running: $p"; break }
  }
  if($peerDown){ $endReason = $peerDown; break }
  if($world -gt 1 -and $sw.Elapsed.TotalSeconds -gt $RendezvousTimeoutSec -and $a -notmatch 'transport=True'){
    $endReason = 'rendezvous'
    Write-Host ("  no transport line after {0}s; the rendezvous never formed. Check that --dist-master is rank 0's OWN address ({1}) and that the firewall admits llama-cli." -f [int]$sw.Elapsed.TotalSeconds, $Master)
    break
  }
}

Start-Sleep 3
$esc = [char]27
function Clean($s){ $s -replace ("{0}\[[0-9;?]*[A-Za-z]" -f [regex]::Escape($esc)),'' }
$raw = @{}
for($r = 0; $r -lt $world; $r++){
  # Fetch each log as base64 BYTES. Reading it as text over ssh round-trips every byte through the
  # remote console code page, which mangles the generated text and makes the output hash useless.
  # FileShare.ReadWrite because llama-cli may still hold the log open. The node also reports the
  # byte length, so a transfer cut short by the ssh timeout is caught.
  $readShared = @"
`$p = '$($logs[$r])'
if(Test-Path `$p){
  `$fs = [IO.File]::Open(`$p, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  `$ms = New-Object IO.MemoryStream
  `$fs.CopyTo(`$ms); `$fs.Close()
  'LEN=' + `$ms.Length
  [Convert]::ToBase64String(`$ms.ToArray())
}
"@
  $bytes = $null; $raw[$r] = ''
  $out = "$((Invoke-Node -N $nodeList[$r] -Ps -TimeoutSec 600 -Cmd $readShared).StdOut)"
  $want = if($out -match 'LEN=(\d+)'){ [int64]$Matches[1] } else { -1 }
  $b64 = ($out -replace 'LEN=\d+','') -replace '\s',''
  try { if($b64){ $bytes = [Convert]::FromBase64String($b64) } } catch { $bytes = $null }
  $dest = Join-Path $LOGDIR "$Tag.rank$r.log"
  if($bytes -and $bytes.Length -eq $want){
    [IO.File]::WriteAllBytes($dest, $bytes)
    $raw[$r] = Clean ([Text.Encoding]::UTF8.GetString($bytes))
  } else {
    # Never leave a stale file from an earlier run of this tag in place of a log that was not read.
    Remove-Item $dest -EA SilentlyContinue
    Write-Host ("  WARNING: rank $r log on $($nodeList[$r]) was empty, unreadable or truncated (got {0} of {1} bytes)" -f $(if($bytes){$bytes.Length}else{0}), $want)
  }
}
foreach($n in $nodeList){ Invoke-Node -N $n -Ps -Cmd $clean -TimeoutSec 120 | Out-Null }

# ---- analyse EVERY rank's log (see Get-RunAnalysis in distprof.ps1 for why rank 0 is not enough)
$an = Get-RunAnalysis -RankTexts @(for($r = 0; $r -lt $world; $r++){ "$($raw[$r])" }) -Transport $Transport -ExpectedNext @($next | ForEach-Object { "$_" })
$r0 = "$($raw[0])"
$prof = $an.prof
$meshTcp = $(if($Mesh){ Get-MeshTcpRanks @($raw.Values | ForEach-Object { "$_" }) }else{ $null })
# Largest all-reduce payload against the ND ceiling: anything above it went over TCP.
$overCeiling = [bool]($Transport -eq 'rdma' -and $prof -and $an.nd_max_bytes -gt 0 -and $prof.max_coll_bytes -gt $an.nd_max_bytes)

Write-Host ""
Write-Host "---- transport actually selected ----"
$transportLines = @(($r0 -split "`r?`n") | Where-Object { $_ -match 'dist-tp transport:|dist-tp all-reduce:|ring edge|falling back to TCP|no usable NetworkDirect|staying on TCP|NetworkDirect not established' })
if($transportLines.Count){ $transportLines | ForEach-Object { Write-Host ("  " + $_.Trim()) } } else { Write-Host "  (no transport line; single rank)" }
if($world -gt 1){ Write-Host ("  per-rank headline: {0}" -f (($an.headlines | ForEach-Object { if($_){$_}else{'(none)'} }) -join ', ')) }
if($Transport -eq 'rdma'){
  Write-Host ("  RDMA edges up (all ranks): {0} of {1}   edge summary: {2}" -f $(if($null -ne $an.nd_edges_up){$an.nd_edges_up}else{'unknown'}), $an.nd_edges_expected, $(if($an.edge_summary){$an.edge_summary}else{'n/a'}))
  if($an.nd_failed_edges.Count){ Write-Host ("  *** PARTIAL RDMA: edges {0} stayed on TCP ***" -f ($an.nd_failed_edges -join ', ')) }
  if($null -eq $an.nd_edges_up){ Write-Host "  *** not every rank logged a transport line; the RDMA ring cannot be assessed ***" }
  if($Mesh){
    Write-Host ("  ranks with a mesh direction on TCP: {0} of {1}" -f $meshTcp, $world)
    if($meshTcp){ Write-Host "  *** a mesh direction fell back to TCP: this is NOT an all-RDMA mesh ***" }
  }
  if($prof){
    Write-Host ("  largest all-reduce payload: {0} B, ND ceiling: {1}" -f $prof.max_coll_bytes, $(if($null -eq $an.nd_max_bytes){'unknown'}elseif($an.nd_max_bytes -eq 0){'none'}else{"$($an.nd_max_bytes) B"}))
    if($overCeiling){ Write-Host "  *** payloads above the ND ceiling went over TCP (the prompt, usually); this binary predates multi-chunk ND messages ***" }
  }
}
if($an.foreign_rank_logs.Count){ Write-Host ("  *** rank(s) {0}: the log dials a different ring-next than this run set; it is from another run ***" -f ($an.foreign_rank_logs -join ', ')) }
if($an.head_mismatch_ranks.Count){ Write-Host ("  *** ranks {0} ran a different transport than requested ***" -f ($an.head_mismatch_ranks -join ', ')) }

Write-Host "---- outcome ----"
Write-Host ("  ended by: $endReason")
if($an.oom_lines.Count){ $an.oom_lines | Select-Object -First 3 | ForEach-Object { Write-Host ("  OOM " + $_.Trim()) } }
else { Write-Host "  no allocation failure in any rank's log" }
if(@($an.builds | Where-Object { $_ } | Select-Object -Unique).Count -gt 1){ Write-Host ("  *** ranks run DIFFERENT builds: {0} ***" -f ($an.builds -join ', ')) }

Write-Host "---- correctness ----"
# A broken run can still print a plausible t/s. Hash the generated text and read a sample. With
# LLAMA_DIST_TP_WIRE_F32=1 the same model, prompt and seed must hash the same at every rank count and
# on either transport; the default f16 wire must still hash the same on either transport.
$an.fatal_lines | Select-Object -First 5 | ForEach-Object { Write-Host "  ERR $_" }
Write-Host ("  output chars={0}  SHA256={1}" -f $an.gen_text.Length, $(if($an.out_sha256){$an.out_sha256}else{'n/a (no output)'}))
if($an.gen_text.Length -gt 0){ Write-Host ("  sample: " + $an.gen_text.Substring(0, [Math]::Min(160, $an.gen_text.Length))) }

if($prof){
  Write-Host ""
  Write-Host "---- per-token cost, rank 0 (mean over $($prof.tokens) decode graphs; $($prof.excluded) prompt graphs excluded) ----"
  Write-Host ("  total={0} ms  gpu_wait(sync+get)={1} ms  net={2} ms  setadd={3} ms  other={4} ms  bytes={5}  nbnd={6}" -f `
      $prof.ms_total, $prof.ms_gpu_wait, $prof.ms_net, $prof.ms_setadd, $prof.ms_other, $prof.bytes, $prof.nbnd)
  Write-Host ("  net by rank (ms): {0}" -f ($an.ms_net_by_rank -join ', '))
  Write-Host ("  prompt: {0} tokens in {1:N1} s of prompt graphs" -f $prof.prompt_tokens, ($prof.prefill_ms / 1000.0))
  Write-Host ("  cross-check: profiler implies {0} tok/s vs llama.cpp {1}{2}" -f $prof.implied_decode_tps, $an.decode_tps, $(if($an.prof_suspect){'   *** MORE THAN 10% OFF: per-token figures unreliable ***'}else{''}))
}

Write-Host ""
Write-Host ("==== RESULT $Tag :: world=$world transport=$Transport status=$($an.status) prefill=$($an.prefill_tps) t/s  decode=$($an.decode_tps) t/s ====")

$rec = [ordered]@{
  tag=$Tag; utc=(Get-Date).ToUniversalTime().ToString('s')
  world_size=$world; nodes=($nodeList -join ','); ring=((0..($world-1) | ForEach-Object { $nodeList[$_] + '->' + $next[$_] }) -join ' ')
  transport=$Transport; rank_headlines=($an.headlines -join ','); transport_fell_back=$an.transport_fell_back
  nd_edges_up=$an.nd_edges_up; nd_edges_expected=$an.nd_edges_expected; nd_failed_edges=($an.nd_failed_edges -join ';')
  rdma_clean=$an.rdma_clean; edge_summary=$an.edge_summary; nd_max_bytes=$an.nd_max_bytes
  max_coll_bytes=$(if($prof){$prof.max_coll_bytes}else{$null}); over_nd_ceiling=$overCeiling
  status=$an.status; end_reason=$endReason; build=$(if($an.builds.Count){$an.builds[0]}else{$null})
  model=(Split-Path $Model -Leaf); prompt=(Split-Path $Prompt -Leaf); n_predict=$NPred; ctx=$Ctx; extra_args=$ExtraArgs
  threads=$Threads; ngl=$Ngl; mesh=[bool]$Mesh
  extra_env=($envExtra -join ';')
  mesh_up=$(if($Mesh){ @($raw.Values | Where-Object { "$_" -match 'dist-tp all-reduce: full mesh' }).Count }else{ $null })
  mesh_tcp_ranks=$meshTcp; device=$Device; master=$(if($world -gt 1){$Master}else{$null})
  prefill_tps=$an.prefill_tps; decode_tps=$an.decode_tps
  ms_total=$prof.ms_total; ms_gpu_wait=$prof.ms_gpu_wait; ms_sync=$prof.ms_sync; ms_get=$prof.ms_get
  ms_net=$prof.ms_net; ms_net_by_rank=($an.ms_net_by_rank -join ';'); ms_setadd=$prof.ms_setadd; ms_other=$prof.ms_other
  bytes_per_token=$prof.bytes; nbnd=$prof.nbnd; prof_tokens=$prof.tokens; prof_excluded=$prof.excluded
  prefill_ms=$prof.prefill_ms; prompt_tokens=$prof.prompt_tokens; implied_decode_tps=$prof.implied_decode_tps; prof_suspect=$an.prof_suspect
  out_chars=$an.gen_text.Length; out_sha256=$an.out_sha256
}
($rec | ConvertTo-Json -Compress) | Add-Content $ResultsFile
Write-Host "  appended to $ResultsFile"
