# Bring the USB4 fabric to a known-good state before a run. Idempotent and safe to repeat. Nodes,
# cables and paths come from nodes.ps1.
#
#  1. Reboot each participating node, one at a time, and prove it rebooted (its boot time must
#     change), so every run starts from a fresh boot. On integrated-GPU systems a second large model
#     load in the same boot can run much slower than the first.
#
#  2. Set MTU 1500 on every USB4 P2P leg, and read it back. Legs can come up at MTU 62000, where bulk
#     TCP completes its handshake and then stalls on the first large segment, while ping, ARP, a
#     bare TCP connect and an ndping test all still pass.
#
#  3. Re-apply leg addresses by measurement on every node marked Readdress in nodes.ps1, then
#     restart that node's legs one at a time (3b) so the NetworkDirect provider picks up the final
#     addresses.
#
#  4. Set the P2P connection profiles to Private, and read it back. They can revert to Public on a
#     boot, which blocks the rendezvous.
#
#  5. Prove every edge between participating nodes accepts a TCP connection. That is a
#     reachability check only; the RDMA proof is each rank's transport line in the run itself, or
#     nd_pair.ps1.
#
#  6. Prove every leg carries full-size frames. A leg can pass small pings, TCP connects and ND
#     connection setup but drop every frame above ~1000 bytes in one direction; the run then fails
#     at its first exchange. Rebooting the sending node clears it, so that is done here
#     automatically, up to -FrameRetries times, and steps 2-6 run again.
param(
  # Comma string, not [string[]]: powershell.exe -File passes "A,B,C" as ONE element to an array
  # parameter. Get-NodeList splits it. Empty = every node in nodes.ps1.
  [string]$RankNodes = '',
  [switch]$SkipReboot,
  [int]$Mtu = 1500,
  # Extra reboots of a node whose leg drops full-size frames (step 6) before giving up.
  [int]$FrameRetries = 2
)
. "$PSScriptRoot\nodes.ps1"
$ErrorActionPreference = 'Stop'

if(-not $RankNodes){ $RankNodes = @($CLUSTER_NODES.Keys) -join ',' }
$nodeList = @(Get-NodeList $RankNodes)

# Both directions of every cable in $CLUSTER_P2P whose two ends are both participating.
$EDGES = @(Get-P2PEdges $nodeList)

function Say($s){ Write-Output ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $s) }
function T([string]$a,[int]$ms=4000){ $c=[Net.Sockets.TcpClient]::new(); $o=$false; try{$o=$c.ConnectAsync($a,22).Wait($ms)}catch{}; try{$c.Close()}catch{}; return $o }
function BootTime([string]$n){ "$((Invoke-Node -N $n -Ps -TimeoutSec 120 -Cmd '(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString(''s'')').StdOut)".Trim() }

# Every node must answer BEFORE anything is rebooted.
foreach($n in $nodeList){ if(-not (T (Node-Ip $n) 8000)){ throw "$n ($(Node-Ip $n)) does not answer on TCP:22; not rebooting anything" } }

function RebootNode([string]$n){
  Say "reboot $n"
  $before = BootTime $n
  if(-not $before){ throw "could not read $n's boot time over ssh (key, user, or admin rights?); not rebooting blind" }
  try { Invoke-Node -N $n -Ps -TimeoutSec 90 -Cmd 'taskkill /F /IM llama-cli.exe 2>&1 | Out-Null; shutdown /r /t 3 /f' | Out-Null } catch {}
  $sw=[Diagnostics.Stopwatch]::StartNew(); while($sw.Elapsed.TotalSeconds -lt 240 -and (T (Node-Ip $n))){ Start-Sleep 5 }
  $sw.Restart(); $up=$false
  while($sw.Elapsed.TotalSeconds -lt 900){ if(T (Node-Ip $n)){ $up=$true; break }; Start-Sleep 8 }
  if(-not $up){ throw "$n did not return from reboot" }
  Start-Sleep 30
  # "It went away and came back" is not proof: a node that never went down "comes back" at once.
  $after = BootTime $n
  if(-not $after -or $after -eq $before){ throw "$n did not actually reboot (boot time still $before)" }
  Say ("  $n rebooted: boot time $before -> $after (UTC)")
}

# ---- 1. reboot -------------------------------------------------------------------------------
if($SkipReboot){ Say "reboot SKIPPED (-SkipReboot): only valid if no model has been loaded this boot" }
else { foreach($n in $nodeList){ RebootNode $n } }

# ---- 2 + 4. MTU and connection profile on every P2P leg --------------------------------------
# Done per node in one remote pass, so the adapter set is enumerated after the reboot, never from
# a cached name or index.
$fix = @"
`$out = @()
`$gw = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -EA SilentlyContinue | ForEach-Object { `$_.ifIndex })
foreach(`$a in @(Get-NetAdapter -EA SilentlyContinue | Where-Object { `$_.InterfaceDescription -match '$($CLUSTER_CFG.P2PMatch)' -and `$_.Status -eq 'Up' -and `$gw -notcontains `$_.ifIndex })){
  # Never touch the normal network adapter. Two guards: the P2PMatch description filter, and
  # skipping any adapter that holds a default route (a P2P leg never has one).
  netsh interface ipv4 set subinterface "`$(`$a.Name)" mtu=$Mtu store=persistent | Out-Null
  try { Set-NetConnectionProfile -InterfaceIndex `$a.ifIndex -NetworkCategory Private -EA Stop } catch {}
  `$ip  = (Get-NetIPAddress -InterfaceIndex `$a.ifIndex -AddressFamily IPv4 -EA SilentlyContinue | ForEach-Object { `$_.IPAddress }) -join ','
  `$mtu = (Get-NetIPInterface -InterfaceIndex `$a.ifIndex -AddressFamily IPv4 -EA SilentlyContinue).NlMtu
  `$prf = (Get-NetConnectionProfile -InterfaceIndex `$a.ifIndex -EA SilentlyContinue).NetworkCategory
  # Read back, never assume: netsh and Set-NetConnectionProfile can both fail quietly (for example
  # without admin rights).
  `$tag = if(`$mtu -ne $Mtu){ '  <-- MTU NOT APPLIED' } elseif("`$prf" -ne 'Private'){ '  <-- profile not Private' } else { '' }
  `$out += ('  LEG {0,-12} idx={1,-4} ip=[{2,-15}] mtu={3,-6} profile={4}{5}' -f `$a.Name, `$a.ifIndex, `$ip, `$mtu, `$prf, `$tag)
}
`$out
"@
# Steps 2-6 run as one unit so that a node rebooted again by step 6 gets its legs set up again.
# The body stays at column 0 because it holds a here-string, whose closing "@ must start its line.
function PrepLegs {
Write-Output ""
Say "MTU $Mtu + Private profile on every USB4 leg (adapters matching '$($CLUSTER_CFG.P2PMatch)')"
foreach($n in $nodeList){
  Write-Output "  [$n]"
  $legOut = "$((Invoke-Node -N $n -Ps -TimeoutSec 300 -Cmd $fix).StdOut)".TrimEnd()
  Write-Output $legOut
  # Every cable to another participating node must show up as a leg here. Zero legs means
  # P2PMatch does not fit this hardware.
  $want = @($EDGES | Where-Object { $_.From -eq $n }).Count
  $have = @(($legOut -split "`r?`n") | Where-Object { $_ -match '^\s*LEG ' }).Count
  if($have -lt $want){ throw "$n shows $have USB4 leg(s) matching '$($CLUSTER_CFG.P2PMatch)' but `$CLUSTER_P2P gives it $want cable(s) to participating nodes; fix P2PMatch in nodes.ps1 or the cabling" }
  if($legOut -match 'MTU NOT APPLIED'){ throw "MTU $Mtu did not stick on $n (is the ssh user a local Administrator?)" }
  if($legOut -match 'profile not Private'){ throw "a USB4 leg on $n is not on a Private network profile; the rendezvous would be blocked" }
}

# ---- 3. leg addresses by measurement, on every node marked Readdress -------------------------
foreach($n in @($nodeList | Where-Object { $CLUSTER_NODES[$_].Readdress })){
  Write-Output ""
  Say "re-applying $n's leg addresses (Readdress in nodes.ps1)"
  & powershell @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$PSScriptRoot\fix_p2p_legs.ps1",
                 '-Node', $n, '-RankNodes', ($nodeList -join ',')) | ForEach-Object { Write-Output ("  " + $_) }
  if($LASTEXITCODE -ne 0){ throw "fix_p2p_legs.ps1 failed for $n" }

  # 3b. Restart that node's legs, one at a time, so the ND provider registers the final addresses.
  # Without it, ND connects to this node can hang while TCP still works. One leg at a time, with a
  # settle wait: restarting every adapter at once can break ND on all of them.
  Say "restarting $n's legs one at a time so NetworkDirect picks up the re-applied addresses"
  $bounce = @"
`$ProgressPreference='SilentlyContinue'
`$gw = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -EA SilentlyContinue | ForEach-Object { `$_.ifIndex })
foreach(`$a in @(Get-NetAdapter -EA SilentlyContinue | Where-Object { `$_.InterfaceDescription -match '$($CLUSTER_CFG.P2PMatch)' -and `$_.Status -eq 'Up' -and `$gw -notcontains `$_.ifIndex })){
  Disable-NetAdapter -Name `$a.Name -Confirm:`$false; Start-Sleep 4; Enable-NetAdapter -Name `$a.Name -Confirm:`$false
  `$sw=[Diagnostics.Stopwatch]::StartNew()
  while(`$sw.Elapsed.TotalSeconds -lt 60 -and (Get-NetAdapter -Name `$a.Name).Status -ne 'Up'){ Start-Sleep 2 }
  Start-Sleep 25
  `$x = Get-NetAdapter -Name `$a.Name
  '  {0,-12} {1} {2} ip={3}' -f `$x.Name, `$x.Status, `$x.LinkSpeed, ((Get-NetIPAddress -InterfaceIndex `$x.ifIndex -AddressFamily IPv4 -EA SilentlyContinue).IPAddress -join ',')
}
"@
  $bOut = "$((Invoke-Node -N $n -Ps -TimeoutSec 600 -Cmd $bounce).StdOut)".TrimEnd()
  Write-Output $bOut
  if($bOut -notmatch '\bUp\b' -or $bOut -match 'Disconnected|Disabled'){ throw "a USB4 leg on $n did not come back Up after the restart" }
}

# ---- 5. prove every edge accepts a connection ---------------------------------------------------
Write-Output ""
Say "edge reachability (TCP:22 connect; ICMP can be filtered on a healthy leg)"
$bad = @()
foreach($e in $EDGES){
  $probe = "`$c=[Net.Sockets.TcpClient]::new();`$o=`$false;try{`$o=`$c.ConnectAsync('$($e.To)',22).Wait(8000)}catch{};try{`$c.Close()}catch{};`$o"
  $r = "$((Invoke-Node -N $e.From -Ps -TimeoutSec 200 -Cmd $probe).StdOut)".Trim()
  $ok = ($r -match 'True')
  Write-Output ("  {0,-10} {1,-16} {2}" -f $e.Label, $e.To, $(if($ok){'OK'}else{'FAIL'}))
  if(-not $ok){ $bad += $e.Label }
}
if($bad.Count){ throw ("edges down: " + ($bad -join ', ')) }

# ---- 6. full-size frames on every leg -----------------------------------------------------------
# A 1472-byte don't-fragment ping is a full 1500-byte frame. Replies are counted by their "Reply
# from <peer>: bytes=" line, because ping's own "Received =" also counts error replies. A direction
# whose 32-byte ping gets no reply is ICMP-filtered, and is reported, not judged.
Write-Output ""
Say "full-size frames on every leg (32 B and 1472 B don't-fragment pings)"
$script:frameBad = @()
foreach($e in $EDGES){
  $reply = "Reply from $($e.To): bytes="
  $probe = "`$s=@(ping -n 2 -w 1500 -l 32 $($e.To) | Select-String -SimpleMatch '$reply').Count; " +
           "`$b=@(ping -n 2 -w 1500 -f -l 1472 $($e.To) | Select-String -SimpleMatch '$reply').Count; 'small=' + `$s + ' big=' + `$b"
  $r = "$((Invoke-Node -N $e.From -Ps -TimeoutSec 200 -Cmd $probe).StdOut)".Trim()
  $small = if($r -match 'small=(\d+)'){ [int]$Matches[1] } else { -1 }
  $big   = if($r -match 'big=(\d+)'){ [int]$Matches[1] } else { -1 }
  $verdict = if($small -le 0){ 'not judged (small ping filtered)' } elseif($big -le 0){ 'FULL-SIZE FRAMES DROPPED' } else { 'OK' }
  Write-Output ("  {0,-10} {1,-16} 32B {2}/2  1472B-DF {3}/2  {4}" -f $e.Label, $e.To, $small, $big, $verdict)
  if($verdict -like 'FULL*' -and $script:frameBad -notcontains $e.From){ $script:frameBad += $e.From }
}
}

$frameTries = 0
while($true){
  PrepLegs
  if(-not $script:frameBad.Count){ break }
  $who = $script:frameBad -join ','
  if($SkipReboot){ throw "full-size frames dropped on a leg from $who, and -SkipReboot forbids the reboot that clears it" }
  if($frameTries -ge $FrameRetries){ throw "full-size frames still dropped on a leg from $who after $frameTries extra reboot(s); re-plug that cable" }
  $frameTries++
  Write-Output ""
  Say "full-size frames dropped from $who; rebooting it again (attempt $frameTries of $FrameRetries), then steps 2-6 again"
  foreach($n in $script:frameBad){ RebootNode $n }
}

Write-Output ""
# Presence only. verify_binaries.ps1 hashes the driver each service actually loads, and the run's
# per-rank transport line is the final word.
Say "ND provider dll present per node (presence only, not proof that RDMA works)"
$ndprov = Join-Path $CLUSTER_CFG.NdDir 'ndprov.dll'
foreach($n in $nodeList){
  $r = "$((Invoke-Node -N $n -Ps -TimeoutSec 250 -Cmd "if(Test-Path '$ndprov'){'provider dll present'}else{'ndprov.dll MISSING at $ndprov'}").StdOut)".Trim()
  Write-Output ("  [{0}] {1}" -f $n, $r)
}

Write-Output ""
Say "fabric ready: $($nodeList -join ',')"
