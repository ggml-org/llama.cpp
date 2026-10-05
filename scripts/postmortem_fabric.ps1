# Record the state of the fabric right after a run dies, before anything is rebooted.
#
# A NetworkDirect abort (0xC0000241 on a receive) names only the rank that saw it, not the cable,
# and prep_fabric.ps1 reboots every node before the next run, which wipes the evidence. Link drops
# do not necessarily reach any event log, so this window is the place to tell two causes apart:
#   - the cable itself stopped carrying traffic (pings, TCP and ND all fail on one leg), or
#   - the RDMA connection was torn down while the cable stayed up (everything below passes
#     except ND).
#
# Read-only: TCP connects, pings, ndping and Get-* queries. Nothing is reset or reconfigured, and
# the normal network adapter is excluded by the same two guards prep_fabric.ps1 uses (P2PMatch on
# the description, and no adapter that holds a default route).
#
#   .\postmortem_fabric.ps1 -RankNodes A,B,C
param(
  # Comma string, as in prep_fabric.ps1.
  [Parameter(Mandatory)][string]$RankNodes,
  # Skip the ndping pass (about a minute per cable).
  [switch]$NoNd,
  [int]$NdWaitSec = 60
)
. "$PSScriptRoot\nodes.ps1"
$nodeList = @(Get-NodeList $RankNodes)
$EDGES = @(Get-P2PEdges $nodeList)
function Say($s){ Write-Output ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $s) }

Say "post-mortem: fabric state after the failure, before any reboot"

# ---- 1. leftover ranks and the USB4 adapters on every node ----------------------------------------
# The counters are cumulative since boot, and prep_fabric.ps1 starts every run on a fresh boot, so
# they cover this run only (plus prep's probes).
$adapters = @"
`$gw = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -EA SilentlyContinue | ForEach-Object { `$_.ifIndex })
'llama-cli alive=' + @(Get-Process llama-cli -EA SilentlyContinue).Count
foreach(`$a in @(Get-NetAdapter -EA SilentlyContinue | Where-Object { `$_.InterfaceDescription -match '$($CLUSTER_CFG.P2PMatch)' -and `$gw -notcontains `$_.ifIndex })){
  `$s = Get-NetAdapterStatistics -Name `$a.Name -EA SilentlyContinue
  `$ip = @(Get-NetIPAddress -InterfaceIndex `$a.ifIndex -AddressFamily IPv4 -EA SilentlyContinue | ForEach-Object { `$_.IPAddress }) -join ','
  `$mtu = (Get-NetIPInterface -InterfaceIndex `$a.ifIndex -AddressFamily IPv4 -EA SilentlyContinue).NlMtu
  '{0} ip={1} status={2} speed={3} mtu={4} rxPk={5} txPk={6} rxErr={7} rxDisc={8} txErr={9} txDisc={10}' -f `$a.Name, `$ip, `$a.Status, `$a.LinkSpeed, `$mtu, `$s.ReceivedUnicastPackets, `$s.SentUnicastPackets, `$s.ReceivedPacketErrors, `$s.ReceivedDiscardedPackets, `$s.OutboundPacketErrors, `$s.OutboundDiscardedPackets
}
"@
foreach($n in $nodeList){
  $r = Invoke-Node -N $n -Ps -TimeoutSec 200 -Cmd $adapters
  foreach($ln in ("$($r.StdOut)" -split "`r?`n" | Where-Object { $_.Trim() })){ Write-Output ("  [{0}] {1}" -f $n, $ln.Trim()) }
}

# ---- 2. every direction of every cable: TCP connect, small ping, full-size ping -------------------
Write-Output ""
Say "every leg: TCP:22 connect, 32 B ping, 1472 B don't-fragment ping"
foreach($e in $EDGES){
  $reply = "Reply from $($e.To): bytes="
  $probe = "`$c=[Net.Sockets.TcpClient]::new();`$o=`$false;try{`$o=`$c.ConnectAsync('$($e.To)',22).Wait(8000)}catch{};try{`$c.Close()}catch{};" +
           "`$s=@(ping -n 2 -w 1500 -l 32 $($e.To) | Select-String -SimpleMatch '$reply').Count; " +
           "`$b=@(ping -n 2 -w 1500 -f -l 1472 $($e.To) | Select-String -SimpleMatch '$reply').Count; 'tcp=' + `$o + ' small=' + `$s + ' big=' + `$b"
  $r = "$((Invoke-Node -N $e.From -Ps -TimeoutSec 200 -Cmd $probe).StdOut)".Trim()
  Write-Output ("  {0,-10} {1,-16} {2}" -f $e.Label, $e.To, $r)
}

# ---- 3. NetworkDirect on every cable, one direction each ------------------------------------------
# The first direction listed for a cable dials, the other end serves. A cable that passes step 2
# and fails here has a broken ND stack on a live link; one that fails both is down.
if(-not $NoNd){
  $seen = @{}
  foreach($e in $EDGES){
    $key = (@($e.From, $e.ToNode) | Sort-Object) -join '|'
    if($seen[$key]){ continue }
    $seen[$key] = $true
    Write-Output ""
    & powershell @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'nd_pair.ps1'),
                   '-ServerNode', $e.ToNode, '-ClientNode', $e.From, '-WaitSec', "$NdWaitSec") 2>&1 |
      Where-Object { "$_" -match '====|RESULT|provider addresses|Connect returned|Test failed' } | ForEach-Object { Write-Output ("  " + "$_".Trim()) }
  }
}
Write-Output ""
Say "post-mortem done"
