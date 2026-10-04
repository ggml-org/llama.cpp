# Put one node's USB4 P2P legs on the right subnets, BY MEASUREMENT rather than by name.
# prep_fabric.ps1 calls this for every node marked Readdress in nodes.ps1.
#
# On some machines the USB4 P2P adapters come back as new device instances after a boot, so a
# static address does not stay attached to the cable it was set on. A leg can then carry the other
# cable's address, or a duplicate of the far end's, and ARP fails on both legs while the adapters
# still report connected. That reads as two dead cables; it is not a cable fault.
#
# Neither the adapter name nor the ifIndex identifies which cable is which, and both can move across
# boots. So: assign, probe the far end over TCP:22 from this node, and if that fails swap and probe
# again. The end state is read back from the host, never assumed.
#
# Only cables to nodes in -RankNodes are assigned and probed. Handles one or two legs; more than two
# needs a permutation search and fails loudly instead of guessing.
param(
  [Parameter(Mandatory)][string]$Node,
  # Comma string, not [string[]]: powershell -File passes "A,B,C" as one element. Empty = every
  # node in nodes.ps1.
  [string]$RankNodes = '',
  [switch]$WhatIf
)
. "$PSScriptRoot\nodes.ps1"

$null = Get-NodeList $Node
if(-not $RankNodes){ $RankNodes = @($CLUSTER_NODES.Keys) -join ',' }
$peers = @(Get-NodeList $RankNodes | Where-Object { $_ -ne $Node })

# This node's cables to participating peers: my address on the cable, and the far end's.
$want = @(foreach($p in $peers){
  try { [pscustomobject]@{ Peer=$p; Mine=(Get-P2PAddr $p $Node); Far=(Get-P2PAddr $Node $p) } } catch {}
})
if(-not $want.Count){ Write-Output "$Node has no cable to any of $($peers -join ','); nothing to do"; return }
if($want.Count -gt 2){ throw "$Node has $($want.Count) cables to participating nodes; this script handles at most 2" }

# Every configured /24, so a stale address from ANY cable is stripped.
$pfx = (@(Get-P2PPrefixes) | ForEach-Object { "'$_'" }) -join ','

function Legs(){
  $c = @"
`$gw = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -EA SilentlyContinue | ForEach-Object { `$_.ifIndex })
Get-NetAdapter -EA SilentlyContinue |
  Where-Object { `$_.InterfaceDescription -match '$($CLUSTER_CFG.P2PMatch)' -and `$_.Status -eq 'Up' -and `$gw -notcontains `$_.ifIndex } |
  ForEach-Object {
    `$ip = (Get-NetIPAddress -InterfaceIndex `$_.ifIndex -AddressFamily IPv4 -EA SilentlyContinue |
           ForEach-Object { `$_.IPAddress }) -join ','
    '{0}|{1}|{2}' -f `$_.Name, `$_.ifIndex, `$ip
  }
"@
  $out = (Invoke-Node -N $Node -Ps -TimeoutSec 300 -Cmd $c).StdOut
  $legs = @()
  foreach($ln in ($out -split "`r?`n")){
    $p = $ln.Trim() -split '\|'
    if($p.Count -ge 2 -and $p[1] -match '^\d+$'){
      $legs += [pscustomobject]@{ Name=$p[0]; Idx=[int]$p[1]; Ip=$(if($p.Count -ge 3){$p[2]}else{''}) }
    }
  }
  return $legs
}

# ICMP can be filtered on a healthy leg, so probe TCP:22. Returns $true only when EVERY far end
# answers.
function Probe(){
  $targets = ($want | ForEach-Object { "'$($_.Far)'" }) -join ','
  $c = @"
`$r = @()
foreach(`$t in @($targets)){
  `$s=[Net.Sockets.TcpClient]::new(); `$o=`$false
  try{ `$o=`$s.ConnectAsync(`$t,22).Wait(8000) }catch{}
  try{ `$s.Close() }catch{}
  `$r += ('{0}={1}' -f `$t, `$o)
}
`$r -join ' '
"@
  $o = (Invoke-Node -N $Node -Ps -TimeoutSec 300 -Cmd $c).StdOut.Trim()
  # Write-Host, not Write-Output: anything written to the pipeline here would become part of the
  # return value, and a non-empty array is always true in `if(Probe)`.
  Write-Host ("  probe: " + $o)
  $all = $true
  foreach($w in $want){ if($o -notmatch ([regex]::Escape($w.Far) + '=True')){ $all = $false } }
  return $all
}

# $names[i] gets $want[i].Mine. Strips every configured-subnet address from EVERY leg first: a stale
# one left behind puts two addresses on an adapter (or the same address on two), and Windows then
# picks a source address that has no route to the peer.
function Assign($names){
  # 'name|address' strings, split on the node. Nested arrays do not survive here: with ONE leg,
  # @(@('Ethernet 4','192.168.77.2')) flattens to two strings.
  $pairs = (0..($want.Count - 1) | ForEach-Object { "'$($names[$_])|$($want[$_].Mine)'" }) -join ','
  $all = ($legs | ForEach-Object { "'$($_.Name)'" }) -join ','
  $c = @"
`$pfx = @($pfx)
foreach(`$n in @($all)){
  `$a = Get-NetAdapter -Name `$n -EA SilentlyContinue
  if(-not `$a){ continue }
  Get-NetIPAddress -InterfaceIndex `$a.ifIndex -AddressFamily IPv4 -EA SilentlyContinue |
    Where-Object { `$pfx -contains ((`$_.IPAddress -split '\.')[0..2] -join '.') } |
    ForEach-Object { Remove-NetIPAddress -InputObject `$_ -Confirm:`$false -EA SilentlyContinue }
}
foreach(`$pair in @($pairs)){
  `$p = `$pair -split '\|'
  `$a = Get-NetAdapter -Name `$p[0] -EA SilentlyContinue
  if(-not `$a){ '  {0}: no such adapter' -f `$p[0]; continue }
  try { New-NetIPAddress -InterfaceIndex `$a.ifIndex -IPAddress `$p[1] -PrefixLength 24 -EA Stop | Out-Null }
  catch { '  {0} -> {1} FAILED: {2}' -f `$p[0], `$p[1], `$_.Exception.Message; continue }
  try { Set-NetConnectionProfile -InterfaceIndex `$a.ifIndex -NetworkCategory Private -EA Stop } catch {}
  '  {0} -> {1}' -f `$p[0], `$p[1]
}
"@
  Write-Output ((Invoke-Node -N $Node -Ps -TimeoutSec 300 -Cmd $c).StdOut.TrimEnd())
  Start-Sleep 5
}

$legs = @(Legs)
Write-Output "== $Node's connected USB4 P2P legs =="
$legs | ForEach-Object { Write-Output ("  {0,-12} idx={1,-4} ip=[{2}]" -f $_.Name, $_.Idx, $_.Ip) }
if($legs.Count -lt $want.Count -or $legs.Count -gt 2){
  throw "$Node has $($legs.Count) connected USB4 P2P leg(s) and $($want.Count) cable(s) to participating nodes; check the cables"
}

# If the current assignment already works, change nothing.
Write-Output "`n== current state"
if(Probe){ Write-Output "OK: every edge already live, leaving $Node's addressing untouched"; return }

if($WhatIf){ Write-Output "-WhatIf: would re-address, stopping here"; return }

# Assign inside each branch. `$orders = if(...){ @(,@(x)) }` would unroll one level through the if
# statement's output, and the single-leg case would then iterate the characters of the name.
if($legs.Count -eq 1){ $orders = @(,@($legs[0].Name)) } else { $orders = @(@($legs[0].Name, $legs[1].Name), @($legs[1].Name, $legs[0].Name)) }
foreach($order in $orders){
  Write-Output ("`n== trying: " + ((0..($want.Count - 1) | ForEach-Object { "{0}-facing='{1}' -> {2}" -f $want[$_].Peer, $order[$_], $want[$_].Mine }) -join '   ') + " ==")
  Assign $order
  if(Probe){
    Write-Output "`nOK: $Node is live on every edge. Config read back from the host:"
    Legs | ForEach-Object { Write-Output ("  {0,-12} idx={1,-4} ip=[{2}]" -f $_.Name, $_.Idx, $_.Ip) }
    return
  }
}
throw "$Node could not reach $(($want | ForEach-Object { "$($_.Peer) ($($_.Far))" }) -join ' and ') on any leg assignment; check the cables, and that the far ends are addressed"
