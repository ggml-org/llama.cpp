# Test NetworkDirect connect in both directions of every USB4 cable between the given nodes, and
# print a summary. Use it to choose the ring order in nodes.ps1.
#
# Each direction is one nd_pair.ps1 call: the server node accepts, the client node connects.
# dist-tp's lower rank accepts and the higher rank connects on every ring edge, so a ring may only
# use directions that pass here.
#
# A failed test ends with ndping killed mid-handshake, which can leave ND state broken for the
# tests after it. If a direction fails, re-run the ones after it on their own with nd_pair.ps1.
#
#   .\nd_all_directions.ps1                  # every cable in nodes.ps1
#   .\nd_all_directions.ps1 -RankNodes A,B   # only the cable(s) between these nodes
param(
  # Comma string. Empty = every node in nodes.ps1.
  [string]$RankNodes = '',
  [int]$WaitSec = 45
)
. "$PSScriptRoot\nodes.ps1"
if(-not $RankNodes){ $RankNodes = @($CLUSTER_NODES.Keys) -join ',' }
$nodeList = @(Get-NodeList $RankNodes)
$edges = @(Get-P2PEdges $nodeList)
if(-not $edges.Count){ throw "no USB4 cable between any two of $($nodeList -join ',') in `$CLUSTER_P2P" }

$summary = @()
foreach($e in $edges){
  # Edge From -> To(Node): From dials, so From is the client and the far end the server.
  $cli = $e.From; $srv = $e.ToNode
  $out = & "$PSScriptRoot\nd_pair.ps1" -ServerNode $srv -ClientNode $cli -WaitSec $WaitSec *>&1 | ForEach-Object { "$_" }
  $out | Where-Object { $_ -match 'RESULT|^\s+\d+l\s+\d+l|provider addresses' } | ForEach-Object { Write-Output $_ }
  $res = if($out -match 'ND PASS'){ 'PASS' } else { 'FAIL' }
  $summary += ('  {0} -> {1}  {2}' -f $cli, $srv, $res)
}
Write-Output ''
Write-Output 'SUMMARY (client connects -> server accepts):'
$summary | ForEach-Object { Write-Output $_ }
Write-Output ('PASSED {0} of {1}' -f @($summary | Where-Object { $_ -match 'PASS' }).Count, $summary.Count)
