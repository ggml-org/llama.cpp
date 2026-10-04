# Test NetworkDirect between two nodes, in a named direction, with ndping.
#
# Direction matters: dist-tp brings up each ring edge as one rank connecting and the other
# accepting, so an RDMA stack that can connect but not accept gives a partial ring: some edges on
# ND, the rest on TCP, with a headline that still says RDMA.
#
#   .\nd_pair.ps1 -ServerNode A -ClientNode B      # does A accept from B?
#   .\nd_pair.ps1 -ServerNode B -ClientNode A      # does B accept from A?
#
# -ServerIp defaults to the server's end of the cable between the two, from $CLUSTER_P2P.
#
# Two traps this script handles, both of which make a healthy stack look dead:
#   - ndping takes the IP as the LAST token; a flag after it is parsed as the address.
#   - server mode must be killed, and a killed ndping loses its buffered stdout, so its output is
#     redirected to a file on the node rather than read from the ssh channel.
param(
  [Parameter(Mandatory)][string]$ServerNode,
  [string]$ServerIp = '',
  [Parameter(Mandatory)][string]$ClientNode,
  [int]$WaitSec = 120
)
. "$PSScriptRoot\nodes.ps1"
$null = Get-NodeList "$ServerNode,$ClientNode"
if(-not $ServerIp){ $ServerIp = Get-P2PAddr $ClientNode $ServerNode }
$NDPING = Join-Path $CLUSTER_CFG.NdDir 'ndping.exe'
$W = 'C:\ProgramData\disttp_nd'

Write-Output "==== ndping: server on $ServerNode ($ServerIp)  <-  client on $ClientNode ===="

foreach($n in @($ServerNode, $ClientNode)){
  Invoke-Node -N $n -Ps -TimeoutSec 150 -Cmd 'Get-Process ndping -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue; "clean"' | Out-Null
}

$srv = @"
`$ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force -Path '$W' | Out-Null
Remove-Item '$W\ndsrv.out' -EA SilentlyContinue
`$cl = 'cmd /c $NDPING -s $ServerIp > $W\ndsrv.out 2>&1'
`$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=`$cl}
Start-Sleep 3
foreach(`$x in @(Get-Process -Name ndping -EA SilentlyContinue)){ try{ `$x.PriorityClass='High' }catch{} }
'server pid=' + `$p.ProcessId + ' alive=' + @(Get-Process ndping -EA SilentlyContinue).Count
"@
Write-Output ("  " + (Invoke-Node -N $ServerNode -Ps -TimeoutSec 200 -Cmd $srv).StdOut.Trim())
Start-Sleep 5

$cli = @"
`$ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force -Path '$W' | Out-Null
Get-Process ndping -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Remove-Item '$W\ndcli.out' -EA SilentlyContinue
`$cl = 'cmd /c $NDPING -c $ServerIp > $W\ndcli.out 2>&1'
`$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=`$cl}
Start-Sleep 3
foreach(`$x in @(Get-Process -Name ndping -EA SilentlyContinue)){ try{ `$x.PriorityClass='High' }catch{} }
for(`$i=0; `$i -lt $([int]($WaitSec/3)); `$i++){ Start-Sleep 3; if(-not (Get-Process ndping -EA SilentlyContinue)){ break } }
Get-Process ndping -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Get-Content '$W\ndcli.out' -EA SilentlyContinue | Out-String
"@
$out = (Invoke-Node -N $ClientNode -Ps -TimeoutSec ($WaitSec + 180) -Cmd $cli).StdOut

Invoke-Node -N $ServerNode -Ps -TimeoutSec 150 -Cmd 'Get-Process ndping -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue' | Out-Null

# The provider address list is the first gate: "Provider returned 0 addresses" means no
# RDMA-capable driver is loaded, and every dist-tp run on this node would be TCP.
if($out -match 'Provider returned (\d+) addresses'){ Write-Output ("  provider addresses on $ClientNode : " + $Matches[1]) }
foreach($ln in ($out -split "`r?`n")){
  if($ln -match 'addr\[\d+\]|NdResolveAddress|Connect returned|Test failed|Test Result|End of test'){ Write-Output ("  " + $ln.Trim()) }
}
if($out -match 'End of test'){
  Write-Output ""
  Write-Output "  RESULT: $ClientNode -> $ServerNode  ND PASS"
  ($out -split "`r?`n") | Where-Object { $_ -match '^\s+\d+l\s+\d+l' } | Select-Object -First 4 | ForEach-Object { Write-Output ("    " + $_.Trim()) }
} else {
  Write-Output ""
  Write-Output "  RESULT: $ClientNode -> $ServerNode  ND FAIL (server on $ServerNode did not complete the exchange)"
}
