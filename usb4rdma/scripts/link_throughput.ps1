# Bulk TCP throughput over ONE USB4 cable. -Server sends, -Client receives, over the cable between
# them (from $CLUSTER_P2P in nodes.ps1); both default to the 2-rank ring's two nodes. Run it once
# per cable, before and after a session: a USB4 link can degrade under sustained load while still
# reporting Up, and a degraded link stalls a run with no error.
#
# Needs an Administrator ssh user on the server (HttpListener binds http://+:<Port>/) and TCP
# <Port> allowed inbound on the server's USB4 leg (Private profile).
#
# The measurement is sized so that startup cost does not matter:
#   1. 4 GiB payload, generated in memory on the server (no disk read on the sender).
#   2. The first 512 MiB is discarded before the clock starts, so connect, slow start and process
#      start are outside the measured window: this reports steady-state throughput. A small
#      payload measures mostly fixed overhead and under-reads a healthy link badly.
#   3. Both endpoints run at High priority (Windows EcoQoS can throttle detached processes).
#
# Exit code: 0 = at or above -FailBelowMBs, 1 = below it or no figure at all.
param([string]$Server = '', [string]$Client = '', [int]$Port = 8902, [double]$FailBelowMBs = 150)
. "$PSScriptRoot\nodes.ps1"
if(-not $Server -or -not $Client){
  $pair = @(Get-NodeList $CLUSTER_RINGS[2])
  if(-not $Server){ $Server = $pair[0] }
  if(-not $Client){ $Client = $pair[1] }
}
$null = Get-NodeList "$Server,$Client"

$SRV_IP   = Get-P2PAddr $Client $Server        # the server's end of the cable
$W        = 'C:\ProgramData\disttp_link'       # node-local scratch for the two helper scripts
$TOTAL    = 4GB
$WARMUP   = 512MB

# ---------- generator on the server ----------
$serverPs = @"
`$ErrorActionPreference='Continue'
try{ (Get-Process -Id `$PID).PriorityClass='High' }catch{}
`$listener = [System.Net.HttpListener]::new()
`$listener.Prefixes.Add('http://+:$Port/')
`$listener.Start()
`$chunk = New-Object byte[] 4194304
(New-Object Random 1234).NextBytes(`$chunk)
while(`$listener.IsListening){
  try {
    `$ctx = `$listener.GetContext()
    `$ctx.Response.ContentType = 'application/octet-stream'
    `$ctx.Response.ContentLength64 = $TOTAL
    if(`$ctx.Request.HttpMethod -eq 'HEAD'){ `$ctx.Response.Close(); continue }
    `$sent = 0L
    try {
      while(`$sent -lt $TOTAL){
        `$ctx.Response.OutputStream.Write(`$chunk, 0, `$chunk.Length)
        `$sent += `$chunk.Length
      }
    } catch { }
    try { `$ctx.Response.Close() } catch { }
  } catch { }
}
"@
$launchServer = @"
`$ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force -Path '$W' | Out-Null
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | ? { `$_.CommandLine -like '*linksrv.ps1*' } | % { Stop-Process -Id `$_.ProcessId -Force -EA SilentlyContinue }
Set-Content -Path '$W\linksrv.ps1' -Value @'
$serverPs
'@ -Encoding ASCII
`$cl = 'powershell -NoProfile -ExecutionPolicy Bypass -File $W\linksrv.ps1'
`$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=`$cl}
Start-Sleep 3
try{ (Get-Process -Id `$p.ProcessId).PriorityClass='High'; `$r='raised' }catch{ `$r='raise_failed' }
'SRV_PID=' + `$p.ProcessId + ' ' + `$r
"@
Write-Output "== bulk generator on ${Server}:$Port ($SRV_IP) =="
Write-Output ("  " + (Invoke-Node -N $Server -Ps -TimeoutSec 120 -Cmd $launchServer).StdOut.Trim())
Start-Sleep 5

# ---------- timed sink on the client ----------
$sinkPs = @"
`$ErrorActionPreference='Stop'
try{ (Get-Process -Id `$PID).PriorityClass='High' }catch{}
Add-Type -AssemblyName System.Net.Http
try {
  `$hc = [System.Net.Http.HttpClient]::new()
  `$hc.Timeout = [TimeSpan]::FromMinutes(20)
  `$resp = `$hc.GetAsync('http://${SRV_IP}:$Port/bulk', [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  `$in = `$resp.Content.ReadAsStreamAsync().Result
  `$buf = New-Object byte[] 4194304
  `$tot = 0L; `$measured = 0L; `$sw = `$null
  while(`$true){
    `$n = `$in.Read(`$buf, 0, `$buf.Length)
    if(`$n -le 0){ break }
    `$tot += `$n
    if(`$sw -eq `$null -and `$tot -ge $WARMUP){ `$sw = [Diagnostics.Stopwatch]::StartNew() }
    elseif(`$sw -ne `$null){ `$measured += `$n }
  }
  `$sw.Stop()
  `$mb = `$measured / 1MB
  `$s  = `$sw.Elapsed.TotalSeconds
  Set-Content -Path '$W\link.out' -Encoding ASCII -Value ("STEADY={0:N1} MB/s  measured={1} bytes in {2:N2}s  total={3} bytes" -f (`$mb/`$s), `$measured, `$s, `$tot)
} catch {
  Set-Content -Path '$W\link.out' -Encoding ASCII -Value ('ERR: ' + `$_.Exception.Message)
}
"@
$launchSink = @"
`$ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force -Path '$W' | Out-Null
Remove-Item $W\link.out -EA SilentlyContinue
Set-Content -Path '$W\linksink.ps1' -Value @'
$sinkPs
'@ -Encoding ASCII
`$cl = 'powershell -NoProfile -ExecutionPolicy Bypass -File $W\linksink.ps1'
`$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=`$cl}
Start-Sleep 3
try{ (Get-Process -Id `$p.ProcessId).PriorityClass='High'; `$r='raised' }catch{ `$r='raise_failed' }
'SINK_PID=' + `$p.ProcessId + ' ' + `$r
"@
Write-Output "== timed sink on $Client =="
Write-Output ("  " + (Invoke-Node -N $Client -Ps -TimeoutSec 120 -Cmd $launchSink).StdOut.Trim())

$verdict = $null
for($i=0; $i -lt 40; $i++){
  Start-Sleep 10
  $o = "$((Invoke-Node -N $Client -Ps -TimeoutSec 120 -Cmd "if(Test-Path '$W\link.out'){ Get-Content '$W\link.out' -Raw }").StdOut)".Trim()
  if($o -match 'STEADY=([\d,.]+)\s*MB/s'){
    $mbps = [double](($Matches[1]) -replace ',','')
    Write-Output ("  RESULT: " + $o)
    if($mbps -ge $FailBelowMBs){ Write-Output ("  VERDICT: link healthy ({0:N1} MB/s steady-state)" -f $mbps) }
    else { Write-Output ("  VERDICT: DEGRADED at {0:N1} MB/s steady-state" -f $mbps) }
    $verdict = $mbps; break
  }
  if($o -match '^ERR'){ Write-Output ("  " + $o); break }
}
if($null -eq $verdict){ Write-Output "  WARN: no steady-state figure produced" }

# Stop the generator so it does not hold the port.
Invoke-Node -N $Server -Ps -TimeoutSec 90 -Cmd "Get-CimInstance Win32_Process -Filter `"Name='powershell.exe'`" | ? { `$_.CommandLine -like '*linksrv.ps1*' } | % { Stop-Process -Id `$_.ProcessId -Force -EA SilentlyContinue }; 'srv stopped'" | Out-Null

if($null -eq $verdict -or $verdict -lt $FailBelowMBs){ exit 1 }
exit 0
