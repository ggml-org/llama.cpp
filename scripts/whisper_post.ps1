param(
    [string]$OutputFile = 'C:\Users\Pc\Kellen.cpp\scripts\whisper_out.txt',
    [string]$ReceiverUrl = 'http://127.0.0.1:9000/test'
)

$lines = Get-Content -Path $OutputFile -ErrorAction Stop
$transcriptLines = $lines | Where-Object { $_ -match '^\[' } | ForEach-Object { $_ -replace '^\[.*?\]\s*', '' }
$transcript = ($transcriptLines -join ' ')
$transcript = $transcript.Trim()
if (-not $transcript) {
    Write-Error 'No transcript found in output file.'
    exit 1
}
$json = @{ prompt = $transcript } | ConvertTo-Json -Compress
Write-Output 'TRANSCRIPT:'
Write-Output $transcript
Invoke-RestMethod -Uri $ReceiverUrl -Method Post -Body $json -ContentType 'application/json' -ErrorAction Stop
Write-Output 'POST done'