# Entry script of the QDC POWERSHELL job; QDC extracts the artifact to C:\Temp\TestContent.
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Logs = 'C:\Temp\QDC_Logs'
New-Item -ItemType Directory -Path $Logs -Force | Out-Null
Start-Transcript -Path (Join-Path $Logs 'run_tests.transcript.log') -Force | Out-Null

# Trust the cert that signed libggml-htp.cat (shipped in the package by the build job)
foreach ($store in 'Root', 'TrustedPublisher') {
    certutil.exe -addstore -f $store llama.cpp\ggml-htp-v1.cer | Out-Host
}

# The device image has no Python: unpack the bundled embeddable one
Expand-Archive -LiteralPath python-embed-arm64.zip -DestinationPath C:\Temp\python-embed -Force
$pth = 'C:\Temp\python-embed\python311._pth'
(Get-Content $pth -Raw) -replace '#\s*import site', 'import site' | Set-Content $pth -Encoding ascii
$py = 'C:\Temp\python-embed\python.exe'
& $py get-pip.py --no-warn-script-location | Out-Host
& $py -m pip install pytest | Out-Host

& $py -m pytest tests\ -v -s --junitxml=results.xml
$rc = $LASTEXITCODE
Copy-Item results.xml $Logs -ErrorAction SilentlyContinue

Stop-Transcript | Out-Null
New-Item -ItemType File -Path C:\Temp\QDCTestDone.txt -Force | Out-Null   # QDC completion sentinel
exit $rc
