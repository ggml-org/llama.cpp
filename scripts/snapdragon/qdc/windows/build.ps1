# Builds llama.cpp for Snapdragon Windows into pkg-snapdragon/llama.cpp (run from the repo root).
# The HTP catalog is signed with a throwaway cert; its .cer ships in the package so the
# QDC device test can trust exactly what signed it. No secrets involved.
$ErrorActionPreference = 'Stop'

function Invoke-Native { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE) { exit $LASTEXITCODE } }

# Signing cert
$cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=GGML.HTP.v1' -CertStoreLocation Cert:\CurrentUser\My
$pfx = Join-Path $env:TEMP 'ggml-htp-v1.pfx'
$cer = Join-Path $env:TEMP 'ggml-htp-v1.cer'
[IO.File]::WriteAllBytes($pfx, $cert.Export([Security.Cryptography.X509Certificates.X509ContentType]::Pfx))
Export-Certificate -Cert $cert -FilePath $cer | Out-Null
$env:HEXAGON_HTP_CERT = $pfx

# setup-sdk.py stores the SDK paths in the user environment
foreach ($n in 'HEXAGON_SDK_ROOT', 'HEXAGON_TOOLS_ROOT', 'OPENCL_SDK_ROOT') {
    Set-Item "env:$n" ([Environment]::GetEnvironmentVariable($n, 'User'))
}
$env:WINDOWS_SDK_BIN = (Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.*" -Directory |
    Where-Object { Test-Path "$($_.FullName)\arm64\signtool.exe" } |
    Sort-Object Name | Select-Object -Last 1).FullName

Copy-Item docs/backend/snapdragon/CMakeUserPresets.json .
Invoke-Native cmake --preset arm64-windows-snapdragon-release -B build-wos
Invoke-Native cmake --build build-wos
Invoke-Native cmake --install build-wos --prefix pkg-snapdragon/llama.cpp
Copy-Item $cer pkg-snapdragon/llama.cpp
