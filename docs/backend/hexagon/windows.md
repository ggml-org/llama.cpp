## Install latest Hexagon SDK Community Edition

Download complete official version from
   https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK?version=6.4.0.2

Or use the trimmed down version (optimized for CI) from
   https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.4.0.2/hexagon-sdk-v6.4.0.2-arm64-wos.tar.xz

Unzip/untar into
   c:\Qualcomm\Hexagon_SDK\6.4.0.2

## Install latest Qualcomm NPU driver

Download from
   https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_HND

Make sure to install all components (qcnspmcdm8380, qcnspmcdm8380_ext)
Make sure that the Hexagon NPU device shows up in Device Manager (under Neural Processors).

## Enabled Test Signatures for drivers

See detailed guide at
   https://learn.microsoft.com/en-us/windows-hardware/drivers/install/the-testsigning-boot-configuration-option

To enable test-signing
   bcdedit -set TESTSIGNING ON

(Secure Boot may need to be disabled for this to work)

Make sure test-signing is enabled after reboot

   bcdedit /enum
   ...
   testsigning             Yes
   ...

## Create personal certificate

See detailed guide at 
   https://learn.microsoft.com/en-us/windows-hardware/drivers/install/introduction-to-test-signing

Tools required for this procedure are available as part of Windows SDK and are located in

   C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64

(replace 10.0.26100.0 with correct version).

To create private self-signed certificate run

   cd c:\Users\MyUser
   mkdir Certs
   cd Certs
   makecert -r -pe -ss PrivateCertStore -n CN=GGML.HTP.v1 -eku 1.3.6.1.5.5.7.3.3 -sv ggml-htp-v1.pkv ggml-htp-v1.cer
   pvk2pfx.exe' -pvk ggml-htp-v1.pvk -spc ggml-htp-v1.cer -pfx ggml-htp-v1.pfx

Add this certificate to "Trusted Root Certification Authorities" and "Trusted Publishers".

## Build

   $env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.4.0.2"
   $env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.4.0.2\tools\HEXAGON_Tools\19.0.04"
   $env:WINDOWS_SDK_BIN="x"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64"

   cmake --preset arm64-windows-snapdragon -B build-wos -D GGML_HEXAGON_HTP_CERT="c:\Users\MyUsers\Certs\ggml-htp-v1.pfx"
   cmake --install build-wos --prefix pkg-snapdragon
