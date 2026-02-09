## Overview

The document covers procedures for installing the latest GPU and NPU drivers, and OpenCL and Hexagon SDKs.


In order to use Hexagon NPU on Snapdragon Windows devices the underlying HTP Ops libraries (e.g libggml-htp-v73.so)
must be included in the .cat file digitally signed with a trusted certificate.

This document covers details on how to generate personal certificate files (.pfx) and how to configure the system
to allow for test signatures (aka test-signing).

## Install the latest Adreno OpenCL SDK

Either use the trimmed down version (optimized for CI) from

    https://github.com/snapdragon-toolchain/opencl-sdk/releases/download/v2.3.2/adreno-opencl-sdk-v2.3.2-arm64-wos.tar.xz

Or download the complete official version from

    https://softwarecenter.qualcomm.com/catalog/item/Adreno_OpenCL_SDK?version=2.3.2

Unzip/untar the archive into
```
c:\Qualcomm\OpenCL_SDK\2.3.2
```

## Install the latest Hexagon SDK Community Edition

Either use the trimmed down version (optimized for CI) from

    https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.4.0.2/hexagon-sdk-v6.4.0.2-arm64-wos.tar.xz

Or download the complete official version from

    https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK?version=6.4.0.2

Unzip/untar the archive into
```
c:\Qualcomm\Hexagon_SDK\6.4.0.2
```

## Install the latest Adreno GPU driver

Download the driver from

    https://softwarecenter.qualcomm.com/catalog/item/Windows_Graphics_Driver

After the automated installation and reboot please make sure that the GPU device shows up in the `Device Manager` (under 'Display Adapters`)

## Install the latest Qualcomm NPU driver

Download the driver from

    https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_HND

After the automated installation and reboot please make sure that the Hexagon NPU device shows up in the `Device Manager` (under `Neural Processors`).

If the device is not available you can try installing all components (`qcnspmcdm8380`, `qcnspmcdm8380_ext`) manually.
The components are extracted into
```
c:\QCDrivers\qcnspmcdm...
```

## Enable NPU driver test signatures

Please note that the following steps are required only for the Hexagon NPU.
Adreno GPU backend does not require test signatures.

### Enable testsigning

Use `bcdedit` to enable test-signing
```
> bcdedit /set TESTSIGNING ON
```
(Secure Boot may need to be disabled for this to work)

Make sure test-signing is enabled after reboot
```
> bcdedit /enum
...
testsigning             Yes
...
```
For additional details see Microsoft guide at

   https://learn.microsoft.com/en-us/windows-hardware/drivers/install/the-testsigning-boot-configuration-option

### Create personal certificate

The tools required for this procedure are available as part of Windows SDK and Windows Driver Kit which should be
installed as part of the MS Visual Studio.
They are typically located at
```
c:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0
```
(replace 10.0.26100.0 with correct version).

To create personal self-signed certificate run the following commands (either from cmd or power-shell):
```
> cd c:\Users\MyUser
> mkdir Certs
> cd Certs
> makecert -r -pe -ss PrivateCertStore -n CN=GGML.HTP.v1 -eku 1.3.6.1.5.5.7.3.3 -sv ggml-htp-v1.pvk ggml-htp-v1.cer
> pvk2pfx.exe -pvk ggml-htp-v1.pvk -spc ggml-htp-v1.cer -pfx ggml-htp-v1.pfx
```
(replace `MyUser` with your username).

Add this certificate to `Trusted Root Certification Authorities` and `Trusted Publishers` stores.
This can be done using `certlm` Certificate Manager tool.
Right click on the certificate store, select `All Tasks -> Import` and follow the prompts to import the certificate from the
PFX file you created above.

For additional details see Microsoft guide at

    https://learn.microsoft.com/en-us/windows-hardware/drivers/install/introduction-to-test-signing

Make sure to save the PFX file, you will need it for the build procedures.
Please note that the same certificate can be used for signing any number of builds.

## Build Hexagon backend with signed HTP ops libraries

The overall Hexagon backend build procedure for Windows on Snapdragon is the same as for other platforms.
However, additional settings are required for generating and signing HTP Ops libraries.
```
> $env:OPENCL_SDK_ROOT="C:\Qualcomm\OpenCL_SDK\2.3.2"
> $env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.4.0.2"
> $env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.4.0.2\tools\HEXAGON_Tools\19.0.04"
> $env:HEXAGON_HTP_CERT="c:\Users\MyUsers\Certs\ggml-htp-v1.pfx"
> $env:WINDOWS_SDK_BIN="C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64"

> cmake --preset arm64-windows-snapdragon-release -B build-wos
...
> cmake --install build-wos --prefix pkg-snapdragon
```

Once the build is complete HTP ops libraries will be installed like this
```
> dir pkg-snapdragon/lib
...
-a----         1/22/2026   6:01 PM         187656 libggml-htp-v73.so
-a----         1/22/2026   6:01 PM         191752 libggml-htp-v75.so
-a----         1/22/2026   6:01 PM         187656 libggml-htp-v79.so
-a----         1/22/2026   6:01 PM         187656 libggml-htp-v81.so
-a----         1/22/2026   6:01 PM           4139 libggml-htp.cat
```

The .cat file, the signature and proper certicate installation can be verified with

```
> signtool.exe verify /v /pa .\pkg-snapdragon\lib\libggml-htp.cat
Verifying: .\pkg-snapdragon\lib\libggml-htp.cat

Signature Index: 0 (Primary Signature)
Hash of file (sha256): 9820C664DA59D5EAE31DBB664127FCDAEF59CDC31502496BC567544EC2F401CF

Signing Certificate Chain:
        Issued to: GGML.HTP.v1
...
Successfully verified: .\pkg-snapdragon\lib\libggml-htp.cat
...
```
