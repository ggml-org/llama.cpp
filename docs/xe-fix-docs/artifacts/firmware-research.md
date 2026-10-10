# A770 DG2 firmware research, 2026-09-29

Read-only investigation. No installation, firmware flash, configuration change,
driver unload, service interruption, reboot, or GPU benchmark performed.

## Conclusion

There is no newer applicable DG2 GuC firmware in upstream linux-firmware main
than the 70.53.0 already running here. The downloaded upstream binary exactly
matches the decompressed installed file. The current kernel's initramfs has the
same GuC and DMC bytes as disk, so an old initramfs is not hiding an available
update. This closes a concrete firmware-version mismatch hypothesis; it does
not establish that GuC itself is bug-free.

Upstream also lists DG2 HuC 7.10.16 and DMC 2.08, matching the reported i915
baseline and current DMC. Firmware version numbers for newer GPU families,
including Battlemage GuC 70.7x, are not upgrade candidates for DG2.
[linux-firmware WHENCE](https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/WHENCE)

## Direct host evidence

Observed with read-only commands during this investigation:

| Item | Observed |
| --- | --- |
| Kernel | `7.3.0-rc1-273-tkg-bore` |
| Firmware package | `linux-firmware-git 20260929.33b68e2c-1` |
| GPU | `0000:03:00.0`, `8086:56a0`, revision 08 |
| Board subsystem | `172f:3937` |
| xe GuC debugfs | `RUNNING`, wanted 70.53.0, found 70.53.0, compatibility 1.26.0 |
| xe HuC debugfs | firmware `(null)`, status `N/A`, wanted 0.0.0 |
| DMC debugfs | initialized and loaded, `i915/dg2_dmc_ver2_08.bin`, version 2.8 |
| firmware_class custom path | empty |
| `/usr/lib/firmware/updates` | absent |
| Board GSC firmware from `sudo -n igsc list-devices --info` | `DG02_1.3266` |
| Board OPROM CODE | `14 00 31 04 00 00 00 00` |
| Board OPROM DATA | `14 00 28 04 00 00 00 00` |

GuC SHA256, decompressed, 381760 bytes:

```text
e6e3f8b4480ba976c89c491ac736d6ce41fa82e0cf5726d0763142da3fe8a63b
```

Identical for all three:

- `/usr/lib/firmware/i915/dg2_guc_70.bin.zst`
- the GuC in `/boot/initramfs-linux73-tkg-bore.img`
- a fresh download of [upstream dg2_guc_70.bin](https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/i915/dg2_guc_70.bin)

DMC SHA256, decompressed:

```text
cac5204087bba70a81c53778846340e57a4e35e5959b7b42006969f6e5f45466
```

Disk and that initramfs match. Initramfs inspection used a temporary extraction
directory, removed on completion. Other installed kernel initramfs images were
not examined. Live debugfs confirms the loaded GuC version, not a cryptographic
hash of GPU memory. Kernel journal/dmesg queries did not return firmware lines;
debugfs supplied the live-state evidence.

## Driver compatibility and HuC

Both current upstream driver firmware tables select DG2 GuC 70.53.0 using the
`i915/dg2_guc_70.bin` name. The directory name does not imply xe uses a wrong
driver's firmware. The xe table deliberately names that same file. Its DG2 HuC
parser comment explicitly says DG2 HuC is not supported; the observed N/A state
therefore is not evidence of a missing linux-firmware package.
[xe firmware source](https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/xe/xe_uc_fw.c),
[i915 firmware source](https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/i915/gt/uc/intel_uc_fw.c)

These are upstream sources, not proof of every patch in this custom TKG kernel.
The live wanted/found versions independently agree. No source or experiment here
links HuC availability to the observed memcpy deadlock or BCS command fault.

## Where updates belong

1. Linux runtime blobs: use the distro's linux-firmware packaging, with
   [upstream linux-firmware](https://gitlab.com/kernel-firmware/linux-firmware)
   as the reference. This machine is already using a same-day git package and
   its relevant GuC equals upstream; reinstalling it would not change GuC.
2. If Intel publishes an explicitly DG2-compatible replacement, preserve the
   old package/blob and record hashes, use a managed package or a documented
   firmware override, rebuild the initramfs that actually boots, then verify
   the loaded version after reboot. Merely copying to the root filesystem
   does not replace a blob already packed into early boot or running on GPU.
3. The kernel searches `/lib/firmware/updates` before `/lib/firmware`, with a
   configurable firmware_class path taking priority. Overrides should be
   recorded because they can silently outrank subsequent package updates.
   [Kernel firmware search paths](https://docs.kernel.org/driver-api/firmware/fw_search_path.html)
4. [Intel's GPU firmware backport repository](https://github.com/intel-gpu/intel-gpu-firmware)
   is another primary source, but its README requires matching tags across its
   driver collection. The API listing inspected today showed DG2's explicitly
   versioned GuC files only through 70.44.1, plus the generic major-version
   filename. That listing supplies no evidence of a newer DG2 update than
   linux-firmware's 70.53.0. Do not transplant a different platform's blob or
   infer compatibility from the major number alone.

## Board firmware is a separate update track

GuC/HuC/DMC files loaded by Linux are distinct from persistent board GSC and
OPROM firmware. Intel's IGSC library accesses GSC through MEI and supports
GSC, OPROM data and OPROM code updates. Its documentation describes identity
and version checking, including subsystem vendor/device checks for OPROM.
That matters for this board's `172f:3937`: a reference Intel A770 image is not
automatically the appropriate board image.
[Intel IGSC introduction](https://raw.githubusercontent.com/intel/igsc/master/doc/introduction.rst)

Intel's public Arc support article, last reviewed December 2024, says its Linux
driver package does not update board firmware and directs users to Windows
for the firmware update. That is a supported distribution route, not a claim
that Linux flashing is technically impossible: IGSC demonstrates the latter.
[Intel Arc Linux firmware support](https://www.intel.com/content/www/us/en/support/articles/000096950/graphics.html)

No authoritative newer board image applicable to this exact A770 subsystem
was established in this investigation. The appropriate next source is the
board vendor's support/download channel or its supported Intel Windows driver
update path, comparing the offered image against `DG02_1.3266` and the OPROM
versions above. Do not describe unrelated Data Center Flex or Battlemage
firmware as an A770 upgrade. A board flash is not currently an evidence-backed
fix for the copy-path failure, and no flash was attempted.

## Confidence and limits

- High: installed/current-initramfs/upstream GuC equality; live GuC 70.53.0;
  DMC disk/initramfs equality and loaded 2.8; board version inventory.
- High: latest DG2 entries in upstream WHENCE at research time and upstream
  xe's documented DG2 HuC omission.
- Unknown: whether Intel has an unpublished or board-vendor-only newer DG2
  firmware; whether any firmware change fixes this workload; newest applicable
  board GSC/OPROM package for subsystem 172f:3937.
- No inference that absence of a newer firmware means investigation should
  stop: the kernel/runtime copy and synchronization paths remain actionable.
