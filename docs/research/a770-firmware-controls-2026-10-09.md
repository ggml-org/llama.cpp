# A770 firmware, controls, and benchmark provenance

Observed on 2026-10-09, starting from repository `6d13d0a79` on
`xe-kmd-copy-engine`. Target: general A770 performance and stability.

## Firmware outcome

Sparkle Arc A770 16 GB, PCI `0000:03:00.0`, device `8086:56a0`, subsystem
`172f:3937`, running xe. GSC firmware now repeatedly reports `DG02_1.3270`,
updated from `DG02_1.3266` with the installed IGSC 0.9.5 tool.

Source: [Intel graphics driver 32.0.101.9034](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html),
[installer download](https://downloadmirror.intel.com/929959/gfx_win_101.9034.exe).
The installer matched Intel's published SHA-512:

```text
3fc6f9b4c8ef40bf066f1ad77f3c6aea55d47bae6dc8ab9575a7a9fe8eee84bf0c79333f81825cd7538ea1bb7ca78b6ddb0b75f96b321b4dcf524dec843cf3f0
```

Extracted `Graphics/ifwi/acm/fwcode/dg2_gfx_fwupdate_SOC1.bin`, SHA-256:

```text
c32374f58b338370f42c6edf019ac7438bcc968884d10fab95e1263d01cf8544
```

IGSC accepted the SOC1 hardware compatibility check. No GPU/MEI device holders
were present before the update. Used normal version/compatibility checks,
without force or downgrade flags. The updater printed contradictory status:

```text
Error: Update process failed
Firmware status: Success (0x0)
Device: FW Version: DG02_1.3270
exit status: 0
```

The transfer progress last showed 44%, with an MEI firmware disconnect in the
kernel log. The error's precise cause is unresolved. Upstream IGSC CLI source
can replace an update error with the result of its subsequent version check;
exit zero alone is therefore insufficient evidence. Independent fresh device
queries returned 3270, Sysman enumeration worked, and a Level Zero SYCL kernel
produced 65,536 correct integer results. This establishes the observed version
and basic GPU execution, not a clean update log or long-term stability.

Unchanged: OPROM CODE `14 00 31 04 00 00 00 00`, OPROM DATA
`14 00 28 04 00 00 00 00`; firmware data format 1, major 101, OEM data 0, VCN 1.
The installer also contains newer generic OPROM code, but no matching
Sparkle A770 board-data payload was established; no option ROM or board data
was flashed. Running GuC remains 70.53.0. The installed DG2 GuC file matched
the current linux-firmware upstream download byte-for-byte.

Artifacts: `/mnt/mrgr/intel-gpu-artifacts/20261009/firmware/` and
`/mnt/mrgr/intel-gpu-artifacts/20261009/review/a770-*`.
Root-only update log and pre-update inventory are under
`/var/backups/a770-firmware-20261009/`. That directory does **not** contain a
readback/rollback image of the old firmware.

## Exposed controls

Paths below are relative to `/sys/bus/pci/devices/0000:03:00.0`.
Values are observations, not recommendations to raise limits. No tuning
settings were changed in this investigation.

| Control | Observed value / support | Interface |
| --- | --- | --- |
| Requested frequency range | 600-2400 MHz; hardware range 300-2400 MHz | `tile0/gt0/freq0/{min_freq,max_freq}`, Sysman |
| Power limit | 230 W; file writable, Sysman reports controllable | `hwmon/hwmon0/power2_max` (microwatts) |
| Power profile | `base` selected; `power_saving` offered | `power_profile` |
| Compute partitioning | `ccs_mode=1`, four compute slices | `tile0/gt0/{ccs_mode,num_cslices}` |
| Fan control | Two fans enumerated, both `canControl=0`; no PWM file found | Sysman / hwmon |
| Overclocking | `UNSUPPORTED_FEATURE` | Sysman |
| Performance factors | No handles exposed | Sysman |

The hwmon number can change after reboot. Writable files and capability flags
do not prove every proposed value is accepted. CCS changes need a separate
idle-device experiment and workload measurements. Existing long recovery
timeouts trade slower hang recovery for tolerance; do not increase them as a
performance optimization.

The physical upstream PCIe link reports 16 GT/s x16 and ReBAR is 16 GB.
The DG2 internal endpoint's Gen1/x1 report is not the physical host-link speed.

## Software and codebase findings

The saved container runtimes are older than this host's oneAPI 2026.1 / IGC
2.41.10 stack. They were not installed over Arch packages. Current
[SGL Kernel XPU CMake](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/CMakeLists.txt)
accepts BMG/CRI targets, so its binaries are not established DG2 replacements.
Its source may inform experiments, but no kernel was ported or benchmarked.

[Device Management Toolkit](https://github.com/device-management-toolkit)
and its sample web UI manage Intel AMT/vPro platforms. They do not provide Arc
GPU tuning or firmware upgrades for this Ryzen host. IGSC handles GSC firmware;
xe sysfs and Level Zero Sysman expose the controls above.

Fixed two omissions in `scripts/bench-a770-fork-unique.py`:

- Preserve `MKL` and `ONEDNN` attention-route records already emitted by
  `fattn.cpp`, rather than silently dropping them.
- Record installed `-git` variants of compute-runtime, IGC and Level Zero
  packages, alongside stable package names. Preserve command failure metadata.

Regression tests passed: `python -m unittest scripts.test_bench_a770_fork_unique`
(37 tests). An actual package-query smoke check captured the host's git runtime
and loader packages. No inference kernel or default tuning policy changed.

Unrestricted `sycl-ls` failed with an LLVM duplicate
`PointerFlowAnalysisResult` registration error. Setting
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` succeeded, as did the compute smoke test
with copy engines disabled. The harness already defaults to Level Zero.
The unrestricted enumeration failure was not diagnosed or repaired here.

## Not claimed

No reboot/power-cycle verification, model inference soak, matched before/after
throughput benchmark, fan/clock/power-control write test, or firmware readback
comparison. No evidence yet that 3270 improves llama.cpp speed or eliminates
the previously documented copy-engine failure. Keep controlled placement,
runtime, and attention-route provenance in the next benchmark campaign.
