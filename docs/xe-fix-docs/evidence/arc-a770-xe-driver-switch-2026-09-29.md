# Arc A770: i915 -> xe driver switch (2026-09-29)

**Host**: vinbonesjr (Arch, `7.3.0-rc1-273-tkg-bore`, ZFSBootMenu)
**Card**: Intel Arc A770 16 GB, DG2, PCI `0000:03:00.0` (`8086:56a0`, subsys `172F:3937`)
**Driver**: i915 (booted) -> **xe (staged for next boot)**. No live switch was performed - see "Why not live".
**Goal**: xe KMD for llama.cpp SYCL (`llama.cpp-sycl-f16-git b12305`, `intel-compute-runtime 26.35`, `level-zero-loader 1.32`).

## TL;DR

1. `reboot`. Default ZBM entry now boots xe on the A770; i915 is blocked for that PCI ID only.
2. Run the verification block below. Expect `xe-a770-tune.service` active and `ccs` preempt 7.5 s / job timeout 60 s.
3. Re-run the i915 baseline bench (command below). i915 numbers: **pp512 1408 t/s, pp2048 849 t/s, tg128 60.2 t/s**.
4. If the A770 must also do VAAPI/QSV *encode* (AV1 batch transcodes), test `vainfo` first: xe on this kernel ships **no DG2 HuC**.

## Facts established before touching anything (2026-09-29 10:2x-10:4x GMT)

- **Headless compute card.** Both monitors hang off the Raphael iGPU (`card1`, amdgpu, `HDMI-A-5 connected`). `card0` (A770) has no connected connector. A bad GPU driver can never cost the desktop.
- **kwin holds it anyway.** `fuser` on `/dev/dri/card0` + `renderD128`: `systemd`, `systemd-logind`, `Xorg`, `kwin_wayland` (fds 33-38, 44-45, 59, 65-69), `Xwayland`, `electron`, `codex-app-linux`, `codex`, `llama-server`. Chromium-based apps render on the A770 (first render node) and copy to the iGPU. **Why not live**: unbinding i915 with kwin holding `card0` risks the compositor; not acceptable unattended.
- **PCIe is fine.** `00:01.1 -> 01:00.0` (DG2 on-board switch upstream) `LnkSta: 16GT/s x16`. The endpoint `03:00.0` reports `2.5GT/s x1` in both LnkCap and LnkSta - that is the switch's internal virtual link, not a bottleneck. ReBAR: `BAR 2: current size: 16GB`. ASPM L1 enabled on every hop, policy `default`.
- **Kernel xe config** (`/proc/config.gz`): `CONFIG_DRM_XE=m`, `DISPLAY=y`, `GPUSVM=y`, `FORCE_PROBE=""`, no `DRM_XE_DEBUG*`, `JOB_TIMEOUT_MAX=10000`, `PREEMPT_TIMEOUT=640000`, `PREEMPT_TIMEOUT_MAX=10000000`, `ENABLE_SCHEDTIMEOUT_LIMIT=y`.
- **xe firmware for DG2** (`strings xe.ko`): `i915/dg2_guc_70.bin`, `i915/dg2_dmc_ver2_08.bin`. **No `dg2_huc_gsc.bin`.** Under i915 right now: `HuC firmware: i915/dg2_huc_gsc.bin status: RUNNING version 7.10.16`, GuC `70.53.0`.
- **i915 as actually booted** (not what the old modprobe comment claimed): `enable_guc=-1` (auto), `mitigations=auto`, `enable_dc=-1`, `request_timeout_ms=20000`. `/proc/cmdline` never carried `i915.enable_guc=3 i915.mitigations=off i915.enable_dc=0`.
- **i915 engine scheduler** (`/sys/class/drm/card0/engine/*`): `ccs0`/`rcs0` `preempt_timeout_ms=7500`, others 640; `timeslice 1 ms`, `heartbeat 2500 ms`. Freq 300-2400 MHz (RP1 600), idle ~800. hwmon `power1_max=230 W`.
- **i915 is not spotless.** This boot, 2026-09-27 22:09-22:10 GMT: seven `Fence expiration time out i915-0000:03:00.0:test-backend-op[3758008:...]` - the 20 s request timeout fired during `test-backend-ops`.
- **June evidence is gone.** Journal holds two boots (from 2026-09-22). The only record of the 2026-06-27 xe failure is the modprobe comment: "DeviceLost / CAT engine-reset / slow prefill".
- **Compute stack sees the card today**: `sycl-ls` -> `[level_zero:gpu] ... Arc(TM) A770 Graphics 12.55.8 [1.17.39758]`, `[opencl:gpu] ... NEO [26.35.39758]`. `libze_intel_gpu.so.1` contains `drm_xe` symbols (xe KMD support present). Mesa ANV `26.3.0-devel` handles both KMDs.
- **Unrelated but relevant to post-reboot triage**: `llama-gpu@Ornith-1.5-35B-Q4_K_M` restarted 837x on 2026-09-28, each `status=1/FAILURE` ~0.7 s after start (startup failure, not a GPU fault). Do not pin a repeat of that on xe.

## Baseline (i915, before the switch)

Service stopped 10:37 GMT, bench, service restarted (healthy after ~96 s).

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-bench -m /mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf -p 512,2048 -n 128 -r 2 -fa 1 -ngl 99 -o md
```

| model | backend | fa | test | t/s |
|---|---|--:|---|--:|
| llama 8B Q4_K_M 4.58 GiB | SYCL | 1 | pp512 | 1408.08 +/- 2.41 |
| llama 8B Q4_K_M | SYCL | 1 | pp2048 | 848.93 +/- 0.57 |
| llama 8B Q4_K_M | SYCL | 1 | tg128 | 60.15 +/- 0.07 |

build `4e7400c3a (12305)`, `GGML_SYCL_GRAPH: yes`, `GGML_SYCL_F16: yes`. Ornith production numbers on i915 live in `~/projects/local-models/experiments/results.md` (q8/q8 FA @128k tg128 20.7 t/s, 2026-09-27).

## What changed (all on disk, nothing live)

| File | Change |
|---|---|
| `zroot/ROOT` `org.zfsbootmenu:commandline` | `i915.force_probe=56a0` -> `i915.force_probe=!56a0` (kept `xe.force_probe=56a0`). Children `arch`/`default` inherit. |
| `/etc/modprobe.d/intel-xe.conf` | `blacklist xe` removed. Now comments only (selection lives on the cmdline; optional levers listed, all off). Backup `intel-xe.conf.bak-20260929-104144`. |
| `/etc/mkinitcpio.conf` | `MODULES=(i915 amdgpu zfs)` -> `MODULES=(xe amdgpu zfs)`. Backup `mkinitcpio.conf.bak-20260929-104144`. |
| `mkinitcpio -P` | All four images rebuilt 10:42 GMT. `initramfs-linux73-tkg-bore.img` contains `xe.ko`, `amdgpu.ko`, `i915/dg2_guc_70.bin`, `dg2_dmc_ver2_08.bin`, `intel-xe.conf` with 0 blacklist lines, **no `i915.ko`**. |
| `/usr/local/bin/xe-a770-tune` | Sets `ccs`+`rcs` `preempt_timeout_us=7500000`; `ccs` `job_timeout_max` then `job_timeout_ms=60000`. Exits 0 with a message when the card is not on xe. |
| `/etc/systemd/system/xe-a770-tune.service` | oneshot, `ConditionPathIsDirectory=/sys/bus/pci/devices/0000:03:00.0/tile0`, enabled. Under i915 today: `start condition unmet` (verified). |
| `/etc/udev/rules.d/80-xe-a770-tune.rules` | `ACTION=="add|bind", SUBSYSTEM=="pci", KERNEL=="0000:03:00.0", DRIVER=="xe"` -> `SYSTEMD_WANTS+=xe-a770-tune.service`. `udevadm verify` passed. |
| `/etc/systemd/system/llama-gpu@.service.d/10-xe-tune.conf` | `Wants=`/`After=xe-a770-tune.service` - xe sysfs defaults only bind to exec queues created *after* the write. |

Why cmdline for selection: xe's own probe message says it - `use xe.force_probe='%04x' and i915.force_probe='!%04x'`. Both drivers carry `56a0` in their ID tables; the `!` makes the outcome independent of module load order and editable from the ZBM prompt.

## Why these timeouts

| Property | i915 today | xe default (this kernel) | xe after tune |
|---|--:|--:|--:|
| ccs preempt timeout | 7500 ms | 640 ms | 7500 ms |
| rcs preempt timeout | 7500 ms | 640 ms | 7500 ms |
| ccs job timeout | 20000 ms (`request_timeout_ms`) | 5000 ms (max 10000) | 60000 ms (max raised to 60000) |

A compute context that cannot yield within the preempt timeout gets an engine reset -> context banned -> `ZE_RESULT_ERROR_DEVICE_LOST` in the process. That is the June failure shape. 640 ms is an interactive-desktop default; on i915 the card has run for months at 7.5 s. Job timeout: 20 s already fired on i915 during `test-backend-ops` two days ago, so 60 s. Kernel clamps: `job_timeout_ms` must be <= `job_timeout_max`, which is why the script writes `_max` first. If the kernel refuses the raise (see "Not claimed"), the script logs the min/max it saw and continues at the 10 s ceiling.

## After reboot - verification

```bash
# driver + probe
lspci -ks 03:00.0 | grep driver            # expect: xe
journalctl -k -b -g 'xe 0000:03:00.0' | grep -iE 'not officially|GuC|HuC|firmware|error|fail|wedged' | head
ls /sys/class/drm/card0/device/tile0/gt0/engines/

# tuning applied?
systemctl status xe-a770-tune.service --no-pager
for c in ccs rcs; do d=/sys/bus/pci/devices/0000:03:00.0/tile0/gt0/engines/$c; \
  echo "$c preempt=$(cat $d/preempt_timeout_us) job=$(cat $d/job_timeout_ms) job_max=$(cat $d/job_timeout_max) defaults=$(cat $d/.defaults/preempt_timeout_us)/$(cat $d/.defaults/job_timeout_ms)"; done

# compute stack
ONEAPI_DEVICE_SELECTOR='level_zero:*;opencl:gpu' /opt/intel/oneapi/compiler/latest/bin/sycl-ls
systemctl status llama-gpu@Ornith-1.5-35B-Q4_K_M --no-pager; curl -s 127.0.0.1:8089/health

# A/B against the i915 numbers above (stop the Ornith unit first, VRAM is full otherwise)
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-bench -m /mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf -p 512,2048 -n 128 -r 2 -fa 1 -ngl 99 -o md

# encode still there?
vainfo --display drm --device /dev/dri/by-path/pci-0000:03:00.0-render | grep -E 'EncSlice'
cat /sys/kernel/debug/dri/0/gt0/uc/huc_info 2>/dev/null   # (root) may not exist under xe
```

Freq/power under xe live at `/sys/class/drm/card0/device/tile0/gt0/freq0/{min,max,cur,act}_freq` and `/sys/class/drm/card0/device/hwmon/hwmon*/power1_max`.

## Rollback

- **One boot**: at the ZBM prompt edit the cmdline: `i915.force_probe=56a0 xe.force_probe=!56a0`. i915 is not in the initramfs any more, but the card is headless so udev loads it from the root fs. `xe-a770-tune` sees i915 and no-ops.
- **Permanent**: `zfs set org.zfsbootmenu:commandline='...i915.force_probe=56a0 xe.force_probe=!56a0...' zroot/ROOT`, `MODULES=(i915 amdgpu zfs)`, `mkinitcpio -P`. Backups of both files carry the `-20260929-104144` suffix.

## Levers looked at and left alone

- `xe.probe_display=0`: saves display init on a headless card, but the DG2 HDA function (`04:00.0`, `snd_hda_intel`) binds through the display audio component; off would leave it probe-deferring. Not worth the noise.
- `xe.wedged_mode=0`: keep the upstream default (1 = wedge on critical error). A wedge needs a rebind/reboot; with kwin on `card0` that is a reboot either way, so mode 0 would only hide the evidence.
- `xe.guc_log_level=0`: default 1 costs nothing measurable and is the only forensic trail if CAT errors return.
- GT `min_freq` floor / `power1_max` raise / per-hop ASPM (`/sys/bus/pci/devices/0000:03:00.0/link/l1_aspm`): no evidence any of them was in play; measure first.
- `KWIN_DRM_DEVICES=/dev/dri/card1`: would stop kwin from opening the A770 and make future driver swaps live-safe. Session-level change, separate decision.
- ZBM prepends its own `amd_pstate=guided amd_iommu=on iommu=pt split_lock_detect=off nowatchdog`, so `/proc/cmdline` has duplicates (`amd_pstate=active` wins, last write). Cosmetic; lives in the ZBM EFI config, not in ZFS.

## Not claimed

- **xe was not exercised on this card in this session.** No probe, no GuC load, no bench under xe. Everything above is staged config plus reasoning from kernel config and driver strings. First real evidence arrives after reboot.
- **HuC / encode**: verified only that `xe.ko` has no DG2 HuC firmware string and that i915 currently runs HuC 7.10.16. Whether `iHD` still exposes encode entrypoints on xe without HuC was not tested (`vainfo` on the busy device failed to initialise here). If the A770 must keep doing AV1/HEVC encode, that decides i915 vs xe, not llama.cpp.
- **`job_timeout_max` write**: the script assumes the sysfs `_max` store accepts values above the Kconfig ceiling. Not verified against 7.3 source; the script reports what the kernel actually accepted.
- **`xe.force_probe=56a0` may be unnecessary** on 7.3 (DG2 may have left the force-probe list). Harmless either way; the journal line after reboot tells.
- **Engine sysfs paths** (`tile0/gt0/engines/{ccs,rcs}`) are from memory of the xe layout, guarded by globs in the script. If they differ, the service fails loudly and llama-server still starts (`Wants=`, not `Requires=`).
- **Baseline bench** is one model, two repetitions, 8B Q4_K_M with FA + SYCL graph. It brackets the driver, not the Ornith production profile.
- **`linux-lts` initramfs** rebuilt with `ERROR: module not found: zfs` - pre-existing (no zfs module for 6.18.53), not caused by this change, and that image already could not boot zroot.
- **June root cause** remains unknown; the timeout parity is the best available match to the symptom, not a proven fix.

---

## Post-reboot results (booted 11:00:46 GMT, checked 11:08-11:20)

**Driver**: `Kernel driver in use: xe`. `i915` loaded with 0 users (udev alias), harmless. `snd_hda_intel` bound to `04:00.0` via xe's audio component.
**Firmware** (`/sys/kernel/debug/dri/0/gt0/uc/`): GuC `i915/dg2_guc_70.bin` RUNNING, release `70.53.0`, compat `1.26.0`. HuC `(null) / N/A`. GSC `N/A`. As predicted.
**Engines**: sysfs classes `bcs ccs rcs vcs vecs`; debugfs `hw_engines`: `rcs0 bcs0 ccs0 ccs1 ccs2 ccs3`. `tile0/gt0/ccs_mode = 1` (root 0644), `num_cslices` present.
**Live params**: `force_probe=56a0`, `probe_display=Y`, `wedged_mode=1`, `guc_log_level=1`, `vram_bar_size=0`.
**Freq**: `min 750 / max 2400 / rp0 2400 / rpe 750 / rpn 300`. xe floors at RPe (750); i915 floored at RPn (300). Part of the pp gain below may be ramp behaviour, not driver efficiency.
**hwmon** (`name=xe`): `power2_label=pkg power2_max=230 W` (same cap as i915), `temp2 pkg 50  degC`, `temp3 vram 62  degC`, `fan1_input 63`, `fan2 0`. i915 exposed no temps/fans.

### Tune unit: first run failed, fixed

`xe-a770-tune.service` exited 1 at 11:00:46: `job_timeout_max=60000` rejected. The sysfs `_max` store honours `CONFIG_DRM_XE_JOB_TIMEOUT_MAX=10000` from the kernel config, so 60 s is unreachable without a kernel rebuild. Preempt writes had landed. Script rewritten 11:14 (backup `xe-a770-tune.bak-...`): raise attempt, then clamp to the kernel max, exit 0. Now:

| engine | preempt_timeout_us | job_timeout_ms | job_timeout_max |
|---|--:|--:|--:|
| ccs | 7500000 | 10000 | 10000 |
| rcs | 7500000 | 5000 (default) | 10000 |

The running llama-server was created with `job=5000`; it picks up 10000 on its next restart. If `test-backend-ops` jobs need >10 s: rebuild the tkg kernel with `CONFIG_DRM_XE_JOB_TIMEOUT_MAX=60000` (i915 gave 20 s via `request_timeout_ms`).

### A/B bench (same command as the i915 baseline)

| test | i915 (server stopped) | xe (server running, 11:09) | xe (server stopped, 11:46) | delta clean |
|---|--:|--:|--:|--:|
| pp512 | 1408.08 +/- 2.41 | 1494.10 +/- 2.23 | 1495.56 +/- 0.49 | +6.2 % |
| pp2048 | 848.93 +/- 0.57 | 988.93 +/- 0.23 | 990.40 +/- 1.28 | +16.7 % |
| tg128 | 60.15 +/- 0.07 | 60.53 +/- 0.14 | 60.58 +/- 0.01 | +0.7 % |

Clean re-run 11:46 GMT with the Ornith unit stopped (only `kwin_wayland` on `renderD128`): within noise of the contended run, so an idle resident server costs nothing measurable. "slow prefill" from June is not reproduced on 7.3 + NEO 26.35. Ornith restarted 11:47, healthy after ~42 s; its exec queues now carry `job_timeout_ms=10000`.

### Stack: unchanged on top

`sycl-ls`: `[level_zero:gpu] ... Arc(TM) A770 Graphics 12.55.8 [1.17.39758]` and `[opencl:gpu] ... NEO [26.35.39758]`, identical to i915. llama-server loaded Ornith in ~57 s, `{"status":"ok"}`. Mesa prints `Support for this platform is experimental with Xe KMD` once per process (ANV/iris on DG2+xe), cosmetic.

### Encode under xe: CQP yes, bitrate control no

`vainfo` on the A770 failed on *both* drivers because `/etc/environment:23` sets `LIBVA_DRIVER_NAME=radeonsi` globally, forcing the AMD backend onto the Intel node. With the override unset, libva auto-picks correctly (`03:00.0` -> iHD 26.2.4, `0f:00.0` -> radeonsi). Nothing in `/etc/systemd/system`, `/etc/profile.d`, `/etc/environment.d` or the user shell/Plasma env files references it. **Removed 11:48 GMT** (the line had been commented out at 11:45; the commented remnant deleted too), backup `/etc/environment.bak-20260929-114756`. `/etc/environment` is read by pam_env at login, so the running session still exports `LIBVA_DRIVER_NAME=radeonsi` until re-login; systemd system services never saw it. `VDPAU_DRIVER=va_gl` and `MOZ_DRM_DEVICE=/dev/dri/renderD129` (Firefox pinned to the iGPU) stay.

With `LIBVA_DRIVER_NAME=iHD` forced, 18 encode entrypoints are advertised (H.264/HEVC/VP9/AV1 `EncSliceLP`). Real test, 90 frames 1080p `testsrc2`, 11:15 GMT:

| encoder | rc mode | result |
|---|---|---|
| h264_vaapi | CQP qp 24 | OK, 0.4 s, 2.96 MB |
| h264_vaapi | VBR 6M | FAIL `return code -5 (Input/output error)` |
| hevc_vaapi | CQP qp 24 | OK |
| av1_vaapi | CQP qp 60 | OK, 0.3 s, 5.97 MB |
| av1_vaapi | VBR 4M | FAIL, same error |
| av1_vaapi | CBR 4M | FAIL, same error |

Bitrate control (BRC) runs on the HuC; no HuC under xe, so VBR/CBR die at submit. CQP works for all three codecs. Batch archival transcodes at fixed quality are fine on xe; anything that targets a bitrate (Plex/Jellyfin session transcodes, `-b:v`) needs i915 or the iGPU.

### journald was not recording system entries this boot

Until 11:17 GMT: `journalctl -k -b` = 0 lines, `_PID=1` = 0, `_UID=0` = 0, a root `logger` probe never landed; only UID-1000 entries (llama-server via `SplitMode=uid`) were written. `system.journal` mtime frozen at 10:58 (previous boot), `journalctl --verify` PASS, disk fine (`zroot/var/log` 7.4 G free, journal 199 M of 500 M). `systemctl restart systemd-journald` fixed it immediately (kernel lines flowing, probe recorded). Root cause unknown; `dmesg` had also lost the first 27 s of boot to 624 lines of `amdgpu 0000:0f:00.0: [drm] *ERROR* Unsupported screen format RA24` spam. Net: **no kernel-side record of the first xe probe exists.** Firmware state above comes from debugfs instead. Watch `journalctl -k -b | head -1` on the next boot.

### Multi-CCS on this stack (asked 11:17)

Hardware: 4 CCS present (`ccs0-3`), 1 exposed (`ccs_mode=1`), matching Intel's `MULTI_CCS_MODES.md` default. Upstream doc configures it through `/sys/class/drm/cardX/gt/gtY/ccs_mode` - the **i915** layout; xe puts it at `/sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode`. NEO 26.35 (`libze_intel_gpu.so.1`) composes the ccs_mode path from `/sys/class/drm/` + `/gt/gt` + `/ccs_mode` only, so `ZEX_NUMBER_OF_CCS` has no file to write under xe. A root write to the xe path works in principle but xe returns `EBUSY` while any DRM client holds the device (kwin, Xorg, Xwayland, logind, llama-server all do) and then GT-resets. Practically unavailable on this workstation without evicting the desktop from `card0`.

Per-client engine accounting under xe (`/proc/<llama-server>/fdinfo/3`): `drm-cycles-bcs 14 003 849`, `drm-cycles-ccs 8 032 230`, `rcs 0`. The UR L0 adapter already routes copies to the blitter (`bcs0`), which runs concurrently with `ccs0` in hardware. `intel_gpu_top -d pci:slot=0000:03:00.0` shows it live.

## Not claimed (post-reboot addendum)

- Bench delta is one model (8B Q4_K_M), two reps per test, FA on, SYCL graph on. It brackets the driver, not the Ornith production profile.
- Encode test used a synthetic source and 90 frames; QSV (`*_qsv`) paths untested. Only VAAPI via iHD.
- journald root cause not found; the restart is a workaround, recurrence unknown.
- Multi-CCS conclusion rests on strings in `libze_intel_gpu.so.1`, not on NEO source; a newer NEO may add the xe path.

---

## Ornith production model on xe (11:56-12:46 GMT): verdict reversed

Offline `llama-bench` on Ornith-1.5-35B-A3B Q4_K_M with the production flags
(`-fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -p 512 -n 64 -d 0,8192 -r 5 -t 12`),
same package N `4e7400c3a`, compared with the 09-28 i915 rounds. Full table, raw
files, both coredumps, server-crash evidence and diagnostics:
`~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/` (README + `raw/`).

| | i915 (09-28 r2) | xe r1 | xe r2 | xe copy-engine off |
|---|--:|--:|--:|--:|
| pp512 | 125.3 | 197.9 | 202.9 | 202.4 |
| tg64 | 18.13 | 13.85 | 13.38 | 14.50 |
| pp512 @8k | 107.1 | 167.4 | 170.2 | 167.7 |
| tg64 @8k | 16.81 | 12.50 | **hang** | 13.41 |

- Prefill +55-62 %, decode -20 to -29 % (+14 to +22 ms per token), both consistent
  over five runs. Expert placement identical on both drivers, which rules out a
  placement difference but not different hit counts, bytes or sync time. The
  fork's decode path syncs, reads routing IDs back, syncs again, then uploads
  expert ranges via `queue::memcpy` per token; prefill batches those
  boundaries. The 8B dense model (all on device) showed no decode change.
- **3 of 5 xe runs failed in the 8k-depth tests**: two silent hangs (host spinning
  in the L0 adapter's `enqueueMemCopyHelper`, ccs context parked on a device-side
  wait, blitter idle, no kernel messages, clean teardown on SIGTERM) and one
  blitter `Engine reset ... Timedout job` with an xe device coredump (`RING_ESR=0x1`,
  `IPEHR=0xfffff000`, context runtime 0 ms) followed by
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` and SIGABRT. Same failure family as the
  June note (DeviceLost / engine reset), now with artifacts. The device-side-wait
  reading of the hangs and the "who corrupted the bcs stream" question are open;
  see the README's cause ranking. i915 ran the same tests clean in every round.
- Levers: `UR_L0_USE_COPY_ENGINE=0` finished and gained +5-7 % decode (one pass);
  `GGML_SYCL_ENABLE_HOST_PINNED_MEM=0` produced the bcs reset; `GGML_SYCL_ENABLE_GRAPH=0`
  hung in pp512@8k.

**Recommendation**: production Ornith stays on i915. The prefill gain does not
pay for a -25 % decode and a hang rate of 3/5 in the long-context regime the
unit actually serves. Everything staged today stays valid for a later retry
(newer NEO / UR / kernel, or a fork change to the copy path); `xe-a770-tune`
no-ops under i915.

Revert (one boot: edit at the ZBM prompt; permanent: the two commands):

```bash
sudo zfs set org.zfsbootmenu:commandline="$(zfs get -H -o value org.zfsbootmenu:commandline zroot/ROOT | sed 's/i915.force_probe=!56a0 xe.force_probe=56a0/i915.force_probe=56a0 xe.force_probe=!56a0/')" zroot/ROOT
sudo sed -i 's/^MODULES=(xe amdgpu zfs)$/MODULES=(i915 amdgpu zfs)/' /etc/mkinitcpio.conf && sudo mkinitcpio -P
```

The earlier "+16.7 % prefill, decode equal" from the 8B dense bench stands; it is
just not the workload this card serves.


### 13:20 GMT: production server crashed on xe, default config

Second real request through `llama-gpu@Ornith` (542-token prompt, 256 tokens) died with
the pinned-off signature: kernel `Engine reset: engine_class=bcs ... Timedout job:
seqno=7811 ... in llama-server`, then `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` in
`ggml_backend_sycl_set_tensor_async`, SIGABRT, systemd restart (evidence in
`research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/raw/N-xe-server/`, coredump
included). Request 1 had run fine (pp 163.5, tg 38.0 t/s with ngram speculation).
Also established: `--moe-cache` is off by default and pinned off in the unit; the
transfer path is `ggml_backend_sched` shuttling activations to CPU-resident expert
layers, not the streaming cache. 4 failures in 7 long-context exercises on xe today.
The i915 recommendation stands with more force.


### 16:37-17:32 GMT: fix found, verdict reversed again

Pushback taken: three single-pass levers were not a fix attempt. Worked the ladder:
`bcs` job timeout to the 10 s cap; `UR_L0_USE_COPY_ENGINE=0` soaked (4/4 Ornith bench
rounds clean), deployed as `llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf`,
then 6/6 real 7k-14k-token requests through the unit clean. Default path meanwhile:
9 failures in 12 (incl. two production crashes at 13:15 and 13:20, three xe device
coredumps, all `bcs Timedout job`). Five submission-model variants did not rescue the
default path; a `UR_L0_DEBUG` trace shows the copy-engine memcpy chain (immediate list
ordinal 1, event-chained) simply stops completing.

**Production stays on xe** with the drop-in: prefill +55-62 %, decode -20 % on the
random-token bench. i915 revert remains the documented fallback
(`journalctl -k -g "Timedout job"` is the tripwire). Full record:
`~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/README.md`.


### 18:55 GMT: deep-research pass and a correction

`/deep-research` (111 agents, verified claims only) confirmed: both drivers load the same
GuC 70.53.0; NEO puts every xe queue on an LR-mode VM with user fences and no kernel-side
ordering between execs, so ccs<->bcs ordering is the runtime's job (why the copy-engine
lever works); LR queues have no job watchdog, so the "Timedout job" lines were GuC engine
resets surfaced through the TDR error path, not timer expiries (the bcs `job_timeout_ms`
change was moot); HuC on DG2 under xe is unsupported with only a stalled out-of-tree
patch; GuC 70.74.0 for DG2 exists only in drm-firmware `intel-staging` (drop-in file,
unsupported by distros). No upstream report matches our blitter signature; nearest is
compute-runtime #999 on Battlemage. Frequency-floor explanations for the performance
deltas were refuted. Report: `research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/deep-research-xe-i915-2026-09-29.md`.

## Not claimed (Ornith addendum)

- One pass per lever; the hang is intermittent, so "copy engine off finished" is
  not "copy engine off fixes it".
- No UR/L0 debug trace was captured before killing r2; the cross-engine reading is
  inferred from `fdinfo` engine counters, the backtrace and the bcs coredump.
- i915 reference is from 09-28, one day earlier, same package; not re-run today.
