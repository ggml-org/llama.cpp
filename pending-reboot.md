# Keep the desktop off the Arc A770 (two reboots observed, 2026-10-06)

Status, 2026-10-06 ~02:25: all three changes are installed and active, and the post-reboot check
passed on the second boot (02:17). Details in "Observed after reboot".

- First boot (01:30): Xorg, KWin, logind and Xwayland no longer held the A770, but
  **`sddm-greeter` still held `renderD128`**, which the original plan missed. Change 3
  (`71-arc-render-restricted.rules`) was added for it.
- Second boot (02:17): the Arc's client list was empty, `fuser` found no holders, and a runtime
  `ccs_mode` write succeeded where it used to return `EBUSY`. `llama-gpu@Ornith-1.5-35B-A3B-uncensored-Q8_0`
  then started under the new `0660` mode and became the only client.
- The last recorded `ccs_mode` readback was **1** (the production default), followed by the
  service start. Leaving mode 2 active with the service stopped was requested, but no later
  stop, write or readback is recorded here. No workload was measured in mode 2.

This file can be deleted once the questions in #92 that depend on it are closed; the host
configuration it describes does not depend on the file.

## Why this file exists

The goal of this change is that something stops happening. After the reboot, success is an absence:
nothing from the desktop holds the A770. An absence is easy to misread:

- On this host an unprivileged `fuser /dev/dri/renderD128` prints nothing **even while the
  compositor holds the card**. `kwin_wayland` has the file capability `cap_sys_nice=ep`, which makes
  its `/proc/<pid>/fd` unreadable to the same user, so a plain `fuser` looks clean before and after.
- A clean result says nothing about whether anything was gained unless you know what the holders
  were blocking.

This file records what was changed, what each change should stop, what that unlocks, and how to
tell whether it worked. Background and the wider list of work items are in issue #92.

## The problem (measured 2026-10-05)

The A770 (`0000:03:00.0`, `xe`, `card0` / `renderD128`) drives no display: all of its connectors
are disconnected. The only connected output is on the Radeon 610M (`0000:0f:00.0`, `amdgpu`,
`card1` / `renderD129`). Even so, the kernel's client list for the card
(`/sys/kernel/debug/dri/0/clients`) and fdinfo showed the desktop holding it:

| Holder | Node | Why it was open | Work it put on the A770 |
| --- | --- | --- | --- |
| `Xorg` (SDDM's X11 server, root, stays running on vt2 behind the Wayland session; a different process from `sddm-greeter`, see Change 3) | `card0` | Xorg's default `AutoAddGPU` made the Arc a "GPU screen": `modeset(G0): using drv /dev/dri/card0` in `/var/log/Xorg.0.log` | 380 KiB VRAM, glamor; `drm-cycles-rcs` 1482, `ccs` 567 |
| `systemd-logind` (PID 1 keeps a duplicate in logind's fd store) | `card0`, DRM master | It opened the card for KWin through `TakeDevice()` | none |
| `kwin_wayland` | `card0` twice (logind's file, plus its own open) and 5 render-node clients on `renderD128` | KWin's display backend adds every card on the seat. Its render-device manager opens every render node. Its Vulkan instance loads the Intel driver (`libvulkan_intel.so`, mapped in the process), which opens `renderD128` while listing devices | 616 KiB VRAM; `rcs` 4207, `ccs` 382 |
| `Xwayland` | duplicate of KWin's file | Inherited | none |

Why that matters:

1. **`ccs_mode` cannot be changed.** xe rejects a write to
   `/sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode` with `EBUSY` while any DRM file is open on the
   device. That includes writing the value it already holds, and the rejection is logged at debug
   level only (`xe_gt_ccs_mode.c`, `ccs_mode_store()`). With the desktop holding the card, the only
   ways to change the mode were writing it at boot before SDDM starts, or stopping the display
   manager.
2. **No process can be asked to let go.** DRM has no revoke: only the holder can close its file.
   logind's `PauseDevice` drops DRM master but keeps the file open, and KWin closes its card handle
   only on a udev `remove` event.
3. **Sole tenancy is never true.** The desktop's files sit on the card for the whole session, and
   the repo's tenancy gates (unprivileged `fuser`, `renderD128` only) cannot see them. See #92.
4. **The desktop is coupled to the compute GPU.** Xorg's glamor context and KWin's EGL and Vulkan
   devices lived on the A770, so the desktop had state on the GPU that compute jobs hang and reset.

## Change 1: KWin

Two files.

`/etc/udev/rules.d/70-desktop-gpu.rules`: stable names for the Radeon. Card numbers can change
between boots, and `/dev/dri/by-path` names contain `:`, which KWin uses to separate list entries.

```
# Display GPU (Radeon 610M): stable names for KWIN_DRM_DEVICES / KWIN_RENDER_NODES
SUBSYSTEM=="drm", KERNELS=="0000:0f:00.0", KERNEL=="card[0-9]*", SYMLINK+="dri/desktop-card"
SUBSYSTEM=="drm", KERNELS=="0000:0f:00.0", KERNEL=="renderD*",   SYMLINK+="dri/desktop-render"
```

`~/.config/systemd/user/plasma-kwin_wayland.service.d/10-desktop-gpu.conf`: KWin runs as this
user unit, so the variables reach only KWin and its children (Xwayland). Other session apps are
not affected.

```
[Service]
# Keep KWin off the Arc A770 (compute only)
Environment=KWIN_DRM_DEVICES=/dev/dri/desktop-card
Environment=KWIN_RENDER_NODES=/dev/dri/desktop-render
# Not VK_DRIVER_FILES: kwin_wayland has cap_sys_nice=ep, so it runs with AT_SECURE=1 and
# the Vulkan loader reads all of its variables through secure_getenv() and ignores them.
# KWin reads this one itself.
Environment=KWIN_DISABLE_VULKAN=1
```

Each variable closes one of KWin's three ways onto the card (read from KWin master source and
checked against the installed `libkwin.so.6`, 6.7.91):

| KWin path onto the A770 | Without the variable | Variable |
| --- | --- | --- |
| Display backend (`backends/drm/drm_backend.cpp`) | Adds every card whose udev seat is the session's seat; a card with no outputs is not excluded | `KWIN_DRM_DEVICES`: open only the listed cards. Symlinks are resolved, so `desktop-card` works |
| Render-device manager (`core/gpumanager.cpp`) | Opens every render node udev lists, with no seat filter | `KWIN_RENDER_NODES`: open only the listed render nodes |
| Vulkan (`core/renderdevice.cpp`) | KWin creates one Vulkan instance per render device, even for the Radeon alone. Each instance loads every installed driver, and the Intel one opens `renderD128` while listing devices | `KWIN_DISABLE_VULKAN=1`: no Vulkan instance |

The Vulkan row first used `VK_DRIVER_FILES` (the Radeon driver only). That cannot work for KWin:

- `kwin_wayland` carries the file capability `cap_sys_nice=ep`, so the kernel starts it with
  `AT_SECURE=1` (read from the live process's auxv).
- In that mode, the Vulkan loader (1.4.363) reads `VK_DRIVER_FILES`, `VK_ICD_FILENAMES` and the
  `VK_LOADER_DRIVERS_SELECT` / `_DISABLE` filters through `secure_getenv()`, which returns NULL
  (`loader.c:3621`, `loader_environment.c:250`).
- Probe: a copy of `vulkaninfo` given the same capability, with `VK_DRIVER_FILES` set to the
  Radeon driver, still listed the A770 through the Intel driver. The same copy without the
  capability listed only RADV.
- KWin reads its own `KWIN_*` variables with `qEnvironmentVariable()`, a plain `getenv()`, so all
  three of those variables are honoured.

What KWin uses Vulkan for (master source, inferred, not tested): copies between GPUs
(`multigpuswapchain.cpp`) and dmabuf format filtering, both with EGL fallbacks. With one display
GPU, nothing should be lost.

Why a drop-in and not `/etc/environment`:

- `/etc/environment` has a note from 2026-07-29 saying not to set `KWIN_DRM_DEVICES` there,
  because card numbers are unstable and by-path names contain `:`. The udev names avoid both
  problems.
- The variables would then reach every process in the session, not just KWin and Xwayland.

## Change 2: SDDM's greeter Xorg

SDDM uses its default X11 greeter (`DisplayServer=x11`), and that Xorg runs for the whole session.

`/etc/X11/xorg.conf.d/10-display-gpu-only.conf`:

```
# X servers (SDDM greeter, X11 sessions) use only the Radeon 610M.
# The Arc A770 is compute-only: no GPU screen, so Xorg does not keep card0 open.
Section "ServerFlags"
    Option "AutoAddGPU" "off"
EndSection

Section "Device"
    Identifier "Radeon"
    Driver     "modesetting"
    BusID      "PCI:15:0:0"    # 0000:0f:00.0, decimal
EndSection
```

- `AutoAddGPU off` stops Xorg from turning the Arc into a GPU screen. That GPU screen is what kept
  `card0` open and ran glamor on the Arc.
- The `Device` section pins the Radeon as the primary display device instead of relying on
  `boot_vga` and probe order. `modesetting` is the driver Xorg already used.
- The file applies to every X server on the host: the greeter, any X11 session, and xpra. None of
  them has a use for the Arc as a display device.

## Change 3: keep the SDDM greeter off the Arc

Found after the reboot (see "Observed after reboot"). `sddm-greeter` is the Qt Quick login screen,
a client of the Xorg from Change 2, running as user `sddm` (`QT_QPA_PLATFORM=xcb`, theme
Sugar-Candy). It stays alive after login. It keeps fd 40 on `renderD128` with three `rw-s`
mappings (about 1 MiB, xe buffer objects) next to its three `card1` fds, so it renders on both
GPUs. Its environment already has `DRI_PRIME=pci-0000_0f_00_0`, which does not stop it. Why Mesa
opens the Arc node is not traced (suspect: render-node enumeration at context creation).

The render node was `0666` from `/usr/lib/udev/rules.d/50-udev-default.rules:61`. The greeter's
user `sddm` is in no groups, so restricting the node to group `render` blocks it without touching
anything on the desktop. `/etc/udev/rules.d/71-arc-render-restricted.rules`:

```
SUBSYSTEM=="drm", KERNELS=="0000:03:00.0", KERNEL=="renderD*", GROUP="render", MODE="0660"
```

This mode-only restriction assumes no ACL grants access separately. Checked 2026-10-09:
the installed `70-uaccess.rules` tags DRM `card*` nodes only; the Arc render node's
`udevadm info --query=property` output has no access tags, and `getfacl` shows only the
owner/group/other entries. Upstream's [render-node access rule](https://github.com/systemd/systemd/blob/main/rules.d/70-uaccess.rules.in)
is conditional on `GROUP_RENDER_UACCESS`. Recheck tags and ACLs if the packaged rules change;
`MODE="0660"` alone does not remove an ACL grant. This does not replace the still-missing check
with a running, active greeter.

- Applied 2026-10-06 01:58 with `udevadm trigger --action=change --subsystem-match=drm
  --sysname-match=renderD128`. `renderD128` is now `crw-rw---- root render`; `renderD129` (Radeon)
  stays `0666`.
- Checked: `udevadm verify` passes. As user `sddm`, `test -r /dev/dri/renderD128` fails; as
  `svnbjrn` it succeeds.
- Who keeps access: `render` members `svnbjrn`, `plex`, `xpum`; `llama-gpu@.service` runs as
  `User=svnbjrn` with `SupplementaryGroups=render video` and sets no `PrivateDevices=` or
  `DeviceAllow=`; root-run units bypass the mode. Service startup under the new mode was checked
  on the second boot (see below).
- Not covered: Docker containers given `--device /dev/dri/renderD128` whose process is non-root
  need a matching `render` GID. Any other user outside `render` loses the Arc, by design.
- Not changed by this rule: session apps run by `svnbjrn` are in `render` and can still open the
  Arc (see "What it does not do").
- Alternative if the rule causes trouble: `GreeterEnvironment=QT_QUICK_BACKEND=software` in
  `/etc/sddm.conf.d/` stops the greeter doing GL at all, with a possible cosmetic cost in
  Sugar-Candy. Not applied.

## Observed after reboot (2026-10-06 01:30 boot, checked ~01:50)

- Xorg (PID 2890): `AutoAddGPU off` applied; `grep -c 'modeset(G0)' /var/log/Xorg.0.log` is 0;
  only `modeset(0): using drv /dev/dri/card1`. Its four DRM fds are all on `card1`. The log still
  shows `Adding drm device (/dev/dri/card0)` and the `Platform probe` line, the one-off probe the
  doc predicted.
- Arc client list (`/sys/kernel/debug/dri/0000:03:00.0/clients`) held a single entry:
  `sddm-greeter`, tgid 3181, dev 128, uid 890. No `kwin_wayland`, logind, Xorg or Xwayland.
  `sudo fuser -v` on the Arc `card` and `render` by-path nodes listed only that greeter.
- `card0` had no holders.
- Not checked: whether the two `KWIN_*` variables are in KWin's live environment, and the `ccs_mode`
  write itself (still blocked by the greeter fd at that time).
- Correction to the plan above: the doc described "Xorg (SDDM's X11 greeter)" as one holder. The
  Xorg server and `sddm-greeter` are two processes; Change 2 fixed the first, Change 3 targets the
  second.

### Second boot (2026-10-06 02:17, checked 02:18 to 02:23)

Checked as root through `ssh vinbonesjr`.

- Arc client list: header line only. `fuser -v` on the Arc `card` and `render` by-path nodes: no
  holders. `sddm-greeter` was not running at all this boot, so this result alone does not show
  whether rule 71 or the greeter's absence kept it off. The rule's effect is shown separately:
  `renderD128` is `crw-rw---- root render` from boot, and `sudo -u sddm test -r` reports blocked.
- `modeset(G0)` count in the Xorg log: 0. `desktop-card` and `desktop-render` both resolve to
  `0000:0f:00.0`. KWin's live environment has `KWIN_DRM_DEVICES`, `KWIN_RENDER_NODES` and
  `KWIN_DISABLE_VULKAN=1`. No xe reset, hang or timeout in `dmesg` before the `ccs_mode` writes.
- `ccs_mode`, runtime write with no client on the card: `echo 2` returned 0 and read back 2. The
  kernel logged `Setting compute mode to 2`, `reset queued from ccs_mode_store`, `reset started`,
  `reset done`. `echo 1` did the same. Both took about 12 ms from queue to done. This confirms
  that a write succeeds from a desktop session once nothing holds the card.
- `llama-gpu@Ornith-1.5-35B-A3B-uncensored-Q8_0` started at 02:20 as `User=svnbjrn` with
  `SupplementaryGroups=render video`, in mode 1. `/health` returned 200 after about 40 s, the log
  reported `model loaded`, and the client list showed `llama-server` (uid 1000, dev 128) as the
  only client. So the `0660` mode does not stop the service. No xe fault logged.
- Not done: a completion request (the server wants an API key the session did not have), so
  inference and tok/s are unverified; IGT `xe_compute@ccs-mode-*`; any run in mode 2; whether the
  greeter, when it does run, is kept off by rule 71 (the greeter was absent).

## What should work after the reboot

These are expected results, not observed ones, except where "Observed after reboot" says
otherwise. Observed so far: items 1, 2 and 4 (nothing holds the card; a `ccs_mode` write
succeeds; the card has no holders with nothing running). Not tested: 3, 5 and 6.

Card and render-node numbers can change between boots, so every A770 path below uses its PCI
address, `0000:03:00.0`:

- sysfs: `/sys/bus/pci/devices/0000:03:00.0/...`
- debugfs: `/sys/kernel/debug/dri/0000:03:00.0/`
- device nodes: `/dev/dri/by-path/pci-0000:03:00.0-{card,render}`

1. **With no compute process and no session app on the Arc, nothing holds the A770.**
   `/sys/kernel/debug/dri/0000:03:00.0/clients` lists no client. A session app that opened the
   Arc's render node (see "What it does not do") stays listed until it exits. Stopping the compute
   services does not close its file.
2. **`ccs_mode` can be changed at runtime without logging out.**
   - First, stop the compute services and confirm the client list above is empty. Any remaining
     client, including a session app, makes the write fail with `EBUSY`.
   - Then:
     ```bash
     echo 2 | sudo tee /sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode
     ```
     The write should succeed, logging `Setting compute mode to 2` and then a GT reset. Before
     this change the same write returned `EBUSY`.
   - Then restore the production mode with
     `echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode` before restarting the
     compute services. The mode persists until the next write or reboot, and no workload has been
     measured in mode 2.
   - Whether mode 2 or 4 helps is a separate open question (#92). Mode N splits the 4 compute
     slices across N engines; it adds none.
   - Intel's guidance is that a single process does best with mode 1. The case worth measuring is
     two inference servers sharing the card.
3. **Intel's own `ccs_mode` tests can run from a desktop session.** IGT `xe_compute@ccs-mode-basic`
   and `xe_compute@ccs-mode-compute-kernel` need a device with no open clients.
4. **Sole tenancy can actually hold.** With nothing running, the card has no holders.
   - The tenancy gates still need the fix from #92, because unprivileged `fuser` stays blind.
   - Until then, check with `sudo`.
5. **A hang, reset or wedge on the A770 should leave the desktop alone.** The desktop no longer
   keeps contexts or buffers on the Arc. This is expected from how DRM works, not tested.
6. **Unbinding and rebinding `xe` (for example, to recover a wedged GPU) no longer leaves desktop
   files dangling on the device.** Expected, not tested.
   - Switching between i915 and xe still needs a reboot: the kernel command line (`force_probe`)
     decides which driver may bind.

Do not expect benchmark speedups from this change alone. The desktop's footprint on the Arc was
about 1 MiB of VRAM and a few thousand engine cycles.

## What it does not do

- Other Vulkan apps in the session can still open the Arc's render node, because the Intel Vulkan
  driver opens every render node it can when listing devices. `DRI_PRIME` only reorders devices;
  it does not hide any. Browsers already use the Radeon.
  - Such an app holds the node until it exits, and while it does, `ccs_mode` writes fail with
    `EBUSY`. Find it with the client list in "Post-reboot check" and close it before a write.
  - A per-app `VK_DRIVER_FILES` works for ordinary apps. It does not work for any binary with file
    capabilities or setuid, for the same reason it fails for KWin (see Change 1).
- It does not set `ccs_mode`. No boot-time writer is installed; `xe-a770-tune.service` still sets
  timeouts only.
- At greeter start, Xorg still opens and closes `card0` once to probe it (the "Platform probe ...
  card0" line in the log). It does not keep the file.
- The repo's tenancy gates and the `sudo fuser -k /dev/dri/renderD128` advice are unchanged; see
  #92. That command would kill the compositor.
- compute-runtime's `ZEX_NUMBER_OF_CCS` still cannot work on xe. It writes a binary `uint32_t`,
  which the kernel rejects with `EINVAL`; see #92.

## Post-reboot check

Run these from a terminal as the logged-in desktop user. Use `sudo` only where shown for
privileged probes; without root the compositor is invisible to `fuser`. `systemctl --user`
must query the desktop user's manager, not root's. Every device is named by PCI address,
so the checks stay valid even if the card numbers change at boot.

The empty-client expectations below require all Arc compute services and other Arc-using
apps to be stopped first. Stop the relevant units (for example,
`sudo systemctl stop llama-gpu@Ornith-1.5-35B-A3B-uncensored-Q8_0`) and close remaining Arc-using
apps, then run the probes. If compute is intentionally left running, expect its clients in
both lists; their presence is not a desktop-isolation failure, but it prevents `ccs_mode`
writes. Restart any services you stopped after the checks.

```bash
ARC=0000:03:00.0 RADEON=0000:0f:00.0
grep -E 'modeset\(G0\)' /var/log/Xorg.0.log                       # expect: no output
sudo cat /sys/kernel/debug/dri/$ARC/clients                        # expect: header line only
sudo fuser -v /dev/dri/by-path/pci-$ARC-card /dev/dri/by-path/pci-$ARC-render   # expect: no holders
systemctl --user show plasma-kwin_wayland.service -p Environment   # expect: the three variables
for n in card render; do                                           # expect: two "ok" lines
  [ "$(readlink -f /dev/dri/desktop-$n)" = "$(readlink -f /dev/dri/by-path/pci-$RADEON-$n)" ] \
    && echo "desktop-$n ok" || echo "desktop-$n MISMATCH"
done
```

Reading a failed check:

- `modeset(G0)` is still in the Xorg log: the Xorg file was not read or did not apply. Check
  `/var/log/Xorg.0.log` for `Using config directory` and for parse errors.
- `kwin_wayland` is in the client list: the drop-in did not reach KWin. Check that the unit lists
  the drop-in (`DropInPaths`) and that both `desktop-*` links exist.
  - If the drop-in is in effect and only render-node clients remain (`dev` 128 or higher in the
    list), `KWIN_DISABLE_VULKAN` was not honoured. Check the live process with
    `sudo sh -c 'tr "\0" "\n" < /proc/$(pgrep -x kwin_wayland)/environ' | grep KWIN_`.
- `systemd-logind` with DRM master is in the client list: KWin still took `card0` through logind,
  which means `KWIN_DRM_DEVICES` was not in effect.
- A process from the session other than KWin or Xorg: that is an app opening the render node (see
  "What it does not do"), not a failure of these changes.
- `sddm-greeter` in the client list: check rule 71 and the Arc's current permissions. Resolve
  the PCI-named link before listing permissions, testing access or tracing rules. The listing
  should show `crw-rw---- root render`, and the access test should print `blocked`:

  ```bash
  (
    set -e -o pipefail
    ARC_RENDER=$(readlink -e /dev/dri/by-path/pci-0000:03:00.0-render)
    test -c "$ARC_RENDER"
    ls -l "$ARC_RENDER"
    sudo -u sddm sh -c 'test -r "$1" && echo readable || echo blocked' sh "$ARC_RENDER"
    sudo udevadm test "/sys/class/drm/${ARC_RENDER##*/}" 2>&1 | grep 71-arc
  )
  ```

  A missing node stops the check. The udev trace includes stderr, where debug messages go.

## Rollback

From a text console (Ctrl+Alt+F3):

- No greeter: delete `/etc/X11/xorg.conf.d/10-display-gpu-only.conf`, then
  `sudo systemctl restart sddm`.
- Black screen after login: delete
  `~/.config/systemd/user/plasma-kwin_wayland.service.d/10-desktop-gpu.conf`, then run
  `systemctl --user daemon-reload` as that user, then `sudo systemctl restart sddm`.
  - The reload is needed because logging in on the text console keeps the user's systemd manager
    alive across the SDDM restart. That manager still has the deleted drop-in cached, and would
    start KWin with it again.
  - The likely cause is a missing or wrong `desktop-card` link, which leaves KWin with no GPU.
- The `70-desktop-gpu.rules` udev rule only adds links and is safe to leave in place.
- Rule 71: delete `/etc/udev/rules.d/71-arc-render-restricted.rules`, then reload the rules and
  trigger an `add` event for the Arc's current render node. The installed
  `50-udev-default.rules` skips its render-node mode assignment on `change` events; `add`
  reapplies the host's `0666` default. Do this if a non-root service outside `render` loses
  the Arc:

  ```bash
  (
    set -e
    ARC_RENDER=$(readlink -e /dev/dri/by-path/pci-0000:03:00.0-render)
    test -c "$ARC_RENDER"
    sudo udevadm control --reload
    sudo udevadm trigger --settle --action=add --subsystem-match=drm --sysname-match="${ARC_RENDER##*/}"
    ls -l "$ARC_RENDER"  # expect: crw-rw-rw- root render (0666)
  )
  ```

## Stale host configuration noticed, not changed

- `/etc/environment` lines 7-9: the 2026-07-29 note against `KWIN_DRM_DEVICES`. It should point to
  the drop-in instead.
- `/etc/environment` lines 13-14 say the Arc is absent. It is not.
- `/etc/environment`: `MOZ_DRM_DEVICE=/dev/dri/renderD129` uses a render-node number;
  `/dev/dri/desktop-render` would be stable.
- `conky-refresh.path` (disabled) watches `card0-DP-1` and `card0-HDMI-A-4`, which are now the
  Arc's connectors.

## Evidence

- Before state: live probes on 2026-10-05. Sources: kernel client list, `/proc/<pid>/fdinfo`, `fuser`
  as root, `/var/log/Xorg.0.log`, `udevadm info`.
- KWin behaviour: KWin master source (`drm_backend.cpp`, `gpumanager.cpp`, `renderdevice.cpp`,
  `utils/udev.cpp`, `multigpuswapchain.cpp`, `eglbackend.cpp`), and the variable names
  (`KWIN_DRM_DEVICES`, `KWIN_RENDER_NODES`, `KWIN_DISABLE_VULKAN`) checked in the installed
  `libkwin.so.6`.
- `VK_DRIVER_FILES` ignored by KWin, checked 2026-10-05:
  - The live `kwin_wayland` has `AT_SECURE=1` in its auxv.
  - The installed `libvulkan.so.1` (1.4.363) imports `secure_getenv`.
  - Vulkan-Loader v1.4.363 source reads the driver variables through it.
  - The `vulkaninfo` copy with `cap_sys_nice=ep` listed the A770 despite `VK_DRIVER_FILES`; the
    copy without the capability did not. Both copies were removed afterwards.
- Xorg config syntax: xserver 21.1.24 parser (`Flags.c`, `Device.c`, `scan.c`). Trailing `#`
  comments are accepted inside sections.
- `ccs_mode` behaviour: `drivers/gpu/drm/xe/xe_gt_ccs_mode.c` in mainline, plus a live probe. A
  text write returned `EBUSY` and a binary write `EINVAL`, and the value stayed 1.
- Installed files checked on 2026-10-05: `udevadm verify` passed, both links resolve to
  `0000:0f:00.0`, and after `daemon-reload` the user unit lists the drop-in and all three variables.
  The drop-in was changed the same evening from `VK_DRIVER_FILES` to `KWIN_DISABLE_VULKAN=1`, and
  the reload was repeated.
- PCI-stable paths: `/sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode` reads 1, and
  `/sys/kernel/debug/dri/0000:03:00.0/clients` exists. `fuser` on the `by-path` links reports the
  same holders as on `card0` / `renderD128`.
- After state, first reboot (2026-10-06): observed, see "Observed after reboot". Sources:
  `sudo fuser -v`, `/proc/{2890,3181}/fd` and `maps`, `/sys/kernel/debug/dri/0000:03:00.0/clients`,
  `/var/log/Xorg.0.log`, `ssh vinbonesjr` for root reads.
- Rule 71: `udevadm verify` and `udevadm test` output seen; the access test used `sudo -u sddm` and
  `sudo -u svnbjrn` from a root shell, not the service sandbox.
- After state, second reboot (02:17): observed, see "Second boot". Sources: client list, `fuser`,
  `ls -l /dev/dri`, `/proc/<kwin>/environ`, `dmesg`, `journalctl -u llama-gpu@...`, `/health`.
