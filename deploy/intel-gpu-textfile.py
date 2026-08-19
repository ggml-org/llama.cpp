#!/usr/bin/env python3
"""Export Intel GPU engine and sensor metrics for node_exporter's textfile
collector.

Why not intel_gpu_top: it is i915-only and refuses to run on the xe driver
("Detected Xe device which is not supported by intel_gpu_top"), and gputop --
the tool it defers to -- has no machine-readable output in igt-gpu-tools 2.4
(only -d/-n/-h). Parsing its TUI table would be fragile.

Engine busy comes from the kernel's drm-usage-stats interface in
/proc/<pid>/fdinfo/<fd>. Both drivers implement it, in different forms:

    xe:   drm-cycles-<class> / drm-total-cycles-<class>   driver-native ticks
    i915: drm-engine-<class>                              nanoseconds busy

Both are cumulative, so they are exported as counters and the averaging is left
to PromQL:

    rate(intel_gpu_engine_busy_total) / rate(intel_gpu_engine_elapsed_total)

which is dimensionless and so works unchanged for either form. That beats
sampling: it averages over the whole scrape interval rather than a 1s slice,
and it needs no CAP_PERFMON, no perf, and no igt-gpu-tools.

Temperature, energy and fan come from the GPU's own hwmon device, read here
rather than via node_exporter's hwmon collector because xe exposes no
power1_average -- only a cumulative energy*_input -- and because reading it
directly lets us use the sensors' own *_label strings and skip a group_left
join to separate this GPU from the host's other hwmon devices.

Every series carries a driver="i915"|"xe" label. Hosts often have more than one
GPU (rainbow pairs the Arc with an AMD integrated one), so which device a
number came from should be visible in the data, not just implied by a filter.
Non-Intel GPUs are ignored entirely -- amdgpu and friends are not in DRIVERS.

Engine class labels map the kernel's abbreviations to readable names:

    rcs -> render        bcs -> copy          ccs -> compute
    vcs -> video         vecs -> video-enhance

Scope: fdinfo is per-DRM-client, so engine busy measures what the GPU clients on
this box are doing -- on a dedicated inference host, llama-server. It is not the
whole-card figure a vendor exporter reports. A consequence: in router mode the
child process changes when a model is swapped, so the counters reset. rate()
handles resets, at the cost of one interval across a switch.

Usage:
    intel-gpu-textfile.py /var/lib/node_exporter/textfile_collector/intel_gpu.prom

Environment:
    INTEL_GPU_DRIVERS      comma-separated drm-driver names to accept,
                           default "i915,xe"
    INTEL_GPU_HWMON_ROOT   sysfs hwmon root, default "/sys/class/hwmon"
"""

import os
import re
import sys
import tempfile
import time

DRIVERS = set(
    d.strip() for d in os.environ.get("INTEL_GPU_DRIVERS", "i915,xe").split(",") if d.strip()
)

HWMON_ROOT = os.environ.get("INTEL_GPU_HWMON_ROOT", "/sys/class/hwmon")

ENGINE_NAMES = {
    "rcs":  "render",
    "bcs":  "copy",
    "ccs":  "compute",
    "vcs":  "video",
    "vecs": "video-enhance",
}

# sysfs sensor prefix -> (metric, label key, divisor to reach the base unit)
HWMON_SENSORS = {
    "energy": ("intel_gpu_energy_joules_total", "domain", 1e6),  # microjoules
    "temp":   ("intel_gpu_temp_celsius",        "sensor", 1e3),  # millidegrees
    "fan":    ("intel_gpu_fan_rpm",             "fan",    1),    # rpm
}

HWMON_INPUT_RE = re.compile(r"^(energy|temp|fan)(\d+)_input$")
HWMON_CAP_RE = re.compile(r"^power(\d+)_cap$")

METRICS = {
    "intel_gpu_engine_busy_total": (
        "counter",
        "Cumulative GPU engine busy time, in driver-native units, divided by engine capacity.",
    ),
    "intel_gpu_engine_elapsed_total": (
        "counter",
        "Cumulative reference clock matching intel_gpu_engine_busy_total; divide the rates for utilisation.",
    ),
    "intel_gpu_engine_capacity": ("gauge", "Number of hardware engines in this class."),
    "intel_gpu_clients": (
        "gauge",
        "Number of DRM clients holding a file descriptor on this GPU.",
    ),
    "intel_gpu_energy_joules_total": (
        "counter",
        "Cumulative GPU energy use in joules; rate() this for power in watts.",
    ),
    "intel_gpu_temp_celsius": ("gauge", "GPU temperature sensor reading."),
    "intel_gpu_fan_rpm": ("gauge", "GPU fan speed."),
    "intel_gpu_power_cap_watts": ("gauge", "Configured GPU power limit."),
    "intel_gpu_hwmon_sensors": (
        "gauge",
        "Number of hwmon sensors read from this GPU; 0 means none were found.",
    ),
}


def escape(value: str) -> str:
    """Escape a Prometheus label value."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def fmt(value) -> str:
    """Render a metric value without losing precision.

    These are large cumulative counters (the xe reference clock is already
    ~1e12 ticks), and "%g" would round to 6 significant figures -- quantising
    the very deltas rate() is meant to measure. Keep integers exact.
    """
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return repr(value)


def labels(**pairs) -> str:
    inner = ",".join('%s="%s"' % (k, escape(str(v))) for k, v in pairs.items())
    return "{%s}" % inner if inner else ""


def read_file(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def parse_fdinfo(text: str) -> dict:
    """Pull the drm-* keys we care about out of one fdinfo file."""
    out = {"driver": None, "client_id": None, "busy": {}, "elapsed": {}, "capacity": {}}

    for line in text.splitlines():
        key, _, raw = line.partition(":")
        if not raw or not key.startswith("drm-"):
            continue
        raw = raw.strip()

        if key == "drm-driver":
            out["driver"] = raw
            continue
        if key == "drm-client-id":
            out["client_id"] = raw
            continue

        # values may carry a unit suffix ("0 ns", "16248 KiB"); we only read
        # unitless counters here, so take the first field and require an int
        try:
            value = int(raw.split()[0])
        except (ValueError, IndexError):
            continue

        # order matters: the longer prefixes are checked first, otherwise
        # drm-engine-capacity-vcs parses as class "capacity-vcs" and
        # drm-total-cycles-rcs as a busy counter
        if key.startswith("drm-engine-capacity-"):
            out["capacity"][key[len("drm-engine-capacity-"):]] = value
        elif key.startswith("drm-total-cycles-"):
            out["elapsed"][key[len("drm-total-cycles-"):]] = value
        elif key.startswith("drm-cycles-"):
            out["busy"][key[len("drm-cycles-"):]] = value
        elif key.startswith("drm-engine-"):
            # i915 form: nanoseconds busy, with wall clock as the reference
            out["busy"][key[len("drm-engine-"):]] = value

    return out


def collect_engines() -> dict:
    """Walk every Intel DRM client and aggregate per (driver, engine class)."""
    busy = {}      # summed across clients
    elapsed = {}   # a shared reference clock, so take the max not the sum
    capacity = {}
    clients = {}
    seen = set()
    ns_drivers = set()

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue  # process exited, or not ours to read

        for fd in fds:
            # readlink is much cheaper than reading fdinfo, and reading fdinfo
            # on a DRM fd makes the driver compute stats -- so filter first
            try:
                if not os.readlink(f"{fd_dir}/{fd}").startswith("/dev/dri/"):
                    continue
                info = parse_fdinfo(read_file(f"/proc/{pid}/fdinfo/{fd}") or "")
            except OSError:
                continue

            driver = info["driver"]
            if driver not in DRIVERS:
                continue  # not an Intel GPU (amdgpu, nouveau, ...)

            # dup'd fds report the same client, so count each client once
            key = (driver, info["client_id"] if info["client_id"] is not None else f"{pid}/{fd}")
            if key in seen:
                continue
            seen.add(key)
            clients[driver] = clients.get(driver, 0) + 1

            for cls, value in info["busy"].items():
                busy[(driver, cls)] = busy.get((driver, cls), 0) + value
            for cls, value in info["elapsed"].items():
                k = (driver, cls)
                elapsed[k] = max(elapsed.get(k, 0), value)
            for cls, value in info["capacity"].items():
                k = (driver, cls)
                capacity[k] = max(capacity.get(k, 1), value)

            if info["busy"] and not info["elapsed"]:
                ns_drivers.add(driver)

    if ns_drivers:
        # i915 reports busy nanoseconds with no reference counter, so supply
        # one: CLOCK_MONOTONIC is cumulative and does not step backwards
        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        for driver, cls in busy:
            if driver in ns_drivers:
                elapsed.setdefault((driver, cls), now_ns)

    out = {}

    def engine_label(driver, cls):
        return labels(driver=driver, engine=ENGINE_NAMES.get(cls, cls))

    # busy is divided by engine capacity, so a class with several engines
    # (drm-engine-capacity-vcs is 2 on Arc) still lands in 0..1 after the rate
    # division rather than 0..2
    out["intel_gpu_engine_busy_total"] = [
        (engine_label(driver, cls), value / capacity.get((driver, cls), 1))
        for (driver, cls), value in sorted(busy.items())
        if (driver, cls) in elapsed
    ]
    out["intel_gpu_engine_elapsed_total"] = [
        (engine_label(driver, cls), value) for (driver, cls), value in sorted(elapsed.items())
    ]
    out["intel_gpu_engine_capacity"] = [
        (engine_label(driver, cls), value) for (driver, cls), value in sorted(capacity.items())
    ]
    out["intel_gpu_clients"] = [
        (labels(driver=driver), value) for driver, value in sorted(clients.items())
    ]
    return out


def collect_hwmon() -> dict:
    """Read each Intel GPU's own hwmon device.

    node_exporter's hwmon collector could nearly do this, but xe exposes no
    power1_average -- only a cumulative energy*_input -- and node_exporter's
    name for an energy counter is not something to guess at. Reading sysfs here
    means the metric names are ours, the sensors' own *_label strings are used
    instead of "temp5", and the dashboards need no group_left join to separate
    this GPU from the host's other hwmon devices (rainbow also carries amdgpu,
    nvme, k10temp, spd5118, ...).
    """
    out = {}
    n_sensors = {}

    try:
        entries = sorted(os.listdir(HWMON_ROOT))
    except OSError:
        entries = []

    for entry in entries:
        base = os.path.join(HWMON_ROOT, entry)
        driver = read_file(os.path.join(base, "name"))
        if driver not in DRIVERS:
            continue
        n_sensors.setdefault(driver, 0)

        try:
            files = sorted(os.listdir(base))
        except OSError:
            continue

        for name in files:
            m = HWMON_INPUT_RE.match(name)
            cap = None if m else HWMON_CAP_RE.match(name)

            if m:
                kind, index = m.group(1), m.group(2)
                metric, label_key, divisor = HWMON_SENSORS[kind]
                stem = f"{kind}{index}"
            elif cap:
                metric, label_key, divisor = "intel_gpu_power_cap_watts", "domain", 1e6
                stem = f"power{cap.group(1)}"
            else:
                continue

            raw = read_file(os.path.join(base, name))
            if raw is None:
                continue  # a sensor can return EIO while the GPU is suspended
            try:
                value = int(raw) / divisor
            except ValueError:
                continue

            # prefer the sensor's own label: Arc exposes 20 temp sensors, and
            # "temp5" tells you nothing about which one melted
            label = read_file(os.path.join(base, f"{stem}_label")) or stem
            out.setdefault(metric, []).append(
                (labels(driver=driver, **{label_key: label}), value)
            )
            n_sensors[driver] += 1

    out["intel_gpu_hwmon_sensors"] = [
        (labels(driver=driver), value) for driver, value in sorted(n_sensors.items())
    ]
    return out


def render(collected: dict) -> str:
    lines = []
    for name, values in collected.items():
        values = [v for v in values if v is not None]
        if not values:
            continue
        kind, help_text = METRICS[name]
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for label_str, value in values:
            lines.append(f"{name}{label_str} {fmt(value)}")
    return "\n".join(lines) + "\n" if lines else ""


def write_atomic(path: str, body: str) -> None:
    """node_exporter reads these files unsynchronised, so swap it in atomically."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".intel_gpu.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.prom>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    health = (
        "# HELP intel_gpu_scrape_success 1 if engine counters were read from DRM fdinfo.\n"
        "# TYPE intel_gpu_scrape_success gauge\n"
    )

    collected = {}
    ok = True

    # the two halves fail independently: a GPU with no DRM client still has
    # readable sensors, and vice versa
    for name, fn in (("engines", collect_engines), ("hwmon", collect_hwmon)):
        try:
            collected.update(fn())
        except Exception as e:  # noqa: BLE001 - a failure must still publish health
            print(f"{name} collection failed: {e}", file=sys.stderr)
            ok = False

    if not collected.get("intel_gpu_engine_busy_total"):
        # no GPU client, or the driver exposes no per-engine counters. Say so
        # rather than leaving the last good values in place to go stale.
        print(
            f"no engine counters found (drivers={sorted(DRIVERS)}, "
            f"clients={collected.get('intel_gpu_clients') or 'none'})",
            file=sys.stderr,
        )
        ok = False

    write_atomic(path, health + f"intel_gpu_scrape_success {1 if ok else 0}\n" + render(collected))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
