#!/bin/bash
# Install patched xe.ko with 600s engine timeout (was 10s default)
# Requires: Secure Boot disabled, kernel 7.0.0-10-generic
set -e
KERNEL="$(uname -r)"
BACKUP="/lib/modules/$KERNEL/kernel/drivers/gpu/drm/xe/xe.ko.zst.orig"
TARGET="/lib/modules/$KERNEL/kernel/drivers/gpu/drm/xe/xe.ko.zst"
MODULE="$(dirname "$0")/xe.ko"

if [ ! -f "$MODULE" ]; then
    echo "ERROR: xe.ko not found in $(dirname "$0")"
    exit 1
fi

# Backup original if not already done
if [ ! -f "$BACKUP" ]; then
    echo "Backing up original xe.ko.zst..."
    cp "$TARGET" "$BACKUP"
fi

echo "Loading patched xe.ko..."
# Unbind GPUs
for dev in /sys/bus/pci/drivers/xe/*/; do
    pci=$(basename "$dev")
    echo "$pci" > /sys/bus/pci/drivers/xe/unbind 2>/dev/null || true
done
sleep 1

# Unload and reload
rmmod xe 2>/dev/null || true
sleep 1
insmod "$MODULE"
sleep 2

# Rebind GPUs
for dev in /sys/bus/pci/devices/*/vendor; do
    vendor=$(cat "$dev")
    if [ "$vendor" = "0x8086" ]; then
        pci=$(basename "$(dirname "$dev")")
        echo "$pci" > /sys/bus/pci/drivers/xe/bind 2>/dev/null || true
    fi
done
sleep 2

# Set timeouts
for eng in /sys/class/drm/card*/device/tile*/gt*/engines/*/job_timeout_ms; do
    echo 600000 > "$eng" 2>/dev/null || true
done

# Disable D3cold
for dev in /sys/bus/pci/drivers/xe/*/d3cold_allowed; do
    echo 0 > "$dev" 2>/dev/null || true
done

echo "Done. job_timeout_max: $(cat /sys/class/drm/card0/device/tile0/gt0/engines/ccs/job_timeout_max 2>/dev/null)"
echo "      job_timeout_ms:  $(cat /sys/class/drm/card0/device/tile0/gt0/engines/ccs/job_timeout_ms 2>/dev/null)"
