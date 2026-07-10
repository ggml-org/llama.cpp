"""Egress-тест: offline-инвариант контура (ноль исходящих за пределы loopback).

Проверяет сам МЕХАНИЗМ изоляции, на который опирается развёртывание: в изолированном
сетевом namespace (unshare -rn) с поднятым только loopback (a) внутренние соединения
работают, (b) любой внешний адрес недостижим. Это тот же контур, в котором DevOps
запускает gateway+llama-server (systemd IPAddressDeny / docker internal:true / nftables).

Требует Linux + unprivileged netns (unshare -rn) + iproute2. Иначе тест пропускается
(на macOS/в ограниченном CI), а не даёт ложно-зелёный результат.
"""
import shutil
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import socket, subprocess, sys
    subprocess.run(["ip", "link", "set", "lo", "up"], check=False)
    # 1) loopback обязан работать (внутренний трафик контура)
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    c = socket.socket()
    try:
        c.settimeout(2); c.connect(("127.0.0.1", port))
    except Exception as e:
        print("LOOPBACK_FAIL", e); sys.exit(2)
    finally:
        c.close(); srv.close()
    # 2) внешний адрес обязан быть недостижим (нет маршрута наружу == ноль egress)
    ext = socket.socket()
    try:
        ext.settimeout(2); ext.connect(("8.8.8.8", 53))
        print("EGRESS_LEAK"); sys.exit(3)
    except OSError:
        pass
    finally:
        ext.close()
    print("OK"); sys.exit(0)
    """
)


def _netns_available():
    if sys.platform != "linux":
        return False
    if not shutil.which("unshare") or not shutil.which("ip"):
        return False
    try:
        r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.egress
def test_runtime_has_zero_egress():
    if not _netns_available():
        pytest.skip("нужен Linux + unshare -rn (unprivileged netns) + iproute2")
    r = subprocess.run(
        ["unshare", "-rn", "python3", "-c", _CHILD],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"offline-инвариант нарушен: rc={r.returncode} out={r.stdout!r} err={r.stderr!r}"
    )
