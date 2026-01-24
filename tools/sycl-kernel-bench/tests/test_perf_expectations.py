import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAIN_CPP = os.path.join(ROOT, "tools", "sycl-kernel-bench", "main.cpp")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_usage_includes_expect_flags():
    text = _read(MAIN_CPP)
    for flag in ("--expect-tps", "--expect-tops", "--expect-bandwidth", "--expect-xmx-util", "--xmx-peak-tops"):
        assert flag in text
    assert "--dpas-ntiles" in text
    for flag in ("--dpas-device-opt", "--dpas-autotune", "--dpas-autotune-force",
                 "--dpas-autotune-cache", "--dpas-autotune-metric", "--dpas-autotune-override-ntiles",
                 "--dpas-autotune-override-prefetch"):
        assert flag in text


def test_parse_kv_supports_expect_flags():
    text = _read(MAIN_CPP)
    for flag in ("--expect-tps", "--expect-tops", "--expect-bandwidth", "--expect-xmx-util", "--xmx-peak-tops"):
        pattern = re.escape(f'if (key == "{flag}")')
        assert re.search(pattern, text), f"Missing parse_kv handler for {flag}"
    assert re.search(re.escape('if (key == "--dpas-ntiles")'), text)
    for flag in ("--dpas-autotune-cache", "--dpas-autotune-override-ntiles",
                 "--dpas-autotune-override-prefetch"):
        pattern = re.escape(f'if (key == "{flag}")')
        assert re.search(pattern, text), f"Missing parse_kv handler for {flag}"
    assert re.search(re.escape('if (key == "--dpas-autotune-metric")'), text)


def test_parse_supports_dpas_tuning_flags():
    text = _read(MAIN_CPP)
    for flag in ("--dpas-device-opt", "--dpas-autotune", "--dpas-autotune-force"):
        pattern = re.escape(f'if (arg == "{flag}")')
        assert re.search(pattern, text), f"Missing parse handler for {flag}"


def test_expect_flags_in_value_arg_list():
    text = _read(MAIN_CPP)
    assert "--expect-tps" in text and "--expect-tops" in text
    assert "--expect-bandwidth" in text and "--expect-xmx-util" in text
    assert "--xmx-peak-tops" in text
