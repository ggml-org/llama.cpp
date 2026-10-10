"""Exercise the SYCL diagnostic's success/failure reporting without GPU work."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {
    "repro_bcs_reset.cpp": 10,
    "bcs_owned_memory_control.cpp": 32,
    "repro_mem_pressure.cpp": 32,
}
SYCL_STUB = r"""
#pragma once
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace sycl {
using exception_list = std::vector<std::exception_ptr>;
struct event {};
namespace info { namespace device { struct name {}; } }
struct device {
    template<class T> std::string get_info() { return "stub device"; }
};
struct handler {
    void depends_on(event) {}
    template<class F> void single_task(F f) { f(); }
};
class queue {
    std::function<void(exception_list)> on_error;
    int copies = 0, kernels = 0;
public:
    explicit queue(std::function<void(exception_list)> f) : on_error(f) {}
    device get_device() { return {}; }
    event memcpy(void *, const void *, size_t) { ++copies; return {}; }
    template<class F> void submit(F f) { handler h; f(h); ++kernels; }
    void wait_and_throw() {
        assert(copies == std::atoi(std::getenv("SYCL_TEST_BATCH")) && kernels == copies);
        const std::string mode = std::getenv("SYCL_TEST_MODE");
        if (mode == "hang") {
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        if (mode == "sync") {
            throw std::runtime_error("injected synchronous failure");
        }
        if (mode == "copy" || mode == "kernel") {
            on_error({std::make_exception_ptr(std::runtime_error("injected " + mode + " failure"))});
        }
        copies = kernels = 0;
    }
    ~queue() { assert(std::uncaught_exceptions() || (copies == 0 && kernels == 0)); }
};
inline void * malloc_device(size_t n, queue &) {
    const bool pressure = n == 12ULL * 1024 * 1024 * 1024;
    const std::string mode = std::getenv("SYCL_TEST_MODE");
    if ((pressure && mode == "pressure_alloc") || (!pressure && mode == "payload_alloc")) {
        return nullptr;
    }
    // The stub never touches the pressure pool; no 12 GiB host allocation is needed.
    return std::malloc(pressure ? 1 : n);
}
inline void free(void * p, queue &) { std::free(p); }
}
"""


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="bcs-control-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        (root / "sycl").mkdir()
        (root / "sycl/sycl.hpp").write_text(SYCL_STUB)
        cls.binaries = {}
        for source in CONTROLS:
            cls.binaries[source] = root / Path(source).stem
            subprocess.run([
                "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root),
                str(ROOT / source), "-o", str(cls.binaries[source]),
            ], check=True, timeout=30)

    def run_control(self, source, mode):
        return subprocess.run(
            ["timeout", "--kill-after=1s", "1s", str(self.binaries[source])],
            env={**os.environ, "SYCL_TEST_MODE": mode, "SYCL_TEST_BATCH": str(CONTROLS[source])},
            capture_output=True, text=True, timeout=10,
        )

    def test_success_is_labeled_as_control(self):
        for source in CONTROLS:
            with self.subTest(source=source):
                result = self.run_control(source, "success")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("success does not test the mmap defect", result.stdout)
                self.assertIn("Done.", result.stdout)

    def test_async_copy_and_kernel_failures_cannot_report_success(self):
        for source in CONTROLS:
            for mode in ("copy", "kernel"):
                with self.subTest(source=source, mode=mode):
                    result = self.run_control(source, mode)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(f"Asynchronous SYCL failure: injected {mode} failure", result.stderr)
                    self.assertNotIn("Done.", result.stdout)

    def test_synchronous_failure_cannot_report_success(self):
        for source in CONTROLS:
            with self.subTest(source=source):
                result = self.run_control(source, "sync")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("SYCL control failed: injected synchronous failure", result.stderr)
                self.assertNotIn("Done.", result.stdout)

    def test_pressure_allocation_failure_stops_before_workload(self):
        result = self.run_control("repro_mem_pressure.cpp", "pressure_alloc")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Failed to allocate 12 GiB VRAM pressure pool", result.stderr)
        self.assertNotIn("Starting stress loop", result.stdout)
        self.assertNotIn("Done.", result.stdout)

    def test_payload_allocation_failure_cannot_report_success(self):
        for source in CONTROLS:
            with self.subTest(source=source):
                result = self.run_control(source, "payload_alloc")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Failed to allocate control buffers", result.stderr)
                self.assertNotIn("Done.", result.stdout)

    def test_timeout_wrapper_stops_stalled_controls(self):
        for source in CONTROLS:
            with self.subTest(source=source):
                result = self.run_control(source, "hang")
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertNotIn("Done.", result.stdout)


if __name__ == "__main__":
    unittest.main()
