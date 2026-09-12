"""Exact P6 VNNI GEMV versus original, P6 AVX2 and scalar independent oracle."""
import unittest
import numpy as np

import support
import test_q4kp_kernel as fixtures


class VNNIKernelTests(unittest.TestCase):
    cases = 0
    output_floats = 0

    @classmethod
    def setUpClass(cls):
        cls.dll = support.library()
        if not cls.dll.q4kp_vnni_supported():
            raise unittest.SkipTest("vnni ISA unavailable")

    def compare(self, n, nc, packed, q8):
        original, activations = packed.copy(), q8.copy()
        recoded = packed.copy()
        self.assertEqual(self.dll.q4kp_recode(recoded.ctypes.data, recoded.nbytes), 0)
        p6_before = recoded.copy()
        outs = [np.full(nc + 2, np.nan, dtype=np.float32) for _ in range(4)]
        for fn, matrix, out in ((self.dll.q4kp_original_gemv, packed, outs[0]),
                               (self.dll.q4kp_gemv, recoded, outs[1]),
                               (self.dll.q4kp_vnni_gemv, recoded, outs[2])):
            fn(n, out.ctypes.data + 4, 789, matrix.ctypes.data, q8.ctypes.data, 1, nc)
        self.dll.q4kp_scalar_gemv(n, outs[3].ctypes.data + 4, packed.ctypes.data, q8.ctypes.data, nc)
        for out in outs[1:]:
            np.testing.assert_array_equal(out.view(np.uint32), outs[0].view(np.uint32))
        self.assertTrue(np.isfinite(outs[0][1:-1]).all())
        self.assertTrue(np.isnan(outs[2][[0, -1]]).all())
        np.testing.assert_array_equal(packed, original)
        np.testing.assert_array_equal(q8, activations)
        np.testing.assert_array_equal(recoded, p6_before)
        type(self).cases += 1
        type(self).output_floats += nc

    def test_random_multiblock_and_row_groups(self):
        for n in (256, 512, 4096):
            for nc in (8, 16, 40):
                for seed in (18, 901, 672):
                    self.compare(n, nc, *fixtures.fixtures(n, nc, seed))

    def test_integer_extrema_no_saturation(self):
        for n in (256, 512, 4096):
            for code in (0, 255, 0xF0, 0x0F, 0x87):
                packed, _ = fixtures.fixtures(n, 40, code)
                packed[:, 128:] = code
                packed[:, 32:128] = 255
                for value in (-128, -1, 0, 1, 127):
                    self.compare(n, 40, packed, fixtures.raw_q8(np.full((n // 256, 256), value, dtype=np.int8)))

    def test_every_six_bit_scale_code(self):
        packed, q8 = fixtures.fixtures(256, 512)
        scales = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None], (64, 8, 8)).copy()
        packed[:, 32:128] = fixtures.pack_metadata(scales, 63 - scales)
        self.compare(256, 512, packed, q8)

    def test_half_edges(self):
        packed, q8 = fixtures.fixtures(512, 16)
        halves = np.array([0., -0., 2 ** -24, -(2 ** -24), 2 ** -14, .125, -128, 65504], dtype="<f2")
        packed[:, :16] = halves.view(np.uint8)
        packed[:, 16:32] = halves[::-1].copy().view(np.uint8)
        self.compare(512, 16, packed, q8)

    def test_invalid_args_leave_output_untouched(self):
        packed, q8 = fixtures.fixtures(256, 8)
        self.assertEqual(self.dll.q4kp_recode(packed.ctypes.data, packed.nbytes), 0)
        for n, nr, nc in ((0, 1, 8), (255, 1, 8), (256, 2, 8), (256, 1, 7)):
            out = np.full(8, 123.5, dtype=np.float32)
            self.dll.q4kp_vnni_gemv(n, out.ctypes.data, 0, packed.ctypes.data, q8.ctypes.data, nr, nc)
            self.assertTrue((out == 123.5).all())
        for missing in (0, 1, 2):
            out = np.full(8, 123.5, dtype=np.float32)
            ptrs = [out.ctypes.data, packed.ctypes.data, q8.ctypes.data]
            ptrs[missing] = None
            self.dll.q4kp_vnni_gemv(256, ptrs[0], 0, ptrs[1], ptrs[2], 1, 8)
            self.assertTrue((out == 123.5).all())


if __name__ == "__main__":
    unittest.main()
