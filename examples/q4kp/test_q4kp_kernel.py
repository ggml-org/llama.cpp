"""Q4KP lossless metadata and GEMV/GEMM correctness; no model inference."""
import ctypes
import unittest
import numpy as np
from support import library

def metadata_reference(packed):
    """Read each six-bit field independently, without uint32 word unpacking."""
    out = np.empty((len(packed), 8, 16), dtype=np.uint8)
    for b, block in enumerate(packed):
        codes = block[32:128].reshape(8, 12)
        for group in range(8):
            c = codes[group].astype(np.uint16)
            for row in range(8):
                if row < 4:
                    out[b, group, row] = c[row] & 63
                    out[b, group, 8 + row] = c[row + 4] & 63
                else:
                    out[b, group, row] = (c[row + 4] & 15) | ((c[row - 4] >> 6) << 4)
                    out[b, group, 8 + row] = (c[row + 4] >> 4) | ((c[row] >> 6) << 4)
    return out.reshape(len(packed), 128)


def pack_metadata(scales, mins):
    """Independent encoder from explicit [block, subblock, row] six-bit values."""
    shape = scales.shape
    result = np.zeros((shape[0], 8, 12), dtype=np.uint8)
    for row in range(4):
        result[:, :, row] = scales[:, :, row] | ((scales[:, :, row + 4] >> 4) << 6)
        result[:, :, row + 4] = mins[:, :, row] | ((mins[:, :, row + 4] >> 4) << 6)
        result[:, :, row + 8] = (scales[:, :, row + 4] & 15) | ((mins[:, :, row + 4] & 15) << 4)
    return result.reshape(shape[0], 96)


def fixtures(n, nc, seed=421):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (n // 256 * (nc // 8), 1152), dtype=np.uint8)
    packed[:, :32] = rng.uniform(-.2, .2, (len(packed), 16)).astype("<f2").view(np.uint8).reshape(-1, 32)
    values = rng.integers(-128, 128, (n // 256, 256), dtype=np.int16).astype(np.int8)
    q8 = raw_q8(values, rng.uniform(-.3, .3, n // 256).astype(np.float32))
    return packed, q8


def raw_q8(values, delta=None):
    values = np.ascontiguousarray(values, dtype=np.int8).reshape(-1, 256)
    raw = np.zeros((len(values), 292), dtype=np.uint8)
    delta = np.ones(len(values), dtype="<f4") if delta is None else np.asarray(delta, dtype="<f4")
    raw[:, :4] = delta.view(np.uint8).reshape(-1, 4)
    raw[:, 4:260] = values.view(np.uint8)
    raw[:, 260:] = values.astype(np.int16).reshape(-1, 16, 16).sum(axis=2).astype("<i2").view(np.uint8).reshape(-1, 32)
    return raw


def decode_p6(packed):
    decoded = np.empty((len(packed), 8, 16), dtype=np.uint8)
    for b, block in enumerate(packed):
        for g in range(8):
            code = block[32 + g * 12:32 + (g + 1) * 12].tobytes()
            scales, mins = int.from_bytes(code[:6], "little"), int.from_bytes(code[6:], "little")
            decoded[b, g, :8] = [(scales >> (6 * row)) & 63 for row in range(8)]
            decoded[b, g, 8:] = [(mins >> (6 * row)) & 63 for row in range(8)]
    return decoded


def raw_q8x4(values, delta):
    values = np.asarray(values, dtype=np.int8)
    nr, nb, _ = values.shape
    raw = np.zeros((nr // 4 * nb, 1168), dtype=np.uint8)
    assert raw.ctypes.data % 16 == 0  # original GEMM uses aligned d loads
    for r in range(0, nr, 4):
        for b in range(nb):
            block = raw[(r // 4) * nb + b]
            group = values[r:r + 4, b]
            block[:16] = np.ascontiguousarray(delta[r:r + 4, b], dtype="<f4").view(np.uint8)
            block[16:1040] = group.reshape(4, 32, 8).transpose(1, 0, 2).copy().view(np.uint8).reshape(-1)
            sums = group.astype(np.int16).reshape(4, 4, 4, 16).sum(axis=3).astype("<i2")
            block[1040:] = sums.transpose(1, 0, 2).copy().view(np.uint8).reshape(-1)
    return raw


class Q4KPKernelTests(unittest.TestCase):
    gemv_cases = 0
    gemm_cases = 0
    output_floats = 0

    @classmethod
    def setUpClass(cls):
        cls.dll = library()

    def recode(self, packed):
        recoded = packed.copy()
        self.assertEqual(self.dll.q4kp_recode(recoded.ctypes.data, recoded.nbytes), 0)
        decoded = decode_p6(recoded)
        np.testing.assert_array_equal(decoded.reshape(len(packed), 128), metadata_reference(packed))
        # Bijective round-trip back to all original 96 metadata bytes.
        np.testing.assert_array_equal(pack_metadata(decoded[:, :, :8], decoded[:, :, 8:]), packed[:, 32:128])
        np.testing.assert_array_equal(recoded[:, :32], packed[:, :32])
        np.testing.assert_array_equal(recoded[:, 128:], packed[:, 128:])
        return recoded

    def compare_gemv(self, n, nc, packed, q8):
        original, a_before = packed.copy(), q8.copy()
        recoded = self.recode(packed)
        before = recoded.copy()
        old, new, scalar = (np.full(nc + 2, np.nan, dtype=np.float32) for _ in range(3))
        self.dll.q4kp_original_gemv(n, old.ctypes.data + 4, 777, packed.ctypes.data, q8.ctypes.data, 1, nc)
        self.dll.q4kp_gemv(n, new.ctypes.data + 4, 777, recoded.ctypes.data, q8.ctypes.data, 1, nc)
        self.dll.q4kp_scalar_gemv(n, scalar.ctypes.data + 4, packed.ctypes.data, q8.ctypes.data, nc)
        np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32), "GEMV old/new bits")
        np.testing.assert_array_equal(new.view(np.uint32), scalar.view(np.uint32), "GEMV scalar bits")
        self.assertTrue(np.isfinite(new[1:-1]).all())
        self.assertTrue(np.isnan(new[[0, -1]]).all())
        np.testing.assert_array_equal(packed, original)
        np.testing.assert_array_equal(q8, a_before)
        np.testing.assert_array_equal(recoded, before)
        type(self).gemv_cases += 1
        type(self).output_floats += nc

    def compare_gemm(self, n, nr, nc, packed, q8):
        original, a_before = packed.copy(), q8.copy()
        recoded = self.recode(packed)
        before = recoded.copy()
        stride = nc + 3  # Exercise bs, disjoint rows and output guards.
        old, new, scalar = (np.full((nr, stride), np.nan, dtype=np.float32) for _ in range(3))
        self.dll.q4kp_original_gemm(n, old.ctypes.data, stride, packed.ctypes.data, q8.ctypes.data, nr, nc)
        self.dll.q4kp_gemm(n, new.ctypes.data, stride, recoded.ctypes.data, q8.ctypes.data, nr, nc)
        self.dll.q4kp_scalar_gemm(n, scalar.ctypes.data, stride, packed.ctypes.data, q8.ctypes.data, nr, nc)
        np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32), "GEMM old/new bits")
        np.testing.assert_array_equal(new.view(np.uint32), scalar.view(np.uint32), "GEMM scalar bits")
        self.assertTrue(np.isfinite(new[:, :nc]).all())
        self.assertTrue(np.isnan(new[:, nc:]).all())
        np.testing.assert_array_equal(packed, original)
        np.testing.assert_array_equal(q8, a_before)
        np.testing.assert_array_equal(recoded, before)
        type(self).gemm_cases += 1
        type(self).output_floats += nr * nc

    def test_random_gemv_multiblock_and_rowgroups(self):
        for n in (256, 512, 4096):
            for nc in (8, 16, 40):
                for seed in (18, 901):
                    self.compare_gemv(n, nc, *fixtures(n, nc, seed))

    def test_gemm_both_avx2_branches_and_tails(self):
        for n in (256, 512, 4096):
            for nr in (4, 8, 16, 20, 32):
                for nc in (8, 16, 40):
                    for seed in (46, 357):
                        rng = np.random.default_rng(seed)
                        packed, _ = fixtures(n, nc, seed)
                        values = rng.integers(-128, 128, (nr, n // 256, 256), dtype=np.int16).astype(np.int8)
                        delta = rng.uniform(-.3, .3, (nr, n // 256)).astype(np.float32)
                        self.compare_gemm(n, nr, nc, packed, raw_q8x4(values, delta))

    def test_every_scale_code_roundtrip(self):
        packed, q8 = fixtures(256, 512)
        scales = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None], (64, 8, 8)).copy()
        mins = 63 - scales
        packed[:, 32:128] = pack_metadata(scales, mins)
        self.compare_gemv(256, 512, packed, q8)

    def test_q8_and_nibble_extremes(self):
        for code in (0, 255, 0xF0, 0x0F, 0x87):
            packed, _ = fixtures(512, 16, code)
            packed[:, 128:] = code
            packed[:, 32:128] = 255
            for val in (-128, -1, 0, 1, 127):
                self.compare_gemv(512, 16, packed, raw_q8(np.full((2, 256), val, dtype=np.int8)))
                values = np.full((20, 2, 256), val, dtype=np.int8)
                self.compare_gemm(512, 20, 16, packed, raw_q8x4(values, np.ones((20, 2), dtype=np.float32)))

    def test_finite_half_edges(self):
        packed, q8 = fixtures(512, 16)
        halves = np.array([0., -0., 2 ** -24, -(2 ** -24), 2 ** -14, .125, -128, 65504], dtype="<f2")
        packed[:, :16] = halves.view(np.uint8)
        packed[:, 16:32] = halves[::-1].copy().view(np.uint8)
        self.compare_gemv(512, 16, packed, q8)
        values = np.resize(np.array([-128, 127, -1, 1], dtype=np.int8), (20, 2, 256))
        self.compare_gemm(512, 20, 16, packed, raw_q8x4(values, np.ones((20, 2), dtype=np.float32)))

    def test_invalid_recode_atomic_failure_and_guards(self):
        packed, _ = fixtures(256, 16)
        for variant in ("short", "extra", "null", "nan", "inf", "negative_inf"):
            data = packed.copy()
            size, pointer = data.nbytes, data.ctypes.data
            if variant == "short": size -= 1
            if variant == "extra": size += 1
            if variant == "null": pointer = None
            if variant in ("nan", "inf", "negative_inf"):
                value = {"nan": 0x7E01, "inf": 0x7C00, "negative_inf": 0xFC00}[variant]
                data[-1, 30:32] = np.frombuffer(value.to_bytes(2, "little"), dtype=np.uint8)
            before = data.copy()
            self.assertEqual(self.dll.q4kp_recode(pointer, size), -3 if variant in ("nan", "inf", "negative_inf") else -1)
            np.testing.assert_array_equal(data, before)
        self.assertEqual(self.dll.q4kp_recode(None, 0), 0)
        self.assertEqual(self.dll.q4kp_recode(ctypes.c_size_t(-1).value - 4, 1152), -1)
        guarded = np.full(packed.nbytes + 2, 0xA5, dtype=np.uint8)
        guarded[1:-1] = packed.reshape(-1)
        self.assertEqual(self.dll.q4kp_recode(guarded.ctypes.data + 1, packed.nbytes), 0)
        np.testing.assert_array_equal(guarded[1:-1].reshape(-1, 1152), self.recode(packed))
        self.assertEqual(guarded[0], 0xA5)
        self.assertEqual(guarded[-1], 0xA5)

    def test_invalid_dimensions_and_null_pointers_do_not_write(self):
        packed, q8 = fixtures(256, 8)
        recoded = self.recode(packed)
        for kind, cases in (("gemv", [(0, 1, 8, 8), (255, 1, 8, 8), (256, 2, 8, 8), (256, 1, 7, 8)]),
                            ("gemm", [(256, 0, 8, 8), (256, 3, 8, 8), (256, 4, 8, 7), (255, 4, 8, 8)])):
            fn = getattr(self.dll, "q4kp_" + kind)
            out = np.full(128, 123.5, dtype=np.float32)
            for n, nr, nc, stride in cases:
                fn(n, out.ctypes.data, stride, recoded.ctypes.data, q8.ctypes.data, nr, nc)
                self.assertTrue((out == 123.5).all())
            for missing in ("s", "x", "y"):
                args = {"s": out.ctypes.data, "x": recoded.ctypes.data, "y": q8.ctypes.data}
                args[missing] = None
                fn(256, args["s"], 8, args["x"], args["y"], 1 if kind == "gemv" else 4, 8)
                self.assertTrue((out == 123.5).all())


if __name__ == "__main__":
    unittest.main()
