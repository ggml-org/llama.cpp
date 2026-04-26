#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path
from typing import Tuple


def skip(message: str) -> None:
    print(f"SKIP: {message}", file=sys.stderr)
    sys.exit(77)


try:
    import torch
except ModuleNotFoundError as exc:
    skip(f"missing dependency: {exc.name}")


if not hasattr(torch, "float8_e4m3fn"):
    skip("torch does not support float8_e4m3fn")


def load_converter(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", repo_root / "convert_hf_to_gguf.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create convert_hf_to_gguf module spec")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        skip(f"missing dependency: {exc.name}")
    return module


def make_u8(shape: Tuple[int, ...]) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= dim
    return (torch.arange(n, dtype=torch.int32) % 256).to(torch.uint8).reshape(shape)


def assert_rejects_float_scale(fn, weight: torch.Tensor, scale: torch.Tensor, name: str) -> None:
    try:
        fn(weight, scale.float(), name)
    except ValueError as exc:
        assert "scale dtype" in str(exc), str(exc)
    else:
        raise AssertionError(f"{name} accepted a multi-byte float scale")


def reference_pack_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    col_blocks = cols // 128

    weight_u8 = weight.view(torch.uint8)
    scale_u8 = scale.view(torch.uint8)
    out = torch.empty((rows, col_blocks, 129), dtype=torch.uint8)
    out[:, :, 0].copy_(scale_u8.repeat_interleave(128, dim=0))
    out[:, :, 1:].copy_(weight_u8.reshape(rows, col_blocks, 128))
    return out.reshape(rows, col_blocks * 129)


def test_pack_fp8(pack_fp8) -> None:
    rows, cols = 256, 384
    weight = make_u8((rows, cols)).view(torch.float8_e4m3fn)
    scale = make_u8((rows // 128, cols // 128))

    actual = pack_fp8(weight, scale, "fp8.weight")
    expected = reference_pack_fp8(weight, scale)
    assert torch.equal(actual, expected)
    assert actual.shape == (rows, (cols // 128) * 129)

    if hasattr(torch, "float8_e8m0fnu"):
        scale_e8 = scale.view(torch.float8_e8m0fnu)
        assert torch.equal(pack_fp8(weight, scale_e8, "fp8.e8.weight"), expected)

    assert_rejects_float_scale(pack_fp8, weight, scale, "fp8.float-scale.weight")


def reference_pack_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, packed_cols = weight.shape
    groups = packed_cols // 16

    hf = weight.view(torch.uint8).reshape(rows, groups, 16)
    vals = torch.empty((rows, groups, 32), dtype=torch.uint8)
    vals[:, :, 0::2].copy_(hf & 0x0F)
    vals[:, :, 1::2].copy_(hf >> 4)

    out = torch.empty((rows, groups, 17), dtype=torch.uint8)
    out[:, :, 0].copy_(scale.view(torch.uint8)[:, :groups])
    out[:, :, 1:].copy_(vals[:, :, :16] | (vals[:, :, 16:] << 4))
    return out.reshape(rows, groups * 17)


def test_pack_mxfp4(pack_mxfp4) -> None:
    rows, packed_cols = 5, 48
    weight = make_u8((rows, packed_cols)).view(torch.int8)
    scale = make_u8((rows, packed_cols // 16 + 2))

    actual = pack_mxfp4(weight, scale, "experts.weight")
    expected = reference_pack_mxfp4(weight, scale)
    assert torch.equal(actual, expected)
    assert actual.shape == (rows, (packed_cols // 16) * 17)

    if hasattr(torch, "float8_e8m0fnu"):
        scale_e8 = scale.view(torch.float8_e8m0fnu)
        assert torch.equal(pack_mxfp4(weight, scale_e8, "experts.e8.weight"), expected)

    assert_rejects_float_scale(pack_mxfp4, weight, scale, "experts.float-scale.weight")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    converter = load_converter(repo_root)

    test_pack_fp8(converter.DeepseekV4Model._pack_fp8_e4m3_b128)
    test_pack_mxfp4(converter.DeepseekV4Model._pack_mxfp4)


if __name__ == "__main__":
    main()
