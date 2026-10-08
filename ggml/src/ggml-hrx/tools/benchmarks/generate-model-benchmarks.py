#!/usr/bin/env python3
#
# Generate model-scoped Loom benchmarks from HRX command program dumps.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HRX_DIR = SCRIPT_DIR.parents[1]
BENCHMARK_DIR = HRX_DIR / "benchmarks"
KERNEL_CORPUS_DIR = HRX_DIR / "kernel-corpus" / "kernels"
KERNEL_TARGET_RE = re.compile(
    r"kernel\.def\s+target\(@(?P<target>[A-Za-z0-9_.$-]+)\)"
    r"(?:\s+export\(\"[^\"]+\"\))?\s+@(?P<symbol>[A-Za-z0-9_.$-]+)\s*\("
)


@dataclass
class ScenarioBenchmarks:
    scenario: str
    command_count: int
    dispatch_count: int
    generated_count: int
    kernel_counts: dict[str, int]
    dispatches: list[dict[str, Any]]
    cases: list[str]
    used_exports: dict[str, dict[str, Any]]


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def sanitize_symbol(text: str) -> str:
    text = text.split(":")[-1]
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "unknown"
    if text[0].isdigit():
        return "k_" + text
    return text


def int_param(command: dict[str, Any], name: str, default: int | None = None) -> int:
    params = command.get("integer_parameters", {})
    if name in params:
        return int(params[name])
    if default is not None:
        return default
    fail(f"command {command.get('ordinal')} {command.get('kernel')} is missing integer parameter {name}")


def config_int(command: dict[str, Any], name: str, default: int | None = None) -> int:
    params = command.get("compile_parameters", {})
    if name in params:
        return int(params[name])
    if default is not None:
        return default
    fail(f"command {command.get('ordinal')} {command.get('kernel')} is missing compile parameter {name}")


def binding_length(command: dict[str, Any], name: str) -> int:
    for binding in command.get("bindings", []):
        if binding.get("name") == name:
            return int(binding["length"])
    fail(f"command {command.get('ordinal')} {command.get('kernel')} is missing binding {name}")


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def tensor_type_for_format(format_value: int) -> str | None:
    if format_value == 16:
        return "f16"
    if format_value == 30:
        return "bf16"
    if format_value == 32:
        return "f32"
    return None


def fill_tensor(name: str, value: str, shape: str, dtype: str) -> str:
    return f"  %{name} = check.generate.fill value({value}) : tensor<{shape}x{dtype}>"


def iota_tensor(name: str, offset: str, step: str, shape: str, dtype: str, period: int | None = None) -> str:
    period_text = "" if period is None else f" period({period})"
    return f"  %{name} = check.generate.iota offset({offset}) step({step}){period_text} : tensor<{shape}x{dtype}>"


def case_binary(symbol: str, command: dict[str, Any]) -> str:
    count = int_param(command, "element_count")
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %element_count = check.literal value({count}) : index",
            fill_tensor("lhs", "2.0", f"{count}", "f32"),
            fill_tensor("rhs", "3.0", f"{count}", "f32"),
            fill_tensor("output", "0.0", f"{count}", "f32"),
            f"  kernel.launch @ggml_binary_f32[%element_count](%element_count, %lhs, %rhs, %output) : [index](index, tensor<{count}xf32>, tensor<{count}xf32>, tensor<{count}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_rmsnorm_binary(symbol: str, command: dict[str, Any]) -> str:
    token_count = int_param(command, "token_count")
    hidden_size = config_int(command, "ggml.rmsnorm_binary_f32.hidden_size")
    shape = f"{token_count}x{hidden_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("input", "2.0", shape, "f32"),
            fill_tensor("rhs", "3.0", f"{hidden_size}", "f32"),
            fill_tensor("output", "0.0", shape, "f32"),
            f"  kernel.launch @ggml_rmsnorm_binary_f32[%token_count](%token_count, %input, %rhs, %output) : [index](index, tensor<{shape}xf32>, tensor<{hidden_size}xf32>, tensor<{shape}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_mul_mat(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    if kernel == "ggml_mul_mat_f32_f32_decode_wave64":
        input_size = int_param(command, "input_size")
        output_size = int_param(command, "output_size")
        accumulation = 0
        weight_format = config_int(command, "ggml.mul_mat_f32_f32_decode.weight_format")
    else:
        input_size = config_int(command, "ggml.mul_mat.input_size")
        output_size = config_int(command, "ggml.mul_mat.output_size")
        accumulation = config_int(command, "ggml.mul_mat.output_accumulation", 0)
        weight_format = config_int(command, "ggml.mul_mat.weight_format")
    weight_type = tensor_type_for_format(weight_format)
    if weight_type is None:
        return None
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    shape_out = f"{token_count}x{output_size}"
    output_init = 1.0 if accumulation else 0.0
    if kernel == "ggml_mul_mat_f32_f32_decode_wave64":
        launch_params = "%token_count, %input_size, %output_size"
        workload_params = "%token_count, %input_size, %output_size"
        workload_types = "[index, index, index]"
        launch_types = "index, index, index"
        literals = [
            f"  %token_count = check.literal value({token_count}) : index",
            f"  %input_size = check.literal value({input_size}) : index",
            f"  %output_size = check.literal value({output_size}) : index",
        ]
    else:
        launch_params = "%token_count"
        workload_params = "%token_count"
        workload_types = "[index]"
        launch_types = "index"
        literals = [f"  %token_count = check.literal value({token_count}) : index"]
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            *literals,
            fill_tensor("input", "1.0", shape_in, "f32"),
            fill_tensor("weight", "1.0", shape_weight, weight_type),
            fill_tensor("output", f"{output_init:.1f}", shape_out, "f32"),
            f"  kernel.launch @{kernel}[{workload_params}]({launch_params}, %input, %weight, %output) : {workload_types}({launch_types}, tensor<{shape_in}xf32>, tensor<{shape_weight}x{weight_type}>, tensor<{shape_out}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_mul_mat_add_decode(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = int_param(command, "input_size")
    output_size = int_param(command, "output_size")
    weight_format = config_int(command, "ggml.mul_mat_f32_f32_decode.weight_format")
    weight_type = tensor_type_for_format(weight_format)
    if weight_type is None:
        return None

    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    shape_out = f"{token_count}x{output_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            f"  %input_size = check.literal value({input_size}) : index",
            f"  %output_size = check.literal value({output_size}) : index",
            fill_tensor("input", "1.0", shape_in, "f32"),
            fill_tensor("weight", "1.0", shape_weight, weight_type),
            fill_tensor("residual_input", "0.25", shape_out, "f32"),
            fill_tensor("residual_output", "0.0", shape_out, "f32"),
            f"  kernel.launch @{kernel}[%token_count, %input_size, %output_size](%token_count, %input_size, %output_size, %input, %weight, %residual_input, %residual_output) : [index, index, index](index, index, index, tensor<{shape_in}xf32>, tensor<{shape_weight}x{weight_type}>, tensor<{shape_out}xf32>, tensor<{shape_out}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_mul_mat_postops(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = config_int(command, "ggml.mul_mat_postops.input_size")
    output_size = config_int(command, "ggml.mul_mat_postops.output_size")
    weight_format = config_int(command, "ggml.mul_mat_postops.weight_format")
    weight_type = tensor_type_for_format(weight_format)
    if weight_type is None:
        return None

    has_bias = "bias" in kernel
    has_residual = "add" in kernel
    has_rmsnorm = "next_rmsnorm" in kernel
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    shape_out = f"{token_count}x{output_size}"
    completion_count = (token_count + 31) // 32

    tensors = [
        fill_tensor("input", "1.0", shape_in, "f32"),
        fill_tensor("weight", "1.0", shape_weight, weight_type),
    ]
    launch_values = ["%token_count", "%input", "%weight"]
    launch_types = ["index", f"tensor<{shape_in}xf32>", f"tensor<{shape_weight}x{weight_type}>"]

    if has_bias:
        tensors.append(fill_tensor("bias", "0.5", f"{output_size}", "f32"))
        launch_values.append("%bias")
        launch_types.append(f"tensor<{output_size}xf32>")
    if has_residual:
        tensors.append(fill_tensor("residual_input", "0.25", shape_out, "f32"))
        tensors.append(fill_tensor("residual_output", "0.0", shape_out, "f32"))
        launch_values.extend(["%residual_input", "%residual_output"])
        launch_types.extend([f"tensor<{shape_out}xf32>", f"tensor<{shape_out}xf32>"])
    else:
        tensors.append(fill_tensor("output", "0.0", shape_out, "f32"))
        launch_values.append("%output")
        launch_types.append(f"tensor<{shape_out}xf32>")

    if has_rmsnorm:
        tensors.append(fill_tensor("norm_weight", "1.0", f"{output_size}", "f32"))
        tensors.append(fill_tensor("normalized_output", "0.0", shape_out, "f32"))
        tensors.append(fill_tensor("completion_counters", "0", f"{completion_count}", "i32"))
        launch_values.extend(["%norm_weight", "%normalized_output", "%completion_counters"])
        launch_types.extend([f"tensor<{output_size}xf32>", f"tensor<{shape_out}xf32>", f"tensor<{completion_count}xi32>"])

    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            *tensors,
            f"  kernel.launch @{kernel}[%token_count]({', '.join(launch_values)}) : [index]({', '.join(launch_types)})",
            "  check.return",
            "}",
        ]
    )


def case_swiglu(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = config_int(command, "ggml.mul_mat_swiglu.input_size")
    output_size = config_int(command, "ggml.mul_mat_swiglu.output_size")
    gate_format = config_int(command, "ggml.mul_mat_swiglu.gate_weight_format")
    up_format = config_int(command, "ggml.mul_mat_swiglu.up_weight_format")
    gate_type = tensor_type_for_format(gate_format)
    up_type = tensor_type_for_format(up_format)
    if gate_type is None or up_type is None:
        return None
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    shape_out = f"{token_count}x{output_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("input", "0.0", shape_in, "f32"),
            fill_tensor("gate_weight", "1.0", shape_weight, gate_type),
            fill_tensor("up_weight", "1.0", shape_weight, up_type),
            fill_tensor("output", "1.0", shape_out, "f32"),
            f"  kernel.launch @{kernel}[%token_count](%token_count, %input, %gate_weight, %up_weight, %output) : [index](index, tensor<{shape_in}xf32>, tensor<{shape_weight}x{gate_type}>, tensor<{shape_weight}x{up_type}>, tensor<{shape_out}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_llm_attention_q_matmul_rope(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = config_int(command, "llm.attention_qkv.input_size")
    output_size = config_int(command, "llm.attention_qkv.output_size")
    weight_format = config_int(command, "llm.attention_qkv.weight_format")
    head_size = config_int(command, "llm.attention_qkv.head_size")
    head_count = config_int(command, "llm.attention_qkv.head_count")
    weight_type = tensor_type_for_format(weight_format)
    if weight_type is None:
        return None
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    shape_out = f"{token_count}x{head_count}x{head_size}"
    half_head_size = head_size // 2
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("input", "1.0", shape_in, "f32"),
            fill_tensor("weight", "1.0", shape_weight, weight_type),
            fill_tensor("positions", "0", f"{token_count}", "i32"),
            fill_tensor("theta", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("freq_factors", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("output", "0.0", shape_out, "f32"),
            f"  kernel.launch @{kernel}[%token_count](%token_count, %input, %weight, %positions, %theta, %freq_factors, %output) : [index](index, tensor<{shape_in}xf32>, tensor<{shape_weight}x{weight_type}>, tensor<{token_count}xi32>, tensor<{half_head_size}xf32>, tensor<{half_head_size}xf32>, tensor<{shape_out}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_llm_attention_k_matmul_rope_set_rows(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = config_int(command, "llm.attention_qkv.input_size")
    output_size = config_int(command, "llm.attention_qkv.output_size")
    weight_format = config_int(command, "llm.attention_qkv.weight_format")
    head_size = config_int(command, "llm.attention_qkv.head_size")
    head_count = config_int(command, "llm.attention_qkv.head_count")
    cache_rows = config_int(command, "llm.attention_qkv.cache_row_count")
    cache_format = config_int(command, "llm.attention_qkv.cache_output_format")
    weight_type = tensor_type_for_format(weight_format)
    cache_type = tensor_type_for_format(cache_format)
    if weight_type is None or cache_type is None:
        return None
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    cache_shape = f"{cache_rows}x{head_count}x{head_size}"
    half_head_size = head_size // 2
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("input", "1.0", shape_in, "f32"),
            fill_tensor("weight", "1.0", shape_weight, weight_type),
            fill_tensor("positions", "0", f"{token_count}", "i32"),
            iota_tensor("indices", "0", "1", f"{token_count}", "i64", period=cache_rows),
            fill_tensor("theta", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("freq_factors", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("cache", "0.0", cache_shape, cache_type),
            f"  kernel.launch @{kernel}[%token_count](%token_count, %input, %weight, %positions, %indices, %theta, %freq_factors, %cache) : [index](index, tensor<{shape_in}xf32>, tensor<{shape_weight}x{weight_type}>, tensor<{token_count}xi32>, tensor<{token_count}xi64>, tensor<{half_head_size}xf32>, tensor<{half_head_size}xf32>, tensor<{cache_shape}x{cache_type}>)",
            "  check.return",
            "}",
        ]
    )


def case_llm_attention_v_matmul_set_rows(symbol: str, command: dict[str, Any], kernel: str) -> str | None:
    token_count = int_param(command, "token_count")
    input_size = config_int(command, "llm.attention_qkv.input_size")
    output_size = config_int(command, "llm.attention_qkv.output_size")
    weight_format = config_int(command, "llm.attention_qkv.weight_format")
    cache_rows = config_int(command, "llm.attention_qkv.cache_row_count")
    cache_format = config_int(command, "llm.attention_qkv.cache_output_format")
    weight_type = tensor_type_for_format(weight_format)
    cache_type = tensor_type_for_format(cache_format)
    if weight_type is None or cache_type is None:
        return None
    shape_in = f"{token_count}x{input_size}"
    shape_weight = f"{output_size}x{input_size}"
    cache_shape = f"{cache_rows}x{output_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("input", "1.0", shape_in, "f32"),
            fill_tensor("weight", "1.0", shape_weight, weight_type),
            iota_tensor("indices", "0", "1", f"{token_count}", "i64", period=cache_rows),
            fill_tensor("cache", "0.0", cache_shape, cache_type),
            f"  kernel.launch @{kernel}[%token_count](%token_count, %input, %weight, %indices, %cache) : [index](index, tensor<{shape_in}xf32>, tensor<{shape_weight}x{weight_type}>, tensor<{token_count}xi64>, tensor<{cache_shape}x{cache_type}>)",
            "  check.return",
            "}",
        ]
    )


def case_flash_attention(symbol: str, command: dict[str, Any], kernel: str, prefix: str) -> str:
    query_tokens = int_param(command, "query_token_count")
    kv_tokens = int_param(command, "key_value_token_count")
    if prefix == "ggml":
        query_heads = config_int(command, "ggml.flash_attention.query_head_count")
        kv_heads = config_int(command, "ggml.flash_attention.key_value_head_count")
        legacy_head_size = config_int(command, "ggml.flash_attention.head_size", 0)
        qk_head_size = config_int(command, "ggml.flash_attention.qk_head_size", legacy_head_size)
        value_head_size = config_int(command, "ggml.flash_attention.value_head_size", qk_head_size)
    else:
        query_heads = config_int(command, f"{prefix}.attention.query_head_count")
        kv_heads = config_int(command, f"{prefix}.attention.key_value_head_count")
        qk_head_size = 128
        value_head_size = 128
    query_shape = f"{query_tokens}x{query_heads}x{qk_head_size}"
    key_shape = f"{kv_tokens}x{kv_heads}x{qk_head_size}"
    value_shape = f"{kv_tokens}x{kv_heads}x{value_head_size}"
    output_shape = f"{query_tokens}x{query_heads}x{value_head_size}"
    mask_shape = f"{query_tokens}x{kv_tokens}"
    if prefix == "ggml":
        gate_bytes = binding_length(command, "gate")
        if gate_bytes % 4 != 0:
            fail(
                f"command {command.get('ordinal')} {command.get('kernel')} has "
                f"non-f32 gate binding length {gate_bytes}"
            )
        gate_elements = gate_bytes // 4
        apply_gate = config_int(command, "ggml.flash_attention.apply_gate", 0)
        if apply_gate:
            gate_stride_head = config_int(command, "ggml.flash_attention.gate_stride_head")
            gate_stride_token = config_int(command, "ggml.flash_attention.gate_stride_token")
            required_gate_elements = (
                (query_tokens - 1) * gate_stride_token
                + (query_heads - 1) * gate_stride_head
                + value_head_size
            )
            if gate_elements < required_gate_elements:
                fail(
                    f"command {command.get('ordinal')} {command.get('kernel')} gate binding "
                    f"has {gate_elements} f32 elements but the strided access requires "
                    f"{required_gate_elements}"
                )
        launch = (
            f"  kernel.launch @{kernel}[%query_token_count, %key_value_token_count]"
            f"(%query_token_count, %key_value_token_count, %query, %key, %value, %mask, %gate, %output)"
            f" : [index, index](index, index, tensor<{query_shape}xf32>, tensor<{key_shape}xf16>, "
            f"tensor<{value_shape}xf16>, tensor<{mask_shape}xf16>, tensor<{gate_elements}xf32>, "
            f"tensor<{output_shape}xf32>)"
        )
    else:
        launch = (
            f"  kernel.launch @{kernel}[%query_token_count, %key_value_token_count]"
            f"(%query_token_count, %key_value_token_count, %query, %key, %value, %mask, %output)"
            f" : [index, index](index, index, tensor<{query_shape}xf32>, tensor<{key_shape}xf16>, "
            f"tensor<{value_shape}xf16>, tensor<{mask_shape}xf16>, tensor<{output_shape}xf32>)"
        )
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %query_token_count = check.literal value({query_tokens}) : index",
            f"  %key_value_token_count = check.literal value({kv_tokens}) : index",
            fill_tensor("query", "0.0", query_shape, "f32"),
            fill_tensor("key", "0.0", key_shape, "f16"),
            fill_tensor("value", "0.0", value_shape, "f16"),
            fill_tensor("mask", "0.0", mask_shape, "f16"),
            *([fill_tensor("gate", "0.0", f"{gate_elements}", "f32")] if prefix == "ggml" else []),
            fill_tensor("output", "1.0", output_shape, "f32"),
            launch,
            "  check.return",
            "}",
        ]
    )


def case_flash_attention_decode_split(symbol: str, command: dict[str, Any], kernel: str) -> str:
    kv_tokens = int_param(command, "key_value_token_count")
    kv_capacity = config_int(command, "ggml.flash_attention.decode.key_value_token_capacity")
    query_heads = config_int(command, "ggml.flash_attention.query_head_count")
    kv_heads = config_int(command, "ggml.flash_attention.key_value_head_count")
    legacy_head_size = config_int(command, "ggml.flash_attention.head_size", 0)
    qk_head_size = config_int(command, "ggml.flash_attention.qk_head_size", legacy_head_size)
    value_head_size = config_int(command, "ggml.flash_attention.value_head_size", qk_head_size)
    partial_blocks = ceil_div(kv_capacity, 64)
    query_shape = f"{query_heads}x{qk_head_size}"
    key_shape = f"{kv_tokens}x{kv_heads}x{qk_head_size}"
    value_shape = f"{kv_tokens}x{kv_heads}x{value_head_size}"
    mask_shape = f"{kv_tokens}"
    partial_shape = f"{kv_heads}x{partial_blocks}x16"
    partial_output_shape = f"{kv_heads}x{partial_blocks}x16x{value_head_size}"
    completion_shape = f"{kv_heads}"
    output_shape = f"{query_heads}x{value_head_size}"
    tensors = [
        fill_tensor("query", "0.0", query_shape, "f32"),
        fill_tensor("key", "0.0", key_shape, "f16"),
        fill_tensor("value", "0.0", value_shape, "f16"),
        fill_tensor("mask", "0.0", mask_shape, "f16"),
        fill_tensor("partial_max", "0.0", partial_shape, "f32"),
        fill_tensor("partial_sum", "0.0", partial_shape, "f32"),
        fill_tensor("partial_output", "0.0", partial_output_shape, "f16"),
        fill_tensor("completion_counter", "0", completion_shape, "i32"),
        fill_tensor("output", "1.0", output_shape, "f32"),
    ]
    launch_values = [
        "%key_value_token_count",
        "%query",
        "%key",
        "%value",
        "%mask",
        "%partial_max",
        "%partial_sum",
        "%partial_output",
        "%completion_counter",
        "%output",
    ]
    launch_types = [
        "index",
        f"tensor<{query_shape}xf32>",
        f"tensor<{key_shape}xf16>",
        f"tensor<{value_shape}xf16>",
        f"tensor<{mask_shape}xf16>",
        f"tensor<{partial_shape}xf32>",
        f"tensor<{partial_shape}xf32>",
        f"tensor<{partial_output_shape}xf16>",
        f"tensor<{completion_shape}xi32>",
        f"tensor<{output_shape}xf32>",
    ]
    if kernel.endswith("_next_q8"):
        next_q8_bytes = binding_length(command, "next_q8_output")
        tensors.append(fill_tensor("next_q8_output", "0", f"{next_q8_bytes}", "i8"))
        launch_values.append("%next_q8_output")
        launch_types.append(f"tensor<{next_q8_bytes}xi8>")
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %key_value_token_count = check.literal value({kv_tokens}) : index",
            *tensors,
            f"  kernel.launch @{kernel}[%key_value_token_count]({', '.join(launch_values)}) : [index]({', '.join(launch_types)})",
            "  check.return",
            "}",
        ]
    )


def case_rope(symbol: str, command: dict[str, Any], kernel: str, prefix: str) -> str:
    token_count = int_param(command, "token_count")
    head_size = config_int(command, f"{prefix}.head_size")
    head_count = config_int(command, f"{prefix}.head_count")
    half_head_size = head_size // 2
    data_shape = f"{token_count}x{head_count}x{head_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            fill_tensor("positions", "0", f"{token_count}", "i32"),
            fill_tensor("input", "0.0", data_shape, "f32"),
            fill_tensor("theta", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("freq_factors", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("output", "1.0", data_shape, "f32"),
            f"  kernel.launch @{kernel}[%token_count](%token_count, %positions, %input, %theta, %freq_factors, %output) : [index](index, tensor<{token_count}xi32>, tensor<{data_shape}xf32>, tensor<{half_head_size}xf32>, tensor<{half_head_size}xf32>, tensor<{data_shape}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_rope_set_rows(symbol: str, command: dict[str, Any]) -> str | None:
    token_count = int_param(command, "token_count")
    cache_rows = int_param(command, "cache_row_count")
    head_size = config_int(command, "ggml.rope_set_rows_f32.head_size")
    head_count = config_int(command, "ggml.rope_set_rows_f32.head_count")
    output_format = config_int(command, "ggml.rope_set_rows_f32.output_format")
    output_type = tensor_type_for_format(output_format)
    if output_type is None:
        return None
    half_head_size = head_size // 2
    input_shape = f"{token_count}x{head_count}x{head_size}"
    cache_shape = f"{cache_rows}x{head_count}x{head_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            f"  %cache_row_count = check.literal value({cache_rows}) : index",
            fill_tensor("positions", "0", f"{token_count}", "i32"),
            iota_tensor("indices", "0", "1", f"{token_count}", "i64", period=cache_rows),
            fill_tensor("input", "0.0", input_shape, "f32"),
            fill_tensor("theta", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("freq_factors", "0.0", f"{half_head_size}", "f32"),
            fill_tensor("cache", "0.0", cache_shape, output_type),
            f"  kernel.launch @ggml_rope_set_rows_f32[%token_count, %cache_row_count](%token_count, %cache_row_count, %positions, %indices, %input, %theta, %freq_factors, %cache) : [index, index](index, index, tensor<{token_count}xi32>, tensor<{token_count}xi64>, tensor<{input_shape}xf32>, tensor<{half_head_size}xf32>, tensor<{half_head_size}xf32>, tensor<{cache_shape}x{output_type}>)",
            "  check.return",
            "}",
        ]
    )


def case_set_rows(symbol: str, command: dict[str, Any]) -> str | None:
    token_count = int_param(command, "token_count")
    cache_rows = int_param(command, "cache_row_count")
    hidden_size = int_param(command, "hidden_size")
    input_format = config_int(command, "ggml.set_rows.input_format")
    output_format = config_int(command, "ggml.set_rows.output_format")
    input_type = tensor_type_for_format(input_format)
    output_type = tensor_type_for_format(output_format)
    if input_type is None or output_type is None:
        return None
    rows_shape = f"{token_count}x{hidden_size}"
    cache_shape = f"{cache_rows}x{hidden_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            f"  %cache_row_count = check.literal value({cache_rows}) : index",
            f"  %hidden_size = check.literal value({hidden_size}) : index",
            fill_tensor("rows", "0.0", rows_shape, input_type),
            iota_tensor("indices", "0", "1", f"{token_count}", "i64", period=cache_rows),
            fill_tensor("cache", "0.0", cache_shape, output_type),
            f"  kernel.launch @ggml_set_rows[%token_count, %cache_row_count, %hidden_size](%token_count, %cache_row_count, %hidden_size, %rows, %indices, %cache) : [index, index, index](index, index, index, tensor<{rows_shape}x{input_type}>, tensor<{token_count}xi64>, tensor<{cache_shape}x{output_type}>)",
            "  check.return",
            "}",
        ]
    )


def case_get_rows(symbol: str, command: dict[str, Any]) -> str | None:
    token_count = int_param(command, "token_count")
    row_count = int_param(command, "row_count")
    hidden_size = int_param(command, "hidden_size")
    weight_format = config_int(command, "ggml.get_rows_f32.weight_format")
    weight_type = tensor_type_for_format(weight_format)
    if weight_type is None:
        return None
    output_shape = f"{token_count}x{hidden_size}"
    weight_shape = f"{row_count}x{hidden_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %token_count = check.literal value({token_count}) : index",
            f"  %row_count = check.literal value({row_count}) : index",
            f"  %hidden_size = check.literal value({hidden_size}) : index",
            fill_tensor("token_ids", "0", f"{token_count}", "i32"),
            fill_tensor("weight", "0.0", weight_shape, weight_type),
            fill_tensor("output", "1.0", output_shape, "f32"),
            f"  kernel.launch @ggml_get_rows_f32[%token_count, %row_count, %hidden_size](%token_count, %row_count, %hidden_size, %token_ids, %weight, %output) : [index, index, index](index, index, index, tensor<{token_count}xi32>, tensor<{weight_shape}x{weight_type}>, tensor<{output_shape}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_gather_add(symbol: str, command: dict[str, Any]) -> str:
    source_tokens = int_param(command, "source_token_count")
    output_tokens = int_param(command, "output_token_count")
    hidden_size = int_param(command, "hidden_size")
    source_shape = f"{source_tokens}x{hidden_size}"
    output_shape = f"{output_tokens}x{hidden_size}"
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            f"  %source_token_count = check.literal value({source_tokens}) : index",
            f"  %output_token_count = check.literal value({output_tokens}) : index",
            f"  %hidden_size = check.literal value({hidden_size}) : index",
            fill_tensor("attention", "0.0", source_shape, "f32"),
            fill_tensor("residual", "0.0", source_shape, "f32"),
            fill_tensor("output_ids", "0", f"{output_tokens}", "i32"),
            fill_tensor("output", "1.0", output_shape, "f32"),
            f"  kernel.launch @ggml_gather_add_f32[%source_token_count, %output_token_count, %hidden_size](%source_token_count, %output_token_count, %hidden_size, %attention, %residual, %output_ids, %output) : [index, index, index](index, index, index, tensor<{source_shape}xf32>, tensor<{source_shape}xf32>, tensor<{output_tokens}xi32>, tensor<{output_shape}xf32>)",
            "  check.return",
            "}",
        ]
    )


def case_generic(symbol: str, command: dict[str, Any], export: dict[str, Any]) -> str | None:
    integer_parameters = command.get("integer_parameters", {})
    binding_lengths = {
        binding.get("name"): int(binding["length"])
        for binding in command.get("bindings", [])
        if binding.get("name") is not None and "length" in binding
    }
    workload_parameters = list(export.get("workload_parameters", []))
    launch_parameters = list(export.get("launch_parameters", []))
    bindings = list(export.get("bindings", []))

    scalar_names: list[str] = []
    for parameter in [*workload_parameters, *launch_parameters]:
        name = parameter["name"]
        if name not in scalar_names:
            scalar_names.append(name)
    missing_scalars = [name for name in scalar_names if name not in integer_parameters]
    missing_bindings = [name for name in bindings if name not in binding_lengths]
    if missing_scalars or missing_bindings:
        return None

    literals = [
        f"  %{name} = check.literal value({int(integer_parameters[name])}) : index"
        for name in scalar_names
    ]
    tensors = [
        fill_tensor(name, "0", f"{binding_lengths[name]}", "i8")
        for name in bindings
    ]
    workload_values = ", ".join(f"%{parameter['name']}" for parameter in workload_parameters)
    launch_values = [f"%{parameter['name']}" for parameter in launch_parameters]
    launch_values.extend(f"%{name}" for name in bindings)
    workload_types = ", ".join("index" for _ in workload_parameters)
    launch_types = ["index" for _ in launch_parameters]
    launch_types.extend(f"tensor<{binding_lengths[name]}xi8>" for name in bindings)
    return "\n".join(
        [
            f"check.case public @{symbol}_case {{",
            *literals,
            *tensors,
            f"  kernel.launch @{export['symbol']}[{workload_values}]({', '.join(launch_values)}) : [{workload_types}]({', '.join(launch_types)})",
            "  check.return",
            "}",
        ]
    )


def render_case(symbol: str, command: dict[str, Any], export: dict[str, Any]) -> str | None:
    kernel = command["kernel"].split(":")[-1]
    case: str | None
    if kernel == "ggml_binary_f32":
        case = case_binary(symbol, command)
    elif kernel == "ggml_rmsnorm_binary_f32":
        case = case_rmsnorm_binary(symbol, command)
    elif kernel in ("ggml_mul_mat_f32_f32_wmma", "ggml_mul_mat_f32_f32_decode_wave64"):
        case = case_mul_mat(symbol, command, kernel)
    elif kernel == "ggml_mul_mat_add_f32_f32_decode_wave64":
        case = case_mul_mat_add_decode(symbol, command, kernel)
    elif kernel in (
        "ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
        "ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
    ):
        case = case_mul_mat_postops(symbol, command, kernel)
    elif kernel in ("ggml_mul_mat_swiglu_f32_f32_wmma", "ggml_mul_mat_swiglu_f32_f32_decode_wave64"):
        case = case_swiglu(symbol, command, kernel)
    elif kernel in ("llm_attention_q_matmul_rope_f32_f32_wmma", "llm_attention_q_matmul_rope_decode_f32_f32"):
        case = case_llm_attention_q_matmul_rope(symbol, command, kernel)
    elif kernel in (
        "llm_attention_k_matmul_rope_set_rows_f32_f32_wmma",
        "llm_attention_k_matmul_rope_set_rows_decode_f32_f32",
    ):
        case = case_llm_attention_k_matmul_rope_set_rows(symbol, command, kernel)
    elif kernel in ("llm_attention_v_matmul_set_rows_f32_f32_wmma", "llm_attention_v_matmul_set_rows_decode_f32_f32"):
        case = case_llm_attention_v_matmul_set_rows(symbol, command, kernel)
    elif kernel == "ggml_flash_attention_f32_f16_wmma":
        case = case_flash_attention(symbol, command, kernel, "ggml")
    elif kernel in ("ggml_flash_attention_decode_split_f32_f16_wmma", "ggml_flash_attention_decode_split_f32_f16_wmma_next_q8"):
        case = case_flash_attention_decode_split(symbol, command, kernel)
    elif kernel == "ggml_rope_f32":
        case = case_rope(symbol, command, kernel, "ggml.rope_f32")
    elif kernel == "ggml_rope_set_rows_f32":
        case = case_rope_set_rows(symbol, command)
    elif kernel == "ggml_set_rows":
        case = case_set_rows(symbol, command)
    elif kernel == "ggml_get_rows_f32":
        case = case_get_rows(symbol, command)
    elif kernel == "ggml_gather_add_f32":
        case = case_gather_add(symbol, command)
    else:
        case = None
    if case is not None:
        return case
    return case_generic(symbol, command, export)


def loom_target_for_export(corpus_dir: Path, export: dict[str, Any]) -> str:
    source = corpus_dir / export["source"]
    if not source.is_file():
        return ""
    text = read_text(source)
    symbol = export["symbol"]
    for match in KERNEL_TARGET_RE.finditer(text):
        if match.group("symbol") == symbol:
            return match.group("target")
    return ""


def load_exports() -> dict[str, dict[str, Any]]:
    exports: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(KERNEL_CORPUS_DIR.glob("*/manifest.json")):
        manifest = load_json(manifest_path)
        corpus_dir = manifest_path.parent
        for export in manifest.get("exports", []):
            name = export.get("name")
            if not name:
                continue
            export = dict(export)
            export["corpus_dir"] = str(corpus_dir.relative_to(HRX_DIR))
            export["loom_target"] = loom_target_for_export(corpus_dir, export)
            previous = exports.get(name)
            if previous is None or previous.get("target_selector") and not export.get("target_selector"):
                exports[name] = export
    return exports


def render_decl(export: dict[str, Any]) -> str:
    symbol = export["symbol"]
    workload = ", ".join(f"%{param['name']}: {param['type']}" for param in export.get("workload_parameters", []))
    launch_items = [f"%{param['name']}: {param['type']}" for param in export.get("launch_parameters", [])]
    launch_items += [f"%{name}: buffer" for name in export.get("bindings", [])]
    launch = ", ".join(launch_items)
    target = export.get("loom_target", "")
    target_attr = "" if not target else f" target(@{target})"
    return f"kernel.decl{target_attr} @{symbol}({workload}) launch({launch})"


def command_shape_key(command: dict[str, Any]) -> str:
    shape_data = {
        "kernel": command.get("kernel"),
        "integer_parameters": command.get("integer_parameters", {}),
        "compile_parameters": command.get("compile_parameters", {}),
        "binding_lengths": [binding.get("length") for binding in command.get("bindings", [])],
    }
    encoded = json.dumps(shape_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_program_commands(dump_dir: Path) -> list[dict[str, Any]]:
    program_paths = sorted(dump_dir.glob("program-*/program.json"))
    if not program_paths:
        fail(f"no program.json files found under {dump_dir}")
    commands: list[dict[str, Any]] = []
    for program_path in program_paths:
        program = load_json(program_path)
        for command in program.get("commands", []):
            command = dict(command)
            command["program"] = {
                "directory": program_path.parent.name,
                "dump_id": program.get("dump_id"),
                "shape_hash": program.get("shape_hash"),
                "target": program.get("target"),
            }
            commands.append(command)
    return commands


def source_paths_for_export(export: dict[str, Any]) -> list[str]:
    recipe = export.get("compile_recipe", {})
    paths = []
    for source in recipe.get("primary_sources", []):
        paths.append(source)
    for source in recipe.get("library_sources", []):
        paths.append(source)
    return list(dict.fromkeys(paths))


def primary_source_paths_for_export(export: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(export.get("compile_recipe", {}).get("primary_sources", [])))


def library_source_paths_for_export(export: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(export.get("compile_recipe", {}).get("library_sources", [])))


def generate_scenario(model_slug: str,
                      scenario: str,
                      commands: list[dict[str, Any]],
                      exports: dict[str, dict[str, Any]]) -> ScenarioBenchmarks:
    dispatch_entries: list[dict[str, Any]] = []
    cases: list[str] = []
    used_exports: dict[str, dict[str, Any]] = {}
    shape_keys = [command_shape_key(command) for command in commands]
    shape_counts = Counter(shape_keys)
    shape_commands: dict[str, list[dict[str, Any]]] = {}
    for command, shape_key in zip(commands, shape_keys, strict=True):
        shape_commands.setdefault(shape_key, []).append(command)
    per_kernel_counts = Counter(command["kernel"] for command in commands)

    for shape_key, represented_commands in shape_commands.items():
        command = represented_commands[0]
        kernel_name = command["kernel"]
        export_name = kernel_name.split(":")[-1]
        export = exports.get(export_name)
        if export is None:
            continue
        else:
            used_exports[export_name] = export
            symbol = f"{model_slug}_{scenario}_{len(dispatch_entries):03d}_{sanitize_symbol(export_name)}"
            try:
                case = render_case(symbol, command, export)
            except SystemExit as exc:
                case = None
                print(f"skipped {kernel_name}: {exc}")
            else:
                if case is None:
                    print(f"skipped {kernel_name}: unsupported template")
            if case is not None:
                cases.append(case)
                benchmark_symbol = "@" + symbol
            else:
                continue

        dispatch_entries.append(
            {
                "benchmark": benchmark_symbol,
                "kernel": kernel_name,
                "symbol": export["symbol"],
                "count": shape_counts[shape_key],
                "integer_parameters": command.get("integer_parameters", {}),
                "compile_parameters": command.get("compile_parameters", {}),
                "workload_parameters": export.get("workload_parameters", []),
                "sources": source_paths_for_export(export),
                "primary_sources": primary_source_paths_for_export(export),
                "library_sources": library_source_paths_for_export(export),
                "corpus_dir": export.get("corpus_dir"),
            }
        )

    return ScenarioBenchmarks(
        scenario=scenario,
        command_count=len(commands),
        dispatch_count=len(shape_commands),
        generated_count=len(dispatch_entries),
        kernel_counts=dict(sorted(per_kernel_counts.items())),
        dispatches=dispatch_entries,
        cases=cases,
        used_exports=used_exports,
    )


def write_model_loom(model_file: Path,
                     model_slug: str,
                     scenarios: list[ScenarioBenchmarks],
                     used_exports: dict[str, dict[str, Any]]) -> None:
    scenario_names = ", ".join(scenario.scenario for scenario in scenarios)
    targets = sorted({
        export["loom_target"]
        for export in used_exports.values()
        if export.get("loom_target")
    })
    decls = [render_decl(export) for _, export in sorted(used_exports.items())]
    lines = [
        f"// Generated by tools/benchmarks/generate-model-benchmarks.py for {model_slug} scenarios: {scenario_names}.",
        "// Regenerate from an HRX command program dump rather than editing by hand.",
        "",
        *(f"target.decl @{target}" for target in targets),
        "",
        *decls,
        "",
        "",
    ]
    for scenario in scenarios:
        lines.append(f"// Scenario: {scenario.scenario}")
        lines.append("")
        for case in scenario.cases:
            lines.append(case)
            lines.append("")
            symbol_match = re.match(r"check\.case public @(.+?)_case", case)
            if symbol_match:
                benchmark_symbol = symbol_match.group(1)
                lines.append(f"check.benchmark<@{benchmark_symbol}_case> @{benchmark_symbol}")
                lines.append("")

    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manifest(manifest_file: Path, model_file: Path, model_slug: str, scenario: ScenarioBenchmarks) -> None:
    write_json(
        manifest_file,
        {
            "schema": "ggml-hrx-model-loom-benchmarks-v2",
            "model": model_slug,
            "scenario": scenario.scenario,
            "loom_source": str(model_file.relative_to(HRX_DIR)),
            "command_count": scenario.command_count,
            "dispatch_count": scenario.dispatch_count,
            "generated_count": scenario.generated_count,
            "kernel_counts": scenario.kernel_counts,
            "dispatches": scenario.dispatches,
        },
    )


def parse_scenario_dump(value: str) -> tuple[str, Path]:
    scenario, separator, dump_dir = value.partition("=")
    if not separator or not scenario or not dump_dir:
        fail(f"invalid --scenario-dump value {value!r}; expected <scenario>=<dump-dir>")
    return scenario, Path(dump_dir)


def scenario_dumps_from_args(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.scenario_dump:
        if args.scenario is not None or args.dump_dir is not None:
            fail("--scenario-dump cannot be combined with --scenario or --dump-dir")
        if args.output_manifest is not None:
            fail("--output-manifest cannot be used with multiple --scenario-dump values")
        return [parse_scenario_dump(value) for value in args.scenario_dump]
    if args.scenario is None or args.dump_dir is None:
        fail("either --scenario and --dump-dir, or one or more --scenario-dump values, are required")
    return [(args.scenario, args.dump_dir)]


def generate(args: argparse.Namespace) -> None:
    exports = load_exports()
    model_slug = args.model
    model_file = args.output_loom or (BENCHMARK_DIR / "loom" / f"{model_slug}.loom")
    scenario_dumps = scenario_dumps_from_args(args)
    scenarios: list[ScenarioBenchmarks] = []
    used_exports: dict[str, dict[str, Any]] = {}
    for scenario, dump_dir in scenario_dumps:
        commands = load_program_commands(dump_dir)
        generated = generate_scenario(model_slug, scenario, commands, exports)
        scenarios.append(generated)
        used_exports.update(generated.used_exports)

    write_model_loom(model_file, model_slug, scenarios, used_exports)
    for scenario in scenarios:
        manifest_file = args.output_manifest or (BENCHMARK_DIR / "loom" / f"{model_slug}.{scenario.scenario}.json")
        write_manifest(manifest_file, model_file, model_slug, scenario)
        print(f"wrote {manifest_file}")

    print(f"wrote {model_file}")
    for scenario in scenarios:
        print(
            f"{scenario.scenario}: commands={scenario.command_count} "
            f"dispatches={scenario.dispatch_count} generated={scenario.generated_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model slug, for example llama32_3b_f16.")
    parser.add_argument("--scenario", help="Scenario slug, for example pp512.")
    parser.add_argument("--dump-dir", type=Path, help="Directory containing HRX program-*/program.json dumps.")
    parser.add_argument(
        "--scenario-dump",
        action="append",
        help="Scenario and dump directory as <scenario>=<dump-dir>. May be repeated.",
    )
    parser.add_argument("--output-loom", type=Path, help="Generated model Loom file.")
    parser.add_argument("--output-manifest", type=Path, help="Generated scenario manifest.")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
