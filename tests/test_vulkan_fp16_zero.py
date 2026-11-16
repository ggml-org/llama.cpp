"""Check DIAG/TRI storage-only FP16 zero stores without a build or GPU.

Run: python3 -B -m unittest discover -s tests -p test_vulkan_fp16_zero.py -v
Requires glslc, spirv-val and spirv-dis on PATH, or executable paths in GLSLC,
SPIRV_VAL and SPIRV_DIS. GGML_VULKAN_SHADER_DIR can select another shader tree.
This test is intentionally invoked manually; it is not registered with CTest.
"""

import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


def resolve_tool(variable, executable):
    selected = os.environ.get(variable, executable)
    resolved = shutil.which(selected)
    if resolved is None:
        raise RuntimeError(f"Required tool {selected!r} not found; set {variable} to its executable path or add {executable} to PATH")
    return resolved


def run_tool(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Could not run {shlex.join(command)}: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def check_stores(test, assembly, width):
    # Only the scalar store paths emitted for these two shaders are supported.
    definitions = {}
    instructions = []
    block = None
    for line in assembly.splitlines():
        tokens = line.split(";", 1)[0].split()
        if not tokens:
            continue
        if len(tokens) > 2 and tokens[1] == "=":
            result, tokens = tokens[0], tokens[2:]
            definitions[result] = tokens
            if tokens[0] == "OpLabel":
                block = result
        instructions.append((block, tokens))

    def definition(value):
        test.assertIn(value, definitions, f"Unsupported SPIR-V value {value}")
        return definitions[value]

    def float_width(type_id):
        node = definition(type_id)
        test.assertEqual(node[0], "OpTypeFloat", f"Expected scalar float, got {node}")
        bits = int(node[1])
        test.assertIn(bits, (16, 32), f"Unsupported float width {bits}")
        return bits

    def base_pointer(value):
        seen = set()
        while True:
            test.assertNotIn(value, seen, "Cyclic pointer dataflow")
            seen.add(value)
            node = definition(value)
            if node[0] in ("OpAccessChain", "OpInBoundsAccessChain"):
                value = node[2]
            else:
                test.assertEqual(node[0], "OpVariable", f"Unsupported pointer dataflow: {node}")
                return value

    def trace(value):
        conversions = []
        seen = set()
        stored_width = None
        while True:
            test.assertNotIn(value, seen, "Cyclic store dataflow")
            seen.add(value)
            node = definition(value)
            bits = float_width(node[1])
            if stored_width is None:
                stored_width = bits
            if conversions:
                operation, target_width = conversions[-1]
                if operation == "OpCopyObject":
                    test.assertEqual(bits, target_width, "Copy changed scalar width")
                else:
                    test.assertNotEqual(bits, target_width, "Conversion did not change scalar width")
            if node[0] in ("OpFConvert", "OpCopyObject"):
                conversions.append((node[0], bits))
                value = node[2]
            elif node[0] == "OpConstant":
                test.assertEqual(len(node), 3, f"Unsupported floating constant: {node}")
                literal = node[2]
                number = float.fromhex(literal) if "0x" in literal.lower() else float(literal)
                positive_zero = number == 0 and math.copysign(1, number) == 1
                return "constant", stored_width, positive_zero, literal
            elif node[0] == "OpConstantNull":
                return "constant", stored_width, True, "null"
            elif node[0] == "OpLoad":
                test.assertEqual(bits, width, "Input load has wrong precision")
                return "load", stored_width, False, base_pointer(node[2])
            else:
                test.fail(f"Unsupported store dataflow: {node}")

    capabilities = [node[1] for _, node in instructions if node[0] == "OpCapability"]
    test.assertNotIn("Float16", capabilities, "FP16 arithmetic must not be enabled")
    if width == 16:
        test.assertIn("StorageBuffer16BitAccess", capabilities)
    entrypoints = [node for _, node in instructions if node[0] == "OpEntryPoint"]
    test.assertEqual(len(entrypoints), 1, "Expected one entrypoint")
    test.assertEqual(entrypoints[0][1:2] + entrypoints[0][3:4], ["GLCompute", '"main"'])
    execution_modes = [node for _, node in instructions if node[0] in ("OpExecutionMode", "OpExecutionModeId")]
    test.assertEqual(execution_modes, [["OpExecutionMode", entrypoints[0][2], "LocalSize", "512", "1", "1"]])

    bindings = {}
    descriptor_sets = {}
    for _, node in instructions:
        if node[0] == "OpDecorate" and node[2] == "Binding":
            bindings[node[1]] = int(node[3])
        elif node[0] == "OpDecorate" and node[2] == "DescriptorSet":
            descriptor_sets[node[1]] = int(node[3])
    test.assertEqual(sorted(bindings.values()), [0, 1], "Expected input/output buffer bindings")
    test.assertEqual({value: descriptor_sets.get(value) for value in bindings}, dict.fromkeys(bindings, 0))

    stores = []
    for store_block, node in instructions:
        if node[0] == "OpStore":
            root = base_pointer(node[1])
            test.assertEqual((descriptor_sets.get(root), bindings.get(root)), (0, 1), "Store is not to output binding")
            stores.append((store_block, trace(node[2])))
    test.assertEqual(len(stores), 2, "Expected zero and passthrough stores")
    constants = [(label, value) for label, value in stores if value[0] == "constant"]
    loads = [(label, value) for label, value in stores if value[0] == "load"]
    test.assertEqual(len(constants), 1, "Expected one constant output store")
    test.assertEqual(len(loads), 1, "Expected one nonconstant input passthrough store")
    for _, value in stores:
        test.assertEqual(value[1], width, "Stored value has wrong precision")
    test.assertEqual((descriptor_sets.get(loads[0][1][3]), bindings.get(loads[0][1][3])), (0, 0), "Passthrough must originate at input binding")
    branches = [node for _, node in instructions if node[0] == "OpBranchConditional" and node[2:4] == [loads[0][0], constants[0][0]]]
    test.assertEqual(len(branches), 1, "Zero store must be the false branch opposite input passthrough")
    test.assertTrue(constants[0][1][2], f"Output zero branch is not positive zero: {constants[0][1]}")


class TestVulkanFP16Zero(unittest.TestCase):
    def test_shader_stores(self):
        glslc = resolve_tool("GLSLC", "glslc")
        validator = resolve_tool("SPIRV_VAL", "spirv-val")
        disassembler = resolve_tool("SPIRV_DIS", "spirv-dis")
        default = Path(__file__).resolve().parents[1] / "ggml/src/ggml-vulkan/vulkan-shaders"
        shaders = Path(os.environ.get("GGML_VULKAN_SHADER_DIR", str(default))).resolve()
        for shader in ("diag", "tri"):
            for data_type, width in (("float16_t", 16), ("float", 32)):
                for mode in ("none", "short", "long"):
                    with self.subTest(shader=shader, width=width, dependencies=mode), tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        module = directory / "module.spv"
                        command = [glslc, "-fshader-stage=compute", "--target-env=vulkan1.2", str(shaders / (shader + ".comp")), "-o", str(module), "-O", "-DA_TYPE=" + data_type, "-DD_TYPE=" + data_type]
                        if mode != "none":
                            dependency = directory / ("short.d" if mode == "short" else "long-dependency-path/ggml/src/ggml-vulkan/shader.cpp.d")
                            dependency.parent.mkdir(parents=True, exist_ok=True)
                            command += ["-MD", "-MF", str(dependency)]
                        run_tool(command)
                        run_tool([validator, "--target-env", "vulkan1.2", str(module)])
                        assembly = run_tool([disassembler, "--raw-id", str(module)])
                        check_stores(self, assembly, width)


if __name__ == "__main__":
    unittest.main()
