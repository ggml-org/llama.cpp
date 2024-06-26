#!/usr/bin/env python

import logging
import argparse
import asyncio
import os
from tempfile import gettempdir

logger = logging.getLogger("ggml-vk-generate-shaders")

GLSLC = "glslc"

type_names = [
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
]

ASYNCIO_CONCURRENCY = 64

input_dir = "vulkan-shaders"
output_dir = gettempdir()

lock = asyncio.Lock()
shader_fnames = []


async def string_to_spv(name, in_fname, defines, fp16=True):
    name = f"{name}{'_fp32' if not fp16 else ''}"
    out_fname = os.path.join(output_dir, f"{name}.spv")

    in_path = os.path.join(input_dir, in_fname)

    cmd = [GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", in_path, "-o", out_fname]

    cmd.extend([f"-D{key}={value}" for key, value in defines.items()])

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    stdout = stdout.decode()
    error = stderr.decode()

    if proc.returncode:
        cmd = " ".join(cmd)
        logger.error(f"cannot compile {name}\n\n{cmd}\n\n{error}")
        return

    async with lock:
        shader_fnames.append((name, out_fname))


def matmul_shaders(tasks, fp16, matmul_id):
    if fp16:
        load_vec = "8"
        aligned_b_type_f32 = "mat2x4"
        aligned_b_type_f16 = "f16mat2x4"
    else:
        load_vec = "4"
        aligned_b_type_f32 = "vec4"
        aligned_b_type_f16 = "f16vec4"

    base_dict = {"FLOAT_TYPE": "float" if not fp16 else "float16_t"}
    shader_name = "matmul"

    if matmul_id:
        base_dict["MUL_MAT_ID"] = "1"
        shader_name = "matmul_id"

    if fp16:
        base_dict["FLOAT16"] = "1"

    # Shaders with f16 B_TYPE
    tasks.append(string_to_spv(f"{shader_name}_f32_f16", "mul_mm.comp", base_dict | {"DATA_A_F32": "1", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))
    tasks.append(string_to_spv(f"{shader_name}_f32_f16_aligned", "mul_mm.comp", base_dict | {"DATA_A_F32": "1", "LOAD_VEC_A": load_vec, "LOAD_VEC_B": load_vec, "B_TYPE": aligned_b_type_f16, "D_TYPE": "float"}, fp16))

    tasks.append(string_to_spv(f"{shader_name}_f16", "mul_mm.comp", base_dict | {"DATA_A_F16": "1", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))
    tasks.append(string_to_spv(f"{shader_name}_f16_aligned", "mul_mm.comp", base_dict | {"DATA_A_F16": "1", "LOAD_VEC_A": load_vec, "LOAD_VEC_B": load_vec, "B_TYPE": aligned_b_type_f16, "D_TYPE": "float"}, fp16))

    for tname in type_names:
        data_a_key = f"DATA_A_{tname.upper()}"
        load_vec_a = load_vec if tname in ("f32", "f16") else "2"
        tasks.append(string_to_spv(f"{shader_name}_{tname}_f32", "mul_mm.comp", base_dict | {data_a_key: "1", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv(f"{shader_name}_{tname}_f32_aligned", "mul_mm.comp", base_dict | {data_a_key: "2", "LOAD_VEC_A": load_vec_a, "LOAD_VEC_B": load_vec, "B_TYPE": aligned_b_type_f32, "D_TYPE": "float"}, fp16))


async def main():
    logger.info("ggml_vulkan: Generating and compiling shaders to SPIR-V")

    tasks = []

    for fp16 in (False, True):
        # MUL_MAT
        matmul_shaders(tasks, fp16, False)
        # MUL_MAT_ID
        matmul_shaders(tasks, fp16, True)

    for tname in type_names:
        base_dict = {"FLOAT_TYPE": "float"}

        # mul mat vec
        data_a_key = f"DATA_A_{tname.upper()}"
        shader = f"mul_mat_vec_{tname}.comp" if tname.endswith("_k") else "mul_mat_vec.comp"

        tasks.append(string_to_spv(f"mul_mat_vec_{tname}_f32_f32", shader, base_dict | {data_a_key: "1", "B_TYPE": "float", "D_TYPE": "float"}))
        tasks.append(string_to_spv(f"mul_mat_vec_{tname}_f16_f32", shader, base_dict | {data_a_key: "1", "B_TYPE": "float16_t", "D_TYPE": "float"}))

        tasks.append(string_to_spv(f"mul_mat_vec_id_{tname}_f32", shader, base_dict | {"MUL_MAT_ID": "1", data_a_key: "1", "B_TYPE": "float", "D_TYPE": "float"}))

        # Dequant shaders
        if tname != "f16":
            tasks.append(string_to_spv(f"dequant_{tname}", f"dequant_{tname}.comp", base_dict | {data_a_key: "1", "D_TYPE": "float16_t"}))

        # get_rows
        if not tname.endswith("_k"):
            shader = "get_rows.comp" if tname in ("f32", "f16") else "get_rows_quant.comp"

            if tname == "f16":
                tasks.append(string_to_spv(f"get_rows_{tname}", shader, {data_a_key: "1", "B_TYPE": "int", "D_TYPE": "float16_t", "OPTIMIZATION_ERROR_WORKAROUND": "1"}))
            else:
                tasks.append(string_to_spv(f"get_rows_{tname}", shader, {data_a_key: "1", "B_TYPE": "int", "D_TYPE": "float16_t"}))
            tasks.append(string_to_spv(f"get_rows_{tname}_f32", shader, {data_a_key: "1", "B_TYPE": "int", "D_TYPE": "float"}))

    tasks.append(string_to_spv("mul_mat_vec_p021_f16_f32", "mul_mat_vec_p021.comp", {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}))

    # Norms
    tasks.append(string_to_spv("norm_f32", "norm.comp", base_dict | {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rms_norm_f32", "rms_norm.comp", base_dict | {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("cpy_f32_f32", "copy.comp", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("cpy_f32_f16", "copy.comp", {"A_TYPE": "float", "D_TYPE": "float16_t"}))
    tasks.append(string_to_spv("cpy_f16_f16", "copy.comp", {"A_TYPE": "float16_t", "D_TYPE": "float16_t", "OPTIMIZATION_ERROR_WORKAROUND": "1"}))

    tasks.append(string_to_spv("add_f32", "add.comp", {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {}))

    tasks.append(string_to_spv("mul_f32", "mul.comp", {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("div_f32", "div.comp", {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("scale_f32", "scale.comp", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("sqr_f32", "square.comp", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("clamp_f32", "clamp.comp", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("gelu_f32", "gelu.comp", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("silu_f32", "silu.comp", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("relu_f32", "relu.comp", {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("soft_max_f32", "soft_max.comp", base_dict | {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("soft_max_f32_f16", "soft_max.comp", base_dict | {"A_TYPE": "float", "B_TYPE": "float16_t", "D_TYPE": "float"}))

    tasks.append(string_to_spv("rope_norm_f32", "rope_norm.comp", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rope_norm_f16", "rope_norm.comp", {"A_TYPE": "float16_t", "D_TYPE": "float16_t"}))

    tasks.append(string_to_spv("rope_neox_f32", "rope_neox.comp", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rope_neox_f16", "rope_neox.comp", {"A_TYPE": "float16_t", "D_TYPE": "float16_t"}))

    tasks.append(string_to_spv("argsort_f32", "argsort.comp", {"A_TYPE": "float"}))

    tasks.append(string_to_spv("sum_rows_f32", "sum_rows.comp", base_dict | {"A_TYPE": "float", "D_TYPE": "float"}))

    # Helper to decorate tasks with semaphore acquisition.
    async def withSemaphore(sem, task):
        async with sem:
            return await task

    # Run tasks concurrently guarded by a concurrency limit.
    sem = asyncio.Semaphore(ASYNCIO_CONCURRENCY)
    await asyncio.gather(*(withSemaphore(sem, task) for task in tasks))

    with open("ggml-vulkan-shaders.hpp", "w") as f:
        f.write("#include <cstdint>\n\n")
        for name, path in sorted(shader_fnames):

            with open(path, "rb") as spv:
                counter = 0
                newline_counter = 0
                f.write(f"unsigned char {name}_data[] = {{\n")
                for val in spv.read():
                    f.write(f"0x{val:02x},")
                    newline_counter += 1
                    counter += 1
                    if newline_counter >= 12:
                        newline_counter = 0
                        f.write("\n")
            f.write("\n};\n")
            f.write(f"const uint64_t {name}_len = {counter};\n\n")
            os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGML Vulkan Shader Generator")

    parser.add_argument("--glslc", help="Path to glslc")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.glslc:
        GLSLC = args.glslc

    asyncio.run(main())
