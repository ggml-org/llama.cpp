import os
import re
import ast
import argparse


def extract_block(text, name):
    pattern = rf'#define\({name}\)\s*(.*?)#end\({name}\)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Missing block: {name}")
    return match.group(1).strip()


def parse_decls(decls_text):
    decls = {}
    for name, code in re.findall(r'#decl\((.*?)\)\s*(.*?)#enddecl\(\1\)', decls_text, re.DOTALL):
        decls[name.strip()] = code.strip()
    return decls


def replace_placeholders(shader_text, replacements):
    for key, val in replacements.items():
        # Match {{KEY}} literally, where KEY is escaped
        pattern = r'{{\s*' + re.escape(key) + r'\s*}}'
        shader_text = re.sub(pattern, str(val), shader_text)
    return shader_text


def expand_includes(shader, input_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(input_dir, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            included_code = f.read()
        # Recursively expand includes inside the included file
        return expand_includes(included_code, input_dir)

    return include_pattern.sub(replacer, shader)


def write_shader(shader_name, shader_code, output_dir, outfile):
    if output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") as f_out:
            f_out.write(shader_code)
    outfile.write(f'const char* wgsl_{shader_name} = R"({shader_code})";\n\n')


def generate_variants(fname, input_dir, output_dir, outfile):
    shader_path = os.path.join(input_dir, fname)
    shader_base_name = fname.split(".")[0]

    with open(shader_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        variants = ast.literal_eval(extract_block(text, "VARIANTS"))
    except ValueError:
        write_shader(shader_base_name, text, output_dir, outfile)
    else:
        try:
            decls_map = parse_decls(extract_block(text, "DECLS"))
        except ValueError:
            decls_map = {}

        with open(os.path.join(input_dir, "common_decls.tmpl"), "r", encoding="utf-8") as f:
            common_decls = f.read()
        decls_map.update(parse_decls(common_decls))

        shader_template = extract_block(text, "SHADER")
        for variant in variants:
            if "DECLS" in variant:
                decls = variant["DECLS"]
            else:
                decls = []
            decls_code = ""
            for key in decls:
                if key not in decls_map:
                    raise ValueError(f"DECLS key '{key}' not found.")
                decls_code += decls_map[key] + "\n\n"

            shader_variant = replace_placeholders(shader_template, variant["REPLS"])
            final_shader = re.sub(r'\bDECLS\b', decls_code, shader_variant)
            final_shader = expand_includes(final_shader, input_dir)

            if "SRC0_TYPE" in variant["REPLS"] and "SRC1_TYPE" in variant["REPLS"]:
                output_name = f"{shader_base_name}_" + "_".join([variant["REPLS"]["SRC0_TYPE"], variant["REPLS"]["SRC1_TYPE"]])
            elif "TYPE_SUFFIX" in variant["REPLS"]:
                output_name = f"{shader_base_name}_" + variant["REPLS"]["TYPE_SUFFIX"]
            elif "TYPE" in variant["REPLS"]:
                output_name = f"{shader_base_name}_" + variant["REPLS"]["TYPE"]
            else:
                output_name = shader_base_name
            write_shader(output_name, final_shader, output_dir, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as out:
        out.write("// Auto-generated shader embedding\n\n")
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(".wgsl"):
                generate_variants(fname, args.input_dir, args.output_dir, out)


if __name__ == "__main__":
    main()
