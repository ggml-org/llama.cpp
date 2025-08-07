import os
import re
import ast
import argparse

variants_regex = re.compile(r'//\s*Variants:\s*\n(\[.*?\])', re.DOTALL)

def remove_variants_block(template_text):
    return re.sub(variants_regex, '', template_text)

def extract_variants(template_text):
    match = re.search(variants_regex, template_text)
    if not match:
        return None
    return ast.literal_eval(match.group(1))

def write_shader(shader_name, shader_code, output_dir, outfile):
    if output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, 'w', encoding='utf-8') as f_out:
            f_out.write(shader_code)
    outfile.write(f'const char* wgsl_{shader_name} = R"({shader_code})";\n')
    outfile.write('\n')

def generate_variants(shader_path, output_dir, outfile):
    shader_base_name = shader_path.split("/")[-1].split(".")[0]
    with open(shader_path, 'r', encoding='utf-8') as f:
        shader_code = f.read()
    variants = extract_variants(shader_code)
    shader_code = remove_variants_block(shader_code)
    if not variants:
        write_shader(shader_base_name, shader_code, output_dir, outfile)
    else:
        for variant in variants:
            shader_variant = shader_code
            parts = []
            for key, val in variant.items():
                parts.append(val)
                shader_variant = shader_variant.replace(key, val)
            output_name = f"{shader_base_name}_" + "_".join(parts)
            write_shader(output_name, shader_variant, output_dir, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as out:
        out.write("// Auto-generated shader embedding \n\n")
        for fname in sorted(os.listdir(args.input_dir)):
            if not fname.endswith('.wgsl'):
                continue
            shader_path = os.path.join(args.input_dir, fname)
            generate_variants(shader_path, args.output_dir, out)

if __name__ == '__main__':
    main()
