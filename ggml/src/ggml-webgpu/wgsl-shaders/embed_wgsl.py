import os
import argparse


def escape_triple_quotes(wgsl):
    # Simple defense in case of embedded """
    return wgsl.replace('"""', '\\"""')


def to_cpp_string_literal(varname, content):
    return f'const char* wgsl_{varname} = R"({content})";\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.output, 'w', encoding='utf-8') as out:
        out.write("// Auto-generated shader embedding \n\n")
        for fname in sorted(os.listdir(args.input)):
            if not fname.endswith('.wgsl'):
                continue
            shader_path = os.path.join(args.input, fname)
            varname = os.path.splitext(fname)[0]
            with open(shader_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = escape_triple_quotes(content)
            out.write(to_cpp_string_literal(varname, content))
            out.write('\n')


if __name__ == '__main__':
    main()
