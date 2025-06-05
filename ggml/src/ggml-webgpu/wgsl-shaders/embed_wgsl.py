import os
import argparse

def to_c_array(name, data):
    varname = f"wgsl_{name}"
    byte_array = ', '.join(f'0x{b:02x}' for b in data)
    return f"""\
const unsigned char {varname}[] = {{
    {byte_array}
}};
const unsigned int {varname}_len = sizeof({varname});
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input directory with .wgsl files')
    parser.add_argument('--output', required=True, help='Output .hpp file path')
    args = parser.parse_args()

    with open(args.output, 'w', encoding='utf-8') as out:
        out.write("// Auto-generated WGSL header\n\n")
        for fname in sorted(os.listdir(args.input)):
            if fname.endswith('.wgsl'):
                path = os.path.join(args.input, fname)
                varname = os.path.splitext(fname)[0]
                with open(path, 'rb') as f:
                    data = f.read()
                out.write(to_c_array(varname, data))
                out.write('\n')

if __name__ == '__main__':
    main()
