#

import sys
import logging
import re
import os

logger = logging.getLogger("opencl-embed-kernel")

INCLUDE_PATTERN = re.compile(r'#include\s+"(.*)".*')


def parse_file_line(ifile, ofile, base_path: str):
    for i in ifile:
        if m := INCLUDE_PATTERN.match(i):
            include_file = os.path.join(base_path, m.group(1))
            logger.info(f"Embedding file: {include_file}")
            with open(include_file, "r") as incf:
                parse_file_line(incf, ofile, base_path)
        else:
            ofile.write('R"({})"\n'.format(i))


def main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ipath = os.path.dirname(sys.argv[1])
    with open(sys.argv[1], "r") as ifile, open(sys.argv[2], "w") as ofile:
        parse_file_line(ifile, ofile, ipath)


if __name__ == "__main__":
    main()
