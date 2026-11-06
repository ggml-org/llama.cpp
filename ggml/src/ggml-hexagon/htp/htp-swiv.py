#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import struct
import sys
import tempfile
import zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_INDEX = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_VERSION_INDEX = 6
EV_CURRENT = 1
MAX_PHDR_COUNT = 100

LOAD_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_SIZE_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


class Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    def __init__(self, data):
        unpacked_data = (Elf_Ehdr_common.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]


class Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    def __init__(self, data):
        unpacked_data = (Elf32_Ehdr.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]
        self.e_entry = unpacked_data[4]
        self.e_phoff = unpacked_data[5]
        self.e_shoff = unpacked_data[6]
        self.e_flags = unpacked_data[7]
        self.e_ehsize = unpacked_data[8]
        self.e_phentsize = unpacked_data[9]
        self.e_phnum = unpacked_data[10]
        self.e_shentsize = unpacked_data[11]
        self.e_shnum = unpacked_data[12]
        self.e_shstrndx = unpacked_data[13]

    def get_packed_data(self):
        values = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_version,
                  self.e_entry,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        return (Elf32_Ehdr.s).pack(*values)


class Elf32_Phdr:
    s = struct.Struct('I' * 8)

    def __init__(self, data):
        unpacked_data = (Elf32_Phdr.s).unpack(data)
        self.p_type = unpacked_data[0]
        self.p_offset = unpacked_data[1]
        self.p_vaddr = unpacked_data[2]
        self.p_paddr = unpacked_data[3]
        self.p_filesz = unpacked_data[4]
        self.p_memsz = unpacked_data[5]
        self.p_flags = unpacked_data[6]
        self.p_align = unpacked_data[7]

    def get_packed_data(self):
        values = [self.p_type,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        return (Elf32_Phdr.s).pack(*values)


class Elf32_Shdr:
    s = struct.Struct('I' * 10)

    def __init__(self, data):
        unpacked_data = (Elf32_Shdr.s).unpack(data)
        self.sh_name = unpacked_data[0]
        self.sh_type = unpacked_data[1]
        self.sh_flags = unpacked_data[2]
        self.sh_addr = unpacked_data[3]
        self.sh_offset = unpacked_data[4]
        self.sh_size = unpacked_data[5]
        self.sh_link = unpacked_data[6]
        self.sh_info = unpacked_data[7]
        self.sh_addralign = unpacked_data[8]
        self.sh_entsize = unpacked_data[9]
        self.sh_name_str = ""

    def get_packed_data(self):
        values = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_size,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        return (Elf32_Shdr.s).pack(*values)


class Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    def __init__(self, data):
        unpacked_data = (Elf64_Ehdr.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]
        self.e_entry = unpacked_data[4]
        self.e_phoff = unpacked_data[5]
        self.e_shoff = unpacked_data[6]
        self.e_flags = unpacked_data[7]
        self.e_ehsize = unpacked_data[8]
        self.e_phentsize = unpacked_data[9]
        self.e_phnum = unpacked_data[10]
        self.e_shentsize = unpacked_data[11]
        self.e_shnum = unpacked_data[12]
        self.e_shstrndx = unpacked_data[13]

    def get_packed_data(self):
        values = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_version,
                  self.e_entry,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        return (Elf64_Ehdr.s).pack(*values)


class Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    def __init__(self, data):
        unpacked_data = (Elf64_Phdr.s).unpack(data)
        self.p_type = unpacked_data[0]
        self.p_flags = unpacked_data[1]
        self.p_offset = unpacked_data[2]
        self.p_vaddr = unpacked_data[3]
        self.p_paddr = unpacked_data[4]
        self.p_filesz = unpacked_data[5]
        self.p_memsz = unpacked_data[6]
        self.p_align = unpacked_data[7]

    def get_packed_data(self):
        values = [self.p_type,
                  self.p_flags,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        return (Elf64_Phdr.s).pack(*values)


class Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    def __init__(self, data):
        unpacked_data = (Elf64_Shdr.s).unpack(data)
        self.sh_name = unpacked_data[0]
        self.sh_type = unpacked_data[1]
        self.sh_flags = unpacked_data[2]
        self.sh_addr = unpacked_data[3]
        self.sh_offset = unpacked_data[4]
        self.sh_size = unpacked_data[5]
        self.sh_link = unpacked_data[6]
        self.sh_info = unpacked_data[7]
        self.sh_addralign = unpacked_data[8]
        self.sh_entsize = unpacked_data[9]
        self.sh_name_str = ""

    def get_packed_data(self):
        values = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_size,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        return (Elf64_Shdr.s).pack(*values)


class SWIV_Seg:
    s = struct.Struct('4sIQ')

    def __init__(self):
        self.magic = SWIV_MAGIC
        self.crc = 0
        self.padding = 0

    def get_packed_data(self):
        values = [self.magic,
                  self.crc,
                  self.padding
                  ]

        return (SWIV_Seg.s).pack(*values)


def is_elf64(elf_header):
    return elf_header.e_ident[ELF_CLASS_INDEX] == ELFCLASS64


def verify_elf_header(elf_header):
    ident = elf_header.e_ident
    return (ident.startswith(ELF_MAGIC)
            and ident[ELF_CLASS_INDEX] in (ELFCLASS32, ELFCLASS64)
            and ident[ELF_VERSION_INDEX] == EV_CURRENT)


def file_copy_offset(in_fp, in_off, out_fp, out_off, num_bytes):
    in_fp.seek(in_off)
    read_in = in_fp.read(num_bytes)
    out_fp.seek(out_off)
    out_fp.write(read_in)

    return num_bytes


def preprocess_elf_file(elf_file_name):
    with open(elf_file_name, 'rb') as elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.read(Elf_Ehdr_common.s.size))

        if verify_elf_header(elf_header) is False:
            raise RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        if is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.read(Elf64_Ehdr.s.size))
        else:
            elf_header = Elf32_Ehdr(elf_fp.read(Elf32_Ehdr.s.size))

        phdr_table = []

        phdr_size = elf_header.e_phentsize

        file_offset = elf_header.e_phoff
        elf_fp.seek(file_offset)

        for _ in range(elf_header.e_phnum):
            if is_elf64(elf_header):
                phdr_table.append(Elf64_Phdr(elf_fp.read(phdr_size)))
            else:
                phdr_table.append(Elf32_Phdr(elf_fp.read(phdr_size)))

    return [elf_header, phdr_table]


def process_section_header(elf_file_name):
    with open(elf_file_name, 'rb') as elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.read(Elf_Ehdr_common.s.size))

        if verify_elf_header(elf_header) is False:
            raise RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        if is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.read(Elf64_Ehdr.s.size))
        else:
            elf_header = Elf32_Ehdr(elf_fp.read(Elf32_Ehdr.s.size))

        shdr_table = []

        shdr_size = elf_header.e_shentsize
        elf_fp.seek(elf_header.e_shoff)

        for _ in range(elf_header.e_shnum):
            if is_elf64(elf_header):
                shdr_table.append(Elf64_Shdr(elf_fp.read(shdr_size)))
            else:
                shdr_table.append(Elf32_Shdr(elf_fp.read(shdr_size)))

    return shdr_table


def crc32_update_file(elf_fp, offset, size, crc):
    elf_fp.seek(offset)
    remaining = size
    while remaining > 0:
        chunk = elf_fp.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("Unexpected EOF while calculating CRC")
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    return crc


def crc32_update_zeros(size, crc):
    zero_chunk = b'\x00' * min(1024 * 1024, size)
    remaining = size
    while remaining > 0:
        chunk = zero_chunk[:min(len(zero_chunk), remaining)]
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    return crc


def crc32_calculate(elf_fp, phdr_table):
    crc = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    for curr_phdr in phdr_table:
        if curr_phdr.p_type != LOAD_TYPE:
            continue

        crc = crc32_update_file(elf_fp, curr_phdr.p_offset, curr_phdr.p_filesz, crc)

        zero_size = curr_phdr.p_memsz - curr_phdr.p_filesz
        if zero_size > 0:
            crc = crc32_update_zeros(zero_size, crc)

    crc &= 0xffffffff
    logger.info("calculated CRC: %#x", crc)

    return crc


def verify_swiv_seg(elf_in_fp, offset):

    elf_in_fp.seek(offset)
    swiv_seg = SWIV_Seg()
    unpacked_data = (SWIV_Seg.s).unpack(elf_in_fp.read(SWIV_SEG_SIZE_MIN))
    swiv_seg.magic = unpacked_data[0]

    return swiv_seg.magic == SWIV_MAGIC


def same_output_file(input_elf, output_elf):
    try:
        return os.path.exists(output_elf) and os.path.samefile(input_elf, output_elf)
    except FileNotFoundError:
        return False


def make_output_target(input_elf, output_elf):
    if not same_output_file(input_elf, output_elf):
        return output_elf, None

    output_dir = os.path.dirname(os.path.abspath(output_elf))
    output_base = os.path.basename(output_elf)
    fd, temp_output = tempfile.mkstemp(prefix=output_base + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(input_elf, temp_output)
    return temp_output, temp_output


def swiv_generate(input_elf, output_elf):
    output_target, temp_output = make_output_target(input_elf, output_elf)

    try:
        with open(input_elf, "rb") as elf_in_fp, open(output_target, "wb+") as elf_out_fp:
            [elf_header, phdr_table] = preprocess_elf_file(input_elf)

            num_phdrs = elf_header.e_phnum
            num_shdrs = elf_header.e_shnum
            phdr_size = elf_header.e_phentsize

            if num_phdrs == 0 or num_phdrs > MAX_PHDR_COUNT:
                raise RuntimeError("Input ELF has invalid number of program headers.")

            if elf_header.e_shoff == 0 or num_shdrs == 0:
                raise RuntimeError("Input ELF does not contain a section header table.")

            shdr_table_input = process_section_header(input_elf)
            shdr_table_output = process_section_header(input_elf)

            last_phdr = phdr_table[num_phdrs - 1]
            if last_phdr.p_memsz == SWIV_SEG_SIZE_MIN and verify_swiv_seg(elf_in_fp, last_phdr.p_offset):
                raise RuntimeError("A SWIV segment is detected in the input ELF.")

            last_shdr = shdr_table_input[elf_header.e_shnum - 1]
            if last_shdr.sh_size == SWIV_SEG_SIZE_MIN and verify_swiv_seg(elf_in_fp, last_shdr.sh_offset):
                raise RuntimeError("A SWIV section is detected in the input ELF.")

            if is_elf64(elf_header):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.size)
            else:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.size)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_size = SWIV_SEG_SIZE_MIN
            swiv_shdr.sh_offset = elf_header.e_shoff + (elf_header.e_shnum + 1) * elf_header.e_shentsize
            shdr_table_output.append(swiv_shdr)
            elf_header.e_shnum += 1

            if elf_header.e_phoff != 0:
                for phdr in phdr_table:
                    file_copy_offset(elf_in_fp, phdr.p_offset, elf_out_fp, phdr.p_offset, phdr.p_filesz)

            elf_out_fp.seek(0)
            elf_out_fp.write(elf_header.get_packed_data())

            phdr_start = elf_header.e_phoff
            for phdr in phdr_table:
                elf_out_fp.seek(phdr_start)
                elf_out_fp.write(phdr.get_packed_data())
                phdr_start += phdr_size

            for shdr in shdr_table_input[:num_shdrs]:
                if shdr.sh_type == SHT_NOBITS or shdr.sh_size == 0:
                    continue
                file_copy_offset(elf_in_fp, shdr.sh_offset, elf_out_fp, shdr.sh_offset, shdr.sh_size)

            shdr_start = elf_header.e_shoff
            for shdr in shdr_table_output:
                elf_out_fp.seek(shdr_start)
                elf_out_fp.write(shdr.get_packed_data())
                shdr_start += elf_header.e_shentsize

            elf_out_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.crc = crc32_calculate(elf_out_fp, phdr_table)
            elf_out_fp.seek(swiv_shdr.sh_offset)
            elf_out_fp.write(swiv_seg.get_packed_data())

        if temp_output is not None:
            os.replace(temp_output, output_elf)
            temp_output = None

        return 0
    finally:
        if temp_output is not None and os.path.exists(temp_output):
            os.unlink(temp_output)


def main(argv):
    parser = argparse.ArgumentParser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.add_argument("-i", "--input", dest="input_file", metavar="INPUT", required=True)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="show calculated CRC")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    try:
        return swiv_generate(args.input_file, args.output_file)
    except RuntimeError as exc:
        logger.error("ERROR: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
