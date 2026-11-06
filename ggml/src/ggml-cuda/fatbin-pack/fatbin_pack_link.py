#!/usr/bin/env python3
"""Link wrapper for GGML_CUDA_FATBIN_PACK (RULE_LAUNCH_LINK of ggml-cuda).

nvcc compresses each image of each TU alone, so it cannot use the redundancy between images.
This wrapper packs .nv_fatbin of all objects into one zstd frame per arch family, then links with .nv_fatbin as NOBITS.
fatbin-unpack.c rebuilds the fatbins at load time.
"""

import argparse
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time

FATBIN_MAGIC = 0xBA55ED50
FATBIN_KIND_PTX = 1
SHT_NOBITS = 8
SHF_WRITE = 1
# entries of one TU stay adjacent, so a 64 MiB window gets almost all of the gain of a 2 GiB window
ZSTD_WINDOW_LOG = 26

MARKER_ASM = """\
.section .nv_fatbin,"aw",@nobits
.balign 8
.globl ggml_cuda_fatbin_section
.hidden ggml_cuda_fatbin_section
ggml_cuda_fatbin_section:
.section .note.GNU-stack,"",@progbits
"""


def expand_response_files(cmd):
    out = []
    for arg in cmd:
        if arg.startswith("@"):
            with open(arg[1:]) as f:
                out += shlex.split(f.read())
        else:
            out.append(arg)
    return out


def parse_fatbins(buf):
    tus = []
    off = 0
    while off + 16 <= len(buf):
        magic, _version, header_size, fat_size = struct.unpack_from("<IHHQ", buf, off)
        if magic != FATBIN_MAGIC:
            off += 8
            continue
        pos = off + header_size
        end = pos + fat_size
        entries = []
        while pos < end:
            kind, _, entry_header_size, size = struct.unpack_from("<HHIQ", buf, pos)
            arch = struct.unpack_from("<I", buf, pos + 28)[0]
            entries.append(((kind << 16) | arch, pos, entry_header_size + size))
            pos += entry_header_size + size
        assert pos == end, f"bad fatbin at offset {off}"
        tus.append((off, bytes(buf[off:off + 16]), entries))
        off = end
    return tus


def group_name(key):
    kind, arch = key >> 16, key & 0xffff
    return "ptx" if kind == FATBIN_KIND_PTX else f"sass{arch // 10}"


def pack(section, blob_path, zstd, work):
    buf = memoryview(section)
    tus = parse_fatbins(buf)
    assert tus, "no fatbin found in .nv_fatbin"

    images = sorted({key for tu in tus for key, _pos, _size in tu[2]})
    groups = {}
    for key in images:
        groups.setdefault(group_name(key), []).append(key)

    t_start = time.monotonic()
    frames = []
    tables = []
    for name in sorted(groups):
        raw = os.path.join(work, name + ".raw")
        table = []
        with open(raw, "wb") as f:
            for idx, (_off, _header, entries) in enumerate(tus):
                for key in groups[name]:
                    for entry_key, pos, size in entries:
                        if entry_key == key:
                            f.write(buf[pos:pos + size])
                            table.append((idx, key, size))
        t_group = time.monotonic()
        subprocess.run([zstd, "-19", f"--long={ZSTD_WINDOW_LOG}", "-T0", "-q", "-f", raw, "-o", raw + ".zst"], check=True)
        with open(raw + ".zst", "rb") as f:
            frames.append(f.read())
        tables.append(table)
        print(f"fatbin pack: {name}: {os.path.getsize(raw) / 2**20:.1f} MiB -> {len(frames[-1]) / 2**20:.1f} MiB in {time.monotonic() - t_group:.1f} s")
        os.unlink(raw)
        os.unlink(raw + ".zst")

    head = b"GGMLFBZ1" + struct.pack("<IIII", len(tus), len(frames), len(images), 0)
    tu_table = b"".join(struct.pack("<Q", tu[0]) + tu[1] for tu in tus)
    image_table = b"".join(struct.pack("<I", key) for key in images)
    entries_offset = len(head) + len(tu_table) + 32 * len(frames) + len(image_table)
    data_offset = entries_offset + sum(16 * len(table) for table in tables)
    group_table = b""
    for frame, table in zip(frames, tables):
        group_table += struct.pack("<QQQII", data_offset, len(frame), entries_offset, len(table), 0)
        data_offset += len(frame)
        entries_offset += 16 * len(table)
    with open(blob_path, "wb") as f:
        f.write(head + tu_table + group_table + image_table)
        for table in tables:
            f.write(b"".join(struct.pack("<IIQ", *entry) for entry in table))
        for frame in frames:
            f.write(frame)

    print(f"fatbin pack: {len(buf) / 2**20:.1f} MiB -> {os.path.getsize(blob_path) / 2**20:.1f} MiB, {len(tus)} TUs, {len(images)} images, {time.monotonic() - t_start:.1f} s")
    return {tu[0] for tu in tus}


def make_nobits_copy(src, dst):
    """Writes src to dst with .nv_fatbin as NOBITS and returns the old section content."""
    with open(src, "rb") as f:
        buf = bytearray(f.read())
    shoff = struct.unpack_from("<Q", buf, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", buf, 0x3A)
    stroff = struct.unpack_from("<Q", buf, shoff + strndx * entsize + 0x18)[0]
    content = None
    for i in range(num):
        hdr = shoff + i * entsize
        name_off = stroff + struct.unpack_from("<I", buf, hdr)[0]
        if buf[name_off:buf.index(0, name_off)] == b".nv_fatbin":
            flags = struct.unpack_from("<Q", buf, hdr + 8)[0]
            offset, size = struct.unpack_from("<QQ", buf, hdr + 0x18)
            content = bytes(buf[offset:offset + size])
            struct.pack_into("<I", buf, hdr + 4, SHT_NOBITS)
            struct.pack_into("<Q", buf, hdr + 8, flags | SHF_WRITE)
    assert content is not None, f"no .nv_fatbin in {src}"
    with open(dst, "wb") as f:
        f.write(buf)
    return content


def check_layout(lib, offsets, section_size):
    sections = {}
    text = subprocess.run(["readelf", "-SW", lib], capture_output=True, text=True, check=True).stdout
    for line in text.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]{16})\s+[0-9a-f]+\s+([0-9a-f]+)", line)
        if m:
            sections[m.group(1)] = (m.group(2), int(m.group(3), 16), int(m.group(4), 16))
    kind, base, size = sections[".nv_fatbin"]
    _, seg, seg_size = sections[".nvFatBinSegment"]
    assert kind == "NOBITS" and size == section_size, f".nv_fatbin is {kind} with size {size}, expected NOBITS with size {section_size}"

    # __fatDeviceText wrappers are 24 bytes, the fatbin pointer is at offset 8
    pointers = set()
    text = subprocess.run(["readelf", "-rW", lib], capture_output=True, text=True, check=True).stdout
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 4 and p[2].endswith("_RELATIVE"):
            off = int(p[0], 16)
            if seg <= off < seg + seg_size and (off - seg) % 24 == 8:
                pointers.add(int(p[3], 16) - base)
    assert pointers == offsets, "fatbin wrappers do not match the packed fatbin offsets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--zstd", required=True)
    parser.add_argument("--zstd-dir", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[0] == "--" else args.cmd

    link = expand_response_files(cmd)
    lib = link[link.index("-o") + 1]
    here = os.path.dirname(os.path.abspath(__file__))
    work = tempfile.mkdtemp(prefix="fatbin-pack-", dir=os.path.dirname(os.path.abspath(lib)))
    try:
        # no link with .nv_fatbin as PROGBITS: with many archs it can pass 2 GiB and overflow relocations
        relink = []
        section = bytearray()
        marker = os.path.join(work, "fatbin-marker.o")
        n_obj = 0
        for arg in link:
            if arg.endswith(".cu.o"):
                if n_obj == 0:
                    relink.append(marker)
                obj = os.path.join(work, f"{n_obj}.o")
                section += make_nobits_copy(arg, obj)
                section += bytes(-len(section) % 8)
                relink.append(obj)
                n_obj += 1
            else:
                relink.append(arg)

        offsets = pack(section, os.path.join(work, "fatbin.blob"), args.zstd, work)
        section_size = len(section)
        del section

        marker_src = os.path.join(work, "fatbin-marker.S")
        with open(marker_src, "w") as f:
            f.write(MARKER_ASM)
        unpack = os.path.join(work, "fatbin-unpack.o")
        subprocess.run([args.cc, "-c", "-fPIC", marker_src, "-o", marker], check=True)
        subprocess.run([args.cc, "-c", "-O2", "-fPIC", "-fvisibility=hidden", "-w", f"-I{args.zstd_dir}", f"-I{work}",
                        os.path.join(here, "fatbin-unpack.c"), "-o", unpack], check=True)

        relink.append(unpack)
        subprocess.run(relink, check=True)
        check_layout(lib, offsets, section_size)
    finally:
        shutil.rmtree(work, ignore_errors=True)


main()
