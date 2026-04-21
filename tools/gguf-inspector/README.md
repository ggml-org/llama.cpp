# gguf-inspector

A Qt6 annotated hex viewer for GGUF model files. Every byte in the
header, KV metadata, tensor info table, padding, and tensor data blobs
is colored and labeled, and hovering or clicking a byte tells you
exactly what it is.

> Demo video: _(link pending)_

## What it shows

- **Structure tree** (left): header → metadata (KV pairs) → tensors
  grouped by layer index parsed from names like `blk.N.*`,
  `layers.N.*`, `h.N.*`. Clicking any entry jumps the hex view there.
- **Hex pane** (top right): virtualized — only visible rows are
  rendered, so multi-GB files open instantly. Each byte is colored by
  the GGUF field it belongs to; the currently selected region is
  outlined.
- **Annotation pane** (bottom right): full description of the
  hovered/clicked byte — its offset, the region it's part of (kind,
  range, size), the field label, and a multi-line explanation
  (including byte order, pointers, layer index, type info, etc.).
- **Legend** (bottom left): color chips for every region category.
- **Status bar**: file summary (size, version, endianness, tensor/KV
  counts, tensor-data offset) and the current hovered offset.

Tool-tips on hover repeat the annotation in place so you can scan the
file without taking your eyes off the hex.

## Requirements

- Qt 6.2 or newer (Widgets)
- A C++17 compiler
- CMake 3.16+

The tool does **not** link against llama/common/ggml — it parses GGUF
bytes directly, so it's fully self-contained.

## Build

Standalone:

```sh
cd tools/gguf-inspector
cmake -B build
cmake --build build -j
./build/gguf-inspector path/to/model.gguf
```

Fedora package names for the Qt dependency: `qt6-qtbase-devel`.
Debian/Ubuntu: `qt6-base-dev`. Arch: `qt6-base`.

## Usage

### GUI

```sh
gguf-inspector [path/to/model.gguf]
```

- `Ctrl+O` opens a file picker.
- Mouse wheel / arrow keys / PgUp / PgDn / Home / End scroll the hex
  view.
- Click a byte to lock the highlight on its enclosing region and
  update the annotation pane; hover anywhere for a tool-tip.
- Click a tree row to jump to that field in the hex view.

### Headless validation (`--check`)

```sh
gguf-inspector --check path/to/model.gguf
```

Parses the file and prints a one-line summary to stdout on success, or
a specific `ERROR` message to stderr on failure. Exits with 0/1. Useful
for CI / scripting — try it on a truncated or corrupt file to see the
error detail it produces.

## What the colors mean

| Group         | Fields                                                         |
|---------------|----------------------------------------------------------------|
| Header        | `magic`, `version`, `tensor_count`, `kv_count`                 |
| KV metadata   | `key_len`, `key`, `type`, `value`, `str_len`, `str`, `arr_*`   |
| Tensor info   | `name_len`, `name`, `n_dims`, `dims`, `type`, `offset`         |
| Align / data  | `padding`, `tensor data` (alternating shades per tensor)       |

Hover any byte for the full description.

## Error handling

The parser rejects and explains:

- non-existent / non-regular files
- files under 24 bytes (minimum GGUF header)
- wrong magic (shows the hex + ASCII of what's actually at offset 0)
- unsupported versions (shows both native and byte-swapped readings)
- truncated / out-of-bounds field reads (names the field and offset)
- implausible key / name lengths (> 1 MiB)
- implausible dimension counts (outside 1–8)
- overflow in element-count or byte-size multiplications
- tensor data that extends past EOF

## AI-assistance disclosure

Per the llama.cpp [AGENTS.md](../../AGENTS.md) policy: this tool was
drafted with AI assistance. It is intended as a personal inspection
utility and not as an upstream PR without substantial human
review/rewrite.
