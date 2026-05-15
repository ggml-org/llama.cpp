import argparse
from gguf import GGUFReader


def tensor_nbytes(tensor):
    # tensor.data is usually a numpy view / memmap-like array.
    if hasattr(tensor.data, "nbytes"):
        return int(tensor.data.nbytes)

    raise TypeError(f"Cannot determine nbytes for tensor {tensor.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--grep", default="")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    reader = GGUFReader(args.gguf)

    shown = 0
    total = 0

    for t in reader.tensors:
        name = t.name
        if args.grep and args.grep not in name:
            continue

        nbytes = tensor_nbytes(t)
        total += nbytes

        print(
            f"{name:80s} "
            f"shape={list(t.shape)} "
            f"type={t.tensor_type} "
            f"bytes={nbytes}"
        )

        shown += 1
        if shown >= args.limit:
            break

    print(f"\nshown={shown}")
    print(f"shown_bytes={total}")


if __name__ == "__main__":
    main()
