## gguf

This is a Python package for writing binary files in the [GGUF](https://github.com/ggerganov/ggml/pull/302)
(GGML Universal File) format.

See [convert-llama-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py)
as an example for its usage.

## Installation
```sh
pip install gguf
```

## API Examples/Simple Tools

[examples/writer.py](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/examples/writer.py) — Generates `example.gguf` in the current directory to demonstrate generating a GGUF file. Note that this file cannot be used as a model.

[scripts/gguf-dump.py](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/scripts/gguf-dump.py) — Dumps a GGUF file's metadata to the console.

[scripts/gguf-set-metadata.py](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/scripts/gguf-set-metadata.py) — Allows changing simple metadata values in a GGUF file by key.

[scripts/gguf-convert-endian.py](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/scripts/gguf-convert-endian.py) — Allows converting the endianness of GGUF files.

[scripts/gguf-new-metadata.py](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/scripts/gguf-new-metadata.py) — Copies a GGUF file with added/modified/removed metadata values.

## Development
Maintainers who participate in development of this package are advised to install it in editable mode:

```sh
cd /path/to/llama.cpp/gguf-py

pip install --editable .
```

**Note**: This may require to upgrade your Pip installation, with a message saying that editable installation currently requires `setup.py`.
In this case, upgrade Pip to the latest:

```sh
pip install --upgrade pip
```

## Automatic publishing with CI

There's a GitHub workflow to make a release automatically upon creation of tags in a specified format.

1. Bump the version in `pyproject.toml`.
2. Create a tag named `gguf-vx.x.x` where `x.x.x` is the semantic version number.

```sh
git tag -a gguf-v1.0.0 -m "Version 1.0 release"
```

3. Push the tags.

```sh
git push origin --tags
```

## Manual publishing
If you want to publish the package manually for any reason, you need to have `twine` and `build` installed:

```sh
pip install build twine
```

Then, follow these steps to release a new version:

1. Bump the version in `pyproject.toml`.
2. Build the package:

```sh
python -m build
```

3. Upload the generated distribution archives:

```sh
python -m twine upload dist/*
```

## TODO
- [ ] Add tests
- [ ] Include conversion scripts as command line entry points in this package.
