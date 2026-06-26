
# llama-gguf-hash

CLI to hash GGUF files to detect difference on a per model and per tensor level.

**Command line options:**

- `--help`: display help message
- `--xxh64`: use xhash 64bit hash mode (default)
- `--sha1`: use sha1
- `--uuid`: use uuid
- `--sha256`: use sha256
- `--all`: use all hash
- `--no-layer`: exclude per layer hash
- `--uuid`: generate UUIDv5 ID
- `-c`, `--check <manifest>`:  verify against a manifest

## About

While most POSIX systems already have hash checking programs like sha256sum, it
is designed to check entire files. This is not ideal for our purpose if we want
to check for consistency of the tensor data even if the metadata content of the
gguf KV store has been updated.

This program is designed to hash a gguf tensor payload on a 'per tensor layer'
in addition to a 'entire tensor model' hash. The intent is that the entire
tensor layer can be checked first but if there is any detected inconsistencies,
then the per tensor hash can be used to narrow down the specific tensor layer
that has inconsistencies.

For Maintainers:
- Detection of tensor inconsistency during development and automated tests
    - This is served by xxh64 which is fast
    - This is also served by having per tensor layer to assist in narrowing down
      the location of the faulty tensor layer
    - This is also served by sha1 which is much slower but more widely supported

For Model Creators:
- Optional consistent UUID generation based on model tensor content
    - This is served by UUIDv5 which is useful for databases keys
        - llama.cpp UUIDv5 Namespace: `ef001206-dadc-5f6d-a15f-3359e577d4e5`
            - Made via UUIDv5 URL namespace of `en.wikipedia.org/wiki/Llama.cpp`

For Model Users:
- Assurance of tensor layer integrity even if metadata was updated
    - This is served by sha256 which is still considered very secure as of 2024

### Design Note

- The default behavior of this program if no arguments is provided is to hash
  using xxhash's xxh32 mode because it is very fast and is primarily targeted
  towards maintainers who may want to use this in automated tests.
- xxhash support xxh32 and xxh128 for 32bit hash and 128bit hash respectively
  however we picked 64bit xxhash as most computers are 64bit as of 2024 and thus
  would have a better affinity to calculating hash that is 64bit in size.

## Compile Example

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_FATAL_WARNINGS=ON
make -C build clean
make -C build llama-gguf-hash VERBOSE=1
./build/bin/llama-gguf-hash test.gguf
./build/bin/llama-gguf-hash --xxh64 test.gguf
./build/bin/llama-gguf-hash --sha1 test.gguf
./build/bin/llama-gguf-hash --uuid test.gguf
./build/bin/llama-gguf-hash --sha256 test.gguf
```

## Generation and Verification Example

To generate we may use this command

```bash
./llama-gguf-hash --all test.gguf > test.gguf.manifest
```

Which would generate a manifest that looks like below, which contains multiple hash type and per tensor layer hashes as well
(This excludes UUID as that is an ID not a hash)

```bash
xxh64     f66e9cd66a4396a0  test.gguf:tensor_0
sha1      59f79ecefd8125a996fdf419239051a7e99e5f20  test.gguf:tensor_0
sha256    c0510d38fa060c46265e0160a85c7243096b01dd31c2f355bdbb5516b20de1bd  test.gguf:tensor_0
xxh64     7d3a1f9ac04d0537  test.gguf:tensor_1
sha1      4765f592eacf096df4628ba59476af94d767080a  test.gguf:tensor_1
sha256    8514cbcc73692a2c56bd7a33a022edd5ff819614bd23b19915d7224387f397a7  test.gguf:tensor_1
xxh64     a0af5d700049693b  test.gguf:tensor_2
sha1      25cbfbad4513cc348e2c95ebdee69d6ff2fd8753  test.gguf:tensor_2
sha256    947e6b36e20f2cc95e1d2ce1c1669d813d574657ac6b5ac5196158d454d35180  test.gguf:tensor_2
xxh64     e83fddf559d7b6a6  test.gguf:tensor_3
sha1      a9cba73e2d90f2ee3dae2548caa42bef3fe6a96c  test.gguf:tensor_3
sha256    423b044e016d8ac73c39f23f60bf01bedef5ecb03c0230accd824c91fe86f1a1  test.gguf:tensor_3
xxh64     1257733306b7992d  test.gguf:tensor_4
sha1      d7bc61db93bb685ce9d598da89717c66729b7543  test.gguf:tensor_4
sha256    79737cb3912d4201384cf7f16a1a37ff7823f23ea796cb205b6ca361ab9e3ebf  test.gguf:tensor_4
xxh64     d238d16ba4711e58  test.gguf:tensor_5
sha1      0706566c198fe1072f37e0a5135b4b5f23654c52  test.gguf:tensor_5
sha256    60949be8298eced0ecdde64487643d018407bd261691e061d9e9c3dbc9fd358b  test.gguf:tensor_5
xxh64     3fbc3b65ab8c7f39  test.gguf:tensor_6
sha1      73922a0727226a409049f6fc3172a52219ca6f00  test.gguf:tensor_6
sha256    574f4c46ff384a3b9a225eb955d2a871847a2e8b3fa59387a8252832e92ef7b0  test.gguf:tensor_6
xxh64     c22021c29854f093  test.gguf:tensor_7
sha1      efc39cece6a951188fc41e354c73bbfe6813d447  test.gguf:tensor_7
sha256    4c0410cd3c500f078ae5b21e8dc9eb79e29112713b2ab58a882f82a3868d4d75  test.gguf:tensor_7
xxh64     936df61f5d64261f  test.gguf:tensor_8
sha1      c2490296d789a4f34398a337fed8377d943d9f06  test.gguf:tensor_8
sha256    c4401313feeba0261275c3b25bd2d8fe40ce04e0f440c2980ed0e9674c30ff01  test.gguf:tensor_8
xxh64     93fd20c64421c081  test.gguf:tensor_9
sha1      7047ce1e78437a6884337a3751c7ee0421918a65  test.gguf:tensor_9
sha256    23d57cf0d7a6e90b0b3616b41300e0cd354781e812add854a5f95aa55f2bc514  test.gguf:tensor_9
xxh64     5a54d3aad816f302  test.gguf
sha1      d15be52c4ff213e823cb6dd13af7ee2f978e7042  test.gguf
sha256    7dd641b32f59b60dbd4b5420c4b0f6321ccf48f58f6ae201a3dbc4a58a27c6e4  test.gguf
```

We can then use the normal check command which will by default check for the highest security strength hash and verify against that:

```bash
$ ./llama-gguf-hash --check test.gguf.manifest test.gguf
manifest  test.gguf.manifest  sha256  sha1  xxh64
sha256    c0510d38fa060c46265e0160a85c7243096b01dd31c2f355bdbb5516b20de1bd  test.gguf:tensor_0  -  Ok
sha256    8514cbcc73692a2c56bd7a33a022edd5ff819614bd23b19915d7224387f397a7  test.gguf:tensor_1  -  Ok
sha256    947e6b36e20f2cc95e1d2ce1c1669d813d574657ac6b5ac5196158d454d35180  test.gguf:tensor_2  -  Ok
sha256    423b044e016d8ac73c39f23f60bf01bedef5ecb03c0230accd824c91fe86f1a1  test.gguf:tensor_3  -  Ok
sha256    79737cb3912d4201384cf7f16a1a37ff7823f23ea796cb205b6ca361ab9e3ebf  test.gguf:tensor_4  -  Ok
sha256    60949be8298eced0ecdde64487643d018407bd261691e061d9e9c3dbc9fd358b  test.gguf:tensor_5  -  Ok
sha256    574f4c46ff384a3b9a225eb955d2a871847a2e8b3fa59387a8252832e92ef7b0  test.gguf:tensor_6  -  Ok
sha256    4c0410cd3c500f078ae5b21e8dc9eb79e29112713b2ab58a882f82a3868d4d75  test.gguf:tensor_7  -  Ok
sha256    c4401313feeba0261275c3b25bd2d8fe40ce04e0f440c2980ed0e9674c30ff01  test.gguf:tensor_8  -  Ok
sha256    23d57cf0d7a6e90b0b3616b41300e0cd354781e812add854a5f95aa55f2bc514  test.gguf:tensor_9  -  Ok
sha256    7dd641b32f59b60dbd4b5420c4b0f6321ccf48f58f6ae201a3dbc4a58a27c6e4  test.gguf  -  Ok

Verification results for test.gguf.manifest - Success
```

Or we may explicitly ask for a faster hash like:

```bash
$ ./llama-gguf-hash --check test.gguf.manifest --xxh64 test.gguf
manifest  test.gguf.manifest  sha256  sha1  xxh64
xxh64     f66e9cd66a4396a0  test.gguf:tensor_0  -  Ok
xxh64     7d3a1f9ac04d0537  test.gguf:tensor_1  -  Ok
xxh64     a0af5d700049693b  test.gguf:tensor_2  -  Ok
xxh64     e83fddf559d7b6a6  test.gguf:tensor_3  -  Ok
xxh64     1257733306b7992d  test.gguf:tensor_4  -  Ok
xxh64     d238d16ba4711e58  test.gguf:tensor_5  -  Ok
xxh64     3fbc3b65ab8c7f39  test.gguf:tensor_6  -  Ok
xxh64     c22021c29854f093  test.gguf:tensor_7  -  Ok
xxh64     936df61f5d64261f  test.gguf:tensor_8  -  Ok
xxh64     93fd20c64421c081  test.gguf:tensor_9  -  Ok
xxh64     5a54d3aad816f302  test.gguf  -  Ok

Verification results for test.gguf.manifest - Success
```

Or maybe we want to just check that all the hash is valid:

```bash
$./llama-gguf-hash --check test.gguf.manifest --all test.gguf.manifest
manifest  test.gguf.manifest  sha256  sha1  xxh64
xxh64     f66e9cd66a4396a0  test.gguf:tensor_0  -  Ok
sha1      59f79ecefd8125a996fdf419239051a7e99e5f20  test.gguf:tensor_0  -  Ok
sha256    c0510d38fa060c46265e0160a85c7243096b01dd31c2f355bdbb5516b20de1bd  test.gguf:tensor_0  -  Ok
xxh64     7d3a1f9ac04d0537  test.gguf:tensor_1  -  Ok
sha1      4765f592eacf096df4628ba59476af94d767080a  test.gguf:tensor_1  -  Ok
sha256    8514cbcc73692a2c56bd7a33a022edd5ff819614bd23b19915d7224387f397a7  test.gguf:tensor_1  -  Ok
xxh64     a0af5d700049693b  test.gguf:tensor_2  -  Ok
sha1      25cbfbad4513cc348e2c95ebdee69d6ff2fd8753  test.gguf:tensor_2  -  Ok
sha256    947e6b36e20f2cc95e1d2ce1c1669d813d574657ac6b5ac5196158d454d35180  test.gguf:tensor_2  -  Ok
xxh64     e83fddf559d7b6a6  test.gguf:tensor_3  -  Ok
sha1      a9cba73e2d90f2ee3dae2548caa42bef3fe6a96c  test.gguf:tensor_3  -  Ok
sha256    423b044e016d8ac73c39f23f60bf01bedef5ecb03c0230accd824c91fe86f1a1  test.gguf:tensor_3  -  Ok
xxh64     1257733306b7992d  test.gguf:tensor_4  -  Ok
sha1      d7bc61db93bb685ce9d598da89717c66729b7543  test.gguf:tensor_4  -  Ok
sha256    79737cb3912d4201384cf7f16a1a37ff7823f23ea796cb205b6ca361ab9e3ebf  test.gguf:tensor_4  -  Ok
xxh64     d238d16ba4711e58  test.gguf:tensor_5  -  Ok
sha1      0706566c198fe1072f37e0a5135b4b5f23654c52  test.gguf:tensor_5  -  Ok
sha256    60949be8298eced0ecdde64487643d018407bd261691e061d9e9c3dbc9fd358b  test.gguf:tensor_5  -  Ok
xxh64     3fbc3b65ab8c7f39  test.gguf:tensor_6  -  Ok
sha1      73922a0727226a409049f6fc3172a52219ca6f00  test.gguf:tensor_6  -  Ok
sha256    574f4c46ff384a3b9a225eb955d2a871847a2e8b3fa59387a8252832e92ef7b0  test.gguf:tensor_6  -  Ok
xxh64     c22021c29854f093  test.gguf:tensor_7  -  Ok
sha1      efc39cece6a951188fc41e354c73bbfe6813d447  test.gguf:tensor_7  -  Ok
sha256    4c0410cd3c500f078ae5b21e8dc9eb79e29112713b2ab58a882f82a3868d4d75  test.gguf:tensor_7  -  Ok
xxh64     936df61f5d64261f  test.gguf:tensor_8  -  Ok
sha1      c2490296d789a4f34398a337fed8377d943d9f06  test.gguf:tensor_8  -  Ok
sha256    c4401313feeba0261275c3b25bd2d8fe40ce04e0f440c2980ed0e9674c30ff01  test.gguf:tensor_8  -  Ok
xxh64     93fd20c64421c081  test.gguf:tensor_9  -  Ok
sha1      7047ce1e78437a6884337a3751c7ee0421918a65  test.gguf:tensor_9  -  Ok
sha256    23d57cf0d7a6e90b0b3616b41300e0cd354781e812add854a5f95aa55f2bc514  test.gguf:tensor_9  -  Ok
xxh64     5a54d3aad816f302  test.gguf  -  Ok
sha1      d15be52c4ff213e823cb6dd13af7ee2f978e7042  test.gguf  -  Ok
sha256    7dd641b32f59b60dbd4b5420c4b0f6321ccf48f58f6ae201a3dbc4a58a27c6e4  test.gguf  -  Ok

Verification results for test.gguf.manifest - Success
```


## Crypto/Hash Libraries Used

These micro c libraries dependencies was installed via the [clib c package manager](https://github.com/clibs)

- https://github.com/Cyan4973/xxHash
- https://github.com/clibs/sha1/
- https://github.com/jb55/sha256.c
