# llama.cpp/examples/retrieval

Demonstration of simple retrieval technique based on cosine similarity

More info:
https://github.com/ggml-org/llama.cpp/pull/6193

### How to use

`retieval.cpp` has parameters of its own:
- `--context-file`: file to be embedded - state this option multiple times to embed multiple files
- `--chunk-size`: minimum size of each text chunk to be embedded
- `--chunk-separator`: STRING to divide chunks by. newline by default

`retrieval` example can be tested as follows:

```bash
make -j && ./llama-retrieval --model ./models/bge-base-en-v1.5-f16.gguf --top-k 3 --context-file README.md --context-file License --chunk-size 100 --chunk-separator .
```

This chunks and embeds all given files and starts a loop requesting query inputs:

```
Enter query:
```

On each query input, top k chunks are shown along with file name, chunk position within file and original text:

```
Enter query: describe the mit license
batch_decode: n_tokens = 6, n_seq = 1
Top 3 similar chunks:
filename: README.md
filepos: 119
similarity: 0.762334
textdata:
png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[Roadmap](https://github.
--------------------
filename: License
filepos: 0
similarity: 0.725146
textdata:
MIT License

Copyright (c) 2023 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
--------------------
filename: README.md
filepos: 9178
similarity: 0.621722
textdata:
com/cztomsik/ava) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal)
- [pythops/tenere](https://github.
--------------------
```
