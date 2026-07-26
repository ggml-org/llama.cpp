# llama-eval

Simple evaluation tool for llama.cpp with support for multiple datasets.

For a full description, usage examples, and sample results, see:

- [PR 21152](https://github.com/ggml-org/llama.cpp/pull/21152)

## Quick start

```bash
# Single server
python3 llama-eval.py \
  --server http://localhost:8033 \
  --model my-model \
  --dataset gsm8k --n_cases 100 \
  --grader-type regex --threads 32

# Multiple servers (comma-separated URLs and thread counts)
python3 llama-eval.py \
  --server http://server1:8033,http://server2:8033 \
  --server-name server1,server2 \
  --threads 16,16 \
  --dataset aime2025 --n_cases 240 \
  --grader-type regex
```

## Agentic suite

`--dataset agentic` is a different shape from the other code suites. It runs
**SACB** (Simple Agent Coding Benchmark), hosted at
[`ilintar/SACB`](https://huggingface.co/datasets/ilintar/SACB) -- 60 tasks over
three hand-written repositories, in Python and TypeScript. Rather than
answering one prompt, the model is given a small but real repository, a
deliberately narrow tool set, and a bug report, and drives its own multi-turn
conversation until it declares itself finished. It is then graded by running a
test suite it was never allowed to see.

```bash
python3 llama-eval.py \
  --server http://localhost:8033 --model my-model \
  --dataset agentic --threads 1 --temperature 0
```

### The tools

| tool | purpose |
|---|---|
| `list_files` | enumerate the tree |
| `read_file` | read, with 1-based line numbers and optional line-range narrowing |
| `search` | regular-expression search across the repository |
| `edit_lines` | replace an inclusive line range |
| `edit_replace` | replace an exact snippet, which must be unique |
| `write_file` | write a whole file |
| `lint` | run the language's linter and type checker |
| `finish` | declare the change complete |

Three constraints are deliberate:

**There is no shell and no way to run the tests.** The only feedback channel is
`lint`. A syntax or type error can be driven out mechanically; a logic error has
to be reasoned about. A harness that lets an agent iterate against the test
suite measures something closer to search than to understanding.

**There are two edit tools**, one line-addressed and one content-addressed.
Which one a model reaches for, and whether it keeps line numbers straight after
its own earlier edit, is itself a signal.

**The tests never touch the working tree.** They are held outside it and copied
into a throwaway copy only after the agent has stopped, so they can be neither
read nor edited. Every tool path is resolved and confinement-checked against the
repository root, so a symlink planted inside the tree cannot reach outside it.

### Scoring

A task is **resolved** when every test that was failing now passes *and* nothing
that was already passing broke. Both sets are derived by running the tests twice
when the corpus is built -- once against the defective tree and once against the
reference fix -- never hand-written, so a task whose defect no test exercises is
rejected rather than shipped.

Because many tasks contain several independent defects, the fraction of target
tests passed is reported alongside, which separates "fixed two of the three
faults" from "changed nothing". Process metrics are reported too: turns, tool
calls, rejected calls, edits, and the peak conversation size.

### Dependencies

Nothing here is installed unless you select this suite, and then only for the
language you actually run. `--agentic-lang python` provisions ruff and mypy;
`--agentic-lang typescript` provisions the TypeScript compiler and uses node's
own test runner. Neither touches the other, and no other suite touches either.

### Useful flags

| flag | meaning |
|---|---|
| `--agentic-lang` | restrict to one language, and provision only its toolchain |
| `--agentic-max-turns` | give up after this many assistant turns (default 50) |
| `--agentic-max-tokens` | cap generation per turn (default 2048); bounds a model that falls into a repetition loop |
| `--agentic-context-limit` | abandon an episode once its context exceeds this many tokens |


## Code suites (HumanEval, ClassEval)

`--dataset humaneval` runs the 164 OpenAI HumanEval problems; `--dataset
classeval` runs the 100 ClassEval class-generation tasks. Unlike the other
suites these are not graded by matching text: the model's reply is parsed for a
```` ```python ```` block, that code is combined with the task's own unit tests,
and the result is **executed**.

ClassEval asks for a whole class rather than one function, so it is scored the
way upstream does — **class level** (every one of the task's tests passes, which
is what feeds pass@k) and **method level** (what fraction of all 2196 test
methods passed). The second number separates "generated nothing usable" from
"got most of the class right", and both are printed at the end of a run.

Give it more room to generate than HumanEval — a class with ten methods does not
fit in a short budget:

```bash
python3 llama-eval.py \
  --server http://localhost:8033 --model my-model \
  --dataset classeval --threads 8 --n_predict 4096 --temperature 0
```

```bash
# pass@1, greedy
python3 llama-eval.py \
  --server http://localhost:8033 --model my-model \
  --dataset humaneval --threads 8 --n_predict 1024 --temperature 0

# pass@10 -- each extra multiple of 164 is another sample per problem
python3 llama-eval.py \
  --server http://localhost:8033 --model my-model \
  --dataset humaneval --n_cases 1640 --temperature 0.8 --top-p 0.95
```

`--grader-type exec` is the default for code suites, so neither it nor
`--dataset-source` normally needs to be passed.

### Execution sandbox — no Docker required

Generated code is untrusted, but these suites need far less than a container to
run safely. **The package set is derived from the dataset's own imports at
startup**, not hardcoded and not taken from the upstream `requirements.txt` —
which for ClassEval lists `torch`, `openai` and `transformers` that its *tests*
never touch. So each suite installs exactly what it needs:

| suite | third-party packages |
|---|---|
| HumanEval | **none** — every problem imports only stdlib (`typing, math, random, copy, string, collections, hashlib, re`) |
| ClassEval | 13: `beautifulsoup4, gensim, lxml, netifaces, nltk, numpy, openpyxl, pandas, pillow, pypdf2, python-docx, reportlab, scipy` |

Two of ClassEval's are *declared* rather than derived, because no import
statement mentions them: `lxml` is selected by string
(`BeautifulSoup(html, 'lxml')`, ClassEval_44 — without it that task silently
drops from 23/23 to 7/23), and `scipy` is a hard gensim requirement. Note also
that `PyPDF2` must **not** be modernised to `pypdf`: ClassEval_69's test imports
`PdfFileReader`, a name pypdf removed.

The venv is cached under `~/.cache/llama-eval/` (override with
`LLAMA_EVAL_CACHE`) and rebuilt only when the package set, interpreter version
or provisioning steps change; force one with `--rebuild-sandbox`. `uv` is used
when available, which makes even the ClassEval venv build in seconds.

**Interpreter selection.** A suite can prefer a different Python than the one
running the tool, and declares it: ClassEval asks for 3.13, because gensim
publishes no wheel past cp313 and its generated C no longer compiles on 3.14
(it still uses `PyLongObject.ob_digit`). That interpreter is found automatically,
including via `uv python`; `--sandbox-python` overrides it.

3.14 is fully supported anyway. ClassEval uses exactly two gensim APIs
(`utils.decode_htmlentities`, `matutils.unitvec`), both pure Python upstream, so
on 3.14+ the suite drops gensim from the install set and provisions a small shim
instead. Both paths score **100/100**.

**Offline provisioning.** A suite that would otherwise fetch data at test time
declares a provisioning step that runs once at venv build time, while the
network is still up. ClassEval uses this to pre-seed exactly three nltk corpora
(`punkt_tab`, `averaged_perceptron_tagger_eng`, `wordnet`, ~33 MB), so the tests
themselves run with no network at all. Those are the *current* names — the
dataset asks for `punkt` and `averaged_perceptron_tagger`, which nltk has since
renamed, and seeding the legacy names too is not harmless: with a network
available nltk considers them satisfied and re-fetches the modern ones anyway.
The corpora are written *inside* the venv, not `~/nltk_data`, because `$HOME` is
a tmpfs under the strongest isolation tier.

Each candidate program runs in a throwaway directory as its own process group,
under the strongest isolation the host offers:

| tier | requires | isolates |
|---|---|---|
| `bwrap` | bubblewrap installed | read-only root, tmpfs `/tmp` and `$HOME`, **no network** |
| `unshare` | `unshare --map-current-user` | **no network** |
| `unshare-root` | any `unshare -r` | no network, but **runs as uid 0** — see below |
| `rlimit` | always available | cpu/memory/file-size/process caps only |

**Candidate code must not run as root**, which is why `unshare -r` is a
last-resort tier that prints a warning. Root bypasses file permission bits, so a
test asserting that writing to a `chmod 0444` file *fails* instead sees it
succeed — ClassEval_50 scores 14/16 under uid 0 and 16/16 under uid 1000, with
nothing in the output to suggest the sandbox caused it. Install bubblewrap.

All tiers additionally cap CPU time, address space (4 GiB), file size and core
dumps, and kill the whole process group on timeout (`--exec-timeout`, default
15 s — raise it for ClassEval, whose tests do real work). Writes to the
throwaway working directory are allowed, which is all the file-handling tasks in
ClassEval need.

Note that the weakest tier does not contain filesystem access. If you are
scoring output from a model you do not trust, check the tier printed at startup.

### Reproducibility

Several tasks draw their test inputs from the *global, unseeded* RNG, so a
verdict is not reproducible run to run — which adds noise exactly when you are
comparing two models or two quantisations. `--exec-seed` (default `0`) seeds the
RNG before each test; pass `--exec-seed -1` for the stock upstream behaviour.
This does not change what the model is asked to compute.

It matters more than it sounds. HumanEval/38, /50 and /53 are affected, and
ClassEval_58 (duplicate mine placement) fails about 30% of the time on its own
*reference* solution — measured here at 21/30 passes unseeded versus **30/30
seeded**.

### Dataset fidelity

The canonical dataset runs **unmodified on current Python** — all 164 reference
solutions pass on CPython 3.14 with no patches, so no forked or re-hosted copy
is needed. `--dataset-source` accepts a local `.jsonl` or a HuggingFace repo id
if you do want to point at a modified copy.

HumanEval does contain a handful of long-standing defects where the *prompt*
contradicts the graded test (notably HumanEval/47, /116 and /148). Those
penalise a model that reads the docstring carefully, but correcting them makes
the benchmark marginally easier and the resulting scores no longer comparable
with published HumanEval numbers, so nothing is patched by default.

ClassEval is different: the upstream data has defects that cap the achievable
score below 100% for reasons unrelated to the model, so `--dataset classeval`
defaults to a **patched fork**,
[`ilintar/ClassEval`](https://huggingface.co/datasets/ilintar/ClassEval).
Six edits over five tasks lift the reference solutions from 96/100 classes and
2182/2196 test methods to **100/100 and 2196/2196**. Three repair tests nothing
could pass:

- **ClassEval_17** was a time bomb — a hard-coded 2024 date checked against
  `datetime.now()`, unpassable since January 2024.
- **ClassEval_48** asserted `get_hostname('0.0.0.0') == 'LAPTOP-2CS86KUM'`, the
  dataset author's own machine name.
- **ClassEval_31** compared floats with `assertEqual`, failing on a 4-ULP numpy
  difference.

The other three fix only the *reference* solutions (NumPy 2.0 removals in
ClassEval_51, a 30%-flaky mine generator in ClassEval_58) and cannot affect a
model's score. No `skeleton` was touched, so what the model is asked to write is
byte-identical to upstream. Pass `--dataset-source FudanSELab/ClassEval` for the
unpatched original.
