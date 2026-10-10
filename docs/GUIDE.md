# Repository field guide

This is a navigation and source map for the Raudbjorn ggml-llama.cpp fork.
Start here to find the right code, guide, test or evidence collection; use the
[documentation index](README.md) for the complete curated document list and the
[benchmark collection guide](benchmarks/README.md) for the separately inventoried
local campaign archive.

## Scope and provenance

Curated on 2026-10-08 against source revision `d2f1f9834` and the working tree.
The inventory below uses `git ls-files`, not a recursive count of build outputs.
At inventory time there were **3,636 tracked paths**, including **646 under
`docs/`**, of which **109 were Markdown**. `docs/research/` accounted for 564
tracked paths, including raw artifacts. `docs/benchmarks/` was a separate,
untracked collection and is deliberately excluded from those totals. Counts
precede creation of this guide and do not imply that every source file was read.

This guide synthesizes existing indexes and inspected build, API, routing and
harness entry points. It establishes locations and relationships, not build
success, complete feature support or new performance results. Sources outrank
this snapshot when they change.

## Start with the task

| Task | Read first | Continue in source or evidence |
| --- | --- | --- |
| Understand the fork's purpose and supported scope | [Root README](../README.md) | [Upstream preservation rules](development/upstream-merge.md) |
| Build a local binary | [Build guide](build/build.md) | [Root CMake](../CMakeLists.txt), then the selected backend guide |
| Run on Intel Arc | [SYCL guide](backend/SYCL.md) | [Stack pins](research/software-stack/sycl-build-runtime-pins.md); check current host separately |
| Embed inference in another application | [SDK guide](SDK.md) | [Public C API](../include/llama.h), [C++ helpers](../include/llama-cpp.h), [simple example](../examples/simple/) |
| Run a CLI or HTTP service | [CLI guide](../tools/cli/README.md), [server guide](../tools/server/README.md) | [Argument parser](../common/arg.cpp), [presets](user/preset.md) |
| Change the Web UI | [UI guide](../tools/ui/README.md) | [UI source](../tools/ui/src/), [asset build selection](../scripts/ui-assets.cmake) |
| Convert or add a model | [Model guide](user/models.md), [add-model guide](development/HOWTO-add-model.md) | [Converter](../convert_hf_to_gguf.py), [conversion modules](../conversion/), [model graphs](../src/models/) |
| Change TurboQuant representation or KV policy | [KV quantization](turboquant/KV-cache-quantization.md) | Codec, graph and cache source map below; [quality evidence](turboquant/quality-benchmarks.md) |
| Diagnose an operator/backend mismatch | [Op support tables](ops.md) | [Backend operation tests](../tests/test-backend-ops.cpp), selected backend implementation |
| Measure speed, quality or capacity | [Scripts catalog](../scripts/README.md) | [Benchmark collection](benchmarks/README.md), [research index](research/README.md) |
| Work on drafting and acceptance | [Speculative decoding](features/speculative.md) | [Speculative implementation](../common/speculative.cpp), [performance harness findings](../scripts/perf/FINDINGS.md) |
| Sync upstream or prepare the generated SDK | [Upstream merge runbook](development/upstream-merge.md), [SDK guide](SDK.md) | [SDK generator](../scripts/prune-to-lib.sh); generation changes branch state |
| Delegate a bounded maintenance task | [Agent roster and contract](development/agents.md) | Shared role bodies in [.claude/agents](../.claude/agents/) and Codex wrappers in [.codex/agents](../.codex/agents/) |

## Repository map

Counts are tracked paths at the revision above, not executable counts or measures
of importance. "Evidence" means retained output, not a fresh validation claim.

| Surface | Paths | Purpose and status |
| --- | ---: | --- |
| [include/](../include/) | 2 | Public llama C API and C++ ownership helpers; embedding interface |
| [src/](../src/) | 227 | Inference runtime, model loading, graphs, vocabulary, sampling and KV/memory implementations |
| [ggml/](../ggml/) | 630 | Tensor library, quantization, backend API and shipped backend implementations |
| [common/](../common/) | 95 | Application helpers: arguments, presets, sampling, chat, downloads and speculative decoding |
| [tools/](../tools/) | 1,019 | CLI, server, UI, benchmark, quantization, evaluation and multimodal tools; includes tests/assets |
| [app/](../app/) | 3 | Unified executable dispatch; its CMake target links the individual tool implementations |
| [examples/](../examples/) | 217 | API demonstrations, bindings and platform examples; presence alone does not prove platform support |
| [conversion/](../conversion/) | 94 | Model-family conversion modules selected by the root converter |
| [gguf-py/](../gguf-py/) | 29 | GGUF Python package, readers/writers and format tools |
| [models/](../models/) | 121 | Repository model-related fixtures and assets; not a manifest of locally installed full models |
| [grammars/](../grammars/) | 9 | Grammar assets for constrained generation |
| [tests/](../tests/) | 116 | Runtime, backend, codec and regression tests; registration and prerequisites vary |
| [scripts/](../scripts/) | 220 | Maintenance and campaign runners, analyzers, fixtures and script tests; [catalog](../scripts/README.md) |
| [docs/](./) | 646 | Instructions, reference, plans and historical evidence; see the map below |
| [bench-a770/](../bench-a770/) | 70 | Retained A770 product/provenance/sample evidence, separate from runner source |
| [benches/](../benches/) | 6 | DGX Spark/Nemotron reports and artifacts; historical hardware evidence, not an A770 execution suite |
| [pocs/](../pocs/) | 4 | Small vector-dot proof-of-concept programs with their own CMake definitions |
| [cmake/](../cmake/) | 12 | Build helpers and package configuration templates |
| [packaging/](../packaging/) | 4 | Arch package recipe and service/config templates; files do not establish installed host state |
| [requirements/](../requirements/) | 12 | Python dependency lists for conversion and related tooling |
| [vendor/](../vendor/) | 27 | Vendored dependencies and build integration |
| [.claude/](../.claude/), [.codex/](../.codex/) | 14 each | Tracked project agent definitions; local untracked additions excluded |
| [.gemini/](../.gemini/), [skills/](../skills/) | 1, 2 | Additional agent context and task workflows |
| [media/](../media/), [licenses/](../licenses/) | 11, 1 | Documentation assets and additional licensing material |
| Repository root | 30 | Main README/instructions, CMake, license, converters and project configuration |

The separate [docs/benchmarks](benchmarks/README.md) inventory covers working-tree
artifacts beyond this tracked map. Local build directories, downloaded models,
`.scratch/`, repository packs and ignored files are not part of the source census.

## Runtime and architecture: where a change travels

A useful reading order is application -> common helpers -> llama runtime -> ggml
scheduler/backend -> kernel. Model conversion is an input path into that runtime,
not part of each inference call.

| Layer | Entry points | Follow when changing |
| --- | --- | --- |
| Application and configuration | [tools/server](../tools/server/), [tools/cli](../tools/cli/), [app/CMakeLists.txt](../app/CMakeLists.txt), [common/arg.cpp](../common/arg.cpp) | User-visible options, service behavior and shared argument handling |
| Model format and loading | [convert_hf_to_gguf.py](../convert_hf_to_gguf.py), [conversion/__init__.py](../conversion/__init__.py), [gguf-py](../gguf-py/), [llama-model-loader.cpp](../src/llama-model-loader.cpp) | Tensor names, architecture metadata and file compatibility |
| Public inference interface | [llama.h](../include/llama.h), [llama-context.cpp](../src/llama-context.cpp) | Calls such as `llama_decode`, batch processing and context state |
| Model graphs and attention | [src/models](../src/models/), [llama-graph.cpp](../src/llama-graph.cpp) | Architecture-specific operations and shared attention construction |
| Persistent sequence state | [llama-kv-cache.cpp](../src/llama-kv-cache.cpp), [llama-memory.cpp](../src/llama-memory.cpp), [llama-memory-hybrid.cpp](../src/llama-memory-hybrid.cpp) | Cache layout, allocation, recurrent/hybrid memory and policy |
| Scheduling and backend contract | [ggml-backend.cpp](../ggml/src/ggml-backend.cpp), [ggml-backend.h](../ggml/include/ggml-backend.h) | Tensor placement, graph dispatch and backend interfaces |
| Shipped backends | [ggml/src/CMakeLists.txt](../ggml/src/CMakeLists.txt) | CPU, BLAS, SYCL, Vulkan and OpenVINO registration; retained reports for other hardware do not add backends |
| Speculative decoding | [common/speculative.cpp](../common/speculative.cpp), [llama-ext.h](../src/llama-ext.h), [tools/server](../tools/server/) | Draft generation, verification and server integration; follow callers across layers |

For TurboQuant, begin with the type contract in [ggml.h](../ggml/include/ggml.h),
block layouts in [ggml-common.h](../ggml/src/ggml-common.h), and the CPU reference
in [ggml-turbo-quant.c](../ggml/src/ggml-turbo-quant.c). Then read graph-side
`ggml_turbo_wht` calls in [llama-graph.cpp](../src/llama-graph.cpp) and cache policy
in [llama-kv-cache.cpp](../src/llama-kv-cache.cpp). Layout or enum changes affect
serialized data and backend consumers; kernel-only fixes have a different scope.

For SYCL attention, follow [fattn.cpp](../ggml/src/ggml-sycl/fattn.cpp) from
`ggml_sycl_get_best_fattn_kernel` to the selected implementation. Read
[ggml-sycl.cpp](../ggml/src/ggml-sycl/ggml-sycl.cpp) for backend dispatch and
runtime configuration, [set_rows.cpp](../ggml/src/ggml-sycl/set_rows.cpp) for cache
writes, and [turbo-quants.hpp](../ggml/src/ggml-sycl/turbo-quants.hpp) for device
codec helpers. The [VEC contract note](research/turbo/turbo-fa-research-artifact.md)
explains why graph rotation and kernel dequantization must agree. Check current
routing predicates before reusing a historical kernel-performance conclusion.

## Build, configuration, tests and exports

| Question | Authoritative surface | Practical boundary |
| --- | --- | --- |
| Which components are built? | [Root CMake](../CMakeLists.txt), [src CMake](../src/CMakeLists.txt) | `LLAMA_BUILD_COMMON`, `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_EXAMPLES`, `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_APP` depend on standalone/subproject setup; inspect the actual cache |
| Which backend is enabled? | [ggml CMake options](../ggml/CMakeLists.txt), [backend registration](../ggml/src/CMakeLists.txt) | Build-time availability differs from runtime device selection and per-op support |
| What sets a CLI/environment option? | [common/arg.cpp](../common/arg.cpp), [preset guide](user/preset.md), selected backend source | `LLAMA_ARG_*` application parsing and backend-specific environment variables are separate surfaces |
| Does this build download models? | [common/CMakeLists.txt](../common/CMakeLists.txt), [SDK guide](SDK.md) | `LLAMA_DOWNLOAD=OFF` selects a downloader stub; do not infer network behavior from a flag appearing in parser help |
| How are UI assets supplied? | [scripts/ui-assets.cmake](../scripts/ui-assets.cmake), [UI guide](../tools/ui/README.md) | Build configuration determines prebuilt/downloaded/local assets |
| What does installation export? | [Root install rules](../CMakeLists.txt), [cmake templates](../cmake/) | llama headers/library, CMake package and pkg-config metadata; `llama-common` install is gated on common plus shared builds |
| How is the library-only tree produced? | [SDK guide](SDK.md), [prune-to-lib.sh](../scripts/prune-to-lib.sh) | Generated `lib` branch; make source changes in the full development tree |
| Which tests actually register? | [tests/CMakeLists.txt](../tests/CMakeLists.txt) | Backend, build and fixture guards matter; a test source existing does not mean CTest selected it |

Use the [build guide](build/build.md) for commands and the [SYCL](backend/SYCL.md)
or [OpenVINO](backend/OPENVINO.md) guide for dependencies. Avoid copying compiler
pins into another guide: the [dated stack record](research/software-stack/sycl-build-runtime-pins.md)
and a live host probe are needed to establish the environment.

Validation is layered. [test-turbo-quant.c](../tests/test-turbo-quant.c) exercises
codec behavior; [test-backend-ops.cpp](../tests/test-backend-ops.cpp) compares
operators; [test-sycl-turbo-correctness.cpp](../tests/test-sycl-turbo-correctness.cpp)
is the synthetic CPU/SYCL oracle. [Server tests](../tools/server/tests/README.md)
cover another boundary, while `scripts/test_*.py` test campaign bookkeeping and
failure handling. Passing harness tests does not validate model output or GPU
performance. Use [debugging tests](development/debugging-tests.md) to locate and
run the relevant target rather than assuming every test is model-free.

GPU runs require explicit resource coordination, time limits and before/after
fault evidence; see the [agent contract](development/agents.md). A clean filtered
log is not proof of health if reading the kernel log failed. Do not execute a
campaign merely to refresh documentation or apply service-control snippets from
old notes without understanding their effects.

## Documentation and evidence map

| Collection | Tracked paths | Read as |
| --- | ---: | --- |
| [build/](build/), [user/](user/), [features/](features/) | 7, 4, 16 | Build/how-to and feature reference, including relocated upstream docs |
| [backend/](backend/) | 3 | Backend and MoE-cache operational guides |
| [turboquant/](turboquant/) | 20 | Codec policy, quality methodology and per-model result tables |
| [development/](development/) | 11 | Maintainer runbooks and implementation guidance |
| [research/](research/README.md) | 564 | Dated reports plus supporting JSON, CSV, logs, patches and probes |
| [plans/](plans/README.md) | 13 | Proposed work and dependencies; a plan does not prove implementation or validation |
| [ops/](ops/), [ops.md](ops.md) | 5 plus one root doc | Backend support inputs and generated table; hardware/device-specific snapshots |
| Documentation root | 3 | Existing documentation index, SDK guide and generated op table before this guide |
| [benchmarks/](benchmarks/README.md) | Outside tracked census | Separately curated local campaign collection; consult its own coverage and provenance limits |

Research is arranged into [SYCL](research/sycl/), [TurboQuant](research/turbo/),
[speculative decoding](research/speculative/) and [software stack](research/software-stack/)
topics. Start at the research index, then a report, then its raw artifacts. For
campaign code, start at the scripts catalog. A `product.json`, trace or log is
an observation to interpret with its command, build and environment, not a guide
to the current implementation.

Evidence also lives outside `docs/`: [bench-a770](../bench-a770/),
[benches](../benches/), [scripts/perf](../scripts/perf/) and tool-specific benchmark
directories. These are distinct collections. The large local benchmark guide
must not silently absorb or reclassify them as files it inventoried.

## Reading old results without importing old assumptions

Prefer current source and live output for behavior, dated runs for observations,
and plans for intentions. Keep the distinction in summaries. The
[known-stale-prose ledger](development/agents.md) explicitly lists drift in root
instructions, including CI, graph replay, fusion and older oracle descriptions.
For example, current SYCL source reads `GGML_SYCL_ENABLE_GRAPH`; copying an old
"graph replay killed" conclusion would hide an implemented opt-in path.

Match driver, hardware, model/quantization, source/build identity, context depth,
KV configuration and workload before comparing numbers. i915-era A770 results
are not an xe baseline. An unsupported-backend name in an old DGX report or
platform example does not override current backend registration. The removed
upstream CI directories are documented in the merge runbook; do not treat old
workflow references as proof of present CI coverage.

Recorded absolute artifact paths may belong to another checkout or host. Keep
them as provenance, label missing inputs, and find surviving local evidence
before claiming reproduction. Do not silently rewrite a historical result to
match current assumptions. Several software-stack survey notes are explicitly
unverified model-generated reference in the research index.

## Keeping this map useful

### Existing Claude/Repomix reference skills

The local [llamacpp-a770 skill](../.claude/skills/llama.cpp-a770/SKILL.md)
already provides a packed repository reference. Its
[summary](../.claude/skills/llama.cpp-a770/references/summary.md) explains
coverage and exclusions; its
[directory index](../.claude/skills/llama.cpp-a770/references/project-structure.md)
locates files, and its
[tech-stack index](../.claude/skills/llama.cpp-a770/references/tech-stacks.md)
lists detected packages. The separate
[benchmark skill](benchmarks/.claude/skills/llama.cpp-a770-benchmarks/SKILL.md)
provides the same search workflow for the archive.

Use these existing indexes for discovery, then verify the original file.
Their `references/files.md` packs are compressed snapshots of about 463 MB
and 392 MB respectively. Search a specific `## File:` header instead of
loading either pack in full:

```bash
rg -n '^## File: ggml/src/ggml-sycl/fattn.cpp$' \
  .claude/skills/llama.cpp-a770/references/files.md
```

The root pack advertises 11,940 files and includes local benchmark material;
that is a different scope from the 3,636 tracked-path census above. Both
pack summaries explicitly exclude benchmark `product.json` files. Read
those original JSON files to inspect provenance and validity diagnostics.
The packs and their skills were inspected, not regenerated or edited during
this curation; they remain local untracked references. The field guides add
task routes, synthesis and evidence caveats around these existing indexes.

### Refresh procedure

1. Inventory tracked paths with `git ls-files`. Separately enumerate untracked
   Markdown with `git ls-files --others --exclude-standard -- '*.md'`; explicitly
   scope any ignored artifact collection before counting it.
2. Read existing indexes first. For new entries, inspect the document and its
   runner/configuration source; summarize purpose, inputs, output and evidence
   status. Do not infer success from a filename or completed-plan checkbox.
3. Add a task route here only when it changes how readers navigate. Keep detailed
   runner usage in the scripts/tool guide, dated findings in research, and
   prospective work in plans. Preserve raw artifacts and their recorded paths.
4. When changing behavior, check callers, tests, fixtures, configuration and
   public/export surfaces. Update the owning guide and its index, then this map
   if the entry point changed. Preserve the fork's layout during upstream merges.
5. Verify relative links against current files, including new and untracked
   Markdown. Scan edited docs for ASCII and compare broken-link sets before and
   after moves. External URLs and Markdown anchors require separate validation.
6. Record the source revision, inventory boundary and checks performed. Add dated
   corrections that link superseding evidence; do not promote old measurements
   into current performance claims by re-indexing them.

For the local benchmark collection, [index_benchmarks.py](../scripts/index_benchmarks.py)
regenerates a TSV of paths, file kinds, sizes, SHA-256 hashes, identical-content
references and symlink targets. It includes hidden/ignored artifacts, does not
follow symlinks, and excludes the four collection-root guide/index files. Its
[unit tests](../scripts/test_index_benchmarks.py) check inventory semantics; they
do not validate benchmark results. Follow the [collection guide](benchmarks/README.md)
for regeneration and interpretation.

No indexing service or external dependency is required for this map. Repository
packs can help a reading pass but are snapshots, not the authority for paths or
state.
