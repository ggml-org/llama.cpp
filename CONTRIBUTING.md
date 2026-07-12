# Contributing

This fork follows the same general contribution workflow as upstream
llama.cpp, but its authoritative policy surface is `AGENTS.md` in the
repository root. Read `AGENTS.md` first; it covers:

- ASCII-only text rules (no em dash, Unicode arrow, multiplication
  sign, or ellipsis; use `-`, `->`, `x`, `...`).
- Pull-request targeting: PRs go from the current branch to
  `master` of the `Raudbjorn/ggml-llama.cpp` fork.
- Code and commit standards, including reuse of existing
  infrastructure and pattern conformance with the surrounding code.

## Security issues

Do not disclose suspected vulnerabilities in a public issue or pull request.
Report them privately to `sveinbjorn@sveinbjorn.dev` with the affected commit,
reproduction details, impact, and any proposed mitigation.

## Hardware and runtime notes

Build, run, and benchmark guidance specific to this fork lives in
`docs/backend/SYCL.md` and the dated `docs/research/` notes. Those
documents are versioned with the source tree; do not copy their
numbers into pull-request descriptions or commit messages verbatim.
