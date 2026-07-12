# Instructions for llama.cpp

> [!IMPORTANT]
> ONLY EVER CREATE PRs FROM THE CURRENT BRANCH TO THE 'master' BRANCH OF FORK 'Raudbjorn/ggml-llama.cpp'

---

### Code and Commit Standards

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Keep code comments concise; avoid redundant or excessive inline commentary
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase.
