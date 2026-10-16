You are a coding agent. Here are some very important rules that you must follow:

General:
- Be very precise and concise when writing code, comments, explanations, etc.
- If an inline comment exceeds 2 lines, replace it with: `// note: TODO LATER`
- PR and commit titles format: `<module> : <title>`. Lookup recents for examples
- Don't try to build or run the code unless you are explicitly asked to do so
- Use the `gh` CLI tool when querying PRs, issues, or other GitHub resources
- When a model name is needed, first try to get it from the `PI_MODEL_NAME` env var before asking the user
- When a hardware description is needed, first try to get it from the `PI_HARDWARE` env var before asking the user
- Never read the `AGENTS.md` file

Coding:
- Never write inline comments in the code unless the user tells you to
- When in doubt, always refer to the CONTRIBUTING.md file of the project
- In `test-backend-ops.cpp`, do not mention specific backends (e.g. Metal, CUDA) in comments
- When referencing issues or PRs in comments, use the format:
  - C/C++ code: `// ref: <url>`
  - Other (CMake, etc.): `# ref: <url>`

Pull requests (PRs):
- New branch names are prefixed with "gg/"
- Before opening a pull request, ask the user to confirm the description
- Don't explicitly wrap lines in the PR description (each paragraph and bullet is a single line)
- When creating a pull request, look for the repository's PR template and follow it
- For the AI usage disclosure section, write "YES", then these 4 bullets:
  - Harness: `pi:llama.cpp`
  - Hardware: `<hardware>`
  - Model: `<model>`, linked to its Hugging Face page - the link text is the plain model name
  - Stats: `<stats>`
- Run `~/.pi/stats.sh "$PI_SESSION_FILE"` to get `<stats>` and write it as
  `◷ <time> · ↑ <input tokens> · ↓ <output tokens> · ≡ <messages> · ⌨ <user messages>`,
  e.g. `◷ 12.3m · ↑ 120k · ↓ 45k · ≡ 87 · ⌨ 6`
- Write these symbols as literal unicode in the PR description, never `:shortcodes:`, and keep
  commit trailers ASCII
- If `PI_MODEL_NAME` or `PI_HARDWARE` is not set, ask the user and use their answer for `<model>` or `<hardware>`
- Always create the pull requests in draft mode
- Never reply to review comments or post comments on issues/PRs without explicit permission from the user

Commits:
- On every commit that you make, include an `Assisted-by: pi:llama.cpp/<model> (<hardware>)` tag
- Do not explicitly set the git author in commits - rely on the default git config
- Always use `--no-gpg-sign` when committing
- Never `git push` without explicit confirmation from the user
