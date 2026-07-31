# Skills

A skill is a folder holding a `SKILL.md` manifest. `TesseraSkillLoader`
scans its search directories for `SKILL.md` and `<name>/SKILL.md`, parses
the YAML frontmatter, and keeps only skills whose folder name matches the
frontmatter `name` (mismatches are skipped and logged). Matching skills are
injected into the agent's system prompt on demand via
`systemPromptFragment(for:)`.

Default search directories: the module resource bundle's `Skills/` folder
(if present) and `~/Documents/TesseraStudio/Skills`.

## Format

```
---
name: apple-reminders
description: Create, list, and complete Apple Reminders via the reminders CLI.
emoji: "⏰"
os: ["darwin"]
requires:
  bins: ["reminders"]
install:
  - brew install reminders
---

## When to Use
...
## When NOT to Use
...
## Setup
...
## Common Commands
...
```

Frontmatter keys: `name` (required), `description`, `emoji`, `os` (list),
`requires.bins` (list), `install` (list). A value is a scalar, an inline
`[a, b]` list, or an indented `- item` block list. The body is free
markdown; the `## When to Use`, `## When NOT to Use`, `## Setup`, and
`## Common Commands` sections are extracted for prompt injection.
