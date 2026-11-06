# Instructions for tools/ui

These add to the repository's [AGENTS.md](../../AGENTS.md) for work in `tools/ui`: **Llama UI**, the design system and component library, and **Llama Chat**, the chat client built from it that llama-server bundles.

- **Design:** [DESIGN.md](DESIGN.md) holds the tokens, roles and rules, written for agents. Follow it for anything visual; when it and the CSS disagree, the CSS wins and DESIGN.md is updated.
- **Why:** Storybook's **Introduction › Design philosophy** ([DesignPhilosophy.svx](tests/stories/introduction/DesignPhilosophy.svx)) explains the reasons, for people.
- **Components:** the `SKILL.md` next to a component folder (such as [src/lib/components/app/SKILL.md](src/lib/components/app/SKILL.md)) holds that folder's conventions.

## Content is Markdown, custom pieces are Svelte: .svx

Anything that is mostly content (a doc, an overview, a showcase or a Storybook page with prose) is written in Markdown, and whatever Markdown cannot express (a live example, a diagram, a real component) is a Svelte component placed in it. Together that is a `.svx` page, compiled by [mdsvex](https://mdsvex.pngwk.com).

- Write the prose as plain Markdown, so it reads well as source, for people and for agents alike.
- Put each custom piece in its own `.svelte` file and import it in the page's `<script>`. Keep logic and styles out of the `.svx` file.
- Frame the page with a Svelte component that sets its typography, and show it in Storybook as a fullscreen story, as `DesignPhilosophy.stories.svelte` and `LogoShowcase.stories.svelte` do. Storybook's MDX is rendered with React and cannot hold Svelte components, so use MDX only for pages that are Markdown alone, such as Design guidelines.
- Do not add another Markdown-to-HTML pipeline. Content goes through mdsvex; user and assistant messages go through `MarkdownContent`.

### mdsvex versions

- `tools/ui` is on mdsvex `0.12.7`, which only Storybook's `.svx` pages use. The llama.brand showcase and specimen are on `1.0.0-next.3`, and `LogoShowcase.svx` comes from there.
- mdsvex 1.0 is the direction. Move `tools/ui` to it, pinned to an exact `next` version, while it only serves Storybook pages; write new `.svx` pages so they work on both.
- Messages: 1.0's `@mdsvex/render` can render a document as it streams in (`@mdsvex/render/stream`) and as Svelte (`Node.svelte`). The plan is to swap the unified/remark/rehype pipeline inside `MarkdownContent` for it, keeping `MarkdownContent`'s API, so chat messages, the live-preview Markdown editor and the Notebook (which all render through `MarkdownContent`) switch together. Check parity first: GFM, math, Mermaid, the safe-HTML config, file-mention badges and the code-block actions, plus bundle size and streaming performance.
