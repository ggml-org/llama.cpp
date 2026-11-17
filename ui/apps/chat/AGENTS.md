# Instructions for ui/apps/chat

These add to the repository's [AGENTS.md](../../../AGENTS.md) for work in `ui/apps/chat`: **Llama Chat**, the chat client that llama-server bundles. **Llama UI**, the library, is `ui/lib/ui` (`@ggml-org/llama-ui`): Chat imports its tokens, fonts and base components from there (`@ggml-org/llama-ui/ui/<name>`) and keeps no copy of them. `DESIGN.md` and the Storybook live here; the brand book, `BRAND.md`, and the brand and communications pages are in [ggml-org/llama-brand](https://github.com/ggml-org/llama-brand), with their own Storybook.

- **Design:** [DESIGN.md](DESIGN.md) holds the tokens, roles and rules, written for agents. Follow it for anything visual; when it and the CSS disagree, the CSS wins and DESIGN.md is updated.
- **Stories:** a story lives with the code it shows. Llama UI's (its foundations, base components, app components and Model UI) are in `ui/lib/ui/tests/stories`, titled `Llama UI/...`, and run as the library's tests; this app's Storybook shows them too. Llama Chat's are in `tests/stories/<area>`, one folder per `src/lib/components/app/<area>` (`chat/` has one per Chat folder), titled `Llama Chat/<Area>/<Component>`. An accessibility story sits next to its component's, as `<Component>.a11y.stories.svelte`, titled `.../<Component>/Accessibility`.
- **Components:** the `SKILL.md` next to a component folder (such as [src/lib/components/app/SKILL.md](src/lib/components/app/SKILL.md)) holds that folder's conventions.

## Content is Markdown, custom pieces are Svelte: .svx

Anything that is mostly content (a doc, an overview, a showcase or a Storybook page with prose) is written in Markdown, and whatever Markdown cannot express (a live example, a diagram, a real component) is a Svelte component placed in it. Together that is a `.svx` page, compiled by [mdsvex](https://mdsvex.pngwk.com).

- Write the prose as plain Markdown, so it reads well as source, for people and for agents alike.
- Put each custom piece in its own `.svelte` file and import it in the page's `<script>`. Keep logic and styles out of the `.svx` file.
- Frame the page with a Svelte component that sets its typography, and show it in Storybook as a fullscreen story, as `introduction/Introduction.stories.svelte` does. Storybook's MDX is rendered with React and cannot hold Svelte components, so use MDX only for pages that are Markdown alone, such as Design guidelines.
- Do not add another Markdown-to-HTML pipeline. Content goes through mdsvex; user and assistant messages go through `MarkdownContent`.

### mdsvex versions

- `ui/apps/chat` is on mdsvex `0.12.7`, which only Storybook's `.svx` pages use. llama-brand's Storybook pages, the logo showcase and the type specimen among them, are on `1.0.0-next.3`.
- mdsvex 1.0 is the direction. Move `ui/apps/chat` to it, pinned to an exact `next` version, while it only serves Storybook pages; write new `.svx` pages so they work on both.
- Messages: 1.0's `@mdsvex/render` can render a document as it streams in (`@mdsvex/render/stream`) and as Svelte (`Node.svelte`). The plan is to swap the unified/remark/rehype pipeline inside `MarkdownContent` for it, keeping `MarkdownContent`'s API, so chat messages, the live-preview Markdown editor and the Notebook (which all render through `MarkdownContent`) switch together. Check parity first: GFM, math, Mermaid, the safe-HTML config, file-mention badges and the code-block actions, plus bundle size and streaming performance.
