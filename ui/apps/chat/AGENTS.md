# Instructions for ui/apps/chat

These add to the repository's [AGENTS.md](../../../AGENTS.md) for work in `ui/apps/chat`: **Llama Chat**, the chat client that llama-server bundles. **Llama UI**, the library, is `ui/lib/ui` (`@ggml-org/llama-ui`): Chat imports its tokens, fonts and base components from there (`@ggml-org/llama-ui/ui/<name>`) and keeps no copy of them. `DESIGN.md` lives here.

- **Design:** [DESIGN.md](DESIGN.md) holds the tokens, roles and rules, written for agents. Follow it for anything visual; when it and the CSS disagree, the CSS wins and DESIGN.md is updated.
- **Components:** the `SKILL.md` next to a component folder (such as [src/lib/components/app/SKILL.md](src/lib/components/app/SKILL.md)) holds that folder's conventions.
