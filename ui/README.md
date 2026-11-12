# ui

The UI of llama.cpp: the packages in `lib` and the apps built on them.

```text
ui/
├── lib/
│   └── ui/     @ggml/llama-ui: design tokens, fonts, surfaces and components
└── apps/
    └── chat/   Llama Chat, the chat app that llama-server serves
```

- [`lib/ui`](lib/ui/README.md) draws the parts and handles interaction, with no business logic. It is built with
  `@sveltejs/package` and published to npm as `@ggml/llama-ui`.
- [`apps/chat`](apps/chat/README.md) talks to llama-server, stores conversations and runs the agentic loop. CMake
  builds it into llama-server (see `scripts/ui-assets.cmake`).

Each folder is its own npm package with its own `package-lock.json`: run `npm ci` in the one you work on.
