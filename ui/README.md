# ui

The UI of llama.cpp: the packages in `lib` and the apps built on them.

```text
ui/
├── lib/
│   └── ui/     @ggml-org/llama-ui: design tokens, fonts, surfaces and components
└── apps/
    └── chat/   Llama Chat, the chat app that llama-server serves
```

- [`lib/ui`](lib/ui/README.md) draws the parts and handles interaction, with no business logic. It is built with
  `@sveltejs/package` and published to npm as `@ggml-org/llama-ui`. Apps in this repo skip the package: they compile
  its sources as part of their own build, so they hold no copy of a component, a token or a font.
- [`apps/chat`](apps/chat/README.md) talks to llama-server, stores conversations and runs the agentic loop. CMake
  builds it into llama-server (see `scripts/ui-assets.cmake`).

Each folder is its own npm package with its own `package-lock.json`: run `npm ci` in the one you work on. Building,
checking and testing `apps/chat` needs no install in `lib/ui`.
