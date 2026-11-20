# ui

The UI of llama.cpp: the packages in `lib` and the apps built on them.

```text
ui/
├── lib/
│   ├── ui/     @ggml-org/llama-ui: design tokens, fonts, surfaces and components
│   └── voice/  @ggml-org/llama-voice: microphone capture, transcription and dictation
└── apps/
    └── chat/   Llama Chat, the chat app that llama-server serves
```

- [`lib/ui`](lib/ui/README.md) draws the parts and handles interaction, with no business logic. It is built with
  `@sveltejs/package` and published to npm as `@ggml-org/llama-ui`. Apps in this repo skip the package: they compile
  its sources as part of their own build, so they hold no copy of a component, a token or a font.
- [`lib/voice`](lib/voice/README.md) records the microphone and turns speech into text through llama-server, with no
  UI: Llama UI draws the voice parts. It is plain TypeScript built with `tsc`, plus a Svelte entry
  (`@ggml-org/llama-voice/svelte`). Apps in this repo compile its sources too. It is not published to npm yet.
- [`apps/chat`](apps/chat/README.md) talks to llama-server, stores conversations and runs the agentic loop. CMake
  builds it into llama-server (see `scripts/ui-assets.cmake`), and it is published to npm as `@ggml-org/llama-chat`
  (with `llama-chat` as its short name), so `npx llama-chat` runs it against any llama-server.

Each folder is its own npm package with its own `package-lock.json`: run `npm ci` in the one you work on. Building,
checking and testing `apps/chat` needs no install in `lib/ui` or `lib/voice`.

## Releases

The npm packages are versioned by hand, with semver, and published by CI:

| Package | Version in | Published by |
| --- | --- | --- |
| `@ggml-org/llama-ui` | `lib/ui/package.json` | [`ui-lib-publish.yml`](../.github/workflows/ui-lib-publish.yml) |
| `@ggml-org/llama-chat` and `llama-chat` | `apps/chat/package.json` and `apps/chat/alias/package.json`, always the same | [`ui-chat-publish.yml`](../.github/workflows/ui-chat-publish.yml) |

To release, bump the version in the PR that should ship the change: a patch for fixes, a minor for new or changed
features or API while the version is 0.x. When that PR reaches `master` with a version npm does not have yet, the
workflow builds the package and publishes it; any other commit publishes nothing. A version lower than the one on npm
fails the run.

Llama Chat as bundled in llama-server is not versioned this way: it ships in every llama.cpp release, as `b<build>`.
